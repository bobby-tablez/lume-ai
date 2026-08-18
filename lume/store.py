"""Local session persistence: one append-only JSONL file per chat.

Nothing in this module touches the network — transcripts stay on the machine that
produced them. The design bias is *durability over speed*: a lost conversation is
worse than a slow listing.

Layout::

    <root>/                      0o700
      sessions/                  0o700
        <id>.jsonl               0o600   line 1 = meta header, later lines = messages
        <id>.jsonl.lock          0o600   flock target (never replaced, so a rewrite
                                         cannot orphan another process's lock)
        <id>.jsonl.bak           0o600   rescue copy, only ever made before a write
                                         that would otherwise overwrite a header we
                                         could not read
        index.json               0o600   *cache only* — list() works without it

Because a message is one line written with a single ``write(2)`` under an
advisory lock, a crash can only ever damage the line being written; every earlier
line is already durable. Readers therefore skip a garbled line instead of failing,
and the meta header is replaced with temp-file + :func:`os.replace` so an
interrupted rewrite leaves either the old file or the new one — never a mixture.

Everything a reader needs is therefore *derived* from the records, never kept in
sync by rewriting them. That includes the token and cost totals: `record_usage`,
which the app calls after every assistant reply, appends a ``{"type":"usage"}``
*checkpoint* line carrying the running totals, and a reader folds the newest
checkpoint over whatever the header says. Rewriting the header on every turn made
a conversation quadratic — turn 600 rewrote the 2.7 MB of turns 1..599 — and put
every earlier message through :func:`os.replace` once per reply for no reason.

One machine per store: a temp file is attributed to a *(host, pid)* pair because
a pid seen on a store shared over NFS or SMB says nothing about a process here.
``$LUME_HOME`` is still best kept off a shared filesystem — advisory locking over
NFS is not something this module can vouch for.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import io
import json
import os
import pathlib
import re
import stat as stat_mod
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime

from .ansi import strip_ansi, truncate

try:  # POSIX advisory locking
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:  # Windows byte-range locking
    import msvcrt
except ImportError:
    msvcrt = None

__all__ = [
    "Message", "SessionMeta", "Store",
    "StoreWarning", "AmbiguousRefError",
    "default_root", "new_id", "auto_title",
    "EXT", "TITLE_WIDTH",
]

EXT = ".jsonl"
INDEX_NAME = "index.json"
INDEX_VERSION = 1
TITLE_WIDTH = 48
DIR_MODE = 0o700
FILE_MODE = 0o600
MAX_ID_LEN = 64
_TMP_PREFIX = ".tmp-"
#: Shortest record a reader would accept, in bytes: enough to bound a cached
#: message count against the file size.
_MIN_RECORD_BYTES = len(b'{"type":"message","role":"u","content":""}\n')
#: How far back from the end of a file :meth:`Store._counters_at` will look for a
#: usage checkpoint before giving up and making a full pass instead.
_TAIL_CAP = 4 << 20
#: Ceiling on the meta header line. Only a defence against a file that is one
#: enormous unterminated line; a real header is a few hundred bytes.
_MAX_HEADER_BYTES = 8 << 20
#: How far back from the end of a file :meth:`Store._all_metas` reads looking for
#: the newest usage checkpoint before it has to fall back on the cached totals.
_INDEX_TAIL = 1 << 15
#: Counter fields a usage checkpoint carries, cumulative for the whole session.
_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)


if hasattr(os, "pread"):
    _pread = os.pread
else:                                   # pragma: no cover - Windows has no pread
    def _pread(fd: int, n: int, offset: int) -> bytes:
        """``os.pread`` for platforms without one, via seek/read/seek-back.

        Not atomic the way pread is, so it relies on what every caller here
        already holds: the file lock. The original file position is restored,
        and O_APPEND writes are unaffected by the seek either way.
        """
        keep = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            out = b""
            while len(out) < n:
                chunk = os.read(fd, n - len(out))
                if not chunk:
                    break
                out += chunk
            return out
        finally:
            os.lseek(fd, keep, os.SEEK_SET)
#: A session path that is a FIFO would park ``os.open`` until somebody opened the
#: other end; on a regular file this flag does nothing at all.
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

#: Ids we generate are lowercase base32; the allowlist is deliberately wider so a
#: hand-written id still works, and deliberately narrow enough that no accepted id
#: can contain a path separator, a drive letter, a dot-dot, or a control byte.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,%d}$" % (MAX_ID_LEN - 1))

#: Device names that are *files* on Windows no matter which directory you are in.
_WIN_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"]
    + ["com%d" % i for i in range(1, 10)]
    + ["lpt%d" % i for i in range(1, 10)]
)

ROLE_LABELS = {"user": "You", "assistant": "Claude", "system": "System"}


class StoreWarning(UserWarning):
    """Raised through :mod:`warnings` when damaged data is skipped or repaired."""


class AmbiguousRefError(LookupError):
    """A session reference matched more than one session.

    Deliberately *not* a :class:`KeyError` (which is also a ``LookupError``) so
    callers can tell "no such session" from "say which one you meant".
    """


# ------------------------------------------------------------------------ paths


def default_root(env=None) -> pathlib.Path:
    """Storage root: ``$LUME_HOME``, else ``$XDG_DATA_HOME/lume``, else the
    platform default (``~/Library/Application Support/lume`` on macOS)."""
    env = os.environ if env is None else env
    home = (env.get("LUME_HOME") or "").strip()
    if home:
        return pathlib.Path(home).expanduser()
    xdg = (env.get("XDG_DATA_HOME") or "").strip()
    if xdg and os.path.isabs(os.path.expanduser(xdg)):
        return pathlib.Path(xdg).expanduser() / "lume"
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Application Support" / "lume"
    return pathlib.Path.home() / ".local" / "share" / "lume"


def _validate_id(session_id) -> str:
    """Return `session_id` if it can only ever name a file *inside* the store."""
    if not isinstance(session_id, str):
        raise ValueError("session id must be a string, got %s" % type(session_id).__name__)
    s = session_id
    if not s or len(s) > MAX_ID_LEN:
        raise ValueError("invalid session id: bad length")
    if "\x00" in s:
        raise ValueError("invalid session id: contains NUL")
    if any(ch < " " or ch == "\x7f" for ch in s):
        raise ValueError("invalid session id: contains a control character")
    if s in (".", "..") or ".." in s:
        raise ValueError("invalid session id: path traversal")
    if "/" in s or "\\" in s or ":" in s:
        raise ValueError("invalid session id: contains a path separator")
    if os.path.isabs(s) or os.path.splitdrive(s)[0]:
        raise ValueError("invalid session id: absolute path")
    if s[-1] in ". ":  # Windows silently strips these, so two ids could collide
        raise ValueError("invalid session id: trailing dot or space")
    if s.split(".")[0].lower() in _WIN_RESERVED:
        raise ValueError("invalid session id: reserved device name")
    if not _ID_RE.match(s):
        raise ValueError("invalid session id: illegal characters")
    return s


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except (ValueError, OSError):  # different drives, or an unusable path
        return False


_B32 = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford: no i/l/o/u, so no lookalikes
_id_guard = threading.Lock()
_last_ms = 0


def _b32(n: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_B32[n & 31])
        n >>= 5
    return "".join(reversed(out))


def new_id(now: float | None = None) -> str:
    """A 26-char ULID-style id: millisecond prefix + 80 random bits.

    Lexicographic order matches creation order, the alphabet is filename-safe on
    every platform, and the counter never goes backwards inside a process so ids
    minted in the same millisecond still sort.
    """
    global _last_ms
    ms = int((time.time() if now is None else now) * 1000)
    with _id_guard:
        if ms <= _last_ms:
            ms = _last_ms + 1
        _last_ms = ms
    return _b32(ms, 10) + _b32(int.from_bytes(os.urandom(10), "big"), 16)


# ------------------------------------------------------------------------ titles


def _clean_title(text, width: int = TITLE_WIDTH) -> str:
    """Collapse `text` to a single short line safe to store and print."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    s = strip_ansi(text)
    s = "".join(" " if (ch < " " or ch == "\x7f") else ch for ch in s)
    s = re.sub("[\u2028\u2029\ufeff\u200b\xa0]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Drop a leading markdown marker so titles read as prose, not as syntax.
    s = re.sub(r"^(?:```+\s*\w*|~~~+\s*\w*|[#>*+\-]+|\d+[.)])\s*", "", s).strip()
    if not s:
        return ""
    return truncate(s, max(1, width)).strip()


def _role_label(role) -> str:
    """Display name for a role, safe to drop into structural markup.

    A stored role is whatever was written to the file, which is not necessarily
    one of the three we know. ``**ESC[31mFAKEROLE**`` in a heading is a stored
    string steering a rendered document, so an unknown role goes through the same
    cleaning as a title instead of straight through.
    """
    label = ROLE_LABELS.get(role)
    if label:
        return label
    return _clean_title(role, 32) or "unknown"


def auto_title(text, width: int = TITLE_WIDTH) -> str:
    """Derive a display title from a message body. Never raises, never empty."""
    return _clean_title(text, width) or "Untitled"


# ----------------------------------------------------------------------- records


def _num(v, default: float = 0.0) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf poison sorting
        return default
    return float(v)


def _int(v, default: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    try:
        return int(v)
    except (ValueError, OverflowError):
        return default


def _limit(value) -> int:
    """Coerce a ``list(limit=...)`` argument, refusing junk out loud.

    A CLI hands over whatever was on the command line, so ``"2"`` has to mean 2.
    Anything that is not a number at all used to fall back to the ``_int``
    default of 0 and quietly return an empty list — "you have no sessions".
    """
    if isinstance(value, bool):
        raise TypeError("limit must be an integer, got a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError("limit must be a finite number, got %r" % value)
        return int(value)
    if isinstance(value, str):
        n = value.strip()
        sign = ""
        if n[:1] in ("+", "-"):
            sign, n = n[0], n[1:]
        # Not int(): it reads "1_0" as 10 and the Arabic-Indic digit "\u0661" as 1.
        # resolve() refuses both for the same reason and says why; a limit typed at
        # a prompt is ASCII, and the length cap keeps int() away from its
        # digit-conversion limit.
        if n.isascii() and n.isdigit() and len(n) <= 18:
            return int(sign + n)
        raise TypeError("limit must be an integer, got %r" % value)
    raise TypeError("limit must be an integer, got %s" % type(value).__name__)


def _text(v) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    if isinstance(v, (list, dict)):  # e.g. an API content-block list
        try:
            return json.dumps(v, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def _json_default(o):
    """Last-resort encoder: a weird value must not cost the user their message."""
    return str(o)


def _dumps_line(obj: dict) -> bytes:
    """Encode one record as exactly one line of UTF-8 (JSON escapes newlines)."""
    try:
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                          default=_json_default).encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates in user input
        data = json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                          default=_json_default).encode("ascii", "backslashreplace")
    if b"\n" in data:  # pragma: no cover - json cannot emit a raw newline
        raise ValueError("record contains a newline")
    return data + b"\n"


@dataclass
class Message:
    """One turn in a conversation."""

    role: str
    content: str
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=new_id)
    model: str | None = None
    usage: dict | None = None
    thinking: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Serialisable form; unset optional fields are omitted to keep lines lean."""
        d = {"role": _text(self.role), "content": _text(self.content),
             "ts": _num(self.ts, 0.0), "id": _text(self.id) or new_id()}
        if self.model:
            d["model"] = self.model
        if self.usage:
            d["usage"] = self.usage
        if self.thinking:
            d["thinking"] = self.thinking
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """Rebuild from a stored record, coercing anything odd rather than raising."""
        return cls(
            role=_text(d.get("role")) or "user",
            content=_text(d.get("content")),
            ts=_num(d.get("ts")),
            id=_text(d.get("id")) or new_id(),
            model=d.get("model") if isinstance(d.get("model"), str) else None,
            usage=d.get("usage") if isinstance(d.get("usage"), dict) else None,
            thinking=d.get("thinking") if isinstance(d.get("thinking"), str) else None,
            error=d.get("error") if isinstance(d.get("error"), str) else None,
        )


@dataclass
class SessionMeta:
    """Header record for a session, plus counters derived from its messages."""

    id: str
    title: str = ""
    created: float = 0.0
    updated: float = 0.0
    model: str = ""
    system: str | None = None
    message_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    pinned: bool = False
    tags: list = field(default_factory=list)
    #: Header keys this version does not know about. Kept verbatim so that an
    #: older build renaming a session written by a newer one does not silently
    #: erase whatever the newer one stored there.
    extra: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.extra.items()
               if isinstance(k, str) and k != "type" and k not in _META_FIELDS}
        out.update({
            "id": self.id, "title": self.title,
            "created": float(self.created), "updated": float(self.updated),
            "model": self.model, "system": self.system,
            "message_count": int(self.message_count),
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "cache_read_tokens": int(self.cache_read_tokens),
            "cache_write_tokens": int(self.cache_write_tokens),
            "cost_usd": float(self.cost_usd),
            "pinned": bool(self.pinned),
            "tags": list(self.tags),
        })
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "SessionMeta":
        d = d if isinstance(d, dict) else {}
        tags = d.get("tags")
        tags = [t[:64] for t in tags if isinstance(t, str)][:32] if isinstance(tags, list) else []
        system = d.get("system")
        return cls(
            id=_text(d.get("id")),
            title=_text(d.get("title")),
            created=_num(d.get("created")),
            updated=_num(d.get("updated")),
            model=_text(d.get("model")),
            system=system if isinstance(system, str) else None,
            message_count=_int(d.get("message_count")),
            input_tokens=_int(d.get("input_tokens")),
            output_tokens=_int(d.get("output_tokens")),
            cache_read_tokens=_int(d.get("cache_read_tokens")),
            cache_write_tokens=_int(d.get("cache_write_tokens")),
            cost_usd=_num(d.get("cost_usd")),
            pinned=bool(d.get("pinned")),
            tags=tags,
            extra={k: v for k, v in d.items()
                   if isinstance(k, str) and k != "type" and k not in _META_FIELDS},
        )


#: Header keys this version owns; everything else in a header is passed through.
_META_FIELDS = frozenset(f.name for f in dataclass_fields(SessionMeta)) - {"extra"}


# ------------------------------------------------------------------------- locks


class _PathLock:
    __slots__ = ("rlock", "users", "fd")

    def __init__(self):
        self.rlock = threading.RLock()
        self.users = 0      # holders + waiters; the entry is dropped when it hits 0
        self.fd = None


_locks: dict = {}
_locks_guard = threading.Lock()

#: Lock failures that mean "this machine is out of something", not "this
#: filesystem cannot lock". Degrading to in-process exclusion on one of these
#: would drop cross-process safety exactly when the machine is under stress —
#: and the descriptor leak this module used to have made EMFILE *reachable*.
_LOCK_FATAL_ERRNOS = frozenset(
    e for e in (getattr(errno, n, None)
                for n in ("EMFILE", "ENFILE", "ENOSPC", "EDQUOT")) if e is not None)


def _lock_failed(key: str, exc: OSError) -> None:
    """Degrade to in-process exclusion, or re-raise if degrading is not honest."""
    if exc.errno in _LOCK_FATAL_ERRNOS:
        raise exc
    warnings.warn("could not lock %s.lock (%s); writes from other processes "
                  "are no longer excluded" % (key, exc), StoreWarning, stacklevel=5)


def _lock_acquire(fd) -> None:
    """Take the OS lock on `fd`, or raise :class:`OSError`.

    Failure must reach the caller. ``msvcrt.locking(LK_LOCK)`` retries for about
    ten seconds and then raises; swallowing that raise would leave the caller
    writing while it believes it holds an exclusive lock.
    """
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)
    elif msvcrt is not None:  # pragma: no cover - Windows
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)


def _lock_release(fd) -> None:
    if fcntl is not None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def _file_lock(path):
    """Cross-process exclusive lock for one session.

    The lock lives in a sidecar ``.lock`` file rather than the data file: a meta
    rewrite replaces the data file's inode, and a lock held on the old inode would
    stop excluding anyone.

    Deliberately *not* re-entrant. ``flock()`` is a property of the open file
    description, so a nested acquire would open a second fd and block against
    itself forever; the per-path ``RLock`` would let that nesting through in
    silence, so it is detected and raised instead of hanging.
    """
    # realpath the *directory*, not the file: two names for one sessions
    # directory (a symlinked $LUME_HOME, /var vs /private/var) are one session,
    # and keying them apart put two open file descriptions on one .lock inode —
    # so a nested acquire blocked on itself for ever instead of being refused,
    # and a lock that had to degrade to in-process exclusion excluded nobody.
    # The final component is left alone: a session file that *is* a symlink is
    # refused by every caller, and resolving it would put the sidecar beside
    # somebody else's file.
    full = os.path.abspath(str(path))
    key = os.path.join(os.path.realpath(os.path.dirname(full)), os.path.basename(full))
    with _locks_guard:
        pl = _locks.get(key)
        if pl is None:
            pl = _locks[key] = _PathLock()
        pl.users += 1           # pins the entry while anyone holds or waits for it
    try:
        pl.rlock.acquire()
        try:
            if pl.fd is not None:  # only reachable from this thread: we hold the RLock
                raise RuntimeError("_file_lock(%s) is not re-entrant" % key)
            fd = None
            try:
                fd = os.open(key + ".lock", os.O_RDWR | os.O_CREAT | _O_NOFOLLOW | _O_BINARY,
                             FILE_MODE)
            except OSError as exc:
                _lock_failed(key, exc)      # degrade to in-process exclusion only
            else:
                try:
                    _lock_acquire(fd)
                except BaseException as exc:
                    # Not ``except OSError``. ``flock()`` blocks, Ctrl-C is how a
                    # user cancels a TUI, and a KeyboardInterrupt raised inside the
                    # blocking call skipped every close path here: one descriptor
                    # leaked per interrupted acquire until the process hit EMFILE,
                    # at which point the open above failed and the store quietly
                    # stopped excluding other processes for the rest of its life.
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    fd = None
                    if not isinstance(exc, OSError):
                        raise
                    _lock_failed(key, exc)
            pl.fd = fd
            try:
                yield
            finally:
                if pl.fd is not None:
                    _lock_release(pl.fd)
                    with contextlib.suppress(OSError):
                        os.close(pl.fd)
                    pl.fd = None
        finally:
            pl.rlock.release()
    finally:
        with _locks_guard:       # the table must not grow once per session touched
            pl.users -= 1
            if pl.users <= 0 and _locks.get(key) is pl:
                del _locks[key]


# ------------------------------------------------------------------- file access


def _chmod(path, mode: int) -> None:
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(path, mode)


def _fsync_dir(path) -> None:
    """Make a create/replace/unlink durable. Not supported everywhere; optional."""
    if not hasattr(os, "O_DIRECTORY"):  # pragma: no cover - Windows
        return
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _no_session(session_id, exc) -> KeyError:
    """Turn an unusable session path into a :class:`KeyError`.

    ``load('x')`` where ``x.jsonl`` is a directory, or has mode 0, used to raise
    a raw ``IsADirectoryError`` / ``PermissionError`` at the caller. Callers of a
    session API expect "no such session"; the operating system's reason is kept
    in the message and in ``__cause__``.
    """
    return KeyError("%s: %s" % (session_id, exc))


def _host_tag() -> str:
    """A short filename-safe name for this machine. Never contains ``-``."""
    name = ""
    uname = getattr(os, "uname", None)
    if uname is not None:
        with contextlib.suppress(OSError, AttributeError):
            name = uname().nodename
    if not name:
        name = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or ""
    return "".join(ch for ch in name.lower() if ch.isascii() and ch.isalnum())[:12] or "nohost"


#: Identifies this machine in temp-file names. See :func:`_owner_from_temp_name`.
_HOST = _host_tag()


def _tmp_prefix() -> str:
    """Temp-file prefix carrying host and pid, so litter can be attributed."""
    return "%s%s-%d-" % (_TMP_PREFIX, _HOST, os.getpid())


def _owner_from_temp_name(name: str):
    """The ``(host, pid)`` a temp file's name claims, or ``(None, None)``.

    The host matters. ``os.kill(pid, 0)`` answers a question about *this*
    machine, so a pid written by another host — a store on an NFS or SMB home
    directory, or another pid namespace — reads back as ProcessLookupError, i.e.
    "definitely dead". Two machines sharing a store therefore deleted each
    other's in-flight rewrites 15 seconds in, and the victim's ``os.replace``
    came back as FileNotFoundError out of ``update()``.
    """
    parts = name[len(_TMP_PREFIX):].split("-")
    if len(parts) < 2:
        return None, None
    host, pid = parts[0], parts[1]
    if not (host.isascii() and host.isalnum()):
        return None, None
    if not (pid.isascii() and pid.isdigit() and len(pid) <= 9):
        return None, None       # an older build's name: no host, so no pid check
    return host, int(pid)


def _pid_is_gone(pid) -> bool:
    """True only when we are sure no such process exists any more."""
    if not pid or not hasattr(os, "kill"):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False        # EPERM: alive and owned by somebody else
    return False


def _write_all(fd, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _open_read(path) -> int:
    """Open for reading, refusing to follow a symlink at the final component.

    ``O_NONBLOCK`` because a FIFO left under a session name would otherwise park
    the process inside ``os.open`` until somebody opened the write end — the way
    ``load()`` and ``export()`` used to hang where ``append()`` raised KeyError.
    It has no effect on a regular file.
    """
    try:
        return os.open(str(path), os.O_RDONLY | _O_NOFOLLOW | _O_BINARY | _O_NONBLOCK)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError("refusing to follow symlink: %s" % path) from e
        raise


def _read_bytes(path) -> bytes:
    if os.path.islink(str(path)):
        raise ValueError("refusing to follow symlink: %s" % path)
    fd = _open_read(path)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _line_reader(fd, chunk_size: int):
    try:
        buf = b""
        while True:
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                parts = buf.split(b"\n")
                buf = parts.pop()
                for line in parts:
                    yield line
        if buf:
            yield buf           # a last line a crash cut short
    finally:
        os.close(fd)


def _iter_lines(path, chunk_size: int = 1 << 18):
    """Yield a file's lines without holding the whole file in memory.

    Reading a 200 MB transcript whole, splitting it whole and parsing it whole —
    which is what a header rewrite used to do — took peak RSS to several times
    the size of the file. Nothing on the write path needs more than one line at
    a time.
    """
    if os.path.islink(str(path)):
        raise ValueError("refusing to follow symlink: %s" % path)
    return _line_reader(_open_read(path), chunk_size)


def _meta_record(line: bytes):
    """`line` parsed as a meta header, or None if it is not one.

    The substring test first, for the same reason as :func:`_usage_record`: a
    compaction asks this of every line it copies and must not pay a JSON parse
    per line to do it.
    """
    if b'"meta"' not in line:
        return None
    try:
        obj = json.loads(line.decode("utf-8", "replace"))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and obj.get("type") == "meta" else None


def _usage_record(line: bytes):
    """`line` parsed as a usage checkpoint, or None if it is not one.

    The substring test first: scanning a transcript for checkpoints then costs a
    memchr per line instead of a JSON parse per line.
    """
    if b'"usage"' not in line:
        return None
    try:
        obj = json.loads(line.decode("utf-8", "replace"))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and obj.get("type") == "usage" else None


class _Scan:
    """What one pass over a session file found, without keeping the messages."""

    __slots__ = ("header", "count", "last_ts", "title", "usage", "skipped", "reason")

    def __init__(self, header=None, count: int = 0):
        self.header = header
        self.count = count
        self.last_ts = 0.0
        self.title = ""         # the first user message with any text in it
        self.usage = None
        self.skipped = 0
        self.reason = ""


def _fold_line(scan: _Scan, line: bytes):
    """Fold one raw line into `scan`; return it if it is a message record."""
    if not line.strip():
        return None
    try:
        obj = json.loads(line.decode("utf-8", "replace"))
        if not isinstance(obj, dict):
            raise ValueError("record is not an object")
    except Exception as e:  # truncated tail, binary junk, half-written line
        scan.skipped += 1
        scan.reason = scan.reason or str(e)
        return None
    kind = obj.get("type")
    if kind == "meta":
        if scan.header is None:
            scan.header = obj
        else:
            scan.skipped += 1
            scan.reason = scan.reason or "duplicate meta header"
        return None
    if kind == "usage":
        scan.usage = obj        # the newest valid checkpoint wins
        return None
    if kind == "message" and isinstance(obj.get("role"), str) and "content" in obj:
        scan.count += 1
        ts = _num(obj.get("ts"))
        if ts > scan.last_ts:
            scan.last_ts = ts
        if not scan.title and obj.get("role") == "user":
            # The first user message that *has* a title, not simply the first
            # one: an opening "   " would otherwise leave the session with no
            # title for good, and leave _maybe_autotitle re-reading the whole
            # transcript on every later turn looking for one it cannot find.
            scan.title = _clean_title(obj.get("content"))
        return obj
    scan.skipped += 1
    scan.reason = scan.reason or "unrecognised record"
    return None


def _scan_meta(path) -> _Scan:
    """One streaming pass, keeping the counters and dropping the messages."""
    scan = _Scan()
    for line in _iter_lines(path):
        _fold_line(scan, line)
    return scan


def _read_records(path):
    """Parse a session file whole: ``(scan, messages)``.

    Damaged lines are counted on the scan, never fatal: the caller decides
    whether to warn. Only :meth:`Store.load` and a content search need the
    message records themselves; everything else uses :func:`_scan_meta`.
    """
    scan = _Scan()
    messages = []
    for line in _iter_lines(path):
        obj = _fold_line(scan, line)
        if obj is not None:
            messages.append(obj)
    return scan, messages


def _read_header(path, limit: int = _MAX_HEADER_BYTES):
    """Parse just line 1 of a session file, or None if it is absent/unparsable.

    Read in chunks up to the first newline, not in one fixed-size gulp. A 64 KB
    gulp meant that pasting a long system prompt dropped the session off the
    :meth:`Store.list` fast path *and* stopped a title ever being persisted for
    it — permanently, silently, and with no way for the user to connect the two.
    """
    if os.path.islink(str(path)):
        raise ValueError("refusing to follow symlink: %s" % path)
    fd = _open_read(path)
    chunks = []
    got = 0
    # Start small and grow. A header is a few hundred bytes, and list() reads one
    # per session: gulping 64 KB per session to find it read two orders of
    # magnitude more than it used, and a long system prompt is still served by
    # the second read.
    want = 1 << 12
    try:
        while got < limit:
            chunk = os.read(fd, min(want, limit - got))
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
            if b"\n" in chunk:
                break
            want = min(want * 4, 1 << 16)
    finally:
        os.close(fd)
    head = b"".join(chunks)
    nl = head.find(b"\n")
    if nl < 0:
        return None
    try:
        obj = json.loads(head[:nl].decode("utf-8", "replace"))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and obj.get("type") == "meta" else None


def _tail_usage(path, size: int, window: int = _INDEX_TAIL):
    """The newest usage checkpoint near the end of a file.

    Returns ``(checkpoint_or_None, decisive)``. `decisive` is True when the
    answer is the whole answer: either a checkpoint was found — in which case it
    is the newest one, because they are appended in order — or the window
    reached byte 0, so there is none anywhere.

    This is what lets :meth:`Store.list` derive a session's token and cost totals
    from the file itself for the price of one small read, instead of trusting a
    number in the index cache. It never falls back to a full pass: a listing of a
    thousand sessions must not turn into a thousand full reads because a long
    message happens to sit between the tail and the newest checkpoint.
    """
    start = max(0, size - window)
    fd = _open_read(path)
    try:
        data = _pread(fd, size - start, start) if size > start else b""
    finally:
        os.close(fd)
    if len(data) != size - start:
        return None, False              # the file changed under us
    if start:
        nl = data.find(b"\n")
        if nl < 0:
            return None, False          # one line longer than the whole window
        data = data[nl + 1:]
    for line in reversed(data.split(b"\n")):
        found = _usage_record(line)
        if found is not None:
            return found, True
    return None, start == 0


def _row_covers(row, meta: SessionMeta) -> bool:
    """True if a cached totals row is at least what the header itself proves.

    The only writer of header totals is a compaction folding checkpoints into
    them, so a file's totals can never be below its header's, and a cached row
    that is lower is stale or forged. ``count`` has had an arithmetic bound
    against the file size since it was first cached; the money figure had none
    at all, and a zeroed row reported a session's whole cost as $0.00.
    """
    for value, key in zip(row, _USAGE_KEYS):
        if max(0, _int(value)) < getattr(meta, key):
            return False
    return max(0.0, _num(row[4])) >= meta.cost_usd


def _apply_usage(meta: SessionMeta, usage) -> None:
    """Overlay the newest usage checkpoint's totals on the header's — upwards only.

    A checkpoint carries absolute totals, so folding it is an assignment rather
    than an addition: that is what makes re-reading one idempotent. But the only
    thing that ever writes totals into the header is a compaction folding the
    checkpoints *in*, so a file's true totals can never be below what its header
    already says, and a checkpoint claiming less than the header is therefore
    always wrong. Believing one was silent and permanent: ``load()`` reported the
    smaller number, and the next ``update()`` baked it into the header and
    dropped the checkpoint it came from, so a single flipped digit — or an old
    build that rewrote the header and left a stale checkpoint behind it — cost
    the whole token and cost history with nothing left to recover it from.

    A caller that really means to lower a total sets it explicitly through
    :meth:`Store.update`, which writes the header and drops the checkpoints.
    """
    if not isinstance(usage, dict):
        return
    for key in _USAGE_KEYS:
        if key in usage:
            setattr(meta, key, max(max(0, _int(getattr(meta, key, 0))),
                                   max(0, _int(usage.get(key)))))
    if "cost_usd" in usage:
        meta.cost_usd = max(max(0.0, _num(meta.cost_usd)),
                            max(0.0, _num(usage.get("cost_usd"))))


def _meta_from(scan: _Scan, session_id: str, mtime: float = 0.0) -> SessionMeta:
    """Build meta from the stored header, with counters derived from the file.

    Derivation is what makes an append a *pure* append: the header does not have
    to be rewritten for the message count or the modification time to be right,
    and a session whose header was lost still lists sensibly. Token totals work
    the same way — the newest usage checkpoint supersedes the header — and the
    counts a checkpoint carries for itself are ignored here, because messages we
    have just counted beat a number some writer once claimed.
    """
    meta = SessionMeta.from_dict(scan.header if isinstance(scan.header, dict) else {})
    meta.id = session_id
    meta.message_count = scan.count
    if not meta.created:
        meta.created = scan.last_ts or mtime or time.time()
    meta.updated = max(meta.updated, meta.created, scan.last_ts)
    if not meta.title:
        meta.title = scan.title or ""
    _apply_usage(meta, scan.usage)
    return meta


def _drop_from_body(line: bytes) -> bool:
    """True for a line the new header replaces: an old header, or a checkpoint."""
    return _meta_record(line) is not None or _usage_record(line) is not None


def _copy_body(rfd, wfd, buffer_size: int = 1 << 18) -> None:
    """Copy `rfd` to `wfd`, dropping the old header and the usage checkpoints.

    Every other line is copied byte-for-byte, including a trailing line that a
    crash left without its newline, and including lines this build does not
    understand.

    Which line is dropped is decided by *parsing* it, not by its position. This
    used to drop line 1 unconditionally, on the reasoning that line 1 is the
    header — but the header is only wherever it happens to be. A file whose
    header had moved (a crash between two writers, a hand-edited transcript, a
    header a reader had to reconstruct) lost a real message to every rename, and
    lost it silently: no warning, and no rescue copy, because as far as
    :meth:`Store._meta_for_rewrite` was concerned the header was perfectly
    readable. Dropping every header line also keeps a rewrite idempotent — a
    file that somehow holds two never grows a third.
    """
    out = bytearray()
    buf = b""

    def flush():
        if out:
            _write_all(wfd, bytes(out))
            del out[:]

    while True:
        chunk = os.read(rfd, buffer_size)
        if not chunk:
            break
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line, buf = buf[:nl], buf[nl + 1:]
            if _drop_from_body(line):
                continue
            out += line
            out += b"\n"
            if len(out) >= buffer_size:
                flush()
    if buf and not _drop_from_body(buf):
        out += buf
    flush()


def _file_digest(path) -> bytes:
    """SHA-256 of a file, read a chunk at a time."""
    digest = hashlib.sha256()
    fd = _open_read(path)
    try:
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.digest()


# ------------------------------------------------------------------------- store


class _Doc:
    """A document assembled a piece at a time, without a copy of itself.

    Trailing whitespace is held back rather than written, so the finished text
    needs no ``rstrip()`` — which would have meant a second full copy of the
    document at the moment the first was complete, which is the whole thing this
    class exists to avoid.
    """

    __slots__ = ("buf", "held")

    def __init__(self):
        self.buf = io.StringIO()
        self.held = ""

    def write(self, text: str) -> None:
        if not text:
            return
        body = text.rstrip()
        if not body:
            self.held += text
            return
        if self.held:
            self.buf.write(self.held)
        self.buf.write(body)
        self.held = text[len(body):]

    def finish(self) -> str:
        self.buf.write("\n")
        return self.buf.getvalue()

    def close(self) -> None:
        self.buf.close()


class Store:
    """Append-only, local-only session storage rooted at a single directory."""

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = pathlib.Path(root).expanduser() if root is not None else default_root()
        self.sessions_dir = self.root / "sessions"
        self.index_path = self.sessions_dir / INDEX_NAME
        self._dirs_ready = False
        self._ensure_dirs()

    def __repr__(self) -> str:  # no secrets here, but keep it terse
        return "Store(root=%r)" % str(self.root)

    # -- layout

    def _ensure_dirs(self) -> None:
        if self._dirs_ready and self.sessions_dir.is_dir():
            return
        for d in (self.root, self.sessions_dir):
            if not d.is_dir():
                os.makedirs(d, mode=DIR_MODE, exist_ok=True)
                _chmod(d, DIR_MODE)  # makedirs' mode is filtered by the umask
        self._dirs_ready = True

    def path_for(self, session_id: str) -> pathlib.Path:
        """Absolute path of a session file.

        Raises :class:`ValueError` for any id that could name something outside
        the store — traversal, absolute paths, NUL, separators, or a symlink
        planted in the sessions directory.
        """
        sid = _validate_id(session_id)
        path = self.sessions_dir / (sid + EXT)
        real = os.path.realpath(str(path))
        root = os.path.realpath(str(self.sessions_dir))
        if not _is_within(real, root) or os.path.basename(real) != sid + EXT:
            raise ValueError("session id escapes the store root: %r" % session_id)
        return path

    def exists(self, session_id: str) -> bool:
        """True if `session_id` names a readable session file."""
        try:
            path = self.path_for(session_id)
        except ValueError:
            return False
        return path.is_file() and not path.is_symlink()

    # -- writing

    def create(self, *, model: str, system: str | None = None,
               title: str | None = None) -> SessionMeta:
        """Create a new session file and return its meta."""
        self._ensure_dirs()
        now = time.time()
        meta = SessionMeta(
            id="", title=_clean_title(title), created=now, updated=now,
            model=_text(model), system=system if isinstance(system, str) else None,
        )
        for _ in range(8):  # id collision is ~impossible; loop instead of trusting that
            meta.id = new_id()
            path = self.path_for(meta.id)
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | _O_NOFOLLOW | _O_BINARY, FILE_MODE)
            except FileExistsError:
                continue
            try:
                _write_all(fd, _dumps_line({"type": "meta", **meta.to_dict()}))
                os.fsync(fd)
            except BaseException:
                os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(str(path))
                raise
            os.close(fd)
            _chmod(path, FILE_MODE)
            _fsync_dir(self.sessions_dir)
            return meta
        raise RuntimeError("could not allocate a unique session id")

    def append(self, session_id: str, message: Message) -> None:
        """Append one message. Atomic against other threads and processes."""
        if not isinstance(message, Message):
            raise TypeError("message must be a Message, got %s" % type(message).__name__)
        path = self.path_for(session_id)
        # Before the lock: a lock taken for an id that does not exist leaves a
        # .lock sidecar and a process-global table entry behind for nothing.
        self._require_session(session_id, path)
        line = _dumps_line({"type": "message", **message.to_dict()})
        with _file_lock(path):
            try:
                fd = os.open(str(path), os.O_RDWR | os.O_APPEND | _O_NOFOLLOW | _O_BINARY)
            except FileNotFoundError:
                raise KeyError(session_id) from None
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.EMLINK):   # O_NOFOLLOW fired
                    raise ValueError("refusing to follow symlink: %s" % path) from exc
                raise _no_session(session_id, exc) from exc
            try:
                info = os.fstat(fd)
                if info.st_nlink > 1:
                    # O_NOFOLLOW and islink() only see symlinks. A hardlink from
                    # the sessions directory to a file outside it would otherwise
                    # be appended to through this name.
                    raise ValueError("refusing to write a hardlinked session file: %s" % path)
                size = info.st_size
                if size and _pread(fd, 1, size - 1) != b"\n":
                    # A previous crash left a half-written line: terminate it so the
                    # damage stays confined to that one line.
                    _write_all(fd, b"\n")
                    warnings.warn("repaired a truncated line in %s" % path,
                                  StoreWarning, stacklevel=2)
                _write_all(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        if message.role == "user" and _clean_title(message.content):
            # A cheap gate on the message in hand, so that a transcript of
            # whitespace never costs a full read per turn. What the title *is*
            # is still decided by the file, under the lock, in _maybe_autotitle.
            self._maybe_autotitle(path)

    def _maybe_autotitle(self, path) -> None:
        """Persist a title derived from the first user message, if there is none.

        :func:`_meta_from` derives this title at *read* time whatever happens
        here, so this is not what makes ``load()`` show a title — it only makes
        the session eligible for the :meth:`list` fast path, which cannot derive
        a title without reading every message. That makes it cache warming, and
        cache warming must be cheap: the "does it already have one?" question is
        answered from line 1, never by parsing the whole transcript. Reading the
        transcript here is what made every user append cost O(file) and a
        conversation O(n²).

        The title comes from the *first user message in the file*, which is what
        a reader derives, and not from the message that happened to trigger this
        call. Stamping the triggering message made the stored title and the
        derived one disagree permanently: ``rename(sid, "")`` restores the
        derived title, and the next turn would then persist a different one.
        """
        try:
            header = _read_header(path)
        except (OSError, ValueError):
            return
        if header is None or _text(header.get("title")):
            return
        try:
            with _file_lock(path):
                scan = _scan_meta(path)
                if scan.header is None or _text(scan.header.get("title")):
                    return
                if not scan.title:
                    return
                meta = _meta_from(scan, self._id_of(path))
                meta.title = scan.title
                self._rewrite_header(path, meta)
        except (OSError, ValueError):
            return  # the message is already durable; the title is only a cache

    @staticmethod
    def _id_of(path) -> str:
        name = os.path.basename(str(path))
        return name[:-len(EXT)] if name.endswith(EXT) else name

    @staticmethod
    def _require_session(session_id, path, regular: bool = True) -> None:
        """Fail *before* locking if `path` is not a usable session file."""
        try:
            info = os.lstat(str(path))
        except FileNotFoundError:
            raise KeyError(session_id) from None
        except OSError as exc:
            raise _no_session(session_id, exc) from exc
        if stat_mod.S_ISLNK(info.st_mode):
            raise ValueError("refusing to follow symlink: %s" % path)
        if regular and not stat_mod.S_ISREG(info.st_mode):
            # A real OSError, not a string: callers (and tests) read the reason
            # out of __cause__, and a FIFO or a directory here is exactly the
            # case where the reason is the whole story.
            code = errno.EISDIR if stat_mod.S_ISDIR(info.st_mode) else errno.EINVAL
            exc = OSError(code, "not a regular file", str(path))
            raise _no_session(session_id, exc) from exc

    @staticmethod
    def _rescue_names(path):
        """Candidate rescue names for `path`, oldest event first.

        The numbered names run out after a hundred damage events, and running out
        used to end the session: :meth:`_meta_for_rewrite` refuses to rewrite a
        damaged header it could not copy aside first, so ``update``, ``rename``,
        pinning and re-modelling all died with a ``KeyError`` that read like a
        missing session while ``load`` and ``append`` carried on working. The
        timestamped names after them are the way out; they are never reused, so
        the hundred-and-first rescue is as safe as the first.
        """
        yield str(path) + ".bak"
        for n in range(1, 100):
            yield "%s.bak.%d" % (path, n)
        for _ in range(8):
            yield "%s.bak.%d" % (path, time.time_ns())

    def _rescue_copy(self, path):
        """Copy `path` aside before a write that would lose bytes we cannot read.

        Returns ``(name, made_a_new_copy)``.

        The first copy is the valuable one, so it is never overwritten — but a
        *second* damage event brings a second set of irreplaceable bytes, and
        returning early because ``<id>.jsonl.bak`` already existed threw those
        away while the warning said they had been kept. Every event that has new
        bytes to keep therefore gets its own name, and an existing copy is
        reused only when it is byte-for-byte what we were about to write.

        "Does a copy exist" is also not an ``lexists`` question: an empty file, a
        directory, a symlink and a dangling symlink all satisfy it, and any of
        them used to disable the rescue completely.
        """
        try:
            size = os.stat(str(path)).st_size
        except OSError:
            size = -1
        digest = None
        for dest in self._rescue_names(path):
            try:
                info = os.lstat(dest)
            except FileNotFoundError:
                return self._copy_aside(path, dest), True
            except OSError:
                continue
            # Size first: a copy of a different length cannot be a copy of these
            # bytes, and hashing the transcript once per existing rescue name —
            # under the session lock — turned a damaged header into a hundred
            # full reads of the file.
            if stat_mod.S_ISREG(info.st_mode) and info.st_size > 0 and info.st_size == size:
                with contextlib.suppress(OSError, ValueError):
                    if digest is None:
                        digest = _file_digest(path)
                    if _file_digest(dest) == digest:
                        return dest, False      # these exact bytes are already safe
        raise OSError("no free rescue name beside %s" % path)

    def _copy_aside(self, src, dest: str) -> str:
        """Stream `src` to a fresh file and link it into place as `dest`."""
        fd, tmp = tempfile.mkstemp(prefix=_tmp_prefix(), suffix=".bak",
                                   dir=str(self.sessions_dir))
        try:                                    # mkstemp always creates 0600
            rfd = _open_read(src)
            try:
                while True:
                    chunk = os.read(rfd, 1 << 20)
                    if not chunk:
                        break
                    _write_all(fd, chunk)
            finally:
                os.close(rfd)
            os.fsync(fd)
            os.close(fd)
            fd = None
            try:
                os.link(tmp, dest)      # never clobbers, even under a race
            except OSError:
                if os.path.lexists(dest):
                    return dest         # somebody else won; their copy is the same
                os.replace(tmp, dest)
                tmp = None
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
        _fsync_dir(self.sessions_dir)
        return dest

    def _meta_for_rewrite(self, session_id: str, path) -> SessionMeta:
        """Read a session for a header rewrite, protecting a damaged header.

        Reconstructing an unreadable header is right for *reads*; persisting the
        reconstruction is not. The reconstruction is all defaults, so writing it
        back would replace a title, a system prompt and a token/cost history that
        are merely unparsed with ones that are genuinely gone. The original bytes
        are copied aside first and the loss is announced.
        """
        try:
            scan = _scan_meta(path)
        except FileNotFoundError:
            raise KeyError(session_id) from None
        except OSError as exc:
            raise _no_session(session_id, exc) from exc
        if scan.header is None and (scan.count or scan.skipped or scan.usage):
            try:
                backup, made = self._rescue_copy(path)
            except OSError as exc:
                # No rescue copy means no rewrite: losing the header is not an
                # acceptable side effect of a rename.
                raise _no_session(session_id, "cannot rescue a damaged header: %s" % exc) from exc
            warnings.warn(
                "%s has no readable meta header: the stored title, model, token "
                "totals and cost cannot be recovered, and this write replaces them "
                "with defaults. %s" % (path, "The file as it was is kept at %s." % backup
                                       if made else
                                       "The same bytes were already kept at %s." % backup),
                StoreWarning, stacklevel=3)
        return _meta_from(scan, session_id)

    def _rewrite_header(self, path, meta: SessionMeta) -> None:
        """Replace line 1 in place, keeping every message byte-for-byte.

        Caller must hold the session lock. The new file is fully written and
        fsynced before :func:`os.replace` swaps it in, so an interrupted rewrite
        leaves the previous file intact and never a partial one. The body is
        streamed rather than read into memory: one ``update()`` on a 200 MB
        transcript used to take RSS from 22 MB to nearly 600 MB.

        Usage checkpoints are dropped on the way past. `meta` comes from a full
        read, so it already carries their totals; leaving them in the file would
        count every one of them a second time.
        """
        head = _dumps_line({"type": "meta", **meta.to_dict()})
        fd, tmp = tempfile.mkstemp(prefix=_tmp_prefix(), suffix=EXT, dir=str(self.sessions_dir))
        try:                                    # mkstemp always creates 0600
            _write_all(fd, head)
            rfd = _open_read(path)
            try:
                _copy_body(rfd, fd)
            finally:
                os.close(rfd)
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(tmp, str(path))
            tmp = None
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
        _fsync_dir(self.sessions_dir)

    def update(self, session_id: str, **fields) -> SessionMeta:
        """Update mutable header fields (title, model, system, pinned, tags, counters).

        ``updated`` is *not* bumped implicitly: renaming or pinning a session must
        not reorder the list as if it had new activity.
        """
        sid = _validate_id(session_id)
        path = self.path_for(sid)
        self._require_session(sid, path)
        with _file_lock(path):
            meta = self._meta_for_rewrite(sid, path)
            for key, value in fields.items():
                self._apply_field(meta, key, value)
            self._rewrite_header(path, meta)
        return meta

    @staticmethod
    def _apply_field(meta: SessionMeta, key: str, value) -> None:
        if key in ("title",):
            meta.title = _clean_title(value)
        elif key in ("model",):
            meta.model = _text(value)
        elif key in ("system",):
            meta.system = value if isinstance(value, str) else None
        elif key in ("pinned",):
            meta.pinned = bool(value)
        elif key in ("tags",):
            if not isinstance(value, (list, tuple)):
                raise ValueError("tags must be a list of strings")
            meta.tags = [_clean_title(t, 64) for t in value if isinstance(t, str)][:32]
        elif key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            setattr(meta, key, max(0, _int(value)))
        elif key in ("cost_usd",):
            meta.cost_usd = max(0.0, _num(value))
        elif key in ("created",):
            meta.created = _num(value)
        elif key in ("updated",):
            # It used to accept this and hand back a SessionMeta showing the value
            # it was given, while every later read clamped it to the newest
            # message. A field that cannot be lowered is not a field you can set.
            raise ValueError(
                "'updated' is derived from the messages, not stored; set 'created' instead")
        else:
            raise ValueError("unknown or derived field: %r" % key)

    def rename(self, session_id: str, title: str) -> SessionMeta:
        """Set the title. An empty title restores the auto-derived one."""
        return self.update(session_id, title=title)

    def record_usage(self, session_id: str, usage: dict, cost_usd: float) -> SessionMeta:
        """Add one turn's token usage and cost to the session totals.

        Appends a *usage checkpoint* — one more line carrying the running totals
        — instead of rewriting the header. The app calls this after every
        assistant reply, and rewriting there made a conversation quadratic: turn
        N re-read, re-parsed and re-wrote turns 1..N-1, so 600 turns wrote
        809 MB for a 2.7 MB transcript and each turn cost more than the last. An
        append costs the same at any file size, and it is also the durable
        shape: a crash mid-checkpoint costs one turn's counters, never a
        message, and never the transcript ``os.replace`` was about to swap.

        Readers fold the newest checkpoint over the header's totals; the next
        header rewrite folds them in for real and drops them from the file.
        """
        usage = usage if isinstance(usage, dict) else {}

        def pick(*names):
            for n in names:
                if n in usage:
                    return max(0, _int(usage.get(n)))
            return 0

        sid = _validate_id(session_id)
        path = self.path_for(sid)
        self._require_session(sid, path)
        with _file_lock(path):
            try:
                fd = os.open(str(path), os.O_RDWR | os.O_APPEND | _O_NOFOLLOW | _O_BINARY)
            except FileNotFoundError:
                raise KeyError(sid) from None
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.EMLINK):   # O_NOFOLLOW fired
                    raise ValueError("refusing to follow symlink: %s" % path) from exc
                raise _no_session(sid, exc) from exc
            try:
                info = os.fstat(fd)
                if info.st_nlink > 1:
                    raise ValueError("refusing to write a hardlinked session file: %s" % path)
                size = info.st_size
                scan = self._counters_at(path, fd, size)
                meta = _meta_from(scan, sid)
                meta.input_tokens += pick("input_tokens", "in")
                meta.output_tokens += pick("output_tokens", "out")
                meta.cache_write_tokens += pick("cache_creation_input_tokens",
                                                "cache_write_tokens")
                meta.cache_read_tokens += pick("cache_read_input_tokens", "cache_read_tokens")
                meta.cost_usd = round(meta.cost_usd + max(0.0, _num(cost_usd)), 10)
                line = _dumps_line({
                    "type": "usage", "ts": time.time(),
                    # n/mts are what this writer believed about the messages
                    # before it, so that the next checkpoint can carry on from
                    # here without re-reading them. A reader that has counted
                    # the records itself ignores both.
                    "n": meta.message_count, "mts": scan.last_ts,
                    "input_tokens": meta.input_tokens,
                    "output_tokens": meta.output_tokens,
                    "cache_read_tokens": meta.cache_read_tokens,
                    "cache_write_tokens": meta.cache_write_tokens,
                    "cost_usd": meta.cost_usd,
                })
                if size and _pread(fd, 1, size - 1) != b"\n":
                    # A previous crash left a half-written line: terminate it so
                    # the damage stays confined to that one line.
                    _write_all(fd, b"\n")
                    warnings.warn("repaired a truncated line in %s" % path,
                                  StoreWarning, stacklevel=2)
                _write_all(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        return meta

    @staticmethod
    def _counters_at(path, fd, size: int) -> _Scan:
        """The counters a reader would derive, without reading the whole file.

        Walks backwards from the end to the newest usage checkpoint and folds
        only the records written after it. That window is one turn's worth of
        bytes whatever the transcript weighs, which is what keeps
        :meth:`record_usage` flat. If no checkpoint turns up within
        ``_TAIL_CAP`` — the first checkpoint of a long session, or the first
        after a header rewrite dropped them — it falls back to a full pass:
        correct, and at worst once per session.

        The header is read separately (line 1, one read) because the window
        rarely reaches it. A title derived from a first user message outside the
        window is not seen here; nothing persists this meta, and
        :meth:`_maybe_autotitle` has already put that title in the header.
        """
        scan = _Scan()
        # One turn's worth of bytes, not a flat 64 KB: the answer is a ~150 byte
        # checkpoint sitting behind the messages of the turn that wrote it, and
        # gulping 64 KB to find it read 14x more of the file than it used. The
        # window still grows when a turn really is bigger than this.
        chunk = 1 << 14
        pos = size
        buf = b""
        while pos > 0:
            start = max(0, pos - chunk)
            data = _pread(fd, pos - start, start)
            if len(data) != pos - start:
                break                       # short read: fall back to a full pass
            buf = data + buf
            pos = start
            nl = buf.find(b"\n")
            if pos and nl < 0:              # no complete line in the window yet
                if size - pos >= _TAIL_CAP:
                    break
                chunk = min(chunk * 4, 1 << 22)
                continue
            lines = (buf if pos == 0 else buf[nl + 1:]).split(b"\n")
            for i in range(len(lines) - 1, -1, -1):
                found = _usage_record(lines[i])
                if found is not None:
                    scan.usage = found
                    for line in lines[i + 1:]:
                        _fold_line(scan, line)
                    scan.count += max(0, _int(found.get("n")))
                    scan.last_ts = max(scan.last_ts, _num(found.get("mts")))
                    with contextlib.suppress(OSError, ValueError):
                        scan.header = _read_header(path)
                    return scan
            if pos == 0:                    # no checkpoint anywhere: the header wins
                for line in lines:
                    _fold_line(scan, line)
                return scan
            if size - pos >= _TAIL_CAP:
                break
            chunk = min(chunk * 4, 1 << 22)
        return _scan_meta(path)

    def delete(self, session_id: str) -> None:
        """Remove a session permanently, with its lock and rescue sidecars."""
        path = self.path_for(session_id)
        try:
            # Not `regular=True`: a symlink planted under a session name is
            # something delete() should be able to clear away, and unlink()
            # never follows one.
            self._require_session(session_id, path, regular=False)
        except KeyError:
            # The transcript is gone but a rescue copy of it may not be, and a
            # rescue copy is a full plaintext transcript. Refusing here before
            # sweeping anything left the one file a privacy-minded user most
            # wants gone with no way to remove it through this API at all.
            self._remove_sidecars(path)
            _fsync_dir(self.sessions_dir)
            raise
        with _file_lock(path):
            try:
                os.unlink(str(path))
            except FileNotFoundError:
                raise KeyError(session_id) from None
            except OSError as exc:
                raise _no_session(session_id, exc) from exc
        self._remove_sidecars(path)
        _fsync_dir(self.sessions_dir)

    def _remove_sidecars(self, path) -> None:
        """Delete every ``.lock`` / ``.bak`` / ``.bak.N`` file beside `path`."""
        base = os.path.basename(str(path))
        try:
            names = os.listdir(str(self.sessions_dir))
        except OSError:
            return
        for name in names:
            if not name.startswith(base + "."):
                continue
            rest = name[len(base) + 1:]
            if rest in ("lock", "bak") or rest.startswith("bak."):
                with contextlib.suppress(OSError):
                    os.unlink(str(self.sessions_dir / name))

    # -- reading

    def load(self, session_id: str) -> tuple:
        """Return ``(meta, messages)``.

        Damaged lines are skipped with a :class:`StoreWarning`; a missing or
        unreadable header is reconstructed from the filename and the messages.
        """
        path = self.path_for(session_id)
        # Before reading: a FIFO under a session name parks os.open() until
        # somebody opens the write end. append() and update() have always
        # refused it here; load() and export() were the two that hung instead.
        self._require_session(session_id, path)
        try:
            scan, records = _read_records(path)
        except FileNotFoundError:
            raise KeyError(session_id) from None
        except OSError as exc:
            raise _no_session(session_id, exc) from exc
        meta = self._meta_after_read(session_id, path, scan, stacklevel=3)
        return meta, [Message.from_dict(r) for r in records]

    @staticmethod
    def _meta_after_read(session_id, path, scan: _Scan, stacklevel: int = 3) -> SessionMeta:
        """Announce what a read had to skip, and build the meta it derived."""
        if scan.skipped:
            warnings.warn("skipped %d damaged line(s) in %s (%s)"
                          % (scan.skipped, path, scan.reason), StoreWarning,
                          stacklevel=stacklevel)
        if scan.header is None:
            warnings.warn("missing meta header in %s; reconstructed" % path,
                          StoreWarning, stacklevel=stacklevel)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return _meta_from(scan, _validate_id(session_id), mtime)

    @staticmethod
    def _iter_messages(path):
        """Yield a session's messages one at a time, holding none of them."""
        scan = _Scan()
        for line in _iter_lines(path):
            obj = _fold_line(scan, line)
            if obj is not None:
                yield Message.from_dict(obj)

    def list(self, limit: int | None = None, query: str | None = None) -> list:
        """Sessions, pinned first then newest first.

        `query` matches the title and any message text, case-insensitively.
        """
        metas = self._all_metas()
        if query:
            needle = str(query).strip().lower()
            if needle:
                metas = [m for m in metas if self._matches(m, needle)]
        metas.sort(key=lambda m: (0 if m.pinned else 1, -m.updated, m.id))
        if limit is not None:
            metas = metas[:max(0, _limit(limit))]
        return metas

    def _matches(self, meta: SessionMeta, needle: str) -> bool:
        if needle in meta.title.lower() or needle in meta.id.lower():
            return True
        try:
            scan = _Scan()
            for line in _iter_lines(self.path_for(meta.id)):
                record = _fold_line(scan, line)
                if record is not None and needle in _text(record.get("content")).lower():
                    return True
        except (OSError, ValueError):
            return False
        return False

    def latest(self) -> SessionMeta | None:
        """The most recently active session, ignoring pinning."""
        metas = self._all_metas()
        if not metas:
            return None
        return max(metas, key=lambda m: (m.updated, m.id))

    def resolve(self, ref: str) -> str:
        """Resolve a full id, unique id prefix, 1-based list index, or ``'last'``.

        Order of interpretation: exact id, ``last``, numeric index into
        :meth:`list`, then case-insensitive id prefix. Raises :class:`KeyError`
        when nothing matches and :class:`AmbiguousRefError` when a prefix matches
        several sessions.
        """
        if not isinstance(ref, str):
            raise KeyError(ref)
        r = ref.strip()
        if not r:
            raise KeyError(ref)
        with contextlib.suppress(ValueError):
            if self.exists(r):
                return _validate_id(r)
        if r.lower() == "last":
            meta = self.latest()
            if meta is None:
                raise KeyError("no sessions yet")
            return meta.id
        n = r[1:] if r.startswith("#") else r
        # str.isdigit() is true for '²' (which int() rejects) and for '١' (which
        # int() happily reads as 1). A list index typed at a prompt is ASCII, and
        # the length cap keeps int() away from its digit-conversion limit.
        if n.isascii() and n.isdigit() and len(n) <= 9:
            rows = self.list()
            idx = int(n)
            if 1 <= idx <= len(rows):
                return rows[idx - 1].id
        low = r.lower()
        hits = sorted(m.id for m in self._all_metas() if m.id.lower().startswith(low))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AmbiguousRefError(
                "%r matches %d sessions: %s" % (ref, len(hits), ", ".join(hits[:5])))
        raise KeyError(ref)

    def export(self, session_id: str, fmt: str = "markdown") -> str:
        """Render a session as ``markdown``, ``json``, or ``text``.

        Written a record at a time. Building the document as a list of pieces and
        joining it held the messages, the pieces and the finished string all at
        once — 610 MB of resident memory to export a 200 MB transcript, on a
        machine that might not have it. The counters in the preamble are derived
        from a full pass, so the file is read twice and never held — which means
        a message appended between the two passes is rendered but not counted in
        the preamble. A transcript is append-only, so the worst of that is a
        heading one message behind the body it introduces.
        """
        fmt = (fmt or "markdown").strip().lower()
        if fmt not in ("markdown", "md", "json", "text", "txt"):
            raise ValueError("unknown export format: %r" % fmt)
        path = self.path_for(session_id)
        # Before reading: a FIFO under a session name parks os.open() until
        # somebody opens the write end.
        self._require_session(session_id, path)
        try:
            meta = self._meta_after_read(session_id, path, _scan_meta(path), stacklevel=3)
        except FileNotFoundError:
            raise KeyError(session_id) from None
        except OSError as exc:
            raise _no_session(session_id, exc) from exc
        doc = _Doc()
        try:
            if fmt == "json":
                self._export_json(doc, path, meta)
            elif fmt in ("markdown", "md"):
                self._export_markdown(doc, path, meta)
            else:
                self._export_text(doc, path, meta)
            return doc.finish()
        except FileNotFoundError:
            # Deleted from under the second pass. A caller of a session API
            # expects "no such session", not a stray OSError.
            raise KeyError(session_id) from None
        except OSError as exc:
            raise _no_session(session_id, exc) from exc
        finally:
            doc.close()

    def _export_json(self, doc, path, meta: SessionMeta) -> None:
        def block(obj, pad):
            return json.dumps(obj, ensure_ascii=False, indent=2).replace("\n", "\n" + pad)
        doc.write('{\n  "meta": ')
        doc.write(block(meta.to_dict(), "  "))
        doc.write(',\n  "messages": [')
        sep = "\n    "
        for message in self._iter_messages(path):
            doc.write(sep)
            doc.write(block(message.to_dict(), "    "))
            sep = ",\n    "
        doc.write("\n  ]\n}" if sep != "\n    " else "]\n}")

    def _export_markdown(self, doc, path, meta: SessionMeta) -> None:
        doc.write("# %s\n\n" % (meta.title or "Untitled session"))
        doc.write("*%s · %s · %d message%s · %d in / %d out · $%.4f*\n"
                  % (meta.model or "unknown model", _when(meta.created),
                     meta.message_count, "" if meta.message_count == 1 else "s",
                     meta.input_tokens, meta.output_tokens, meta.cost_usd))
        if meta.system:
            doc.write("\n> **System**\n\n")
            for line in meta.system.splitlines():
                doc.write("> %s\n" % line)
        for message in self._iter_messages(path):
            doc.write("\n---\n\n**%s** · %s\n\n"
                      % (_role_label(message.role), _when(message.ts, "%H:%M")))
            if message.thinking:
                doc.write("> *thinking*\n\n")
                for line in message.thinking.splitlines():
                    doc.write("> %s\n" % line)
                doc.write("\n")
            doc.write(message.content)
            doc.write("\n")
            if message.error:
                doc.write("\n> **error:** %s\n" % message.error)

    def _export_text(self, doc, path, meta: SessionMeta) -> None:
        doc.write("%s\n%s · %s\n%s\n"
                  % (meta.title or "Untitled session", meta.model or "unknown model",
                     _when(meta.created), "=" * 60))
        if meta.system:
            doc.write("\nSystem:\n%s\n" % meta.system)
        for message in self._iter_messages(path):
            doc.write("\n[%s] %s:\n" % (_when(message.ts, "%H:%M"),
                                        _role_label(message.role)))
            doc.write(message.content)
            doc.write("\n")
            if message.error:
                doc.write("error: %s\n" % message.error)

    # -- index cache

    def _all_metas(self) -> list:
        """Every session's meta.

        The directory — never the index — decides which sessions exist, and every
        header field (title, model, pin, tags) is re-read from the session file
        on each call. The index contributes only values that would otherwise
        cost a full parse — the message count, the last-activity time and the
        token/cost totals folded from the usage checkpoints — and only for a
        file whose size and mtime still match the ones that were counted;
        anything else triggers a full re-scan. A corrupt, stale, or absent index
        therefore costs time, not correctness.
        """
        self._ensure_dirs()
        cached = self._read_index()
        fresh = {}
        metas = []
        try:
            names = sorted(os.listdir(str(self.sessions_dir)))
        except OSError:
            return []
        live = frozenset(n for n in names if n.endswith(EXT))
        for name in names:
            if name.startswith(_TMP_PREFIX):
                self._sweep_temp(name)   # a rewrite that was killed before os.replace
                continue
            if not name.endswith(EXT):
                self._sweep_sidecar(name, live)
                continue
            sid = name[:-len(EXT)]
            try:
                _validate_id(sid)
            except ValueError:
                continue  # a file we would never have written
            path = self.sessions_dir / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                st = path.stat()
                header = _read_header(path)
            except (OSError, ValueError):
                continue
            meta = None
            entry = cached.get(sid)
            totals = entry.get("usage") if isinstance(entry, dict) else None
            if (header is not None and _text(header.get("title"))
                    and isinstance(entry, dict)
                    and entry.get("size") == st.st_size
                    and entry.get("mtime_ns") == st.st_mtime_ns
                    and isinstance(entry.get("count"), int) and entry["count"] >= 0
                    # The shortest record this reader would accept is
                    # {"type":"message","role":"u","content":""} plus a newline, so
                    # a count above this is arithmetically impossible and the entry
                    # is junk. It is a sanity bound on a *cache*, not a defence: a
                    # writer who can edit index.json can edit the session file too.
                    and entry["count"] * _MIN_RECORD_BYTES <= st.st_size):
                scan = _Scan(header, entry["count"])
                try:
                    scan.usage, decisive = _tail_usage(path, st.st_size)
                except (OSError, ValueError):
                    scan.usage, decisive = None, False
                meta = _meta_from(scan, sid, st.st_mtime)
                if not decisive:
                    # The newest checkpoint is further back than one tail read.
                    # Only then are the cached totals used at all, and only where
                    # they are at least what the header already proves.
                    if _is_totals_row(totals) and _row_covers(totals, meta):
                        for key, value in zip(_USAGE_KEYS, totals):
                            setattr(meta, key, max(0, _int(value)))
                        meta.cost_usd = max(0.0, _num(totals[4]))
                    else:
                        meta = None
                if meta is not None:
                    # Activity cannot postdate the file that records it.
                    meta.updated = max(meta.updated,
                                       min(_num(entry.get("updated")), st.st_mtime))
            if meta is None:
                try:
                    meta = _meta_from(_scan_meta(path), sid, st.st_mtime)
                except (OSError, ValueError):
                    continue
            metas.append(meta)
            fresh[sid] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                          "count": meta.message_count, "updated": meta.updated,
                          "usage": [meta.input_tokens, meta.output_tokens,
                                    meta.cache_read_tokens, meta.cache_write_tokens,
                                    meta.cost_usd]}
        # Compare, don't guess: a session the fast path cannot serve (an untitled
        # one, say) is re-scanned every time, but re-scanning it to the same
        # numbers is not a reason to rewrite the file.
        if fresh != cached:
            self._write_index(fresh)
        return metas

    def _sweep_sidecar(self, name: str, live: frozenset, lock_age: float = 300.0) -> None:
        """Remove a ``.lock`` / ``.bak`` sidecar whose session no longer exists.

        A ``.bak`` is a full plaintext transcript, so an orphaned one is exactly
        the leftover a privacy-minded user would least expect and — before
        :meth:`delete` learned to sweep first — could not remove through this
        API at all. A ``.lock`` is empty and may still be held by somebody
        finishing a delete, so it has to be stale before it goes.
        """
        for suffix in (".lock", ".bak"):
            cut = name.find(EXT + suffix)
            if cut < 0:
                continue
            rest = name[cut + len(EXT) + len(suffix):]
            if rest and not (rest[:1] == "." and rest[1:].isascii() and rest[1:].isdigit()):
                return
            if name[:cut + len(EXT)] in live:
                return
            path = self.sessions_dir / name
            if suffix == ".lock":
                try:
                    if time.time() - path.stat().st_mtime <= lock_age:
                        return
                except OSError:
                    return
            with contextlib.suppress(OSError):
                os.unlink(str(path))
            return

    def _sweep_temp(self, name: str, max_age: float = 300.0, dead_age: float = 15.0,
                    live_age: float = 86400.0) -> None:
        """Remove a temp file left behind by a rewrite that died.

        A temp file may still belong to a live writer, so it is only removed once
        we know better: either the process that created it is gone (its pid is in
        the name), or enough time has passed that no plausible writer is still
        mid-rewrite. Waiting an hour for the second case just kept a killed
        writer's litter around for an hour.

        Age alone is not enough on its own, though: "no plausible writer" was a
        guess, and a rewrite of a very large transcript on a slow disk can sit
        inside one ``fsync`` for longer than the guess allows. Sweeping it then
        deleted a *live* writer's temp file and its ``os.replace`` came back as a
        raw ``FileNotFoundError`` out of ``update()``. So a name that claims this
        machine and a pid that still exists is left alone — until a day has
        passed, by which point the pid is far likelier to have been recycled than
        to belong to the same rewrite.
        """
        path = self.sessions_dir / name
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return
        host, pid = _owner_from_temp_name(name)
        mine = host == _HOST and pid and not _pid_is_gone(pid)
        if mine and age <= live_age:
            return
        if age > max_age or (age > dead_age and host == _HOST and _pid_is_gone(pid)):
            with contextlib.suppress(OSError):
                os.unlink(str(path))

    def _read_index(self) -> dict:
        try:
            raw = _read_bytes(self.index_path)
        except (OSError, ValueError):
            return {}
        try:
            doc = json.loads(raw.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            return {}
        if not isinstance(doc, dict) or doc.get("version") != INDEX_VERSION:
            return {}
        entries = doc.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _write_index(self, entries: dict) -> None:
        """Refresh the cache. Failure here is never fatal — it is only a cache."""
        doc = {"version": INDEX_VERSION, "written": time.time(), "entries": entries}
        fd = tmp = None
        try:
            fd, tmp = tempfile.mkstemp(prefix=_tmp_prefix(), suffix=".json",
                                       dir=str(self.sessions_dir))
            _write_all(fd, _dumps_line(doc))     # mkstemp always creates 0600
            os.close(fd)
            fd = None
            os.replace(tmp, str(self.index_path))
            tmp = None
        except OSError:
            pass
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)


def _is_totals_row(row) -> bool:
    """True for a cached ``[in, out, cache_read, cache_write, cost]`` row."""
    return (isinstance(row, list) and len(row) == 5
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and v == v and v not in (float("inf"), float("-inf")) for v in row))


def _when(ts: float, fmt: str = "%Y-%m-%d %H:%M") -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime(fmt)
    except (ValueError, OSError, OverflowError):
        return "?"
