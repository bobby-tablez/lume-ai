"""The interactive prompt: readline when we can have it, a plain reader when we can't.

Three environments have to behave: a real terminal (readline, history, tab
completion, emacs keys), a terminal without the ``readline`` module (raw-mode
line reading, paste still arrives whole), and a pipe or file on stdin (no marker
noise, one line per submission). :class:`Prompt` picks between them per read, so
the same object works in all three.

Submission rules — these are the whole contract, and ``/help`` prints them:

* **Enter** submits the line.
* A line ending in an **odd number of backslashes** continues onto the next line:
  one backslash is consumed, the rest are literal text. An even number submits
  the line exactly as typed.
* A line whose first non-space characters are ``\"\"\"`` opens a **verbatim block**.
  It ends at the first line that ends with ``\"\"\"`` — including a line that
  arrives in the middle of a paste, in which case everything after that line is
  handed back as the next submission. The markers are removed and a blank
  first/last line is dropped. Nothing inside a block is special.
* ``\"\"\"text\"\"\"`` on one line is a complete block.
* A **paste** is one submission, every line of it, exactly as pasted — bar
  the control characters no message has a use for (see ``sanitize_text``
  below), and bar the one ambiguity point 6 admits to.

How a submission ends, exactly, because the guarantee is only as good as the
mechanism — and none of it is a clock except where it says so:

1. A **bare CR** is the Return key. Once ``ICRNL`` is off nothing else produces
   one, so it ends the submission there and then. Two commands typed ahead while
   lume was printing are two commands, however close together they arrived, and
   a typed Enter never waits for anything.
2. **LF and CRLF are content.** No key sends them: they came from a paste, or
   from a program writing into the terminal. So they stay *inside* one
   submission — which is what makes a multi-line paste one message even on a
   terminal that frames nothing. Under readline, which accepts a line on either,
   ``\\C-j`` types :data:`RL_EOL_MARK` first: the mark is how a line that ended
   on pasted content gets told apart from one that ended on Return, and the
   rest of that paste is then read here rather than through readline.
3. lume turns **bracketed paste on once, for as long as the prompt lives**
   (``ESC[?2004h``; off again on close and from the exit hook). A terminal that
   brackets its pastes wraps them in ``ESC[200~``/``ESC[201~``, and everything
   between the two guards is one submission *however slowly it arrives*.
4. readline would otherwise swallow that frame itself — and its paste handler
   rewrites every CR as a newline, which turns a CRLF paste into a blank line
   between every line and cannot be undone afterwards. So lume takes the frame
   away from it: ``ESC[200~`` is bound to a macro that types
   :data:`RL_PASTE_MARK` and accepts the line, which hands the prompt back the
   instant a paste starts and leaves the body in the terminal's queue, where
   lume reads it **as bytes**. That is the only way the two paths can agree, and
   it is why a paste is verbatim in both. A single-line paste is put back in the
   line for editing; several lines are a message and are sent.
5. The one timed thing: when a submission was ended by a *pasted* line ending on
   a terminal that has never shown us a frame, lume waits :data:`PASTE_WINDOW`
   (100 ms, restarted on every newline) for more of the same paste. A source
   slower than that — a badly congested ssh link — still splits it. That is the
   one case the guarantee does not cover, and it is why the secret filter runs
   per line as well as per submission.
6. What is left over is the one genuine ambiguity: a paste whose lines are
   separated by *bare CRs* (some terminals convert them on the way out) on a
   terminal that does not bracket. It is byte-for-byte identical to typing, so
   lume treats it as typing.
7. Piped stdin is *not* coalesced: a pipe is a script, and each line there is its
   own submission.

The terminal mode is set **once**, at construction, and restored on close:

* ``ICRNL`` off with ``VEOL`` set to CR — Enter still ends a line, but a CRLF
  paste stays CRLF instead of gaining a newline per line. Once, not per read,
  because the bytes that need it most arrive while lume is busy printing; there
  is nothing left to guess about afterwards and nothing to repair.
* ``NOFLSH``, so a Ctrl-C can be told apart from a Ctrl-C over a half-typed line.
* bracketed paste on, so a paste made during a reply is framed too.
* ``ECHOCTL`` is left exactly as found: it is what renders a pasted escape
  sequence as ``^[`` rather than executing it, in the one window (lume printing)
  where the terminal is doing the echoing.

For the duration of each read the plain reader additionally clears ``ICANON``
and ``ECHO``:

* ``ICANON`` off because the canonical line buffer is 4096 bytes and silently
  drops everything past it — a minified line, a base64 blob or a data: URI would
  arrive truncated, or wedge the prompt outright. Line assembly is lume's job
  anyway.
* ``ECHO`` off because with the terminal echoing, a pasted ``ESC]0;…BEL`` is
  handed straight back to the terminal to execute. lume paints the line itself,
  through :func:`~lume.ansi.sanitize_text`, and handles Backspace, Ctrl-U,
  Ctrl-W and Ctrl-D itself.

Everything is restored in a ``finally``, and :func:`lume.ansi.on_exit` restores
it again if the process is signalled away (SIGTERM/SIGHUP/SIGQUIT) — otherwise
closing the window would leave the user's shell needing ``stty sane``.

* **Alt+Enter** (see ``multiline_key``) inserts a newline without sending; it is
  implemented as a readline macro that appends ``\\`` and accepts the line.

**libedit** — what Apple and several distributions ship under the name
``readline`` — takes the plain path too, for the reasons in :func:`_is_libedit`.
It keeps bracketed paste, the paste framing and the editing keys; what it gives
up is libedit's own line editor, which is why Up/Down are implemented here (see
:meth:`Prompt._recall`) rather than left to the library.

**Windows** has no ``readline`` in a stock Python and no ``select`` on a console
handle, so it takes the plain path: the console driver does the line editing, and
``msvcrt.kbhit`` — the one thing that will say whether more input is already
waiting — keeps a paste in one piece. There is no key to bind there, so Alt+Enter
is neither offered nor listed by ``/help`` (Windows Terminal spends that chord on
full-screen anyway); ``\\`` and ``\"\"\"`` are the multi-line keys, and end-of-file
is Ctrl-Z Enter. :func:`interrupt_guard` covers the other half of Ctrl-C there.

Ctrl-C never kills lume from here: it throws away whatever is pending and returns
a fresh prompt. Only a second Ctrl-C in a row on an already-empty prompt raises
``KeyboardInterrupt``, so the app can offer a way out. Ctrl-D on an empty prompt
raises ``EOFError``; with a half-typed block it submits what is there instead.

Every submission goes through :func:`~lume.ansi.sanitize_text` before it is
returned or remembered: tabs and newlines survive, control characters and escape
sequences do not. A message has no use for them and a history file that replays
them when it is ``cat``-ed is an injection channel.

The history file is per-user and strictly one entry per line: multi-line
submissions stay in memory only. It is opened without following symlinks, must be
a regular file, is only ever *tightened* towards ``0600``, and lume never changes
the mode of the directory it lives in. It is capped in lines *and* in bytes, an
over-long entry is kept in memory only, and a rewrite that cannot take the lock
is skipped rather than raced. If the location is not writable, history is dropped
for the session rather than forced.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import re
import signal
import stat
import sys
import threading
import time
from pathlib import Path

from . import commands
from .ansi import Caps, Console, char_width, display_width, on_exit, sanitize_text
from .theme import Theme

try:                                    # absent on some minimal builds; not fatal
    import readline
except Exception:                       # pragma: no cover - platform dependent
    readline = None

try:
    import select
except Exception:                       # pragma: no cover - platform dependent
    select = None

try:
    import termios
except Exception:                       # pragma: no cover - platform dependent
    termios = None

try:
    import fcntl
except Exception:                       # pragma: no cover - platform dependent
    fcntl = None

try:                                    # Windows console: the only way to tell
    import msvcrt                       # whether more input is already waiting
except Exception:                       # pragma: no cover - platform dependent
    msvcrt = None

__all__ = [
    "Prompt", "default_completer", "default_history_path", "interrupt_guard",
    "looks_secret", "readline_escape", "HISTORY_MAX", "HISTORY_BYTES",
    "HISTORY_ENTRY_MAX", "PASTE_WINDOW", "RL_PASTE_MARK", "RL_EOL_MARK",
]

#: Entries kept in the history file (and in readline's memory).
HISTORY_MAX = 2000

#: Bytes the history file may take. SPEC calls it size-capped; a line cap alone
#: is not one — fifty 110 KB pastes made startup take two seconds.
HISTORY_BYTES = 1 << 20

#: An entry longer than this is remembered in RAM but never written: one line is
#: not worth a megabyte, and the secret scan is capped at the same length.
HISTORY_ENTRY_MAX = 8192

#: Fallback coalescing window for terminals that do not bracket their pastes.
#: The clock restarts on every newline, so this is the gap *between lines* a
#: paste may have and still arrive whole — not the time the whole paste may take.
PASTE_WINDOW = 0.1

#: How long to wait for the closing guard of a paste that has already started.
#: Only a terminal that lies about bracketed paste ever hits this.
FRAME_TIMEOUT = 5.0

FENCE = '"""'

PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"
_BP_ON = b"\x1b[?2004h"
_BP_OFF = b"\x1b[?2004l"

#: What readline types into the line when a paste starts, and when a line was
#: ended by a *pasted* line ending rather than by the Return key (see the module
#: docstring). Both are typed by a macro, so they have to be ordinary printable
#: text; they are chosen to be unmistakable, and to say what happened in the
#: unlikely event that a terminal leaves one on screen.
RL_PASTE_MARK = "[lume-paste]"
RL_EOL_MARK = "[lume-eol]"

# Runs of escape sequences, so they can be hidden from readline's column maths.
_ESC_RUN = re.compile(
    r"(?:\x1b\[[0-9;:?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_])+"
)

# Bracketed-paste guards: the frame around a paste, and the noise to strip once
# the frame has been used.
_PASTE_MARKS = re.compile(r"\x1b\[20[01]~")

# Any of the three line endings, whichever the terminal happens to send.
_EOL = re.compile(r"\r\n|\r|\n")

# Backspace, Ctrl-U, Ctrl-W, Ctrl-D: the keys the reader has to handle itself
# once the line discipline is out of the way. Everything else is text.
#: Keys the plain reader acts on itself. The arrows are matched whole so a
#: three-byte sequence is never mistaken for an ESC followed by text; an
#: escape that is not one of these stays text and is neutralised on the way
#: to the screen.
_EDIT_KEYS = re.compile(r"[\x7f\x08\x15\x17\x04]|\x1b[\[O][AB]")
_UP_KEYS = ("\x1b[A", "\x1bOA")
_DOWN_KEYS = ("\x1b[B", "\x1bOB")

# ---------------------------------------------------------------------- secrets
#
# Anything that smells like a credential never reaches the history file. The
# shapes are deliberately broad: a false positive costs one forgotten history
# entry, a false negative writes a live key to disk in cleartext. What they must
# not be is *slow* — looks_secret runs inside read(), so a pattern that rescans
# its input at every start position freezes the prompt on a long line.

#: Characters scanned per line and per whole entry. An entry longer than this is
#: never written to the file anyway (:data:`HISTORY_ENTRY_MAX`), so scanning
#: further would cost time to prove nothing.
SECRET_SCAN_MAX = HISTORY_ENTRY_MAX

_SECRET_RE = re.compile(
    r"sk-ant-\S*"                                  # the SPEC rule: any sk-ant- token
    # sk-…, sk_live_…, pk_test_…, rk_… — the separator is a class on purpose:
    # the whole Stripe family uses '_' and used to walk straight past.
    r"|(?<![A-Za-z0-9])[sprk]k[-_][A-Za-z0-9][A-Za-z0-9_-]{11,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"                 # github classic tokens
    r"|github_pat_[A-Za-z0-9_]{20,}"               # github fine-grained tokens
    r"|xox[a-z]-[A-Za-z0-9-]{10,}"                 # slack: xoxb/xoxp/xoxc/xoxe/xoxs
    r"|xapp-[A-Za-z0-9-]{10,}"                     # slack app-level token
    r"|npm_[A-Za-z0-9]{30,}"                       # npm automation token
    r"|dop_v1_[A-Za-z0-9]{32,}"                    # digitalocean
    r"|glpat-[A-Za-z0-9_-]{16,}"                   # gitlab
    r"|A(?:KIA|SIA|GPA|IDA|ROA|NPA|NVA)[0-9A-Z]{12,}"   # aws key ids
    r"|AIza[A-Za-z0-9_-]{20,}"                     # google api key
    r"|ya29\.[A-Za-z0-9_-]{20,}"                   # google oauth token
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\."  # a jwt, bearer or not
    r"|bearer\s+[A-Za-z0-9._~+/-]{12,}"            # with or without 'Authorization:'
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    # Bounded and anchored to a word start on purpose: an unbounded greedy run
    # in front of a literal is quadratic, and this filter runs inside read().
    r"|(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+.-]{0,30}://[^/\s:@]+:[^/\s@]+@",
    re.IGNORECASE,
)

# name = value, in two strengths. The strict noun list takes any value; the wide
# one (Account/Client/Consumer/Master/… — anything at all in front of the noun)
# wants a value that looks like a token, so that 'monkey = bananas' survives.
_KV_STRICT = re.compile(
    r"(?:(?:api|auth|access|secret|private|session|refresh|client|consumer"
    r"|master|account|personal|bot|app|user|admin|db|database)"
    r"[ _.-]?(?:keys|key|tokens|token|secret|password|passwd|pass|pwd"
    r"|credentials|credential)"
    r"|(?<![A-Za-z])(?:password|passwd|secret|token)(?![A-Za-z]))"
    r"\s*[:=]\s*[\"']?(?P<v>[^\s\"'`]{4,})",
    re.IGNORECASE,
)
# No prefix group: search() finds the noun *inside* a word by itself (the 'Key'
# of 'AccountKey='), and a greedy [A-Za-z]* in front of it costs 1.6 seconds on
# an 8 KB line for exactly nothing.
_KV_WIDE = re.compile(
    r"(?:keys|key|tokens|token|secret|password|passwd|pwd)"
    r"\s*[:=]\s*[\"']?(?P<v>[^\s\"'`]{6,})",
    re.IGNORECASE,
)

# A private-key body line: 40+ base64 characters with upper, lower and digit in
# them. Bounded, and *case-sensitive* — under IGNORECASE the two case tests match
# either case, which degrades the whole rule into "40 characters of base64" and
# quietly eats every git SHA and hex digest. The character tests are done in
# Python: as lookaheads they rescanned the run at every start position, which
# cost 19 seconds on a 60,000-character line.
_KEY_BODY = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,512}={0,2}")

_HEX = set("0123456789abcdefABCDEF")

#: Values that name a secret without being one. `API_KEY=$STRIPE_KEY` in a shell
#: snippet is a question about configuration, not a leak.
_PLACEHOLDER = (
    "process.env", "os.environ", "os.getenv", "getenv(", "environ[", "env.",
    "$", "${", "%(", "{{", "<", "self.", "this.", "config.", "settings.",
    "secrets.", "vault.", "***", "xxx", "your", "changeme", "none", "null",
    "true", "false", "undefined", "redacted", "hunter2",
)

# multiline_key -> readline key sequence. The macro inserts a backslash and then
# accepts the line, which lands on the trailing-backslash continuation rule above.
_KEYSEQ = {
    "alt+enter": r"\e\r",
    "alt+return": r"\e\r",
    "meta+enter": r"\e\r",
    "esc+enter": r"\e\r",
    "ctrl+j": r"\C-j",
    "ctrl+o": r"\C-o",
}


def _is_key_body(run: str) -> bool:
    """Does this run of base64 characters look like key material rather than a digest?"""
    lower = upper = digit = False
    hexonly = True
    for ch in run:
        if ch.islower():
            lower = True
        elif ch.isupper():
            upper = True
        elif ch.isdigit():
            digit = True
        if hexonly and ch not in _HEX:
            hexonly = False
    return lower and upper and digit and not hexonly


def _is_secret_value(value: str, strong: bool = False) -> bool:
    """Is the right-hand side of ``name = value`` a credential, or a reference to one?"""
    value = value.strip("\"'`,;.")
    if len(value) < 6:
        return False
    low = value.lower()
    if low.startswith(_PLACEHOLDER) or low in _PLACEHOLDER:
        return False
    if set(low) <= set("x*.-_"):                   # already masked
        return False
    if not strong:
        return True
    # The wide noun list has to earn it: a token has a digit in it, or is long.
    return any(c.isdigit() for c in value) or len(value) >= 20


def _scan_secret(chunk: str) -> bool:
    if _SECRET_RE.search(chunk) is not None:
        return True
    for match in _KV_STRICT.finditer(chunk):
        if _is_secret_value(match.group("v")):
            return True
    for match in _KV_WIDE.finditer(chunk):
        if _is_secret_value(match.group("v"), strong=True):
            return True
    for match in _KEY_BODY.finditer(chunk):
        if _is_key_body(match.group(0)):
            return True
    return False


def looks_secret(text: str) -> bool:
    """True if `text` — as a whole *or* on any one of its lines — looks like a credential.

    Both passes matter: a paste that gets split into several submissions must be
    caught line by line, and a single submission must be caught even when the
    credential is only part of it. Each pass is capped at
    :data:`SECRET_SCAN_MAX` characters, which is also the longest entry that can
    reach the file, so nothing that could be written goes unscanned.
    """
    if not text:
        return False
    if _scan_secret(text[:SECRET_SCAN_MAX]):
        return True
    if "\n" not in text:
        return False
    return any(_scan_secret(line[:SECRET_SCAN_MAX]) for line in text.splitlines())


def readline_escape(s: str) -> str:
    """Wrap escape sequences in ``\\001``/``\\002`` so readline can count columns.

    GNU readline measures the prompt to know where the cursor is; without these
    markers every colour code is counted as printable and editing a long line
    smears. The visible width of the result is unchanged.
    """
    return _ESC_RUN.sub(lambda m: "\001" + m.group(0) + "\002", s)


def _is_libedit(rl) -> bool:
    """True for the libedit shim Apple and several distros ship as `readline`.

    It has no `enable-bracketed-paste`, so nothing would consume an `ESC[200~`
    guard, and it breaks a pasted CRLF into pieces of its own before any of this
    code sees the bytes -- so a paste cannot be reassembled on that path at all.
    lume gives it the plain reader instead. Detected by the module docstring,
    because libedit reports no version of its own.
    """
    return rl is not None and "libedit" in (getattr(rl, "__doc__", "") or "")


def _strip_eol(line: str) -> str:
    """Drop one trailing line ending, whichever of the three it is."""
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def _install_console_ctrl(callback):
    """Register a Windows console control handler; returns an unregister callable.

    Behind a function of its own so the POSIX path never imports ctypes and so a
    test can stand in for it. ``None`` means "not available here", which is the
    answer everywhere except a real Windows console.
    """
    if os.name != "nt":
        return None
    try:                                              # pragma: no cover - Windows
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None
    try:                                              # pragma: no cover - Windows
        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def route(event):
            if event in (0, 1):                       # CTRL_C_EVENT, CTRL_BREAK_EVENT
                try:
                    callback()
                except Exception:
                    pass
                # "Handled": the default action for Ctrl-C is to kill lume, and
                # the whole point here is that it should only stop the reply.
                return True
            return False

        routine = handler_type(route)                 # must outlive the registration
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.SetConsoleCtrlHandler(routine, True):
            return None

        def remove():
            kernel32.SetConsoleCtrlHandler(routine, False)

        remove.routine = routine                      # keep the thunk alive
        return remove
    except Exception:
        return None


@contextlib.contextmanager
def interrupt_guard(on_interrupt):
    """Call `on_interrupt()` when the user presses Ctrl-C, even mid-syscall.

    Used to stop a reply that is still streaming. On POSIX a plain SIGINT handler
    does it: the signal interrupts the blocked socket read, and Python runs the
    handler as the call unwinds.

    Windows needs the second half. A parked ``recv`` there is *not* interrupted,
    and a Python-level signal handler cannot run until the main thread next
    reaches a bytecode boundary — so while the model was quiet (thinking, or slow
    to the first token) Ctrl-C did nothing at all, for as long as the quiet
    lasted. ``SetConsoleCtrlHandler`` is called on a thread the console owns
    rather than the one that is stuck, so the callback lands straight away.

    Restores whatever was in place on the way out, and is a no-op off the main
    thread, where installing a signal handler is not allowed.
    """
    def handler(signum=None, frame=None):
        try:
            on_interrupt()
        except Exception:                             # a guard must never raise
            pass

    previous = None
    installed = False
    if threading.current_thread() is threading.main_thread():
        try:
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handler)
            installed = True
        except (ValueError, OSError):                 # pragma: no cover - no signals
            previous = None
    remove_console = _install_console_ctrl(handler)
    try:
        yield handler
    finally:
        if remove_console is not None:                # pragma: no cover - Windows
            try:
                remove_console()
            except Exception:
                pass
        if installed and previous is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGINT, previous)


def default_history_path() -> Path:
    """``$LUME_HISTORY``, else the lume data directory's ``history`` file."""
    env = os.environ
    override = env.get("LUME_HISTORY")
    if override:
        return Path(override).expanduser()
    home = env.get("LUME_HOME")
    if home:
        return Path(home).expanduser() / "history"
    xdg = env.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "lume" / "history"
    return Path.home() / ".local" / "share" / "lume" / "history"


def default_completer(text: str, line: str) -> list:
    """Complete slash commands and their arguments; prose gets no suggestions."""
    stripped = line.lstrip()
    if not stripped.startswith("/"):
        return []
    if " " in stripped or "\t" in stripped:
        return commands.suggest(stripped)
    return commands.suggest(text or stripped)


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _fileno(stream):
    try:
        fd = stream.fileno()
    except Exception:
        return None
    return fd if isinstance(fd, int) and fd >= 0 else None


def _trailing_backslashes(s: str) -> int:
    return len(s) - len(s.rstrip("\\"))


def _norm_eol(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _open_frame(buf: str) -> bool:
    """True while a bracketed paste has started but not finished."""
    start = buf.rfind(PASTE_START)
    return start >= 0 and PASTE_END not in buf[start:]


def _submissions(text: str) -> tuple:
    """Cut a raw chunk from a terminal into whole submissions.

    Returns ``(units, tail)``: the submissions a boundary has already closed, and
    everything after the last one. Three kinds of line ending, and they do not
    mean the same thing:

    * A **bare CR** is the Return key — no terminal sends anything else for it
      once ``ICRNL`` is off — so it ends a submission. Two commands typed ahead
      while lume was printing are two commands, however close together they
      arrived.
    * **LF and CRLF** are content: no key produces them, so they came from a
      paste or from a program writing into the terminal, and they stay *inside*
      one submission. The caller decides when a trailing one ends it — at once
      when the terminal frames its pastes, otherwise after :data:`PASTE_WINDOW`.
    * A **bracketed-paste frame** is atomic: everything between the guards is one
      submission, every line of it, however slowly it arrives.
    """
    first = text.find(PASTE_START)
    head = text if first < 0 else text[:first]
    if PASTE_END in head:                              # a guard with no frame
        text = head.replace(PASTE_END, "") + ("" if first < 0 else text[first:])
    out, parts, i, n = [], [], 0, len(text)
    while i < n:
        frame = text.find(PASTE_START, i)
        stop = frame if frame >= 0 else n
        m = _EOL.search(text, i, stop)
        if m is None:
            parts.append(text[i:stop])
            if frame < 0:
                break
            end = text.find(PASTE_END, frame)
            if end < 0:
                parts.append(text[frame:])         # the frame is still open
                break
            parts.append(_norm_eol(text[frame + len(PASTE_START):end]))
            i = end + len(PASTE_END)
            continue
        parts.append(text[i:m.start()])
        i = m.end()
        if m.group(0) == "\r":                     # the Return key: a boundary
            out.append("".join(parts))
            parts = []
        else:
            parts.append("\n")                     # pasted LF or CRLF: content
    return out, "".join(parts)


def _soft_done(tail: str) -> bool:
    """Is `tail` a submission waiting only for the caller to say the paste is over?

    Not while a frame is still open: the guards outrank every line ending inside
    them, which is the whole point of them.
    """
    return tail.endswith("\n") and not _open_frame(tail)


def _cursor_after(col: int, text: str, width: int) -> tuple:
    """Where the cursor ends up after painting `text` from column `col`."""
    row = 0
    for ch in text:
        if ch == "\n":
            row += 1
            col = 0
            continue
        w = char_width(ch)
        if w <= 0:
            continue
        if col + w > width:
            row += 1
            col = w
        else:
            col += w
    return row, col


class Prompt:
    """Reads one submission at a time from the user.

    ``console`` is used for the marker and for hints (never for the readline
    prompt itself, which readline must print so it can redraw it). ``completer``
    is ``callable(text, line) -> list[str]``; the default completes slash commands
    and their arguments. ``history_path`` defaults to the per-user history file —
    pass ``""`` or ``False`` to keep nothing on disk. ``stdin`` is an escape hatch
    for tests and embedding; it defaults to ``sys.stdin``, resolved per read.
    Passing anything *other* than the real ``sys.stdin`` turns readline off for
    that prompt: readline's ``input()`` reads the process's own stdin and would
    quietly ignore the stream you passed.

    Constructing one on a terminal takes the terminal's input mode with it (see
    the module docstring) and hands it back on :meth:`close`, which is also run
    from an exit hook if the process is signalled away.
    """

    MARKER = "❯ "
    MARKER_ASCII = "> "
    CONT = "⋮ "
    CONT_ASCII = ". "

    def __init__(self, console: Console, theme: Theme, history_path=None,
                 completer=None, multiline_key: str = "alt+enter",
                 *, stdin=None, history_max: int = HISTORY_MAX) -> None:
        self.console = console
        self.theme = theme
        self.completer = completer or default_completer
        self.multiline_key = multiline_key or ""
        self.history_max = max(1, int(history_max))
        self._stdin = stdin
        self._rl = readline
        # libedit takes the plain path; see _readline_ready.
        self._libedit = _is_libedit(readline)
        self._matches = []
        self._completer_fn = self._complete    # one object, so it can be recognised
        self._interrupts = 0
        self._closed = False
        self._auto_history_off = False
        self._history = []
        self._file_lines = 0
        self._file_bytes = 0
        self._typed = False
        self._brackets = False
        self._pending = []           # whole submissions read but not yet returned
        self._partial = ""           # the tail of a read that is not a line yet
        self._owner = None           # (fd, saved attrs) while we hold the terminal
        self._hooked = False

        if history_path is None:
            self.history_path = default_history_path()
        elif not history_path:
            self.history_path = None
        else:
            self.history_path = Path(history_path)

        self._load_history()
        self._setup_readline()
        self._own_terminal()

    # ------------------------------------------------------------------ marker

    @property
    def caps(self) -> Caps:
        return self.console.caps

    @property
    def uses_readline(self) -> bool:
        """Whether the *next* read will go through readline (needs a tty on both ends)."""
        return self._readline_ready(self._stdin if self._stdin is not None else sys.stdin)

    def marker(self, continuation: bool = False) -> str:
        """The bare prompt glyph, two columns wide in both unicode and ASCII."""
        if continuation:
            return self.CONT if self.caps.unicode else self.CONT_ASCII
        return self.MARKER if self.caps.unicode else self.MARKER_ASCII

    def styled_marker(self, continuation: bool = False, readline_safe: bool = False) -> str:
        """The themed marker; `readline_safe` hides its escapes from readline."""
        glyph = self.marker(continuation)
        # The trailing space stays outside the style so a reversed/underlined
        # theme cannot paint the gap the cursor sits in.
        body = self.theme.render(glyph.rstrip(" "), "prompt.marker", self.caps)
        out = body + " " * (len(glyph) - len(glyph.rstrip(" ")))
        return readline_escape(out) if readline_safe else out

    def marker_width(self, continuation: bool = False, readline_safe: bool = False) -> int:
        """Visible columns the marker takes up — what readline must not miscount."""
        s = self.styled_marker(continuation, readline_safe)
        return display_width(s.replace("\001", "").replace("\002", ""))

    # ------------------------------------------------------------------- read

    def read(self, prefix: str = "", placeholder: str = "") -> str:
        """Read one submission, applying the multi-line rules in the module docstring.

        `prefix` pre-fills the line (readline lets you edit it; otherwise it is
        echoed and prepended). `placeholder` is a one-off dim hint printed above
        the prompt on a tty. Returns ``''`` for an empty line. Raises ``EOFError``
        on Ctrl-D at an empty prompt, and ``KeyboardInterrupt`` only on a second
        consecutive Ctrl-C with nothing pending.
        """
        if self._closed:
            raise ValueError("Prompt is closed")
        self._hist_at, self._hist_draft = None, ""   # each read starts at the live line
        if placeholder and self.caps.is_tty:
            self.console.print(self.theme.render(placeholder, "prompt.hint", self.caps))

        lines = []
        fence = False
        used_fence = False
        pending = prefix or ""

        while True:
            try:
                raw = self._read_line(bool(lines) or fence, pending)
            except KeyboardInterrupt:
                had = (bool(lines) or fence or bool(pending.strip())
                       or bool(self._pending) or bool(self._partial.strip())
                       or self._half_typed())
                self._typed = False
                self._pending = []          # cancel means cancel, type-ahead too
                self._partial = ""
                if self._on_interrupt(had):
                    raise
                lines, fence, used_fence, pending = [], False, False, ""
                continue
            except EOFError:
                if lines or fence:
                    break
                raise
            pending = ""
            self._interrupts = 0
            self._typed = False

            if fence:                                   # inside a verbatim block
                closed = False
                block = raw.split("\n")
                for idx, line in enumerate(block):
                    body = line.rstrip()
                    if body.endswith(FENCE):
                        lines.append(body[: -len(FENCE)])
                        rest = "\n".join(block[idx + 1:])
                        if rest.strip():                # the paste ran past the fence
                            self._pending.insert(0, rest)
                        closed = True
                        break
                    lines.append(line)
                if closed:
                    fence = False
                    break
                continue

            if "\n" in raw:                             # a paste: verbatim, one message
                lines.append(raw)
                break

            head = raw.lstrip()
            if head.startswith(FENCE):
                used_fence = True
                rest = head[len(FENCE):]
                tail = rest.rstrip()
                if tail.endswith(FENCE):                # """all on one line"""
                    lines.append(tail[: -len(FENCE)])
                    break
                lines.append(rest)
                fence = True
                continue

            if _trailing_backslashes(raw) % 2 == 1:
                lines.append(raw[:-1])
                continue

            lines.append(raw)
            break

        if used_fence:
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

        text = sanitize_text("\n".join(lines))
        if text.strip():
            self.add_history(text)
        return text

    def _read_line(self, continuation: bool = False, prefill: str = "") -> str:
        if self._partial:                       # whole lines in it are owed first
            subs, rest = _submissions(self._partial)
            if subs:
                self._pending = subs + self._pending
                self._partial = rest
        if self._pending:                       # already read, still owed to the caller
            text = self._pending.pop(0)
            if self.caps.is_tty:
                self.console.write(self.styled_marker(continuation) + prefill)
                self._echo(text + "\n")
            return prefill + text
        stdin = self._stdin if self._stdin is not None else sys.stdin
        ready = self._readline_ready(stdin)
        marker = self.styled_marker(continuation, readline_safe=ready)
        if ready:
            return self._read_readline(marker, prefill, stdin, continuation)
        return self._read_plain(marker, prefill, stdin)

    def _readline_ready(self, stdin) -> bool:
        # readline only makes sense when it can own the line: a tty at both ends,
        # and the *process's own* stdin — input() cannot be pointed at anything
        # else, so an injected stream must take the plain path or be ignored.
        # libedit is excluded on purpose: it has no bracketed-paste support and
        # splits a pasted CRLF into pieces of its own before anything here sees
        # the bytes, so a paste cannot be put back together on that path. The
        # plain reader below reads the terminal directly and frames pastes
        # properly, which matters more than libedit's editing keys.
        return bool(self._rl is not None and not self._libedit and self.caps.is_tty
                    and stdin is sys.stdin and _isatty(stdin))

    def _read_readline(self, prompt: str, prefill: str, stdin,
                       continuation: bool = False) -> str:
        rl = self._rl
        fd = _fileno(stdin)
        if self._partial:                                 # never lost, just late
            prefill = self._partial + prefill
            self._partial = ""
        while True:
            if prefill:
                self._call(rl, "set_startup_hook",
                           lambda: self._call(rl, "insert_text", prefill))
            try:
                line = input(prompt)
            except KeyboardInterrupt:
                self._note_typed(self._call(rl, "get_line_buffer") or "")
                raise
            finally:
                if prefill:
                    self._call(rl, "set_startup_hook", None)
                self._forget_auto_history()
            if fd is None:                                # pragma: no cover
                return line
            if RL_PASTE_MARK not in line:
                break
            head, _, tail = line.rpartition(RL_PASTE_MARK)
            body = self._take_rl_paste(fd)
            if body is None:                     # nobody pasted: it was typed
                break
            self._unecho(line, continuation)     # take the mark off the screen
            text = head + body[0] + tail
            if not body[1]:
                # No line ending: it is still being written, so hand it back to
                # readline to go on editing. Sanitised, because a control
                # character in a line buffer is a control character the terminal
                # will act on the next time the line is redrawn.
                prefill = sanitize_text(text)
                continue
            # It ended a line, so it is a message — exactly as it would be on
            # the plain path, which submits on the same signal.
            if self.caps.is_tty:
                self.console.write(self.styled_marker(continuation))
                self._echo(text + "\n")
            return text
        if line.endswith(RL_EOL_MARK):
            # A pasted line ending, not the Return key: the rest of the paste is
            # on its way, and this line is the first of one message.
            self._unecho(line, continuation)
            return self._absorb_soft(line[: -len(RL_EOL_MARK)], fd, continuation)
        return self._absorb_paste(line, fd)

    def _take_rl_paste(self, fd: int):
        """Read the body of the paste readline has just told us about, as bytes.

        readline's own paste handler rewrites every CR as a newline before we can
        see it, which turns a CRLF paste into a blank line between every line and
        cannot be undone afterwards — by then the two shapes are identical. So
        the ``ESC[200~`` guard is bound to a macro that types
        :data:`RL_PASTE_MARK` and accepts the line: the body is still in the
        terminal's queue, unread and unconverted, and this is what reads it.

        Returns ``(text, ended_a_line)``, or ``None`` if nothing arrived at all —
        which means the user typed the mark's text by hand rather than pasting.
        A paste that ended a line is a message and is sent; one that did not is
        put back in the line to go on editing, which is what the plain reader
        does with the same bytes.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buf = ""
        with self._reading_mode(fd):
            deadline = time.monotonic() + PASTE_WINDOW
            while PASTE_END not in buf:
                # Clock-free once a paste has started: only "did one start at
                # all?" is timed, and only because the mark is typeable.
                timeout = FRAME_TIMEOUT if buf else max(0.0, deadline - time.monotonic())
                try:
                    ready, _, _ = select.select([fd], [], [], timeout)
                except (OSError, ValueError):         # pragma: no cover - fd raced away
                    break
                if not ready:
                    break
                chunk = self._read_available(fd, decoder)
                if chunk is None:
                    break
                buf += chunk
        if not buf:
            return None
        if PASTE_END in buf:
            self._brackets = True
        body = _PASTE_MARKS.sub("", buf)
        return _clean_paste(buf), bool(_norm_eol(body).endswith("\n"))

    def _unecho(self, echoed: str, continuation: bool) -> None:
        """Take the line readline echoed for the paste mark back off the screen."""
        if not self.caps.is_tty:
            return
        width = max(4, self.caps.width or 80)
        rows, _ = _cursor_after(self.marker_width(continuation), echoed, width)
        self.console.write("\x1b[%dA" % (rows + 1) + "\r\x1b[J")

    def _read_plain(self, marker: str, prefill: str, stdin) -> str:
        fd = _fileno(stdin)
        if fd is not None and select is not None and os.name != "nt" and _isatty(stdin):
            return prefill + self._read_tty(fd, marker, prefill)
        if self.caps.is_tty:
            self.console.write(marker + prefill)
        raw = stdin.readline()
        if raw == "":
            raise EOFError
        return prefill + self._absorb_console(_strip_eol(raw), stdin)

    def _absorb_console(self, line: str, stdin) -> str:
        """Windows: take in the rest of a paste the console is still holding.

        There is no ``select`` on a console handle, but ``msvcrt.kbhit`` answers
        the only question that matters: is more input already waiting? Asked in
        the microseconds after a line arrives, "yes" means the terminal handed
        over more than one line at once — a paste — because nobody types a whole
        line that fast. Without this a pasted block became one submission per
        line on Windows, which is the one thing pasting must not do.

        Every extra line is read through the same ``stdin`` as the first, never
        through ``msvcrt``, so nothing is taken from behind the runtime's back.
        """
        if msvcrt is None or not _isatty(stdin):
            return line
        out = [line]
        deadline = time.monotonic() + PASTE_WINDOW
        while time.monotonic() < deadline:
            try:
                if not msvcrt.kbhit():
                    break
            except OSError:                           # pragma: no cover - Windows
                break
            more = stdin.readline()
            if more == "":
                break
            out.append(_strip_eol(more))
            deadline = time.monotonic() + PASTE_WINDOW   # each line buys the next
        return "\n".join(out)

    # -------------------------------------------------------------- tty reading

    def _read_tty(self, fd: int, marker: str = "", prefill: str = "") -> str:
        """Read one submission straight from the tty.

        Unbuffered on purpose: a ``TextIOWrapper`` would read ahead into its own
        buffer, and the bytes it hid would be invisible both to the paste framing
        below and to readline on the next prompt. Where a submission ends is
        :func:`_submissions`' business; the only thing timed here is the tail of
        a paste on a terminal that does not frame them.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        with self._reading_mode(fd):
            if marker and self.caps.is_tty:
                self.console.write(marker + prefill)
            buf = self._partial
            self._partial = ""
            buf += self._take_queued(fd, decoder)
            shown = ""
            if buf:
                self._echo(buf)
                shown = buf
            units, tail = _submissions(buf)
            try:
                while not units and not _soft_done(tail):
                    timeout = FRAME_TIMEOUT if _open_frame(buf) else None
                    try:
                        ready, _, _ = select.select([fd], [], [], timeout)
                    except (OSError, ValueError):   # pragma: no cover - fd raced away
                        break
                    if timeout is not None and not ready:
                        break                       # a frame the terminal never closed
                    chunk = self._read_available(fd, decoder)
                    if chunk is None:
                        if buf:
                            break
                        raise EOFError
                    buf, shown, eof = self._consume(buf, shown, chunk, marker)
                    if eof:                         # Ctrl-D: send it, or leave
                        if not buf:
                            raise EOFError
                        buf += "\n"
                    units, tail = _submissions(buf)
            except KeyboardInterrupt:
                self._note_typed(buf + self._peek_pending(fd))
                raise
            if not units and not self._brackets:
                # Only a pasted line ending closed this one. On a terminal that
                # does not frame its pastes, the window is the only thing that
                # can tell the rest of a paste from the next message.
                buf += self._drain(fd, decoder)
                units, tail = _submissions(buf)
            self._echo("\n")
            if units:
                self._partial = tail
                self._pending.extend(units[1:])
                return units[0]
            self._partial = ""
            if _soft_done(tail):
                return tail.rstrip("\n")
            return _clean_paste(buf)                # EOF, or a frame never closed

    def _read_available(self, fd: int, decoder):
        """One read plus whatever is already behind it; ``None`` at end of input.

        Greedy on purpose: it keeps a CRLF from being split across two reads,
        where the CR would look like the Return key and the LF like the start of
        the next message.
        """
        out = b""
        while True:
            try:
                data = os.read(fd, 65536)
            except OSError:
                data = b""
            if not data:
                return decoder.decode(out) if out else None
            out += data
            try:
                if not select.select([fd], [], [], 0)[0]:
                    return decoder.decode(out)
            except (OSError, ValueError):            # pragma: no cover
                return decoder.decode(out)

    def _consume(self, buf: str, shown: str, chunk: str, marker: str) -> tuple:
        """Fold a live chunk into the buffer, painting it and honouring edit keys.

        The terminal's own echo is off for the duration of the read (a pasted
        escape sequence must not be handed back to it), so this is where the line
        appears on screen — and, inside a paste frame, where nothing is treated
        as an edit key, because a byte between the guards is text.
        """
        if PASTE_START in chunk:
            self._brackets = True                     # this terminal frames its pastes
        i, n = 0, len(chunk)
        while i < n:
            # In segments, not characters: a hundred-kilobyte paste examined a
            # character at a time (with the whole buffer rescanned for an open
            # frame each time) took seven seconds to arrive.
            m = _EDIT_KEYS.search(chunk, i)
            stop = m.start() if m else n
            if stop > i:
                seg = chunk[i:stop]
                buf += seg
                self._echo(seg)
                shown += seg
                i = stop
            if m is None:
                break
            key = m.group(0)
            i += len(key)
            if _open_frame(buf):                      # between the guards it is text
                buf += key
                self._echo(key)
                shown += key
                continue
            if key == "\x04":                         # Ctrl-D: send it, or leave
                return buf, shown, True
            if key in _UP_KEYS or key in _DOWN_KEYS:
                recalled = self._recall(key in _UP_KEYS, buf)
                if recalled is None:
                    continue
                buf = recalled
            elif key in ("\x7f", "\x08"):             # Backspace
                buf = buf[:-1]
            elif key == "\x15":                       # Ctrl-U: kill the line
                buf = buf[:buf.rfind("\n") + 1]
            else:                                     # Ctrl-W: kill a word
                head = buf.rstrip(" \t")
                buf = buf[:max(head.rfind(" "), head.rfind("\t"),
                               head.rfind("\n")) + 1]
            shown = self._repaint(marker, shown, buf)
        return buf, shown, False

    #: Where Up/Down have walked to, and the line that was being typed when the
    #: walk began. Both reset at the start of every read.
    _hist_at = None
    _hist_draft = ""

    def _recall(self, up: bool, buf: str):
        """Walk history for the reader that has no readline to do it.

        Returns the line to show, or None to leave the line alone -- Down on a
        line nobody walked away from, or Up with nothing to walk to. Multi-line
        entries are skipped: they are kept in memory for the app, but a reader
        that paints one line cannot put one back for editing honestly.
        """
        entries = [h for h in self._history if "\n" not in h]
        if not entries:
            return None
        if self._hist_at is None:
            if not up:
                return None                  # already at the live line
            self._hist_draft = buf           # keep what was being typed
            self._hist_at = len(entries)
        pos = self._hist_at + (-1 if up else 1)
        if pos < 0:
            return None                      # oldest entry: stay on it
        if pos >= len(entries):
            self._hist_at = None
            return self._hist_draft          # back out to the live line
        self._hist_at = pos
        return entries[pos]

    def _echo(self, text: str) -> None:
        """Paint what the terminal is no longer echoing for us, harmlessly.

        Sanitised, because the whole reason the tty's echo is off is that it
        would hand a pasted ``ESC]0;…BEL`` straight back to the terminal.
        """
        if not text or not self.caps.is_tty:
            return
        self.console.write(sanitize_text(_norm_eol(_PASTE_MARKS.sub("", text))))

    def _repaint(self, marker: str, shown: str, buf: str) -> str:
        """Redraw the line after an edit; returns what is now on screen.

        A backspace cannot simply back over a character: the line may have
        wrapped, and at column zero ``\\b`` does not climb. Painting the line
        again is both simpler and correct, and an edit is rare enough to afford it.
        """
        if not self.caps.is_tty:
            return buf
        width = max(4, self.caps.width or 80)
        text = sanitize_text(_norm_eol(_PASTE_MARKS.sub("", shown)))
        rows, _ = _cursor_after(display_width(marker), text, width)
        up = "\x1b[%dA" % rows if rows else ""
        self.console.write(up + "\r\x1b[J" + marker
                           + sanitize_text(_norm_eol(_PASTE_MARKS.sub("", buf))))
        return buf

    def _read_rl_paste(self, line: str, fd: int, continuation: bool = False) -> str:
        """Take a paste that readline has just told us about, byte for byte.

        readline's own paste handler rewrites every CR as a newline before we can
        see it, which turns a CRLF paste into a blank line between every line and
        cannot be undone afterwards — the two shapes are identical by then. So
        the ``ESC[200~`` guard is bound to a macro that types
        :data:`RL_PASTE_MARK` and accepts the line: the body is still in the
        terminal's queue, unread and unconverted, and this reads it.
        """
        head, _, tail = line.rpartition(RL_PASTE_MARK)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buf = ""
        with self._reading_mode(fd):
            deadline = time.monotonic() + PASTE_WINDOW
            while PASTE_END not in buf:
                # Clock-free once the paste has started: only the question "did
                # one start at all?" is timed, and only because a user may have
                # typed the mark's text by hand.
                timeout = FRAME_TIMEOUT if buf else max(0.0, deadline - time.monotonic())
                try:
                    ready, _, _ = select.select([fd], [], [], timeout)
                except (OSError, ValueError):         # pragma: no cover - fd raced away
                    break
                if not ready:
                    break
                try:
                    data = os.read(fd, 65536)
                except OSError:                       # pragma: no cover
                    break
                if not data:
                    break
                buf += decoder.decode(data)
        if PASTE_END in buf:
            self._brackets = True
        body = _clean_paste(buf)
        text = head + body + tail
        self._redraw_paste(line, text, continuation)
        return text

    def _redraw_paste(self, echoed: str, text: str, continuation: bool) -> None:
        """Replace the line readline left on screen with the paste it stood for."""
        if not self.caps.is_tty:
            return
        marker = self.styled_marker(continuation)
        width = max(4, self.caps.width or 80)
        rows, _ = _cursor_after(self.marker_width(continuation), echoed, width)
        self.console.write("\x1b[%dA" % (rows + 1) + "\r\x1b[J" + marker)
        self._echo(text + "\n")

    def _absorb_paste(self, line: str, fd: int) -> str:
        """Glue on a paste that readline split at a pasted line ending.

        readline accepts a line on the CR of a CRLF; the LF is the first byte
        still in the queue, and it is *proof* that the break was pasted content
        rather than the Return key — no keyboard sends CR LF. Nothing is waited
        for without that proof, which is why a typed Enter costs nothing at all.

        (A line that ended on a bare LF does not come here at all: it arrives
        carrying :data:`RL_EOL_MARK`, and :meth:`_absorb_soft` finishes it.)
        """
        if not self._waiting(fd):
            return line
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        with self._reading_mode(fd):
            queued = self._take_queued(fd, decoder)
            if not queued.startswith("\n"):
                self._partial = queued + self._partial   # type-ahead: not this line
                return line
            if not self._brackets:
                queued += self._drain(fd, decoder)
        units, tail = _submissions(line + queued)
        if units:
            self._pending.extend(units[1:])
            self._partial = tail + self._partial
            return units[0]
        self._partial = ""
        return tail.rstrip("\n")

    def _absorb_soft(self, line: str, fd: int, continuation: bool = False) -> str:
        """Finish a submission readline accepted on a *pasted* line ending.

        The remaining lines of the paste are still in the terminal, and they are
        read here rather than through readline, so the whole of it arrives as one
        message with its line endings intact. A terminal that frames its pastes
        never comes through here; one that does not gets :data:`PASTE_WINDOW`,
        the single timed thing in the reader.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        with self._reading_mode(fd):
            rest = self._take_queued(fd, decoder)
            if not self._brackets:
                rest += self._drain(fd, decoder)
        units, tail = _submissions(line + "\n" + rest)
        if units:
            self._pending.extend(units[1:])
            self._partial = tail + self._partial
            text = units[0]
        else:
            self._partial = ""
            text = tail.rstrip("\n")
        if self.caps.is_tty:
            self.console.write(self.styled_marker(continuation))
            self._echo(text + "\n")
        return text

    def _drain(self, fd: int, decoder=None) -> str:
        """Pull whatever else the terminal has, for terminals that do not bracket.

        The deadline restarts on every newline: a paste delivered a line at a
        time stays whole as long as consecutive lines are less than one window
        apart, while a dribble of bytes with no line ending cannot hold the
        prompt open indefinitely.
        """
        if decoder is None:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
        out = ""
        deadline = time.monotonic() + PASTE_WINDOW
        while True:
            if _open_frame(out):
                timeout = FRAME_TIMEOUT               # a paste started: see it out
            else:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
            try:
                ready, _, _ = select.select([fd], [], [], timeout)
            except (OSError, ValueError):            # pragma: no cover - fd raced away
                break
            if not ready:
                break
            try:
                data = os.read(fd, 65536)
            except OSError:                          # pragma: no cover
                break
            if not data:
                break
            chunk = decoder.decode(data)
            out += chunk
            if PASTE_START in chunk:
                self._brackets = True
            if "\n" in chunk or "\r" in chunk:
                deadline = time.monotonic() + PASTE_WINDOW
            if PASTE_END in chunk:
                self._brackets = True
                break                                 # the frame closed: nothing follows
        return out

    def _take_queued(self, fd: int, decoder=None) -> str:
        """Whatever the terminal was already holding, before this read began.

        It shares the caller's decoder: a multi-byte character split between the
        queue and the bytes that arrive next must not become two replacements.
        """
        if select is None:
            return ""
        if decoder is None:                          # pragma: no cover - tests only
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
        out = ""
        try:
            while select.select([fd], [], [], 0)[0]:
                data = os.read(fd, 65536)
                if not data:
                    break
                out += decoder.decode(data)
        except (OSError, ValueError):                       # pragma: no cover
            pass
        if PASTE_START in out:
            self._brackets = True
        return out

    def _waiting(self, fd: int) -> bool:
        """Is the terminal holding bytes right now? (Nothing is consumed.)"""
        if select is None or fd is None:
            return False
        try:
            return bool(select.select([fd], [], [], 0)[0])
        except (OSError, ValueError):                       # pragma: no cover
            return False

    # ------------------------------------------------------- terminal ownership

    def _own_terminal(self) -> None:
        """Take the terminal's input mode for as long as this prompt lives.

        Once, not per read: the bytes that most need the mode to be right are the
        ones that arrive *while lume is printing a reply*, and by the time a read
        starts the line discipline has already had them. Getting it right up
        front is what leaves nothing to guess at afterwards.
        """
        if self._owner is not None or termios is None or os.name == "nt":
            return
        stdin = self._stdin if self._stdin is not None else sys.stdin
        fd = _fileno(stdin)
        if fd is None or not _isatty(stdin):
            return
        try:
            saved = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            new[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR)
            new[3] |= getattr(termios, "NOFLSH", 0)
            cc = list(new[6])
            cc[termios.VEOL] = b"\r"
            new[6] = cc
            termios.tcsetattr(fd, termios.TCSANOW, new)
        except (OSError, ValueError, termios.error):        # pragma: no cover
            return
        self._owner = (fd, saved)
        # Off, not merely "not on": a terminal left in bracketed-paste mode by
        # some earlier program would otherwise send guards nothing here can read.
        _write_fd(fd, _BP_ON)
        if not self._hooked:
            self._hooked = True
            on_exit(self._release_terminal)

    def _release_terminal(self) -> None:
        """Hand the terminal back. Runs from close() *and* from a signal handler."""
        owner, self._owner = self._owner, None
        if owner is None:
            return
        fd, saved = owner
        _write_fd(fd, _BP_OFF)
        if termios is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, saved)
            except Exception:                               # pragma: no cover
                pass

    @contextlib.contextmanager
    def _noncanon(self, fd: int):
        """Take the line discipline out of the way, and put it back intact.

        * ``ICANON`` off (with ``VMIN``/``VTIME`` for a blocking read) because
          the canonical buffer is 4096 bytes and silently drops the rest of a
          longer line — or, with a paste frame open, everything after it.
        * ``ECHO`` off because the terminal would otherwise execute a pasted
          escape sequence on its way back to the screen; lume paints the line.
        * ``ICRNL`` off and ``NOFLSH`` on, in case this prompt does not own the
          terminal (an injected stream, or a platform without ``tcsetattr``).
        """
        saved = None
        if termios is not None:
            try:
                saved = termios.tcgetattr(fd)
                new = termios.tcgetattr(fd)
                new[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR)
                new[3] &= ~(termios.ICANON | termios.ECHO)
                new[3] |= getattr(termios, "NOFLSH", 0)
                cc = list(new[6])
                cc[termios.VMIN] = 1
                cc[termios.VTIME] = 0
                new[6] = cc
                termios.tcsetattr(fd, termios.TCSANOW, new)
            except (OSError, ValueError, termios.error):   # pragma: no cover
                saved = None
        try:
            yield saved
        finally:
            if saved is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, saved)
                except (OSError, ValueError, termios.error):   # pragma: no cover
                    pass

    @contextlib.contextmanager
    def _reading_mode(self, fd: int):
        """:meth:`_noncanon` for the length of a read, with bracketed paste on.

        A prompt that owns the terminal has already turned bracketed paste on and
        keeps it on; one that does not (an injected stream) turns it on for the
        read and off again, so it leaves nothing behind.
        """
        with self._noncanon(fd) as saved:
            if self._owner is None:
                _write_fd(fd, _BP_ON)
            try:
                yield saved
            finally:
                if self._owner is None:
                    _write_fd(fd, _BP_OFF)

    def _peek_pending(self, fd: int) -> str:
        """What the user had typed when Ctrl-C arrived (``NOFLSH`` kept it).

        Canonical mode holds an unfinished line back from ``read``; dropping into
        non-canonical mode for a moment releases it, which is the only way to
        tell "cancel this line" from "I have nothing to cancel".
        """
        if termios is None or select is None:
            return ""
        try:
            saved = termios.tcgetattr(fd)
            raw = termios.tcgetattr(fd)
            raw[3] &= ~termios.ICANON
            cc = list(raw[6])
            cc[termios.VMIN] = 0
            cc[termios.VTIME] = 0
            raw[6] = cc
            termios.tcsetattr(fd, termios.TCSANOW, raw)
        except (OSError, ValueError, termios.error):       # pragma: no cover
            return ""
        out = b""
        try:
            while select.select([fd], [], [], 0)[0]:
                data = os.read(fd, 65536)
                if not data:
                    break
                out += data
        except (OSError, ValueError):                      # pragma: no cover
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, saved)
            except (OSError, ValueError, termios.error):   # pragma: no cover
                pass
        return out.decode("utf-8", "replace")

    def _note_typed(self, text: str) -> None:
        if text and text.strip():
            self._typed = True

    # ---------------------------------------------------------------- interrupt

    def _half_typed(self) -> bool:
        """Was there text on the line that Ctrl-C just threw away?

        Under readline the buffer survives the SIGINT and answers directly; the
        plain reader has to be told, which :meth:`_peek_pending` does.
        """
        if self._typed:
            return True
        buf = self._call(self._rl, "get_line_buffer")
        return bool(buf and buf.strip())

    def _on_interrupt(self, had_pending: bool) -> bool:
        """Handle Ctrl-C. Returns True if the caller should re-raise."""
        if self.caps.is_tty:
            self.console.write("\n")
        if had_pending:
            self._interrupts = 0
            return False
        self._interrupts += 1
        if self._interrupts >= 2:
            self._interrupts = 0
            return True
        if self.caps.is_tty:
            self.console.print(self.theme.render(
                "Ctrl-C again or %s to exit." % commands.EOF_KEY,
                "prompt.hint", self.caps))
        return False

    # ------------------------------------------------------------------ history

    def history(self) -> list:
        """Everything this prompt has remembered this session, oldest first."""
        return list(self._history)

    def add_history(self, text: str) -> None:
        """Remember a submission. Secrets, blanks and immediate repeats are dropped.

        Multi-line submissions stay in memory (so Up recalls them) but are not
        written to the file, which is strictly one entry per line — that is what
        keeps the size cap and the secret scan honest. So is an entry too long to
        be worth a line of the file.
        """
        if not text or not text.strip() or self._closed:
            return
        if looks_secret(text):
            self._forget_auto_history()
            return
        if self._history and self._history[-1] == text:
            return
        self._history.append(text)
        self._call(self._rl, "add_history", text)
        if "\n" in text or len(text.encode("utf-8", "replace")) > HISTORY_ENTRY_MAX:
            return
        self._append_history_file(text)

    # -- the file itself -------------------------------------------------------
    #
    # Every open goes through _open_history: O_NOFOLLOW so a symlink planted at
    # the predictable path cannot redirect the write, and a regular-file check on
    # the *open descriptor* so the answer cannot change underneath us.

    def _open_history(self, mode: str):
        path = self.history_path
        if path is None:
            return None
        extra = (getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NONBLOCK", 0)       # NONBLOCK: never hang on a fifo
                 | getattr(os, "O_BINARY", 0))        # Windows: no "\n" -> "\r\n"
        if mode == "r":
            flags = os.O_RDONLY
        elif mode == "a":
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        else:
            flags = os.O_RDWR | os.O_CREAT
        try:
            fd = os.open(path, flags | extra, 0o600)
        except OSError:
            return None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
                os.close(fd)
                return None
        except OSError:                               # pragma: no cover
            os.close(fd)
            return None
        return fd

    @staticmethod
    def _tighten(fd: int) -> None:
        """Take group/other access away — never grant it, and never on a path.

        Windows has no ``os.fchmod`` and no group/other bits to take away, so
        there is nothing to do there; ACLs already keep the file to its owner.
        """
        fchmod = getattr(os, "fchmod", None)
        if fchmod is None:                            # pragma: no cover - Windows
            return
        try:
            st = os.fstat(fd)
            want = stat.S_IMODE(st.st_mode) & ~0o077
            if want != stat.S_IMODE(st.st_mode):
                fchmod(fd, want)
        except OSError:
            pass

    @staticmethod
    def _lock(fd: int, timeout: float = 1.0) -> bool:
        """Best-effort exclusive lock. Never blocks forever, never deadlocks."""
        if fcntl is None:
            return False
        end = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                if time.monotonic() >= end:
                    return False
                time.sleep(0.005)

    @staticmethod
    def _unlock(fd: int) -> None:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:                           # pragma: no cover
                pass

    def _load_history(self) -> None:
        path = self.history_path
        if path is None:
            return
        parent = path.parent
        if not parent.is_dir():
            try:
                parent.mkdir(mode=0o700, parents=True)   # ours, so ours to lock down
            except FileExistsError:                      # someone else got there first
                pass
            except OSError:
                self.history_path = None
                return
        raw = []
        fd = self._open_history("r")                  # read first: a file we may not own
        if fd is not None:
            try:
                raw = _read_all(fd).decode("utf-8", "replace").splitlines()
            except OSError:                           # pragma: no cover
                raw = []
            finally:
                os.close(fd)
        fd = self._open_history("a")                  # creates it, 0600, no symlink
        if fd is None:                                # unwritable, or not a plain file
            self.history_path = None                  # remember in RAM, write nothing
        else:
            self._tighten(fd)
            os.close(fd)
        entries = [ln for ln in raw if ln.strip() and not looks_secret(ln)]
        self._file_lines = len(raw)
        self._file_bytes = sum(len(ln.encode("utf-8", "replace")) + 1 for ln in raw)
        for line in entries[-self.history_max:]:
            if not self._history or self._history[-1] != line:
                self._history.append(line)
                self._call(self._rl, "add_history", line)
        # A secret an older lume left behind is recognised here and would
        # otherwise be left lying on disk, recognised and untouched, forever.
        if len(entries) != len(raw) or self._over_cap():
            self._trim_history_file()

    def _over_cap(self) -> bool:
        """Is the file due a trim? Both caps carry slack, for the same reason.

        A trim rewrites the whole file, so triggering one on the exact byte the
        cap is reached would rewrite it again on every single line after that.
        """
        return (self._file_lines > self.history_max + max(8, self.history_max // 10)
                or self._file_bytes > HISTORY_BYTES + HISTORY_BYTES // 8)

    def _append_history_file(self, text: str) -> None:
        if self.history_path is None:
            return
        fd = self._open_history("a")
        if fd is None:
            self.history_path = None                  # degrade quietly, never force
            return
        locked = False
        try:
            locked = self._lock(fd)
            self._tighten(fd)
            data = (text + "\n").encode("utf-8", "replace")
            os.write(fd, data)
            self._file_lines += 1
            self._file_bytes += len(data)
        except OSError:
            return
        finally:
            if locked:
                self._unlock(fd)
            os.close(fd)
        if self._over_cap():
            self._trim_history_file()

    def _trim_history_file(self) -> None:
        """Cap the file, in place and under a lock, so a second lume loses nothing.

        In place on purpose: rewriting to a temporary file and renaming would give
        every other process an fd onto an orphaned inode, and their appends would
        vanish with it. Read, filter and rewrite all happen inside one lock — and
        if the lock cannot be had, the trim is simply skipped. A rewrite that goes
        ahead without it destroys whatever another lume appended in the meantime,
        and a trim is always deferrable; an append never is.
        """
        if self.history_path is None:
            return
        fd = self._open_history("w")
        if fd is None:
            return
        locked = self._lock(fd)
        if fcntl is not None and not locked:
            os.close(fd)
            return
        try:
            self._tighten(fd)
            raw = _read_all(fd).decode("utf-8", "replace").splitlines()
            keep = [ln for ln in raw if ln.strip() and not looks_secret(ln)]
            keep = _cap_bytes(keep[-self.history_max:], HISTORY_BYTES)
            body = ("\n".join(keep) + "\n") if keep else ""
            data = body.encode("utf-8", "replace")
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, data)
            os.ftruncate(fd, len(data))
            self._file_lines = len(keep)
            self._file_bytes = len(data)
        except OSError:
            pass
        finally:
            if locked:
                self._unlock(fd)
            os.close(fd)

    # ------------------------------------------------------------------ readline

    @staticmethod
    def _call(obj, name: str, *args):
        """Call an optional readline entry point; libedit is missing several."""
        if obj is None:
            return None
        fn = getattr(obj, name, None)
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception:
            return None

    def _setup_readline(self) -> None:
        rl = self._rl
        if rl is None:
            return
        libedit = "libedit" in (getattr(rl, "__doc__", "") or "")
        # '/' is a completer delimiter by default, which would hide the slash from
        # the completer; a command line only ever splits on whitespace.
        self._call(rl, "set_completer_delims", " \t\n")
        self._call(rl, "set_completer", self._completer_fn)
        self._call(rl, "set_history_length", self.history_max)
        self._auto_history_off = self._set_auto_history(False)
        binds = ["bind ^I rl_complete"] if libedit else [
            "tab: complete",
            # Off, deliberately: readline's own paste handler rewrites CR as
            # newline, which cannot be undone. The binding below hands the paste
            # to lume instead, guards and line endings intact.
            "set enable-bracketed-paste off",
            '"%s": "%s\\C-m"' % (r"\e[200~", RL_PASTE_MARK),
            # ...and the closing guard is swallowed: on its own (a terminal that
            # started a paste it never finished) readline would leave a stray
            # '~' in the line.
            '"%s": ""' % r"\e[201~",
            # LF is not the Return key: no keyboard sends one once ICRNL is
            # off, so a line readline accepted on one is pasted content with
            # more of the same behind it. The mark is how that gets back here.
            '"%s": "%s\\C-m"' % (r"\C-j", RL_EOL_MARK),
            "set completion-ignore-case on",
            "set skip-completed-text on",
            "set show-all-if-ambiguous on",       # one Tab, not three, to disambiguate
        ]
        seq = _KEYSEQ.get(self.multiline_key.lower().replace(" ", "").replace("-", "+"))
        if seq and not libedit:
            binds.append(r'"%s": "\\\C-m"' % seq)
        for b in binds:
            self._call(rl, "parse_and_bind", b)
        # Editing mode is left alone on purpose: readline already defaults to
        # emacs, and a user's ~/.inputrc (vi mode, say) deserves to win.

    def _set_auto_history(self, on: bool) -> bool:
        rl = self._rl
        if rl is None or getattr(rl, "set_auto_history", None) is None:
            return False
        try:
            rl.set_auto_history(on)
            return True
        except Exception:                            # pragma: no cover
            return False

    def _forget_auto_history(self) -> None:
        """Drop the entry readline added behind our back (libedit has no switch)."""
        rl = self._rl
        if rl is None or self._auto_history_off:
            return
        n = self._call(rl, "get_current_history_length")
        if isinstance(n, int) and n > 0:
            self._call(rl, "remove_history_item", n - 1)

    def _complete(self, text: str, state: int):
        if state == 0:
            line = self._call(self._rl, "get_line_buffer") or text
            end = self._call(self._rl, "get_endidx")
            if isinstance(end, int) and 0 <= end <= len(line):
                line = line[:end]
            try:
                matches = self.completer(text, line)
            except TypeError:                        # a one-argument completer
                matches = self.completer(text)
            except Exception:
                matches = []
            self._matches = [str(m) for m in (matches or [])]
        try:
            return self._matches[state]
        except IndexError:
            return None

    # --------------------------------------------------------------------- close

    def close(self) -> None:
        """Flush and cap the history file, hand the terminal back, let go of readline."""
        if self._closed:
            return
        self._closed = True
        if self.history_path is not None and (self._file_lines > self.history_max
                                              or self._file_bytes > HISTORY_BYTES):
            # Exactly, now that nothing follows.
            self._trim_history_file()
        self._release_terminal()
        rl = self._rl
        if rl is not None:
            # Only if it is still ours: another live Prompt may have taken over,
            # and disarming *its* completer on our way out would be rude.
            if self._call(rl, "get_completer") is self._completer_fn:
                self._call(rl, "set_completer", None)
            self._call(rl, "set_startup_hook", None)
            if self._auto_history_off:
                self._set_auto_history(True)

    def __enter__(self) -> "Prompt":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _write_fd(fd: int, data: bytes) -> None:
    """Write a terminal mode sequence to the tty we are reading from."""
    try:
        os.write(fd, data)
    except OSError:
        pass


def _read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    out = b""
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            return out
        out += chunk


def _cap_bytes(lines: list, limit: int) -> list:
    """The newest lines that fit in `limit` bytes, oldest first."""
    total = 0
    keep = []
    for line in reversed(lines):
        total += len(line.encode("utf-8", "replace")) + 1
        if total > limit and keep:
            break
        keep.append(line)
    keep.reverse()
    return keep


def _clean_paste(text: str) -> str:
    """Normalise a chunk read straight from a tty into plain '\\n' lines.

    The single place newlines are normalised: the reader keeps CR and CRLF
    verbatim (see :meth:`Prompt._reading_mode`) precisely so this function can
    tell them apart, and bracketed-paste guards are the terminal's business, not
    the message's.
    """
    text = _PASTE_MARKS.sub("", text)
    text = _norm_eol(text)
    if text.endswith("\n"):
        text = text[:-1]
    return text
