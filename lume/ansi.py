"""Terminal primitives: capability detection, colour, width-aware text, Console.

Zero dependencies. Everything above this module renders through the helpers here,
so terminal quirks are handled in exactly one place.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import signal
import sys
import threading
import unicodedata
import weakref
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

RGB = tuple

ESC = "\x1b"
CSI = "\x1b["
RESET = "\x1b[0m"

# CSI sequences, OSC sequences (BEL- or ST-terminated), and lone charset selects.
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;:?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_]"
)


# --------------------------------------------------------------------------- caps


@dataclass(frozen=True)
class Caps:
    """What the attached terminal can actually do."""

    color: int = 0          # bits of colour: 0 (none), 4, 8, or 24
    unicode: bool = False   # safe to emit non-ASCII box/braille glyphs
    is_tty: bool = False
    width: int = 80
    height: int = 24
    hyperlinks: bool = False
    animation: bool = False  # tty + not dumb + motion not disabled

    @property
    def truecolor(self) -> bool:
        return self.color >= 24

    def with_size(self, width: int, height: int) -> "Caps":
        return replace(self, width=width, height=height)


def _env_flag(env, name: str) -> bool:
    """True when a variable is set to something other than an explicit falsey word."""
    v = env.get(name)
    return v is not None and v.strip().lower() not in ("", "0", "false", "no")


def _is_set(env, name: str) -> bool:
    """no-color.org semantics: present and non-empty, *whatever* the value says."""
    v = env.get(name)
    return v is not None and v != ""


def detect_background(env=None) -> str:
    """Guess the terminal's background: "light", "dark", or "" if unknown.

    `COLORFGBG` is the only widely-honoured convention for this (rxvt, konsole,
    iTerm and others set it). Getting it wrong is what makes a dark theme
    unreadable on a white terminal, so it is worth the six lines.
    """
    env = os.environ if env is None else env
    raw = env.get("COLORFGBG") or ""
    parts = [p for p in raw.split(";") if p.strip().isdigit()]
    if not parts:
        return ""
    background = int(parts[-1])
    # 0-6 and 8 are the dark slots; 7 and 9-15 are the light ones.
    return "light" if background == 7 or background >= 9 else "dark"


def enable_windows_vt() -> bool:
    """Turn on VT processing for legacy Windows consoles. No-op elsewhere."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):  # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:
        return False


def detect_caps(stream=None, env=None) -> Caps:
    """Detect terminal capabilities, honouring NO_COLOR / CLICOLOR / FORCE_COLOR / TERM."""
    stream = stream if stream is not None else sys.stdout
    env = os.environ if env is None else env

    try:
        is_tty = bool(stream.isatty())
    except Exception:
        is_tty = False

    term = (env.get("TERM") or "").lower()
    term_program = env.get("TERM_PROGRAM") or ""
    dumb = term == "dumb"
    windows_no_vt = os.name == "nt" and is_tty and not enable_windows_vt()

    # 1. Baseline from what the terminal advertises.
    color = 0
    if is_tty:
        colorterm = (env.get("COLORTERM") or "").lower()
        if colorterm in ("truecolor", "24bit") or "direct" in term:
            color = 24
        elif "256" in term:
            color = 8
        else:
            color = 4
        if term_program in ("iTerm.app", "WezTerm", "vscode", "ghostty", "Hyper", "rio"):
            color = max(color, 24)
        # Windows Terminal / ConEmu / ANSICON advertise through their own variables.
        if any(env.get(k) for k in ("WT_SESSION", "ConEmuANSI", "ANSICON")):
            color = max(color, 24)

    # 2. Explicit forcing, which also works when output is redirected.
    if _env_flag(env, "CLICOLOR_FORCE"):
        color = max(color, 4)      # "colour even when redirected", not "24-bit"
    force = env.get("FORCE_COLOR")
    if force is not None:
        # chalk/supports-color levels: 0 off, 1 = 16, 2 = 256, 3 = truecolor.
        level = force.strip().lower()
        if level in ("0", "false", "no"):
            color = 0
        elif level in ("true", "yes", ""):
            color = 4
        else:
            try:                       # clamp to 0-3; unparseable means level 1
                color = (0, 4, 8, 24)[min(max(int(level), 0), 3)]
            except ValueError:
                color = 4

    # 3. Explicit disabling wins over forcing.
    if _is_set(env, "NO_COLOR") or (env.get("CLICOLOR") or "").strip() == "0":
        color = 0

    # 4. A dumb terminal cannot be talked into colour by anything.
    if dumb:
        color = 0

    # Unicode
    enc = (getattr(stream, "encoding", None) or "").lower()
    lang = " ".join(env.get(k, "") for k in ("LC_ALL", "LC_CTYPE", "LANG")).lower()
    uni = "utf" in enc or "utf-8" in lang or "utf8" in lang
    if os.name == "nt" and (enc.startswith("utf") or env.get("WT_SESSION")):
        uni = True
    if _env_flag(env, "LUME_ASCII"):
        uni = False

    # Size: honour the injected env first so the seam works in tests.
    width = height = None
    try:
        width = int(env["COLUMNS"])
        if width < 10:
            width = None       # 0 or negative means "unset", not a 2-column tty
    except (KeyError, ValueError, TypeError):
        width = None
    try:
        height = int(env["LINES"])
        if height < 2:
            height = None
    except (KeyError, ValueError, TypeError):
        height = None
    if width is None or height is None:
        size = shutil.get_terminal_size(fallback=(80, 24))
        width = size.columns if width is None else width
        height = size.lines if height is None else height
    # shutil reads os.environ["COLUMNS"] itself, so a nonsense value can arrive
    # through the fallback as well as through the injected env.
    if width < 10:
        width = 80
    if height < 2:
        height = 24
    width = max(2, min(width, 1000))
    height = max(1, min(height, 500))

    # Hyperlinks are an OSC feature, not a colour feature: NO_COLOR must not kill them.
    hyperlinks = bool(is_tty and not dumb and term_program not in ("Apple_Terminal",))

    animation = bool(is_tty and not dumb
                     and not _env_flag(env, "LUME_NO_MOTION")
                     and not _is_set(env, "NO_COLOR"))

    if windows_no_vt:
        # A legacy console without VT processing renders escapes literally, but
        # its size is still worth knowing.
        color, uni, hyperlinks, animation = 0, False, False, False

    return Caps(
        color=color,
        unicode=uni,
        is_tty=is_tty,
        width=width,
        height=height,
        hyperlinks=hyperlinks,
        animation=animation,
    )


# ------------------------------------------------------------------------- colour


def hex_rgb(value: str) -> RGB:
    """'#1f2c3d' or '1f2c3d' -> (31, 44, 61)."""
    s = value.lstrip("#").strip()
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bad hex colour: {value!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def blend(a: RGB, b: RGB, t: float) -> RGB:
    """Linear interpolation between two colours; t in [0, 1]."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient(stops: Sequence[RGB], n: int) -> list:
    """n colours spread across an arbitrary number of stops."""
    if n <= 0:
        return []
    stops = list(stops)
    if len(stops) == 1:
        return [tuple(stops[0])] * n
    if n == 1:
        return [tuple(stops[0])]
    out = []
    segs = len(stops) - 1
    for i in range(n):
        pos = i / (n - 1) * segs
        idx = min(int(pos), segs - 1)
        out.append(blend(stops[idx], stops[idx + 1], pos - idx))
    return out


# The 16 ANSI slots as most terminals render them by default. These are
# user-configurable in practice, which is exactly why we only fall back to them.
_ANSI16 = (
    (0, 0, 0), (170, 0, 0), (0, 170, 0), (170, 85, 0),
    (0, 0, 170), (170, 0, 170), (0, 170, 170), (170, 170, 170),
    (85, 85, 85), (255, 85, 85), (85, 255, 85), (255, 255, 85),
    (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
)

_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)


def _build_xterm256():
    """Indices 16-255 of the xterm palette: the 6x6x6 cube, then the grey ramp."""
    out = []
    for r in _CUBE_LEVELS:
        for g in _CUBE_LEVELS:
            for b in _CUBE_LEVELS:
                out.append((r, g, b))
    out.extend((8 + 10 * i,) * 3 for i in range(24))
    return tuple(out)


_XTERM256 = _build_xterm256()  # index i corresponds to colour number i + 16


def _srgb_to_lab(c: RGB):
    """sRGB -> CIE L*a*b* (D65). Nearest-colour in Lab matches what the eye picks;
    nearest in raw sRGB does not, and drags everything toward grey."""
    def lin(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in c)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


_LAB256 = tuple(_srgb_to_lab(c) for c in _XTERM256)
_LAB16 = tuple(_srgb_to_lab(c) for c in _ANSI16)


def _nearest(lab, table):
    best, best_d = 0, None
    for i, ref in enumerate(table):
        d = (lab[0] - ref[0]) ** 2 + (lab[1] - ref[1]) ** 2 + (lab[2] - ref[2]) ** 2
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


@lru_cache(maxsize=4096)
def rgb_to_256(c: RGB) -> int:
    """Nearest xterm-256 colour, searching only 16-255: the first 16 slots are
    whatever the user's theme says they are, so they are not safe targets."""
    return 16 + _nearest(_srgb_to_lab(c), _LAB256)


@lru_cache(maxsize=4096)
def rgb_to_16(c: RGB) -> int:
    return _nearest(_srgb_to_lab(c), _LAB16)


def fg(color: RGB, caps: Caps) -> str:
    if not caps.color or color is None:
        return ""
    if caps.color >= 24:
        return f"{CSI}38;2;{color[0]};{color[1]};{color[2]}m"
    if caps.color >= 8:
        return f"{CSI}38;5;{rgb_to_256(color)}m"
    i = rgb_to_16(color)
    return f"{CSI}{(90 + i - 8) if i >= 8 else (30 + i)}m"


def bg(color: RGB, caps: Caps) -> str:
    if not caps.color or color is None:
        return ""
    if caps.color >= 24:
        return f"{CSI}48;2;{color[0]};{color[1]};{color[2]}m"
    if caps.color >= 8:
        return f"{CSI}48;5;{rgb_to_256(color)}m"
    i = rgb_to_16(color)
    return f"{CSI}{(100 + i - 8) if i >= 8 else (40 + i)}m"


# -------------------------------------------------------------------------- style


@dataclass(frozen=True)
class Style:
    """An immutable text style. Call it to wrap text in the right escapes."""

    fg: RGB = None
    bg: RGB = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    reverse: bool = False

    def codes(self, caps: Caps) -> str:
        if not caps.color:
            # No colour still leaves the whole attribute vocabulary, which is
            # exactly what the `mono` theme encodes its meaning in.
            attrs = []
            for flag, code in ((self.bold, "1"), (self.dim, "2"), (self.italic, "3"),
                               (self.underline, "4"), (self.reverse, "7"),
                               (self.strike, "9")):
                if flag:
                    attrs.append(code)
            return f"{CSI}{';'.join(attrs)}m" if attrs and caps.is_tty else ""
        parts = []
        if self.bold:
            parts.append("1")
        if self.dim:
            parts.append("2")
        if self.italic:
            parts.append("3")
        if self.underline:
            parts.append("4")
        if self.reverse:
            parts.append("7")
        if self.strike:
            parts.append("9")
        out = f"{CSI}{';'.join(parts)}m" if parts else ""
        out += fg(self.fg, caps) + bg(self.bg, caps)
        return out

    def __call__(self, text: str, caps: Caps) -> str:
        if text == "":
            return ""
        codes = self.codes(caps)
        return f"{codes}{text}{RESET}" if codes else text

    def open(self, caps: Caps) -> str:
        """Just the opening escapes, for callers managing their own nesting."""
        return self.codes(caps)

    def close(self, caps: Caps, restore: "Style" = None) -> str:
        """Close this style, optionally re-opening the one that enclosed it.

        `style(text, caps)` always closes with a full RESET, which is right for a
        standalone run and wrong inside a nested one — hence this pair.
        """
        if not self.codes(caps):
            return ""
        return RESET + (restore.codes(caps) if restore is not None else "")

    def merge(self, other: "Style") -> "Style":
        """Overlay `other` on top of self; unset fields fall through."""
        return Style(
            fg=other.fg if other.fg is not None else self.fg,
            bg=other.bg if other.bg is not None else self.bg,
            bold=self.bold or other.bold,
            dim=self.dim or other.dim,
            italic=self.italic or other.italic,
            underline=self.underline or other.underline,
            strike=self.strike or other.strike,
            reverse=self.reverse or other.reverse,
        )

    def __add__(self, other: "Style") -> "Style":
        return self.merge(other)


NULL_STYLE = Style()


# --------------------------------------------------------------------------- text


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


# Conjoining Hangul jamo occupy no column of their own; they compose onto the
# preceding syllable. Both the original block and Extended-B behave this way.
_JAMO_RANGES = ((0x1160, 0x11FF), (0xD7B0, 0xD7FF))
_ZERO_WIDTH_CATS = frozenset(("Mn", "Me", "Cf"))
#: Characters that promote a preceding narrow base to emoji (double-width) presentation.
_PRESENTATION = ("️", "⃣")


def char_width(ch: str) -> int:
    """Columns a single character occupies. See `display_width` for text."""
    o = ord(ch)
    if o == 0:
        return 0
    if o < 32 or 0x7F <= o < 0xA0:
        return 0
    for lo, hi in _JAMO_RANGES:
        if lo <= o <= hi:
            return 0
    # Category is the right test: `unicodedata.combining()` reports the canonical
    # combining class, which is non-zero for spacing marks that do take a column.
    if unicodedata.category(ch) in _ZERO_WIDTH_CATS:
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


def display_width(s: str) -> int:
    """Rendered column count: ANSI-stripped, wide-, combining- and VS16-aware.

    Known limit: ZWJ sequences and skin-tone modifiers are measured per component
    (as `wcwidth` also does); collapsing them needs full grapheme clustering.
    """
    total = 0
    last = 0
    promoted = False
    for ch in strip_ansi(s):
        if ch in _PRESENTATION:
            # U+FE0F / U+20E3 turn a text-presentation glyph into an emoji, which
            # every modern terminal advances two columns for.
            if last == 1 and not promoted:
                total += 1
                promoted = True
            continue
        w = char_width(ch)
        if w:
            last, promoted = w, False
        total += w
    return total


# ------------------------------------------------------------------ SGR state

_SGR_SET = {1: "bold", 2: "dim", 3: "italic", 4: "underline", 5: "blink",
            7: "reverse", 8: "conceal", 9: "strike", 53: "overline"}
_SGR_UNSET = {21: ("bold",), 22: ("bold", "dim"), 23: ("italic",), 24: ("underline",),
              25: ("blink",), 27: ("reverse",), 28: ("conceal",), 29: ("strike",),
              55: ("overline",)}
_ATTR_CODE = {"bold": "1", "dim": "2", "italic": "3", "underline": "4", "blink": "5",
              "reverse": "7", "conceal": "8", "strike": "9", "overline": "53"}
_ATTR_ORDER = ("bold", "dim", "italic", "underline", "blink", "reverse", "conceal",
               "strike", "overline")
_SGR_RE = re.compile(r"\x1b\[([0-9;:]*)m")
_OSC8_RE = re.compile(r"\x1b\]8;;([^\x07\x1b]*)(?:\x07|\x1b\\)")


class _SgrState:
    """Canonical SGR state, so a wrapped line re-opens with one short sequence
    instead of replaying every escape that led up to the break."""

    __slots__ = ("fg", "bg", "attrs", "extra")

    def __init__(self):
        self.fg = None
        self.bg = None
        self.attrs = set()
        self.extra = []      # sequences we chose not to model, replayed verbatim

    def copy(self) -> "_SgrState":
        other = _SgrState()
        other.fg, other.bg = self.fg, self.bg
        other.attrs = set(self.attrs)
        other.extra = list(self.extra)
        return other

    def reset(self) -> None:
        self.fg = self.bg = None
        self.attrs.clear()
        self.extra.clear()

    def feed(self, seq: str) -> None:
        m = _SGR_RE.fullmatch(seq)
        if m is None:
            return                                   # not an SGR sequence; ignore
        params = m.group(1)
        if ":" in params:
            # Colon-delimited colour forms are rare and fiddly; keep them verbatim.
            self.extra.append(seq)
            return
        if params == "":
            self.reset()
            return
        parts = [int(x) if x else 0 for x in params.split(";")]
        i = 0
        while i < len(parts):
            code = parts[i]
            if code == 0:
                self.reset()
            elif code in _SGR_SET:
                self.attrs.add(_SGR_SET[code])
            elif code in _SGR_UNSET:
                self.attrs.difference_update(_SGR_UNSET[code])
            elif code in (38, 48):
                mode = parts[i + 1] if i + 1 < len(parts) else 0
                take = 5 if mode == 2 else 3 if mode == 5 else 1
                chunk = ";".join(str(x) for x in parts[i:i + take])
                if code == 38:
                    self.fg = chunk
                else:
                    self.bg = chunk
                i += take
                continue
            elif 30 <= code <= 37 or 90 <= code <= 97:
                self.fg = str(code)
            elif code == 39:
                self.fg = None
            elif 40 <= code <= 47 or 100 <= code <= 107:
                self.bg = str(code)
            elif code == 49:
                self.bg = None
            i += 1

    def empty(self) -> bool:
        return not (self.fg or self.bg or self.attrs or self.extra)

    def render(self) -> str:
        if self.empty():
            return ""
        parts = [_ATTR_CODE[a] for a in _ATTR_ORDER if a in self.attrs]
        if self.fg:
            parts.append(self.fg)
        if self.bg:
            parts.append(self.bg)
        out = f"{CSI}{';'.join(parts)}m" if parts else ""
        return out + "".join(self.extra)


# ------------------------------------------------------------------ tokenizing


def _expand_tabs(text: str, tabsize: int) -> str:
    """Expand tabs against the *visible* column, ignoring escapes."""
    if "\t" not in text:
        return text
    out = []
    col = 0
    pos = 0
    for m in _ANSI_RE.finditer(text):
        chunk, esc = text[pos:m.start()], m.group(0)
        col = _expand_chunk(chunk, col, tabsize, out)
        out.append(esc)
        pos = m.end()
    _expand_chunk(text[pos:], col, tabsize, out)
    return "".join(out)


def _expand_chunk(chunk, col, tabsize, out):
    for ch in chunk:
        if ch == "\t":
            gap = tabsize - (col % tabsize)
            out.append(" " * gap)
            col += gap
        elif ch == "\n":
            out.append(ch)
            col = 0
        else:
            out.append(ch)
            col += char_width(ch)
    return col


def _tokenize(s: str):
    """Split into (kind, text) where kind is 'ansi' | 'link' | 'nl' | 'ws' | 'word'."""
    out = []
    pos = 0
    for m in _ANSI_RE.finditer(s):
        if m.start() > pos:
            out.extend(_split_plain(s[pos:m.start()]))
        seq = m.group(0)
        link = _OSC8_RE.fullmatch(seq)
        out.append(("link", link.group(1)) if link else ("ansi", seq))
        pos = m.end()
    if pos < len(s):
        out.extend(_split_plain(s[pos:]))
    return out


def _split_plain(s: str):
    out = []
    buf = ""
    kind = None
    for ch in s:
        k = "nl" if ch == "\n" else "ws" if ch in " \t" else "word"
        if k == "nl":
            if buf:
                out.append((kind, buf))
                buf = ""
            out.append(("nl", "\n"))
            kind = None
            continue
        if kind is None or k == kind:
            buf += ch
            kind = k
        else:
            out.append((kind, buf))
            buf, kind = ch, k
    if buf:
        out.append((kind, buf))
    return out


def _take_columns(word: str, width: int, force: bool = True):
    """Longest prefix of `word` fitting `width` columns, plus the remainder.

    Measures exactly as `display_width` does — including emoji-presentation
    promotion — because two measuring functions that disagree is how a wrapper
    starts overflowing the width it was given.
    """
    if width <= 0:
        return ("", word) if not force else (word[:1], word[1:])
    taken, w = 0, 0
    last, promoted = 0, False
    for i, ch in enumerate(word):
        if ch in _PRESENTATION:
            add = 1 if (last == 1 and not promoted) else 0
            if w + add > width:
                break
            w += add
            promoted = promoted or bool(add)
            taken = i + 1
            continue
        cw = char_width(ch)
        if w + cw > width:
            break
        w += cw
        if cw:
            last, promoted = cw, False
        taken = i + 1
    if taken == 0 and force:     # a single glyph wider than the whole line
        taken = 1
    return word[:taken], word[taken:]


def _fit_indent(indent: str, width: int) -> str:
    """Never let an indent eat the line: leave room for at least one wide glyph.

    ANSI-aware, because a char-based cut can slice an escape sequence in half and
    leak an unclosed style into the rest of the terminal.
    """
    room = max(0, width - 2)
    if display_width(indent) <= room:
        return indent
    out, used = "", 0
    state = _SgrState()
    for kind, tok in _tokenize(indent):
        if kind == "ansi":
            out += tok
            state.feed(tok)
            continue
        if kind == "link":
            continue             # a hyperlink in an indent is not worth carrying
        take, rest = _take_columns(tok, room - used, force=False)
        out += take
        used += display_width(take)
        if rest:
            break
    return out + (RESET if not state.empty() else "")


def _link_open(url: str) -> str:
    return f"{ESC}]8;;{url}{ESC}\\"


_LINK_CLOSE = f"{ESC}]8;;{ESC}\\"
_LINK_MARK = f"{ESC}]8;;"


def wrap(text: str, width: int, initial_indent: str = "", subsequent_indent: str = None,
         tabsize: int = 8) -> list:
    """ANSI-aware word wrap.

    Style state and open hyperlinks are closed at each break and re-opened on the
    next line, so every line stands alone. No line exceeds `width` columns unless
    `width < 2`, where a double-width glyph physically cannot fit.

    Interior blank lines are preserved; trailing ones are dropped, so `"text\n"`
    wraps to one line rather than two. Tabs are expanded against the visible column.
    """
    if subsequent_indent is None:
        subsequent_indent = initial_indent
    width = max(1, int(width))
    text = _expand_tabs(text, tabsize)
    initial_indent = _expand_tabs(initial_indent, tabsize)
    subsequent_indent = _expand_tabs(subsequent_indent, tabsize)

    lines = []
    state = _SgrState()
    link = None
    line_state = state.copy()
    line_link = None
    indent = _fit_indent(initial_indent, width)
    avail = max(1, width - display_width(indent))
    cur, curw = "", 0
    pending_ws = ""            # spaces held back: they may fall on a line break
    pending_esc = ""           # escapes seen while spaces are held back
    hard_start = True          # at the head of a logical (newline-delimited) line

    def emit():
        """Close the current visual line and start a fresh one."""
        nonlocal cur, curw, pending_ws, pending_esc, indent, avail
        nonlocal line_state, line_link, hard_start
        # Queued zero-width escapes belong to this line: one of them may be the
        # hyperlink close, and dropping it leaves the link open over everything
        # printed afterwards — including the user's shell prompt.
        if pending_esc:
            cur += pending_esc
            pending_esc = ""
        opener = line_state.render() + (_link_open(line_link) if line_link else "")
        closer = ""
        # Close a hyperlink only if this line actually opened one and it is still
        # open — otherwise we emit a close for a link that belongs to the next line.
        if link is not None and (line_link or _LINK_MARK in cur):
            closer += _LINK_CLOSE
        if not line_state.empty() or not state.empty() or "\x1b[" in cur:
            closer += RESET
        lines.append(indent + opener + cur + closer if cur else (indent.rstrip() or ""))
        cur, curw, pending_ws, pending_esc = "", 0, "", ""
        indent = _fit_indent(subsequent_indent, width)
        avail = max(1, width - display_width(indent))
        line_state = state.copy()
        line_link = link
        hard_start = False

    def place(text_run):
        """Append zero-width escapes, deferring them if spaces are still pending."""
        nonlocal cur, pending_esc
        if pending_ws:
            pending_esc += text_run
        else:
            cur += text_run

    for kind, tok in _tokenize(text):
        if kind == "ansi":
            state.feed(tok)
            if curw == 0 and not pending_ws:
                # At the head of a line the opener re-renders the state, so
                # emitting the raw escape here would duplicate it.
                line_state = state.copy()
            else:
                place(tok)
            continue

        if kind == "link":
            link = tok or None
            if curw == 0 and not pending_ws:
                line_link = link
            else:
                place(_link_open(tok) if tok else _LINK_CLOSE)
            continue

        if kind == "nl":
            emit()
            hard_start = True
            continue

        if kind == "ws":
            # Leading whitespace is meaningful at the head of a logical line and
            # noise at the head of a soft-wrapped continuation.
            if curw == 0 and not hard_start:
                continue
            if curw == 0 and hard_start:
                kept, _rest = _take_columns(tok, avail)
                cur += kept
                curw += display_width(kept)
                continue
            pending_ws += tok
            continue

        # word
        w = display_width(tok)
        gap = display_width(pending_ws) if curw else 0
        if curw and curw + gap + w > avail:
            emit()
            gap = 0
            pending_ws = ""
        if w > avail and curw == 0:
            # Split lazily: `avail` changes after each break when the continuation
            # indent differs from the current one.
            cur += pending_esc
            pending_esc = ""
            rest = tok
            while rest:
                chunk, rest = _take_columns(rest, avail)
                cur += chunk
                curw += display_width(chunk)
                if rest:
                    emit()
            pending_ws = ""
            continue
        cur += pending_ws + pending_esc + tok
        curw += gap + w
        pending_ws = ""
        pending_esc = ""

    emit()
    while len(lines) > 1 and strip_ansi(lines[-1]).strip() == "":
        lines.pop()
    return lines


def truncate(s: str, width: int, ellipsis: str = "…") -> str:
    """Cut to `width` columns, keeping ANSI codes intact and closing every run.

    A newline ends the result: a "line" that wrapped is no longer one line.
    """
    if width <= 0:
        return ""
    if display_width(s) <= width and "\n" not in strip_ansi(s):
        state = _SgrState()
        link = None
        for kind, tok in _tokenize(s):
            if kind == "ansi":
                state.feed(tok)
            elif kind == "link":
                link = tok or None
        return s + (_LINK_CLOSE if link else "") + (RESET if not state.empty() else "")
    if display_width(ellipsis) > width:
        ellipsis, _ = _take_columns(ellipsis, width)
    budget = max(0, width - display_width(ellipsis))
    out, used = "", 0
    state = _SgrState()
    link = None
    clipped = False
    for kind, tok in _tokenize(s):
        if kind == "ansi":
            out += tok
            state.feed(tok)
            continue
        if kind == "link":
            link = tok or None
            out += _link_open(tok) if tok else _LINK_CLOSE
            continue
        if kind == "nl":
            clipped = True
            break
        take, rest = _take_columns(tok, budget - used, force=False)
        out += take
        used += display_width(take)
        if rest:
            clipped = True
            break
    tail = (_LINK_CLOSE if link else "") + (RESET if not state.empty() else "")
    if not clipped and display_width(out) <= width:
        return out + tail
    return out + ellipsis + tail


def pad(s: str, width: int, align: str = "left", fill: str = " ") -> str:
    """Pad to `width` display columns. align: left | right | center."""
    if not fill or char_width(fill[0]) != 1:
        fill = " "          # a wide or empty filler cannot land on an exact column
    fill = fill[0]
    gap = width - display_width(s)
    if gap <= 0:
        return s
    if align == "right":
        return fill * gap + s
    if align == "center":
        left = gap // 2
        return fill * left + s + fill * (gap - left)
    return s + fill * gap


#: C0 controls (minus tab/newline), DEL, and the C1 range, which includes the
#: 8-bit CSI/OSC introducers. Model output is untrusted text: left unfiltered it
#: can set the window title, clear the screen, switch to the alternate buffer, or
#: write the user's clipboard via OSC 52.
_UNSAFE_TEXT = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize_text(text: str, keep_newlines: bool = True) -> str:
    """Strip control characters from untrusted text before it reaches the terminal.

    Everything the renderer itself emits is added *after* this, so removing every
    escape here costs nothing and closes the injection channel entirely.
    """
    cleaned = _UNSAFE_TEXT.sub("", text)
    if not keep_newlines:
        cleaned = cleaned.replace("\n", " ")
    return cleaned


_URL_UNSAFE = re.compile(r"[\x00-\x20\x7f-\x9f]")
#: Schemes a terminal may safely be handed. `javascript:` and `file:` are not.
_URL_SCHEMES = ("http://", "https://", "mailto:", "ftp://", "ftps://", "irc://",
                "ircs://", "news:", "nntp://", "tel:", "sms:")


def sanitize_url(url: str, limit: int = 2048) -> str:
    """Make a model-supplied URL safe to hand to the terminal.

    Strips every control byte (including the C1 range, whose U+009D is an 8-bit
    OSC introducer that would terminate ours early), caps the length, and allows
    only schemes that are meaningful to click. A relative or scheme-less URL is
    kept; anything with an unrecognised scheme is rejected outright.
    """
    cleaned = _URL_UNSAFE.sub("", str(url))[:limit]
    if not cleaned:
        return ""
    head = cleaned.split("/", 1)[0]
    if ":" in head:
        lowered = cleaned.lower()
        if not any(lowered.startswith(scheme) for scheme in _URL_SCHEMES):
            return ""
    return cleaned


def hyperlink(url: str, label: str, caps: Caps) -> str:
    if not caps.hyperlinks:
        return label
    safe = sanitize_url(url)
    if not safe:
        return label
    return f"{ESC}]8;;{safe}{ESC}\\{label}{ESC}]8;;{ESC}\\"


# ------------------------------------------------------------------------- cursor

HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"
CLEAR_LINE = f"{CSI}2K\r"
CLEAR_TO_END = f"{CSI}0K"
CLEAR_SCREEN = f"{CSI}2J{CSI}H"
CLEAR_SCROLLBACK = f"{CSI}3J"


def up(n: int = 1) -> str:
    return f"{CSI}{n}A" if n > 0 else ""


def down(n: int = 1) -> str:
    return f"{CSI}{n}B" if n > 0 else ""


def col(n: int = 1) -> str:
    return f"{CSI}{n}G"


# ------------------------------------------------------------------------ console


#: Live consoles, so an exit hook can put every hidden cursor back.
_CONSOLES = weakref.WeakSet()


#: Callbacks to run when the process is leaving, however it leaves. Registered by
#: anything that puts the terminal into a non-default state — raw mode, bracketed
#: paste, the alternate screen — so one exit path restores all of it.
_EXIT_HOOKS = []


def on_exit(callback) -> None:
    """Register a cleanup callback for normal exit and for SIGTERM/SIGHUP/SIGQUIT.

    Callbacks must be fast and must not raise; they run from a signal handler.
    """
    if callback not in _EXIT_HOOKS:
        _EXIT_HOOKS.append(callback)


def _run_exit_hooks():
    for callback in list(_EXIT_HOOKS):
        try:
            callback()
        except Exception:
            pass


def _restore_all_cursors():
    """Put every hidden cursor back, without ever blocking.

    This runs at exit and from signal handlers, so it must not wait on a lock a
    daemon thread may be holding — that would hang the process instead of tidying
    it. A single six-byte write is atomic enough for the last thing we ever do.
    """
    _run_exit_hooks()
    for console in list(_CONSOLES):
        try:
            if not console._cursor_hidden:
                continue
            got = console.lock.acquire(timeout=0.05)
            try:
                console._cursor_hidden = False
                erase = CLEAR_LINE if console._transient else ""
                console._transient = False
                console.stream.write(erase + SHOW_CURSOR)
                console.stream.flush()
            finally:
                if got:
                    console.lock.release()
        except Exception:
            pass


_signal_net_installed = False


def install_signal_net() -> bool:
    """Restore the cursor if the process is signalled away.

    Deliberately *not* done at import: a library that rewrites its host's signal
    handlers as a side effect of being imported will break hosts that set their
    own — notably one that has set SIGHUP to SIG_IGN, which `nohup` and every
    daemonised process do. `cli.py` calls this; embedders may choose not to.
    """
    global _signal_net_installed
    if _signal_net_installed:
        return True
    installed = False
    for name in ("SIGTERM", "SIGHUP", "SIGQUIT"):
        num = getattr(signal, name, None)
        if num is None:
            continue
        try:
            previous = signal.getsignal(num)
        except (ValueError, OSError):
            continue
        if previous in (signal.SIG_IGN, None):
            continue        # the host asked to ignore this signal; respect that

        def handler(signum, frame, _previous=previous):
            _restore_all_cursors()
            if callable(_previous):
                _previous(signum, frame)
                return
            # Default disposition: die *of the signal*, so the exit status and
            # any waiting parent see what actually happened.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        try:
            signal.signal(num, handler)
            installed = True
        except (ValueError, OSError):
            pass            # not the main thread, or signals unavailable here
    _signal_net_installed = installed
    return installed


atexit.register(_restore_all_cursors)


class Console:
    """The single writer to the terminal.

    One `write()` call is atomic against other writers. A *sequence* of writes is
    not — use `with console.batch():` when several writes form one frame.
    """

    def __init__(self, stream=None, caps: Caps = None, fixed_size: bool = False):
        self._fixed_size = fixed_size
        self.stream = stream if stream is not None else sys.stdout
        self.caps = caps if caps is not None else detect_caps(self.stream)
        self.lock = threading.RLock()
        self._cursor_hidden = False
        self._dead = False
        self._transient = False
        # A hidden cursor outliving the process is the one unrecoverable mess we
        # can leave in someone's terminal, so guarantee the restore centrally.
        _CONSOLES.add(self)

    # -- sizing
    def refresh(self) -> Caps:
        """Re-read the terminal size. A console built for a fixed width keeps it."""
        if self._fixed_size:
            return self.caps
        size = shutil.get_terminal_size(fallback=(self.caps.width, self.caps.height))
        self.caps = self.caps.with_size(
            max(2, min(size.columns, 1000)), max(1, min(size.lines, 500))
        )
        return self.caps

    @property
    def width(self) -> int:
        return self.caps.width

    @property
    def height(self) -> int:
        return self.caps.height

    # -- writing
    def write(self, *parts: str, flush: bool = True) -> bool:
        """Write parts as one atomic unit. Returns False once the sink is gone."""
        text = "".join(parts)
        if self._dead:
            return False
        if not text:
            if flush:
                self._flush()
            return not self._dead
        with self.lock:
            try:
                self.stream.write(text)
                if flush:
                    self._flush()
            except BrokenPipeError:
                self._dead = True          # the reader hung up: stop writing
            except (OSError, ValueError):
                # EBADF (fd closed under us) or EIO (terminal detached).
                self._dead = True
        return not self._dead

    @property
    def alive(self) -> bool:
        """False once the output sink has gone away (`lume | head` and friends)."""
        return not self._dead

    def set_transient(self, active: bool) -> None:
        """Declare that the current line holds redrawable output (a spinner frame).

        The exit hook erases it before restoring the cursor, so a process killed
        mid-animation does not leave a frozen frame with a shell prompt after it.
        """
        with self.lock:
            self._transient = bool(active)

    @contextmanager
    def batch(self):
        """Hold the lock across several writes so a frame cannot be torn."""
        with self.lock:
            yield self

    def print(self, text: str = "", end: str = "\n") -> None:
        self.write(text + end)

    def _flush(self):
        try:
            self.stream.flush()
        except BrokenPipeError:
            self._dead = True
        except (OSError, ValueError):
            self._dead = True

    def style(self, text: str, style: Style) -> str:
        return style(text, self.caps)

    # -- cursor
    def hide_cursor(self) -> None:
        with self.lock:
            if self.caps.is_tty and not self._cursor_hidden:
                self._cursor_hidden = True
                self.write(HIDE_CURSOR)

    def show_cursor(self) -> None:
        with self.lock:
            if self.caps.is_tty and self._cursor_hidden:
                self._cursor_hidden = False
                self.write(SHOW_CURSOR)

    def clear_line(self) -> None:
        if self.caps.is_tty:
            self.write(CLEAR_LINE)

    def clear_screen(self, scrollback: bool = False) -> None:
        if self.caps.is_tty:
            self.write(CLEAR_SCREEN + (CLEAR_SCROLLBACK if scrollback else ""))
