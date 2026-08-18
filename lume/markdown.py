"""Streaming Markdown renderer and a hand-written syntax highlighter.

Assistant replies arrive as a token stream, so the renderer is *incremental and
append-only*: :meth:`MarkdownStream.feed` returns only text whose meaning can no
longer change, and everything else waits in a buffer. The headline invariant is

    "".join(stream.feed(c) for c in chunks) + stream.close()
        == render_markdown(whole_text, ...)

for **any** chunking of ``whole_text`` — so nothing here may depend on where a
chunk boundary happened to fall. Two rules keep that true: complete source lines
are the only thing the block machine ever sees, and a partial trailing line is
only rendered up to the last position whose meaning is already decided
(see :func:`_parse_inline` and its ``unresolved`` result).

Model output is untrusted, so every control character is stripped on the way in
(:meth:`MarkdownStream.feed`) and every *displayed* URL goes through
:func:`~lume.ansi.sanitize_url`. Everything the renderer emits itself is added
after that filter, so it costs the output nothing.

Documented choices:
  * Long lines inside a code block are **truncated** with an ellipsis, never
    wrapped — code keeps its column structure, which is the whole point of it.
  * A code box always fills the available width. Its top border is printed
    before the first code line has arrived, so the box cannot be sized to its
    contents without buffering the whole block — which is exactly the streaming
    the box exists to show. A table *is* buffered whole, so it is content-sized.
  * A ``=====`` underline makes the line above a level-1 heading, but only
    while none of the paragraph has been printed yet: promoting it later
    would mean reprinting. ``---`` on its own line is always a thematic
    break, never a setext underline.
  * A table is recognised only at the start of a block, never as an interruption
    of a paragraph, and the whole table is buffered until it ends (column widths
    are unknowable before then).
  * Reference-link definitions are not resolved; each is shown on a line of its
    own rather than glued into the surrounding prose.
  * Containers nest at most ``_MAX_CONTAINER_DEPTH`` deep, and only while
    ``_MIN_CONTENT`` columns are left for the content. Deeper levels render
    flat — unbounded nesting overflows the terminal and then the stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .ansi import (
    RESET,
    Caps,
    Style,
    display_width,
    hyperlink,
    pad,
    sanitize_text,
    sanitize_url,
    strip_ansi,
    truncate,
    wrap,
)
from .theme import Theme

__all__ = [
    "MarkdownStream",
    "render_markdown",
    "highlight",
    "supported_languages",
]


# ----------------------------------------------------------------------- glyphs


class _Glyphs:
    """Box-drawing and marker characters, with an ASCII twin for every one."""

    def __init__(self, unicode: bool) -> None:
        self.unicode = unicode
        if unicode:
            self.tl, self.tr, self.bl, self.br = "╭", "╮", "╰", "╯"
            self.h, self.v = "─", "│"
            self.lt, self.rt, self.tt, self.bt, self.cross = (
                "├", "┤", "┬", "┴", "┼",
            )
            self.bar = "▌"
            self.rule = "─"
            self.h1 = "━"
            self.h2 = "─"
            self.bullets = ("•", "◦", "▪")
            self.check = "✓"
            self.ell = "…"
            self.image = "▣"
        else:
            self.tl = self.tr = self.bl = self.br = "+"
            self.h, self.v = "-", "|"
            self.lt = self.rt = self.tt = self.bt = self.cross = "+"
            self.bar = "|"
            self.rule = "-"
            self.h1 = "="
            self.h2 = "-"
            self.bullets = ("-", "*", "+")
            self.check = "x"
            self.ell = "..."
            self.image = "[img]"


# ----------------------------------------------------------------- inline parse

_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

_URL_RE = re.compile(
    r"(?:https?://|ftp://)[^\s<>\[\]{}\"'`\\]+"
    r"|www\.[^\s<>\[\]{}\"'`\\]+"
    r"|mailto:[^\s<>\[\]{}\"'`\\]+"
)
_AUTOLINK_RE = re.compile(r"<((?:[a-zA-Z][a-zA-Z0-9+.\-]{1,31}:|mailto:)[^<>\s]+)>")


def _trim_url(url: str) -> tuple:
    """Strip trailing sentence punctuation that a bare URL almost never owns."""
    tail = ""
    while url and url[-1] in ".,;:!?":
        tail = url[-1] + tail
        url = url[:-1]
    while url.endswith(")") and url.count("(") < url.count(")"):
        tail = ")" + tail
        url = url[:-1]
    return url, tail


def _find_run(s: str, i: int, ch: str) -> int:
    j = i
    while j < len(s) and s[j] == ch:
        j += 1
    return j


def _parse_inline(s: str, breaks: list = None):
    """Parse inline Markdown into nodes.

    Returns ``(nodes, unresolved)`` where *unresolved* is the index of the first
    construct that a later chunk could still change (``len(s)`` when everything
    is decided). Streaming uses that index to know how much is safe to print.

    When a list is passed as *breaks* it collects the offsets of every space that
    sits at the top level of this string — outside every link, code span and
    emphasis run. Those are the only positions where cutting the source cannot
    change how the text before the cut renders (see :func:`_stable_cut`).
    """
    nodes = []
    buf = []
    unresolved = len(s)
    n = len(s)
    i = 0
    # A '[' past the last ']' can never close, and re-discovering that with a
    # full scan for every one of them is what made "[" * 5000 quadratic.
    last_close = s.rfind("]")

    def flush():
        if buf:
            nodes.append(("text", "".join(buf)))
            del buf[:]

    def unres(pos):
        nonlocal unresolved
        if pos < unresolved:
            unresolved = pos

    while i < n:
        c = s[i]

        if c == "\\":
            if i + 1 >= n:
                unres(i)          # the escaped character has not arrived yet
                buf.append("\\")
                i += 1
                continue
            if s[i + 1] in _PUNCT:
                buf.append(s[i + 1])
                i += 2
                continue
            buf.append("\\")
            i += 1
            continue

        if c == "`":
            j = _find_run(s, i, "`")
            run = s[i:j]
            k = j
            close = -1
            while True:
                k = s.find(run, k)
                if k < 0:
                    break
                if k + len(run) < n and s[k + len(run)] == "`":
                    k = _find_run(s, k, "`")
                    continue
                close = k
                break
            if close < 0:
                unres(i)
                buf.append(run)
                i = j
                continue
            content = s[j:close]
            if len(content) >= 2 and content[0] == " " and content[-1] == " " and content.strip():
                content = content[1:-1]
            flush()
            nodes.append(("code", content))
            i = close + len(run)
            continue

        if c == "!" and i + 1 < n and s[i + 1] == "[":
            if i + 1 > last_close:
                unres(i)
                buf.append(c)
                i += 1
                continue
            node, end, bad = _parse_link(s, i, True)
            if node is not None:
                flush()
                nodes.append(node)
                i = end
                continue
            if bad:
                unres(i)
            buf.append(c)
            i += 1
            continue

        if c == "[":
            if i > last_close:
                unres(i)
                buf.append(c)
                i += 1
                continue
            node, end, bad = _parse_link(s, i, False)
            if node is not None:
                flush()
                nodes.append(node)
                i = end
                continue
            if bad:
                unres(i)
            buf.append(c)
            i += 1
            continue

        if c == "<":
            m = _AUTOLINK_RE.match(s, i)
            if m:
                flush()
                nodes.append(("link", [("text", m.group(1))], m.group(1), False))
                i = m.end()
                continue
            rest = s[i:]
            if ">" not in rest and not re.search(r"\s", rest):
                unres(i)
            buf.append(c)
            i += 1
            continue

        if c in "*_~" and (i == 0 or s[i - 1] != "\\"):
            j = _find_run(s, i, c)
            run_len = j - i
            prev = s[i - 1] if i > 0 else " "
            nxt = s[j] if j < n else " "
            ok_len = run_len in (1, 2, 3) if c != "~" else run_len == 2
            can_open = ok_len and not nxt.isspace()
            if c == "_":
                can_open = can_open and not (prev.isalnum() or prev == "_")
            if can_open:
                close = _find_closer(s, j, c, run_len)
                if close < 0:
                    unres(i)
                else:
                    inner, _ = _parse_inline(s[j:close])
                    kind = {1: "em", 2: "strong", 3: "emstrong"}[run_len]
                    if c == "~":
                        kind = "strike"
                    flush()
                    nodes.append((kind, inner))
                    i = close + run_len
                    continue
            elif j >= n:
                unres(i)          # the run itself may still grow
            buf.append(s[i:j])
            i = j
            continue

        if (c in "hwfm") and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] in "@/._-")):
            m = _URL_RE.match(s, i)
            if m:
                url, tail = _trim_url(m.group(0))
                if url:
                    flush()
                    href = url if "://" in url or url.startswith("mailto:") else "https://" + url
                    nodes.append(("link", [("text", url)], href, False))
                    i = m.end() - len(tail)
                    continue

        if breaks is not None and c in " \t":
            breaks.append(i)
        buf.append(c)
        i += 1

    flush()
    return nodes, unresolved


def _parse_link(s: str, i: int, image: bool):
    """Try to read ``[label](dest)`` at *i*. Returns (node, end, maybe_later)."""
    n = len(s)
    b = i + 1 if image else i        # index of '['
    depth = 0
    j = b
    end = -1
    while j < n:
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "`":
            k = _find_run(s, j, "`")
            run = s[j:k]
            nxt = s.find(run, k)
            if nxt < 0:
                return None, i, True       # the span may still close and hide the ']'
            j = nxt + len(run)
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = j
                break
        j += 1
    if end < 0:
        return None, i, True                       # a ']' may still arrive
    if end + 1 >= n:
        return None, i, True                       # a '(' may still arrive
    if s[end + 1] != "(":
        return None, i, False
    depth = 1
    k = end + 2
    while k < n:
        ch = s[k]
        if ch == "\\":
            k += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        k += 1
    if k >= n:
        return None, i, True                       # unbalanced, may still close
    dest = s[end + 2:k].strip()
    if dest.startswith("<") and dest.endswith(">"):
        dest = dest[1:-1]
    dest = dest.split(" ", 1)[0] if " " in dest else dest
    label, _ = _parse_inline(s[b + 1:end])
    return ("link", label, dest, image), k + 1, False


def _find_closer(s: str, start: int, ch: str, run_len: int) -> int:
    """Index of the delimiter run that closes an emphasis opener, or -1."""
    n = len(s)
    k = start
    while k < n:
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "`":
            j = _find_run(s, k, "`")
            run = s[k:j]
            nxt = s.find(run, j)
            if nxt < 0:
                return -1                  # undecided: a later '`' swallows this region
            k = nxt + len(run)
            continue
        if c == ch:
            j = _find_run(s, k, ch)
            if j - k == run_len and k > start:
                prev = s[k - 1]
                nxt = s[j] if j < n else " "
                ok = not prev.isspace()
                if ch == "_":
                    ok = ok and not (nxt.isalnum() or nxt == "_")
                if ok:
                    return k
            k = j
            continue
        k += 1
    return -1


def _cut_breaks(s: str) -> tuple:
    """``(stable cut, every top-level break offset)`` — see :func:`_stable_cut`."""
    breaks = []
    _, unresolved = _parse_inline(s, breaks)
    cut = min(unresolved, len(s))
    ws = -1
    for pos in breaks:
        if pos >= cut:
            break
        ws = pos
    return (ws if ws > 0 else 0), breaks


def _stable_cut(s: str) -> int:
    """How much of *s* can be rendered now: decided text, ending on a word break.

    The break has to be a *top-level* space. Cutting inside a construct that is
    already complete (``[a link](url)`` -> ``[a``) would render the tail as
    something else entirely, and the escapes that fall on the resulting line
    break would not survive the rest of the text arriving — which is exactly the
    append-only invariant.
    """
    return _cut_breaks(s)[0]


_DUP_RESET_RE = re.compile("(?:" + re.escape(RESET) + "){2,}")


def _same_lines(a: list, b: list) -> bool:
    """Line-by-line equality, ignoring repeated resets.

    ``wrap()`` closes the style once per break and once more at end of text, so
    the last line of a prefix carries one reset more than the same line does in
    the middle of a longer text. Nothing is printed differently, so the extra
    reset must not veto a checkpoint (see :meth:`MarkdownStream._checkpoint`).
    """
    if len(a) != len(b):
        return False
    return all(_DUP_RESET_RE.sub(RESET, x) == _DUP_RESET_RE.sub(RESET, y)
               for x, y in zip(a, b))


_SGR_WS_RE = re.compile(r"((?:\x1b\[[0-9;]*m)+)( +)")


def _tidy(line: str) -> str:
    """Move a space that sits inside the style codes of the next word out of them.

    `wrap()` defers whitespace until it knows the next word fits, so the space
    lands *after* the SGR codes and gets underlined/highlighted along with the
    word. Nothing here changes the printed columns — only which style paints
    the gap between two words.
    """
    if "\x1b" not in line:
        return line

    def swap(m):
        codes, ws = m.group(1), m.group(2)
        cut = codes.rfind(RESET)
        if cut < 0:
            return ws + codes
        cut += len(RESET)
        return codes[:cut] + ws + codes[cut:]

    return _SGR_WS_RE.sub(swap, line)


def _attrs(st: Style) -> Style:
    """Just the attribute bits of a style — used when the base colour must win."""
    return Style(
        bold=st.bold, dim=st.dim, italic=st.italic,
        underline=st.underline, strike=st.strike,
    )


def _render_nodes(nodes, theme: Theme, caps: Caps, base: Style, keep_color: bool,
                  g: _Glyphs, markers: bool = True) -> str:
    """Render inline nodes.

    `markers` keeps the literal ``**``/``*``/``~~`` when the terminal cannot draw
    a style that differs from the surrounding text — losing the emphasis entirely
    would lose meaning. Blocks that are already visually distinct on their own
    (headings) pass False: an asterisk there reads as a rendering failure.
    """
    out = []
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(base(node[1], caps))
        elif kind == "code":
            st = theme["md.code"]
            if caps.color >= 8:
                st = st + theme["md.code_bg"]
            # With no attributes to spare, the backticks are the only signal left.
            text = node[1] if st.codes(caps) else "`" + node[1] + "`"
            out.append(st(text, caps))
        elif kind in ("em", "strong", "emstrong", "strike"):
            token = {
                "em": "md.italic", "strong": "md.bold",
                "emstrong": "md.bold", "strike": "md.strike",
            }[kind]
            over = theme[token]
            if kind == "emstrong":
                over = over + _attrs(theme["md.italic"])
            style = base + over if keep_color else base + _attrs(over)
            body = _render_nodes(node[1], theme, caps, style, keep_color, g, markers)
            if markers and style.codes(caps) == base.codes(caps):
                mark = {"em": "*", "strong": "**", "emstrong": "***", "strike": "~~"}[kind]
                body = base(mark, caps) + body + base(mark, caps)
            out.append(body)
        elif kind == "link":
            _, label, url, image = node
            # Both branches below *display* the URL, so it goes through the same
            # filter `hyperlink()` uses — a dimmed "(url)" reaches the terminal
            # just as directly as an OSC 8 payload does.
            url = sanitize_url(url) if url else url
            style = base + theme["md.link"] if keep_color else base + _attrs(theme["md.link"])
            body = _render_nodes(label, theme, caps, style, keep_color, g, markers)
            plain = strip_ansi(body)
            if image:
                body = theme.render(g.image + " ", "md.link_url", caps) + body
            if caps.hyperlinks and url:
                out.append(hyperlink(url, body, caps))
            elif url and url != plain and url != "mailto:" + plain and not image:
                out.append(body + theme.render(" (" + url + ")", "md.link_url", caps))
            elif url and image:
                out.append(body + theme.render(" (" + url + ")", "md.link_url", caps))
            else:
                out.append(body)
    return "".join(out)


# ------------------------------------------------------------------ block syntax

_ATX_RE = re.compile(r"(#{1,6})(?:[ \t]+(.*?))?[ \t]*(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"(`{3,}|~{3,})[ \t]*(.*)$")
_HR_RE = re.compile(r"(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$")
_ITEM_RE = re.compile(r"([-*+]|\d{1,9}[.)])([ \t]+|$)(.*)$")
_TASK_RE = re.compile(r"\[([ xX])\](?:[ \t]+|$)(.*)$")
_DELIM_RE = re.compile(r"^[ \t]*:?-+:?[ \t]*$")
_HARD_BREAK_RE = re.compile(r"(?:[ ]{2,}|\\)$")
_SETEXT_RE = re.compile(r"[ ]{0,3}=+[ \t]*$")
#: ``[label]: dest "title"`` — a definition, not prose. Nothing here resolves
#: references, but gluing the URL into the next sentence is strictly worse than
#: showing the definition on a line of its own.
_LINKDEF_RE = re.compile(
    r"[ ]{0,3}\[([^\]\n]{1,300})\]:[ \t]*(\S+)"
    r"(?:[ \t]+([\"'(][^\n]*))?[ \t]*$")

#: First characters that could still turn a partial line into another block.
_RISKY_START = set("#>-*+~`_=|0123456789\t")

#: Columns a nested container must leave for its content. A quote or a list item
#: prints its prefix *outside* the child's width, so without a floor a document
#: like ``">" * 200`` walks the content width down past zero — first overflowing
#: the terminal, then blowing the interpreter stack one child renderer at a time.
_MIN_CONTENT = 8

#: And a hard ceiling for the cases a wide terminal would otherwise allow: every
#: level is a nested renderer, so depth costs stack frames on every fed chunk.
_MAX_CONTAINER_DEPTH = 12


def _is_block_start(line: str) -> bool:
    """Would this line interrupt a paragraph?"""
    s = line.lstrip(" ")
    if not s.strip():
        return True
    if len(line) - len(s) >= 4:
        return False                      # lazy continuation ignores indentation
    if s[0] == "#" and _ATX_RE.match(s):
        return True
    if s[0] in "`~" and _FENCE_RE.match(s):
        return True
    if s[0] in "*-_" and _HR_RE.match(s):
        return True
    if s[0] == ">":
        return True
    m = _ITEM_RE.match(s)
    if m and m.group(3).strip():
        marker = m.group(1)
        if marker[0].isdigit():
            return marker[:-1] == "1"
        return True
    return False


def _partial_ok(p: str, at_block_start: bool) -> bool:
    """Can this unfinished line already be treated as flowing text?

    Conservative on purpose: anything that a later character could turn into a
    different block (a list marker, a fence, a table row) stays buffered.
    """
    s = p.lstrip(" ")
    if not s:
        return False
    if len(p) - len(s) >= 4:
        return False
    if s[0] in _RISKY_START:
        return False
    # A '[' only decides between prose and a link definition once the line ends,
    # and the two render as different *blocks* — so it has to wait.
    if at_block_start and (s[0] == "[" or "|" in s):
        return False
    return True


def _split_cells(row: str) -> list:
    """Split a table row on unescaped pipes, dropping the outer border pipes."""
    s = row.strip()
    cells, buf, i = [], [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            buf.append(s[i:i + 2])
            i += 2
            continue
        if c == "|":
            cells.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    cells.append("".join(buf))
    if cells and not cells[0].strip() and s.startswith("|"):
        cells.pop(0)
    if cells and not cells[-1].strip() and s.endswith("|") and not s.endswith("\\|"):
        cells.pop()
    return [c.strip() for c in cells]


def _is_delimiter_row(line: str, ncols: int) -> bool:
    s = line.strip()
    if "-" not in s or not s:
        return False
    if not set(s) <= set("-:| \t"):
        return False
    cells = _split_cells(s)
    if not cells or len(cells) != ncols:
        return False
    return all(_DELIM_RE.match(c) for c in cells)


def _alignments(delim: str, ncols: int) -> list:
    out = []
    for c in _split_cells(delim):
        c = c.strip()
        if c.startswith(":") and c.endswith(":"):
            out.append("center")
        elif c.endswith(":"):
            out.append("right")
        else:
            out.append("left")
    while len(out) < ncols:
        out.append("left")
    return out[:ncols]


# ------------------------------------------------------------------------- flow


class _Flow:
    """A run of source lines that renders as wrapped, inline-formatted text.

    The source is kept *joined* rather than as a list of lines, and the part of
    it that has already been committed to the output is dropped
    (:meth:`MarkdownStream._checkpoint`). Committed lines can never change, so
    carrying their source forward only means re-parsing and re-wrapping the
    whole paragraph on every character that arrives — quadratic in the length of
    a paragraph, which is exactly the shape a model emits prose in.
    """

    __slots__ = ("segs", "text", "cur", "cur_done", "base", "split",
                 "emitted", "token", "width", "opened")

    def __init__(self, token: str, width: int) -> None:
        self.segs = []          # segments closed by a hard break, not yet emitted
        self.text = ""          # open segment: joined text of the resolved lines
        self.cur = ""           # current source line, minus the committed prefix
        self.cur_done = False   # ... and whether that line has finished arriving
        self.base = 0           # chars of the current source line already dropped
        self.split = False      # a hard break has closed a segment at some point
        self.emitted = 0
        self.token = token
        self.width = width
        self.opened = False

    def _fold(self, raw: str) -> None:
        """Absorb a source line now known *not* to be the last one.

        Only then is a trailing backslash or double space a hard break: at the end of
        a paragraph it is just trailing whitespace, so the decision waits until
        the next line proves there is one.
        """
        hard = bool(_HARD_BREAK_RE.search(raw))
        t = raw[:-1] if (hard and raw.endswith("\\")) else raw
        t = t.strip()
        self.text = (self.text + " " + t) if (self.text and t) else (self.text or t)
        if hard:
            self.segs.append(self.text)
            self.text = ""
            self.split = True

    def add(self, line: str) -> None:
        """A complete source line."""
        if self.cur_done:
            self._fold(self.cur)
            self.base = 0
            self.cur = line
        else:
            self.cur = line[self.base:]     # the same line, now finished
        self.cur_done = True

    def partial(self, line: str) -> None:
        """The unfinished trailing line, re-sent in full on every chunk."""
        if self.cur_done:
            self._fold(self.cur)
            self.base = 0
        self.cur = line[self.base:]
        self.cur_done = False

    def piece(self, final: bool) -> str:
        # A line that may still grow keeps its trailing spaces: they only become
        # a hard break once another line follows.
        return self.cur.strip() if final else self.cur.lstrip(" ")

    def open_text(self, final: bool) -> tuple:
        """``(text of the open segment, offset of the current line within it)``."""
        p = self.piece(final)
        if self.text and p:
            return self.text + " " + p, len(self.text) + 1
        return (self.text or p), (len(self.text) if self.text else 0)


class _Sub:
    """A nested renderer plus the prefix its output lines wear."""

    __slots__ = ("child", "kind", "first", "rest", "fed", "used")

    def __init__(self, child, kind, first, rest):
        self.child = child
        self.kind = kind          # "quote" | "item"
        self.first = first
        self.rest = rest
        self.fed = ""             # part of the current source line already sent
        self.used = False

    def push(self, text: str, complete: bool) -> str:
        """Feed the child only the part of the current source line it has not seen.

        Streaming shows a partial line to the child first and the finished line
        later; sending the delta is what keeps the child's view identical to the
        one it would have had from a single un-chunked feed."""
        delta = text[len(self.fed):] if text.startswith(self.fed) else text
        if complete:
            self.fed = ""
            return self.child.feed(delta + "\n")
        self.fed = text
        return self.child.feed(delta)


@dataclass
class _List:
    ordered: bool
    indent: int
    content_indent: int = 0
    number: int = 1
    blanks: int = 0
    marker_width: int = 2


class MarkdownStream:
    """Incremental Markdown renderer.

    Feed it Markdown in arbitrary pieces; each :meth:`feed` returns the text that
    is ready to print *now*. Nothing already returned is ever revised, so the
    caller can write straight to the terminal.
    """

    def __init__(self, theme: Theme, caps: Caps, width: int = None, indent: str = "",
                 _depth: int = 0, _text_token: str = "app.text", _cdepth: int = 0) -> None:
        self.theme = theme
        self.caps = caps
        self.width = max(1, int(width) if width else caps.width)
        self.g = _Glyphs(caps.unicode)
        self.depth = _depth
        self.cdepth = _cdepth
        self.text_token = _text_token
        # The indent is *part of* the line, so it has to be fitted before the
        # content width is derived from it: an indent wider than the terminal
        # would otherwise push every line past `width`.
        room = max(0, self.width - _MIN_CONTENT)
        if display_width(indent) > room:
            indent = truncate(indent, room, "")
        self.indent = indent
        self.inner = max(1, self.width - display_width(indent))
        # Nothing deeper than this can be drawn without overflowing, so deeper
        # containers render flat (see :meth:`_can_nest`).
        self.flat = not self._can_nest(2)

        self._buf = ""
        self._chunks = []
        self._closed = False

        self._state = None            # None|para|fence|code|table|table_pending|quote|list
        self._flow = None
        self._sub = None
        self._list = None
        self._fence = None
        self._icode_blanks = 0
        self._icode_open = False
        self._table = None
        self._pending_row = None

        self._blank_pending = False
        self._emitted_any = False
        self._last_blank = False
        self._prev_kind = None

    # ------------------------------------------------------------------ public

    def feed(self, chunk: str) -> str:
        """Consume a chunk of Markdown; return text ready to print now.

        The chunk is untrusted model output, so every control character is
        stripped here, at the one place text enters the renderer. Left in, a
        reply could set the window title, clear the screen, switch to the
        alternate buffer, overwrite the line with ``\\r``, or write the user's
        clipboard with OSC 52. Everything the renderer emits is added *after*
        this, so the filter costs the output nothing.

        Filtering per chunk is safe for the append-only invariant because
        :func:`~lume.ansi.sanitize_text` is a per-character filter: sanitising
        the pieces and joining is the same as sanitising the join.
        """
        if self._closed or not chunk:
            return self._take()
        self._buf += sanitize_text(chunk)
        self._run(final=False)
        return self._take()

    def close(self) -> str:
        """Flush everything still buffered. A second call returns ''."""
        if self._closed:
            return ""
        self._closed = True
        self._run(final=True)
        return self._take()

    # ----------------------------------------------------------------- plumbing

    def _take(self) -> str:
        if not self._chunks:
            return ""
        out = "".join(self._chunks)
        del self._chunks[:]
        return out

    def _run(self, final: bool) -> None:
        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break
            line, self._buf = self._buf[:nl], self._buf[nl + 1:]
            self._line(line[:-1] if line.endswith("\r") else line)
        if final:
            if self._buf:
                line, self._buf = self._buf, ""
                self._line(line[:-1] if line.endswith("\r") else line)
            self._close_blocks()
        elif self._buf:
            # A trailing CR is almost always the first half of a CRLF that has
            # not finished arriving; treating it as text would let a partial
            # line render differently from the complete one.
            partial = self._buf[:-1] if self._buf.endswith("\r") else self._buf
            if partial:
                self._partial(partial)

    def _emit(self, line: str = "") -> None:
        if line and strip_ansi(line).strip():
            self._chunks.append(self.indent + line + "\n")
            self._last_blank = False
        else:
            self._chunks.append("\n")
            self._last_blank = True
        self._emitted_any = True

    def _sep(self, kind: str) -> None:
        """One blank line between blocks — never two, never a leading one."""
        if self._emitted_any:
            tight = not self._blank_pending and (
                (self._prev_kind == "para" and kind == "list")
                or (self._prev_kind == "linkdef" and kind == "linkdef"))
            if not tight and not self._last_blank:
                self._emit()
        self._blank_pending = False
        self._prev_kind = kind

    def _inline(self, text: str, token: str = "app.text") -> str:
        nodes, _ = _parse_inline(text)
        return _render_nodes(nodes, self.theme, self.caps, self.theme[token],
                             token == "app.text", self.g)

    def _rule_style(self, text: str, token: str) -> str:
        return self.theme.render(text, token, self.caps)

    # ------------------------------------------------------------ line routing

    def _line(self, line: str) -> None:
        st = self._state
        if st == "fence":
            self._fence_line(line)
        elif st == "code":
            self._icode_line(line)
        elif st == "table":
            self._table_line(line)
        elif st == "table_pending":
            self._table_pending_line(line)
        elif st == "quote":
            self._quote_line(line)
        elif st == "list":
            self._list_line(line)
        elif st == "para":
            self._para_line(line)
        else:
            self._dispatch(line)

    def _dispatch(self, line: str, no_table: bool = False) -> None:
        s = line.lstrip(" ")
        ind = len(line) - len(s)
        if not s.strip():
            self._blank_pending = True
            return
        if ind >= 4 or line.startswith("\t"):
            self._icode_line(line)
            return
        if s[0] == "#":
            m = _ATX_RE.match(s)
            if m:
                self._heading(len(m.group(1)), (m.group(2) or "").strip())
                return
        if s[0] in "`~":
            m = _FENCE_RE.match(s)
            if m and not (s[0] == "`" and "`" in m.group(2)):
                self._open_fence(m.group(1), m.group(2).strip())
                return
        if s[0] in "*-_" and _HR_RE.match(s):
            self._hr()
            return
        if s[0] == ">" and not self.flat:
            self._open_quote()
            self._quote_line(line)
            return
        m = _ITEM_RE.match(s)
        if m and not self.flat:
            self._open_list(ind, m.group(1)[0].isdigit())
            self._new_item(ind, m)
            return
        if s[0] == "[":
            m = _LINKDEF_RE.match(s)
            if m:
                self._linkdef(m.group(1), m.group(2), m.group(3) or "")
                return
        if s[0] == "|" and not no_table:
            self._pending_row = line
            self._state = "table_pending"
            return
        self._open_para()
        self._flow.add(line)
        self._flow_emit(False)

    def _partial(self, p: str) -> None:
        """Render as much of an unfinished line as is already unambiguous."""
        st = self._state
        if st == "para":
            if _partial_ok(p, False):
                self._flow.partial(p)
                self._flow_emit(False)
        elif st is None:
            if _partial_ok(p, True):
                self._open_para()
                self._flow.partial(p)
                self._flow_emit(False)
        elif st == "quote":
            s = p.lstrip(" ")
            if s.startswith(">"):
                t = s[1:]
                self._pump(self._sub.push(t[1:] if t.startswith(" ") else t, False))
            elif s and _partial_ok(p, False):
                self._pump(self._sub.push(s, False))
        elif st == "list":
            lst = self._list
            s = p.lstrip(" ")
            ind = len(p) - len(s)
            # Mirror `_list_line` exactly: only *spaces* count toward the content
            # indent, so a tab-led line stays text here and becomes an indented
            # code block there, instead of the two paths disagreeing.
            if ind >= lst.content_indent:
                self._flush_blanks()
                self._pump(self._sub.push(p[lst.content_indent:], False))
            elif s and lst.blanks == 0 and ind < lst.content_indent and _partial_ok(p, False):
                self._pump(self._sub.push(s, False))

    def _close_blocks(self) -> None:
        st = self._state
        if st == "para":
            self._flush_para()
        elif st == "fence":
            self._close_fence()
        elif st == "code":
            self._close_icode()
        elif st == "table":
            self._flush_table()
        elif st == "table_pending":
            row, self._pending_row, self._state = self._pending_row, None, None
            self._dispatch(row, no_table=True)
            self._close_blocks()
        elif st == "quote":
            self._close_quote()
        elif st == "list":
            self._close_list()

    # ---------------------------------------------------------------- headings

    def _heading(self, level: int, text: str) -> None:
        self._sep("heading")
        token = "md.h1" if level == 1 else "md.h2" if level == 2 else "md.h3"
        base = self.theme[token]
        if level >= 4:
            base = base + Style(dim=True)
        body = _render_nodes(_parse_inline(text)[0], self.theme, self.caps, base, False,
                             self.g, markers=False)
        w = 0
        if text.strip():
            for ln in wrap(body, self.inner):
                ln = _tidy(ln)
                self._emit(ln)
                w = max(w, display_width(ln))
        if level <= 2:
            ch = self.g.h1 if level == 1 else self.g.h2
            w = min(self.inner, max(3, w))
            self._emit(self._rule_style(ch * w, "md.rule"))

    def _linkdef(self, label: str, dest: str, title: str) -> None:
        """One reference-link definition, on a line of its own."""
        self._sep("linkdef")
        base = self.theme[self.text_token]
        nodes = [("text", "[" + label + "] "),
                 ("link", [("text", dest)], dest, False)]
        body = _render_nodes(nodes, self.theme, self.caps, base, True, self.g)
        if title:
            body += self.theme.render(" " + title, "app.muted", self.caps)
        for ln in wrap(body, self.inner):
            self._emit(_tidy(ln))
        self._state = None
        self._prev_kind = "linkdef"

    def _hr(self) -> None:
        self._sep("rule")
        self._emit(self._rule_style(self.g.rule * self.inner, "md.rule"))

    # --------------------------------------------------------------- paragraph

    def _open_para(self, token: str = None) -> None:
        self._state = "para"
        self._flow = _Flow(token or self.text_token, self.inner)

    def _para_line(self, line: str) -> None:
        if not line.strip():
            self._flush_para()
            self._blank_pending = True
            return
        if _SETEXT_RE.match(line) and self._setext_ready():
            text = self._flow.open_text(True)[0]
            self._flow = None
            self._state = None
            self._heading(1, text)
            return
        if self._is_start(line):
            self._flush_para()
            self._dispatch(line)
            return
        self._flow.add(line)
        self._flow_emit(False)

    def _setext_ready(self) -> bool:
        """Can the paragraph so far still become an underlined heading?

        Only while nothing of it has been printed. Once a line is out it is out —
        promoting the paragraph afterwards would mean reprinting it, which is the
        one thing this renderer never does. `flow.opened` is set the first time a
        line is emitted and, by the append-only invariant, its value at a given
        line boundary does not depend on how the input was chunked.
        """
        f = self._flow
        return (f is not None and not f.opened and not f.segs
                and bool(f.open_text(True)[0].strip()))

    def _flush_para(self) -> None:
        self._flow_emit(True)
        self._flow = None
        self._state = None
        self._prev_kind = "para"

    def _wrapped(self, text: str, flow: _Flow) -> list:
        return [_tidy(x) for x in wrap(self._inline(text, flow.token), flow.width)]

    def _flow_lines(self, flow: _Flow, final: bool) -> tuple:
        """``(every line this flow can print yet, how many came from closed segments)``."""
        out = []
        for seg in flow.segs:
            out.extend(self._wrapped(seg, flow))
        nclosed = len(out)
        text, _ = flow.open_text(final)
        if not final:
            text = text[:_stable_cut(text)].rstrip()
            if not text:
                return out, nclosed
        elif not text.strip() and flow.split:
            return out, nclosed
        lines = self._wrapped(text, flow)
        if not final:
            lines = lines[:-1]      # the last line can still grow another word
        out.extend(lines)
        return out, nclosed

    def _flow_emit(self, final: bool) -> None:
        flow = self._flow
        if flow is None:
            return
        lines, nclosed = self._flow_lines(flow, final)
        fresh = len(lines) > flow.emitted
        for ln in lines[flow.emitted:]:
            if not flow.opened:
                flow.opened = True
                self._sep("para")
            self._emit(ln)
        flow.emitted = max(flow.emitted, len(lines))
        if nclosed:
            # A closed segment is complete, so its lines have all been printed.
            flow.segs = []
            flow.emitted -= nclosed
        if fresh and not final and flow.emitted:
            self._checkpoint(flow, lines[nclosed:])

    def _checkpoint(self, flow: _Flow, done: list) -> None:
        """Drop the source of lines that have already been printed.

        The cut must be exactly a committed line boundary *and* a top-level
        space — nothing may span it, or re-rendering the tail on its own would
        not give back the same lines. That is checked rather than assumed: a cut
        that does not reproduce `done` exactly is abandoned, and the paragraph
        simply stays whole until the next line offers a better one.
        """
        k = len(done)
        text, off = flow.open_text(False)
        cut, breaks = _cut_breaks(text)
        cand = [b for b in breaks if 0 < b < cut and text[b] == " "]
        if not cand:
            return
        # Line count grows monotonically with the prefix, so the boundary of the
        # last committed line is the largest prefix that still wraps to `k` lines.
        lo, hi, best = 0, len(cand) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if len(self._wrapped(text[:cand[mid]].rstrip(), flow)) <= k:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best < 0:
            return
        b = cand[best]
        if not _same_lines(self._wrapped(text[:b].rstrip(), flow), done):
            return
        rest = text[b + 1:]
        rest = rest[:_stable_cut(rest)].rstrip()
        if rest and len(self._wrapped(rest, flow)) > 1:
            return                      # the tail alone would commit more lines
        flow.emitted = 0
        bp = b - off
        if bp < -1:
            flow.text = flow.text[b + 1:]
            return
        shift = (len(flow.cur) - len(flow.piece(False))) + bp + 1
        flow.base += shift
        flow.cur = flow.cur[shift:]
        flow.text = ""

    # -------------------------------------------------------------- code blocks

    def _box_top(self, label: str) -> str:
        g, w = self.g, self.inner
        if w < 3:                       # no room for two corners and a side
            return ""
        if not label:
            return self._rule_style(g.tl + g.h * (w - 2) + g.tr, "md.code_border")
        lab = " " + label + " "
        room = w - 3 - display_width(lab)
        if room < 1:
            return self._rule_style(g.tl + g.h * (w - 2) + g.tr, "md.code_border")
        return (self._rule_style(g.tl + g.h, "md.code_border")
                + self.theme.render(lab, "app.muted", self.caps)
                + self._rule_style(g.h * room + g.tr, "md.code_border"))

    def _box_bottom(self) -> str:
        g = self.g
        if self.inner < 3:
            return ""
        return self._rule_style(g.bl + g.h * (self.inner - 2) + g.br, "md.code_border")

    def _code_line(self, text: str) -> str:
        """One code row: highlighted, clipped to the box, painted with the code bg."""
        hl = self._fence["hl"] if self._fence else None
        styled = hl.feed_line(text) if hl else self.theme.render(text, "syn.plain", self.caps)
        g = self.g
        room = self.inner - 4
        if self.inner < 3:              # the gutter itself would not fit
            return truncate(styled, self.inner, g.ell)
        if room < 4:
            return self._rule_style(g.v, "md.code_border") + " " + truncate(
                styled, max(1, self.inner - 2), g.ell)
        body = " " + pad(truncate(styled, room, g.ell), room) + " "
        if self.caps.color >= 8:
            codes = self.theme["md.code_bg"].codes(self.caps)
            if codes:
                body = codes + body.replace(RESET, RESET + codes) + RESET
        bar = self._rule_style(g.v, "md.code_border")
        return bar + body + bar

    def _open_fence(self, marker: str, info: str) -> None:
        self._sep("code")
        label = info.split()[0].strip("{}.") if info.split() else ""
        lang = _resolve_lang(label)
        self._fence = {
            "char": marker[0],
            "len": len(marker),
            "hl": _Highlighter(lang, self.theme, self.caps) if lang else None,
        }
        self._state = "fence"
        self._emit(self._box_top(label))

    def _fence_line(self, line: str) -> None:
        s = line.lstrip(" ")
        if len(line) - len(s) < 4 and s and s[0] == self._fence["char"]:
            j = _find_run(s, 0, s[0])
            if j >= self._fence["len"] and not s[j:].strip():
                self._close_fence()
                return
        self._emit(self._code_line(line.expandtabs(4)))

    def _close_fence(self) -> None:
        self._emit(self._box_bottom())
        self._fence = None
        self._state = None
        self._prev_kind = "code"

    def _icode_line(self, line: str) -> None:
        """Indented (four-space) code block — same box, no language label."""
        if not line.strip():
            if self._icode_open:
                self._icode_blanks += 1
            return
        if not (line.startswith("    ") or line.startswith("\t")):
            self._close_icode()
            self._blank_pending = True
            self._dispatch(line)
            return
        if not self._icode_open:
            self._sep("code")
            self._state = "code"
            self._icode_open = True
            self._fence = {"char": "", "len": 0, "hl": None}
            self._emit(self._box_top(""))
        for _ in range(self._icode_blanks):
            self._emit(self._code_line(""))
        self._icode_blanks = 0
        text = line[1:] if line.startswith("\t") else line[4:]
        self._emit(self._code_line(text.expandtabs(4)))

    def _close_icode(self) -> None:
        if self._icode_open:
            self._emit(self._box_bottom())
        self._icode_open = False
        self._icode_blanks = 0
        self._fence = None
        self._state = None
        self._prev_kind = "code"

    # -------------------------------------------------------------------- table

    def _table_pending_line(self, line: str) -> None:
        header = self._pending_row
        cells = _split_cells(header)
        if cells and _is_delimiter_row(line, len(cells)):
            self._pending_row = None
            self._state = "table"
            self._table = {"rows": [cells], "aligns": _alignments(line, len(cells))}
            return
        self._pending_row = None
        self._state = None
        self._dispatch(header, no_table=True)
        self._line(line)

    def _table_line(self, line: str) -> None:
        if not line.strip() or "|" not in line:
            self._flush_table()
            self._line(line)
            return
        n = len(self._table["rows"][0])
        cells = _split_cells(line)
        self._table["rows"].append((cells + [""] * n)[:n])

    def _flush_table(self) -> None:
        t, self._table, self._state = self._table, None, None
        if not t:
            return
        self._sep("table")
        rows, aligns = t["rows"], t["aligns"]
        n = len(rows[0])
        g = self.g
        cells = []
        for r, row in enumerate(rows):
            token = "md.table_head" if r == 0 else "app.text"
            base = self.theme[token]
            cells.append([
                _render_nodes(_parse_inline(c)[0], self.theme, self.caps, base, r > 0, self.g)
                for c in row
            ])
        widths = [max(1, max(display_width(row[i]) for row in cells)) for i in range(n)]
        avail = self.inner - (3 * n + 1)
        if avail < n:                       # far too narrow for a real table
            for row in cells:
                self._emit(truncate(("  " + self._rule_style(g.v, "md.table_border") + " ")
                                    .join(row), self.inner, g.ell))
            self._prev_kind = "table"
            return
        guard = 0
        while sum(widths) > avail and guard < 10000:
            guard += 1
            k = widths.index(max(widths))
            if widths[k] <= 3:
                break
            widths[k] -= 1
        while sum(widths) > avail:          # every column is already minimal
            k = widths.index(max(widths))
            widths[k] -= 1

        bar = self._rule_style(g.v, "md.table_border")
        line_of = lambda l, m, r: self._rule_style(
            l + m.join(g.h * (w + 2) for w in widths) + r, "md.table_border")

        self._emit(line_of(g.tl, g.tt, g.tr))
        for r, row in enumerate(cells):
            cols = [[_tidy(x) for x in wrap(c, widths[i])] for i, c in enumerate(row)]
            height = max(len(c) for c in cols)
            for k in range(height):
                parts = []
                for i in range(n):
                    piece = cols[i][k] if k < len(cols[i]) else ""
                    parts.append(" " + pad(piece, widths[i], aligns[i]) + " ")
                self._emit(bar + bar.join(parts) + bar)
            if r == 0:
                self._emit(line_of(g.lt, g.cross, g.rt))
        self._emit(line_of(g.bl, g.bt, g.br))
        self._prev_kind = "table"

    # --------------------------------------------------------------- containers

    def _can_nest(self, cost: int) -> bool:
        """Is there room — in columns and in stack — for one more container?

        `cost` is the width of the prefix the parent will print to the left of
        every child line. The child's width is never clamped back up, so this is
        the one place that decides; past it, deeper levels render flat.
        """
        return (self.cdepth < _MAX_CONTAINER_DEPTH
                and self.inner - cost >= _MIN_CONTENT)

    def _is_start(self, line: str) -> bool:
        """`_is_block_start`, minus the containers this renderer will not open."""
        if not _is_block_start(line):
            return False
        if not self.flat:
            return True
        s = line.lstrip(" ")
        if not s.strip():
            return True
        if s[0] == ">":
            return False
        if s[0] in "*-_" and _HR_RE.match(s):
            return True
        m = _ITEM_RE.match(s)
        return not (m and m.group(3).strip())

    def _child(self, width: int, token: str = None, deeper: bool = False) -> "MarkdownStream":
        # `deeper` only tracks *list* nesting, which is what picks the bullet glyph:
        # a quote inside a quote must not shift its bullets.
        return MarkdownStream(self.theme, self.caps, width=max(1, width), indent="",
                              _depth=self.depth + (1 if deeper else 0),
                              _text_token=token or self.text_token,
                              _cdepth=self.cdepth + 1)

    def _pump(self, text: str) -> None:
        if not text:
            return
        for line in text.split("\n")[:-1]:
            self._emit_sub_line(line)

    def _emit_sub_line(self, line: str) -> None:
        sub = self._sub
        blank = not strip_ansi(line).strip()
        if sub.kind == "quote":
            self._emit(sub.first if blank else sub.first + " " + line)
            return
        if blank:
            self._emit("")
            return
        self._emit((sub.first if not sub.used else sub.rest) + line)
        sub.used = True

    # quotes -------------------------------------------------------------------

    def _open_quote(self) -> None:
        self._sep("quote")
        bar = self._rule_style(self.g.bar, "md.quote_bar")
        self._state = "quote"
        self._sub = _Sub(self._child(self.inner - 2, "md.quote"), "quote", bar, bar)

    def _quote_line(self, line: str) -> None:
        s = line.lstrip(" ")
        if s.startswith(">"):
            t = s[1:]
            self._pump(self._sub.push(t[1:] if t.startswith(" ") else t, True))
            return
        if not line.strip():
            self._close_quote()
            self._blank_pending = True
            return
        if not self._is_start(line):
            self._pump(self._sub.push(s, True))     # lazy continuation
            return
        self._close_quote()
        self._dispatch(line)

    def _close_quote(self) -> None:
        self._pump(self._sub.child.close())
        self._sub = None
        self._state = None
        self._prev_kind = "quote"

    # lists --------------------------------------------------------------------

    def _open_list(self, indent: int, ordered: bool) -> None:
        self._sep("list")
        self._state = "list"
        self._list = _List(ordered=ordered, indent=indent)

    def _marker(self, ordered: bool, number: int, task: str) -> tuple:
        """Return (styled marker padded to its column, width)."""
        if task is not None:
            done = task.lower() == "x"
            text = "[" + (self.g.check if done else " ") + "]"
            styled = self.theme.render(text, "md.task_done" if done else "md.task_todo", self.caps)
            return styled + " ", display_width(text) + 1
        if ordered:
            text = f"{number}."
            return self.theme.render(text, "md.number", self.caps) + " ", len(text) + 1
        glyph = self.g.bullets[self.depth % len(self.g.bullets)]
        return self.theme.render(glyph, "md.bullet", self.caps) + " ", display_width(glyph) + 1

    def _new_item(self, indent: int, m) -> None:
        lst = self._list
        if lst.blanks:
            if not self._last_blank and self._emitted_any:
                self._emit()
            lst.blanks = 0
        marker, gap, rest = m.group(1), m.group(2), m.group(3)
        gapn = len(gap.expandtabs(4)) if gap else 0
        if gapn == 0 or gapn > 4 or not rest.strip():
            gapn = 1
        lst.content_indent = indent + len(marker) + gapn
        task = None
        tm = _TASK_RE.match(rest)
        if tm and not lst.ordered:
            task = tm.group(1)
            rest = tm.group(2)
        number = int(marker[:-1]) if lst.ordered else 0
        text, mw = self._marker(lst.ordered, number, task)
        room = max(1, self.inner - 1)
        if mw > room:                   # a nine-digit marker in a narrow column
            text = truncate(text, room, self.g.ell)
            mw = display_width(strip_ansi(text))
        lst.marker_width = mw
        lst.number = number + 1
        self._sub = _Sub(self._child(self.inner - mw, deeper=True), "item", text, " " * mw)
        self._pump(self._sub.push(rest, True))

    def _end_item(self) -> None:
        if self._sub is None:
            return
        self._pump(self._sub.child.close())
        if not self._sub.used:
            self._emit(self._sub.first.rstrip())
        self._sub = None

    def _flush_blanks(self) -> None:
        while self._list.blanks > 0:
            self._list.blanks -= 1
            self._pump(self._sub.push("", True))

    def _list_line(self, line: str) -> None:
        lst = self._list
        s = line.lstrip(" ")
        ind = len(line) - len(s)
        if not s.strip():
            lst.blanks += 1
            return
        if ind >= lst.content_indent:
            self._flush_blanks()
            self._pump(self._sub.push(line[lst.content_indent:], True))
            return
        m = _ITEM_RE.match(s)
        if m and ind <= lst.indent + 3 and (m.group(1)[0].isdigit()) == lst.ordered:
            self._end_item()
            self._new_item(ind, m)
            return
        if lst.blanks == 0 and not self._is_start(line):
            self._pump(self._sub.push(s, True))     # lazy continuation
            return
        self._close_list()
        self._dispatch(line)

    def _close_list(self) -> None:
        blanks = self._list.blanks if self._list else 0
        self._end_item()
        self._list = None
        self._state = None
        self._prev_kind = "list"
        if blanks:
            self._blank_pending = True


def render_markdown(text: str, theme: Theme, caps: Caps, width: int = None,
                    indent: str = "") -> str:
    """Render a whole Markdown document in one go."""
    stream = MarkdownStream(theme, caps, width=width, indent=indent)
    return stream.feed(text) + stream.close()


# ------------------------------------------------------------------ highlighting

_TOKEN_STYLE = {
    "kw": "syn.keyword", "str": "syn.string", "num": "syn.number",
    "com": "syn.comment", "fn": "syn.func", "bi": "syn.builtin",
    "op": "syn.operator", "pun": "syn.punct", "ty": "syn.type",
    # `syn.variable` is the same colour as `syn.plain` in every theme, so a
    # `$HOME` split out of a shell string would render as unstyled code and make
    # the string look like it had ended mid-token. `syn.decorator` is distinct
    # from both `syn.plain` and `syn.string` at 24-, 8- and 4-bit colour.
    "dec": "syn.decorator", "var": "syn.decorator", "txt": "syn.plain",
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(
    r"0[xXbBoO][0-9a-fA-F_]+[uUlLfF]*"
    r"|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+\-]?\d+)?[a-zA-Z_]*"
)
_OPS = set("+-*/%=<>!&|^~?")
_PUNCT_CH = set("()[]{},;.:")


@dataclass(frozen=True)
class _Lang:
    """Everything the generic tokenizer needs to know about one language."""

    name: str
    keywords: frozenset = frozenset()
    builtins: frozenset = frozenset()
    types: frozenset = frozenset()
    line_comments: tuple = ()
    block_comments: tuple = ()          # ((open, close), ...)
    strings: tuple = ()                 # ((open, close, escapes, multiline), ...)
    decorator: str = ""
    var_prefix: str = ""
    interp: str = ""                    # string openers that interpolate variables
    ci: bool = False                    # case-insensitive keywords (SQL)
    upper_type: bool = False            # Capitalised identifiers are types
    comment_boundary: bool = False      # '#' only starts a comment after a space
    string_prefixes: str = ""           # f"" / b'' / r"" style prefixes
    lifetime: bool = False              # 'a is a lifetime, not a char literal
    raw_hash: bool = False              # r#"..."# raw strings
    def_kw: tuple = ()                  # next identifier is a function name
    type_kw: tuple = ()                 # next identifier is a type name
    custom: str = ""


def _words(s: str) -> frozenset:
    return frozenset(s.split())


def _scan_to(s: str, i: int, closer: str, esc: bool) -> int:
    n = len(s)
    while i < n:
        if esc and s[i] == "\\":
            i += 2
            continue
        if s.startswith(closer, i):
            return i
        i += 1
    return -1


def _push_string(out, text: str, L: _Lang, interp: bool) -> None:
    """Append a string token, splitting out any variables it interpolates.

    `"$HOME"` in a shell is a string *and* a variable reference; painting the
    whole thing as one string hides the substitution that actually happens.
    """
    if not (interp and L.var_prefix and L.var_prefix in text):
        if text:
            out.append(("str", text))
        return
    i = start = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == L.var_prefix and i + 1 < n:
            j = i + 1
            if text[j] == "{":
                k = text.find("}", j)
                j = n if k < 0 else k + 1
            elif text[j].isalnum() or text[j] == "_":
                m = _IDENT_RE.match(text, j)
                j = m.end() if m else j + 1
            elif text[j] in "?@*#$!-":
                j += 1                          # $?, $@, $# and friends
            else:
                i += 1
                continue
            if start < i:
                out.append(("str", text[start:i]))
            out.append(("var", text[i:j]))
            i = start = j
            continue
        i += 1
    if start < n:
        out.append(("str", text[start:]))


def _tok_generic(L: _Lang, line: str, state):
    out = []
    i, n = 0, len(line)
    last_kw = ""
    if state:
        kind, closer, esc = state
        interp = kind == "str" and closer in L.interp
        j = _scan_to(line, 0, closer, esc)
        if j < 0:
            if line:
                if kind == "str":
                    _push_string(out, line, L, interp)
                else:
                    out.append((kind, line))
            return out, state
        if kind == "str":
            _push_string(out, line[:j + len(closer)], L, interp)
        else:
            out.append((kind, line[:j + len(closer)]))
        i = j + len(closer)
    while i < n:
        c = line[i]
        if c in " \t":
            j = i
            while j < n and line[j] in " \t":
                j += 1
            out.append(("txt", line[i:j]))
            i = j
            continue
        hit = False
        for lc in L.line_comments:
            if line.startswith(lc, i):
                if L.comment_boundary and lc == "#" and i > 0 and line[i - 1] not in " \t;|&(":
                    continue
                out.append(("com", line[i:]))
                return out, None
        for op, cl in L.block_comments:
            if line.startswith(op, i):
                j = _scan_to(line, i + len(op), cl, False)
                if j < 0:
                    out.append(("com", line[i:]))
                    return out, ("com", cl, False)
                out.append(("com", line[i:j + len(cl)]))
                i = j + len(cl)
                hit = True
                break
        if hit:
            continue
        if L.lifetime and c == "'":
            # `'a` is a lifetime; `'a'` is a char literal. The character after
            # the identifier is the whole difference.
            m = _IDENT_RE.match(line, i + 1)
            if m and (m.end() >= n or line[m.end()] != "'"):
                out.append(("ty", line[i:m.end()]))
                i = m.end()
                continue
        if L.raw_hash and c == "r":
            k = i + 1
            while k < n and line[k] == "#":
                k += 1
            if k > i + 1 and k < n and line[k] == '"':
                cl = '"' + "#" * (k - i - 1)
                j = line.find(cl, k + 1)
                if j < 0:
                    out.append(("str", line[i:]))
                    return out, ("str", cl, False)
                out.append(("str", line[i:j + len(cl)]))
                i = j + len(cl)
                continue
        pre = 0
        if L.string_prefixes:
            k = i
            while k < n and k - i < 2 and line[k] in L.string_prefixes:
                k += 1
            if k < n and line[k] in "\"'":
                pre = k - i
        for op, cl, esc, ml in L.strings:
            if line.startswith(op, i + pre):
                interp = op in L.interp
                j = _scan_to(line, i + pre + len(op), cl, esc)
                if j < 0:
                    _push_string(out, line[i:], L, interp)
                    return out, (("str", cl, esc) if ml else None)
                _push_string(out, line[i:j + len(cl)], L, interp)
                i = j + len(cl)
                hit = True
                break
        if hit:
            continue
        if c.isdigit() or (c == "." and i + 1 < n and line[i + 1].isdigit()):
            m = _NUM_RE.match(line, i)
            if m:
                out.append(("num", m.group(0)))
                i = m.end()
                continue
        if L.var_prefix and c == L.var_prefix:
            j = i + 1
            if j < n and line[j] == "{":
                j = line.find("}", j)
                j = n if j < 0 else j + 1
            elif j < n and (line[j].isalnum() or line[j] == "_"):
                m = _IDENT_RE.match(line, j)
                j = m.end() if m else j + 1
            elif j < n:
                j += 1
            out.append(("var", line[i:j]))
            i = j
            continue
        if L.decorator and c == L.decorator and i + 1 < n and (line[i + 1].isalpha() or line[i + 1] == "_"):
            j = i + 1
            while j < n and (line[j].isalnum() or line[j] in "_."):
                j += 1
            out.append(("dec", line[i:j]))
            i = j
            continue
        m = _IDENT_RE.match(line, i)
        if m:
            word = m.group(0)
            key = word.lower() if L.ci else word
            if last_kw in L.def_kw:
                kind = "fn"
            elif last_kw in L.type_kw:
                kind = "ty"
            elif key in L.keywords:
                kind = "kw"
            elif key in L.types:
                kind = "ty"
            elif key in L.builtins:
                kind = "bi"
            else:
                j = m.end()
                while j < n and line[j] == " ":
                    j += 1
                if j < n and line[j] in "(<" and line[j] == "(":
                    kind = "fn"
                elif L.upper_type and word[0].isupper():
                    kind = "ty"
                else:
                    kind = "txt"
            last_kw = key if kind == "kw" else ""
            out.append((kind, word))
            i = m.end()
            continue
        if c in _OPS:
            j = i
            while j < n and line[j] in _OPS:
                j += 1
            out.append(("op", line[i:j]))
            i = j
            continue
        if c in _PUNCT_CH:
            out.append(("pun", c))
            i += 1
            continue
        out.append(("txt", c))
        i += 1
    return out, None


def _tok_json(L, line, state):
    out = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            j = i
            while j < n and line[j] in " \t":
                j += 1
            out.append(("txt", line[i:j]))
            i = j
        elif c == '"':
            j = _scan_to(line, i + 1, '"', True)
            end = n if j < 0 else j + 1
            k = end
            while k < n and line[k] == " ":
                k += 1
            out.append(("ty" if k < n and line[k] == ":" else "str", line[i:end]))
            i = end
        elif c.isdigit() or (c == "-" and i + 1 < n and line[i + 1].isdigit()):
            m = _NUM_RE.match(line, i + 1 if c == "-" else i)
            end = m.end() if m else i + 1
            out.append(("num", line[i:end]))
            i = end
        elif line.startswith("true", i) or line.startswith("null", i):
            out.append(("bi", line[i:i + 4]))
            i += 4
        elif line.startswith("false", i):
            out.append(("bi", line[i:i + 5]))
            i += 5
        elif c in "{}[],:":
            out.append(("pun", c))
            i += 1
        else:
            out.append(("txt", c))
            i += 1
    return out, None


_YAML_KEY_RE = re.compile(r"([ \t]*)(-[ \t]+)?([\"']?[\w.\-/ ]+[\"']?)(:)(?=[ \t]|$)")


def _tok_yaml(L, line, state):
    out = []
    stripped = line.strip()
    if stripped.startswith("#"):
        return [("com", line)], None
    if stripped in ("---", "..."):
        return [("pun", line)], None
    i = 0
    m = _YAML_KEY_RE.match(line)
    if m:
        out.append(("txt", m.group(1)))
        if m.group(2):
            out.append(("op", m.group(2)))
        out.append(("ty", m.group(3)))
        out.append(("pun", ":"))
        i = m.end()
    n = len(line)
    while i < n:
        c = line[i]
        if c == "#" and (i == 0 or line[i - 1] in " \t"):
            out.append(("com", line[i:]))
            return out, None
        if c in "\"'":
            j = _scan_to(line, i + 1, c, True)
            end = n if j < 0 else j + 1
            out.append(("str", line[i:end]))
            i = end
            continue
        if c in "&*":
            m2 = _IDENT_RE.match(line, i + 1)
            if m2:
                out.append(("dec", line[i:m2.end()]))
                i = m2.end()
                continue
        if c.isdigit() and (i == 0 or not line[i - 1].isalnum()):
            m2 = _NUM_RE.match(line, i)
            if m2:
                out.append(("num", m2.group(0)))
                i = m2.end()
                continue
        m2 = _IDENT_RE.match(line, i)
        if m2:
            word = m2.group(0)
            kind = "bi" if word.lower() in ("true", "false", "null", "yes", "no", "on", "off") else "txt"
            out.append((kind, word))
            i = m2.end()
            continue
        if c in "-[]{},":
            out.append(("op" if c == "-" else "pun", c))
            i += 1
            continue
        out.append(("txt", c))
        i += 1
    return out, None


_TOML_KEY_RE = re.compile(r"([ \t]*)([\w.\-\"']+)([ \t]*)(=)")


def _tok_toml(L, line, state):
    stripped = line.strip()
    if stripped.startswith("#"):
        return [("com", line)], None
    if stripped.startswith("["):
        return [("ty", line)], None
    out = []
    i = 0
    m = _TOML_KEY_RE.match(line)
    if m:
        out.append(("txt", m.group(1)))
        out.append(("fn", m.group(2)))
        out.append(("txt", m.group(3)))
        out.append(("op", "="))
        i = m.end()
    rest, st = _tok_generic(_LANGS["_toml_value"], line[i:], state)
    out.extend(rest)
    return out, st


def _tok_diff(L, line, state):
    # Inside a hunk every line is content, so a diff *of a diff* — where an added
    # line reads `+++ b/x` — must not be painted as a file header.
    in_hunk = state == ("hunk",)
    if line.startswith(("diff ", "index ", "old mode", "new mode",
                        "similarity index", "rename ")):
        return [("dec", line)], None
    if not in_hunk and line.startswith(("--- ", "+++ ")):
        return [("dec", line)], None
    if line.startswith("@@"):
        return [("kw", line)], ("hunk",)
    if line.startswith("+"):
        return [("str", line)], state
    if line.startswith("-"):
        return [("ty", line)], state
    if line.startswith("\\"):
        return [("com", line)], state
    return [("txt", line)], state


_HTML_TAG_RE = re.compile(r"</?([A-Za-z][\w:-]*)")
_HTML_ATTR_RE = re.compile(r"([A-Za-z_:][\w:.\-]*)")


def _tok_html(L, line, state):
    out = []
    i, n = 0, len(line)
    in_tag = state == ("html_tag",)
    if state == ("html_comment",):
        j = line.find("-->")
        if j < 0:
            return [("com", line)], state
        out.append(("com", line[:j + 3]))
        i = j + 3
        state = None
    while i < n:
        c = line[i]
        if not in_tag:
            if line.startswith("<!--", i):
                j = line.find("-->", i)
                if j < 0:
                    out.append(("com", line[i:]))
                    return out, ("html_comment",)
                out.append(("com", line[i:j + 3]))
                i = j + 3
                continue
            m = _HTML_TAG_RE.match(line, i)
            if m:
                out.append(("pun", line[i:m.start(1)]))
                out.append(("kw", m.group(1)))
                i = m.end()
                in_tag = True
                continue
            if c == "&":
                j = line.find(";", i)
                if 0 < j - i < 12:
                    out.append(("bi", line[i:j + 1]))
                    i = j + 1
                    continue
            j = i + 1                 # a '<' or '&' that starts nothing is text
            while j < n and line[j] not in "<&":
                j += 1
            out.append(("txt", line[i:j]))
            i = j
            continue
        if c in "\"'":
            j = _scan_to(line, i + 1, c, False)
            end = n if j < 0 else j + 1
            out.append(("str", line[i:end]))
            i = end
            continue
        if c == ">" or line.startswith("/>", i):
            k = i + (2 if line.startswith("/>", i) else 1)
            out.append(("pun", line[i:k]))
            i = k
            in_tag = False
            continue
        if c == "=":
            out.append(("op", c))
            i += 1
            continue
        m = _HTML_ATTR_RE.match(line, i)
        if m:
            out.append(("fn", m.group(0)))
            i = m.end()
            continue
        out.append(("txt", c))
        i += 1
    return out, (("html_tag",) if in_tag else None)


_CSS_PROP_RE = re.compile(r"[-\w]+")


def _tok_css(L, line, state):
    out = []
    in_block = bool(state and state[0] == "css_block")
    if state and state[0] == "css_comment":
        j = line.find("*/")
        if j < 0:
            return [("com", line)], state
        out.append(("com", line[:j + 2]))
        line_start = j + 2
        state = ("css_block",) if state[1] else None
        in_block = bool(state)
    else:
        line_start = 0
    i, n = line_start, len(line)
    while i < n:
        c = line[i]
        if line.startswith("/*", i):
            j = line.find("*/", i + 2)
            if j < 0:
                out.append(("com", line[i:]))
                return out, ("css_comment", in_block)
            out.append(("com", line[i:j + 2]))
            i = j + 2
            continue
        if c in " \t":
            j = i
            while j < n and line[j] in " \t":
                j += 1
            out.append(("txt", line[i:j]))
            i = j
            continue
        if c in "\"'":
            j = _scan_to(line, i + 1, c, True)
            end = n if j < 0 else j + 1
            out.append(("str", line[i:end]))
            i = end
            continue
        if c == "{":
            in_block = True
            out.append(("pun", c))
            i += 1
            continue
        if c == "}":
            in_block = False
            out.append(("pun", c))
            i += 1
            continue
        if c == "@":
            m = _CSS_PROP_RE.match(line, i + 1)
            out.append(("dec", line[i:m.end() if m else i + 1]))
            i = m.end() if m else i + 1
            continue
        if c == "#" and not in_block:
            m = _CSS_PROP_RE.match(line, i + 1)
            out.append(("ty", line[i:m.end() if m else i + 1]))
            i = m.end() if m else i + 1
            continue
        if c == "#" or c.isdigit() or (c == "." and i + 1 < n and line[i + 1].isdigit()):
            m = re.compile(r"#[0-9a-fA-F]{3,8}|[\d.]+[a-z%]*").match(line, i)
            if m:
                out.append(("num", m.group(0)))
                i = m.end()
                continue
        m = _CSS_PROP_RE.match(line, i)
        if m and (m.group(0)[0].isalpha() or m.group(0)[0] == "-"):
            word = m.group(0)
            j = m.end()
            k = j
            while k < n and line[k] == " ":
                k += 1
            if in_block and k < n and line[k] == ":":
                kind = "fn"
            elif in_block and k < n and line[k] == "(":
                kind = "bi"
            elif in_block:
                kind = "txt"
            else:
                kind = "ty"
            out.append((kind, word))
            i = j
            continue
        out.append(("pun" if c in _PUNCT_CH else "op" if c in _OPS else "txt", c))
        i += 1
    return out, (("css_block",) if in_block else None)


_MD_HEAD_RE = re.compile(r"[ ]{0,3}#{1,6}[ \t]")


def _tok_markdown(L, line, state):
    if state == ("md_fence",):
        if line.strip().startswith(("```", "~~~")):
            return [("com", line)], None
        return [("txt", line)], state
    if line.strip().startswith(("```", "~~~")):
        return [("com", line)], ("md_fence",)
    if _MD_HEAD_RE.match(line):
        return [("kw", line)], None
    if line.lstrip().startswith(">"):
        return [("com", line)], None
    out = []
    i, n = 0, len(line)
    m = re.match(r"([ \t]*)([-*+]|\d+[.)])([ \t]+)", line)
    if m:
        out.append(("txt", m.group(1)))
        out.append(("op", m.group(2)))
        out.append(("txt", m.group(3)))
        i = m.end()
    while i < n:
        c = line[i]
        if c == "`":
            j = line.find("`", i + 1)
            end = n if j < 0 else j + 1
            out.append(("str", line[i:end]))
            i = end
            continue
        if c in "*_~" and i + 1 < n:
            j = i
            while j < n and line[j] == c:
                j += 1
            out.append(("kw", line[i:j]))
            i = j
            continue
        if c == "[":
            j = line.find("]", i)
            end = n if j < 0 else j + 1
            out.append(("fn", line[i:end]))
            i = end
            continue
        if c == "(" and out and out[-1][1].endswith("]"):
            j = line.find(")", i)
            end = n if j < 0 else j + 1
            out.append(("str", line[i:end]))
            i = end
            continue
        out.append(("txt", c))
        i += 1
    return out, None


_CUSTOM = {
    "json": _tok_json, "yaml": _tok_yaml, "toml": _tok_toml, "diff": _tok_diff,
    "html": _tok_html, "css": _tok_css, "markdown": _tok_markdown,
}


_C_STRINGS = (('"', '"', True, False), ("'", "'", True, False))
_PY_STRINGS = (
    ('"""', '"""', True, True), ("'''", "'''", True, True),
    ('"', '"', True, False), ("'", "'", True, False),
)

_LANGS = {}


def _lang(name: str, **kw) -> _Lang:
    L = _Lang(name=name, **kw)
    _LANGS[name] = L
    return L


_lang(
    "python",
    keywords=_words("""and as assert async await break class continue def del elif else
        except finally for from global if import in is lambda nonlocal not or pass raise
        return try while with yield match case"""),
    builtins=_words("""True False None self cls print len range int str float list dict set
        tuple bool open isinstance issubclass super type object repr hash id dir getattr
        setattr hasattr delattr staticmethod classmethod property format input bytes
        bytearray callable divmod exec eval frozenset globals locals iter next slice vars
        round pow ord chr abs sum min max sorted reversed enumerate zip map filter any all
        Exception ValueError TypeError KeyError IndexError RuntimeError StopIteration
        NotImplementedError OSError __name__ __init__ __main__"""),
    line_comments=("#",),
    strings=_PY_STRINGS,
    decorator="@",
    string_prefixes="rbfuRBFU",
    def_kw=("def",),
    type_kw=("class",),
)

_JS_KW = _words("""var let const function return if else for while do break continue new
    delete typeof instanceof in of class extends super this null undefined true false
    async await yield try catch finally throw switch case default import export from as
    static get set void with debugger enum interface type implements public private
    protected readonly declare namespace abstract satisfies keyof infer require""")
_JS_BUILTINS = _words("""console window document Math JSON Object Array String Number
    Boolean Promise Symbol Map Set WeakMap WeakSet RegExp Date Error module exports process
    globalThis parseInt parseFloat isNaN encodeURIComponent decodeURIComponent setTimeout
    setInterval clearTimeout fetch localStorage""")

_lang(
    "javascript",
    keywords=_JS_KW,
    builtins=_JS_BUILTINS,
    types=_words("number string boolean any unknown never object bigint symbol void"),
    line_comments=("//",),
    block_comments=(("/*", "*/"),),
    strings=(("`", "`", True, True),) + _C_STRINGS,
    decorator="@",
    def_kw=("function",),
    type_kw=("class", "interface", "enum"),
    upper_type=True,
)

_lang(
    "typescript",
    keywords=_JS_KW,
    builtins=_JS_BUILTINS,
    types=_words("number string boolean any unknown never object bigint symbol void Array Record Partial"),
    line_comments=("//",),
    block_comments=(("/*", "*/"),),
    strings=(("`", "`", True, True),) + _C_STRINGS,
    decorator="@",
    def_kw=("function",),
    type_kw=("class", "interface", "enum", "type"),
    upper_type=True,
)

_lang(
    "go",
    keywords=_words("""break case chan const continue default defer else fallthrough for
        func go goto if import interface map package range return select struct switch type
        var nil true false iota"""),
    builtins=_words("""append cap close complex copy delete imag len make new panic print
        println real recover fmt errors context"""),
    types=_words("""bool byte complex64 complex128 error float32 float64 int int8 int16
        int32 int64 rune string uint uint8 uint16 uint32 uint64 uintptr any"""),
    line_comments=("//",),
    block_comments=(("/*", "*/"),),
    strings=(("`", "`", False, True),) + _C_STRINGS,
    def_kw=("func",),
    type_kw=("type", "struct", "interface", "package"),
)

_lang(
    "rust",
    keywords=_words("""as async await break const continue crate dyn else enum extern false
        fn for if impl in let loop match mod move mut pub ref return self Self static struct
        super trait true type unsafe use where while box"""),
    builtins=_words("""println print format vec panic assert assert_eq write writeln Some
        None Ok Err unwrap expect clone iter collect"""),
    types=_words("""i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char
        str String Vec Option Result Box Rc Arc HashMap HashSet"""),
    line_comments=("//",),
    block_comments=(("/*", "*/"),),
    strings=_C_STRINGS,
    decorator="#",
    def_kw=("fn",),
    type_kw=("struct", "enum", "trait", "type", "impl", "mod"),
    string_prefixes="rb",
    lifetime=True,
    raw_hash=True,
)

_C_KW = _words("""alignas alignof auto bool break case catch char class const constexpr
    continue default delete do double else enum explicit export extern false float for
    friend goto if inline int long mutable namespace new noexcept nullptr operator private
    protected public register return short signed sizeof static struct switch template this
    throw true try typedef typename union unsigned using virtual void volatile while
    static_cast dynamic_cast const_cast reinterpret_cast""")

_lang(
    "c",
    keywords=_C_KW,
    builtins=_words("""printf sprintf fprintf scanf malloc calloc realloc free memcpy memset
        strlen strcpy strcmp NULL size_t FILE stdin stdout stderr include define ifdef ifndef
        endif pragma"""),
    types=_words("int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t"),
    line_comments=("//",),
    block_comments=(("/*", "*/"),),
    strings=_C_STRINGS,
    decorator="#",
    def_kw=(),
    type_kw=("struct", "union", "enum", "class", "typename"),
)
_LANGS["cpp"] = _Lang(**{**_LANGS["c"].__dict__, "name": "cpp"})

_lang(
    "java",
    keywords=_words("""abstract assert boolean break byte case catch char class const
        continue default do double else enum extends final finally float for goto if
        implements import instanceof int interface long native new package private protected
        public return short static strictfp super switch synchronized this throw throws
        transient try void volatile while var record sealed permits yield true false null"""),
    builtins=_words("""System String Integer Long Double Float Boolean Object List Map Set
        ArrayList HashMap HashSet Optional Stream Exception RuntimeException Override"""),
    line_comments=("//",),
    block_comments=(("/*", "*/"),),
    strings=_C_STRINGS,
    decorator="@",
    type_kw=("class", "interface", "enum", "record", "extends", "implements"),
    upper_type=True,
)

_lang(
    "bash",
    keywords=_words("""if then else elif fi for while until do done case esac function in
        return break continue local export readonly declare source alias unset shift eval
        exec trap set select time"""),
    builtins=_words("""echo printf cd pwd ls cat grep sed awk find rm mv cp mkdir rmdir touch
        chmod chown curl wget git make sudo test read exit kill ps sleep head tail sort uniq
        wc xargs tar ssh scp docker python python3 pip npm node cargo go java env which
        true false"""),
    line_comments=("#",),
    strings=(('"', '"', True, False), ("'", "'", False, False)),
    var_prefix="$",
    interp='"',
    comment_boundary=True,
    def_kw=("function",),
)

_lang(
    "sql",
    keywords=_words("""select from where insert into values update set delete create table
        drop alter add column primary key foreign references index view join inner left
        right outer full cross on group by order having limit offset union all distinct as
        and or not null is in between like exists case when then else end asc desc cast with
        returning default constraint unique check begin commit rollback transaction using
        grant revoke schema database if temporary procedure function trigger"""),
    builtins=_words("""count sum avg min max coalesce nullif now current_timestamp
        current_date length substring upper lower trim round abs concat"""),
    types=_words("""int integer bigint smallint serial bigserial varchar char text boolean
        bool date timestamp timestamptz time numeric decimal real double float json jsonb
        uuid bytea array"""),
    line_comments=("--",),
    block_comments=(("/*", "*/"),),
    strings=(("'", "'", True, False), ('"', '"', True, False)),
    ci=True,
)

_lang(
    "_toml_value",
    builtins=_words("true false"),
    line_comments=("#",),
    strings=(('"""', '"""', True, True), ("'''", "'''", True, True)) + _C_STRINGS,
    comment_boundary=True,
)

for _n in ("json", "yaml", "toml", "diff", "html", "css", "markdown"):
    _lang(_n, custom=_n)

_ALIASES = {
    "py": "python", "python3": "python", "js": "javascript", "jsx": "javascript",
    "mjs": "javascript", "cjs": "javascript", "node": "javascript",
    "ts": "typescript", "tsx": "typescript", "sh": "bash", "shell": "bash",
    "zsh": "bash", "console": "bash", "shell-session": "bash", "bash": "bash",
    "c++": "cpp", "cxx": "cpp", "cc": "cpp", "h": "c", "hpp": "cpp",
    "golang": "go", "rs": "rust", "yml": "yaml", "md": "markdown",
    "patch": "diff", "jsonc": "json", "json5": "json", "htm": "html",
    "xml": "html", "svg": "html", "vue": "html", "postgres": "sql",
    "postgresql": "sql", "mysql": "sql", "sqlite": "sql", "scss": "css",
    "less": "css", "text": "", "txt": "", "plain": "", "": "",
}


def _resolve_lang(name: str):
    """Map a fence info string to a language spec (None when unknown)."""
    if not name:
        return None
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    return _LANGS.get(key)


def supported_languages() -> list:
    """Every language name and alias `highlight()` understands."""
    names = {n for n in _LANGS if not n.startswith("_")}
    names |= {a for a, t in _ALIASES.items() if t}
    return sorted(names)


class _Highlighter:
    """Line-at-a-time syntax highlighting; state carries multi-line constructs."""

    def __init__(self, lang, theme: Theme, caps: Caps) -> None:
        self.lang = lang
        self.theme = theme
        self.caps = caps
        self.state = None
        self.fn = _CUSTOM.get(lang.custom or lang.name, _tok_generic) if lang else None

    def feed_line(self, line: str) -> str:
        if self.lang is None:
            return self.theme.render(line, "syn.plain", self.caps)
        tokens, self.state = self.fn(self.lang, line, self.state)
        out = []
        prev_kind = None
        buf = []
        for kind, text in tokens:
            if not text:
                continue
            if kind == prev_kind:
                buf.append(text)
                continue
            if buf:
                out.append(self.theme.render("".join(buf), _TOKEN_STYLE[prev_kind], self.caps))
            prev_kind, buf = kind, [text]
        if buf:
            out.append(self.theme.render("".join(buf), _TOKEN_STYLE[prev_kind], self.caps))
        return "".join(out)


def highlight(code: str, lang, theme: Theme, caps: Caps) -> str:
    """Syntax-highlight a block of code. Unknown languages render plainly.

    `code` is untrusted, so control characters are stripped before anything is
    emitted — see :meth:`MarkdownStream.feed`.
    """
    hl = _Highlighter(_resolve_lang(lang) if isinstance(lang, str) else None, theme, caps)
    return "\n".join(hl.feed_line(line) for line in sanitize_text(code).split("\n"))
