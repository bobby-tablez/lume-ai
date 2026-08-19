"""Slash commands: declaration, parsing, completion, and the help screen.

This module is deliberately inert. It knows the *shape* of every command — name,
arguments, one-line blurb, group, aliases — and nothing whatsoever about what a
command does; the handlers live in ``lume.app``. Keeping the two apart is what
stops the parser, the tab-completer and the help screen from drifting out of step:
there is exactly one list of commands and all three read it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .ansi import Caps, display_width, pad, truncate, wrap
from .theme import Theme, theme_names

__all__ = [
    "ALT_ENTER", "Command", "COMMANDS", "EOF_KEY", "GROUPS", "GROUP_BLURBS",
    "INPUT_RULES",
    "MODEL_HINTS",
    "ORIENTATION", "find", "parse", "suggest", "arg_values", "groups", "names",
    "help_text",
]


@dataclass(frozen=True)
class Command:
    """One slash command's declaration. Frozen: the registry is read-only data."""

    name: str
    args: str
    help: str
    group: str
    aliases: tuple = ()
    #: Kept working, kept out of the help screen — an old name that a newer
    #: command has absorbed. It still parses, completes and has its own
    #: ``/help <name>`` page.
    hidden: bool = False

    @property
    def signature(self) -> str:
        """``/rename [ref] <title>`` — what the help screen shows in column one."""
        return f"/{self.name} {self.args}".rstrip()

    def matches(self, token: str) -> bool:
        """True if `token` (with or without a leading '/') names this command."""
        t = token.lstrip("/").lower()
        return t == self.name or t in self.aliases


#: Display order of the help screen's sections.
GROUPS = ("Session", "Conversation", "Model", "Interface")

#: Two or three words next to each heading, so the shape of the screen is legible
#: before any of it is read.
GROUP_BLURBS = {
    "Session": "saved chats",
    "Conversation": "the one you are in",
    "Model": "how it answers",
    "Interface": "look and keys",
    "Input": "how to type",
}

#: The four commands worth knowing on day one, shown under the title.
ORIENTATION = (("/new", "start"), ("/resume", "last"),
               ("/list", "browse"), ("/quit", "leave"))

COMMANDS = (
    # -- Session ------------------------------------------------------------
    Command("new", "[title]", "Start a fresh conversation, optionally with a title.",
            "Session"),
    Command("resume", "[ref]", 'Reopen a session by id, list number, or "last".',
            "Session", ("r",)),
    Command("list", "[query]", "List saved sessions, newest first; filter with a query.",
            "Session", ("ls",)),
    Command("rename", "[ref] <title>", "Give a session a better title.",
            "Session"),
    Command("delete", "<ref>", "Delete a session and its transcript for good.",
            "Session", ("rm",)),
    Command("export", "[format] [path]",
            "Write the transcript out as markdown, json, or text.", "Session"),
    # -- Conversation -------------------------------------------------------
    Command("system", "[text]", "Show, set, or clear the system prompt.",
            "Conversation"),
    Command("retry", "[note]", "Send the last message again, optionally with a nudge.",
            "Conversation"),
    Command("undo", "[n]", "Drop the last exchange, or the last n, from the chat.",
            "Conversation"),
    Command("edit", "[n]", "Reopen an earlier message in $EDITOR and send it again.",
            "Conversation"),
    Command("copy", "[n]", "Copy the last reply, or message n, to the clipboard.",
            "Conversation", ("y",)),
    Command("clear", "[history]",
            "Clear the screen; add 'history' to also forget the conversation.",
            "Conversation", ("cls",)),
    Command("usage", "", "Show the tokens this session has used and what they cost.",
            "Conversation", ("tokens", "cost")),
    # -- Model --------------------------------------------------------------
    Command("model", "[name]", "Show the current model, or switch to another.",
            "Model", ("m",)),
    Command("models", "", "List every model with its context window and price.",
            "Model"),
    Command("think", "[on|off]", "Turn extended thinking on or off.", "Model"),
    Command("effort", "[level]",
            "How hard to think: low, medium, high, xhigh, or max.", "Model"),
    # -- Interface ----------------------------------------------------------
    Command("theme", "[name]", "Show the current colour theme, or switch to another.",
            "Interface"),
    Command("keys", "", "Show the key bindings and the multi-line input rules.",
            "Interface"),
    Command("help", "[topic]", "Show this help, or the detail for one command.",
            "Interface", ("h", "?")),
    Command("quit", "", "Leave lume.", "Interface", ("exit", "q")),
)

#: Alt+Enter is a readline binding, and readline is not part of a stock Windows
#: Python; Windows Terminal claims the chord for full-screen on top of that. A
#: help screen that offers a key which maximises the window instead of adding a
#: line is worse than one that stays quiet, so there it goes and ``\`` and
#: ``"""`` carry multi-line on their own.
def _alt_enter_bindable() -> bool:
    """Whether the newline chord can actually be bound in this interpreter.

    It is a GNU readline macro, so it needs GNU readline: a stock Windows Python
    has no readline at all, and the libedit shim Apple and several distros ship
    under that name cannot bind it (and lume gives libedit the plain reader
    anyway). Detected here rather than imported from `lume.input`, which imports
    this module.
    """
    if os.name == "nt":
        return False
    try:
        import readline
    except Exception:
        return False
    return "libedit" not in (getattr(readline, "__doc__", "") or "")


ALT_ENTER = _alt_enter_bindable()

#: End-of-file is Ctrl-Z Enter on a Windows console and Ctrl-D everywhere else.
#: Naming the wrong one in the help sends the user pressing a key that does
#: nothing, so the label follows the platform the help is printed on.
EOF_KEY = "Ctrl-Z Enter" if os.name == "nt" else "Ctrl-D"

#: How to type things at the prompt. Shown by ``/help`` and ``/keys``; the rules
#: are enforced by :class:`lume.input.Prompt`, and the two must agree.
INPUT_RULES = tuple(row for row in (
    ("Enter", "Send the message."),
    ("\\ at end of line", "Continue on the next line; the backslash is dropped."),
    ('"""', 'Open a block. A line ending in """ closes it and sends.'),
    ("Alt+Enter", "Add a newline without sending.") if ALT_ENTER else None,
    ("paste", "A multi-line paste arrives whole, as one message."),
    ("//text", "Send a line that really does start with a slash."),
    ("Tab", "Complete a command or its argument."),
    ("Up / Down", "Walk back and forth through history."),
    ("Ctrl-C", "Throw the line away and start over."),
    (EOF_KEY, "Exit lume, on an empty line."),
) if row is not None)

#: Completion hints only — ``/models`` is the authoritative list. Declaring them
#: here keeps `commands` free of any dependency on the API client.
MODEL_HINTS = ("opus", "sonnet", "haiku", "fable")

_ARG_VALUES = {
    "think": ("on", "off"),
    "effort": ("low", "medium", "high", "xhigh", "max"),
    "export": ("markdown", "json", "text"),
    "model": MODEL_HINTS,
    "resume": ("last",),
    "system": ("clear", "off", "none"),
    "clear": ("history",),
}

# Registry index: name and alias -> Command. Built once, never mutated.
_INDEX = {}
for _c in COMMANDS:
    _INDEX[_c.name] = _c
    for _a in _c.aliases:
        _INDEX[_a] = _c

# A command name is a letter (or '?') followed by name characters, and must be
# followed by whitespace or end-of-line — so '/usr/bin/env python' stays prose.
_CMD_RE = re.compile(r"/([A-Za-z?][A-Za-z0-9_?-]*)(?=$|[^\S\n]|\n)(.*)", re.DOTALL)


def names() -> tuple:
    """Every canonical command name, in declaration order."""
    return tuple(c.name for c in COMMANDS)


def groups() -> tuple:
    """Group names in display order (only those that actually have commands)."""
    used = {c.group for c in COMMANDS}
    return tuple(g for g in GROUPS if g in used)


def find(name: str) -> "Command | None":
    """Look a command up by name or alias, with or without a leading '/'."""
    if not name:
        return None
    return _INDEX.get(str(name).lstrip("/").strip().lower())


def arg_values(name: str) -> tuple:
    """Static completion values for a command's first argument ( () if free text)."""
    cmd = find(name)
    if cmd is None:
        return ()
    if cmd.name == "theme":
        return tuple(theme_names())
    if cmd.name == "help":
        # Every topic /help understands, each of them once: the group names
        # overlap the command names ("model" is both), and Tab offering the same
        # word twice is Tab admitting it does not know what it is completing.
        topics = names() + tuple(g.lower() for g in groups()) + ("input", "rules")
        return tuple(dict.fromkeys(topics))
    return _ARG_VALUES.get(cmd.name, ())


def parse(line: str) -> tuple:
    """Split a submission into ``(command_name, arguments)``.

    ``'/model sonnet'`` -> ``('model', 'sonnet')``; ordinary prose ->
    ``(None, line)`` with the line handed back untouched. Aliases resolve to the
    canonical name (``'/q'`` -> ``('quit', '')``); an unknown ``'/xyz'`` still
    returns ``('xyz', rest)`` so the app can say "unknown command", and a lone
    ``'/'`` returns ``('', '')``. A line starting ``'//'`` is the escape hatch for
    literal text: one slash is eaten and the rest is prose.

    Anything that only *looks* like a command is prose — ``/usr/bin/env`` (no
    space after the name) and ``/2+2`` (does not start with a letter) both come
    back as text. **So is anything containing a newline**: a slash command is one
    line by construction, so a pasted document whose first line happens to start
    with a slash is a message, not a command with a very large argument.
    """
    if not line:
        return (None, line or "")
    if "\n" in line:                     # a paste is never a command
        return (None, line)
    stripped = line.lstrip(" \t")
    if not stripped.startswith("/"):
        return (None, line)
    if stripped.startswith("//"):
        return (None, stripped[1:])
    if stripped.strip() == "/":
        return ("", "")
    m = _CMD_RE.match(stripped)
    if m is None:
        return (None, line)
    token, rest = m.group(1), m.group(2)
    cmd = find(token)
    return ((cmd.name if cmd is not None else token), rest.strip())


def suggest(prefix: str) -> list:
    """Completion candidates for the text typed so far.

    With no whitespace the prefix is a command name and the results carry a
    leading '/' exactly when the prefix did (readline replaces the whole token).
    Once there is a space the prefix is completed as that command's *argument*
    and bare values come back.

    Two rules keep Tab honest about what Enter would do. An alias is only offered
    when no canonical name matches, so '/l' completes to '/list' rather than a
    mixed bag — but a prefix that *is* itself a command comes back in the list
    even when longer names match it, so completing '/r' can never rewrite it to
    '/re', which Enter would then reject. Leading whitespace is ignored, because
    the prompt tolerates it too.
    """
    if prefix is None:
        return []
    prefix = prefix.lstrip()

    cut = max(prefix.rfind(" "), prefix.rfind("\t"))
    if cut >= 0:                                   # completing an argument
        head, cur = prefix[:cut].strip(), prefix[cut + 1:]
        first = head.split()[0] if head.split() else ""
        if not first.startswith("/"):
            return []
        return [v for v in arg_values(first) if v.startswith(cur)]

    raw = prefix.strip()
    slash = raw.startswith("/")
    token = raw[1:].lower() if slash else raw.lower()
    lead = "/" if slash or token == "" else ""
    hits = [c.name for c in COMMANDS if c.name.startswith(token)]
    if token and not token.startswith("/") and token not in hits and find(token) is not None:
        hits.append(token)                         # what Enter would already run
    if not hits:
        hits = [a for c in COMMANDS for a in c.aliases if a.startswith(token)]
    return [lead + h for h in sorted(hits)]


# ----------------------------------------------------------------- help screen


def _cut(s: str, width: int, caps: Caps) -> str:
    """Truncate to `width`, with an ellipsis the terminal can actually draw."""
    if display_width(s) <= width:
        return s
    return truncate(s, width, "…" if caps.unicode else "...")


def _sig(cmd: Command, theme: Theme, caps: Caps) -> str:
    """``/list, /ls [query]`` — the name, its aliases, then the arguments."""
    out = theme.render("/" + cmd.name, "cmd.name", caps)
    if cmd.aliases:
        out += theme.render(", " + ", ".join("/" + a for a in cmd.aliases),
                            "app.dim", caps)
    if cmd.args:
        out += " " + theme.render(cmd.args, "cmd.args", caps)
    return out


def _two_col(rows, width: int, indent: str = "    ", gutter: int = 2,
             colw: int = None) -> list:
    """Lay out (left, right) pairs as an aligned table that never exceeds `width`.

    `colw` is passed in by the help screen so every section shares one column and
    the whole page lines up. A row whose left side overflows keeps its own line
    rather than squeezing the prose, and a very narrow terminal stacks them all.
    """
    if not rows:
        return []
    iw = display_width(indent)
    avail = max(2, width - iw)
    out = []

    if avail < 34:                                  # too narrow to be a table
        for left, right in rows:
            out.append(indent + truncate(left, avail))
            out.extend(wrap(right, width, indent + "  ", indent + "  "))
        return out

    natural = colw if colw is not None else max(display_width(l) for l, _ in rows)
    colw = min(natural, max(8, avail - 18 - gutter))
    hang = indent + " " * (colw + gutter)
    for left, right in rows:
        if display_width(left) > colw:
            out.append(indent + truncate(left, avail))
            out.extend(wrap(right, width, hang, hang))
        else:
            out.extend(wrap(right, width, indent + pad(left, colw + gutter), hang))
    return out


def _heading(label: str, theme: Theme, caps: Caps, width: int, indent: str = "  ") -> str:
    """``Session · saved chats ─────`` — the label, what it is for, then a rule."""
    head = theme.render(label, "cmd.group", caps)
    blurb = GROUP_BLURBS.get(label, "")
    if blurb:
        dot = "·" if caps.unicode else "-"
        head += theme.render(f" {dot} {blurb}", "app.dim", caps)
    line = indent + head
    room = width - display_width(line) - 1
    if room >= 4 and width >= 52:
        dash = "─" if caps.unicode else "-"
        line += " " + theme.render(dash * room, "app.rule", caps)
    return _cut(line, width, caps)


def _orientation(theme: Theme, caps: Caps, width: int) -> list:
    """One line of orientation: the four commands a new user needs, and nothing else."""
    dot = " · " if caps.unicode else " | "
    sep = theme.render(dot, "app.rule", caps)
    body = sep.join(theme.render(name, "cmd.name", caps) + " "
                    + theme.render(what, "app.dim", caps)
                    for name, what in ORIENTATION)
    plain = dot.join(f"{n} {w}" for n, w in ORIENTATION)
    if display_width(plain) + 2 > width:            # no room: the help table says it all
        return []
    return ["  " + body]


def _detail(cmd: Command, theme: Theme, caps: Caps, width: int) -> list:
    """The ``/help <command>`` view: one command, in full."""
    tag = theme.render(cmd.group.lower(), "cmd.group", caps)
    left = "  " + _sig(cmd, theme, caps)
    room = width - display_width(tag)
    out = [_cut(pad(left, room) + tag if room > display_width(left) + 2 else left,
                width, caps)]
    out.append("")
    out.extend(wrap(theme.render(cmd.help, "cmd.help", caps), width, "  ", "  "))
    vals = arg_values(cmd.name)
    if vals:
        out.append("")
        out.extend(wrap(theme.render("values: ", "cmd.group", caps)
                        + theme.render(" ".join(vals), "cmd.args", caps),
                        width, "  ", "  "))
    return out


def _index(theme: Theme, caps: Caps, width: int, title: str) -> list:
    """The short form of the help screen: what there is, and where to read it.

    A 24-line terminal cannot show 62 lines of table; it shows the *last* 24 of
    them, which is the half without the title, the orientation strip or the
    first two groups. So when the table will not fit, this goes out instead —
    every command still named, one line per group, and a pointer at the page
    that explains them.
    """
    out = [_cut("  " + title, width, caps)]
    out.extend(_orientation(theme, caps, width))
    for label in groups():
        row = [c for c in COMMANDS if c.group == label and not c.hidden]
        if not row:
            continue
        out.append("")
        out.append(_heading(label, theme, caps, width))
        line = theme.render("  ".join("/" + c.name for c in row), "cmd.name", caps)
        out.extend(wrap(line, width, "    ", "    "))
    out.append("")
    out.extend(wrap(theme.render("/help <topic> for what one does, /keys for the "
                                 "typing rules.", "app.dim", caps),
                    width, "  ", "  "))
    return out


def help_text(theme: Theme, caps: Caps, width: int = None, group: str = None,
              height: int = None) -> str:
    """Render the help screen: a grouped, aligned, themed table that fits `width`.

    `group` narrows the output - a group name ("session") shows that section,
    "input" shows the typing rules alone (this is what ``/keys`` prints), and a
    command name or alias ("resume", "/r") shows that one command in full. An
    unknown topic is called out and the whole screen follows. No trailing newline.

    The requested width is honoured, not clamped: 20 columns gets 20-column lines
    (stacked and truncated, but never wrapped by the terminal behind our back).
    So is the *height*: `height` (default ``caps.height``) is how many lines
    there are to write on, and the full table is only printed when it fits.
    Pass ``0`` for no limit. There is no pager here, and a screen that has
    scrolled its own title away has told the reader nothing.
    """
    width = max(1, min(int(width or caps.width or 80), 400))
    dot = "·" if caps.unicode else "-"
    out = []

    topic = (group or "").strip().lstrip("/").lower()
    section = None
    if topic:
        for g in groups():
            if g.lower() == topic:
                section = g
        # "input" is the section /keys prints: the typing rules on their own.
        if topic in ("input", "rules"):
            section = "Input"
        cmd = find(topic)
        if section is None and cmd is not None:
            return "\n".join(_clamp(_detail(cmd, theme, caps, width), width, caps))
        if section is None:
            out.append(_cut(theme.render(f"No help topic {topic!r}.", "app.warn", caps),
                            width, caps))
            out.append("")

    input_rows = [(theme.render(k, "cmd.name", caps), theme.render(v, "cmd.help", caps))
                  for k, v in INPUT_RULES]
    sections = [(g, [(_sig(c, theme, caps), theme.render(c.help, "cmd.help", caps))
                     for c in COMMANDS if c.group == g and not c.hidden])
                for g in ([] if section == "Input" else
                          [section] if section else list(groups()))]
    if section is None or section == "Input":
        sections.append(("Input", input_rows))
    # One column width for the whole page: sections that align with each other
    # read as one table instead of four.
    colw = max((display_width(l) for _, rows in sections for l, _ in rows), default=8)

    title = (theme.render("lume", "app.accent", caps) + " "
             + theme.render(dot, "app.rule", caps) + " "
             + theme.render("commands" if section is None else section.lower(),
                            "app.text", caps))
    hint = theme.render("/help <topic> for detail", "app.dim", caps)
    if width - display_width(title) - display_width(hint) - 2 >= 2:
        title = pad(title, width - display_width(hint) - 2) + hint
    warning = list(out)                  # an unknown topic was called out above
    out.append(_cut("  " + title, width, caps))
    if section is None:
        out.extend(_orientation(theme, caps, width))

    for label, rows in sections:
        if not rows:
            continue
        out.append("")
        out.append(_heading(label, theme, caps, width))
        out.extend(_two_col(rows, width, colw=colw))

    if section is None:
        out.append("")
        out.extend(wrap(
            theme.render("Anything that is not a command is sent to Claude.",
                         "app.dim", caps),
            width, "  ", "  "))

    room = caps.height if height is None else height
    if section is None and room and len(out) > max(4, int(room)) - 1:
        out = warning + _index(theme, caps, width, title)

    return "\n".join(_clamp(out, width, caps))


def _clamp(lines, width: int, caps: Caps) -> list:
    """Last line of defence for the width contract, whatever the layout did."""
    return [_cut(line, width, caps) for line in lines]
