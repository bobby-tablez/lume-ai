"""Named colour themes.

A theme maps a *token* (a semantic name like ``md.h1``) to a :class:`lume.ansi.Style`.
Nothing outside this module hard-codes a colour, so a new palette is a data change.

Legibility is guaranteed by construction, not by taste: every palette declares its
background, and each token declares the minimum WCAG contrast ratio its role needs.
`_ensure` lifts any colour that falls short toward the far end of the scale, so a
pretty-but-invisible hex value cannot ship.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

from .ansi import NULL_STYLE, Caps, Style, blend, hex_rgb

__all__ = ["TOKENS", "Theme", "THEMES", "get_theme", "theme_names", "Style", "contrast"]

#: The complete token vocabulary. Every theme must define every token.
TOKENS = (
    # chrome
    "app.bg", "app.accent", "app.accent2", "app.text", "app.muted", "app.dim",
    "app.rule", "app.error", "app.warn", "app.success", "app.info",
    # conversation
    "user.label", "user.text", "assistant.label", "assistant.text",
    "system.label", "system.text", "thinking.label", "thinking.text",
    # markdown
    "md.h1", "md.h2", "md.h3", "md.bold", "md.italic", "md.strike",
    "md.code", "md.code_bg", "md.code_border", "md.quote", "md.quote_bar",
    "md.bullet", "md.number", "md.link", "md.link_url", "md.rule",
    "md.table_head", "md.table_border", "md.task_done", "md.task_todo",
    # syntax highlighting
    "syn.keyword", "syn.string", "syn.number", "syn.comment", "syn.func",
    "syn.builtin", "syn.operator", "syn.punct", "syn.type", "syn.decorator",
    "syn.variable", "syn.plain",
    # status / prompt / motion
    "status.bar", "status.key", "status.value", "status.sep",
    "prompt.marker", "prompt.text", "prompt.hint",
    "spinner.from", "spinner.to",
    # commands / help
    "cmd.name", "cmd.args", "cmd.help", "cmd.group",
)

#: Minimum contrast against the theme background, by role.
#: Body text clears AA; supporting text clears AA-large; structural rules only
#: need to be *seen*, so they sit deliberately below text thresholds.
_MIN_CONTRAST = {
    "app.text": 7.0, "md.bold": 7.0, "assistant.text": 7.0, "user.text": 7.0,
    "prompt.text": 7.0, "md.italic": 6.0,
    # Headings descend in prominence, and the floors enforce it rather than
    # leaving it to whichever hue a palette happened to pick.
    "md.h1": 11.0, "md.h2": 8.6, "md.table_head": 4.5,
    "user.label": 4.5, "assistant.label": 4.5, "system.label": 4.5,
    "app.error": 4.5, "app.warn": 4.5, "app.success": 4.5, "app.info": 4.5,
    "app.accent": 4.5, "app.accent2": 4.5, "md.link": 4.5, "cmd.name": 4.5,
    "app.muted": 4.5, "md.code": 4.5, "md.bullet": 4.5, "md.number": 4.5,
    "prompt.marker": 4.5, "status.value": 4.5, "md.task_done": 4.5,
    "cmd.args": 4.2, "cmd.help": 4.2, "system.text": 4.2, "md.quote": 4.2,
    "md.quote_bar": 4.0, "md.task_todo": 4.0, "status.bar": 4.2,
    "spinner.from": 4.0, "spinner.to": 4.0, "md.strike": 3.5,
    "app.dim": 4.5, "thinking.label": 4.5, "thinking.text": 4.5,
    "prompt.hint": 4.5, "status.key": 4.5, "md.link_url": 4.5, "cmd.group": 4.5,
    "syn.comment": 4.5, "syn.punct": 4.5, "md.h3": 4.6,
    # Rules are non-text, so WCAG 1.4.11's 3.0 applies rather than 4.5.
    "app.rule": 3.0, "md.rule": 3.0, "md.code_border": 3.0,
    "md.table_border": 3.0, "status.sep": 3.0,
}
_DEFAULT_MIN = 4.0          # everything else, notably the syntax vocabulary

_WARNED = set()


def _relative_luminance(c) -> float:
    def f(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (f(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b) -> float:
    """WCAG contrast ratio between two colours, 1.0 (identical) to 21.0."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _ensure(fg, bg, target: float):
    """Lift `fg` toward white or black until it reaches `target` against `bg`."""
    if contrast(fg, bg) >= target:
        return fg
    toward = (255, 255, 255) if _relative_luminance(bg) < 0.5 else (0, 0, 0)
    lo, hi = 0.0, 1.0
    if contrast(blend(fg, toward, hi), bg) < target:
        return toward                       # even the extreme cannot reach it
    for _ in range(20):
        mid = (lo + hi) / 2
        if contrast(blend(fg, toward, mid), bg) >= target:
            hi = mid
        else:
            lo = mid
    return blend(fg, toward, hi)


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool
    styles: Mapping

    def __getitem__(self, token: str) -> Style:
        try:
            return self.styles[token]
        except KeyError:
            # A typo'd token would otherwise render as plain body text forever.
            key = (self.name, token)
            if key not in _WARNED:
                _WARNED.add(key)
                warnings.warn(f"unknown theme token {token!r}", stacklevel=2)
            return self.styles.get("app.text", NULL_STYLE)

    def get(self, token: str, default: Style = None) -> Style:
        return self.styles.get(token, default if default is not None else NULL_STYLE)

    def render(self, text: str, token: str, caps: Caps) -> str:
        return self[token](text, caps)

    @property
    def background(self):
        return self["app.bg"].bg or ((0, 0, 0) if self.dark else (255, 255, 255))

    def accent_stops(self) -> list:
        """Colours for gradients (banner, spinner).

        `mono` has no colours at all, so it falls back to a neutral ramp: callers
        get a usable gradient and the terminal simply ignores it.
        """
        stops = [self["app.accent"].fg, self["app.accent2"].fg, self["spinner.to"].fg]
        if all(s is None for s in stops):
            return []          # a theme with no colours paints no gradient
        fallback = [(120, 120, 120), (200, 200, 200), (255, 255, 255)]
        return [s if s is not None else fallback[i] for i, s in enumerate(stops)]


def _build(name, dark, palette):
    p = {k: hex_rgb(v) for k, v in palette.items()}
    bg = p["bg"]
    S = Style
    raw = {
        "app.bg": S(bg=bg),
        "app.accent": S(fg=p["accent"], bold=True),
        "app.accent2": S(fg=p["accent2"]),
        "app.text": S(fg=p["text"]),
        "app.muted": S(fg=p["muted"]),
        "app.dim": S(fg=p["faint"]),
        "app.rule": S(fg=p["rule"]),
        "app.error": S(fg=p["error"], bold=True),
        "app.warn": S(fg=p["warn"]),
        "app.success": S(fg=p["success"]),
        "app.info": S(fg=p["accent2"]),

        "user.label": S(fg=p["user"], bold=True),
        "user.text": S(fg=p["text"]),
        "assistant.label": S(fg=p["accent"], bold=True),
        "assistant.text": S(fg=p["text"]),
        "system.label": S(fg=p["warn"], bold=True),
        "system.text": S(fg=p["muted"], italic=True),
        "thinking.label": S(fg=p["faint"], italic=True),
        "thinking.text": S(fg=p["faint"], italic=True),

        # Prominence descends h1 -> h2 -> h3. A third hue would be noise, so h3
        # recedes into the muted neutral instead of outshining the other two.
        "md.h1": S(fg=p["accent"], bold=True),
        "md.h2": S(fg=p["accent2"], bold=True),
        "md.h3": S(fg=p["muted"], bold=True),
        "md.bold": S(fg=p["strong"], bold=True),
        "md.italic": S(fg=p["text"], italic=True),
        "md.strike": S(fg=p["muted"], strike=True),
        "md.code": S(fg=p["code"]),
        "md.code_bg": S(bg=p["code_bg"]),
        "md.code_border": S(fg=p["rule"]),
        "md.quote": S(fg=p["muted"], italic=True),
        "md.quote_bar": S(fg=p["accent2"]),
        "md.bullet": S(fg=p["accent2"], bold=True),
        "md.number": S(fg=p["accent2"]),
        "md.link": S(fg=p["link"], underline=True),
        "md.link_url": S(fg=p["faint"]),
        "md.rule": S(fg=p["rule"]),
        "md.table_head": S(fg=p["accent2"], bold=True),
        "md.table_border": S(fg=p["rule"]),
        "md.task_done": S(fg=p["success"]),
        "md.task_todo": S(fg=p["muted"]),

        "syn.keyword": S(fg=p["kw"], bold=True),
        "syn.string": S(fg=p["str"]),
        "syn.number": S(fg=p["num"]),
        "syn.comment": S(fg=p["comment"], italic=True),
        "syn.func": S(fg=p["func"]),
        "syn.builtin": S(fg=p["builtin"]),
        "syn.operator": S(fg=p["op"]),
        "syn.punct": S(fg=p["punct"]),
        "syn.type": S(fg=p["type"]),
        "syn.decorator": S(fg=p["decorator"]),
        # Deliberately identical: both mean "ordinary identifier text". They
        # exist as separate tokens so a future theme can split them.
        "syn.variable": S(fg=p["code"]),
        "syn.plain": S(fg=p["code"]),

        "status.bar": S(fg=p["muted"]),
        "status.key": S(fg=p["faint"]),
        "status.value": S(fg=p["accent2"]),
        "status.sep": S(fg=p["rule"]),

        "prompt.marker": S(fg=p["user"], bold=True),
        "prompt.text": S(fg=p["text"]),
        "prompt.hint": S(fg=p["faint"], italic=True),

        "spinner.from": S(fg=p["accent"]),
        "spinner.to": S(fg=p["accent2"]),

        "cmd.name": S(fg=p["accent"], bold=True),
        "cmd.args": S(fg=p["accent2"]),
        "cmd.help": S(fg=p["muted"]),
        "cmd.group": S(fg=p["strong"], bold=True),
    }

    styles = {}
    for token, style in raw.items():
        if style.fg is not None:
            target = _MIN_CONTRAST.get(token, _DEFAULT_MIN)
            # Enforce against the surface the token is actually drawn on.
            surface = p["code_bg"] if token.startswith(("syn.", "md.code")) else bg
            lifted = _ensure(style.fg, surface, target)
            style = Style(fg=lifted, bg=style.bg, bold=style.bold, dim=style.dim,
                          italic=style.italic, underline=style.underline,
                          strike=style.strike, reverse=style.reverse)
        styles[token] = style

    missing = [t for t in TOKENS if t not in styles]
    if missing:
        raise AssertionError(f"theme {name!r} missing tokens: {missing}")
    return Theme(name=name, dark=dark, styles=styles)


AURORA = _build("aurora", True, {
    "bg": "#0f1117", "code_bg": "#171a22",
    "accent": "#5fd7bd", "accent2": "#a08cff", "text": "#dfe4ec",
    "muted": "#9aa3b7", "faint": "#6f7889", "rule": "#3a4152",
    "strong": "#f4f7fc", "error": "#ff6b7f", "warn": "#f0b45f",
    "success": "#63d18c", "user": "#63a8ff", "link": "#63a8ff",
    "code": "#c4cbdb", "punct": "#aab3c6",
    "kw": "#c792ea", "str": "#9ede93", "num": "#f0b45f", "comment": "#6f7889",
    "func": "#5fd7bd", "builtin": "#79b8ff", "op": "#f2849e", "type": "#f5c2a7",
    "decorator": "#ffd98f",
})

SOLAR = _build("solar", False, {
    "bg": "#fbfbfd", "code_bg": "#eef0f5",
    "accent": "#0b7a6b", "accent2": "#6740c4", "text": "#1c2129",
    "muted": "#525a68", "faint": "#767e8c", "rule": "#b9c0cb",
    "strong": "#0a0d12", "error": "#c01f31", "warn": "#8a5200",
    "success": "#0f6b36", "user": "#1a52b8", "link": "#1a52b8",
    "code": "#242a34", "punct": "#3d4552",
    "kw": "#7b28a8", "str": "#1a6b35", "num": "#8a5200", "comment": "#6b7280",
    "func": "#0b7a6b", "builtin": "#1a52b8", "op": "#b02a6b", "type": "#a04516",
    "decorator": "#4a5d00",
})

EMBER = _build("ember", True, {
    "bg": "#14100d", "code_bg": "#1e1813",
    "accent": "#f0a15a", "accent2": "#ff8f7a", "text": "#e8e2db",
    "muted": "#a99789", "faint": "#7d6d61", "rule": "#453931",
    "strong": "#fdf7f0", "error": "#ff6b6b", "warn": "#f2ce74",
    "success": "#a8d97e", "user": "#7cc0ff", "link": "#7cc0ff",
    "code": "#d8cfc7", "punct": "#bcaea3",
    "kw": "#d69bf0", "str": "#9fd67f", "num": "#f2ce74", "comment": "#8d7b6e",
    "func": "#f0a15a", "builtin": "#7cc0ff", "op": "#ff8f7a", "type": "#5fc9b0",
    "decorator": "#e0a0b8",
})

# The theme for terminals that have no colour at all: meaning lives entirely in
# SGR attributes. There are only six of those, so they are spent on the roles that
# actually appear next to each other — full syntax highlighting in monochrome is
# noise, so code gets the three distinctions that carry the most meaning.
_MONO_ATTRS = {
    "bold": ("app.accent", "md.h1", "md.bold", "user.label", "assistant.label",
             "system.label", "cmd.name", "md.table_head", "prompt.marker",
             "cmd.group", "app.warn", "app.success", "md.bullet", "md.number",
             "md.task_done", "status.value", "syn.keyword", "spinner.from",
             "spinner.to"),
    "bold_underline": ("md.h2",),
    "bold_italic": ("md.h3", "syn.type"),
    "underline": ("md.link", "syn.func"),
    "italic": ("md.italic", "md.quote", "thinking.label", "system.text",
               "prompt.hint", "syn.string", "app.info", "cmd.args"),
    "dim_italic": ("thinking.text", "syn.comment"),
    "dim": ("app.dim", "status.key", "md.link_url", "app.rule", "md.rule",
            "md.code_border", "md.table_border", "status.sep", "cmd.help",
            "syn.punct", "md.task_todo", "syn.decorator"),
    "strike": ("md.strike",),
    "reverse": ("app.error",),
}
_MONO_STYLES = {t: Style() for t in TOKENS}
for _kind, _tokens in _MONO_ATTRS.items():
    for _t in _tokens:
        _MONO_STYLES[_t] = Style(
            bold="bold" in _kind, dim="dim" in _kind, italic="italic" in _kind,
            underline="underline" in _kind, strike="strike" in _kind,
            reverse="reverse" in _kind,
        )
# markdown.py draws a border around code blocks, so mono needs no block fill —
# reverse video across a whole listing is unreadable.

MONO = Theme(name="mono", dark=True, styles=_MONO_STYLES)

#: No colour and no attributes: for pipes, logs, and `--plain`. `mono` still uses
#: bold/italic/dim to carry meaning; this uses nothing at all.
PLAIN = Theme(name="plain", dark=True, styles={t: Style() for t in TOKENS})

THEMES = {t.name: t for t in (AURORA, SOLAR, EMBER, MONO, PLAIN)}
DEFAULT_THEME = "aurora"


def theme_names() -> list:
    return list(THEMES)


def default_theme_for(background: str = "") -> str:
    """The theme to use when the user has not chosen one.

    A dark palette on a white terminal is unreadable — `aurora`'s body text sits
    at contrast 1.28 there — so honour the terminal's own report when we have one.
    """
    return "solar" if background == "light" else DEFAULT_THEME


def get_theme(name: str = None, caps: Caps = None, background: str = "") -> Theme:
    """Resolve a theme by name, degrading to `mono` when the terminal has no colour."""
    key = str(name).strip().lower() if name else ""
    if key == "plain":
        return PLAIN            # an explicit request for no escapes at all
    if caps is not None and not caps.color:
        return MONO
    fallback = THEMES[default_theme_for(background)]
    if not key or key == "auto":
        return fallback
    return THEMES.get(key, fallback)
