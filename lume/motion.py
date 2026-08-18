"""Motion: transient status spinners, the gradient wordmark, rules, fades.

The whole module obeys one taste rule: *slight* motion. A spinner is a single
cell that never changes width (so the line can never jitter), its colour drifts
slowly along the theme accent ramp, and the elapsed clock only appears once the
wait is long enough to be worth reporting.

The mechanical rule is stricter: the animation thread owns exactly one line. It
writes only through ``console.write`` while holding ``console.lock``, never
emits ``\\n`` (labels are flattened on the way in), and every teardown path —
normal, exception, ``KeyboardInterrupt``, double stop, interpreter exit —
erases that line and shows the cursor again.

This module deliberately installs **no signal handler**. A library that
overwrites its host's ``SIGINT`` to clean one line costs the application its
own Ctrl-C — and ``lume.app`` uses ``SIGINT`` to cancel a reply while
``lume.input`` uses it to clear the prompt line. Nothing is lost: a ``SIGINT``
that raises ``KeyboardInterrupt`` unwinds the ``with`` block and hits the
``finally:`` in :meth:`Animator.status` like any other exception, an atexit
sweep here erases the line if the process leaves a block open, and
``lume.ansi`` owns the process-wide cursor net (its atexit hook always, plus
the SIGTERM/SIGHUP one that ``lume.cli`` opts into).

Three locks are involved, and the order between them is the whole reason this
module can be driven from several threads at once:

* ``console.lock`` — the terminal. A foreground writer may already hold it when
  it calls in here, so it is always the *outermost* of the three. It is also the
  mutex the animator's cursor/line ownership flags live under.
* ``Animator._lock`` — the status stack and the frame thread. Taken alone, for a
  handful of instructions, and **never** held across a ``console.lock``
  acquire: that is the ABBA pair, and it deadlocked a foreground writer.
* ``Animator._flags`` — a leaf, taken innermost and released before anything
  else is called.
"""

from __future__ import annotations

import atexit
import re
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass

from .ansi import (
    CLEAR_LINE,
    CLEAR_TO_END,
    ESC,
    RESET,
    SHOW_CURSOR,
    Caps,
    Console,
    Style,
    blend,
    display_width,
    fg,
    gradient,
    rgb_to_16,
    sanitize_text,
    truncate,
    up,
)
from .ansi import _ANSI16          # the 16 slots, to score a colour after quantising
from .theme import Theme, contrast

__all__ = [
    "Spinner",
    "SPINNERS",
    "DEFAULT_SPINNER",
    "get_spinner",
    "StatusHandle",
    "Animator",
    "banner",
    "rule",
    "fade_in",
    "wordmark_lines",
]


# --------------------------------------------------------------------- tuning

_PHASES = 24            # colour-ramp steps; ~2s per drift cycle at spinner speed
_ELAPSED_AFTER = 2.0    # seconds before the elapsed clock joins the line
_LOCK_TIMEOUT = 0.05    # how long a frame waits for console.lock before skipping
_RESTORE_TIMEOUT = 0.15  # ditto for the final erase; then we bypass the lock
_CLAIM_TIMEOUT = 0.15   # ...and for the cursor claim as a status block opens
_JOIN_TIMEOUT = 1.0
_MAX_FPS = 60.0

#: Anything that would break the one-line invariant, mapped to a space.
_FLATTEN = {ord(c): " " for c in "\n\r\t\v\f\b\a\x00"}

#: The one escape a label may carry: a colour change. Everything else an ESC
#: can start — cursor motion, a screen clear, an OSC title or clipboard write —
#: has no business inside a one-line status label.
_SGR = re.compile(r"\x1b\[[0-9;:]*m")


def _cut(text: str, width: int, caps: Caps) -> str:
    """Truncate to `width` columns with an ellipsis the terminal can draw:
    `truncate`'s default is U+2026, which is mojibake on an ASCII tty."""
    return truncate(text, width, "\u2026" if caps.unicode else "...")


def _oneline(text) -> str:
    """A label that cannot escape its line, or the terminal.

    ``\\n`` is the interesting one: one newline per frame turns CLEAR_LINE into a
    no-op and scrolls the spinner into the scrollback forever. Those become
    spaces, so two words never run together; everything else ``sanitize_text``
    strips — notably DEL and the C1 block, which contains the 8-bit CSI
    introducer ``\\x9b``, i.e. a second way to spell ``ESC[`` and clear the
    screen from inside a label.

    SGR escapes survive, deliberately: a caller may hand us pre-styled text, and
    that is the whole of what a label needs an escape for. An escape that moves
    the cursor, clears the screen or opens an OSC does not survive — it would
    leave the animator's one line and take the terminal with it.
    """
    flat = str(text).translate(_FLATTEN)
    out, pos = [], 0
    for match in _SGR.finditer(flat):
        out.append(sanitize_text(flat[pos:match.start()], keep_newlines=False))
        out.append(match.group())
        pos = match.end()
    out.append(sanitize_text(flat[pos:], keep_newlines=False))
    return "".join(out)


# -------------------------------------------------------------------- spinners


@dataclass(frozen=True)
class Spinner:
    """A named frame set. Every frame of a set is one display column wide, so
    the status line never reflows while it spins."""

    name: str
    unicode: tuple
    ascii: tuple
    interval: float = 0.09

    def frames(self, caps: Caps) -> tuple:
        """The frame tuple this terminal can actually render."""
        return self.unicode if caps.unicode else self.ascii


def _spinners(*specs) -> dict:
    return {s.name: s for s in specs}


#: Named frame sets. Four, not a catalogue: each one is a different *idea*
#: (rotation, breathing, level, fill) and each keeps that idea in ASCII, so no
#: two sets collapse into the same animation on a terminal without unicode.
SPINNERS = _spinners(
    # rotation — the calm default
    Spinner("orbit", tuple("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"), tuple("|/-\\"), 0.085),
    # breathing
    Spinner("pulse", tuple("·∘•●•∘"), tuple(".oO0Oo"), 0.13),
    # a level rising and falling
    Spinner("wave", tuple("▁▂▃▄▅▆▇▆▅▄▃▂"), tuple("_-~-"), 0.075),
    # one cell filling and emptying (ASCII: filling and starting over — a
    # ramp back down would be `pulse`'s breath again, and a space would make
    # the spinner *vanish* for a sixth of every cycle)
    Spinner("bar", tuple("▏▎▍▌▋▊▉▊▋▌▍▎"), tuple("_.-:=#"), 0.075),
)

DEFAULT_SPINNER = "orbit"


def get_spinner(name) -> Spinner:
    """Look up a spinner, falling back to the default rather than raising —
    a mistyped style name must never take down a chat session."""
    if isinstance(name, Spinner):
        return name
    return SPINNERS.get(str(name or "").strip().lower(), SPINNERS[DEFAULT_SPINNER])


def _stops(theme: Theme) -> list:
    """Accent stops with consecutive duplicates removed: several themes point
    two stops at the same colour, and a repeated stop flattens half the ramp."""
    out = []
    for c in theme.accent_stops():
        c = tuple(c)
        if not out or out[-1] != c:
            out.append(c)
    return out or [(255, 255, 255)]


class _Painter:
    """Pre-rendered escapes for one spinner, so a frame costs a few concats."""

    def __init__(self, spinner: Spinner, theme: Theme, caps: Caps):
        self.frames = spinner.frames(caps)
        self.interval = spinner.interval
        if caps.color >= 8:
            # Ping-pong the accent stops so the drift loops without a seam.
            stops = _stops(theme)
            ramp = stops + stops[-2::-1] if len(stops) > 1 else stops
            self.colors = [fg(c, caps) for c in gradient(ramp, _PHASES)]
        elif caps.color:
            # 4-bit colour quantises a gradient into a flicker; stay still.
            self.colors = [theme["spinner.from"].codes(caps)]
        else:
            self.colors = [""]
        self._reset = RESET if any(self.colors) else ""

    def glyph(self, tick: int) -> str:
        color = self.colors[tick % len(self.colors)]
        char = self.frames[tick % len(self.frames)]
        return f"{color}{char}{self._reset}" if color else char


# ------------------------------------------------------------------- elapsed


def _fmt_elapsed(seconds: float) -> str:
    """Whole seconds only: a tenths counter reads as noise, not as progress."""
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# --------------------------------------------------------------------- status


class StatusHandle:
    """The object a ``status()`` block receives. Also the animator's stack frame."""

    __slots__ = ("_animator", "_label", "_spinner", "_start", "_styled", "__weakref__")

    def __init__(self, animator: "Animator", label: str, spinner: Spinner):
        self._animator = animator
        self._label = _oneline(label)
        self._spinner = spinner
        self._start = time.monotonic()
        self._styled = None

    @property
    def label(self) -> str:
        """The text currently shown next to the spinner (flattened to one line)."""
        return self._label

    def update(self, label: str) -> None:
        """Replace the status text. The next frame shows it — at most one
        spinner interval away, sooner if the redraw budget allows."""
        label = _oneline(label)
        if label == self._label:
            return
        self._label = label
        self._styled = None
        self._animator._nudge()

    def elapsed(self) -> float:
        """Seconds since the status block was entered."""
        return time.monotonic() - self._start

    # -- rendering (called on the animation thread)

    def _render(self, theme: Theme, caps: Caps) -> str:
        styled = self._styled
        if styled is None:
            styled = self._styled = theme.render(self._label, "app.muted", caps)
        return styled


class Animator:
    """Owns the transient one-line animation for a console.

    One daemon thread at most, shared by nested ``status()`` blocks: an inner
    block takes over the line and the outer label returns when it exits. The
    thread is torn down — and the line erased, and the cursor restored — by the
    *calling* thread.

    A caller that already holds ``console.lock`` (a foreground writer mid-frame)
    cannot deadlock against the animator, in either direction: ``_lock`` is
    never held across a ``console.lock`` acquire, and ``console.lock`` is an
    RLock, so the caller re-enters its own hold. ``console.lock`` is also the
    mutex the ``_hid_cursor`` / ``_painted`` flags live under, which is what
    makes claiming the cursor and hiding it — or capturing the claim and giving
    the cursor back — a single step against a status opening on another thread.

    The animator owns the current line while a status is running: foreground
    output written during a status block is overwritten by the next frame, so
    print permanent output after the block (or after :meth:`stop`).
    """

    def __init__(self, console: Console, theme: Theme, fps: float = 18.0) -> None:
        """``fps`` is a *ceiling* on repaints, not a target.

        The line is redrawn at ``min(fps, 1 / spinner.interval)``: with the
        default orbit cadence (0.085s) that is ~11.8 redraws a second no matter
        what ``fps`` says. Raising ``fps`` only buys latency for
        :meth:`StatusHandle.update`; lowering it below the spinner's own rate
        genuinely slows the animation down, which is the point on a slow link.
        """
        self.console = console
        self.theme = theme
        self.fps = max(1.0, min(float(fps), _MAX_FPS))
        self._lock = threading.Lock()
        self._wake = threading.Event()    # a repaint is wanted (update/nesting/stop)
        self._quit = threading.Event()    # teardown only; an update must not set it
        self._stack = []
        self._thread = None
        self._stopping = False
        self._gen = 0
        # The fps ceiling belongs to the animator, not to one run: a tight loop
        # of short status blocks would otherwise paint on entry every time.
        self._last_paint = 0.0
        # `_flags` is a leaf lock: it is only ever taken innermost, so it can
        # never take part in a cycle with console.lock or self._lock. The flags
        # it guards are *owned* by console.lock — every mutation below happens
        # while that lock is held, except the one bypass path that could not
        # get it.
        self._flags = threading.Lock()
        self._painted = False
        self._hid_cursor = False
        _LIVE.add(self)

    # -- introspection

    @property
    def enabled(self) -> bool:
        """False when the terminal cannot (or must not) animate."""
        return bool(self.console.caps.animation)

    @property
    def active(self) -> bool:
        """True while a status block is running."""
        with self._lock:
            return bool(self._stack)

    # -- public API

    @contextmanager
    def status(self, label: str, style: str = "orbit"):
        """Show a transient one-line spinner for the duration of the block.

        Yields a :class:`StatusHandle`. On every exit path the line is erased
        and the cursor restored — including ``KeyboardInterrupt``, which
        reaches this ``finally`` exactly like any other exception because the
        animator leaves the host's ``SIGINT`` handler alone. When
        ``caps.animation`` is False this is completely silent — nothing is
        written to stdout or stderr and no thread is created — because the only
        tty-free consumers are pipes and logs, where a spinner is either
        garbage bytes or a lie.
        """
        handle = StatusHandle(self, label, get_spinner(style))
        if not self.enabled:
            yield handle
            return
        self._push(handle)
        try:
            yield handle
        finally:
            self._pop(handle)

    def stop(self) -> None:
        """Stop everything and leave the line clean. Safe before any start,
        safe twice, safe from any thread; the animator stays reusable."""
        self._teardown()

    # -- stack management

    def _push(self, handle: StatusHandle) -> None:
        thread = None
        with self._lock:
            self._stack.append(handle)
            self._stopping = False
            if self._thread is None or not self._thread.is_alive():
                # A generation number retires any thread from a previous run, so
                # a straggler can never repaint a line this run already cleaned.
                self._gen += 1
                self._wake.clear()
                self._quit.clear()
                thread = threading.Thread(
                    target=self._run, args=(self._gen,), name="lume-motion", daemon=True
                )
                self._thread = thread
        if thread is None:
            self._nudge()          # nested: show the new top of the stack now
            return
        # The cursor is claimed *after* `_lock` is released and *under*
        # console.lock, which is both halves of one rule: `_lock` must never be
        # held across a console.lock acquire (a foreground writer holding
        # console.lock and then opening a status is the other half of that ABBA
        # pair), and console.lock is the mutex the flags live under, so the
        # claim and the hide are one step against a _restore racing us.
        # The acquire is bounded: entering a status must not park the caller
        # behind a slow foreground writer. Missing the claim costs a visible
        # cursor next to the spinner, which is the safe way to be wrong.
        if self.console.lock.acquire(timeout=_CLAIM_TIMEOUT):
            try:
                with self._flags:
                    # Only *this* animator's hide gets undone by this animator:
                    # if the app (or an outer animator) already hid the cursor,
                    # revealing it on our way out would leave the console
                    # disagreeing with the terminal. And there is nothing to
                    # claim on a stream that is not a terminal.
                    if not self._hid_cursor and self.console.caps.is_tty:
                        self._hid_cursor = not self.console._cursor_hidden
                self.console.hide_cursor()
            finally:
                self.console.lock.release()
        thread.start()

    def _pop(self, handle: StatusHandle) -> None:
        with self._lock:
            for i in range(len(self._stack) - 1, -1, -1):
                if self._stack[i] is handle:
                    del self._stack[i]
                    break
            remaining = bool(self._stack)
        if remaining:
            # An outer status is still running: repaint its label immediately.
            self._nudge()
            return
        self._teardown()

    def _nudge(self) -> None:
        if not self._wake.is_set():   # a token-by-token update() storm hits this
            self._wake.set()

    def _teardown(self) -> None:
        with self._lock:
            self._stack.clear()
            self._stopping = True
            t = self._thread
            self._thread = None
            # Both are leaf Events, and both are set inside the lock that a
            # _push clears them in. Set outside it, a _push landing in the gap
            # starts its run with `_quit` already set — and the frame loop's
            # rate-limit `wait` then returns instantly, for ever, so the thread
            # spins on a whole CPU core while the screen still looks right.
            self._quit.set()
            self._wake.set()
        if t is not None and t is not threading.current_thread():
            try:
                t.join(_JOIN_TIMEOUT)
            except RuntimeError:
                pass           # stopped in the window before the thread started
        # The flags are captured inside `_restore`, under console.lock — which
        # is also the lock a frame is painted under, after a `_stopping` check
        # this method has already made true. So there is no window between the
        # last paint and the capture: `_painted` is exact, and a status block
        # that never got to draw leaves the terminal untouched instead of
        # erasing a line it never owned.
        self._restore(t is not None)

    def _restore(self, had_thread: bool = False) -> None:
        """Erase the line and give back exactly what this animator took.

        Runs on the *stopping* thread. ``console.lock`` is an RLock, so a caller
        that already holds it (a foreground writer) re-enters instead of
        deadlocking, and this thread holds no other lock, so there is no cycle
        to fall into either.

        The two flags are captured *and* cleared under that same lock, which is
        what makes the undo atomic against a ``status()`` opening on another
        thread: the claim in ``_push`` happens under the same mutex, so the two
        cannot interleave and steal each other's restore.

        The wait is short on purpose — tearing a spinner down must feel instant
        even when another thread is sitting on the lock. It is a bound on
        *contention*, not on the terminal: if the sink itself has stalled (an
        undrained pty, XOFF), the frame thread is blocked inside ``write``
        holding the lock, and the raw bypass below blocks on the same stream.
        Nothing short of non-blocking IO can bound that, and a status spinner is
        not worth putting the terminal into O_NONBLOCK.
        """
        # Nothing owed and no thread that could still be inside a frame: this
        # is `stop()` before any start, or a second `stop()`, and it costs
        # nothing. With a thread we always take the lock — a frame that has just
        # passed its `_stopping` check is a few bytecodes from setting the flag,
        # and reading it without the lock would miss it.
        if not (had_thread or self._painted or self._hid_cursor):
            return
        got = self.console.lock.acquire(timeout=_RESTORE_TIMEOUT)
        try:
            with self._flags:
                painted, self._painted = self._painted, False
                hid, self._hid_cursor = self._hid_cursor, False
            if got:
                if painted:
                    self.console.write(CLEAR_LINE)
                    self.console.set_transient(False)
                if hid:
                    self.console.show_cursor()
            else:
                # Last resort: a stuck foreground writer must not cost the user
                # their cursor. One bypassing write, so it cannot be torn — and
                # the cursor half is tty-only, exactly as `Console.show_cursor`
                # is, so a bypass cannot put escapes into a pipe.
                text = ((CLEAR_LINE if painted else "")
                        + (SHOW_CURSOR if hid and self.console.caps.is_tty else ""))
                if text:
                    self._raw(text)
                # ...and the console's own bookkeeping has to follow the bypass,
                # or it goes on believing the cursor is hidden: the next status
                # block would then decline to hide it, and banner()/fade_in()
                # would skip their own restore, all on the strength of a flag
                # the terminal disagrees with.
                if hid:
                    self.console._cursor_hidden = False
                self.console._transient = False
        finally:
            if got:
                self.console.lock.release()

    def _raw(self, text: str) -> None:
        try:
            self.console.stream.write(text)
            self.console.stream.flush()
        except Exception:
            pass

    # -- the frame loop

    def _run(self, gen: int) -> None:
        console, theme = self.console, self.theme
        painters = {}
        tick = 0
        min_interval = 1.0 / self.fps
        while True:
            now = time.monotonic()
            with self._lock:
                if self._stopping or self._gen != gen or not self._stack:
                    return
                handle = self._stack[-1]
                # The rate limit belongs to the *animator*, not to this run:
                # seeded at zero per run, the first frame of every block painted
                # unconditionally, and a loop of short blocks painted thousands
                # of times a second at fps=18. Read and claim the slot in one
                # step, or two runs either side of the boundary both take it.
                budget = self._last_paint + min_interval - now
                if budget <= 0:
                    self._last_paint = now
            if budget > 0:
                # Rate limit. Waiting on `_quit` (not `_wake`) is what makes
                # `fps` real: an update() storm cannot spin this window away,
                # but stop() still lands immediately.
                self._quit.wait(budget)
                continue
            self._wake.clear()       # this frame answers every update so far
            spinner = handle._spinner
            painter = painters.get(spinner.name)
            if painter is None:
                painter = painters[spinner.name] = _Painter(spinner, theme, console.caps)
            line = self._compose(handle, painter, tick, console.caps)
            # A bounded acquire is what keeps the thread live: if a foreground
            # writer is holding the lock we drop the frame instead of blocking,
            # so stop() can always join us.
            if console.lock.acquire(timeout=_LOCK_TIMEOUT):
                try:
                    if not self._stopping and self._gen == gen:
                        # Claim the line *before* drawing on it: a stop() that
                        # captures the flag between the write and the claim
                        # would skip the final erase and leave a frame behind.
                        # `set_transient` says the same thing to lume.ansi's
                        # signal net, which erases the line before it restores
                        # the cursor — otherwise SIGTERM freezes a half-spun
                        # frame on screen with the shell prompt after it.
                        with self._flags:
                            self._painted = True
                        console.set_transient(True)
                        console.write(CLEAR_LINE, line)
                finally:
                    console.lock.release()
            tick += 1
            self._wake.wait(painter.interval)

    def _compose(self, handle: StatusHandle, painter: _Painter, tick: int, caps: Caps) -> str:
        parts = [painter.glyph(tick), " ", handle._render(self.theme, caps)]
        seconds = time.monotonic() - handle._start
        if seconds >= _ELAPSED_AFTER:
            parts.append("  " + self.theme.render(_fmt_elapsed(seconds), "app.dim", caps))
        # Stay one column short of the edge: a line that reaches the last cell
        # wraps on some terminals, and then \r no longer owns what we drew.
        return _cut("".join(parts), max(1, caps.width - 1), caps)


# A cursor left hidden outlives the process, so keep a registry and a final
# sweep. `lume.ansi` restores cursors at exit too; this hook also erases the
# half-drawn frame, which only the animator knows about. Module-global state.
_LIVE = weakref.WeakSet()


def _restore_all() -> None:
    """Erase every live animator's line and give the cursor back (atexit)."""
    for animator in list(_LIVE):
        try:
            animator.stop()
        except Exception:
            pass


atexit.register(_restore_all)


# --------------------------------------------------------------------- banner

# One source of truth for both wordmarks: a 6-row pixel grid per letter. Six
# rows is the floor at which every glyph keeps its counter — at five, `e` loses
# the bowl and reads as `c`. The unicode banner folds the rows in half with
# ▀▄█ (3 text rows); the ASCII banner prints all six.
_BITMAP = {
    "l": ("# ",
          "# ",
          "# ",
          "# ",
          "# ",
          "##"),
    "u": ("    ",
          "#  #",
          "#  #",
          "#  #",
          "#  #",
          " ## "),
    "m": ("     ",
          "#####",
          "# # #",
          "# # #",
          "# # #",
          "# # #"),
    "e": ("    ",
          " ## ",
          "#  #",     # <- the counter
          "####",
          "#   ",
          " ###"),
}
_ROWS = 6
_GAP = "  "
_HALF = {(True, True): "█", (True, False): "▀", (False, True): "▄", (False, False): " "}


def wordmark_lines(caps: Caps, word: str = "lume") -> list:
    """The wordmark as plain (uncoloured) rows of equal display width.

    Only the four letters of the name have art. Anything else raises
    ``ValueError``, naming what is missing: quietly dropping the characters it
    cannot draw made ``wordmark_lines(caps, "hello")`` answer with the art for
    ``ell`` and ``"LUME"`` answer with nothing at all, which is a different
    question answered without saying so. Case is not the caller's problem.
    """
    word = str(word).lower()
    missing = sorted(set(word) - set(_BITMAP))
    if missing:
        raise ValueError(
            "no wordmark art for %r; the alphabet is %r"
            % ("".join(missing), "".join(sorted(_BITMAP)))
        )
    letters = [_BITMAP[c] for c in word]
    if not letters:
        return []
    rows = [_GAP.join(l[r] for l in letters) for r in range(_ROWS)]
    if not caps.unicode:
        return rows
    folded = []
    for r in range(0, _ROWS, 2):
        top, bot = rows[r], rows[r + 1]
        folded.append("".join(_HALF[(top[i] == "#", bot[i] == "#")] for i in range(len(top))))
    return folded


def _colorize(row: str, colors: list, caps: Caps) -> str:
    """Paint one row column by column; spaces stay bare so the escape stream
    (and the scrollback) keeps only what it needs."""
    if not colors or not caps.color:
        return row
    out = []
    last = ""
    col = 0
    for ch in row:
        # Index the palette by *column*, not by character: `colors` is sized by
        # display width, so one wide glyph would run the two out of step and
        # walk off the end of the list.
        code = "" if ch == " " else fg(colors[min(col, len(colors) - 1)], caps)
        if code != last:
            if last:
                out.append(RESET)
            out.append(code)
            last = code
        out.append(ch)
        col += max(1, display_width(ch))
    if last:
        out.append(RESET)
    return "".join(out)


def _flat_accent(theme: Theme) -> tuple:
    """One accent colour, for a terminal that cannot draw a gradient.

    At 4-bit every stop collapses onto one of sixteen slots, so the ramp turns
    into three or four flat bands — measured on `aurora`, the widest of them was
    plain grey, i.e. the colour drained out of the middle of the word. Pick the
    stop that survives quantisation best: the one whose 16-colour slot keeps the
    most contrast against the theme's background.
    """
    bg = theme.background
    return max(_stops(theme), key=lambda c: contrast(_ANSI16[rgb_to_16(c)], bg))


def banner(console: Console, theme: Theme, subtitle: str = "", animate: bool = True) -> None:
    """Print the gradient ``lume`` wordmark.

    The reveal is a soft light sweeping left to right (unrevealed columns sit
    at the theme's dim colour, the leading edge glints) and finishes in about a
    quarter of a second. It degrades to a single static frame when ``animate``
    is False, when the terminal cannot animate, or when there is no colour to
    sweep. A terminal too narrow for the wordmark gets a one-line lockup
    instead; the two-column indent is the first thing dropped.
    """
    caps = console.caps
    rows = wordmark_lines(caps)
    width = display_width(rows[0]) if rows else 0
    if not rows or width > caps.width or caps.height <= len(rows):
        # Too narrow, or too short to hold the wordmark at all: on a terminal
        # with fewer rows than the wordmark has, the one-liner is what actually
        # fits on screen.
        _banner_compact(console, theme, subtitle, caps)
        return
    indent = "  " if width + 2 <= caps.width else ""

    if caps.color >= 8:
        colors = gradient(_stops(theme), width)
    elif caps.color:
        # `_Painter` already refuses to gradient at 4-bit ("quantises into a
        # flicker"); the wordmark obeys its own rule here rather than painting
        # a four-band stripe with a grey one through the middle of the word.
        colors = [_flat_accent(theme)] * width
    else:
        colors = []
    # The reveal redraws in place with `up(len(rows) - 1)`, so it needs the
    # whole wordmark to still be on screen; the guard above has already sent a
    # shorter terminal to the compact banner.
    moving = bool(animate and caps.animation and caps.color >= 8)

    with console.lock:
        if moving:
            already = console._cursor_hidden
            console.hide_cursor()
            try:
                _sweep(console, theme, rows, colors, indent, caps)
            finally:
                if not already:
                    console.show_cursor()
        else:
            console.write("\n".join(indent + _colorize(r, colors, caps) for r in rows))
        console.write("\n")
        sub = _subtitle(theme, subtitle, caps, caps.width - len(indent))
        if sub:
            console.write(indent, sub, "\n")
        console.write("\n")


def _subtitle(theme: Theme, subtitle: str, caps: Caps, width: int) -> str:
    """The subtitle, flattened and cut to the columns actually available."""
    text = _cut(_oneline(subtitle).strip(), max(0, width), caps)
    return theme.render(text, "app.dim", caps) if text else ""


def _banner_compact(console: Console, theme: Theme, subtitle: str, caps: Caps) -> None:
    """Narrow-terminal fallback: the name, spaced out, still on the gradient.

    Every line it writes fits in ``caps.width`` — `caps.width` really can be 2,
    and a wrapped banner is worse than no banner."""
    word = "l u m e"
    indent = "  " if len(word) + 2 <= caps.width else ""
    word = word[: max(0, caps.width - len(indent))].rstrip()
    if word:
        colors = gradient(_stops(theme), len(word)) if caps.color else []
        console.write(indent, _colorize(word, colors, caps), "\n")
    sub = _subtitle(theme, subtitle, caps, caps.width - len(indent))
    if sub:
        console.write(indent, sub, "\n")
    console.write("\n")


#: How long the whole reveal is allowed to take, however many frames it needs.
_SWEEP_SECONDS = 0.22


def _sweep(console: Console, theme: Theme, rows: list, colors: list, indent: str,
           caps: Caps) -> None:
    dim = theme["app.dim"].fg or (110, 110, 110)
    glint = theme["md.bold"].fg or (255, 255, 255)
    width = len(rows[0])
    soft = max(4, width // 3)
    steps = 10
    frames = []
    for step in range(steps + 1):
        # Stop the light half a softness short of clearing the last column:
        # past that every frame is the finished wordmark, and holding a still
        # image for a fifth of the reveal is just latency.
        edge = -soft + (width + 1.5 * soft) * (step / steps)
        if step == steps:
            frame_colors = colors
        else:
            frame_colors = []
            for i, base in enumerate(colors):
                lit = min(1.0, max(0.0, (edge - i) / soft + 1.0))
                c = blend(dim, base, lit)
                near = 1.0 - min(1.0, abs(i - edge) / soft)
                frame_colors.append(blend(c, glint, 0.45 * near * near))
        body = ("\n").join(
            indent + _colorize(r, frame_colors, caps) + CLEAR_TO_END for r in rows
        )
        # Two steps that *render* the same are a held still image, not an
        # animation. Below truecolor the sweep quantises and neighbouring steps
        # land on the same palette entry — at 8-bit, two of these eleven frames
        # were byte-identical, ~22ms of a 222ms reveal spent redrawing what was
        # already on screen. Drop the repeat instead of holding it.
        if not frames or body != frames[-1]:
            frames.append(body)
    dt = _SWEEP_SECONDS / max(1, len(frames) - 1)
    for i, body in enumerate(frames):
        console.write(("\r" + up(len(rows) - 1) if i else "") + body)
        if i < len(frames) - 1:
            time.sleep(dt)


# ----------------------------------------------------------------------- rule


def rule(width: int, theme: Theme, caps: Caps, label: str = "") -> str:
    """A horizontal rule, optionally with a label, exactly ``width`` columns wide.

    Returns a string with no trailing newline so the caller decides placement.
    """
    if width <= 0:
        return ""
    line = "─" if caps.unicode else "-"
    label = _oneline(label).strip()
    if not label or width < 4:
        return theme.render(line * width, "app.rule", caps)

    lead = 2 if width >= 10 else 0
    gap = 2 if lead else 1                       # spaces reserved around the label
    label = _cut(label, max(1, width - lead - gap), caps)
    tail = max(0, width - lead - gap - display_width(label))
    parts = []
    if lead:
        parts.append(theme.render(line * lead, "app.rule", caps))
        parts.append(" ")
    parts.append(theme.render(label, "app.muted", caps))
    parts.append(" ")
    if tail:
        parts.append(theme.render(line * tail, "app.rule", caps))
    return "".join(parts)


# -------------------------------------------------------------------- fade in


def fade_in(text: str, console: Console, theme: Theme, token: str = "app.text") -> None:
    """Print ``text`` as if the ink were settling: a short ramp from the dim
    colour up to the token colour, then a newline.

    Static (one plain print) whenever motion is unavailable, the theme has no
    colour for the token, a line is wide enough to wrap, or the block is taller
    than the terminal. The last two are the same reason: the redraw walks the
    cursor back up ``len(lines) - 1`` rows, and a wrapped line — or a block that
    has already scrolled — puts that origin somewhere else, so every frame lands
    on the wrong rows and scribbles over the scrollback.
    """
    caps = console.caps
    style = theme[token]
    lines = str(text).split("\n")
    final = "\n".join(style(l, caps) if l else "" for l in lines)

    wraps = any(display_width(l) >= caps.width for l in lines)
    taller = len(lines) >= caps.height
    blank = not str(text).strip()
    if blank or wraps or taller or not (caps.animation and caps.color >= 8 and style.fg):
        console.print(final)
        return

    dim = theme["app.dim"].fg or (110, 110, 110)
    steps = 6
    with console.lock:
        already = console._cursor_hidden
        console.hide_cursor()
        try:
            for step in range(steps + 1):
                if step == steps:
                    body = final
                else:
                    shade = Style(
                        fg=blend(dim, style.fg, (step + 1) / steps),
                        bold=style.bold, italic=style.italic,
                        underline=style.underline, dim=style.dim,
                    )
                    body = "\n".join(shade(l, caps) if l else "" for l in lines)
                body = "\n".join(l + CLEAR_TO_END for l in body.split("\n"))
                console.write(("\r" + up(len(lines) - 1) if step else "") + body)
                if step < steps:
                    time.sleep(0.028)
        finally:
            if not already:
                console.show_cursor()
        console.write("\n")
