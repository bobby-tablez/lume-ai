"""Tests for lume.motion.

Everything runs against an in-memory Console so the emitted byte stream itself
is the assertion target: no stray newline from the animator, the line cleared on
every exit path, the cursor shown again, the thread joined.

Several tests here exist to kill a specific sabotage — they are marked with a
``regression:`` note naming the defect they were watched to catch.

Two things beyond the bytes are asserted, because two real defects lived where
the bytes looked perfect:

* **Lock order.** ``console.lock`` is the outer lock and ``Animator._lock`` must
  never be held across an acquire of it. A wrapper records every acquisition, so
  the inversion is caught as an inversion, not as a hung suite.
* **Cost.** A frame loop that spins through its own rate-limit sleep paints
  exactly as often as it should, so the frame thread's own ``thread_time()`` is
  an assertion target: 99% of a core, at the right frame rate, showing the right
  thing, is still a bug.
"""

import io
import os
import re
import signal
import subprocess
import sys
import threading
import time
import pathlib
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lume.ansi import (
    CLEAR_LINE, CLEAR_TO_END, HIDE_CURSOR, SHOW_CURSOR, Caps, Console,
    display_width, strip_ansi,
)
from lume.theme import get_theme
from lume import ansi, motion


HIDE, SHOW = HIDE_CURSOR, SHOW_CURSOR
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeStream(io.StringIO):
    """StringIO that can claim to be a tty."""

    encoding = "utf-8"

    def __init__(self, tty=True):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def make_console(tty=True, animation=None, color=24, unicode=True, width=80):
    if animation is None:
        animation = tty
    caps = Caps(color=color, unicode=unicode, is_tty=tty, width=width, height=24,
                hyperlinks=tty, animation=animation)
    return Console(stream=FakeStream(tty), caps=caps)


def out(console):
    return console.stream.getvalue()


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def frames_of(text):
    """Reconstruct the successive states of the animated line."""
    return [f for f in text.split(CLEAR_LINE) if f]


INK = set("\u2588\u2580\u2584#")


def cell_colours(frame):
    """The truecolor value painted on each ink cell of one banner frame."""
    colours, current = [], None
    for token in re.split(r"(\x1b\[[0-9;?]*[A-Za-z])", frame):
        if token.startswith("\x1b"):
            match = re.fullmatch(r"\x1b\[38;2;(\d+);(\d+);(\d+)m", token)
            current = tuple(int(g) for g in match.groups()) if match else None
        else:
            colours.extend(current for ch in token if ch in INK)
    return colours


THEME = get_theme("aurora")

#: A spinner with a cadence far faster than any fps setting, so a test can tell
#: the fps ceiling apart from the spinner's own pace. `status(style=...)` takes
#: a Spinner as readily as a name.
FAST = motion.Spinner("test-fast", tuple("|/-\\"), tuple("|/-\\"), 0.005)

#: The opposite: a cadence so slow that anything which fails to *wake* the frame
#: thread — an update, a nested block, a stop — shows up as a whole second of
#: latency instead of hiding inside the next frame.
SLOW = motion.Spinner("test-slow", tuple("|/-\\"), tuple("|/-\\"), 1.0)


# --------------------------------------------------------------------- spinners


class SpinnerSetTests(unittest.TestCase):
    def test_every_spinner_frame_is_one_column(self):
        for name, spin in motion.SPINNERS.items():
            for label, frames in (("unicode", spin.unicode), ("ascii", spin.ascii)):
                self.assertTrue(frames, f"{name}/{label} empty")
                widths = {display_width(f) for f in frames}
                self.assertEqual(widths, {1}, f"{name}/{label} jitters: {widths}")

    def test_ascii_variants_are_ascii(self):
        for name, spin in motion.SPINNERS.items():
            joined = "".join(spin.ascii)
            self.assertTrue(joined.isascii(), f"{name} ascii variant is not ascii")
            self.assertGreaterEqual(len(spin.ascii), 3)

    def test_intervals_are_calm(self):
        for name, spin in motion.SPINNERS.items():
            self.assertGreaterEqual(spin.interval, 0.05, name)
            self.assertLessEqual(spin.interval, 0.3, name)

    def test_every_style_is_a_different_animation(self):
        """regression: orbit/dots/line were the same four ASCII frames, so three
        of eight advertised styles were indistinguishable without unicode."""
        def cycles(frames):
            # A rotation of the same loop is the same animation on screen.
            n = len(frames)
            return {"".join(frames[i:] + frames[:i]) for i in range(n)}

        for alphabet in ("unicode", "ascii"):
            seen = {}
            for name, spin in motion.SPINNERS.items():
                frames = getattr(spin, alphabet)
                for other, previous in seen.items():
                    self.assertFalse(
                        cycles(frames) & cycles(previous),
                        f"{name} and {other} are the same animation in {alphabet}")
                seen[name] = frames

    def test_the_ascii_bar_is_a_fill_not_a_breath(self):
        """regression: `pulse` and `bar` were both small→big→small in ASCII —
        the same animation to the eye, whatever the tuples say (and `bar`'s
        version passed through a blank frame on the way). A fill runs one way
        and starts over; a breath doubles back, and only one of the four may."""
        bar = motion.SPINNERS["bar"].ascii
        self.assertEqual(len(set(bar)), len(bar),
                         f"the ASCII bar doubles back like `pulse`: {bar}")
        pulse = motion.SPINNERS["pulse"].ascii
        self.assertLess(len(set(pulse)), len(pulse), "`pulse` stopped breathing")

    def test_the_catalogue_stays_small(self):
        self.assertLessEqual(len(motion.SPINNERS), 5, "spinner inventory is growing")
        self.assertIn(motion.DEFAULT_SPINNER, motion.SPINNERS)

    def test_frames_follow_caps_unicode(self):
        spin = motion.SPINNERS["orbit"]
        uni = Caps(unicode=True)
        asc = Caps(unicode=False)
        self.assertEqual(spin.frames(uni), spin.unicode)
        self.assertEqual(spin.frames(asc), spin.ascii)

    def test_unknown_name_falls_back_to_default(self):
        self.assertIs(motion.get_spinner("nope"), motion.SPINNERS[motion.DEFAULT_SPINNER])
        self.assertIs(motion.get_spinner(""), motion.SPINNERS[motion.DEFAULT_SPINNER])
        self.assertIs(motion.get_spinner(None), motion.SPINNERS[motion.DEFAULT_SPINNER])
        self.assertIs(motion.get_spinner("ORBIT"), motion.SPINNERS["orbit"])
        self.assertIs(motion.get_spinner(FAST), FAST)


# ----------------------------------------------------------------------- status


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.console = make_console()
        self.anim = motion.Animator(self.console, THEME, fps=30)

    def tearDown(self):
        self.anim.stop()

    def assert_clean(self, console=None, hides=1):
        text = out(console or self.console)
        self.assertNotIn("\n", text, "animator emitted a newline")
        self.assertEqual(text.count(HIDE), hides)
        self.assertEqual(text.count(SHOW), hides)
        self.assertTrue(text.endswith(CLEAR_LINE + SHOW) or text.endswith(SHOW),
                        f"line not cleaned: {text[-40:]!r}")
        # Nothing painted survives the last clear.
        self.assertEqual(strip_ansi(text.rsplit(CLEAR_LINE, 1)[-1]), "")

    def test_renders_frames_then_cleans_up(self):
        with self.anim.status("thinking") as handle:
            self.assertIsInstance(handle, motion.StatusHandle)
            wait_until(lambda: out(self.console).count(CLEAR_LINE) >= 3)
        self.assertGreaterEqual(out(self.console).count(CLEAR_LINE), 3)
        self.assertIn("thinking", strip_ansi(out(self.console)))
        self.assert_clean()

    def test_no_newline_ever(self):
        with self.anim.status("working"):
            wait_until(lambda: CLEAR_LINE in out(self.console))
        self.assertNotIn("\n", out(self.console))
        self.assertNotIn("\r\n", out(self.console))

    def test_a_newline_in_the_label_cannot_escape_the_line(self):
        """regression: `status("Reading file\\nand thinking")` emitted one \\n per
        frame, so CLEAR_LINE erased only the last row and the spinner scrolled
        into the scrollback forever."""
        with self.anim.status("Reading file\nand thinking\r!") as h:
            self.assertEqual(h.label, "Reading file and thinking !")
            wait_until(lambda: out(self.console).count(CLEAR_LINE) >= 3)
            h.update("second\nline\rhere\tand\x00a nul")
            self.assertTrue(wait_until(lambda: "second line here" in strip_ansi(out(self.console))))
        text = out(self.console)
        self.assertNotIn("\n", text)
        # CLEAR_LINE itself ends in \r; nothing else may.
        self.assertNotIn("\r", text.replace(CLEAR_LINE, ""))
        self.assertIn("Reading file and thinking !", strip_ansi(text))
        self.assert_clean()

    def test_thread_is_daemon_and_joined(self):
        with self.anim.status("x"):
            wait_until(lambda: self.anim._thread is not None)
            t = self.anim._thread
            self.assertTrue(t.daemon)
            self.assertTrue(t.is_alive())
        self.assertFalse(t.is_alive())
        self.assertIsNone(self.anim._thread)
        self.assertFalse(self.anim.active)

    def test_frames_carry_glyph_and_label(self):
        with self.anim.status("loading", style="orbit"):
            wait_until(lambda: out(self.console).count(CLEAR_LINE) >= 4)
        seen = frames_of(out(self.console))
        glyphs = {strip_ansi(f)[:1] for f in seen if strip_ansi(f)}
        self.assertTrue(glyphs <= set(motion.SPINNERS["orbit"].unicode), glyphs)
        self.assertGreater(len(glyphs), 1, "spinner never advanced")
        for f in seen:
            if strip_ansi(f):
                self.assertIn("loading", strip_ansi(f))

    def test_update_changes_the_label(self):
        with self.anim.status("first") as h:
            wait_until(lambda: "first" in strip_ansi(out(self.console)))
            h.update("second")
            self.assertEqual(h.label, "second")
            self.assertTrue(wait_until(lambda: "second" in strip_ansi(out(self.console))))
        text = strip_ansi(out(self.console))
        self.assertIn("first", text)
        self.assertIn("second", text)
        self.assert_clean()

    def test_update_to_same_label_is_a_noop(self):
        """regression: an update that changes nothing still woke the frame
        thread, so a stream that re-sends the same label repainted on every
        token for no visible difference."""
        with self.anim.status("same") as h:
            wait_until(lambda: CLEAR_LINE in out(self.console))
            self.anim._wake.clear()
            h.update("same")
            self.assertEqual(h.label, "same")
            self.assertFalse(self.anim._wake.is_set(),
                             "an identical label still asked for a repaint")

    def test_elapsed(self):
        with self.anim.status("x") as h:
            self.assertGreaterEqual(h.elapsed(), 0.0)
            time.sleep(0.05)
            first = h.elapsed()
            time.sleep(0.05)
            self.assertGreater(h.elapsed(), first)

    def test_exception_still_restores(self):
        with self.assertRaises(RuntimeError):
            with self.anim.status("boom"):
                wait_until(lambda: CLEAR_LINE in out(self.console))
                raise RuntimeError("boom")
        self.assert_clean()

    def test_keyboard_interrupt_still_restores(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.anim.status("interrupted"):
                wait_until(lambda: CLEAR_LINE in out(self.console))
                raise KeyboardInterrupt
        self.assert_clean()

    def test_stop_before_start_is_safe(self):
        self.anim.stop()
        self.anim.stop()
        self.assertEqual(out(self.console), "")

    def test_double_stop_and_stop_inside_block(self):
        with self.anim.status("x"):
            wait_until(lambda: CLEAR_LINE in out(self.console))
            self.anim.stop()
            self.anim.stop()
        self.anim.stop()
        self.assert_clean()
        self.assertEqual(out(self.console).count(SHOW), 1)

    def test_reusable(self):
        for label in ("one", "two", "three"):
            with self.anim.status(label):
                wait_until(lambda: label in strip_ansi(out(self.console)))
        text = strip_ansi(out(self.console))
        for label in ("one", "two", "three"):
            self.assertIn(label, text)
        self.assert_clean(hides=3)

    def test_nested_status_shares_one_thread(self):
        with self.anim.status("outer") as outer:
            wait_until(lambda: "outer" in strip_ansi(out(self.console)))
            first = self.anim._thread
            with self.anim.status("inner", style="pulse"):
                self.assertTrue(wait_until(lambda: "inner" in strip_ansi(out(self.console))))
                self.assertIs(self.anim._thread, first)
            # The outer label comes back after the inner block.
            marker = len(out(self.console))
            self.assertTrue(wait_until(lambda: "outer" in strip_ansi(out(self.console)[marker:])))
            self.assertEqual(outer.label, "outer")
        self.assert_clean()
        self.assertEqual(out(self.console).count(HIDE), 1)

    def test_nested_exception_unwinds_cleanly(self):
        with self.assertRaises(ValueError):
            with self.anim.status("outer"):
                with self.anim.status("inner"):
                    raise ValueError
        self.assert_clean()

    def test_out_of_order_exit_is_safe(self):
        outer = self.anim.status("outer")
        inner = self.anim.status("inner")
        outer.__enter__()
        inner.__enter__()
        wait_until(lambda: "inner" in strip_ansi(out(self.console)))
        outer.__exit__(None, None, None)   # pop the outer first
        inner.__exit__(None, None, None)
        self.assert_clean()
        self.assertFalse(self.anim.active)

    def test_line_never_reaches_the_terminal_edge(self):
        console = make_console(width=24)
        anim = motion.Animator(console, THEME)
        try:
            with anim.status("a very long status label that will not fit at all"):
                wait_until(lambda: out(console).count(CLEAR_LINE) >= 2)
        finally:
            anim.stop()
        for frame in frames_of(out(console)):
            self.assertLessEqual(display_width(strip_ansi(frame)), 23)

    def test_elapsed_clock_appears_only_after_a_while(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        try:
            with anim.status("slow") as h:
                wait_until(lambda: CLEAR_LINE in out(console))
                early = strip_ansi(out(console))
                self.assertNotRegex(early, r"\d+s")
                h._start -= motion._ELAPSED_AFTER + 3   # pretend it has been running
                mark = len(out(console))
                self.assertTrue(wait_until(lambda: re.search(r"\ds", strip_ansi(out(console)[mark:]))))
        finally:
            anim.stop()

    def test_ascii_terminal_uses_ascii_frames(self):
        console = make_console(unicode=False, width=24)
        anim = motion.Animator(console, THEME)
        try:
            # Long enough to be truncated: the ellipsis must be ASCII too.
            with anim.status("plain but rather long for this terminal"):
                wait_until(lambda: out(console).count(CLEAR_LINE) >= 3)
        finally:
            anim.stop()
        body = strip_ansi(out(console))
        self.assertTrue(body.isascii(), repr(body))

    def test_monochrome_emits_no_colour(self):
        console = make_console(color=0)
        anim = motion.Animator(console, THEME)
        try:
            with anim.status("mono"):
                wait_until(lambda: out(console).count(CLEAR_LINE) >= 2)
        finally:
            anim.stop()
        text = out(console)
        self.assertNotIn("38;2;", text)
        self.assertNotIn("38;5;", text)

    def test_four_bit_colour_does_not_flicker(self):
        console = make_console(color=4)
        anim = motion.Animator(console, THEME)
        try:
            with anim.status("legacy"):
                wait_until(lambda: out(console).count(CLEAR_LINE) >= 5)
        finally:
            anim.stop()
        codes = set(re.findall(r"\x1b\[(3\d|9\d)m", out(console)))
        self.assertLessEqual(len(codes), 2, codes)


# -------------------------------------------------------------------- redraw rate


class FrameRateTests(unittest.TestCase):
    def test_fps_caps_the_redraw_rate_under_an_update_storm(self):
        """regression: `_wake.wait()` returned instantly on every update, so a
        token-by-token stream repainted ~150 times a second at fps=18."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=10)
        t0 = time.monotonic()
        try:
            with anim.status("x", style=FAST) as h:
                i = 0
                while time.monotonic() - t0 < 0.8:
                    h.update(f"tok {i}")
                    i += 1
        finally:
            anim.stop()
        elapsed = time.monotonic() - t0
        frames = out(console).count(CLEAR_LINE)
        self.assertGreater(i, 500, "the update storm did not happen")
        self.assertLessEqual(frames, int(10 * elapsed) + 3,
                             f"{frames} frames in {elapsed:.2f}s at fps=10")
        self.assertGreaterEqual(frames, 2, "the line stopped animating altogether")

    def test_updates_still_reach_the_screen_promptly(self):
        console = make_console()
        anim = motion.Animator(console, THEME, fps=20)
        try:
            with anim.status("first", style=FAST) as h:
                wait_until(lambda: "first" in strip_ansi(out(console)))
                mark = len(out(console))
                t0 = time.monotonic()
                h.update("second")
                self.assertTrue(wait_until(
                    lambda: "second" in strip_ansi(out(console)[mark:]), 1.0))
                self.assertLess(time.monotonic() - t0, 0.3)
        finally:
            anim.stop()

    def test_fps_is_a_ceiling_not_a_target(self):
        """The documented meaning: the spinner's own cadence wins when it is
        slower, so fps=60 with the 0.085s orbit is still ~12 redraws a second."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        try:
            with anim.status("calm"):
                time.sleep(1.0)
        finally:
            anim.stop()
        frames = out(console).count(CLEAR_LINE)
        self.assertLess(frames, 20, f"{frames} frames/s from a 0.085s spinner")

    def test_a_low_fps_really_slows_the_spinner_down(self):
        console = make_console()
        anim = motion.Animator(console, THEME, fps=4)
        try:
            with anim.status("slow"):
                time.sleep(1.0)
        finally:
            anim.stop()
        frames = out(console).count(CLEAR_LINE)
        self.assertLessEqual(frames, 8, frames)
        self.assertGreaterEqual(frames, 2, frames)

    def test_fps_is_clamped_to_something_sane(self):
        """regression: the old assertion read the ceiling back out of the module,
        so `_MAX_FPS = 10000` passed it. A terminal cannot show 10000 frames a
        second; asking for them only burns a core."""
        console = make_console()
        self.assertEqual(motion.Animator(console, THEME, fps=0).fps, 1.0)
        self.assertEqual(motion.Animator(console, THEME, fps=-5).fps, 1.0)
        self.assertEqual(motion.Animator(console, THEME, fps=1e6).fps, motion._MAX_FPS)
        self.assertLessEqual(motion._MAX_FPS, 120.0, "the fps ceiling is not a ceiling")
        self.assertGreaterEqual(motion._MAX_FPS, 24.0)


# ----------------------------------------------------------- disabled terminals


class DisabledTests(unittest.TestCase):
    def test_non_tty_is_completely_silent(self):
        console = make_console(tty=False, animation=False)
        anim = motion.Animator(console, THEME)
        self.assertFalse(anim.enabled)
        with anim.status("thinking") as h:
            self.assertIsInstance(h, motion.StatusHandle)
            h.update("still thinking")
            self.assertGreaterEqual(h.elapsed(), 0.0)
            time.sleep(0.15)
        anim.stop()
        self.assertEqual(out(console), "")

    def test_no_thread_is_created(self):
        console = make_console(tty=False, animation=False)
        anim = motion.Animator(console, THEME)
        before = threading.active_count()
        with anim.status("x"):
            self.assertEqual(threading.active_count(), before)
            self.assertIsNone(anim._thread)
        self.assertEqual(out(console), "")

    def test_tty_with_motion_disabled(self):
        """LUME_NO_MOTION / TERM=dumb: a tty, but animation is off."""
        console = make_console(tty=True, animation=False)
        anim = motion.Animator(console, THEME)
        with anim.status("quiet"):
            time.sleep(0.12)
        anim.stop()
        self.assertEqual(out(console), "")

    def test_disabled_stop_is_a_noop(self):
        console = make_console(tty=False, animation=False)
        anim = motion.Animator(console, THEME)
        anim.stop()
        self.assertEqual(out(console), "")


# ------------------------------------------------------------------ concurrency


class ConcurrencyTests(unittest.TestCase):
    def test_stop_is_prompt_while_a_foreground_writer_holds_the_lock(self):
        """regression: the frame thread's lock acquire must stay bounded and the
        final erase must not wait seconds. With an unbounded acquire the frame
        thread is stuck, stop() burns the whole join timeout, and a user waits.
        The old test held the lock for 0.25s — inside the join timeout — so it
        could not see the difference."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        holding = threading.Event()
        release = threading.Event()

        def hog():
            with console.lock:
                holding.set()
                release.wait(5.0)

        with anim.status("busy"):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
            t = threading.Thread(target=hog)
            t.start()
            self.assertTrue(holding.wait(3.0))
            time.sleep(0.2)             # the frame thread meets the held lock
            t0 = time.monotonic()
            anim.stop()
            stopped = time.monotonic() - t0
        release.set()
        t.join(5.0)
        self.assertLess(stopped, 0.5, f"stop() blocked the caller for {stopped:.2f}s")
        self.assertTrue(out(console).endswith(SHOW), repr(out(console)[-40:]))
        self.assertNotIn("\n", out(console))

    def test_a_long_foreground_hold_never_deadlocks_a_status_block(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        done = threading.Event()

        def worker():
            with anim.status("busy"):
                wait_until(lambda: CLEAR_LINE in out(console))
                time.sleep(0.4)
            done.set()

        t = threading.Thread(target=worker)
        t.start()
        wait_until(lambda: CLEAR_LINE in out(console))
        with console.lock:            # far longer than the join timeout
            time.sleep(2.2)
            console.write("foreground")
        self.assertTrue(done.wait(2.0), "status block deadlocked")
        t.join(5.0)
        self.assertFalse(t.is_alive())
        anim.stop()
        self.assertIn("foreground", strip_ansi(out(console)))

    def test_stop_while_holding_the_console_lock(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        finished = threading.Event()

        def body():
            with console.lock:        # stop() called with the lock already held
                anim.stop()
            finished.set()

        with anim.status("x"):
            wait_until(lambda: CLEAR_LINE in out(console))
            t = threading.Thread(target=body)
            t.start()
            self.assertTrue(finished.wait(5.0), "stop() deadlocked against console.lock")
            t.join(5.0)
        self.assertTrue(out(console).endswith(SHOW))
        self.assertNotIn("\n", out(console))

    def test_frames_are_atomic_against_a_foreground_writer(self):
        """A foreground print must never land inside a frame's escape sequence."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        with anim.status("streaming"):
            wait_until(lambda: CLEAR_LINE in out(console))
            for i in range(20):
                console.write(f"<{i}>")
                time.sleep(0.005)
        anim.stop()
        text = out(console)
        for i in range(20):
            self.assertIn(f"<{i}>", text)
        self.assertNotIn("\x1b[38;2;<", text)

    def test_every_frame_is_written_under_the_console_lock(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        violations = []
        real = console.stream.write

        def guarded(text):
            if threading.current_thread().name == "lume-motion" and not console.lock._is_owned():
                violations.append(text)
            return real(text)

        console.stream.write = guarded
        try:
            with anim.status("locked"):
                wait_until(lambda: out(console).count(CLEAR_LINE) >= 4)
        finally:
            anim.stop()
            console.stream.write = real
        self.assertEqual(violations, [])

    def test_stop_from_another_thread(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        ctx = anim.status("x")
        ctx.__enter__()
        wait_until(lambda: CLEAR_LINE in out(console))
        t = threading.Thread(target=anim.stop)
        t.start()
        t.join(5.0)
        self.assertFalse(t.is_alive())
        ctx.__exit__(None, None, None)
        self.assertTrue(out(console).endswith(SHOW))

    def test_stop_racing_a_frame_always_ends_shown(self):
        for _ in range(120):
            console = make_console()
            anim = motion.Animator(console, THEME, fps=60)
            ctx = anim.status("race", style=FAST)
            ctx.__enter__()
            anim.stop()
            ctx.__exit__(None, None, None)
            text = out(console)
            self.assertEqual(text.count(HIDE), text.count(SHOW), repr(text[-60:]))
            if text:
                self.assertTrue(text.endswith(SHOW), repr(text[-60:]))

    def test_nested_status_from_many_threads(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        errors = []

        def worker():
            try:
                for _ in range(30):
                    with anim.status("outer"):
                        with anim.status("inner"):
                            pass
            except Exception as exc:          # pragma: no cover - a failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30.0)
        anim.stop()
        self.assertEqual(errors, [])
        self.assertEqual([t for t in threading.enumerate() if t.name == "lume-motion"], [])
        text = out(console)
        self.assertNotIn("\n", text)
        self.assertEqual(text.count(HIDE), text.count(SHOW))

    def test_a_retired_frame_thread_never_outlives_its_run(self):
        """regression: without the generation guard a straggler thread that was
        still composing when stop() ran keeps looping into the *next* run,
        painting alongside the live thread. Nothing in the old suite noticed."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        gate = threading.Event()
        blocked = threading.Event()
        trap = [True]
        writes = []
        real_glyph = motion._Painter.glyph
        real_write = console.stream.write

        def slow_glyph(painter, tick):
            # Compose runs outside console.lock, so a thread parked here holds
            # nothing and stop() proceeds exactly as it would in the wild.
            if trap[0] and threading.current_thread().name == "lume-motion":
                trap[0] = False
                blocked.set()
                gate.wait(5.0)
            return real_glyph(painter, tick)

        def spy(text):
            writes.append(threading.current_thread())
            return real_write(text)

        motion._Painter.glyph = slow_glyph
        console.stream.write = spy
        try:
            ctx = anim.status("first")
            ctx.__enter__()
            self.assertTrue(blocked.wait(3.0))
            straggler = anim._thread
            anim.stop()                      # join times out; straggler survives
            ctx.__exit__(None, None, None)
            with anim.status("second"):
                self.assertTrue(wait_until(lambda: "second" in strip_ansi(out(console))))
                mark = len(writes)
                gate.set()
                self.assertTrue(wait_until(lambda: not straggler.is_alive(), 2.0),
                                "the retired frame thread kept running into a later run")
                time.sleep(0.2)
                late = [t for t in writes[mark:] if t is straggler]
        finally:
            motion._Painter.glyph = real_glyph
            console.stream.write = real_write
            gate.set()
            anim.stop()
        self.assertEqual(late, [], "a retired frame thread repainted a later run")


# ---------------------------------------------------------------------- signals


class SignalTests(unittest.TestCase):
    """motion.py must not touch the host's signal handlers at all: `lume.app`
    installs its own SIGINT to cancel a reply, `lume.input` needs Ctrl-C at the
    prompt, and `lume.ansi` already owns the process-wide cursor net."""

    def setUp(self):
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("needs the main thread")

    def test_no_signal_handler_is_installed_around_a_live_status(self):
        console = make_console()          # a real tty: the case that used to install
        anim = motion.Animator(console, THEME)
        previous = signal.getsignal(signal.SIGINT)
        try:
            with anim.status("x"):
                wait_until(lambda: CLEAR_LINE in out(console))
                self.assertIs(signal.getsignal(signal.SIGINT), previous)
            self.assertIs(signal.getsignal(signal.SIGINT), previous)
        finally:
            anim.stop()
            signal.signal(signal.SIGINT, previous)

    def test_the_host_handler_survives_nested_blocks_and_off_thread_stops(self):
        """regression: motion captured its own handler as `previous` on the
        second install and reinstated it forever, so the app's Ctrl-C was gone
        after one out-of-order teardown."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        cancels = []

        def app_sigint(signum, frame):
            cancels.append(1)

        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, app_sigint)
        try:
            with anim.status("thinking"):
                wait_until(lambda: CLEAR_LINE in out(console))
                t = threading.Thread(target=anim.stop)   # cancel from a worker
                t.start()
                t.join(5.0)
                with anim.status("retrying"):
                    wait_until(lambda: "retrying" in strip_ansi(out(console)))
            self.assertIs(signal.getsignal(signal.SIGINT), app_sigint)
            signal.raise_signal(signal.SIGINT)
            time.sleep(0.05)
            self.assertEqual(cancels, [1], "Ctrl-C never reached the app")
        finally:
            anim.stop()
            signal.signal(signal.SIGINT, previous)

    def test_sigint_that_raises_leaves_the_line_clean(self):
        """With the default handler in place, Ctrl-C raises KeyboardInterrupt
        inside the block — the context manager's finally is the whole story."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            with self.assertRaises(KeyboardInterrupt):
                with anim.status("^C me"):
                    wait_until(lambda: CLEAR_LINE in out(console))
                    signal.raise_signal(signal.SIGINT)
                    time.sleep(0.2)     # in case delivery is deferred
        finally:
            anim.stop()
            signal.signal(signal.SIGINT, previous)
        text = out(console)
        self.assertTrue(text.endswith(SHOW), repr(text[-40:]))
        self.assertNotIn("\n", text)
        self.assertEqual(strip_ansi(text.rsplit(CLEAR_LINE, 1)[-1]), "")

    def test_the_animator_does_not_keep_itself_alive(self):
        """A handler bound into the signal module kept the animator, its thread
        and the console reachable forever."""
        import gc
        import weakref
        console = make_console()
        anim = motion.Animator(console, THEME)
        with anim.status("x"):
            wait_until(lambda: CLEAR_LINE in out(console))
        ref = weakref.ref(anim)
        del anim
        gc.collect()
        self.assertIsNone(ref(), "the animator outlived its last reference")


class ExitHookTests(unittest.TestCase):
    def test_the_atexit_sweep_is_registered(self):
        """regression: deleting `@atexit.register` left the whole suite green.

        Asserted against the source rather than by unregistering the live
        callback and counting: `atexit.unregister` is a no-op on some CPython
        builds, which failed this test on a perfectly registered sweep. What the
        regression actually was is a deleted line, and that is what this reads.
        """
        import ast
        import atexit

        tree = ast.parse(pathlib.Path(motion.__file__).read_text(encoding="utf-8"))
        registered = [
            node.args[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and node.args
            and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "atexit"
            and isinstance(node.args[0], ast.Name)
        ]
        self.assertIn("_restore_all", registered,
                      "motion._restore_all is not registered with atexit")
        self.assertGreaterEqual(atexit._ncallbacks(), 1, "nothing registered at all")

    def test_the_sweep_cleans_a_leaked_status(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        ctx = anim.status("leaked")
        ctx.__enter__()
        wait_until(lambda: CLEAR_LINE in out(console))
        motion._restore_all()           # what atexit runs
        text = out(console)
        self.assertTrue(text.endswith(SHOW))
        self.assertEqual(strip_ansi(text.rsplit(CLEAR_LINE, 1)[-1]), "")
        ctx.__exit__(None, None, None)

    def test_a_real_interpreter_exit_erases_the_line(self):
        """The end-to-end version: a child process that never leaves its status
        block must still exit with the line erased and the cursor back."""
        child = (
            "import sys, time\n"
            f"sys.path.insert(0, {ROOT!r})\n"
            "from lume.ansi import Caps, Console\n"
            "from lume.theme import get_theme\n"
            "from lume import motion\n"
            "caps = Caps(color=0, unicode=False, is_tty=True, width=40, height=24,\n"
            "            animation=True)\n"
            "con = Console(stream=sys.stdout, caps=caps)\n"
            "anim = motion.Animator(con, get_theme('aurora'))\n"
            "ctx = anim.status('leaking')\n"      # kept alive: a leaked block
            "ctx.__enter__()\n"
            "time.sleep(0.35)\n"
        )
        # Bytes, not text=True: universal-newline translation would rewrite the
        # \r inside every CLEAR_LINE into a newline and hide the whole point.
        proc = subprocess.run([sys.executable, "-c", child], capture_output=True,
                              timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        text = proc.stdout.decode("utf-8", "replace")
        self.assertIn(HIDE, text)
        self.assertEqual(text.count(HIDE), text.count(SHOW), repr(text[-60:]))
        self.assertTrue(text.endswith(SHOW), repr(text[-60:]))
        self.assertEqual(strip_ansi(text.rsplit(CLEAR_LINE, 1)[-1]), "",
                         "a painted frame survived the exit")
        self.assertNotIn("\n", text)


# ----------------------------------------------------------------------- cursor


class CursorOwnershipTests(unittest.TestCase):
    """regression: `_push` hid the cursor unconditionally and claimed ownership,
    so the spinner's exit revealed a cursor the *app* had hidden."""

    def test_status_gives_back_only_what_it_hid(self):
        console = make_console()
        console.hide_cursor()             # the app owns the hidden cursor
        anim = motion.Animator(console, THEME)
        with anim.status("x"):
            wait_until(lambda: CLEAR_LINE in out(console))
        anim.stop()
        text = out(console)
        self.assertEqual(text.count(HIDE), 1)
        self.assertEqual(text.count(SHOW), 0, "the animator revealed the app's cursor")
        self.assertTrue(console._cursor_hidden)

    def test_two_animators_on_one_console_hand_the_cursor_back_once(self):
        console = make_console()
        outer = motion.Animator(console, THEME)
        inner = motion.Animator(console, THEME)
        with outer.status("A"):
            wait_until(lambda: CLEAR_LINE in out(console))
            with inner.status("B"):
                wait_until(lambda: "B" in strip_ansi(out(console)))
            self.assertNotIn(SHOW, out(console), "the inner animator showed the cursor")
            wait_until(lambda: "A" in strip_ansi(out(console).rsplit(SHOW, 1)[-1]))
        outer.stop()
        inner.stop()
        text = out(console)
        self.assertEqual(text.count(HIDE), 1)
        self.assertEqual(text.count(SHOW), 1)
        self.assertTrue(text.endswith(SHOW))
        self.assertFalse(console._cursor_hidden)

    def test_banner_gives_back_only_what_it_hid(self):
        console = make_console()
        console.hide_cursor()
        motion.banner(console, THEME, animate=True)
        text = out(console)
        self.assertEqual(text.count(HIDE), 1)
        self.assertEqual(text.count(SHOW), 0)
        self.assertTrue(console._cursor_hidden)

    def test_fade_in_gives_back_only_what_it_hid(self):
        console = make_console()
        console.hide_cursor()
        motion.fade_in("settling", console, THEME, "app.accent")
        text = out(console)
        self.assertEqual(text.count(HIDE), 1)
        self.assertEqual(text.count(SHOW), 0)
        self.assertTrue(console._cursor_hidden)


# ----------------------------------------------------------------------- banner


class WordmarkTests(unittest.TestCase):
    def test_wordmark_rows_are_rectangular(self):
        for uni in (True, False):
            caps = Caps(unicode=uni, width=80)
            rows = motion.wordmark_lines(caps)
            self.assertGreaterEqual(len(rows), 3)
            widths = {display_width(r) for r in rows}
            self.assertEqual(len(widths), 1, widths)
            if not uni:
                self.assertTrue("".join(rows).isascii())

    def test_the_e_keeps_its_counter(self):
        """regression: the `e` was `.##.`/`####`/`#...`/`.###` — no bowl row, so
        the arc sat straight on the crossbar and the wordmark read `lumc`."""
        grid = motion._BITMAP["e"]
        bars = [i for i, row in enumerate(grid) if set(row) == {"#"}]
        self.assertTrue(bars, "the e has no crossbar")
        bowl = grid[bars[0] - 1]
        self.assertEqual(bowl[0], "#", f"no left stem above the crossbar: {bowl!r}")
        self.assertEqual(bowl[-1], "#", f"no right stem above the crossbar: {bowl!r}")
        self.assertIn(" ", bowl[1:-1], f"the bowl has no counter: {bowl!r}")
        # ...and the counter survives the half-block fold.
        folded = motion.wordmark_lines(Caps(unicode=True), "e")
        self.assertIn(" ", "".join(folded)[:len(folded[0]) * 2],
                      "the folded e is a solid block")

    def test_every_letter_is_distinguishable(self):
        for uni in (True, False):
            caps = Caps(unicode=uni, width=80)
            shapes = {c: tuple(motion.wordmark_lines(caps, c)) for c in "lume"}
            self.assertEqual(len(set(shapes.values())), 4, shapes)

    def test_letters_have_enough_rows_for_a_counter(self):
        self.assertGreaterEqual(motion._ROWS, 6)
        for name, grid in motion._BITMAP.items():
            self.assertEqual(len(grid), motion._ROWS, name)
            self.assertEqual(len({len(r) for r in grid}), 1, name)
            self.assertTrue(set("".join(grid)) <= {"#", " "}, name)

    def test_the_wordmark_still_fits_a_small_terminal(self):
        for uni in (True, False):
            width = display_width(motion.wordmark_lines(Caps(unicode=uni), "lume")[0])
            self.assertLessEqual(width, 24, "the wordmark outgrew a narrow tty")


class BannerTests(unittest.TestCase):
    def test_animated_banner_reveals_then_settles(self):
        console = make_console()
        motion.banner(console, THEME, subtitle="terminal chat", animate=True)
        text = out(console)
        self.assertEqual(text.count(HIDE), 1)
        self.assertEqual(text.count(SHOW), 1)
        self.assertGreater(text.count("\x1b[2A"), 3, "no multi-line redraw")
        body = strip_ansi(text)
        rows = motion.wordmark_lines(console.caps)
        self.assertGreater(body.count(rows[-1]), 3, "wordmark drawn only once")
        self.assertIn("terminal chat", body)
        self.assertTrue(body.endswith("\n\n"))

    def test_the_reveal_never_holds_a_still_image(self):
        """regression: the light cleared the last ink column three frames early,
        so ~20% of the reveal was the finished wordmark, redrawn and redrawn.

        Then, at 8-bit colour, two neighbouring steps quantised onto the same
        palette entry and the reveal held *that* frame instead — invisible to
        this test while it only ever ran at color=24, which is why the colour
        depth is a parameter now. Every depth that animates is checked, and at
        every width: `soft` is derived from the wordmark width, so the spacing
        of the steps changes with it."""
        for name in ("aurora", "solar", "ember", "mono"):
            theme = get_theme(name)
            for color in (24, 8):
                for width in (80, 40, 23, 21):
                    with self.subTest(theme=name, color=color, width=width):
                        console = make_console(color=color, width=width)
                        motion.banner(console, theme, animate=True)
                        frames = [f for f in out(console).split("\r") if f.strip()]
                        self.assertGreater(len(frames), 5)
                        self.assertEqual(len(frames), len(set(frames)),
                                         "the sweep repeated a frame")

    def test_the_sweep_never_writes_the_same_frame_twice(self):
        """The general case, independent of any theme's palette: when two steps
        render identically — which is what quantisation does to a ramp — the
        repeat is dropped rather than held on screen."""
        caps = Caps(color=0, unicode=True, is_tty=True, width=80, height=24,
                    hyperlinks=True, animation=True)
        console = Console(stream=FakeStream(), caps=caps)
        writes = []
        console.write = lambda *parts, flush=True: writes.append("".join(parts))
        rows = motion.wordmark_lines(caps)
        # No colour at all: every step of the sweep renders the same bytes.
        motion._sweep(console, THEME, rows, [], "  ", caps)
        self.assertEqual(len(writes), 1,
                         f"{len(writes)} identical frames written instead of one")

    def test_the_reveal_arrives_at_the_wordmark_only_at_the_end(self):
        console = make_console()
        motion.banner(console, THEME, animate=True)
        frames = [f for f in out(console).split("\r") if f.strip()]
        final = cell_colours(frames[-1])
        self.assertTrue(final and all(c is not None for c in final))

        def settled(frame):
            cells = cell_colours(frame)
            return (len(cells) == len(final)
                    and all(c is not None and max(abs(x - y) for x, y in zip(c, f)) <= 8
                            for c, f in zip(cells, final)))

        self.assertEqual([i for i, f in enumerate(frames) if settled(f)],
                         [len(frames) - 1],
                         "the sweep arrives at the finished wordmark before it ends")
        # ...and it still takes enough distinct steps to read as a sweep rather
        # than as two or three jumps.
        self.assertGreaterEqual(len(frames), 9, "the reveal lost most of its steps")

    def test_static_banner_draws_exactly_one_frame(self):
        console = make_console()
        motion.banner(console, THEME, subtitle="hi", animate=False)
        text = out(console)
        self.assertNotIn("\x1b[2A", text)
        self.assertNotIn(HIDE, text)
        rows = motion.wordmark_lines(console.caps)
        self.assertEqual(strip_ansi(text).count(rows[-1]), 1)

    def test_banner_is_static_on_a_non_tty(self):
        console = make_console(tty=False, animation=False)
        motion.banner(console, THEME, animate=True)
        text = out(console)
        self.assertNotIn("\x1b[2A", text)      # no redraw
        self.assertNotIn(HIDE, text)
        self.assertNotIn("\r", text)

    def test_banner_has_no_escapes_without_colour(self):
        console = make_console(tty=False, animation=False, color=0)
        motion.banner(console, THEME, subtitle="plain", animate=True)
        self.assertNotIn("\x1b", out(console))

    def test_banner_ascii_only_without_unicode(self):
        for width in (80, 24, 12, 4):
            console = make_console(unicode=False, animation=False, width=width)
            motion.banner(console, THEME, subtitle="a subtitle that will not fit",
                          animate=False)
            self.assertTrue(strip_ansi(out(console)).isascii(), repr(out(console)))

    def test_banner_finishes_quickly(self):
        console = make_console()
        t0 = time.monotonic()
        motion.banner(console, THEME, subtitle="x", animate=True)
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_banner_uses_the_theme_gradient(self):
        console = make_console()
        motion.banner(console, THEME, animate=False)
        colours = set(re.findall(r"38;2;(\d+;\d+;\d+)m", out(console)))
        self.assertGreater(len(colours), 4, "no gradient across the wordmark")

    def test_no_banner_line_ever_wraps(self):
        """regression: the compact fallback printed an un-truncated, un-indented
        7-column `l u m e` even at caps.width == 2, so it wrapped."""
        subtitle = "a subtitle that is far too long for a narrow terminal"
        for width in range(0, 40):
            for uni in (True, False):
                console = make_console(width=width, unicode=uni, animation=False)
                motion.banner(console, THEME, subtitle=subtitle, animate=False)
                for line in strip_ansi(out(console)).split("\n"):
                    self.assertLessEqual(display_width(line), width,
                                         f"width={width} unicode={uni}: {line!r}")

    def test_the_full_wordmark_is_used_whenever_it_fits(self):
        """The two-column indent is the first thing to go, not the wordmark."""
        for uni in (True, False):
            rows = motion.wordmark_lines(Caps(unicode=uni))
            exact = display_width(rows[0])
            console = make_console(width=exact, unicode=uni, animation=False)
            motion.banner(console, THEME, animate=False)
            body = strip_ansi(out(console))
            self.assertIn(rows[-1], body, f"fell back at exactly {exact} columns")
            console = make_console(width=exact - 1, unicode=uni, animation=False)
            motion.banner(console, THEME, animate=False)
            self.assertIn("l u m", strip_ansi(out(console)))

    def test_compact_banner_keeps_its_indent_when_there_is_room(self):
        console = make_console(width=16, animation=False)
        motion.banner(console, THEME, subtitle="chat", animate=False)
        lines = strip_ansi(out(console)).split("\n")
        self.assertEqual(lines[0], "  l u m e")
        self.assertEqual(lines[1], "  chat")

    def test_a_newline_in_the_subtitle_cannot_add_a_row(self):
        console = make_console(animation=False)
        motion.banner(console, THEME, subtitle="one\ntwo", animate=False)
        body = strip_ansi(out(console))
        self.assertIn("one two", body)
        rows = motion.wordmark_lines(console.caps)
        self.assertEqual(body.rstrip("\n").count("\n"), len(rows))


# ------------------------------------------------------------------------- rule


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.caps = Caps(color=24, unicode=True, is_tty=True, width=80)
        self.ascii = Caps(color=0, unicode=False, is_tty=False, width=80)

    def test_exact_width(self):
        for width in range(0, 81):
            for label in ("", "history", "セッション", "an extremely long label that cannot fit"):
                for caps in (self.caps, self.ascii):
                    r = motion.rule(width, THEME, caps, label)
                    self.assertEqual(display_width(r), max(0, width), (width, label, repr(r)))

    def test_no_newlines(self):
        r = motion.rule(40, THEME, self.caps, "multi\nline\rreturn")
        self.assertNotIn("\n", r)
        self.assertNotIn("\r", r)
        self.assertIn("multi line return", strip_ansi(r))

    def test_zero_and_negative_width(self):
        self.assertEqual(motion.rule(0, THEME, self.caps), "")
        self.assertEqual(motion.rule(-5, THEME, self.caps, "x"), "")

    def test_ascii_fallback(self):
        r = motion.rule(20, THEME, self.ascii, "log")
        self.assertTrue(strip_ansi(r).isascii())
        self.assertIn("-", strip_ansi(r))
        long_label = motion.rule(20, THEME, self.ascii, "a label far too long to fit")
        self.assertTrue(strip_ansi(long_label).isascii(), repr(long_label))

    def test_label_is_present_and_padded(self):
        r = strip_ansi(motion.rule(40, THEME, self.caps, "session"))
        self.assertIn(" session ", r)
        self.assertTrue(r.startswith("──"))
        self.assertTrue(r.endswith("─"))


# ---------------------------------------------------------------------- fade_in


class FadeInTests(unittest.TestCase):
    def test_static_when_motion_is_off(self):
        console = make_console(tty=False, animation=False)
        motion.fade_in("hello there", console, THEME)
        self.assertEqual(out(console),
                         THEME.render("hello there", "app.text", console.caps) + "\n")
        self.assertNotIn("\r", out(console))
        self.assertNotIn(HIDE, out(console))

    def test_animated_ends_on_the_final_colour(self):
        console = make_console()
        motion.fade_in("hello there", console, THEME, "app.accent")
        text = out(console)
        self.assertEqual(text.count(HIDE), 1)
        self.assertEqual(text.count(SHOW), 1)
        self.assertEqual(text.count("\n"), 1)
        self.assertTrue(text.endswith("\n"))
        final = THEME.render("hello there", "app.accent", console.caps)
        self.assertIn(final, text)
        self.assertGreater(len(re.findall(r"hello there", text)), 3, "no fade steps")

    def test_multiline_uses_a_block_redraw(self):
        console = make_console()
        motion.fade_in("one\ntwo\nthree", console, THEME)
        text = out(console)
        self.assertIn("\x1b[2A", text)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(text.count(SHOW), 1)
        settled = strip_ansi(text).rsplit("\r", 1)[-1]
        self.assertEqual(settled.rstrip("\n").split("\n"), ["one", "two", "three"])

    def test_wide_line_is_printed_statically(self):
        console = make_console(width=20)
        text_in = "x" * 40
        motion.fade_in(text_in, console, THEME)
        text = out(console)
        self.assertNotIn(HIDE, text)
        self.assertEqual(strip_ansi(text), text_in + "\n")

    def test_empty_text(self):
        console = make_console()
        motion.fade_in("", console, THEME)
        self.assertEqual(strip_ansi(out(console)), "\n")

    def test_monochrome_is_static(self):
        console = make_console(color=0)
        motion.fade_in("plain", console, THEME)
        self.assertEqual(out(console), "plain\n")



# ------------------------------------------------------------------ lock order


class _Tracked:
    """A lock wrapper that remembers, per thread, that it is held."""

    def __init__(self, lock, depth):
        self._lock, self._depth = lock, depth

    def acquire(self, *args, **kwargs):
        got = self._lock.acquire(*args, **kwargs)
        if got:
            self._depth.n = getattr(self._depth, "n", 0) + 1
        return got

    def release(self):
        self._depth.n -= 1
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def __getattr__(self, name):
        return getattr(self._lock, name)


class _Ordered:
    """console.lock, wrapped so that taking it while `_lock` is held is recorded."""

    def __init__(self, lock, depth, violations):
        self._lock, self._depth, self._violations = lock, depth, violations

    def acquire(self, *args, **kwargs):
        if getattr(self._depth, "n", 0):
            self._violations.append(threading.current_thread().name)
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def __getattr__(self, name):
        return getattr(self._lock, name)


def stop_soon(anim, timeout=3.0):
    """Tear an animator down without ever blocking the test thread: these tests
    exist to catch a deadlock, and the cleanup must not join it."""
    t = threading.Thread(target=anim.stop, daemon=True)
    t.start()
    t.join(timeout)


class LockOrderTests(unittest.TestCase):
    """The ABBA pair. `console.lock` is the outer lock — a foreground writer may
    already hold it when it calls in here — so `Animator._lock` must never be
    held across an acquire of it."""

    def test_the_animator_lock_is_never_held_across_the_console_lock(self):
        """regression: `_push` took `_lock` and then blocked, unboundedly, on
        `console.lock`; every foreground caller takes the two the other way
        round. That is a real deadlock, and both the SPEC and this class's own
        docstring promised it could not happen."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        depth = threading.local()
        violations = []
        anim._lock = _Tracked(anim._lock, depth)
        console.lock = _Ordered(console.lock, depth, violations)
        stop = threading.Event()

        def churn():
            while not stop.is_set():
                with anim.status("outer", style=FAST) as h:
                    h.update("tok")
                    with anim.status("inner", style=FAST):
                        pass
                anim.stop()

        def writer():
            while not stop.is_set():
                with console.batch():
                    console.write("x")
                time.sleep(0.001)

        threads = [threading.Thread(target=churn, daemon=True) for _ in range(3)]
        threads.append(threading.Thread(target=writer, daemon=True))
        for t in threads:
            t.start()
        time.sleep(0.8)
        stop.set()
        for t in threads:
            t.join(3.0)
        stop_soon(anim)
        self.assertEqual(violations, [],
                         "console.lock was taken while the animator's own lock was held")

    def test_a_console_lock_holder_can_open_a_status_from_another_thread(self):
        """regression: T1 holds console.lock and opens a status; T2 is opening
        one at the same time. T2 held `_lock` waiting for console.lock, T1 waited
        for `_lock`: both threads stuck for ever."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        done = []
        holding = threading.Event()

        def writer():
            with console.batch():              # a slow foreground render
                console.write("rendering a frame...")
                holding.set()
                time.sleep(0.3)
                with anim.status("saving", style=FAST):
                    pass
            done.append("writer")

        def other():
            time.sleep(0.05)
            with anim.status("thinking", style=FAST):
                time.sleep(0.02)
            done.append("other")

        w = threading.Thread(target=writer, daemon=True)
        o = threading.Thread(target=other, daemon=True)
        w.start()
        self.assertTrue(holding.wait(3.0))
        o.start()
        w.join(6.0)
        o.join(6.0)
        stop_soon(anim)
        self.assertEqual(sorted(done), ["other", "writer"],
                         "deadlock between a console.lock holder and a status block")

    def test_a_console_lock_holder_can_stop_the_animator(self):
        """The same pair, reached through the documented-safe `stop()`."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        done = []
        holding = threading.Event()

        def writer():
            with console.batch():
                console.write("rendering a frame...")
                holding.set()
                time.sleep(0.3)
                anim.stop()
            done.append("writer")

        def other():
            time.sleep(0.05)
            with anim.status("thinking", style=FAST):
                time.sleep(0.02)
            done.append("other")

        w = threading.Thread(target=writer, daemon=True)
        o = threading.Thread(target=other, daemon=True)
        w.start()
        self.assertTrue(holding.wait(3.0))
        o.start()
        w.join(6.0)
        o.join(6.0)
        stop_soon(anim)
        self.assertEqual(sorted(done), ["other", "writer"],
                         "deadlock between a console.lock holder and stop()")

    def test_stop_is_prompt_while_another_thread_is_entering_a_status(self):
        """regression: with a writer holding console.lock and a second thread
        inside `status()`, `stop()` took 4.7 seconds — the entering thread was
        sitting on `_lock` waiting for the console."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        holding, release = threading.Event(), threading.Event()

        def hog():
            with console.lock:
                holding.set()
                release.wait(5.0)

        h = threading.Thread(target=hog, daemon=True)
        h.start()
        self.assertTrue(holding.wait(3.0))
        entering = threading.Thread(
            target=lambda: anim.status("blocked", style=FAST).__enter__(),
            daemon=True)
        entering.start()
        time.sleep(0.3)
        t0 = time.monotonic()
        stopper = threading.Thread(target=anim.stop, daemon=True)
        stopper.start()
        stopper.join(4.0)
        stopped = time.monotonic() - t0
        release.set()
        h.join(5.0)
        entering.join(5.0)
        stop_soon(anim)
        self.assertFalse(stopper.is_alive(), "stop() never returned")
        self.assertLess(stopped, 0.5, f"stop() took {stopped:.2f}s")

    def test_entering_a_status_does_not_park_the_caller_behind_a_writer(self):
        """The cursor claim takes console.lock, so it is bounded: a foreground
        writer that holds the console for a second must not hold up the caller
        of `status()` for a second. Losing the claim costs a visible cursor next
        to the spinner, which is the safe way to be wrong."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        holding, release = threading.Event(), threading.Event()

        def hog():
            with console.lock:
                holding.set()
                release.wait(5.0)

        h = threading.Thread(target=hog, daemon=True)
        h.start()
        self.assertTrue(holding.wait(3.0))
        t0 = time.monotonic()
        ctx = anim.status("waiting", style=FAST)
        ctx.__enter__()
        entered = time.monotonic() - t0
        release.set()
        h.join(5.0)
        ctx.__exit__(None, None, None)
        stop_soon(anim)
        self.assertLess(entered, 0.5, f"status() blocked its caller for {entered:.2f}s")


class _FlagSpy:
    """`Animator._flags`, wrapped to record whether console.lock was held."""

    def __init__(self, lock, console, seen):
        self._lock, self._console, self._seen = lock, console, seen

    def acquire(self, *args, **kwargs):
        got = self._lock.acquire(*args, **kwargs)
        if got:
            self._seen.append(self._console.lock._is_owned())
        return got

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class CursorFlagTests(unittest.TestCase):
    def test_the_ownership_flags_are_only_touched_under_the_console_lock(self):
        """regression: `_teardown` captured the cursor/paint flags under `_lock`,
        before it ever reached the console. `console.lock` is what serialises
        the claim in `_push` against the capture in `_restore`; capturing it
        anywhere else reopens the race that leaves one run's cursor hidden and
        another run's restore already spent."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        seen = []
        anim._flags = _FlagSpy(anim._flags, console, seen)
        with anim.status("x", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
        anim.stop()
        self.assertTrue(seen, "the flags were never touched at all")
        self.assertTrue(all(seen),
                        "a cursor/paint flag was read or written outside console.lock")

    def test_the_cursor_is_claimed_in_the_same_breath_as_it_is_hidden(self):
        """The claim and the HIDE must be one step under console.lock: a
        `_restore` landing between them gives back a cursor nobody hid yet."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        claimed_at_acquire = []
        real = console.lock

        class Spy:
            def acquire(self, *args, **kwargs):
                got = real.acquire(*args, **kwargs)
                if got:
                    claimed_at_acquire.append(anim._hid_cursor)
                return got

            def release(self):
                real.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *exc):
                self.release()

            def __getattr__(self, name):
                return getattr(real, name)

        console.lock = Spy()
        with anim.status("x", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
        anim.stop()
        console.lock = real
        self.assertTrue(claimed_at_acquire)
        self.assertFalse(claimed_at_acquire[0],
                         "the cursor was claimed before console.lock was held")
        self.assertIn(True, claimed_at_acquire, "the cursor was never claimed")


# ------------------------------------------------------------- what it costs


class _MeasuredAnimator(motion.Animator):
    """An animator that reports how much CPU its frame thread actually burned.

    Frame *rate* is not the whole story: a loop that spins through its own
    rate-limit sleep paints exactly as often as it should while eating a core,
    so the only way to see it is to ask the thread itself."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu = 0.0

    def _run(self, gen):
        start = time.thread_time()
        try:
            super()._run(gen)
        finally:
            self.cpu += time.thread_time() - start


class FrameCostTests(unittest.TestCase):
    def test_the_frame_thread_costs_nothing_under_an_update_storm(self):
        """regression: waiting on `_wake` instead of `_quit` for the rate limit
        cost 45% of a core, and a stale `_quit` cost 99% — both with the frame
        rate, the frames and the screen all still exactly right. Nothing in the
        suite asserted cost, so both were invisible."""
        console = make_console()
        anim = _MeasuredAnimator(console, THEME, fps=18)
        t0 = time.monotonic()
        updates = 0
        try:
            with anim.status("x", style=FAST) as h:
                while time.monotonic() - t0 < 1.0:
                    updates += 1
                    h.update(f"tok {updates}")
        finally:
            anim.stop()
        wall = time.monotonic() - t0
        self.assertGreater(updates, 500, "the update storm did not happen")
        self.assertGreater(out(console).count(CLEAR_LINE), 4, "nothing was painted")
        self.assertLess(anim.cpu, 0.10 * wall,
                        f"the frame thread burned {anim.cpu:.2f}s of CPU in {wall:.2f}s")

    def test_an_idle_status_block_costs_nothing(self):
        console = make_console()
        anim = _MeasuredAnimator(console, THEME, fps=18)
        with anim.status("waiting"):
            time.sleep(0.8)
        anim.stop()
        self.assertLess(anim.cpu, 0.08, f"an idle spinner burned {anim.cpu:.3f}s of CPU")

    def test_a_stop_racing_a_new_status_never_leaves_the_run_flagged_to_quit(self):
        """regression: `_teardown` set `_quit` *outside* `_lock` while `_push`
        cleared it inside. A thread preempted in that gap left the next run with
        `_quit` permanently set, so `self._quit.wait(budget)` — the fps rate
        limit — returned instantly for ever: a whole core burned at the correct
        frame rate, painting the correct thing, so nothing on screen looked
        wrong. Reachable through `status()` and `stop()` alone; the pause is
        staged here so the window does not depend on the scheduler.

        Setting both Events inside `_lock` is what closes it — they are leaf
        Events, so holding the lock across them costs nothing and risks nothing.
        """
        console = make_console()
        anim = _MeasuredAnimator(console, THEME, fps=18)
        real_set = anim._quit.set
        gate = threading.Event()

        def delayed_set():
            gate.wait(3.0)          # the scheduler, pausing _teardown right here
            real_set()

        anim._quit.set = delayed_set
        stopper = threading.Thread(target=anim.stop, daemon=True)
        stopper.start()
        time.sleep(0.05)
        anim._quit.set = real_set

        opened = threading.Event()

        def opener():
            with anim.status("victim", style=FAST):
                opened.set()
                time.sleep(0.6)

        runner = threading.Thread(target=opener, daemon=True)
        runner.start()
        time.sleep(0.05)
        gate.set()                  # ...and the stale set lands on the new run
        self.assertTrue(opened.wait(3.0), "the new status never opened")
        flagged = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.4:
            if anim._quit.is_set() and anim._stack and not anim._stopping:
                flagged.append(1)
            time.sleep(0.005)
        runner.join(3.0)
        stopper.join(3.0)
        anim.stop()
        self.assertEqual(flagged, [],
                         "a live run was left flagged to quit: its rate limit is gone")
        self.assertLess(anim.cpu, 0.1,
                        f"the frame thread burned {anim.cpu:.2f}s of CPU on one status")

    def test_the_fps_ceiling_spans_status_blocks(self):
        """regression: `_run` seeded its rate limit at 0.0, so the *first* frame
        of every block painted unconditionally. A tight loop of short blocks
        produced 13,000 repaints a second at fps=18 — the ceiling was per run,
        not per animator."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=10)
        t0 = time.monotonic()
        blocks = 0
        while time.monotonic() - t0 < 0.6:
            with anim.status("x", style=FAST):
                pass
            blocks += 1
        anim.stop()
        elapsed = time.monotonic() - t0
        # Count what was *painted*: every block also emits its own final erase.
        frames = [f for f in out(console).split(CLEAR_LINE) if strip_ansi(f).strip()]
        self.assertGreater(blocks, 50, "the loop of short blocks did not happen")
        self.assertLessEqual(len(frames), int(10 * elapsed) + 3,
                             f"{len(frames)} repaints from {blocks} blocks at fps=10")

    def test_a_block_that_never_painted_leaves_the_line_alone(self):
        """A status block that came and went inside one frame interval never
        drew anything, so it must not erase the line either — the user's own
        half-written line is not ours to clear."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=1)
        with anim.status("first", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
        mark = len(out(console))
        with anim.status("second", style=FAST):     # inside the fps budget
            pass
        anim.stop()
        tail = out(console)[mark:]
        self.assertNotIn(CLEAR_LINE, tail, f"an unpainted block erased the line: {tail!r}")
        self.assertEqual(strip_ansi(tail), "")


class WakeTests(unittest.TestCase):
    def test_the_wake_flag_is_cleared_once_the_frame_answers_it(self):
        """A frame answers every update received before it was composed, and says
        so — otherwise the loop can never idle between frames."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=30)
        try:
            with anim.status("a", style=FAST) as h:
                self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
                h.update("b")
                self.assertTrue(wait_until(lambda: "b" in strip_ansi(out(console))))
                self.assertTrue(wait_until(lambda: not anim._wake.is_set(), 1.0),
                                "the wake flag is never cleared")
        finally:
            anim.stop()

    def test_an_update_wakes_the_frame_thread_immediately(self):
        """regression: with `_nudge` a no-op an update waits for the spinner's
        own cadence. `FAST` hid that — at 0.005s a lost wake is invisible — so
        this runs at a one-second cadence, where it is the whole latency."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        try:
            with anim.status("first", style=SLOW) as h:
                self.assertTrue(wait_until(lambda: "first" in strip_ansi(out(console))))
                mark = len(out(console))
                t0 = time.monotonic()
                h.update("second")
                self.assertTrue(
                    wait_until(lambda: "second" in strip_ansi(out(console)[mark:]), 0.8),
                    "the update never reached the screen")
                self.assertLess(time.monotonic() - t0, 0.3)
        finally:
            anim.stop()

    def test_an_update_that_lands_while_a_frame_is_composed_is_not_lost(self):
        """regression: `_wake.clear()` has to happen *before* the frame is
        composed. Cleared afterwards, an update that arrives while the frame is
        being built is thrown away with it, and the label sits stale until
        something else happens to wake the loop."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        handle = []
        fired = []
        real_glyph = motion._Painter.glyph

        def glyph(painter, tick):
            if not fired and handle and threading.current_thread().name == "lume-motion":
                fired.append(True)
                handle[0].update("second")     # lands mid-compose
            return real_glyph(painter, tick)

        motion._Painter.glyph = glyph
        try:
            with anim.status("first", style=SLOW) as h:
                handle.append(h)
                self.assertTrue(wait_until(lambda: fired, 2.0), "no frame was composed")
                t0 = time.monotonic()
                self.assertTrue(
                    wait_until(lambda: "second" in strip_ansi(out(console)), 0.8),
                    "an update composed away was never repainted")
                self.assertLess(time.monotonic() - t0, 0.5)
        finally:
            motion._Painter.glyph = real_glyph
            anim.stop()

    def test_a_nested_block_takes_over_the_line_immediately(self):
        """The inner label must not wait for the outer spinner's next tick."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        try:
            with anim.status("outer", style=SLOW):
                self.assertTrue(wait_until(lambda: "outer" in strip_ansi(out(console))))
                mark = len(out(console))
                t0 = time.monotonic()
                with anim.status("inner", style=SLOW):
                    self.assertTrue(
                        wait_until(lambda: "inner" in strip_ansi(out(console)[mark:]), 0.8),
                        "the nested label never appeared")
                    self.assertLess(time.monotonic() - t0, 0.3)
        finally:
            anim.stop()


class TeardownTests(unittest.TestCase):
    def test_stop_is_immediate_while_the_frame_thread_sleeps_between_frames(self):
        """`_wake` is what ends the between-frames sleep; `stop()` must not wait
        out a slow spinner's cadence."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        with anim.status("x", style=SLOW):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
            time.sleep(0.05)                   # the thread is now in its sleep
            t0 = time.monotonic()
            anim.stop()
            stopped = time.monotonic() - t0
        self.assertLess(stopped, 0.3, f"stop() waited {stopped:.2f}s for the spinner")

    def test_stop_is_immediate_while_the_frame_thread_waits_out_the_fps_budget(self):
        """...and `_quit` is what ends the rate-limit wait, which at a low fps is
        the longer of the two."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=1)
        with anim.status("x", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
            time.sleep(0.05)                   # the thread is now in the budget wait
            t0 = time.monotonic()
            anim.stop()
            stopped = time.monotonic() - t0
        self.assertLess(stopped, 0.4, f"stop() waited {stopped:.2f}s for the fps budget")

    def test_a_frame_in_flight_when_stop_lands_is_never_painted(self):
        """regression: `_stopping` is the only thing that can stop a frame that
        was already composed when `stop()` ran — the generation guard cannot,
        because `stop()` does not open a new generation. And the wait for that
        frame is bounded: a parked frame thread must not hold the caller for
        longer than the join timeout."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        blocked, gate = threading.Event(), threading.Event()
        trap = [True]
        real_glyph = motion._Painter.glyph

        def slow_glyph(painter, tick):
            if trap[0] and threading.current_thread().name == "lume-motion":
                trap[0] = False
                blocked.set()
                gate.wait(5.0)
            return real_glyph(painter, tick)

        motion._Painter.glyph = slow_glyph
        ctx = anim.status("x", style=FAST)
        try:
            ctx.__enter__()
            self.assertTrue(blocked.wait(3.0), "the frame thread never composed")
            t0 = time.monotonic()
            anim.stop()
            stopped = time.monotonic() - t0
            mark = len(out(console))
            gate.set()
            time.sleep(0.25)
            late = out(console)[mark:]
        finally:
            motion._Painter.glyph = real_glyph
            gate.set()
            ctx.__exit__(None, None, None)
            anim.stop()
        self.assertLess(stopped, 1.5, f"stop() blocked for {stopped:.2f}s on a parked frame")
        self.assertEqual(late, "", f"a frame landed after stop(): {late!r}")

    def test_the_frame_thread_is_dead_before_the_block_returns(self):
        """The teardown *joins*: a returned status block must not leave a thread
        that can still write to the terminal."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        blocked = threading.Event()
        trap = [True]
        real_glyph = motion._Painter.glyph

        def slow_glyph(painter, tick):
            if trap[0] and threading.current_thread().name == "lume-motion":
                trap[0] = False
                blocked.set()
                time.sleep(0.25)              # still composing when stop() lands
            return real_glyph(painter, tick)

        motion._Painter.glyph = slow_glyph
        try:
            with anim.status("x", style=FAST):
                self.assertTrue(blocked.wait(3.0))
                thread = anim._thread
            self.assertFalse(thread.is_alive(), "the frame thread outlived its block")
        finally:
            motion._Painter.glyph = real_glyph
            anim.stop()

    def test_stop_cleans_up_even_when_the_stack_is_already_empty(self):
        """`stop()` is unconditional: what it has to undo lives in the cursor and
        line flags, not in the stack, and a racing `_pop` can empty the stack
        while a frame is still on screen."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        ctx = anim.status("x", style=FAST)
        ctx.__enter__()
        self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
        anim._stack.clear()                    # as a racing _pop would leave it
        anim.stop()
        text = out(console)                    # ...checked before the block exits
        ctx.__exit__(None, None, None)
        self.assertTrue(text.endswith(SHOW), repr(text[-40:]))
        self.assertEqual(text.count(HIDE), text.count(SHOW))
        self.assertFalse(console._cursor_hidden)

    def test_active_follows_the_stack(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        self.assertFalse(anim.active)
        with anim.status("x", style=FAST):
            self.assertTrue(anim.active, "a running status block does not report itself")
            with anim.status("y", style=FAST):
                self.assertTrue(anim.active)
            self.assertTrue(anim.active)
        self.assertFalse(anim.active)
        with anim.status("z", style=FAST):
            anim.stop()
            self.assertFalse(anim.active, "stop() left the stack behind")
        anim.stop()


# ------------------------------------------------------------- the restore path


class LatchedStream(io.StringIO):
    """A stream that only publishes what has actually been flushed."""

    encoding = "utf-8"

    def __init__(self):
        super().__init__()
        self.committed = ""
        self.pending = ""

    def isatty(self):
        return True

    def write(self, text):
        self.pending += text
        return len(text)

    def flush(self):
        self.committed += self.pending
        self.pending = ""


def hold_console(console, seconds=3.0):
    """Hold console.lock from another thread; returns (thread, release_event)."""
    holding, release = threading.Event(), threading.Event()

    def hog():
        with console.lock:
            holding.set()
            release.wait(seconds)

    t = threading.Thread(target=hog, daemon=True)
    t.start()
    holding.wait(3.0)
    return t, release


class RestorePathTests(unittest.TestCase):
    def test_the_final_erase_waits_for_a_briefly_busy_lock(self):
        """regression: with the restore timeout at zero every teardown took the
        raw bypass. The bypass exists for a *stuck* writer; a writer that is
        merely mid-frame must be waited for, so the erase does not land in the
        middle of someone else's output."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        writes = []
        real = console.stream.write

        def spy(text):
            writes.append((text, console.lock._is_owned()))
            return real(text)

        with anim.status("x", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
            console.stream.write = spy
            t, release = hold_console(console, 3.0)
            threading.Timer(0.06, release.set).start()   # busy, but not stuck
            anim.stop()
        console.stream.write = real
        release.set()
        t.join(3.0)
        restore = [owned for text, owned in writes if SHOW in text]
        self.assertTrue(restore, "the cursor was never given back")
        self.assertTrue(all(restore), "the restore bypassed a lock it could have waited for")

    def test_the_bypass_restores_everything_when_the_lock_is_out_of_reach(self):
        """regression: the bypass dropped CLEAR_LINE (leaving the frame frozen on
        screen), never flushed, and left `console._cursor_hidden` True — so the
        console went on believing a cursor was hidden that the terminal was
        showing, and the *next* status block declined to hide it."""
        stream = LatchedStream()
        caps = Caps(color=24, unicode=True, is_tty=True, width=80, height=24,
                    hyperlinks=True, animation=True)
        console = Console(stream=stream, caps=caps)
        anim = motion.Animator(console, THEME)
        ctx = anim.status("x", style=FAST)
        ctx.__enter__()
        self.assertTrue(wait_until(lambda: CLEAR_LINE in stream.committed))
        t, release = hold_console(console, 5.0)         # a stuck writer
        time.sleep(0.1)
        t0 = time.monotonic()
        anim.stop()
        stopped = time.monotonic() - t0
        ctx.__exit__(None, None, None)
        mark = len(stream.committed)
        self.assertLess(stopped, 0.6, f"stop() waited {stopped:.2f}s for a stuck writer")
        self.assertTrue(stream.committed.endswith(CLEAR_LINE + SHOW),
                        f"the bypass did not erase and restore: {stream.committed[-30:]!r}")
        self.assertFalse(console._cursor_hidden,
                         "the console still thinks the cursor is hidden")
        release.set()
        t.join(3.0)
        with anim.status("again", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in stream.committed[mark:]))
        anim.stop()
        self.assertIn(HIDE, stream.committed[mark:],
                      "the next status block ran with a visible cursor")

    def test_the_bypass_writes_nothing_to_a_stream_that_is_not_a_terminal(self):
        caps = Caps(color=24, unicode=True, is_tty=False, width=80, height=24,
                    hyperlinks=False, animation=True)
        console = Console(stream=FakeStream(tty=False), caps=caps)
        anim = motion.Animator(console, THEME)
        ctx = anim.status("x", style=FAST)
        ctx.__enter__()
        self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
        t, release = hold_console(console, 5.0)
        time.sleep(0.1)
        anim.stop()
        ctx.__exit__(None, None, None)
        release.set()
        t.join(3.0)
        text = out(console)
        self.assertNotIn(HIDE, text)
        self.assertNotIn(SHOW, text)

    def test_a_signal_sweep_erases_the_live_frame(self):
        """regression (with lume.ansi): SIGTERM restored the cursor but left the
        half-spun frame frozen on screen with the shell prompt after it. The
        animator has to *say* the line is transient; only it knows."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        with anim.status("thinking", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
            self.assertTrue(console._transient, "the frame did not claim the line")
            mark = len(out(console))
            ansi._restore_all_cursors()
            swept = out(console)[mark:]
        anim.stop()
        self.assertEqual(swept, CLEAR_LINE + SHOW,
                         "the exit sweep left the frame on screen")
        self.assertFalse(console._transient)

    def test_the_line_is_released_when_the_status_ends(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        with anim.status("x", style=FAST):
            self.assertTrue(wait_until(lambda: CLEAR_LINE in out(console)))
        anim.stop()
        self.assertFalse(console._transient, "the erased line is still marked transient")
        mark = len(out(console))
        ansi._restore_all_cursors()
        self.assertEqual(out(console)[mark:], "", "the exit sweep erased a line nobody owned")


class FrameWriteTests(unittest.TestCase):
    def frame_writes(self, console, anim, label="painting"):
        """Every write the frame thread makes during one status block."""
        seen = []
        real = console.stream.write

        def spy(text):
            if threading.current_thread().name == "lume-motion":
                seen.append((text, anim._painted, console._transient))
            return real(text)

        console.stream.write = spy
        try:
            with anim.status(label, style=FAST):
                wait_until(lambda: out(console).count(CLEAR_LINE) >= 4)
        finally:
            anim.stop()
            console.stream.write = real
        return seen

    def test_each_frame_reaches_the_stream_as_a_single_write(self):
        """regression: split into two writes, a frame can be torn by a foreground
        writer that lands between them — the erase without the frame is a flash,
        the frame without the erase is two labels on one line."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        seen = self.frame_writes(console, anim)
        frames = [text for text, _, _ in seen if text != HIDE]
        self.assertGreaterEqual(len(frames), 4)
        for text in frames:
            self.assertTrue(text.startswith(CLEAR_LINE), repr(text))
            self.assertIn("painting", strip_ansi(text), repr(text))

    def test_the_line_is_claimed_before_it_is_drawn_on(self):
        """regression: `_painted` set *after* the write leaves a window where a
        teardown erases nothing and the last frame stays on screen. The same
        goes for the transient mark the signal sweep reads."""
        console = make_console()
        anim = motion.Animator(console, THEME)
        seen = self.frame_writes(console, anim)
        frames = [(painted, transient) for text, painted, transient in seen
                  if text != HIDE]
        self.assertTrue(frames)
        self.assertTrue(all(painted for painted, _ in frames),
                        "a frame was drawn before the line was claimed")
        self.assertTrue(all(transient for _, transient in frames),
                        "a frame was drawn before the line was marked transient")

    def test_the_label_is_rendered_with_the_muted_token(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        with anim.status("thinking", style=FAST):
            self.assertTrue(wait_until(lambda: "thinking" in strip_ansi(out(console))))
        anim.stop()
        self.assertIn(THEME.render("thinking", "app.muted", console.caps), out(console))

    def test_the_elapsed_clock_is_spaced_off_the_label(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        previous = motion._ELAPSED_AFTER
        motion._ELAPSED_AFTER = 0.0
        try:
            with anim.status("waiting", style=FAST):
                self.assertTrue(wait_until(
                    lambda: re.search(r"waiting\s+\d+s", strip_ansi(out(console)))))
        finally:
            motion._ELAPSED_AFTER = previous
            anim.stop()
        painted = [f for f in frames_of(out(console)) if "waiting" in strip_ansi(f)]
        self.assertTrue(painted)
        self.assertTrue(any(re.search(r"waiting {2}\d+s", strip_ansi(f)) for f in painted),
                        f"the clock is glued to the label: {strip_ansi(painted[-1])!r}")


# --------------------------------------------------------------- colour & glyphs


class _StubTheme:
    """Just enough theme to exercise `_stops`."""

    def __init__(self, stops):
        self._stops = stops

    def accent_stops(self):
        return list(self._stops)


class PainterTests(unittest.TestCase):
    def test_the_spinner_colour_drifts_along_the_ramp(self):
        """The one thing that makes the spinner feel alive rather than blinking:
        with a single phase the glyph is one flat colour for ever."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        try:
            with anim.status("drifting", style=FAST):
                time.sleep(0.5)
        finally:
            anim.stop()
        heads = {re.match(r"\x1b\[[0-9;]+m", f).group()
                 for f in frames_of(out(console)) if re.match(r"\x1b\[[0-9;]+m", f)}
        self.assertGreaterEqual(len(heads), 3, f"the spinner colour never drifted: {heads}")

    def test_every_frame_closes_the_colour_it_opens(self):
        """An unreset colour bleeds into whatever the application prints next."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        try:
            with anim.status("resetting", style=FAST):
                wait_until(lambda: out(console).count(CLEAR_LINE) >= 4)
        finally:
            anim.stop()
        painted = [f for f in frames_of(out(console)) if strip_ansi(f).strip()]
        self.assertTrue(painted)
        for frame in painted:
            self.assertRegex(frame, r"^\x1b\[38;2;\d+;\d+;\d+m.\x1b\[0m ",
                             "the spinner glyph is not reset before the label")

    def test_a_custom_spinner_defaults_to_a_calm_cadence(self):
        self.assertGreaterEqual(motion.Spinner("x", ("a",), ("a",)).interval, 0.05)
        self.assertLessEqual(motion.Spinner("x", ("a",), ("a",)).interval, 0.3)

    def test_no_spinner_frame_is_blank(self):
        """regression: the ASCII `bar` set was " .:#:." — the spinner vanished
        for a sixth of every cycle, which reads as a hang, not as motion."""
        for name, spin in motion.SPINNERS.items():
            for label, frames in (("unicode", spin.unicode), ("ascii", spin.ascii)):
                for frame in frames:
                    self.assertTrue(frame.strip(), f"{name}/{label} has a blank frame")

    def test_stops_collapse_consecutive_duplicates(self):
        """regression: several themes point two accent stops at the same colour,
        and a repeated stop flattens half the ramp into one flat band."""
        self.assertEqual(motion._stops(_StubTheme([(1, 2, 3), (1, 2, 3), (4, 5, 6)])),
                         [(1, 2, 3), (4, 5, 6)])
        self.assertEqual(motion._stops(_StubTheme([(1, 2, 3), (1, 2, 3)])), [(1, 2, 3)])

    def test_stops_never_returns_an_empty_ramp(self):
        """`gradient([])` raises, and a themeless terminal must still get a
        banner rather than a traceback."""
        self.assertEqual(motion._stops(_StubTheme([])), [(255, 255, 255)])


class ElapsedFormatTests(unittest.TestCase):
    def test_whole_seconds_only(self):
        self.assertEqual(motion._fmt_elapsed(0.0), "0s")
        self.assertEqual(motion._fmt_elapsed(0.94), "0s")
        self.assertEqual(motion._fmt_elapsed(5.7), "5s")
        self.assertEqual(motion._fmt_elapsed(59.9), "59s")

    def test_minutes_keep_their_seconds(self):
        self.assertEqual(motion._fmt_elapsed(60), "1m00s")
        self.assertEqual(motion._fmt_elapsed(63), "1m03s")
        self.assertEqual(motion._fmt_elapsed(3599), "59m59s")

    def test_hours_read_as_hours_then_minutes(self):
        self.assertEqual(motion._fmt_elapsed(3600), "1h00m")
        self.assertEqual(motion._fmt_elapsed(3723), "1h02m")


class LabelSafetyTests(unittest.TestCase):
    def test_control_characters_become_spaces_or_disappear(self):
        """regression: `_oneline` re-implemented a weaker `sanitize_text`. It
        mapped the C0 controls it happened to name and left DEL and the C1
        block, which contains \\x9b — a second way to spell ESC[ that a terminal
        will happily act on."""
        self.assertEqual(motion._oneline("a\nb\rc\td\ve\ff\bg\ah\x00i"),
                         "a b c d e f g h i")
        self.assertEqual(motion._oneline("a\x7fb"), "ab")
        self.assertEqual(motion._oneline("a\x9b2Jb"), "a2Jb")
        self.assertEqual(motion._oneline("a\x01\x1fb"), "ab")

    def test_a_label_may_carry_colour_but_not_a_screen_clear(self):
        self.assertEqual(motion._oneline("\x1b[31mred\x1b[0m"), "\x1b[31mred\x1b[0m")
        self.assertNotIn("\x1b[2J", motion._oneline("\x1b[2J\x1b[Hnasty"))
        self.assertIn("nasty", motion._oneline("\x1b[2J\x1b[Hnasty"))
        self.assertNotIn("\x1b]", motion._oneline("\x1b]0;title\x07"))

    def test_a_hostile_label_never_reaches_the_terminal(self):
        console = make_console()
        anim = motion.Animator(console, THEME)
        with anim.status("\x1b[2J\x1b[Hnasty\x9b2J", style=FAST) as h:
            self.assertTrue(wait_until(lambda: "nasty" in strip_ansi(out(console))))
            h.update("\x1b]0;pwned\x07more")
            self.assertTrue(wait_until(lambda: "more" in strip_ansi(out(console))))
        anim.stop()
        text = out(console)
        self.assertNotIn("\x1b[2J", text)
        self.assertNotIn("\x1b]", text)
        self.assertNotIn("\x9b", text)
        self.assertNotIn("\n", text)


class StackTests(unittest.TestCase):
    def test_leaving_the_outer_block_first_keeps_the_inner_label(self):
        """regression: `_pop` removed the top of the stack rather than the handle
        that was actually exiting, so an out-of-order exit swapped the two and
        the finished block's label came back."""
        console = make_console()
        anim = motion.Animator(console, THEME, fps=60)
        outer = anim.status("outer-label", style=FAST)
        inner = anim.status("inner-label", style=FAST)
        outer.__enter__()
        inner.__enter__()
        try:
            self.assertTrue(wait_until(lambda: "inner-label" in strip_ansi(out(console))))
            outer.__exit__(None, None, None)          # out of order, on purpose
            mark = len(out(console))
            self.assertTrue(wait_until(
                lambda: "inner-label" in strip_ansi(out(console)[mark:]), 1.0),
                "the inner label stopped painting when the outer block left")
            self.assertNotIn("outer-label", strip_ansi(out(console)[mark:]),
                             "a finished block's label came back")
        finally:
            inner.__exit__(None, None, None)
            anim.stop()
        self.assertFalse(anim.active)


# ------------------------------------------------------- wordmark, banner, fade


class WordmarkShapeTests(unittest.TestCase):
    def test_the_half_block_fold_keeps_the_wordmark_the_right_way_up(self):
        """regression: swapping ▀ (U+2580 UPPER HALF BLOCK) and ▄ (LOWER HALF)
        renders every letter upside down — and every other assertion about the
        wordmark still passes, because the *set* of rows is unchanged."""
        for char, grid in motion._BITMAP.items():
            folded = motion.wordmark_lines(Caps(unicode=True), char)
            self.assertEqual(len(folded) * 2, len(grid))
            for r, line in enumerate(folded):
                top, bottom = grid[2 * r], grid[2 * r + 1]
                for i, cell in enumerate(line):
                    ink = (top[i] == "#", bottom[i] == "#")
                    expected = {(True, True): "█", (True, False): "▀",
                                (False, True): "▄", (False, False): " "}[ink]
                    self.assertEqual(cell, expected,
                                     f"{char!r} row {r} column {i}: {ink} rendered {cell!r}")
        # ...and the shape of it: the foot of the `l` sits in the lower half of
        # its last row, never the upper.
        self.assertEqual(motion.wordmark_lines(Caps(unicode=True), "l")[-1], "█▄")

    def test_wordmark_lines_refuses_letters_it_cannot_draw(self):
        """regression: it answered a different question in silence —
        `wordmark_lines(caps, "hello")` returned the art for `ell`, and `"LUME"`
        returned nothing at all. It is exported; a caller deserves to be told."""
        caps = Caps(unicode=True, width=80)
        with self.assertRaises(ValueError) as raised:
            motion.wordmark_lines(caps, "hello")
        self.assertIn("h", str(raised.exception))
        with self.assertRaises(ValueError):
            motion.wordmark_lines(caps, "lume!")
        self.assertEqual(motion.wordmark_lines(caps, "LUME"),
                         motion.wordmark_lines(caps, "lume"))
        self.assertEqual(motion.wordmark_lines(caps, ""), [])

    def test_the_palette_is_indexed_by_column_not_by_character(self):
        """regression: `_colorize` walked `colors[i]` by character while the
        palette is sized by display width. Equal only while every glyph is one
        column wide — a single wide glyph in the bitmap becomes an IndexError."""
        from lume.ansi import fg
        caps = Caps(color=24, unicode=True, width=80)
        row = "中#"                       # a two-column glyph, then one ink cell
        colors = [(10, 0, 0), (20, 0, 0), (30, 0, 0)]
        painted = motion._colorize(row, colors, caps)
        self.assertIn(fg(colors[0], caps), painted)
        self.assertIn(fg(colors[2], caps), painted, "the second glyph took the wrong colour")
        self.assertEqual(strip_ansi(painted), row)

    def test_the_gradient_never_paints_a_space(self):
        """A colour on a blank cell is an escape sequence per space, in the
        scrollback for ever, for nothing visible."""
        console = make_console()
        motion.banner(console, THEME, animate=False)
        self.assertIsNone(re.search(r"38;2;[0-9;]+m ", out(console)),
                          "the wordmark painted a space")


class BannerColourTests(unittest.TestCase):
    def test_four_bit_colour_gets_one_flat_accent(self):
        """regression: the banner painted its 21-step gradient at 4-bit, where it
        is not a gradient: measured per column on `aurora` it came out
        CYN×1, cyan×7, light-grey×9, bright-blue×4 — the colour drained out of
        the middle of the word. `_Painter` already refuses to do this."""
        for name in ("aurora", "solar", "ember", "mono"):
            theme = get_theme(name)
            console = make_console(color=4, animation=False)
            motion.banner(console, theme, animate=False)
            colours = set(re.findall(r"\x1b\[(\d+)m", out(console))) - {"0"}
            self.assertEqual(len(colours), 1,
                             f"{name}: {len(colours)} colours in a 4-bit wordmark")

    def test_four_bit_colour_picks_the_stop_that_survives_quantising(self):
        from lume.ansi import _ANSI16, rgb_to_16
        from lume.theme import contrast
        for name in ("aurora", "solar", "ember"):
            theme = get_theme(name)
            chosen = motion._flat_accent(theme)
            ratio = contrast(_ANSI16[rgb_to_16(chosen)], theme.background)
            for stop in motion._stops(theme):
                self.assertGreaterEqual(
                    ratio + 1e-9, contrast(_ANSI16[rgb_to_16(stop)], theme.background),
                    f"{name} chose a lower-contrast accent")
            self.assertGreater(ratio, 3.0, f"{name}: the wordmark is barely visible")

    def test_every_sweep_frame_clears_to_the_end_of_its_rows(self):
        """Without it a frame narrower than the last leaves ghost columns."""
        console = make_console()
        motion.banner(console, THEME, animate=True)
        text = out(console)
        rows = motion.wordmark_lines(console.caps)
        frames = [f for f in text.split("\r") if f.strip()]
        self.assertGreaterEqual(text.count(CLEAR_TO_END), len(rows) * (len(frames) - 1))

    def test_the_leading_edge_of_the_sweep_glints(self):
        """The light has a bright edge — that is what makes it read as a light
        moving across the word rather than a wipe. It overshoots the finished
        colour, so it cannot be mistaken for one of the settled columns."""
        from lume.theme import _relative_luminance as luminance
        console = make_console()
        motion.banner(console, THEME, animate=True)
        frames = [f for f in out(console).split("\r") if f.strip()]
        final = cell_colours(frames[-1])
        overshoot = 0.0
        for frame in frames[:-1]:
            cells = cell_colours(frame)
            if len(cells) != len(final):
                continue
            for lit, settled in zip(cells, final):
                if lit and settled:
                    overshoot = max(overshoot, luminance(lit) - luminance(settled))
        self.assertGreater(overshoot, 0.05,
                           "the sweep is a flat wipe: no cell is brighter than the "
                           "finished wordmark")

    def test_a_terminal_too_short_for_the_wordmark_gets_the_compact_banner(self):
        """regression: the reveal walks the cursor up `len(rows) - 1` rows with
        no comparison to caps.height, so on a short terminal it repaints above
        the top of the viewport, over the scrollback."""
        for height in (1, 2, 3):
            console = make_console(width=80)
            console.caps = console.caps.with_size(80, height)
            motion.banner(console, THEME, subtitle="chat", animate=True)
            text = out(console)
            self.assertNotIn("\x1b[2A", text, f"height={height} redrew in place")
            self.assertIn("l u m e", strip_ansi(text))

    def test_the_compact_banner_ends_with_a_blank_line(self):
        """The banner is a header: what follows it needs the breathing room, and
        the full-size one leaves it."""
        console = make_console(width=16, animation=False)
        motion.banner(console, THEME, subtitle="chat", animate=False)
        self.assertTrue(strip_ansi(out(console)).endswith("\n\n"), repr(out(console)))


class RuleShapeTests(unittest.TestCase):
    def test_a_label_survives_a_narrow_rule(self):
        """A label is the only reason to draw a rule with one; dropping it below
        some width silently loses the caller's text."""
        caps = Caps(color=24, unicode=True, width=80)
        for width in range(4, 12):
            line = strip_ansi(motion.rule(width, THEME, caps, "hi"))
            self.assertIn("h", line, f"width={width}: {line!r}")
            self.assertEqual(display_width(line), width)

    def test_a_non_positive_width_draws_nothing(self):
        caps = Caps(color=24, unicode=True, width=80)
        for width in (0, -1, -80):
            self.assertEqual(motion.rule(width, THEME, caps, "label"), "")


class FadeShapeTests(unittest.TestCase):
    def test_static_when_the_text_is_taller_than_the_terminal(self):
        """regression: `fade_in` guarded horizontal wrap but not vertical, so 30
        lines on a 10-row terminal emitted ESC[29A on every frame — past the top
        of the viewport, repainting over the scrollback."""
        console = make_console(width=40)
        console.caps = console.caps.with_size(40, 10)
        motion.fade_in("\n".join(f"line {i:02d}" for i in range(30)), console, THEME)
        text = out(console)
        self.assertEqual(re.findall(r"\x1b\[(\d+)A", text), [],
                         "the fade walked the cursor off the top of the terminal")
        self.assertIn("line 29", strip_ansi(text))

    def test_the_fade_is_over_quickly(self):
        console = make_console()
        t0 = time.monotonic()
        motion.fade_in("settling into place", console, THEME, "app.accent")
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_the_ramp_starts_dim_and_arrives_at_the_token_colour(self):
        """regression: a ramp that stops at half the token colour, or starts at
        the full one, is not a fade — and both leave the final frame correct, so
        only the shape of the ramp can tell."""
        console = make_console()
        theme = THEME
        style = theme["app.accent"]
        dim = theme["app.dim"].fg
        motion.fade_in("settling", console, theme, "app.accent")
        shades = [tuple(int(g) for g in m)
                  for m in re.findall(r"\x1b\[38;2;(\d+);(\d+);(\d+)m", out(console))]
        self.assertGreaterEqual(len(shades), 4)

        def distance(a, b):
            return sum(abs(x - y) for x, y in zip(a, b))

        self.assertEqual(shades[-1], tuple(style.fg), "the fade did not end on the token")
        self.assertLess(distance(shades[0], dim), distance(shades[0], style.fg),
                        "the first frame is already the final colour")
        self.assertLess(distance(shades[-2], style.fg), distance(shades[-2], dim),
                        "the ramp never gets near the token colour")
        self.assertGreater(len(set(shades)), 3, "the ramp has no steps")


if __name__ == "__main__":
    unittest.main()
