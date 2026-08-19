"""Tests for lume.input — the prompt, driven over a pipe, a pty, and a fake readline.

Nothing here needs a real terminal: the tty paths run against ``os.openpty`` and
the readline path against a stub module, so the three environments Prompt has to
survive are all exercised.
"""

import contextlib
import io
import os
import re
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lume import input as inp
from lume.ansi import Caps, Console, display_width, strip_ansi
from lume.input import (Prompt, default_completer, default_history_path,
                        looks_secret, readline_escape)
from lume.theme import get_theme

#: Permission bits are a POSIX idea: Windows synthesises st_mode and has no
#: fchmod, so the mode assertions below are skipped rather than failed there.
POSIX_MODES = unittest.skipUnless(hasattr(os, "fchmod"), "POSIX permission bits")

PIPE_CAPS = Caps(color=0, unicode=False, is_tty=False, width=80, height=24)
TTY_CAPS = Caps(color=24, unicode=True, is_tty=True, width=80, height=24)


class FakeReadline:
    """Just enough of GNU readline to drive Prompt's readline branch."""

    __doc__ = "GNU readline stub"

    def __init__(self):
        self.history = []
        self.binds = []
        self.completer = None
        self.delims = None
        self.auto_history = True
        self.startup_hook = None
        self.line_buffer = ""
        self.inserted = None
        self.length = None

    def set_completer(self, fn=None):
        self.completer = fn

    def get_completer(self):
        return self.completer

    def set_completer_delims(self, d):
        self.delims = d

    def parse_and_bind(self, s):
        self.binds.append(s)

    def set_history_length(self, n):
        self.length = n

    def set_auto_history(self, on):
        self.auto_history = on

    def add_history(self, s):
        self.history.append(s)

    def get_current_history_length(self):
        return len(self.history)

    def remove_history_item(self, i):
        del self.history[i]

    def insert_text(self, t):
        self.inserted = t

    def set_startup_hook(self, fn=None):
        self.startup_hook = fn

    def get_line_buffer(self):
        return self.line_buffer

    def get_endidx(self):
        return len(self.line_buffer)

    def redisplay(self):
        pass


class PipeStdin(io.StringIO):
    """A StringIO that is honest about not being a terminal."""

    def isatty(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("fileno")


class TTYStdin(io.StringIO):
    """Claims to be a tty but has no fd."""

    def isatty(self):
        return True

    def fileno(self):
        raise io.UnsupportedOperation("fileno")


class TTYLineStdin(io.StringIO):
    """A Windows console stand-in: a tty that only ever yields whole lines."""

    on_read = None

    def isatty(self):
        return True

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def readline(self, *a):
        out = super().readline(*a)
        if out and self.on_read is not None:
            self.on_read()
        return out


class ScriptedStdin:
    """Yields canned lines, and raises whatever exception objects are in the list."""

    def __init__(self, items):
        self.items = list(items)

    def isatty(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def readline(self):
        if not self.items:
            return ""
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class PromptTestCase(unittest.TestCase):
    """Common wiring: no readline unless a test asks for it, no history on disk."""

    caps = PIPE_CAPS

    def setUp(self):
        self.theme = get_theme("aurora")
        self.out = io.StringIO()
        self.console = Console(stream=self.out, caps=self.caps)
        patch = mock.patch.object(inp, "readline", None)
        patch.start()
        self.addCleanup(patch.stop)

    def make(self, text="", **kw):
        kw.setdefault("history_path", False)
        kw.setdefault("stdin", PipeStdin(text))
        p = Prompt(self.console, self.theme, **kw)
        self.addCleanup(p.close)
        return p

    def written(self):
        return self.out.getvalue()


class PipedStdinTests(PromptTestCase):
    def test_reads_one_line_per_submission(self):
        p = self.make("hello\nworld\n")
        self.assertEqual(p.read(), "hello")
        self.assertEqual(p.read(), "world")
        with self.assertRaises(EOFError):
            p.read()

    def test_empty_line_returns_empty_string(self):
        p = self.make("\nafter\n")
        self.assertEqual(p.read(), "")
        self.assertEqual(p.read(), "after")

    def test_missing_final_newline_still_reads(self):
        p = self.make("no newline")
        self.assertEqual(p.read(), "no newline")

    def test_crlf_is_stripped(self):
        p = self.make("windows\r\n")
        self.assertEqual(p.read(), "windows")

    def test_eof_on_empty_prompt_raises(self):
        p = self.make("")
        with self.assertRaises(EOFError):
            p.read()

    def test_no_marker_when_stdout_is_not_a_tty(self):
        p = self.make("hi\n")
        p.read()
        self.assertEqual(self.written(), "")

    def test_works_without_readline_module(self):
        p = self.make("hi\n")
        self.assertFalse(p.uses_readline)
        self.assertEqual(p.read(), "hi")

    def test_prefix_is_prepended(self):
        p = self.make("world\n")
        self.assertEqual(p.read(prefix="hello "), "hello world")

    def test_read_after_close_is_refused(self):
        p = self.make("hi\n")
        p.close()
        p.close()                                   # idempotent
        with self.assertRaises(ValueError):
            p.read()


class TtyStdoutTests(PromptTestCase):
    """stdout is a terminal, stdin is a pipe — the marker must still appear."""

    caps = TTY_CAPS

    def test_marker_is_written(self):
        p = self.make("hi\n")
        self.assertEqual(p.read(), "hi")
        self.assertIn("❯", strip_ansi(self.written()))

    def test_continuation_marker_differs(self):
        p = self.make("a \\\nb\n")
        self.assertEqual(p.read(), "a \nb")
        self.assertIn("⋮", strip_ansi(self.written()))

    def test_placeholder_is_shown_once(self):
        p = self.make("hi\n")
        p.read(placeholder="ask me anything")
        self.assertIn("ask me anything", strip_ansi(self.written()))

    def test_prefix_is_echoed(self):
        p = self.make("world\n")
        p.read(prefix="hello ")
        self.assertIn("hello ", strip_ansi(self.written()))

    def test_ascii_terminal_uses_ascii_marker(self):
        self.console.caps = Caps(color=0, unicode=False, is_tty=True, width=80, height=24)
        p = self.make("hi\n")
        p.read()
        self.assertTrue(self.written().isascii())
        self.assertIn(">", self.written())


class MultilineRuleTests(PromptTestCase):
    def test_trailing_backslash_continues(self):
        p = self.make("one \\\ntwo\n")
        self.assertEqual(p.read(), "one \ntwo")

    def test_several_continuations(self):
        p = self.make("a\\\nb\\\nc\nd\n")
        self.assertEqual(p.read(), "a\nb\nc")
        self.assertEqual(p.read(), "d")

    def test_even_backslashes_submit_unchanged(self):
        p = self.make("path\\\\\nnext\n")
        self.assertEqual(p.read(), "path\\\\")
        self.assertEqual(p.read(), "next")

    def test_backslash_inside_a_line_is_literal(self):
        p = self.make("a\\b\n")
        self.assertEqual(p.read(), "a\\b")

    def test_eof_during_continuation_submits_what_there_is(self):
        p = self.make("half \\\n")
        self.assertEqual(p.read(), "half ")
        with self.assertRaises(EOFError):
            p.read()

    def test_fenced_block(self):
        p = self.make('"""\nalpha\nbeta\n"""\nafter\n')
        self.assertEqual(p.read(), "alpha\nbeta")
        self.assertEqual(p.read(), "after")

    def test_fence_on_one_line(self):
        p = self.make('"""just this"""\n')
        self.assertEqual(p.read(), "just this")

    def test_fence_with_text_on_the_opening_line(self):
        p = self.make('"""alpha\nbeta"""\n')
        self.assertEqual(p.read(), "alpha\nbeta")

    def test_fence_keeps_blank_lines_in_the_middle(self):
        p = self.make('"""\na\n\nb\n"""\n')
        self.assertEqual(p.read(), "a\n\nb")

    def test_fence_is_verbatim(self):
        p = self.make('"""\n/help me\nends with \\\n"""\n')
        self.assertEqual(p.read(), "/help me\nends with \\")

    def test_fence_swallows_twenty_lines_as_one_submission(self):
        body = [f"line {i}" for i in range(20)]
        p = self.make('"""\n' + "\n".join(body) + '\n"""\n')
        self.assertEqual(p.read(), "\n".join(body))

    def test_unterminated_fence_at_eof_submits(self):
        p = self.make('"""\nalpha\n')
        self.assertEqual(p.read(), "alpha")

    def test_fence_indented(self):
        p = self.make('  """\nalpha\n"""\n')
        self.assertEqual(p.read(), "alpha")

    def test_quotes_mid_line_are_not_a_fence(self):
        p = self.make('he said """hello"""\n')
        self.assertEqual(p.read(), 'he said """hello"""')

    def test_piped_lines_are_not_glued_together(self):
        # A pipe is a script: coalescing would turn two commands into one message.
        p = self.make("/help\n/quit\n")
        self.assertEqual(p.read(), "/help")
        self.assertEqual(p.read(), "/quit")


class InterruptTests(PromptTestCase):
    caps = TTY_CAPS

    def test_ctrl_c_clears_the_line_and_reprompts(self):
        p = self.make(stdin=ScriptedStdin([KeyboardInterrupt(), "hi\n"]))
        self.assertEqual(p.read(), "hi")
        self.assertIn("Ctrl-C again", strip_ansi(self.written()))

    def test_ctrl_c_twice_on_an_empty_prompt_raises(self):
        p = self.make(stdin=ScriptedStdin([KeyboardInterrupt(), KeyboardInterrupt(), "hi\n"]))
        with self.assertRaises(KeyboardInterrupt):
            p.read()
        self.assertEqual(p.read(), "hi")            # and the prompt still works

    def test_ctrl_c_with_pending_text_does_not_count_toward_exit(self):
        p = self.make(stdin=ScriptedStdin(["half \\\n", KeyboardInterrupt(),
                                           KeyboardInterrupt(), "hi\n"]))
        self.assertEqual(p.read(), "hi")

    def test_interrupt_count_resets_after_a_submission(self):
        p = self.make(stdin=ScriptedStdin([KeyboardInterrupt(), "a\n",
                                           KeyboardInterrupt(), "b\n"]))
        self.assertEqual(p.read(), "a")
        self.assertEqual(p.read(), "b")


class MarkerTests(PromptTestCase):
    caps = TTY_CAPS

    def test_marker_is_two_columns_wide(self):
        p = self.make()
        for uni in (True, False):
            self.console.caps = Caps(color=24, unicode=uni, is_tty=True, width=80, height=24)
            for cont in (False, True):
                self.assertEqual(p.marker_width(cont), 2)
                self.assertEqual(p.marker_width(cont, readline_safe=True), 2)
                self.assertEqual(display_width(p.styled_marker(cont)), 2)

    def test_readline_marker_hides_every_escape(self):
        p = self.make()
        s = p.styled_marker(readline_safe=True)
        self.assertIn("\001", s)
        self.assertIn("\002", s)
        self.assertNotIn("\x1b", re.sub(r"\001[^\002]*\002", "", s))

    def test_readline_escape_keeps_visible_width(self):
        raw = "\x1b[1m\x1b[38;2;1;2;3m>\x1b[0m "
        esc = readline_escape(raw)
        self.assertEqual(display_width(esc), display_width(raw))
        self.assertEqual(esc.count("\001"), esc.count("\002"))
        # Consecutive escapes collapse into a single guarded run.
        self.assertEqual(esc.count("\001"), 2)

    def test_readline_escape_leaves_plain_text_alone(self):
        self.assertEqual(readline_escape("> "), "> ")

    def test_no_colour_marker_has_no_escapes(self):
        self.console.caps = PIPE_CAPS
        p = self.make()
        self.assertEqual(p.styled_marker(), "> ")
        self.assertEqual(p.styled_marker(readline_safe=True), "> ")


class ReadlineBranchTests(PromptTestCase):
    caps = TTY_CAPS

    def setUp(self):
        super().setUp()
        self.rl = FakeReadline()
        patch = mock.patch.object(inp, "readline", self.rl)
        patch.start()
        self.addCleanup(patch.stop)
        # readline's input() reads the process's own stdin and nothing else, so
        # the readline branch is only reachable when sys.stdin *is* the terminal.
        stdin_patch = mock.patch.object(sys, "stdin", TTYStdin())
        stdin_patch.start()
        self.addCleanup(stdin_patch.stop)

    def make_tty(self, **kw):
        return self.make(stdin=None, **kw)

    def make_rl(self, lines, **kw):
        p = self.make_tty(**kw)
        items = list(lines)

        def fake_input(prompt=""):
            if self.rl.startup_hook is not None:    # readline runs it before editing
                self.rl.startup_hook()
            item = items.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        self.fake_input = mock.patch("builtins.input", side_effect=fake_input)
        self.input_mock = self.fake_input.start()
        self.addCleanup(self.fake_input.stop)
        return p

    def test_uses_readline_when_both_ends_are_ttys(self):
        p = self.make_tty()
        self.assertTrue(p.uses_readline)

    def test_does_not_use_readline_for_a_pipe(self):
        p = self.make(stdin=PipeStdin())
        self.assertFalse(p.uses_readline)

    def test_an_injected_stream_never_goes_through_readline(self):
        # input() reads the process's own stdin whatever we pass here, so a
        # Prompt pointed at another stream must take the plain path instead of
        # silently reading the real terminal. (tests/test_app.py depends on it.)
        p = self.make(stdin=TTYStdin("scripted\n"))
        self.assertFalse(p.uses_readline)
        with mock.patch("builtins.input", side_effect=AssertionError("readline used")):
            self.assertEqual(p.read(), "scripted")

    def test_tab_completion_shows_ambiguous_matches_at_once(self):
        self.make_tty()
        self.assertIn("set show-all-if-ambiguous on", self.rl.binds)

    def test_setup_configures_readline(self):
        self.make_tty()
        self.assertEqual(self.rl.delims, " \t\n")
        self.assertIsNotNone(self.rl.completer)
        self.assertFalse(self.rl.auto_history)      # we manage history ourselves
        self.assertIn("tab: complete", self.rl.binds)
        # readline's own paste handler rewrites CR as newline before we can see
        # it, so lume takes the frame off it and reads the body itself.
        self.assertIn("set enable-bracketed-paste off", self.rl.binds)
        self.assertTrue(any(inp.RL_PASTE_MARK in b and r"\e[200~" in b
                            for b in self.rl.binds), self.rl.binds)
        self.assertTrue(any(inp.RL_EOL_MARK in b and r"\C-j" in b
                            for b in self.rl.binds), self.rl.binds)
        self.assertTrue(any(r"\e\r" in b for b in self.rl.binds), self.rl.binds)

    def test_multiline_key_can_be_disabled(self):
        self.make_tty(multiline_key="")
        self.assertFalse(any(r"\e\r" in b for b in self.rl.binds))

    def test_prompt_passed_to_readline_is_guarded(self):
        p = self.make_rl(["hello"])
        self.assertEqual(p.read(), "hello")
        prompt = self.input_mock.call_args_list[0][0][0]
        self.assertIn("\001", prompt)
        self.assertNotIn("\x1b", re.sub(r"\001[^\002]*\002", "", prompt))
        self.assertEqual(display_width(prompt), 2)

    def test_paste_from_readline_is_one_submission(self):
        body = "\n".join(f"line {i}" for i in range(20))
        p = self.make_rl([body])
        self.assertEqual(p.read(), body)
        self.assertEqual(self.input_mock.call_count, 1)

    def test_eof_raises_and_interrupt_reprompts(self):
        p = self.make_rl([KeyboardInterrupt(), "after", EOFError()])
        self.assertEqual(p.read(), "after")
        with self.assertRaises(EOFError):
            p.read()

    def test_prefill_goes_through_the_startup_hook(self):
        p = self.make_rl(["edited"])
        p.read(prefix="draft")
        self.assertEqual(self.rl.inserted, "draft")
        self.assertIsNone(self.rl.startup_hook)     # cleared afterwards

    def test_completer_completes_commands_and_arguments(self):
        p = self.make_tty()
        self.rl.line_buffer = "/mo"
        self.assertEqual(self.rl.completer("/mo", 0), "/model")
        self.assertEqual(self.rl.completer("/mo", 1), "/models")
        self.assertIsNone(self.rl.completer("/mo", 2))
        self.rl.line_buffer = "/theme au"
        self.assertEqual(self.rl.completer("au", 0), "aurora")

    def test_custom_completer_is_used(self):
        p = self.make_tty(completer=lambda text, line: ["zzz"])
        self.rl.line_buffer = "z"
        self.assertEqual(self.rl.completer("z", 0), "zzz")

    def test_single_argument_completer_is_tolerated(self):
        p = self.make_tty(completer=lambda text: ["one", "two"])
        self.rl.line_buffer = "o"
        self.assertEqual(self.rl.completer("o", 0), "one")

    def test_broken_completer_does_not_crash(self):
        def boom(text, line):
            raise RuntimeError("nope")
        p = self.make_tty(completer=boom)
        self.rl.line_buffer = "x"
        self.assertIsNone(self.rl.completer("x", 0))

    def test_ctrl_c_over_a_half_typed_line_is_only_a_cancel(self):
        p = self.make_tty()
        steps = [("half typed", KeyboardInterrupt()), ("", KeyboardInterrupt()), ("", "hi")]

        def fake_input(prompt=""):
            buf, item = steps.pop(0)
            self.rl.line_buffer = buf       # readline keeps its buffer through SIGINT
            if isinstance(item, BaseException):
                raise item
            return item

        with mock.patch("builtins.input", side_effect=fake_input):
            self.assertEqual(p.read(), "hi")
        self.assertIn("Ctrl-C again", strip_ansi(self.written()))

    def test_two_ctrl_c_on_an_empty_line_raises(self):
        p = self.make_tty()
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                p.read()

    def test_history_is_shared_with_readline(self):
        p = self.make_rl(["remember me"])
        p.read()
        self.assertIn("remember me", self.rl.history)

    def test_close_detaches_from_readline(self):
        p = self.make_tty()
        p.close()
        self.assertIsNone(self.rl.completer)
        self.assertTrue(self.rl.auto_history)


class RealReadlineTests(unittest.TestCase):
    """The real module, if the build has one: setup must not explode."""

    def test_setup_against_the_real_module(self):
        if inp.readline is None:                    # pragma: no cover
            self.skipTest("no readline module")
        console = Console(stream=io.StringIO(), caps=PIPE_CAPS)
        p = Prompt(console, get_theme("aurora"), history_path=False,
                   stdin=PipeStdin("hi\n"))
        try:
            self.assertEqual(p.read(), "hi")
        finally:
            p.close()
        self.assertIsNone(inp.readline.get_completer())


@unittest.skipUnless(hasattr(os, "openpty"), "needs a pty")
class PtyHarness(PromptTestCase):
    """A real terminal on stdin, without readline: raw-mode reads plus paste."""

    caps = TTY_CAPS

    def setUp(self):
        super().setUp()
        window = mock.patch.object(inp, "PASTE_WINDOW", 0.25)
        window.start()
        self.addCleanup(window.stop)
        self.master, slave = os.openpty()
        self.addCleanup(self._close, self.master)
        self.stdin = os.fdopen(slave, "r", buffering=1)
        self.addCleanup(self.stdin.close)

    @staticmethod
    def _close(fd):
        try:
            os.close(fd)
        except OSError:
            pass

    def send(self, text):
        os.write(self.master, text.encode())

    def read_soon(self, prompt, timeout=8.0, **kw):
        box = {}

        def run():
            try:
                box["value"] = prompt.read(**kw)
            except BaseException as exc:            # surfaced by the caller
                box["error"] = exc

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            self.fail("read() never returned")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def read_live(self, prompt, chunks, gap=0.0, lead=0.08, timeout=8.0, **kw):
        """Read with the data arriving *after* the read has started.

        Which is how a terminal actually delivers it: the prompt is drawn, the
        read blocks, and only then does anything arrive. Bytes sent before the
        read are a different thing — type-ahead — and the reader is allowed to
        treat them differently.
        """
        box = {}

        def run():
            try:
                box["value"] = prompt.read(**kw)
            except BaseException as exc:            # surfaced by the caller
                box["error"] = exc

        t = threading.Thread(target=run, daemon=True)
        t.start()
        time.sleep(lead)
        for chunk in ([chunks] if isinstance(chunks, str) else chunks):
            self.send(chunk)
            if gap:
                time.sleep(gap)
        t.join(timeout)
        if t.is_alive():
            self.fail("read() never returned")
        if "error" in box:
            raise box["error"]
        return box["value"]


class HistoryKeyTests(PromptTestCase):
    """Up and Down for the reader that has no readline to provide them.

    libedit builds take the plain path (see `_readline_ready`), so without this
    a macOS user would have no history recall at all.
    """

    def make_walked(self, *entries):
        p = self.make(history_path=False)
        for entry in entries:
            p.add_history(entry)
        return p

    def test_up_walks_back_and_down_returns_to_the_draft(self):
        p = self.make_walked("first", "second")
        self.assertEqual(p._recall(True, "draft"), "second")
        self.assertEqual(p._recall(True, ""), "first")
        self.assertEqual(p._recall(False, ""), "second")
        self.assertEqual(p._recall(False, ""), "draft")

    def test_down_on_a_live_line_does_nothing(self):
        p = self.make_walked("only")
        self.assertIsNone(p._recall(False, "typing"))

    def test_up_stops_at_the_oldest_entry(self):
        p = self.make_walked("oldest")
        self.assertEqual(p._recall(True, ""), "oldest")
        self.assertIsNone(p._recall(True, ""))

    def test_no_history_leaves_the_line_alone(self):
        p = self.make_walked()
        self.assertIsNone(p._recall(True, "typing"))

    def test_multi_line_entries_are_skipped(self):
        # Kept in memory for the app, but a one-line reader cannot put one back
        # for editing without lying about what it is.
        p = self.make_walked("plain", "two\nlines")
        self.assertEqual(p._recall(True, ""), "plain")

    def test_each_read_starts_at_the_live_line(self):
        p = self.make("hello\n", history_path=False)
        p.add_history("earlier")
        p._recall(True, "")                          # walked during a previous read
        self.assertEqual(p.read(), "hello")
        self.assertIsNone(p._hist_at)


@unittest.skipUnless(hasattr(os, "openpty"), "needs a pty")
class HistoryKeyTerminalTests(PtyHarness):
    """The same walk, driven by the actual escape sequences a terminal sends."""

    def test_up_arrow_recalls_the_previous_line(self):
        p = self.make(stdin=self.stdin, history_path=False)
        p.add_history("recalled")
        self.assertEqual(self.read_live(p, "\x1b[A\r"), "recalled")

    def test_application_mode_arrows_work_too(self):
        # Some terminals send ESC O A rather than ESC [ A once the cursor keys
        # are in application mode; both are the same key.
        p = self.make(stdin=self.stdin, history_path=False)
        p.add_history("recalled")
        self.assertEqual(self.read_live(p, "\x1bOA\r"), "recalled")

    def test_an_arrow_inside_a_paste_is_text_not_navigation(self):
        p = self.make(stdin=self.stdin, history_path=False)
        p.add_history("recalled")
        got = self.read_live(p, "\x1b[200~up\x1b[Adown\x1b[201~\r")
        self.assertNotIn("recalled", got)
        self.assertIn("up", got)


class PtyTests(PtyHarness):
    """The submission rules against a real terminal."""

    def test_single_line_from_a_terminal(self):
        p = self.make(stdin=self.stdin)
        self.assertTrue(p.console.caps.is_tty)
        self.send("hello\n")
        self.assertEqual(self.read_soon(p), "hello")

    def test_twenty_line_paste_is_one_submission(self):
        p = self.make(stdin=self.stdin)
        body = [f"line {i}" for i in range(20)]
        self.send("".join(line + "\n" for line in body))
        self.assertEqual(self.read_soon(p), "\n".join(body))

    def test_bare_carriage_returns_are_the_return_key_not_a_paste(self):
        # A bare CR is what the Return key sends and the only thing that sends
        # it. Two of them are two submissions however close together they
        # arrived — which is what stops a typed-ahead command being glued into
        # the message before it. A paste that uses CR for its line breaks is
        # byte-for-byte identical to typing, and is treated as typing.
        p = self.make(stdin=self.stdin)
        self.send("alpha\rbeta\r")
        self.assertEqual(self.read_soon(p), "alpha")
        self.assertEqual(self.read_soon(p), "beta")

    def test_the_same_lines_inside_a_paste_frame_are_one_message(self):
        p = self.make(stdin=self.stdin)
        self.send("\x1b[200~alpha\rbeta\x1b[201~\r")
        self.assertEqual(self.read_soon(p), "alpha\nbeta")

    def test_bracketed_paste_guards_are_stripped(self):
        p = self.make(stdin=self.stdin)
        self.send("\x1b[200~one\ntwo\n\x1b[201~")
        self.assertEqual(self.read_soon(p), "one\ntwo")

    def test_separate_lines_are_separate_submissions(self):
        p = self.make(stdin=self.stdin)
        self.send("first\n")
        self.assertEqual(self.read_soon(p), "first")
        self.send("second\n")
        self.assertEqual(self.read_soon(p), "second")

    def test_hangup_raises_eof(self):
        p = self.make(stdin=self.stdin)
        self._close(self.master)
        with self.assertRaises(EOFError):
            self.read_soon(p)

    def test_crlf_paste_does_not_gain_a_blank_line_between_every_line(self):
        # The tty's ICRNL turns CR into NL, so a CRLF paste used to arrive as a
        # double newline per line. Every paste of Windows-authored text and every
        # PuTTY paste went through this.
        p = self.make(stdin=self.stdin)
        self.send("c1\r\nc2\r\nc3\r\n")
        self.assertEqual(self.read_soon(p), "c1\nc2\nc3")

    def test_a_single_crlf_line_does_not_gain_a_trailing_newline(self):
        p = self.make(stdin=self.stdin)
        self.send("solo\r\n")
        self.assertEqual(self.read_soon(p), "solo")

    def test_crlf_paste_of_a_command_is_still_one_message(self):
        p = self.make(stdin=self.stdin)
        self.send("/list of things\r\nsecond line\r\n")
        self.assertEqual(self.read_soon(p), "/list of things\nsecond line")

    def test_bracketed_paste_arrives_whole_however_slowly_it_is_delivered(self):
        # The frame, not the clock, decides where a paste ends: with the window
        # cut to a millisecond the guards still hold three slow lines together.
        with mock.patch.object(inp, "PASTE_WINDOW", 0.001):
            p = self.make(stdin=self.stdin)
            box = {}

            def run():
                box["value"] = p.read()

            t = threading.Thread(target=run, daemon=True)
            t.start()
            self.send("\x1b[200~one\r")
            time.sleep(0.3)
            self.send("two\r")
            time.sleep(0.3)
            self.send("three\x1b[201~\r")
            t.join(8.0)
            self.assertFalse(t.is_alive(), "read() never returned")
            self.assertEqual(box["value"], "one\ntwo\nthree")

    def test_a_paste_that_starts_but_never_closes_still_returns(self):
        with mock.patch.object(inp, "FRAME_TIMEOUT", 0.3):
            p = self.make(stdin=self.stdin)
            self.send("\x1b[200~orphan\r")
            self.assertEqual(self.read_soon(p), "orphan")

    def test_the_terminal_is_left_exactly_as_it_was_found(self):
        # The mode is taken for as long as the prompt lives — the bytes that
        # need it most arrive while lume is printing — and given back on close.
        import termios
        fd = self.stdin.fileno()
        before = termios.tcgetattr(fd)
        p = self.make(stdin=self.stdin)
        self.assertNotEqual(termios.tcgetattr(fd), before, "the mode was not taken")
        self.send("hello\r")
        self.assertEqual(self.read_soon(p), "hello")
        p.close()
        self.assertEqual(termios.tcgetattr(fd), before)

    def test_while_it_is_held_cr_arrives_as_cr(self):
        import termios
        p = self.make(stdin=self.stdin)
        attrs = termios.tcgetattr(self.stdin.fileno())
        self.assertFalse(attrs[0] & termios.ICRNL, "ICRNL would double a CRLF paste")
        self.assertEqual(attrs[6][termios.VEOL], b"\r", "Enter would not end a line")
        self.assertTrue(attrs[3] & termios.NOFLSH)
        p.close()

    def test_closing_turns_bracketed_paste_back_off(self):
        p = self.make(stdin=self.stdin)
        p.close()
        seen = b""
        while select.select([self.master], [], [], 0.2)[0]:
            seen += os.read(self.master, 65536)
        self.assertGreater(seen.rfind(b"\x1b[?2004l"), seen.rfind(b"\x1b[?2004h"), seen)


class LineEndingMeaningTests(unittest.TestCase):
    """What each line ending means, which is the whole of the paste guarantee.

    A bare CR is the Return key; LF and CRLF are content no keyboard produces;
    a bracketed-paste frame outranks both. Everything else in the reader is
    built on these three sentences.
    """

    def test_a_bare_cr_ends_a_submission(self):
        self.assertEqual(inp._submissions("one\rtwo\r"), (["one", "two"], ""))

    def test_an_lf_is_content(self):
        self.assertEqual(inp._submissions("one\ntwo\n"), ([], "one\ntwo\n"))

    def test_a_crlf_is_content_and_becomes_one_newline(self):
        self.assertEqual(inp._submissions("c1\r\nc2\r\n"), ([], "c1\nc2\n"))

    def test_a_frame_is_atomic_however_it_breaks_its_lines(self):
        for body in ("a\nb", "a\rb", "a\r\nb"):
            self.assertEqual(inp._submissions("\x1b[200~%s\x1b[201~\r" % body),
                             (["a\nb"], ""), body)

    def test_an_open_frame_holds_everything_back(self):
        self.assertEqual(inp._submissions("\x1b[200~a\nb\n"),
                         ([], "\x1b[200~a\nb\n"))
        self.assertFalse(inp._soft_done("\x1b[200~a\nb\n"),
                         "a line ending inside a frame is not the end of a paste")

    def test_a_closing_guard_on_its_own_is_noise_not_text(self):
        # A terminal that starts a paste it never finishes used to leave the
        # guard's own bytes in the message.
        self.assertEqual(inp._submissions("three\x1b[201~\r"), (["three"], ""))

    def test_a_paste_before_a_typed_line_keeps_both(self):
        self.assertEqual(inp._submissions("\x1b[200~a\nb\x1b[201~\rnext\r"),
                         (["a\nb", "next"], ""))

    def test_a_soft_tail_is_a_submission_a_partial_line_is_not(self):
        self.assertTrue(inp._soft_done("done\n"))
        self.assertFalse(inp._soft_done("half"))


class LongLineTests(PtyHarness):
    """The canonical buffer is 4096 bytes and drops the rest without a word.

    A minified file, a base64 blob, a data: URI. Before ICANON was cleared for
    the length of a read, an unframed 6000-character line arrived as 4095
    characters and a *framed* one produced no submission at all — three more
    Enters produced nothing either, and only Ctrl-C got the prompt back.
    """

    def test_a_six_thousand_character_line_arrives_whole(self):
        p = self.make(stdin=self.stdin)
        body = "Z" * 6000
        self.assertEqual(self.read_live(p, body + "\r"), body)

    def test_a_six_thousand_character_paste_arrives_whole(self):
        p = self.make(stdin=self.stdin)
        body = "Y" * 6000
        got = self.read_live(p, "\x1b[200~" + body + "\x1b[201~\r")
        self.assertEqual(got, body)

    def test_the_prompt_still_works_afterwards(self):
        p = self.make(stdin=self.stdin)
        self.read_live(p, "W" * 6000 + "\r")
        self.assertEqual(self.read_live(p, "after\r"), "after")

    def test_lines_either_side_of_the_old_limit(self):
        p = self.make(stdin=self.stdin)
        for n in (4000, 4095, 4096, 4097, 8192):
            self.assertEqual(len(self.read_live(p, "x" * n + "\r")), n, n)

    def test_a_hundred_kilobyte_paste_arrives_whole(self):
        p = self.make(stdin=self.stdin)
        body = "\n".join("line %04d %s" % (i, "q" * 60) for i in range(1200))
        got = self.read_live(p, ["\x1b[200~", body, "\x1b[201~\r"], timeout=20)
        self.assertEqual(got, body)

    def test_and_arrives_promptly(self):
        # Reading it a character at a time, rescanning the whole buffer for an
        # open frame at each one, took seven seconds for this.
        p = self.make(stdin=self.stdin)
        body = "\n".join("line %04d %s" % (i, "q" * 60) for i in range(1200))
        start = time.monotonic()
        self.assertEqual(self.read_live(p, ["\x1b[200~", body, "\x1b[201~\r"],
                                        lead=0.05, timeout=30), body)
        self.assertLess(time.monotonic() - start - 0.05, 2.0)


class TypeAheadTests(PtyHarness):
    """Two lines typed while lume was printing are two lines.

    They were one: everything the terminal was holding when a read began got
    taken in one go and glued together, so '/model sonnet' and '/list' arrived
    as a single two-line message — which parse() then (correctly) calls prose,
    so the command went to the API as text.
    """

    def test_two_commands_typed_ahead_stay_two_commands(self):
        p = self.make(stdin=self.stdin)
        self.send("/model sonnet\r/list\r")          # typed while lume printed
        self.assertEqual(self.read_soon(p), "/model sonnet")
        self.assertEqual(self.read_soon(p), "/list")

    def test_and_each_one_still_parses_as_a_command(self):
        from lume import commands
        p = self.make(stdin=self.stdin)
        self.send("/model sonnet\r/list\r")
        self.assertEqual(commands.parse(self.read_soon(p)), ("model", "sonnet"))
        self.assertEqual(commands.parse(self.read_soon(p)), ("list", ""))

    def test_lines_typed_a_few_milliseconds_apart_are_still_two(self):
        p = self.make(stdin=self.stdin)
        self.assertEqual(self.read_live(p, ["abc\r", "def\r"], gap=0.005), "abc")
        self.assertEqual(self.read_soon(p), "def")

    def test_a_paste_made_while_lume_was_printing_is_still_one_message(self):
        # The other half of the same bug: this must *not* be split. The line
        # endings say which is which — nothing here is timed.
        p = self.make(stdin=self.stdin)
        self.send("Title\n\nBody one\n\nBody two\n")
        self.assertEqual(self.read_soon(p), "Title\n\nBody one\n\nBody two")

    def test_a_crlf_paste_made_while_lume_was_printing_keeps_its_breaks(self):
        p = self.make(stdin=self.stdin)
        self.send("c1\r\nc2\r\nc3\r\n")
        self.assertEqual(self.read_soon(p), "c1\nc2\nc3")

    def test_a_blank_line_inside_a_paste_survives(self):
        # The glue used to test the drained chunk for truthiness after cleaning
        # it, so a deliberately blank second line was dropped on the floor.
        p = self.make(stdin=self.stdin)
        self.assertEqual(self.read_live(p, "a\r\n\r\nb\r\n"), "a\n\nb")

    def test_a_typed_line_and_then_a_paste_come_back_in_order(self):
        p = self.make(stdin=self.stdin)
        self.send("first\r\x1b[200~a\nb\x1b[201~\r")
        self.assertEqual(self.read_soon(p), "first")
        self.assertEqual(self.read_soon(p), "a\nb")

    def test_type_ahead_survives_a_half_written_line(self):
        p = self.make(stdin=self.stdin)
        self.send("done\rhalf")
        self.assertEqual(self.read_soon(p), "done")
        self.assertEqual(self.read_live(p, "-written\r"), "half-written")

    def test_ctrl_c_throws_away_what_was_typed_ahead(self):
        p = self.make(stdin=self.stdin)
        self.send("one\rtwo\r")
        self.assertEqual(self.read_soon(p), "one")
        with mock.patch.object(Prompt, "_read_line", side_effect=KeyboardInterrupt):
            with contextlib.suppress(KeyboardInterrupt):
                p.read()
        self.assertEqual(p._pending, [])
        self.assertEqual(p._partial, "")


class NoDeadTimeTests(PtyHarness):
    """A typed Enter must not wait for a paste that a keyboard cannot produce.

    Every submission used to cost PASTE_WINDOW — 100 ms, measured, on every
    terminal and every line, because the only way to know whether more of a
    paste was coming was to wait and see. A bare CR is proof enough that it is
    not: nothing but Return sends one.
    """

    def test_a_typed_line_returns_at_once(self):
        p = self.make(stdin=self.stdin)
        for i in range(5):
            start = time.monotonic()
            self.assertEqual(self.read_live(p, "hi%d\r" % i, lead=0.02), "hi%d" % i)
            spent = time.monotonic() - start - 0.02
            self.assertLess(spent, inp.PASTE_WINDOW,
                            "the prompt waited a paste window for a typed line")

    def test_a_framed_paste_returns_at_once_too(self):
        p = self.make(stdin=self.stdin)
        start = time.monotonic()
        self.assertEqual(self.read_live(p, "\x1b[200~a\nb\x1b[201~\r", lead=0.02),
                         "a\nb")
        self.assertLess(time.monotonic() - start - 0.02, inp.PASTE_WINDOW)

    def test_an_unframed_pasted_line_ending_still_buys_the_window(self):
        # The one thing that is timed, and it stays timed: on a terminal that
        # has never shown a frame this is the only thing holding a paste
        # together. (Two lines, 45 ms apart, one message.)
        p = self.make(stdin=self.stdin)
        self.assertEqual(self.read_live(p, ["alpha\n", "beta\n"], gap=0.045),
                         "alpha\nbeta")


class EchoTests(PtyHarness):
    """The terminal's echo is lume's problem, because it is an injection channel.

    With the tty echoing, a pasted 'ESC]0;…BEL' goes straight back to the
    terminal, which executes it: the window title, the alternate screen, OSC 52
    and the user's clipboard. lume paints the line itself instead.
    """

    def echoed(self):
        seen = b""
        while select.select([self.master], [], [], 0.1)[0]:
            seen += os.read(self.master, 65536)
        return seen

    def test_a_pasted_escape_sequence_is_not_echoed_raw(self):
        # Two channels to close: the terminal's own echo (ECHO is cleared for
        # the read, so nothing comes back off the tty) and lume's painting of
        # the line, which goes through sanitize_text.
        p = self.make(stdin=self.stdin)
        self.echoed()                                  # drop the setup bytes
        payload = "\x1b]0;PWNED\x07hi"
        self.read_live(p, "\x1b[200~" + payload + "\x1b[201~\r")
        self.assertNotIn(b"\x1b]0;PWNED\x07", self.echoed())
        self.assertNotIn("\x1b]0;PWNED", self.written())
        self.assertNotIn("\x07", self.written())

    def test_it_is_not_in_the_submission_either(self):
        p = self.make(stdin=self.stdin)
        got = self.read_live(p, "\x1b[200~\x1b]0;PWNED\x07hi\x1b[201~\r")
        self.assertNotIn("\x1b", got)
        self.assertNotIn("\x07", got)
        self.assertIn("hi", got)

    def test_nor_in_the_history_file(self):
        path = Path(tempfile.mkdtemp()) / "history"
        p = self.make(stdin=self.stdin, history_path=path)
        self.read_live(p, "\x1b[200~\x1b]0;PWNED\x07hi\x1b[201~\r")
        p.close()
        body = path.read_text()
        self.assertNotIn("\x1b", body)
        self.assertIn("hi", body)

    def test_what_was_typed_is_still_shown(self):
        # Turning the terminal's echo off is only safe if lume paints the line
        # itself: a prompt that shows nothing as you type is not a prompt.
        p = self.make(stdin=self.stdin)
        self.read_live(p, "visible\r")
        self.assertIn("visible", self.written())

    def test_and_shown_once_because_the_terminal_is_not_echoing_too(self):
        # With ECHO left on, everything typed appears twice — once from the tty
        # and once from here — and an edit key lands as ^? in the middle of it.
        p = self.make(stdin=self.stdin)
        self.echoed()
        self.read_live(p, "visible\r")
        self.assertNotIn(b"visible", self.echoed(), "the terminal echoed it as well")

    def test_backspace_does_not_leave_its_own_key_on_the_screen(self):
        p = self.make(stdin=self.stdin)
        self.echoed()
        self.assertEqual(self.read_live(p, "abX\x7f\r"), "ab")
        self.assertNotIn(b"^?", self.echoed())

    def test_backspace_erases_a_character(self):
        p = self.make(stdin=self.stdin)
        self.assertEqual(self.read_live(p, "abcX\x7f\r"), "abc")

    def test_ctrl_u_kills_the_line(self):
        p = self.make(stdin=self.stdin)
        self.assertEqual(self.read_live(p, "throw away\x15kept\r"), "kept")

    def test_ctrl_w_kills_a_word(self):
        p = self.make(stdin=self.stdin)
        self.assertEqual(self.read_live(p, "keep this\x17\r"), "keep ")

    def test_an_edit_key_inside_a_paste_is_text(self):
        p = self.make(stdin=self.stdin)
        got = self.read_live(p, "\x1b[200~a\x7fb\x1b[201~\r")
        self.assertEqual(got, "ab", "the paste's own bytes were treated as keys")

    def test_ctrl_d_on_an_empty_line_is_still_end_of_input(self):
        p = self.make(stdin=self.stdin)
        with self.assertRaises(EOFError):
            self.read_live(p, "\x04")

    def test_ctrl_d_over_a_half_typed_line_sends_it(self):
        p = self.make(stdin=self.stdin)
        self.assertEqual(self.read_live(p, "half\x04"), "half")


class FenceInsideAPasteTests(PtyHarness):
    """The rule is 'the first line that ends with \"\"\"', not 'the last line of the chunk'.

    Pasting into an open block used to check only the chunk's final line, so a
    paste containing the closing marker produced no submission at all and
    swallowed everything after it.
    """

    def test_a_paste_that_closes_the_fence_stops_there(self):
        p = self.make(stdin=self.stdin)
        self.send('"""\r')
        self.assertEqual(self.read_live(
            p, '\x1b[200~inside\n"""\nAFTER\nmore\x1b[201~\r'), "inside")
        self.assertEqual(self.read_soon(p), "AFTER\nmore")

    def test_nothing_after_the_marker_means_nothing_left_over(self):
        p = self.make(stdin=self.stdin)
        self.send('"""\r')
        self.assertEqual(self.read_live(p, '\x1b[200~inside\n"""\x1b[201~\r'),
                         "inside")
        self.assertEqual(p._pending, [])

    def test_a_fence_that_is_never_closed_still_takes_the_whole_paste(self):
        p = self.make(stdin=self.stdin)
        self.send('"""\r')
        got = self.read_live(p, ["\x1b[200~one\ntwo\x1b[201~\r", 'three"""\r'],
                             gap=0.05)
        self.assertEqual(got, "one\ntwo\nthree")


class HistoryFileTests(PromptTestCase):
    def setUp(self):
        super().setUp()
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "nested" / "history"

    def make_hist(self, text="", **kw):
        return self.make(history_path=self.path, stdin=PipeStdin(text), **kw)

    @POSIX_MODES
    def test_file_and_directory_are_private(self):
        p = self.make_hist()
        p.add_history("hello")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)

    @POSIX_MODES
    def test_loose_permissions_are_tightened(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("old\n")
        os.chmod(self.path, 0o644)
        self.make_hist()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_entries_round_trip(self):
        p = self.make_hist()
        p.add_history("first")
        p.add_history("second")
        p.close()
        again = self.make_hist()
        self.assertEqual(again.history(), ["first", "second"])

    def test_reading_records_history(self):
        p = self.make_hist("hello\n")
        p.read()
        self.assertEqual(self.path.read_text(), "hello\n")

    def test_blank_and_duplicate_entries_are_ignored(self):
        p = self.make_hist()
        for text in ("", "   ", "same", "same"):
            p.add_history(text)
        self.assertEqual(p.history(), ["same"])
        self.assertEqual(self.path.read_text(), "same\n")

    def test_secrets_are_never_written(self):
        p = self.make_hist()
        secret = "sk-ant-api03-AAAABBBBCCCCDDDD"
        p.add_history(secret)
        p.add_history(f"my key is {secret} ok")
        p.add_history("ANTHROPIC_API_KEY=abcdefghijklmnop")
        p.close()
        body = self.path.read_text()
        self.assertNotIn("sk-ant-", body)
        self.assertNotIn("abcdefghijklmnop", body)
        self.assertEqual(p.history(), [])

    def test_secret_submissions_are_not_remembered(self):
        p = self.make_hist("sk-ant-api03-ZZZZYYYYXXXX\n")
        self.assertEqual(p.read(), "sk-ant-api03-ZZZZYYYYXXXX")
        self.assertEqual(p.history(), [])
        self.assertEqual(self.path.read_text(), "")

    def test_multiline_entries_stay_in_memory_only(self):
        p = self.make_hist()
        p.add_history("one\ntwo")
        self.assertEqual(p.history(), ["one\ntwo"])
        self.assertEqual(self.path.read_text(), "")

    def test_size_cap_is_enforced(self):
        p = self.make_hist(history_max=5)
        for i in range(40):
            p.add_history(f"entry {i}")
        p.close()
        lines = self.path.read_text().splitlines()
        self.assertLessEqual(len(lines), 5)
        self.assertEqual(lines[-1], "entry 39")

    def test_cap_survives_a_reload(self):
        p = self.make_hist(history_max=3)
        for i in range(10):
            p.add_history(f"e{i}")
        p.close()
        again = self.make_hist(history_max=3)
        self.assertEqual(again.history(), ["e7", "e8", "e9"])

    def test_corrupt_history_file_is_tolerated(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"good\n\xff\xfe binary \x00\nalso good\n")
        p = self.make_hist()
        self.assertIn("good", p.history())
        self.assertIn("also good", p.history())

    def test_history_can_be_disabled(self):
        p = self.make(history_path=False)
        p.add_history("nothing persists")
        self.assertIsNone(p.history_path)
        self.assertEqual(p.history(), ["nothing persists"])

    def test_unwritable_location_degrades_quietly(self):
        p = self.make(history_path=self.dir / "history" / "x" / "y",
                      stdin=PipeStdin("hi\n"))
        os.chmod(self.dir, 0o500)
        self.addCleanup(os.chmod, self.dir, 0o700)
        p.add_history("still fine")
        self.assertEqual(p.read(), "hi")

    def test_default_path_honours_the_environment(self):
        with mock.patch.dict(os.environ, {"LUME_HISTORY": "/tmp/lume-hist"}, clear=False):
            self.assertEqual(default_history_path(), Path("/tmp/lume-hist"))
        env = {k: v for k, v in os.environ.items() if k not in ("LUME_HISTORY",)}
        env["LUME_HOME"] = "/tmp/lume-home"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_history_path(), Path("/tmp/lume-home/history"))
        env = {"XDG_DATA_HOME": "/tmp/xdg"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_history_path(), Path("/tmp/xdg/lume/history"))


class PasteNormalisationTests(unittest.TestCase):
    """_clean_paste sees bytes queued while readline held the tty in raw mode."""

    def test_crlf_and_cr_become_newlines(self):
        self.assertEqual(inp._clean_paste("a\r\nb\rc\n"), "a\nb\nc")

    def test_bracketed_paste_guards_are_removed(self):
        self.assertEqual(inp._clean_paste("\x1b[200~a\nb\x1b[201~"), "a\nb")

    def test_empty(self):
        self.assertEqual(inp._clean_paste(""), "")


class InterruptGuardTests(unittest.TestCase):
    """Stopping a reply mid-stream — the guard app.py wraps the stream loop in."""

    def test_ctrl_c_reaches_the_callback(self):
        seen = []
        with inp.interrupt_guard(lambda: seen.append("stop")):
            signal.raise_signal(signal.SIGINT)
        self.assertEqual(seen, ["stop"])

    def test_the_previous_handler_is_put_back(self):
        before = signal.getsignal(signal.SIGINT)
        with inp.interrupt_guard(lambda: None):
            self.assertIsNot(signal.getsignal(signal.SIGINT), before)
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_the_handler_is_restored_after_an_exception(self):
        before = signal.getsignal(signal.SIGINT)
        with self.assertRaises(ValueError):
            with inp.interrupt_guard(lambda: None):
                raise ValueError("boom")
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_a_raising_callback_never_escapes(self):
        def boom():
            raise RuntimeError("nope")
        with inp.interrupt_guard(boom):
            signal.raise_signal(signal.SIGINT)      # must not propagate

    def test_ctrl_c_does_not_raise_while_guarded(self):
        # The whole point: during a reply Ctrl-C cancels, it does not unwind.
        with inp.interrupt_guard(lambda: None):
            signal.raise_signal(signal.SIGINT)
            self.assertEqual(1 + 1, 2)              # still running

    def test_off_the_main_thread_it_is_a_no_op(self):
        box = {}

        def run():
            try:
                with inp.interrupt_guard(lambda: None):
                    box["ok"] = True
            except BaseException as exc:            # pragma: no cover
                box["error"] = exc

        t = threading.Thread(target=run)
        t.start(); t.join(5)
        self.assertTrue(box.get("ok"), box.get("error"))

    def test_the_console_handler_is_registered_on_windows(self):
        # The POSIX signal alone is not enough there, so the guard must ask for
        # the console handler too, and must take it back down afterwards.
        calls = []

        def fake_install(callback):
            calls.append(callback)
            return lambda: calls.append("removed")

        with mock.patch.object(inp, "_install_console_ctrl", fake_install):
            with inp.interrupt_guard(lambda: None):
                self.assertEqual(len(calls), 1)
        self.assertEqual(calls[-1], "removed")

    def test_no_console_handler_off_windows(self):
        self.assertIsNone(inp._install_console_ctrl(lambda: None))


class WindowsConsolePasteTests(PromptTestCase):
    """The Windows reader: no select(), so msvcrt.kbhit decides what is a paste."""

    caps = TTY_CAPS

    class FakeMsvcrt:
        def __init__(self, pending):
            self.pending = pending                  # lines still "in the console"

        def kbhit(self):
            return self.pending[0] > 0

    def make_console(self, text, queued):
        stdin = TTYLineStdin(text)
        fake = self.FakeMsvcrt(queued)
        stdin.on_read = lambda: queued.__setitem__(0, max(0, queued[0] - 1))
        patch = mock.patch.object(inp, "msvcrt", fake)
        patch.start()
        self.addCleanup(patch.stop)
        return self.make(stdin=stdin)

    def test_a_paste_is_one_submission(self):
        body = [f"line {i}" for i in range(20)]
        p = self.make_console("".join(l + "\r\n" for l in body), [20])
        self.assertEqual(p.read(), "\n".join(body))

    def test_typing_is_still_one_line_at_a_time(self):
        p = self.make_console("first\r\nsecond\r\n", [1])
        self.assertEqual(p.read(), "first")
        self.assertEqual(p.read(), "second")

    def test_without_msvcrt_it_reads_line_by_line(self):
        patch = mock.patch.object(inp, "msvcrt", None)
        patch.start()
        self.addCleanup(patch.stop)
        p = self.make(stdin=TTYLineStdin("a\r\nb\r\n"))
        self.assertEqual(p.read(), "a")

    def test_a_pipe_is_never_coalesced(self):
        fake = self.FakeMsvcrt([9])
        patch = mock.patch.object(inp, "msvcrt", fake)
        patch.start()
        self.addCleanup(patch.stop)
        p = self.make("/help\n/quit\n")             # PipeStdin: not a tty
        self.assertEqual(p.read(), "/help")
        self.assertEqual(p.read(), "/quit")


class SecretDetectionTests(unittest.TestCase):
    def test_positives(self):
        for text in ("sk-ant-api03-abcdefgh",
                     "export ANTHROPIC_API_KEY=sk-ant-oat01-xxxxxxxx",
                     "Authorization: Bearer abc123def456",
                     "ghp_0123456789abcdefghijklmnopqrst",
                     "AKIAIOSFODNN7EXAMPLE",
                     "-----BEGIN RSA PRIVATE KEY-----"):
            self.assertTrue(looks_secret(text), text)

    def test_negatives(self):
        for text in ("", "what is sk-ant?", "tell me about api keys",
                     "the sky is blue", "/model sonnet"):
            self.assertFalse(looks_secret(text), text)


class CompleterTests(unittest.TestCase):
    def test_completes_command_names(self):
        self.assertEqual(default_completer("/mo", "/mo"), ["/model", "/models"])

    def test_completes_arguments(self):
        self.assertEqual(default_completer("au", "/theme au"), ["aurora"])

    def test_ignores_prose(self):
        self.assertEqual(default_completer("hel", "hel"), [])


@unittest.skipUnless(hasattr(os, "openpty"), "needs a pty")
class RealTimingPtyTests(PromptTestCase):
    """PASTE_WINDOW itself, unpatched — the boundary pinned from both sides.

    Every other pty test patches the window, which is how a coalescing bug that
    only shows up at real source-side latency survived a whole test suite.
    """

    caps = TTY_CAPS

    def setUp(self):
        super().setUp()
        self.master, slave = os.openpty()
        self.addCleanup(self._close, self.master)
        self.stdin = os.fdopen(slave, "r", buffering=1)
        self.addCleanup(self.stdin.close)

    @staticmethod
    def _close(fd):
        try:
            os.close(fd)
        except OSError:
            pass

    def read_in_thread(self, prompt):
        box = {}

        def run():
            try:
                box["value"] = prompt.read()
            except BaseException as exc:            # surfaced by the caller
                box["error"] = exc

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return box, t

    def finish(self, box, t, timeout=10.0):
        t.join(timeout)
        if t.is_alive():
            self.fail("read() never returned")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def test_a_paste_delivered_a_line_at_a_time_is_one_submission(self):
        # 45 ms per line is an ordinary ssh/tmux/mosh delivery rate. At the old
        # 30 ms window this arrived as eight separate submissions.
        self.assertGreaterEqual(inp.PASTE_WINDOW, 0.08)
        p = self.make(stdin=self.stdin)
        body = [f"line {i}" for i in range(8)]
        box, t = self.read_in_thread(p)
        for line in body:
            os.write(self.master, (line + "\n").encode())
            time.sleep(0.045)
        self.assertEqual(self.finish(box, t), "\n".join(body))

    def test_lines_far_apart_are_still_separate_submissions(self):
        # The other side of the boundary: a window wide enough to swallow a
        # second thought would be its own bug.
        self.assertLessEqual(inp.PASTE_WINDOW, 0.2)
        p = self.make(stdin=self.stdin)
        box, t = self.read_in_thread(p)
        os.write(self.master, b"first\n")
        time.sleep(0.45)
        os.write(self.master, b"second\n")
        self.assertEqual(self.finish(box, t), "first")
        box, t = self.read_in_thread(p)
        self.assertEqual(self.finish(box, t), "second")

    def test_a_dribble_with_no_newline_cannot_hold_the_drain_open(self):
        # The clock restarts on newlines, not on bytes. In raw mode — which is
        # what readline hands back — a source trickling characters would
        # otherwise extend the window for as long as it kept typing.
        import tty
        fd = self.stdin.fileno()
        tty.setcbreak(fd)                           # partial lines become readable
        p = self.make(stdin=self.stdin)
        box = {}

        def run():
            t0 = time.monotonic()
            box["value"] = p._drain(fd)
            box["elapsed"] = time.monotonic() - t0

        t = threading.Thread(target=run, daemon=True)
        t.start()
        for _ in range(8):
            time.sleep(0.04)
            os.write(self.master, b"x")             # never delimited
        t.join(5.0)
        self.assertFalse(t.is_alive(), "_drain never returned")
        self.assertLess(box["elapsed"], inp.PASTE_WINDOW * 2.5,
                        "the dribble held the drain open")


ROOT = str(Path(__file__).resolve().parent.parent)

CHILD = """
import os, sys
sys.path.insert(0, %r)
import lume.input as inp
if os.environ.get("NORL"):
    inp.readline = None
from lume.ansi import Caps, Console
from lume.input import Prompt
from lume.theme import get_theme
caps = Caps(color=0, unicode=False, is_tty=True, width=80, height=24)
if os.environ.get("NET"):
    from lume.ansi import install_signal_net
    install_signal_net()
p = Prompt(Console(sys.stdout, caps), get_theme(), history_path=os.environ.get("H") or "")
if os.environ.get("BUSY"):
    import time
    sys.stdout.write("> \\n"); sys.stdout.flush()   # the marker start() waits for
    time.sleep(float(os.environ["BUSY"]))           # ...lume, printing a reply
while True:
    try:
        t = p.read()
    except EOFError:
        sys.stdout.write("\\x01EOF\\x02\\n"); sys.stdout.flush(); break
    except KeyboardInterrupt:
        sys.stdout.write("\\x01KBD\\x02\\n"); sys.stdout.flush(); break
    sys.stdout.write("\\x01%%r\\x02\\n" %% (t,)); sys.stdout.flush()
p.close()
""" % ROOT


def _controlling_tty():                             # pragma: no cover - child side
    import fcntl as _fcntl
    import termios as _termios
    os.setsid()
    _fcntl.ioctl(0, _termios.TIOCSCTTY, 0)


@unittest.skipUnless(hasattr(os, "openpty") and sys.platform != "win32", "needs a pty")
class ChildHarness(unittest.TestCase):
    """A real lume prompt in a child process that owns a real controlling terminal.

    Nothing else can test Ctrl-C, a signal or readline honestly: the tty only
    sends SIGINT to the foreground process group of its own session, and
    readline only reads the process's own stdin.
    """

    def start(self, **env):
        import pty
        try:
            import termios
            termios.TIOCSCTTY
        except (ImportError, AttributeError):       # pragma: no cover
            self.skipTest("no TIOCSCTTY")
        master, slave = pty.openpty()
        self.master = master
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD], stdin=slave, stdout=slave, stderr=slave,
            env=dict(os.environ, PYTHONUNBUFFERED="1", **env),
            preexec_fn=_controlling_tty)
        os.close(slave)
        self.proc = proc
        self.addCleanup(self._stop)
        self.buf = bytearray()
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()
        # Wait for the prompt marker rather than guessing: the child is only
        # ready to be typed at once it is inside read().
        if not self.wait_for(lambda: ">" in bytes(self.buf).decode("utf-8", "replace"), 10.0):
            self.fail("the child never reached its prompt")
        return proc

    def _pump(self):
        while True:
            try:
                ready, _, _ = select.select([self.master], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            self.buf += data

    def _stop(self):
        with contextlib.suppress(Exception):
            self.proc.kill()
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            os.close(self.master)

    def wait_for(self, pred, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            time.sleep(0.02)
        return pred()

    def send(self, data, pause=0.25):
        os.write(self.master, data)
        time.sleep(pause)

    def expect(self, count, timeout=5.0):
        self.wait_for(lambda: len(self.submissions()) >= count, timeout)
        return self.submissions()

    def submissions(self):
        text = bytes(self.buf).decode("utf-8", "replace")
        return re.findall(r"\x01(.*?)\x02", text)

    def baseline(self):
        """A fresh pty's modes: what this one has to look like again afterwards."""
        import pty as _pty
        import termios
        master, slave = _pty.openpty()
        try:
            return termios.tcgetattr(master)
        finally:
            os.close(master)
            os.close(slave)


class ChildOnATerminalTests(ChildHarness):
    """Ctrl-C, Ctrl-D and pastes, in a child that owns its terminal."""

    def test_ctrl_c_over_typed_text_cancels_the_line_without_leaving_lume(self):
        # Without readline there is no line buffer to consult, so Prompt has to
        # find the typed text itself; before that, 'abc' then two Ctrl-C quit.
        self.start(NORL="1")
        self.send(b"abc")
        self.send(b"\x03")
        self.send(b"\x03")
        self.send(b"hi\n", pause=0)
        self.assertEqual(self.expect(1), ["'hi'"])
        self.assertIsNone(self.proc.poll(), "lume exited on Ctrl-C")

    def test_two_ctrl_c_on_an_empty_line_still_leaves(self):
        self.start(NORL="1")
        self.send(b"\x03")
        self.send(b"\x03", pause=0)
        self.assertIn("KBD", self.expect(1))

    def test_a_crlf_paste_at_the_prompt_keeps_its_line_breaks(self):
        # ICRNL turns the CR of every CRLF into a second newline, so this used to
        # arrive with a blank line between every line — on both tty paths.
        self.start(NORL="1")
        self.send(b"c1\r\nc2\r\nc3\r\n", pause=0)
        self.assertEqual(self.expect(1), ["'c1\\nc2\\nc3'"])

    def test_a_crlf_paste_at_a_readline_prompt_keeps_its_line_breaks(self):
        # Same paste, with readline in charge: the tail of it is drained by
        # _absorb_paste, which must not add a newline the chunk already carries.
        self.start()
        self.send(b"w1\r\nw2\r\nw3\r\n", pause=0)
        self.assertEqual(self.expect(1), ["'w1\\nw2\\nw3'"])

    def test_a_slow_paste_over_readline_is_one_submission_and_no_secret_lands(self):
        # The whole of the largest gap, end to end: a nine-line paste at 45 ms a
        # line over a real pty, with readline in charge. It used to arrive as six
        # submissions with an OpenSSH key body and an auth token in the history
        # file, in cleartext.
        home = Path(tempfile.mkdtemp())
        hist = home / "history"
        self.start(H=str(hist))
        doc = ('intro\n'
               'cmd = "a\\\n'
               'b"\n'
               '"""\n'
               'quoted\n'
               '"""\n'
               '-----BEGIN OPENSSH PRIVATE KEY-----\n'
               'b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdz\n'
               'ANTHROPIC_AUTH_TOKEN=sk-live-9f2b7c1d4e8a\n')
        for line in doc.splitlines(True):
            os.write(self.master, line.encode())
            time.sleep(0.045)
        subs = self.expect(1, timeout=5.0)
        time.sleep(0.3)                             # ...and no second one follows
        subs = self.submissions()
        self.assertEqual(len(subs), 1, subs)
        self.assertIn("BEGIN OPENSSH PRIVATE KEY", subs[0])
        self.assertIn("quoted", subs[0])
        body = hist.read_text() if hist.exists() else ""
        self.assertEqual(body, "")
        for leak in ("sk-live", "b3BlbnNzaC1rZXktdjEA", "ANTHROPIC_AUTH_TOKEN"):
            self.assertNotIn(leak, body)


class PasteThroughRealReadlineTests(ChildHarness):
    """A bracketed paste through *real* readline, which nothing tested before.

    Every other pty test injects a stream, and injecting one turns readline off
    by construction — so the path that actually runs in production was the one
    path never exercised with a paste-guarded paste in it. It was also the one
    that was wrong: readline's own paste handler rewrites every CR as a newline,
    so a CRLF paste arrived with a blank line between every line and no amount
    of termios could undo it.
    """

    def test_a_guarded_crlf_paste_keeps_its_line_breaks(self):
        self.start()
        self.send(b"\x1b[200~a\r\nb\x1b[201~", pause=0.3)
        self.send(b"\r", pause=0)
        self.assertEqual(self.expect(1), ["'a\\nb'"])

    def test_a_guarded_paste_is_verbatim_whichever_ending_it_uses(self):
        for eol, want in ((b"\n", "a\\nb\\nc"), (b"\r", "a\\nb\\nc"),
                          (b"\r\n", "a\\nb\\nc")):
            with self.subTest(eol=eol):
                self.start()
                body = b"a" + eol + b"b" + eol + b"c"
                self.send(b"\x1b[200~" + body + b"\x1b[201~", pause=0.3)
                self.send(b"\r", pause=0)
                self.assertEqual(self.expect(1), ["'%s'" % want])
                self._stop()

    def test_a_guarded_paste_keeps_its_blank_lines(self):
        # The other direction, and the one a repair heuristic gets wrong: these
        # blank lines are the user's, not a doubled CR.
        self.start()
        self.send(b"\x1b[200~T\n\nB1\n\nB2\x1b[201~", pause=0.3)
        self.send(b"\r", pause=0)
        self.assertEqual(self.expect(1), ["'T\\n\\nB1\\n\\nB2'"])

    def test_a_guarded_paste_arrives_whole_however_slowly(self):
        self.start()
        self.send(b"\x1b[200~one\r\n", pause=0.6)
        self.send(b"two\r\n", pause=0.6)
        self.send(b"three\x1b[201~", pause=0.3)
        self.send(b"\r", pause=0)
        self.assertEqual(self.expect(1), ["'one\\ntwo\\nthree'"])

    def test_a_single_line_paste_is_left_in_the_line_to_edit(self):
        # No line ending in it, so it is not a message yet — the same rule the
        # plain reader uses. Typing after it must land in the same submission.
        self.start()
        self.send(b"\x1b[200~pasted\x1b[201~", pause=0.3)
        self.send(b" and typed\r", pause=0)
        self.assertEqual(self.expect(1), ["'pasted and typed'"])

    def test_the_mark_is_never_left_in_the_message(self):
        self.start()
        self.send(b"\x1b[200~x\ny\x1b[201~", pause=0.3)
        self.send(b"\r", pause=0)
        subs = self.expect(1)
        self.assertNotIn(inp.RL_PASTE_MARK, subs[0])
        self.assertNotIn(inp.RL_EOL_MARK, subs[0])

    def test_typing_the_mark_by_hand_is_ordinary_text(self):
        self.start()
        self.send(inp.RL_PASTE_MARK.encode() + b"\r", pause=0)
        self.assertEqual(self.expect(1), ["%r" % inp.RL_PASTE_MARK])

    def test_two_commands_typed_ahead_run_as_two_commands(self):
        # Typed while lume was printing a reply. They used to arrive as one
        # two-line message, which parse() calls prose — so the command was sent
        # to the API as text.
        self.start(BUSY="1.2")
        self.send(b"/model sonnet\r", pause=0.4)
        self.send(b"/list\r", pause=0)
        self.assertEqual(self.expect(2), ["'/model sonnet'", "'/list'"])

    def test_a_paste_made_while_lume_was_printing_is_still_one_message(self):
        self.start(BUSY="1.2")
        self.send(b"Title\n\nBody one\n\nBody two\n", pause=0)
        self.assertEqual(self.expect(1), ["'Title\\n\\nBody one\\n\\nBody two'"])

    def test_a_typed_line_does_not_wait_for_a_paste_window(self):
        self.start()
        start = time.monotonic()
        self.send(b"hello\r", pause=0)
        self.assertTrue(self.wait_for(lambda: self.submissions(), 3.0))
        self.assertLess(time.monotonic() - start, inp.PASTE_WINDOW * 4,
                        "every submission was paying the fallback window")

    def test_a_pasted_escape_sequence_is_neither_run_nor_kept(self):
        home = Path(tempfile.mkdtemp())
        hist = home / "history"
        self.start(H=str(hist))
        self.send(b"\x1b[200~\x1b]0;PWNED\x07hi\x1b[201~", pause=0.3)
        self.send(b"\r", pause=0)
        subs = self.expect(1)
        self.assertNotIn("\\x1b", subs[0])
        self.assertIn("hi", subs[0])
        body = hist.read_text() if hist.exists() else ""
        self.assertNotIn("\x1b", body)


class SignalledAwayTests(ChildHarness):
    """SIGTERM and SIGHUP: the terminal has to come back, or the shell is broken.

    Closing a terminal window sends SIGHUP. Before the exit hook, that left
    ICRNL off, NOFLSH on, VEOL set to CR and bracketed paste still on — the
    user's next shell needed 'stty sane', and it was not obvious why.
    """

    def modes(self):
        import termios
        return termios.tcgetattr(self.master)

    def check_restored(self, signum):
        before = self.baseline()
        self.start(NET="1")
        self.assertNotEqual(self.modes(), before, "the mode was never taken")
        os.kill(self.proc.pid, signum)
        self.assertTrue(self.wait_for(lambda: self.proc.poll() is not None, 5.0))
        time.sleep(0.2)
        self.assertEqual(self.modes(), before, "the terminal was left in lume's mode")
        seen = bytes(self.buf)
        self.assertGreater(seen.rfind(b"\x1b[?2004l"), seen.rfind(b"\x1b[?2004h"),
                           "bracketed paste was left on")

    def test_sigterm_puts_the_terminal_back(self):
        import signal as _signal
        self.check_restored(_signal.SIGTERM)

    def test_sighup_puts_the_terminal_back(self):
        import signal as _signal
        self.check_restored(_signal.SIGHUP)

    def test_a_normal_exit_puts_the_terminal_back(self):
        before = self.baseline()
        self.start(NET="1")
        self.send(b"\x04", pause=0)                 # Ctrl-D on an empty prompt
        self.assertTrue(self.wait_for(lambda: self.proc.poll() is not None, 5.0))
        self.assertEqual(self.modes(), before)


class HistorySafetyTests(PromptTestCase):
    """The history file is a file lume owns — and nothing else."""

    def setUp(self):
        super().setUp()
        self.dir = Path(tempfile.mkdtemp())

    def make_hist(self, path, **kw):
        return self.make(history_path=path, stdin=PipeStdin(""), **kw)

    @POSIX_MODES
    def test_a_symlink_at_the_history_path_is_refused(self):
        # The path is predictable; without O_NOFOLLOW lume appended the user's
        # lines into whatever the link pointed at, and chmodded it to 0600.
        victim = self.dir / "victim.txt"
        victim.write_text("important\n")
        os.chmod(victim, 0o644)
        link = self.dir / "history"
        os.symlink(victim, link)
        p = self.make_hist(link)
        p.add_history("hello from lume")
        p.close()
        self.assertEqual(victim.read_text(), "important\n")
        self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o644)
        self.assertIsNone(p.history_path)

    @POSIX_MODES
    def test_an_existing_directory_keeps_its_mode(self):
        # LUME_HISTORY=$HOME/hist used to chmod $HOME to 0700.
        shared = self.dir / "shared"
        shared.mkdir()
        os.chmod(shared, 0o755)
        p = self.make_hist(shared / "history")
        p.add_history("entry")
        p.close()
        self.assertEqual(stat.S_IMODE(shared.stat().st_mode), 0o755)

    @POSIX_MODES
    def test_a_read_only_directory_is_not_forced_open(self):
        ro = self.dir / "ro"
        ro.mkdir()
        os.chmod(ro, 0o500)
        self.addCleanup(os.chmod, ro, 0o700)
        p = self.make_hist(ro / "history")
        p.add_history("still fine")
        p.close()
        self.assertEqual(stat.S_IMODE(ro.stat().st_mode), 0o500)
        self.assertFalse((ro / "history").exists())
        self.assertIsNone(p.history_path)
        self.assertEqual(p.history(), ["still fine"])     # remembered, not written

    @POSIX_MODES
    def test_a_read_only_history_file_is_never_loosened(self):
        path = self.dir / "history"
        path.write_text("old\n")
        os.chmod(path, 0o400)
        p = self.make_hist(path)
        p.add_history("new entry")
        p.close()
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
        self.assertEqual(path.read_text(), "old\n")
        self.assertIn("old", p.history())                 # still readable

    @POSIX_MODES
    def test_a_loose_mode_is_tightened_on_the_descriptor(self):
        path = self.dir / "history"
        path.write_text("old\n")
        os.chmod(path, 0o666)
        p = self.make_hist(path)
        p.add_history("mine")
        p.close()
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    @POSIX_MODES
    def test_a_write_only_history_file_does_not_gain_read_permission(self):
        # The mode is only ever tightened: 0600 would have *added* the read bit
        # the owner deliberately took away.
        path = self.dir / "history"
        path.touch()
        os.chmod(path, 0o200)
        p = self.make_hist(path)
        p.add_history("appended")
        p.close()
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o200)

    def test_a_fifo_at_the_history_path_is_refused(self):
        if not hasattr(os, "mkfifo"):                     # pragma: no cover
            self.skipTest("no mkfifo")
        fifo = self.dir / "history"
        os.mkfifo(fifo)
        p = self.make_hist(fifo)
        p.add_history("x")
        p.close()
        self.assertIsNone(p.history_path)

    def test_a_directory_at_the_history_path_is_refused(self):
        (self.dir / "history").mkdir()
        p = self.make_hist(self.dir / "history")
        p.add_history("x")
        p.close()
        self.assertIsNone(p.history_path)

    def test_a_second_lume_loses_nothing_while_this_one_trims(self):
        # Two windows is the normal case. The trim used to be read-filter-write-
        # replace with no lock, so anything appended in that window went with the
        # orphaned inode.
        path = self.dir / "history"
        path.write_text("")
        child = subprocess.Popen(
            [sys.executable, "-c", """
import sys, io, time
sys.path.insert(0, %r)
from lume.ansi import Caps, Console
from lume.input import Prompt
from lume.theme import get_theme
caps = Caps(color=0, unicode=False, is_tty=False, width=80, height=24)
p = Prompt(Console(io.StringIO(), caps), get_theme(), history_path=%r,
           stdin=io.StringIO(""), history_max=10000)
for i in range(60):
    p.add_history("child-%%02d" %% i)
    time.sleep(0.004)
p.close()
""" % (ROOT, str(path))])
        self.addCleanup(child.kill)
        trimmer = self.make_hist(path, history_max=10000)
        deadline = time.monotonic() + 8.0
        while child.poll() is None and time.monotonic() < deadline:
            trimmer._trim_history_file()
            time.sleep(0.002)
        child.wait(timeout=5)
        lines = path.read_text().splitlines()
        missing = [f"child-{i:02d}" for i in range(60)
                   if f"child-{i:02d}" not in lines]
        self.assertEqual(missing, [], f"lost {len(missing)} entries")


class SecretShapeTests(unittest.TestCase):
    """The shapes that were written to disk verbatim before round two."""

    POSITIVES = (
        "sk-ant-api03-AAAABBBBCCCCDDDD",
        "SK-ANT-API03-AAAABBBBCCCCDDDD",
        "Sk-Ant-Api03-AAAABBBBCCCCDDDD",
        "export ANTHROPIC_AUTH_TOKEN=abc123def456ghi789",
        "ANTHROPIC_AUTH_TOKEN=sk-live-9f2b7c1d4e8a",
        "Bearer eyJhbGciOiJIUzI1NiJ9.abcdefgh.ijklmnop",
        "Authorization: Bearer abc123def456",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "sk-proj-abc123def456ghi789jkl",
        "github_pat_11ABCDEFG0abcdefghijklmnop",
        "ghp_0123456789abcdefghijklmnopqrst",
        "AIzaSyA0000000000000000000000000000000",
        "AKIAIOSFODNN7EXAMPLE",
        "postgres://user:s3cr3tpassword@host/db",
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdz",
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "x-api-key: 0123456789abcdef0123",
        "-H 'x-api-key: sk-ant-api03-ZZZZWWWW'",
        # Round three: every one of these reached the file in cleartext. The
        # separator was a literal '-', so the whole sk_/rk_/pk_ family walked
        # past; xox[abprs] missed xoxc and xoxe; there was no xapp- or npm_ at
        # all; and the key/value noun list had no Account, Client or Master.
        "sk_live_EXAMPLE_NOT_A_REAL_STRIPE_KEY",
        "STRIPE_SECRET=sk_live_EXAMPLE_NOT_A_REAL_STRIPE_KEY",
        "rk_test_51H8xYzAbCdEfGhIjKlMnOp",
        "pk_live_51H8xYzAbCdEfGhIjKlMnOp",
        "xoxc-1234567890-1234567890123-abcdefghijkl",
        "xoxe-1-EXAMPLE-NOT-A-REAL-SLACK-TOKEN",
        "xapp-1-A01234ABCDE-1234567890123-abcdef0123456789abcdef",
        "npm_AbCdEf0123456789AbCdEf0123456789ab",
        "AccountKey=abcd1234efgh5678ijkl==",
        "DefaultEndpointsProtocol=https;AccountKey=abcd1234efgh5678ijkl==",
        "ClientSecret: 7Q~aB3dEfGhIjKlMnOpQrStUvWx",
        "account_password = correcthorsebattery",   # no digits: the noun carries it
        "master-key: opensesameplease",
        "consumer_key = 9f8e7d6c5b4a39281706",
        "MasterKey=Zm9vYmFyMTIzNDU2Nzg5MA==",
        "glpat-EXAMPLE-NOT-A-REAL-GITLAB-TOKEN",
        "ya29.a0AfH6SMBx1234567890abcdefghij",
    )

    NEGATIVES = (
        "", "what is sk-ant?", "tell me about api keys", "the sky is blue",
        "/model sonnet", "explain how bearer tokens work",
        "the password field should be masked",
        "https://example.com/docs/authentication",
        "a5f3c9d1b7e2a5f3c9d1b7e2a5f3c9d1",          # a plain hex digest
        # Round three: re.IGNORECASE over the whole pattern made the private-key
        # rule's mixed-case tests match either case, so it decayed into "forty
        # base64 characters" and swallowed these — out of the file *and* out of
        # memory, so Up could not get them back, and nothing was said.
        "e83c5163316f89bfbde7d9ab23ca2e25604af290",   # a git SHA
        "fix regression in e83c5163316f89bfbde7d9ab23ca2e25604af290",
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "E83C5163316F89BFBDE7D9AB23CA2E25604AF290",
        "const apiKey = process.env.API_KEY;",
        "api_key = os.environ['ANTHROPIC_API_KEY']",
        'export ANTHROPIC_API_KEY="$MY_KEY"',
        "password = <your password here>",
        "token: ***",
        "monkey = bananas",
        "why does the donkey key not work",
    )

    def test_positives(self):
        for text in self.POSITIVES:
            self.assertTrue(looks_secret(text), text)

    def test_negatives(self):
        for text in self.NEGATIVES:
            self.assertFalse(looks_secret(text), text)

    def test_a_secret_on_any_line_condemns_the_whole_entry(self):
        entry = "here is the key\nsk-ant-api03-AAAABBBBCCCCDDDD\nthanks"
        self.assertTrue(looks_secret(entry))


class SecretsAndTheFileTests(PromptTestCase):
    def setUp(self):
        super().setUp()
        self.path = Path(tempfile.mkdtemp()) / "history"

    def test_no_shape_reaches_the_file(self):
        p = self.make(history_path=self.path, stdin=PipeStdin(""))
        for text in SecretShapeTests.POSITIVES:
            p.add_history(text)
        p.close()
        body = self.path.read_text()
        self.assertEqual(body, "")

    def test_a_secret_buried_in_a_multiline_paste_is_not_remembered(self):
        p = self.make(history_path=self.path, stdin=PipeStdin(""))
        p.add_history("first line\nANTHROPIC_AUTH_TOKEN=sk-live-9f2b7c1d4e8a\nlast")
        p.close()
        self.assertEqual(p.history(), [])
        self.assertEqual(self.path.read_text(), "")

    def test_secrets_left_in_the_file_by_an_older_lume_are_dropped_on_trim(self):
        self.path.write_text("keep me\nsk-ant-api03-LEFTBEHIND\nkeep me too\n")
        p = self.make(history_path=self.path, stdin=PipeStdin(""), history_max=10)
        p._trim_history_file()
        body = self.path.read_text()
        self.assertNotIn("sk-ant-", body)
        self.assertIn("keep me", body)
        self.assertEqual(p.history(), ["keep me", "keep me too"])


class SecretScanCostTests(unittest.TestCase):
    """looks_secret runs inside read(), so it is a latency budget, not just a filter.

    The base64 rule had two lookaheads that rescanned the whole run at every
    start position: a 10,000-character run of letters cost 0.54 s, 30,000 cost
    4.78 s and 60,000 cost 19.28 s — with the prompt frozen, because read()
    calls add_history before it returns. A FASTA sequence does it. So does a
    base32 blob, and so does _load_history over every line of the file.
    """

    def elapsed(self, text):
        start = time.monotonic()
        looks_secret(text)
        return time.monotonic() - start

    def test_a_long_run_of_letters_is_cheap(self):
        for n in (10_000, 30_000, 60_000):
            with self.subTest(n=n):
                self.assertLess(self.elapsed("ACGT" * (n // 4)), 0.25, n)

    def test_it_is_linear_not_quadratic(self):
        small = self.elapsed("ACGT" * 2500) + 0.001
        large = self.elapsed("ACGT" * 20000) + 0.001
        self.assertLess(large / small, 40, (small, large))

    def test_a_long_line_of_mixed_junk_is_cheap(self):
        body = ("aA1+/" * 12000)
        self.assertLess(self.elapsed(body), 0.25)

    def test_a_big_multiline_paste_is_cheap(self):
        doc = "\n".join("word " * 12 for _ in range(4000))
        self.assertLess(self.elapsed(doc), 0.5)

    def test_a_secret_at_the_end_of_a_big_paste_is_still_caught(self):
        doc = "\n".join("line %d" % i for i in range(3000))
        self.assertTrue(looks_secret(doc + "\nsk-ant-api03-AAAABBBBCCCCDDDD"))

    def test_reading_a_long_line_does_not_stall_the_prompt(self):
        p = Prompt(Console(io.StringIO(), PIPE_CAPS), get_theme("aurora"),
                   history_path=False, stdin=PipeStdin("A" * 60000 + "\n"))
        self.addCleanup(p.close)
        start = time.monotonic()
        self.assertEqual(len(p.read()), 60000)
        self.assertLess(time.monotonic() - start, 1.0)


class OrdinaryTextIsRememberedTests(PromptTestCase):
    """The other half of the secret filter: what it must *not* take away.

    A dropped entry is silent — it never reaches the file and it never reaches
    memory, so Up cannot get it back and nothing says why. A git SHA, a sha256
    digest and a line of code that only mentions a key are all ordinary things
    to type at a chat prompt.
    """

    def setUp(self):
        super().setUp()
        self.path = Path(tempfile.mkdtemp()) / "history"

    def keeps(self, text):
        p = self.make(history_path=self.path, stdin=PipeStdin(""))
        p.add_history(text)
        p.close()
        self.assertIn(text, p.history(), "Up could not get it back")
        self.assertIn(text, self.path.read_text(), "it never reached the file")

    def test_a_git_sha(self):
        self.keeps("revert e83c5163316f89bfbde7d9ab23ca2e25604af290")

    def test_a_bare_sha(self):
        self.keeps("e83c5163316f89bfbde7d9ab23ca2e25604af290")

    def test_a_sha256_digest(self):
        self.keeps("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")

    def test_an_upper_case_digest(self):
        self.keeps("E83C5163316F89BFBDE7D9AB23CA2E25604AF290")

    def test_code_that_only_names_a_key(self):
        self.keeps("const apiKey = process.env.API_KEY;")

    def test_a_shell_line_that_passes_one_through(self):
        self.keeps('export ANTHROPIC_API_KEY="$MY_KEY"')

    def test_a_placeholder(self):
        self.keeps("password = <your password here>")

    def test_an_ordinary_sentence_with_key_in_a_word(self):
        self.keeps("monkey = bananas")

    def test_and_a_real_secret_in_the_same_shapes_is_still_dropped(self):
        p = self.make(history_path=self.path, stdin=PipeStdin(""))
        p.add_history("e83c5163316f89bfbde7d9ab23ca2e25604af290 sk_live_EXAMPLE_NOT_A_REAL_KEY")
        p.close()
        self.assertEqual(p.history(), [])
        self.assertEqual(self.path.read_text(), "")


class HistoryFileCapTests(PromptTestCase):
    """SPEC says the history file is size-capped. A line cap is not a size cap.

    Fifty 110 KB pastes made it 5.5 MB, and the next start-up spent 1.88 s
    reading and secret-scanning it (0.004 s is normal). Framing made it worse by
    reliably delivering big pastes whole.
    """

    def setUp(self):
        super().setUp()
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "history"

    def make_hist(self, **kw):
        return self.make(history_path=self.path, stdin=PipeStdin(""), **kw)

    def test_an_over_long_entry_is_remembered_but_not_written(self):
        p = self.make_hist()
        big = "x" * (inp.HISTORY_ENTRY_MAX + 1)
        p.add_history(big)
        p.add_history("small")
        p.close()
        self.assertIn(big, p.history(), "Up must still recall it")
        body = self.path.read_text()
        self.assertNotIn("x" * 100, body)
        self.assertIn("small", body)

    def test_an_entry_right_on_the_limit_is_written(self):
        p = self.make_hist()
        p.add_history("y" * inp.HISTORY_ENTRY_MAX)
        p.close()
        self.assertIn("y" * 100, self.path.read_text())

    def test_the_file_stays_under_the_byte_cap(self):
        p = self.make_hist()
        entry = "z" * (inp.HISTORY_ENTRY_MAX - 5)
        for i in range(200):                     # 1.6 MB of entries
            p.add_history("%04d%s" % (i, entry))
            self.assertLessEqual(self.path.stat().st_size,
                                 inp.HISTORY_BYTES + inp.HISTORY_BYTES // 4,
                                 "it went over the cap mid-session")
        p.close()
        size = self.path.stat().st_size
        self.assertLessEqual(size, inp.HISTORY_BYTES + inp.HISTORY_BYTES // 4, size)
        self.assertIn("0199", self.path.read_text(), "the newest entry was lost")

    def test_the_cap_survives_a_restart(self):
        p = self.make_hist()
        for i in range(200):
            p.add_history("%04d%s" % (i, "q" * (inp.HISTORY_ENTRY_MAX - 5)))
        p.close()
        start = time.monotonic()
        again = self.make_hist()
        self.assertLess(time.monotonic() - start, 1.0, "start-up read a huge file")
        again.close()
        self.assertLessEqual(self.path.stat().st_size,
                             inp.HISTORY_BYTES + inp.HISTORY_BYTES // 4)

    def test_a_file_already_over_the_cap_is_trimmed_at_start_up(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join("old %06d\n" % i for i in range(200000)))
        self.assertGreater(self.path.stat().st_size, inp.HISTORY_BYTES)
        p = self.make_hist()
        p.close()
        self.assertLessEqual(self.path.stat().st_size,
                             inp.HISTORY_BYTES + inp.HISTORY_BYTES // 4)

    def test_the_line_cap_still_applies(self):
        p = self.make_hist(history_max=5)
        for i in range(40):
            p.add_history("entry %d" % i)
        p.close()
        lines = self.path.read_text().splitlines()
        self.assertLessEqual(len(lines), 5)
        self.assertEqual(lines[-1], "entry 39")

    def test_a_small_cap_is_honoured_exactly(self):
        with mock.patch.object(inp, "HISTORY_BYTES", 2048):
            p = self.make_hist()
            for i in range(400):
                p.add_history("%03d %s" % (i, "w" * 40))
            p.close()
        self.assertLessEqual(self.path.stat().st_size, 2048 + 2048 // 4)
        self.assertIn("399", self.path.read_text())


class SecretsAlreadyOnDiskTests(PromptTestCase):
    """A secret an older lume wrote is recognised at start-up — and was left there.

    _load_history filtered them out of *memory* and never touched the file, so a
    key written before the filter existed stayed in cleartext on disk for good.
    """

    def setUp(self):
        super().setUp()
        self.path = Path(tempfile.mkdtemp()) / "history"

    def make_hist(self, **kw):
        return self.make(history_path=self.path, stdin=PipeStdin(""), **kw)

    def test_loading_rewrites_the_file_without_them(self):
        self.path.write_text("keep me\nsk-ant-api03-LEFTBEHIND\nkeep me too\n")
        p = self.make_hist()
        body = self.path.read_text()
        self.assertNotIn("sk-ant-api03-LEFTBEHIND", body)
        self.assertEqual(body.splitlines(), ["keep me", "keep me too"])
        self.assertEqual(p.history(), ["keep me", "keep me too"])
        p.close()

    def test_every_shape_is_cleaned_out(self):
        self.path.write_text("".join(x + "\n" for x in SecretShapeTests.POSITIVES))
        p = self.make_hist()
        p.close()
        self.assertEqual(self.path.read_text(), "")

    def test_a_clean_file_is_left_alone(self):
        self.path.write_text("one\ntwo\n")
        before = self.path.stat().st_mtime_ns
        p = self.make_hist()
        p.close()
        self.assertEqual(self.path.read_text(), "one\ntwo\n")
        self.assertEqual(self.path.stat().st_mtime_ns, before, "rewritten for nothing")

    def test_a_read_only_file_is_not_a_crash_and_history_still_works(self):
        self.path.write_text("keep me\nsk-ant-api03-LEFTBEHIND\n")
        os.chmod(self.path, 0o400)
        self.addCleanup(os.chmod, self.path, 0o600)
        p = self.make_hist()
        self.assertEqual(p.history(), ["keep me"])
        p.add_history("new one")
        self.assertIn("new one", p.history())
        p.close()


class TrimNeedsTheLockTests(PromptTestCase):
    """A trim that could not take the lock used to rewrite the file anyway.

    Appends are safe whatever happens — O_APPEND is atomic — but the trim is a
    read, filter and write, and another lume's entry that landed in between was
    destroyed by it. A trim is always deferrable; an append never is.
    """

    def setUp(self):
        super().setUp()
        self.path = Path(tempfile.mkdtemp()) / "history"

    def make_hist(self, **kw):
        return self.make(history_path=self.path, stdin=PipeStdin(""), **kw)

    def test_an_unlockable_trim_is_skipped_not_forced(self):
        p = self.make_hist(history_max=5)
        for i in range(30):
            p.add_history("a%02d" % i)
        before = self.path.read_text()
        self.assertGreater(len(before.splitlines()), 5, "no trim was pending")
        with mock.patch.object(Prompt, "_lock", staticmethod(lambda fd, timeout=1.0: False)):
            p._trim_history_file()
        self.assertEqual(self.path.read_text(), before)

    def test_nothing_another_process_appended_is_lost(self):
        a = self.make_hist(history_max=5)
        b = self.make_hist(history_max=5)
        for i in range(30):
            a.add_history("a%02d" % i)
        with mock.patch.object(Prompt, "_lock", staticmethod(lambda fd, timeout=1.0: False)):
            a._trim_history_file()
        b.add_history("b-from-the-other-lume")
        self.assertIn("b-from-the-other-lume", self.path.read_text())

    def test_and_the_trim_still_happens_when_the_lock_is_free(self):
        p = self.make_hist(history_max=5)
        for i in range(30):
            p.add_history("a%02d" % i)
        p._trim_history_file()
        self.assertEqual(len(self.path.read_text().splitlines()), 5)


class CompleterOwnershipTests(unittest.TestCase):
    """close() used to set the *global* readline completer to None.

    Two live prompts (an embedder's, or a nested one) and closing either
    disarmed Tab for both.
    """

    def setUp(self):
        self.rl = FakeReadline()
        patch = mock.patch.object(inp, "readline", self.rl)
        patch.start()
        self.addCleanup(patch.stop)
        self.theme = get_theme("aurora")

    def prompt(self):
        return Prompt(Console(io.StringIO(), TTY_CAPS), self.theme,
                      history_path=False, stdin=PipeStdin(""))

    def test_closing_one_prompt_leaves_the_other_armed(self):
        first = self.prompt()
        second = self.prompt()
        first.close()
        self.assertIs(self.rl.completer, second._completer_fn)
        second.close()
        self.assertIsNone(self.rl.completer)

    def test_a_prompt_still_detaches_its_own(self):
        only = self.prompt()
        self.assertIs(self.rl.completer, only._completer_fn)
        only.close()
        self.assertIsNone(self.rl.completer)


if __name__ == "__main__":
    unittest.main()
