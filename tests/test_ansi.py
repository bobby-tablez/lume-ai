"""Tests for the terminal foundation: capabilities, colour, width, wrapping, Console."""

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lume import ansi
from lume.ansi import Caps, Console, Style


def caps(**kw):
    base = dict(color=24, unicode=True, is_tty=True, width=80, height=24,
                hyperlinks=True, animation=True)
    base.update(kw)
    return Caps(**base)


class FakeStream:
    """A minimal text stream: StringIO refuses to let us set `.encoding`."""

    def __init__(self, tty=True, encoding="utf-8"):
        self._buf = io.StringIO()
        self._tty = tty
        self.encoding = encoding

    def isatty(self):
        return self._tty

    def write(self, s):
        return self._buf.write(s)

    def flush(self):
        self._buf.flush()

    def close(self):
        self._buf.close()

    def getvalue(self):
        return self._buf.getvalue()


class TestDetectCaps(unittest.TestCase):
    def test_non_tty_has_no_colour(self):
        c = ansi.detect_caps(FakeStream(tty=False), env={"TERM": "xterm-256color"})
        self.assertEqual(c.color, 0)
        self.assertFalse(c.is_tty)
        self.assertFalse(c.animation)

    def test_no_color_env_wins_over_truecolor(self):
        env = {"TERM": "xterm-256color", "COLORTERM": "truecolor", "NO_COLOR": "1"}
        self.assertEqual(ansi.detect_caps(FakeStream(), env=env).color, 0)

    def test_no_color_empty_value_is_not_set(self):
        # NO_COLOR="" is treated as unset, matching the common convention here.
        env = {"TERM": "xterm-256color", "COLORTERM": "truecolor", "NO_COLOR": ""}
        self.assertEqual(ansi.detect_caps(FakeStream(), env=env).color, 24)

    def test_force_color_levels_on_a_pipe(self):
        # chalk/supports-color convention: 1 = 16 colours, 2 = 256, 3 = truecolor.
        for level, want in (("1", 4), ("2", 8), ("3", 24)):
            env = {"TERM": "xterm-256color", "FORCE_COLOR": level}
            self.assertEqual(ansi.detect_caps(FakeStream(tty=False), env=env).color, want)

    def test_no_color_any_nonempty_value_disables(self):
        for value in ("1", "0", "false", "anything"):
            env = {"TERM": "xterm-256color", "COLORTERM": "truecolor", "NO_COLOR": value}
            self.assertEqual(ansi.detect_caps(FakeStream(), env=env).color, 0, value)

    def test_clicolor_zero_disables(self):
        env = {"TERM": "xterm-256color", "CLICOLOR": "0"}
        self.assertEqual(ansi.detect_caps(FakeStream(), env=env).color, 0)

    def test_dumb_terminal_cannot_be_forced(self):
        env = {"TERM": "dumb", "FORCE_COLOR": "3", "CLICOLOR_FORCE": "1"}
        self.assertEqual(ansi.detect_caps(FakeStream(), env=env).color, 0)

    def test_direct_terminfo_is_truecolor(self):
        self.assertEqual(ansi.detect_caps(FakeStream(), env={"TERM": "xterm-direct"}).color, 24)

    def test_size_honours_injected_env(self):
        c = ansi.detect_caps(FakeStream(), env={"TERM": "xterm", "COLUMNS": "37", "LINES": "11"})
        self.assertEqual((c.width, c.height), (37, 11))

    def test_hyperlinks_survive_no_color(self):
        env = {"TERM": "xterm-256color", "NO_COLOR": "1"}
        self.assertTrue(ansi.detect_caps(FakeStream(), env=env).hyperlinks)

    def test_force_color_zero_disables(self):
        env = {"TERM": "xterm-256color", "COLORTERM": "truecolor", "FORCE_COLOR": "0"}
        self.assertEqual(ansi.detect_caps(FakeStream(), env=env).color, 0)

    def test_dumb_terminal(self):
        c = ansi.detect_caps(FakeStream(), env={"TERM": "dumb"})
        self.assertEqual(c.color, 0)
        self.assertFalse(c.animation)

    def test_256_colour_terminal(self):
        self.assertEqual(ansi.detect_caps(FakeStream(), env={"TERM": "xterm-256color"}).color, 8)

    def test_unicode_from_locale_and_encoding(self):
        self.assertTrue(ansi.detect_caps(FakeStream(encoding="UTF-8"), env={"TERM": "xterm"}).unicode)
        self.assertFalse(ansi.detect_caps(FakeStream(encoding="ascii"),
                                          env={"TERM": "xterm", "LANG": "C"}).unicode)

    def test_lume_no_motion(self):
        env = {"TERM": "xterm-256color", "COLORTERM": "truecolor", "LUME_NO_MOTION": "1"}
        c = ansi.detect_caps(FakeStream(), env=env)
        self.assertTrue(c.color)
        self.assertFalse(c.animation)


class TestColour(unittest.TestCase):
    def test_hex_rgb_forms(self):
        self.assertEqual(ansi.hex_rgb("#1f2c3d"), (31, 44, 61))
        self.assertEqual(ansi.hex_rgb("1f2c3d"), (31, 44, 61))
        self.assertEqual(ansi.hex_rgb("#abc"), (170, 187, 204))
        with self.assertRaises(ValueError):
            ansi.hex_rgb("#12345")

    def test_blend_clamps(self):
        self.assertEqual(ansi.blend((0, 0, 0), (100, 100, 100), 0.5), (50, 50, 50))
        self.assertEqual(ansi.blend((0, 0, 0), (100, 100, 100), -5), (0, 0, 0))
        self.assertEqual(ansi.blend((0, 0, 0), (100, 100, 100), 5), (100, 100, 100))

    def test_gradient_endpoints_and_length(self):
        g = ansi.gradient([(0, 0, 0), (255, 255, 255)], 5)
        self.assertEqual(len(g), 5)
        self.assertEqual(g[0], (0, 0, 0))
        self.assertEqual(g[-1], (255, 255, 255))
        self.assertEqual(ansi.gradient([(1, 2, 3)], 3), [(1, 2, 3)] * 3)
        self.assertEqual(ansi.gradient([(0, 0, 0), (9, 9, 9)], 0), [])

    def test_gradient_multi_stop_hits_middle_stop(self):
        g = ansi.gradient([(0, 0, 0), (10, 10, 10), (20, 20, 20)], 3)
        self.assertEqual(g[1], (10, 10, 10))

    def test_depth_downgrade(self):
        red = (255, 0, 0)
        self.assertIn("38;2;255;0;0", ansi.fg(red, caps(color=24)))
        self.assertIn("38;5;", ansi.fg(red, caps(color=8)))
        self.assertTrue(ansi.fg(red, caps(color=4)).startswith("\x1b["))
        self.assertEqual(ansi.fg(red, caps(color=0)), "")

    def test_grey_maps_to_greyscale_ramp(self):
        self.assertGreaterEqual(ansi.rgb_to_256((128, 128, 130)), 232)

    def test_quantisation_is_perceptually_optimal(self):
        """The whole point of Lab nearest-neighbour: never a suboptimal choice."""
        import random

        def de(a, b):
            la, lb = ansi._srgb_to_lab(a), ansi._srgb_to_lab(b)
            return sum((la[i] - lb[i]) ** 2 for i in range(3)) ** 0.5

        rng = random.Random(1234)
        for _ in range(300):
            c = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            got = de(c, ansi._XTERM256[ansi.rgb_to_256(c) - 16])
            self.assertLessEqual(got, min(de(c, p) for p in ansi._XTERM256) + 1e-9)
            got16 = de(c, ansi._ANSI16[ansi.rgb_to_16(c)])
            self.assertLessEqual(got16, min(de(c, p) for p in ansi._ANSI16) + 1e-9)

    def test_grey_and_black_map_sensibly(self):
        self.assertEqual(ansi.rgb_to_16((5, 5, 5)), 0)
        self.assertEqual(ansi._XTERM256[ansi.rgb_to_256((255, 255, 255)) - 16], (255, 255, 255))

    def test_dark_neutral_does_not_become_navy(self):
        r, g, b = ansi._XTERM256[ansi.rgb_to_256((31, 36, 48)) - 16]
        self.assertLess(max(r, g, b) - min(r, g, b), 40)


class TestStyle(unittest.TestCase):
    def test_render_and_reset(self):
        out = Style(fg=(255, 0, 0), bold=True)("hi", caps())
        self.assertTrue(out.endswith(ansi.RESET))
        self.assertEqual(ansi.strip_ansi(out), "hi")

    def test_empty_text_is_untouched(self):
        self.assertEqual(Style(fg=(1, 2, 3))("", caps()), "")

    def test_no_colour_keeps_attributes_only(self):
        out = Style(fg=(255, 0, 0), bold=True)("hi", caps(color=0))
        self.assertNotIn("38;2", out)
        self.assertIn("1m", out)
        self.assertEqual(ansi.strip_ansi(out), "hi")

    def test_no_colour_non_tty_is_plain(self):
        self.assertEqual(Style(fg=(255, 0, 0), bold=True)("hi", caps(color=0, is_tty=False)), "hi")

    def test_merge_right_wins_on_colour_and_ors_attributes(self):
        a = Style(fg=(1, 1, 1), bold=True)
        b = Style(fg=(2, 2, 2), italic=True)
        m = a + b
        self.assertEqual(m.fg, (2, 2, 2))
        self.assertTrue(m.bold and m.italic)

    def test_merge_keeps_left_colour_when_right_unset(self):
        self.assertEqual((Style(fg=(1, 1, 1)) + Style(italic=True)).fg, (1, 1, 1))


class TestWidth(unittest.TestCase):
    def test_strip_ansi_removes_sgr_and_osc(self):
        self.assertEqual(ansi.strip_ansi("\x1b[1;31mhi\x1b[0m"), "hi")
        self.assertEqual(ansi.strip_ansi("\x1b]8;;http://x\x1b\\hi\x1b]8;;\x1b\\"), "hi")

    def test_ascii_width(self):
        self.assertEqual(ansi.display_width("hello"), 5)

    def test_wide_east_asian(self):
        self.assertEqual(ansi.display_width("日本語"), 6)

    def test_combining_marks_are_zero_width(self):
        self.assertEqual(ansi.display_width("é"), 1)

    def test_zero_width_joiner_and_variation_selector(self):
        self.assertEqual(ansi.display_width("‍"), 0)
        self.assertEqual(ansi.display_width("️"), 0)

    def test_styled_text_width_ignores_escapes(self):
        self.assertEqual(ansi.display_width(Style(fg=(1, 2, 3))("abc", caps())), 3)

    def test_control_characters_are_zero(self):
        self.assertEqual(ansi.display_width("\x00\x07"), 0)


class TestWrap(unittest.TestCase):
    def test_basic_word_wrap(self):
        self.assertEqual(ansi.wrap("the quick brown fox", 9), ["the quick", "brown fox"])

    def test_every_line_fits(self):
        text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod"
        for w in (10, 17, 23, 40):
            for line in ansi.wrap(text, w):
                self.assertLessEqual(ansi.display_width(line), w, f"width {w}: {line!r}")

    def test_indents(self):
        lines = ansi.wrap("alpha beta gamma delta", 12, initial_indent="> ", subsequent_indent="  ")
        self.assertTrue(lines[0].startswith("> "))
        self.assertTrue(lines[1].startswith("  "))
        for line in lines:
            self.assertLessEqual(ansi.display_width(line), 12)

    def test_over_long_word_is_hard_split(self):
        lines = ansi.wrap("x" * 25, 10)
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertLessEqual(ansi.display_width(line), 10)

    def test_explicit_newlines_break_lines(self):
        self.assertEqual(ansi.wrap("a\nb", 20), ["a", "b"])

    def test_ansi_state_carried_across_break(self):
        styled = "\x1b[31m" + "alpha beta gamma" + ansi.RESET
        lines = ansi.wrap(styled, 8)
        self.assertGreater(len(lines), 1)
        self.assertIn("\x1b[31m", lines[1])
        self.assertEqual(" ".join(ansi.strip_ansi(l) for l in lines), "alpha beta gamma")

    def test_no_text_is_lost(self):
        text = "one two three four five six seven eight nine ten"
        for w in (6, 11, 20, 33):
            joined = " ".join(ansi.strip_ansi(l).strip() for l in ansi.wrap(text, w))
            self.assertEqual(joined.split(), text.split(), f"width {w}")

    def test_wide_characters_respect_width(self):
        for line in ansi.wrap("日本語 テスト です", 8):
            self.assertLessEqual(ansi.display_width(line), 8)

    def test_zero_and_negative_width_do_not_hang(self):
        self.assertTrue(ansi.wrap("abc def", 0))
        self.assertTrue(ansi.wrap("abc def", -3))


class TestTruncateAndPad(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(ansi.truncate("abc", 10), "abc")

    def test_truncate_respects_width(self):
        out = ansi.truncate("abcdefghij", 5)
        self.assertLessEqual(ansi.display_width(out), 5)
        self.assertTrue(out.endswith("…"))

    def test_truncate_preserves_style_and_resets(self):
        styled = Style(fg=(0, 255, 0))("hello world", caps())
        out = ansi.truncate(styled, 8)
        self.assertTrue(out.endswith(ansi.RESET))
        self.assertLessEqual(ansi.display_width(out), 8)

    def test_truncate_wide_chars_never_overflow(self):
        self.assertLessEqual(ansi.display_width(ansi.truncate("日本語テスト", 5)), 5)

    def test_truncate_zero_width(self):
        self.assertEqual(ansi.truncate("abc", 0), "")

    def test_pad_alignments(self):
        self.assertEqual(ansi.pad("ab", 5), "ab   ")
        self.assertEqual(ansi.pad("ab", 5, "right"), "   ab")
        self.assertEqual(ansi.pad("ab", 5, "center"), " ab  ")

    def test_pad_ignores_escapes_when_measuring(self):
        self.assertEqual(ansi.display_width(ansi.pad(Style(fg=(1, 2, 3))("ab", caps()), 6)), 6)

    def test_pad_never_shrinks(self):
        self.assertEqual(ansi.pad("abcdef", 3), "abcdef")


class TestHyperlink(unittest.TestCase):
    def test_emits_osc8_when_supported(self):
        out = ansi.hyperlink("https://x.dev", "x", caps(hyperlinks=True))
        self.assertIn("\x1b]8;;https://x.dev", out)
        self.assertEqual(ansi.strip_ansi(out), "x")

    def test_falls_back_to_label(self):
        self.assertEqual(ansi.hyperlink("https://x.dev", "x", caps(hyperlinks=False)), "x")


class TestConsole(unittest.TestCase):
    def test_write_and_print(self):
        s = FakeStream()
        c = Console(stream=s, caps=caps())
        c.write("a")
        c.print("b")
        self.assertEqual(s.getvalue(), "ab\n")

    def test_cursor_hide_show_is_idempotent(self):
        s = FakeStream()
        c = Console(stream=s, caps=caps())
        c.hide_cursor()
        c.hide_cursor()
        c.show_cursor()
        c.show_cursor()
        self.assertEqual(s.getvalue().count(ansi.HIDE_CURSOR), 1)
        self.assertEqual(s.getvalue().count(ansi.SHOW_CURSOR), 1)

    def test_cursor_control_is_suppressed_off_tty(self):
        s = FakeStream(tty=False)
        c = Console(stream=s, caps=caps(is_tty=False))
        c.hide_cursor()
        c.clear_line()
        c.clear_screen()
        self.assertEqual(s.getvalue(), "")

    def test_broken_pipe_is_swallowed(self):
        class Broken(FakeStream):
            def write(self, s):
                raise BrokenPipeError()

        Console(stream=Broken(), caps=caps()).write("x")  # must not raise

    def test_closed_stream_is_swallowed(self):
        s = FakeStream()
        s.close()
        Console(stream=s, caps=caps()).write("x")  # must not raise




class TestWrapProperties(unittest.TestCase):
    """Randomised properties. These are what caught the indent-overflow bug."""

    ALPHABET = [
        "a", "bb", "ccc", "word", "longerword", "supercalifragilistic",
        "日本語", "漢字テスト", "café", "naïve", "é", "⚠️", "👍", "🇯🇵",
        "https://example.com/a/very/long/path/that/never/ends", "-", "—", "\t",
    ]

    def _samples(self, seed, n=250):
        import random
        rng = random.Random(seed)
        for _ in range(n):
            parts = []
            for _ in range(rng.randrange(1, 14)):
                parts.append(rng.choice(self.ALPHABET))
                if rng.random() < 0.12:
                    parts.append("\n")
                if rng.random() < 0.15:
                    st = Style(fg=(rng.randrange(256), rng.randrange(256), rng.randrange(256)),
                               bold=rng.random() < 0.4, italic=rng.random() < 0.3)
                    parts.append(st(rng.choice(self.ALPHABET), caps()))
            text = " ".join(parts)
            width = rng.randrange(2, 120)
            ind_a = " " * rng.randrange(0, 6)
            ind_b = " " * rng.randrange(0, 12)
            yield text, width, ind_a, ind_b

    def test_no_line_ever_exceeds_the_requested_width(self):
        for text, width, a, b in self._samples(11):
            for line in ansi.wrap(text, width, initial_indent=a, subsequent_indent=b):
                self.assertLessEqual(
                    ansi.display_width(line), width,
                    f"width={width} initial={a!r} subsequent={b!r} text={text!r} line={line!r}")

    def test_wider_continuation_indent_still_fits(self):
        """The exact shape that overflowed before: long word + wider hanging indent."""
        for line in ansi.wrap("x" * 30, 12, initial_indent="", subsequent_indent="        "):
            self.assertLessEqual(ansi.display_width(line), 12, repr(line))

    def test_no_visible_text_is_lost_or_duplicated(self):
        for text, width, a, b in self._samples(12):
            got = "".join(ansi.strip_ansi(l) for l in ansi.wrap(text, width, a, b))
            self.assertEqual("".join(got.split()), "".join(ansi.strip_ansi(text).split()),
                             f"width={width} text={text!r}")

    def test_no_partial_escape_sequences(self):
        for text, width, a, b in self._samples(13):
            for line in ansi.wrap(text, width, a, b):
                self.assertNotIn("\x1b", ansi.strip_ansi(line), repr(line))

    def test_every_styled_line_closes_its_run(self):
        for text, width, a, b in self._samples(14):
            for line in ansi.wrap(text, width, a, b):
                if "\x1b[" in line:
                    self.assertTrue(line.endswith(ansi.RESET), repr(line))

    def test_state_is_reopened_not_replayed(self):
        """Escape overhead must track the styles ON a line, not the ones before it."""
        chunks = [Style(fg=(i * 7 % 256, 40, 200))(f"w{i}", caps()) for i in range(200)]
        text = " ".join(chunks)
        lines = ansi.wrap(text, 40)
        blowup = len("".join(lines)) / len(text)
        self.assertLess(blowup, 1.6, f"output is {blowup:.1f}x the input")
        overhead = [len(l) - ansi.display_width(l) for l in lines]
        # The last line must not be dramatically more escaped than the first.
        self.assertLess(overhead[-1], 3 * (sum(overhead) / len(overhead)) + 60)


class TestWrapDetails(unittest.TestCase):
    def test_background_does_not_paint_the_preceding_space(self):
        line = ansi.wrap("aaa " + Style(bg=(200, 0, 0))("XX", caps()), 40)[0]
        self.assertIn("aaa \x1b[", line)

    def test_leading_whitespace_of_a_hard_line_is_kept(self):
        self.assertEqual(ansi.wrap("a\n    b", 40), ["a", "    b"])

    def test_soft_wrap_does_not_keep_the_break_space(self):
        for line in ansi.wrap("alpha beta gamma delta", 11):
            self.assertFalse(line.startswith(" "), repr(line))

    def test_tabs_are_expanded_to_columns(self):
        line = ansi.wrap("col1\tcol2", 40)[0]
        self.assertNotIn("\t", line)
        self.assertEqual(ansi.display_width(line), len("col1    col2"))

    def test_hyperlink_is_closed_and_reopened_across_a_break(self):
        link = ansi.hyperlink("https://example.com", "alpha beta gamma delta", caps())
        lines = ansi.wrap(link, 11)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertEqual(line.count("\x1b]8;;"), 2, repr(line))

    def test_wide_glyphs_never_overflow(self):
        for w in range(2, 12):
            for line in ansi.wrap("日本語テストです", w):
                self.assertLessEqual(ansi.display_width(line), w)


class TestWidthDetails(unittest.TestCase):
    def test_emoji_presentation_selector_promotes_to_two_columns(self):
        for seq in ("⚠️", "❤️", "✔️", "▶️", "ℹ️"):
            self.assertEqual(ansi.display_width(seq), 2, seq)

    def test_keycap_is_two_columns(self):
        self.assertEqual(ansi.display_width("1️⃣"), 2)

    def test_text_presentation_stays_narrow(self):
        self.assertEqual(ansi.display_width("⚠"), 1)

    def test_spacing_marks_are_not_zeroed(self):
        self.assertGreaterEqual(ansi.display_width("ः"), 1)

    def test_hangul_jamo_extended_b_is_zero_width(self):
        self.assertEqual(ansi.display_width("ힰ"), 0)


class TestTruncateDetails(unittest.TestCase):
    def test_ellipsis_wider_than_width_is_itself_clipped(self):
        self.assertLessEqual(ansi.display_width(ansi.truncate("abcdefgh", 2, ellipsis="...")), 2)

    def test_pad_rejects_a_wide_filler(self):
        self.assertEqual(ansi.display_width(ansi.pad("ab", 6, fill="日")), 6)

    def test_pad_rejects_an_empty_filler(self):
        self.assertEqual(ansi.display_width(ansi.pad("ab", 6, fill="")), 6)


class TestHyperlinkWrapping(unittest.TestCase):
    """A link opener landing exactly on a break used to be emitted twice."""

    def _link_text(self, label):
        return "alpha beta " + ansi.hyperlink("https://ex.com", label, caps()) + " zeta eta"

    def test_opener_on_a_break_is_not_duplicated(self):
        for width in range(6, 40):
            for label in ("gamma", "gamma delta epsilon", "g"):
                for line in ansi.wrap(self._link_text(label), width):
                    self.assertLessEqual(line.count("\x1b]8;;https"), 1,
                                         f"width={width} {line!r}")

    def test_no_line_closes_a_link_it_never_opened(self):
        for width in range(6, 40):
            for label in ("gamma", "gamma delta epsilon", "g"):
                for line in ansi.wrap(self._link_text(label), width):
                    opens = line.count("\x1b]8;;https")
                    closes = line.count("\x1b]8;;\x1b\\")
                    self.assertFalse(closes and not opens,
                                     f"stray close: width={width} {line!r}")
                    self.assertLessEqual(opens, closes, f"unclosed: {line!r}")

    def test_link_text_survives_wrapping(self):
        text = self._link_text("gamma delta epsilon")
        joined = " ".join(ansi.strip_ansi(l).strip() for l in ansi.wrap(text, 11))
        self.assertEqual(joined.split(), ansi.strip_ansi(text).split())


class TestSanitizeText(unittest.TestCase):
    """Model output is untrusted: it must not be able to drive the terminal."""

    ATTACKS = {
        "clipboard (OSC 52)": "text \x1b]52;c;cHduZWQ=\x07 end",
        "window title": "\x1b]0;PWNED\x07",
        "clear screen": "\x1b[2J\x1b[H",
        "alternate screen": "\x1b[?1049h",
        "DCS passthrough": "\x1bP0;1|xx\x1b\\",
        "8-bit CSI": "\x9b31m",
        "carriage return overwrite": "visible\rOVERWRITTEN",
        "backspace": "abc\x08\x08\x08xyz",
        "NUL": "a\x00b",
    }

    def test_no_escape_survives(self):
        for name, attack in self.ATTACKS.items():
            cleaned = ansi.sanitize_text(attack)
            self.assertNotIn("\x1b", cleaned, name)
            self.assertNotIn("\r", cleaned, name)
            for ch in cleaned:
                self.assertFalse(0 <= ord(ch) <= 8 or 11 <= ord(ch) <= 31
                                 or 127 <= ord(ch) <= 159, f"{name}: {ch!r}")

    def test_ordinary_text_is_untouched(self):
        for good in ("hello world", "日本語 テスト", "⚠️ warning", "a\tb", "line\nline",
                     "emoji 👍🏽 and — dashes", "café"):
            self.assertEqual(ansi.sanitize_text(good), good)

    def test_newlines_can_be_folded(self):
        self.assertEqual(ansi.sanitize_text("a\nb", keep_newlines=False), "a b")

    def test_it_is_idempotent(self):
        for attack in self.ATTACKS.values():
            once = ansi.sanitize_text(attack)
            self.assertEqual(ansi.sanitize_text(once), once)


class TestTransientRestore(unittest.TestCase):
    """A process killed mid-animation must not leave a frozen frame behind."""

    def test_a_transient_line_is_erased_before_the_cursor_returns(self):
        stream = FakeStream()
        console = Console(stream=stream, caps=caps())
        console.hide_cursor()
        console.set_transient(True)
        console.write("| thinking")
        ansi._restore_all_cursors()
        tail = stream.getvalue()
        self.assertIn(ansi.CLEAR_LINE, tail)
        self.assertTrue(tail.endswith(ansi.SHOW_CURSOR))
        self.assertFalse(console._transient)

    def test_a_settled_line_is_not_erased(self):
        stream = FakeStream()
        console = Console(stream=stream, caps=caps())
        console.hide_cursor()
        console.write("final answer")
        ansi._restore_all_cursors()
        self.assertNotIn(ansi.CLEAR_LINE, stream.getvalue())

    def test_sigquit_is_covered_by_the_signal_net(self):
        import subprocess
        code = (
            "import signal, sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parent.parent) +
            "import lume.ansi as a;"
            "a.install_signal_net();"
            "print(signal.getsignal(signal.SIGQUIT) is not signal.SIG_DFL)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        self.assertEqual(out, "True")



class TestConsoleSafety(unittest.TestCase):
    """The three defences a critic proved had no coverage at all."""

    def test_batch_prevents_tearing_between_concurrent_writers(self):
        """Without the lock the frames interleave; the assertion must be able to see it."""
        import threading
        import time

        class SlowStream(FakeStream):
            """One character at a time, so a frame can be torn in the middle."""

            def write(self, text):
                for ch in text:
                    super().write(ch)
                return len(text)

        def run(use_batch):
            stream = SlowStream()
            console = Console(stream=stream, caps=caps())
            # A barrier, not a sleep: whether two threads interleave is up to the
            # scheduler, and a lock released and immediately re-taken by the same
            # thread tears nothing at all -- which failed this test on its own
            # vacuousness check rather than on the lock it is about. Here both
            # writers are *made* to sit between the "<" and the tag, so an
            # unbatched frame tears on every run, on every build.
            gate = threading.Barrier(2, timeout=10)

            def frame(tag):
                for _ in range(60):
                    if use_batch:
                        with console.batch():
                            console.write("<", tag, ">", flush=False)
                    else:
                        console.write("<", flush=False)
                        gate.wait()          # the other writer is mid-frame too
                        console.write(tag, flush=False)
                        console.write(">", flush=False)

            threads = [threading.Thread(target=frame, args=(t,)) for t in "AB"]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return stream.getvalue()

        intact = run(True)
        self.assertEqual(intact.count("<A>") + intact.count("<B>"), 120,
                         f"batched frames tore: {intact[:120]!r}")
        # Sanity: the harness must be capable of observing a tear at all, or the
        # assertion above proves nothing.
        torn = run(False)
        self.assertLess(torn.count("<A>") + torn.count("<B>"), 120,
                        "the harness cannot detect tearing, so the test is vacuous")

    def test_exit_hook_restores_a_hidden_cursor(self):
        stream = FakeStream()
        console = Console(stream=stream, caps=caps())
        console.hide_cursor()
        ansi._restore_all_cursors()
        self.assertTrue(stream.getvalue().endswith(ansi.SHOW_CURSOR))
        self.assertFalse(console._cursor_hidden)

    def test_exit_hook_never_blocks_on_a_held_lock(self):
        """A daemon thread holding the lock at exit must not hang the process."""
        import threading
        import time

        stream = FakeStream()
        console = Console(stream=stream, caps=caps())
        console.hide_cursor()
        release = threading.Event()

        def hog():
            with console.lock:
                release.wait(5)

        holder = threading.Thread(target=hog, daemon=True)
        holder.start()
        time.sleep(0.05)
        started = time.monotonic()
        ansi._restore_all_cursors()
        elapsed = time.monotonic() - started
        release.set()
        holder.join(1)
        self.assertLess(elapsed, 1.0, "restore blocked on the lock")
        self.assertIn(ansi.SHOW_CURSOR, stream.getvalue())

    def test_atexit_restores_the_cursor_on_a_normal_exit(self):
        """Proves the hook is registered, by exiting normally and reading the bytes."""
        import subprocess
        code = (
            "import sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parent.parent) +
            "from lume.ansi import Console, Caps;"
            "c = Console(stream=sys.stdout, caps=Caps(color=24, unicode=True,"
            " is_tty=True, width=80, height=24));"
            "c.hide_cursor()"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(proc.stdout.endswith(ansi.SHOW_CURSOR.encode()),
                        f"cursor leaked on normal exit: {proc.stdout!r}")

    def test_signal_net_is_opt_in_not_an_import_side_effect(self):
        import subprocess
        code = (
            "import signal, sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parent.parent) +
            "signal.signal(signal.SIGHUP, signal.SIG_IGN);"
            "import lume.ansi as a;"
            # The claim being tested: importing installs nothing at all.
            "print(signal.getsignal(signal.SIGTERM) is signal.SIG_DFL);"
            "print(signal.getsignal(signal.SIGHUP) is signal.SIG_IGN);"
            "a.install_signal_net();"
            "print(signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL);"
            "print(signal.getsignal(signal.SIGHUP) is signal.SIG_IGN)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=30).stdout.split()
        self.assertEqual(out, ["True", "True", "True", "True"])

    def test_the_signal_net_chains_to_a_pre_existing_handler(self):
        import subprocess
        code = (
            "import os, signal, sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parent.parent) +
            "import lume.ansi as a;"
            "handled=[];"
            "signal.signal(signal.SIGTERM, lambda s, f: (print('CHAINED'), sys.exit(7)));"
            "a.install_signal_net();"
            "os.kill(os.getpid(), signal.SIGTERM)"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=30)
        self.assertIn("CHAINED", proc.stdout)
        self.assertEqual(proc.returncode, 7)

    def test_cursor_is_restored_after_an_uncaught_exception(self):
        import subprocess
        code = (
            "import sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parent.parent) +
            "from lume.ansi import Console, Caps;"
            "c = Console(stream=sys.stdout, caps=Caps(color=24, unicode=True,"
            " is_tty=True, width=80, height=24));"
            "c.hide_cursor();"
            "raise RuntimeError('boom')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
        self.assertTrue(proc.stdout.endswith(ansi.SHOW_CURSOR.encode()),
                        f"cursor leaked: {proc.stdout!r}")


class TestStyleNesting(unittest.TestCase):
    def test_close_restores_the_enclosing_style(self):
        outer, inner = Style(bold=True), Style(fg=(255, 0, 0))
        c = caps()
        text = outer.open(c) + "a" + inner.open(c) + "b" + inner.close(c, outer) + "c"
        self.assertIn(ansi.RESET + outer.codes(c), text)
        self.assertEqual(ansi.strip_ansi(text), "abc")

    def test_close_of_an_empty_style_emits_nothing(self):
        self.assertEqual(Style().close(caps()), "")


class TestSanitizeUrl(unittest.TestCase):
    """URLs come from model output and are handed straight to the terminal."""

    def test_control_bytes_are_stripped_including_c1(self):
        for bad in ("http://x\x1by", "http://x\x07y", "http://x\x9dy",
                    "http://x\x9cy", "http://x\ny", "http://x\x00y"):
            cleaned = ansi.sanitize_url(bad)
            self.assertEqual(cleaned, "http://xy", repr(bad))

    def test_dangerous_schemes_are_refused(self):
        for bad in ("javascript:alert(1)", "JavaScript:alert(1)", "data:text/html,x",
                    "file:///etc/passwd", "vbscript:x"):
            self.assertEqual(ansi.sanitize_url(bad), "", bad)

    def test_ordinary_schemes_survive(self):
        for good in ("https://example.com/a?b=1#c", "http://example.com",
                     "mailto:someone@example.com", "ftp://files.example.com/x"):
            self.assertEqual(ansi.sanitize_url(good), good)

    def test_relative_urls_are_kept(self):
        for good in ("./docs/index.html", "docs/index.html", "#anchor"):
            self.assertEqual(ansi.sanitize_url(good), good)

    def test_length_is_capped(self):
        self.assertLessEqual(len(ansi.sanitize_url("https://x/" + "a" * 9000)), 2048)

    def test_hyperlink_refuses_a_rejected_url(self):
        out = ansi.hyperlink("javascript:alert(1)", "click", caps())
        self.assertEqual(out, "click")


class TestExitHooks(unittest.TestCase):
    """Anything that puts the terminal in a non-default mode registers here."""

    def setUp(self):
        self._saved = list(ansi._EXIT_HOOKS)

    def tearDown(self):
        ansi._EXIT_HOOKS[:] = self._saved

    def test_hooks_run_on_the_exit_path(self):
        ran = []
        ansi.on_exit(lambda: ran.append("restored"))
        ansi._restore_all_cursors()
        self.assertEqual(ran, ["restored"])

    def test_a_raising_hook_does_not_stop_the_others(self):
        ran = []

        def boom():
            raise RuntimeError("nope")

        ansi.on_exit(boom)
        ansi.on_exit(lambda: ran.append("still ran"))
        ansi._restore_all_cursors()
        self.assertEqual(ran, ["still ran"])

    def test_registering_twice_runs_once(self):
        ran = []
        hook = lambda: ran.append(1)
        ansi.on_exit(hook)
        ansi.on_exit(hook)
        ansi._restore_all_cursors()
        self.assertEqual(len(ran), 1)

    def test_hooks_run_on_a_signal(self):
        import subprocess
        code = (
            "import os, signal, sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parent.parent) +
            "import lume.ansi as a;"
            "a.on_exit(lambda: (sys.stdout.write('CLEANED'), sys.stdout.flush()));"
            "a.install_signal_net();"
            "os.kill(os.getpid(), signal.SIGTERM)"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=30)
        self.assertIn("CLEANED", proc.stdout)

if __name__ == "__main__":
    unittest.main()
