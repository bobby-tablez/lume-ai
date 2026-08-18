"""Tests for the theme registry.

The interesting guarantees are legibility (every token clears the contrast its role
declares) and distinguishability (tokens that appear side by side differ visibly).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lume import theme as T
from lume.ansi import Caps, Style, _srgb_to_lab, strip_ansi

CAPS = Caps(color=24, unicode=True, is_tty=True, width=80, height=24)
MONO_CAPS = Caps(color=0, unicode=False, is_tty=True, width=80, height=24)
#: Themes that carry meaning in colour. `mono` uses SGR attributes only and
#: `plain` uses nothing at all, so neither is subject to the palette rules.
ATTRIBUTE_THEMES = ("mono", "plain")
COLOUR_THEMES = {n: t for n, t in T.THEMES.items() if n not in ATTRIBUTE_THEMES}


def delta_e(a, b):
    la, lb = _srgb_to_lab(a), _srgb_to_lab(b)
    return sum((la[i] - lb[i]) ** 2 for i in range(3)) ** 0.5


class TestRegistry(unittest.TestCase):
    def test_every_theme_defines_every_token(self):
        for name, th in T.THEMES.items():
            self.assertEqual([t for t in T.TOKENS if t not in th.styles], [], name)

    def test_no_theme_defines_unknown_tokens(self):
        for name, th in T.THEMES.items():
            self.assertEqual([t for t in th.styles if t not in T.TOKENS], [], name)

    def test_all_styles_are_style_instances(self):
        for name, th in T.THEMES.items():
            for token, st in th.styles.items():
                self.assertIsInstance(st, Style, f"{name}.{token}")

    def test_plain_emits_nothing_at_all(self):
        """`--plain` must be genuinely escape-free, not merely colour-free."""
        plain = T.THEMES["plain"]
        for token in T.TOKENS:
            self.assertEqual(plain.render("x", token, CAPS), "x", token)
            self.assertEqual(plain.render("x", token, MONO_CAPS), "x", token)

    def test_plain_is_reachable_even_on_a_colour_terminal(self):
        self.assertEqual(T.get_theme("plain", CAPS).name, "plain")
        self.assertEqual(T.get_theme("plain", MONO_CAPS).name, "plain")

    def test_light_and_dark_are_both_offered(self):
        self.assertTrue(any(t.dark for t in T.THEMES.values()))
        self.assertTrue(any(not t.dark for t in T.THEMES.values()))

    def test_theme_names_matches_registry(self):
        self.assertEqual(sorted(T.theme_names()), sorted(T.THEMES))


class TestResolution(unittest.TestCase):
    def test_known_name(self):
        self.assertEqual(T.get_theme("ember").name, "ember")

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(T.get_theme("  EmBeR ").name, "ember")

    def test_unknown_or_missing_name_falls_back(self):
        self.assertEqual(T.get_theme("nope").name, T.DEFAULT_THEME)
        self.assertEqual(T.get_theme(None).name, T.DEFAULT_THEME)

    def test_colourless_terminal_forces_mono(self):
        self.assertEqual(T.get_theme("aurora", caps=MONO_CAPS).name, "mono")


class TestContrast(unittest.TestCase):
    """Every token must clear the minimum its role declares, in every theme."""

    def test_all_tokens_meet_their_declared_minimum(self):
        failures = []
        for name, th in COLOUR_THEMES.items():
            bg = th.background
            code_bg = th["md.code_bg"].bg
            for token in T.TOKENS:
                fg = th[token].fg
                if fg is None:
                    continue
                want = T._MIN_CONTRAST.get(token, T._DEFAULT_MIN)
                # Same surface rule `_build` uses; measuring against the wrong one
                # would let a regression in that rule pass unnoticed.
                surface = code_bg if token.startswith(("syn.", "md.code")) else bg
                got = T.contrast(fg, surface)
                if got < want - 0.01:
                    failures.append(f"{name}.{token}: {got:.2f} < {want}")
        self.assertEqual(failures, [])

    def test_body_text_clears_wcag_aa(self):
        for name, th in COLOUR_THEMES.items():
            self.assertGreaterEqual(T.contrast(th["app.text"].fg, th.background), 7.0, name)

    def test_rules_are_visible_even_though_they_are_quiet(self):
        for name, th in COLOUR_THEMES.items():
            for token in ("app.rule", "md.rule", "md.table_border"):
                self.assertGreater(T.contrast(th[token].fg, th.background), 2.5,
                                   f"{name}.{token}")

    def test_no_token_compounds_sgr_dim_on_a_faint_colour(self):
        """SGR 2 roughly halves contrast; a faint colour must not also be dimmed."""
        for name, th in COLOUR_THEMES.items():
            for token in T.TOKENS:
                st = th[token]
                if st.dim and st.fg is not None:
                    self.assertGreater(T.contrast(st.fg, th.background), 7.0,
                                       f"{name}.{token} is dimmed and already faint")

    def test_code_background_is_close_to_the_page(self):
        for name, th in COLOUR_THEMES.items():
            self.assertLess(T.contrast(th["md.code_bg"].bg, th.background), 1.8, name)

    def test_contrast_helper_endpoints(self):
        self.assertAlmostEqual(T.contrast((0, 0, 0), (255, 255, 255)), 21.0, places=1)
        self.assertAlmostEqual(T.contrast((7, 7, 7), (7, 7, 7)), 1.0, places=6)


class TestDistinguishability(unittest.TestCase):
    def test_syntax_tokens_are_mutually_distinct(self):
        """Adjacent-in-code tokens must not share a colour.

        `syn.plain` / `syn.variable` are excluded from each other (both mean
        "ordinary identifier"), and `syn.punct` is meant to sit near plain text.
        """
        syn = [t for t in T.TOKENS if t.startswith("syn.")]
        exempt = {frozenset(("syn.plain", "syn.variable")),
                  frozenset(("syn.plain", "syn.punct")),
                  frozenset(("syn.variable", "syn.punct"))}
        for name, th in COLOUR_THEMES.items():
            for i, a in enumerate(syn):
                for b in syn[i + 1:]:
                    if frozenset((a, b)) in exempt:
                        continue
                    self.assertGreaterEqual(
                        delta_e(th[a].fg, th[b].fg), 15.0, f"{name}: {a} vs {b}")

    def test_comments_are_distinct_from_punctuation(self):
        for name, th in COLOUR_THEMES.items():
            self.assertGreaterEqual(delta_e(th["syn.comment"].fg, th["syn.punct"].fg),
                                    15.0, name)

    def test_headings_descend_in_prominence(self):
        """h1 must not be quieter than h2, and h3 must not outshine either."""
        for name, th in COLOUR_THEMES.items():
            bg = th.background
            h1, h2, h3 = (T.contrast(th[k].fg, bg) for k in ("md.h1", "md.h2", "md.h3"))
            self.assertGreater(h1, h2, f"{name}: h1 {h1:.1f} is quieter than h2 {h2:.1f}")
            self.assertGreater(h2, h3, f"{name}: h2 {h2:.1f} is quieter than h3 {h3:.1f}")

    def test_no_heading_outshines_body_text(self):
        for name, th in COLOUR_THEMES.items():
            bg = th.background
            self.assertLessEqual(T.contrast(th["md.h3"].fg, bg),
                                 T.contrast(th["app.text"].fg, bg) + 0.01, name)

    def test_the_three_heading_levels_are_all_different(self):
        for name, th in COLOUR_THEMES.items():
            h = [th["md.h1"], th["md.h2"], th["md.h3"]]
            self.assertEqual(len({(s.fg, s.bold, s.underline) for s in h}), 3, name)

    def test_error_warn_and_success_are_distinct(self):
        for name, th in COLOUR_THEMES.items():
            for a, b in (("app.error", "app.success"), ("app.error", "app.warn"),
                         ("app.warn", "app.success")):
                self.assertGreaterEqual(delta_e(th[a].fg, th[b].fg), 20.0, f"{name} {a}/{b}")

    def test_distinctions_survive_the_256_colour_downgrade(self):
        from lume.ansi import rgb_to_256
        for name, th in COLOUR_THEMES.items():
            for a, b in (("app.error", "app.success"), ("md.h1", "app.text"),
                         ("syn.keyword", "syn.string"), ("syn.comment", "app.text")):
                self.assertNotEqual(rgb_to_256(th[a].fg), rgb_to_256(th[b].fg),
                                    f"{name} {a}/{b} collide at 256 colours")


class TestMono(unittest.TestCase):
    """`mono` is the theme for terminals with no colour, so meaning lives in attributes."""

    def test_emits_no_colour_codes(self):
        th = T.THEMES["mono"]
        for token in T.TOKENS:
            out = th.render("x", token, CAPS)
            self.assertNotIn("38;2", out)
            self.assertNotIn("38;5", out)

    def test_attribute_styling_survives_a_colourless_terminal(self):
        th = T.THEMES["mono"]
        for token, code in (("md.italic", "3"), ("app.dim", "2"), ("md.strike", "9"),
                            ("md.h1", "1"), ("md.link", "4")):
            self.assertIn(code, th.render("x", token, MONO_CAPS), token)

    def test_key_roles_are_visually_separable(self):
        th = T.THEMES["mono"]
        seen = {}
        for token in ("md.h1", "md.h2", "md.h3", "app.text", "md.italic", "md.strike",
                      "app.dim", "md.link"):
            seen.setdefault(th.render("x", token, MONO_CAPS), []).append(token)
        for rendering, tokens in seen.items():
            self.assertEqual(len(tokens), 1, f"indistinguishable in mono: {tokens}")


class TestRendering(unittest.TestCase):
    def test_render_round_trips_text(self):
        for name, th in T.THEMES.items():
            for token in T.TOKENS:
                self.assertEqual(strip_ansi(th.render("sample", token, CAPS)), "sample",
                                 f"{name}.{token}")

    def test_unknown_token_warns_once_and_degrades(self):
        th = T.get_theme("aurora")
        T._WARNED.clear()
        with self.assertWarns(UserWarning):
            self.assertEqual(strip_ansi(th.render("x", "no.such.token", CAPS)), "x")

    def test_accent_stops_are_usable_in_every_colour_theme(self):
        for name, th in COLOUR_THEMES.items():
            stops = th.accent_stops()
            self.assertGreaterEqual(len(stops), 2, name)
            for s in stops:
                self.assertIsNotNone(s, name)
                self.assertEqual(len(s), 3, name)
                self.assertTrue(all(0 <= v <= 255 for v in s), name)

    def test_a_theme_with_no_colours_paints_no_gradient(self):
        """A grey fallback ramp made `--theme plain` emit escapes anyway."""
        for name in ATTRIBUTE_THEMES:
            self.assertEqual(T.THEMES[name].accent_stops(), [], name)

    def test_background_is_defined_for_every_theme(self):
        for name, th in T.THEMES.items():
            self.assertEqual(len(th.background), 3, name)



class TestAutomaticTheme(unittest.TestCase):
    """A dark palette on a white terminal is unreadable; pick from the terminal."""

    def test_a_light_terminal_gets_a_light_theme(self):
        self.assertFalse(T.get_theme("auto", CAPS, "light").dark)

    def test_a_dark_terminal_gets_a_dark_theme(self):
        self.assertTrue(T.get_theme("auto", CAPS, "dark").dark)

    def test_an_unknown_background_keeps_the_default(self):
        self.assertEqual(T.get_theme("auto", CAPS, "").name, T.DEFAULT_THEME)

    def test_an_explicit_choice_beats_the_terminal(self):
        self.assertEqual(T.get_theme("ember", CAPS, "light").name, "ember")

    def test_colourless_terminals_still_get_mono(self):
        self.assertEqual(T.get_theme("auto", MONO_CAPS, "light").name, "mono")

    def test_the_light_theme_is_legible_on_the_background_it_declares(self):
        solar = T.THEMES["solar"]
        self.assertGreater(T.contrast(solar["app.text"].fg, solar.background), 7.0)

if __name__ == "__main__":
    unittest.main()
