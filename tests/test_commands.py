"""Tests for lume.commands — the registry, the parser, completion, and /help."""

import os
import re
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lume.ansi import Caps, display_width, strip_ansi
import lume.commands as commands_mod
from lume.commands import (COMMANDS, GROUP_BLURBS, GROUPS, INPUT_RULES,
                           ORIENTATION, Command, arg_values, find, groups,
                           help_text, names, parse, suggest)
from lume.theme import THEMES, get_theme, theme_names

#: The vocabulary SPEC.md asked for. `/stream` was dropped deliberately — replies
#: always stream, so a toggle described a feature the app does not have — and
#: `/tokens`/`/cost` became aliases of `/usage` rather than separate commands.
REQUIRED = (
    "help new resume list delete rename model models system theme think effort "
    "clear retry edit copy export tokens cost keys quit"
).split()

#: Added in round two; each needs a handler in app.py (see the report).
ADDED = ("usage", "undo")

VISIBLE = tuple(c for c in COMMANDS if not c.hidden)

DARK = Caps(color=24, unicode=True, is_tty=True, width=80, height=24)
ASCII = Caps(color=0, unicode=False, is_tty=False, width=80, height=24)
#: A terminal with room for the whole table. The default 80x24 one has not got
#: it — see HelpFitsTheScreenTests — so every test that is about the table
#: itself rather than about fitting has to say so.
TALL = Caps(color=24, unicode=True, is_tty=True, width=80, height=200)
TALL_ASCII = Caps(color=0, unicode=False, is_tty=False, width=80, height=200)


def plain(text):
    return strip_ansi(text)


class RegistryTests(unittest.TestCase):
    def test_every_required_command_exists(self):
        for name in REQUIRED:
            with self.subTest(name=name):
                self.assertIsNotNone(find(name))

    def test_new_commands_exist(self):
        for name in ADDED:
            self.assertIsNotNone(find(name), name)
        self.assertIs(find("cls"), find("clear"))

    def test_tokens_and_cost_are_genuine_aliases_of_usage(self):
        # They used to be separate commands whose docstring called them aliases.
        # One dataset, one handler, one help entry.
        for name in ("tokens", "cost"):
            self.assertIs(find(name), find("usage"), name)
            # parse canonicalises, so callers only ever see one name.
            self.assertEqual(parse("/" + name), ("usage", ""))
        self.assertEqual(parse("/cost extra"), ("usage", "extra"))

    def test_names_and_aliases_are_unique(self):
        seen = set()
        for cmd in COMMANDS:
            for token in (cmd.name,) + tuple(cmd.aliases):
                self.assertNotIn(token, seen, f"duplicate token {token!r}")
                seen.add(token)

    def test_declarations_are_well_formed(self):
        for cmd in COMMANDS:
            with self.subTest(cmd=cmd.name):
                self.assertIn(cmd.group, GROUPS)
                self.assertTrue(cmd.name.islower())
                self.assertTrue(cmd.help.endswith("."), cmd.help)
                self.assertTrue(cmd.help[0].isupper(), cmd.help)
                self.assertLess(len(cmd.help), 80)
                self.assertIsInstance(cmd.aliases, tuple)
                # Help text has to survive an ASCII-only terminal.
                self.assertTrue(cmd.help.isascii())
                self.assertTrue(cmd.args.isascii())

    def test_signature_and_matches(self):
        cmd = find("list")
        self.assertEqual(cmd.signature, "/list [query]")
        self.assertEqual(find("models").signature, "/models")
        self.assertTrue(cmd.matches("/ls"))
        self.assertTrue(cmd.matches("LIST"))
        self.assertFalse(cmd.matches("listen"))

    def test_command_defaults(self):
        c = Command("x", "", "Does x.", "Session")
        self.assertEqual(c.aliases, ())
        self.assertFalse(c.hidden)
        self.assertEqual(c.signature, "/x")

    def test_command_is_frozen(self):
        with self.assertRaises(Exception):
            COMMANDS[0].name = "nope"

    def test_find_variants(self):
        self.assertIs(find("/quit"), find("q"))
        self.assertIs(find("EXIT"), find("quit"))
        self.assertIsNone(find("nonesuch"))
        self.assertIsNone(find(""))
        self.assertIsNone(find(None))

    def test_groups_are_all_used(self):
        self.assertEqual(set(groups()), {c.group for c in COMMANDS})
        self.assertEqual(groups(), tuple(g for g in GROUPS if g in groups()))

    def test_names_in_declaration_order(self):
        self.assertEqual(names(), tuple(c.name for c in COMMANDS))


class ParseTests(unittest.TestCase):
    def test_command_with_argument(self):
        self.assertEqual(parse("/model sonnet"), ("model", "sonnet"))

    def test_command_without_argument(self):
        self.assertEqual(parse("/models"), ("models", ""))

    def test_arguments_are_stripped(self):
        self.assertEqual(parse("/model   sonnet  "), ("model", "sonnet"))

    def test_alias_resolves_to_canonical_name(self):
        self.assertEqual(parse("/q"), ("quit", ""))
        self.assertEqual(parse("/ls foo"), ("list", "foo"))
        self.assertEqual(parse("/?"), ("help", ""))

    def test_case_insensitive(self):
        self.assertEqual(parse("/HELP resume"), ("help", "resume"))

    def test_leading_whitespace_tolerated(self):
        self.assertEqual(parse("   /new title"), ("new", "title"))

    def test_prose_comes_back_untouched(self):
        for line in ("hello there", "  indented prose", "2/3 of the way", ""):
            self.assertEqual(parse(line), (None, line))

    def test_unknown_command_still_parses(self):
        self.assertEqual(parse("/xyz a b"), ("xyz", "a b"))

    def test_lone_slash(self):
        self.assertEqual(parse("/"), ("", ""))
        self.assertEqual(parse("/   "), ("", ""))

    def test_double_slash_is_literal_text(self):
        self.assertEqual(parse("//help"), (None, "/help"))
        self.assertEqual(parse("//"), (None, "/"))

    def test_path_like_text_is_not_a_command(self):
        self.assertEqual(parse("/usr/bin/env python"), (None, "/usr/bin/env python"))
        self.assertEqual(parse("/2+2"), (None, "/2+2"))
        self.assertEqual(parse("/-x"), (None, "/-x"))

    def test_a_multiline_paste_is_never_a_command(self):
        # A slash command is one line by construction. Without this, pasting a
        # document whose first line starts with a slash silently ran it as a
        # command and swallowed the rest of the paste as its argument.
        paste = "/list of things\nsecond line\nthird\n"
        self.assertEqual(parse(paste), (None, paste))
        self.assertEqual(parse("/system be terse\nand kind"),
                         (None, "/system be terse\nand kind"))
        self.assertEqual(parse("/quit\n"), (None, "/quit\n"))

    def test_single_line_commands_still_parse(self):
        self.assertEqual(parse("/system be terse"), ("system", "be terse"))

    def test_none_is_safe(self):
        self.assertEqual(parse(None), (None, ""))


class SuggestTests(unittest.TestCase):
    def test_command_prefix(self):
        self.assertEqual(suggest("/mo"), ["/model", "/models"])

    def test_slash_is_preserved_or_absent(self):
        self.assertEqual(suggest("mo"), ["model", "models"])
        self.assertIn("/help", suggest(""))
        self.assertEqual(len(suggest("/")), len(COMMANDS))

    def test_alias_only_when_no_name_matches(self):
        self.assertEqual(suggest("/rm"), ["/rm"])
        self.assertEqual(suggest("/l"), ["/list"])       # not the /ls alias

    def test_no_match(self):
        self.assertEqual(suggest("/zzz"), [])

    def test_argument_completion(self):
        self.assertEqual(suggest("/theme "), list(theme_names()))
        self.assertEqual(suggest("/theme au"), ["aurora"])
        self.assertEqual(suggest("/think o"), ["on", "off"])
        self.assertEqual(suggest("/effort x"), ["xhigh"])

    def test_argument_completion_for_free_text_command(self):
        self.assertEqual(suggest("/new my "), [])

    def test_argument_completion_needs_a_command(self):
        self.assertEqual(suggest("hello wor"), [])

    def test_none_is_safe(self):
        self.assertEqual(suggest(None), [])

    def test_leading_whitespace_still_suggests(self):
        # The prompt tolerates "  /he"; completion has to as well.
        self.assertEqual(suggest("  /he"), ["/help"])
        self.assertEqual(suggest("\t/mo"), ["/model", "/models"])
        self.assertEqual(suggest("  /theme au"), ["aurora"])

    def test_tab_never_rewrites_a_line_that_enter_would_run(self):
        # '/r' runs /resume, so Tab must not expand it to '/re', which is not a
        # command at all. Every prefix that find() resolves stays in its own
        # completion list, and the common prefix of the list is unchanged.
        for token in ("/r", "/q", "/h", "/m", "/y", "/ls", "/rm"):
            with self.subTest(token=token):
                hits = suggest(token)
                self.assertIn(token, hits)
                common = os.path.commonprefix(hits)
                self.assertEqual(common, token)
                self.assertIsNotNone(find(common))
                # ...and what Tab leaves behind is what Enter would have run.
                self.assertEqual(parse(common)[0], find(token).name)

    def test_arg_values(self):
        self.assertEqual(arg_values("/theme"), tuple(theme_names()))
        self.assertIn("resume", arg_values("help"))
        self.assertIn("session", arg_values("help"))
        self.assertEqual(arg_values("new"), ())
        self.assertEqual(arg_values("nonesuch"), ())


class HelpTextTests(unittest.TestCase):
    def setUp(self):
        self.theme = get_theme("aurora")

    def widths(self):
        return (24, 30, 34, 40, 48, 55, 60, 72, 80, 100, 120, 200)

    def test_fits_every_width(self):
        for uni in (True, False):
            for w in self.widths():
                caps = Caps(color=24, unicode=uni, is_tty=True, width=w, height=24)
                for line in help_text(self.theme, caps, w).split("\n"):
                    self.assertLessEqual(display_width(line), w,
                                         f"width={w} unicode={uni}: {line!r}")

    def test_fits_every_theme(self):
        caps = Caps(color=8, unicode=True, is_tty=True, width=76, height=24)
        for theme in THEMES.values():
            for line in help_text(theme, caps, 76).split("\n"):
                self.assertLessEqual(display_width(line), 76, theme.name)

    def test_lists_every_command(self):
        out = plain(help_text(self.theme, TALL, 100))
        for cmd in VISIBLE:
            self.assertIn("/" + cmd.name, out)
            self.assertIn(cmd.help, " ".join(out.split()))

    def test_usage_lists_its_aliases_on_one_line(self):
        out = plain(help_text(self.theme, TALL, 100))
        self.assertIn("/usage, /tokens, /cost", out)
        detail = plain(help_text(self.theme, ASCII, 80, "tokens"))
        self.assertIn("/usage", detail)

    def test_no_command_describes_a_feature_that_does_not_exist(self):
        """/stream claimed to toggle streaming; replies always stream."""
        self.assertIsNone(find("stream"))
        self.assertNotIn("/stream", plain(help_text(self.theme, TALL, 100)))

    def test_group_headings_carry_a_blurb(self):
        out = plain(help_text(self.theme, TALL, 100))
        for label, blurb in GROUP_BLURBS.items():
            self.assertIn(blurb, out, label)

    def test_orientation_strip_is_under_the_title(self):
        lines = plain(help_text(self.theme, DARK, 100)).split("\n")
        self.assertIn("lume", lines[0])
        for name, what in ORIENTATION:
            self.assertIn(name + " " + what, lines[1])
        self.assertIsNotNone(find(ORIENTATION[0][0]))

    def test_narrow_screens_drop_the_orientation_strip_rather_than_wrap_it(self):
        out = plain(help_text(self.theme, ASCII, 40))
        self.assertNotIn("/new start", out)

    def test_shows_every_group_and_the_input_rules(self):
        out = plain(help_text(self.theme, TALL, 100))
        for g in groups():
            self.assertIn(g, out)
        self.assertIn("Input", out)
        for key, _ in INPUT_RULES:
            self.assertIn(key, out)

    def test_columns_line_up_across_groups(self):
        out = plain(help_text(self.theme, TALL, 100))
        starts = set()
        for line in out.split("\n"):
            m = re.match(r"( {4}\S.*?)(\s{2,})(\S.*)$", line)
            if m:
                starts.add(len(m.group(1)) + len(m.group(2)))
        self.assertEqual(len(starts), 1, starts)

    def test_ascii_terminal_gets_ascii_only(self):
        out = help_text(self.theme, ASCII, 80)
        self.assertTrue(out.isascii(), [c for c in out if not c.isascii()][:5])

    def test_no_escapes_without_colour(self):
        out = help_text(self.theme, ASCII, 80)
        self.assertEqual(out, strip_ansi(out))
        self.assertNotIn("\x1b", out)

    def test_colour_terminal_is_themed(self):
        out = help_text(self.theme, DARK, 80)
        self.assertIn("\x1b[", out)

    def test_width_defaults_to_caps(self):
        caps = Caps(color=0, unicode=False, is_tty=False, width=52, height=24)
        for line in help_text(self.theme, caps).split("\n"):
            self.assertLessEqual(display_width(line), 52)

    def test_group_filter(self):
        out = plain(help_text(self.theme, ASCII, 80, "session"))
        self.assertIn("/resume", out)
        self.assertNotIn("/effort", out)
        self.assertNotIn("Ctrl-D", out)

    def test_group_filter_is_case_insensitive(self):
        self.assertEqual(help_text(self.theme, ASCII, 80, "SESSION"),
                         help_text(self.theme, ASCII, 80, "session"))

    def test_input_topic_shows_only_the_typing_rules(self):
        out = plain(help_text(self.theme, ASCII, 80, "input"))
        self.assertIn("Ctrl-D", out)
        # The newline chord is only listed where it can actually be bound; the
        # rules that work on every build are the ones always shown.
        self.assertIn('"""', out)
        if commands_mod.ALT_ENTER:
            self.assertIn("Alt+Enter", out)
        else:
            self.assertNotIn("Alt+Enter", out)
        self.assertNotIn("/resume", out)
        self.assertNotIn("No help topic", out)

    def test_command_detail(self):
        out = plain(help_text(self.theme, ASCII, 80, "resume"))
        self.assertIn("/resume, /r [ref]", out)
        self.assertIn("session", out)
        self.assertIn("last", out)
        self.assertNotIn("/effort", out)
        self.assertEqual(out, plain(help_text(self.theme, ASCII, 80, "/r")))

    def test_unknown_topic_is_reported_then_help_follows(self):
        out = plain(help_text(self.theme, ASCII, 80, "wat"))
        self.assertIn("No help topic", out)
        self.assertIn("/quit", out)

    def test_no_trailing_newline(self):
        out = help_text(self.theme, DARK, 80)
        self.assertFalse(out.endswith("\n"))
        self.assertFalse(out.startswith("\n"))

    def test_widths_below_the_old_floor_are_honoured(self):
        # help_text used to clamp to 24 columns and hand back 24-column lines to
        # a narrower terminal, which then hard-wrapped every one of them.
        for w in range(6, 30):
            for uni in (True, False):
                caps = Caps(color=0, unicode=uni, is_tty=False, width=w, height=24)
                for topic in (None, "session", "input", "resume"):
                    for line in help_text(self.theme, caps, w, topic).split("\n"):
                        self.assertLessEqual(display_width(line), w,
                                             f"width={w} topic={topic}: {line!r}")

    def test_every_width_from_four_to_two_hundred_fits(self):
        caps = Caps(color=24, unicode=True, is_tty=True, width=80, height=24)
        for w in range(4, 201):
            for line in help_text(self.theme, caps, w).split("\n"):
                self.assertLessEqual(display_width(line), w, f"width={w}: {line!r}")

    def test_narrow_terminal_stacks_rather_than_truncating(self):
        caps = Caps(color=0, unicode=False, is_tty=False, width=30, height=200)
        out = plain(help_text(self.theme, caps, 30))
        self.assertIn("/export [format] [path]", out)


class HelpFitsTheScreenTests(unittest.TestCase):
    """/help has to fit the terminal it is printed on. There is no pager here.

    On a default 80x24 terminal the full table is 61 lines: 38 of them scroll
    away, and the ones that go first are the title, the orientation strip and
    the two groups a new user needs most. Every one of these failed before the
    screen was allowed to know how tall it is.
    """

    def setUp(self):
        self.theme = get_theme("aurora")

    def heights(self):
        return (8, 12, 24, 30, 40)

    def test_the_main_screen_fits_an_eighty_by_twentyfour_terminal(self):
        out = help_text(self.theme, DARK, 80)
        self.assertLessEqual(len(out.split("\n")), DARK.height - 1, out)

    def test_it_fits_every_ordinary_terminal_size(self):
        for uni in (True, False):
            for h in self.heights():
                for w in (40, 60, 80, 100, 200):
                    caps = Caps(color=24, unicode=uni, is_tty=True, width=w, height=h)
                    lines = help_text(self.theme, caps, w).split("\n")
                    if h >= 24:
                        self.assertLessEqual(len(lines), h - 1, (w, h, len(lines)))

    def test_the_short_form_still_names_every_command(self):
        out = plain(help_text(self.theme, DARK, 80))
        for cmd in VISIBLE:
            self.assertIn("/" + cmd.name, out, cmd.name)
        for cmd in COMMANDS:
            if cmd.hidden:
                self.assertNotIn("/" + cmd.name, out)

    def test_the_short_form_keeps_the_orientation_strip_and_the_title(self):
        lines = plain(help_text(self.theme, DARK, 80)).split("\n")
        self.assertIn("lume", lines[0])
        for name, what in ORIENTATION:
            self.assertIn(name + " " + what, lines[1])

    def test_the_short_form_says_where_the_detail_is(self):
        out = plain(help_text(self.theme, DARK, 80))
        self.assertIn("/help <topic>", out)
        self.assertIn("/keys", out)

    def test_the_short_form_keeps_every_group_heading(self):
        out = plain(help_text(self.theme, DARK, 80))
        for g in groups():
            self.assertIn(g, out)

    def test_a_tall_terminal_still_gets_the_whole_table(self):
        out = plain(help_text(self.theme, TALL, 80))
        for cmd in VISIBLE:
            self.assertIn(cmd.help, " ".join(out.split()), cmd.name)
        for key, _ in INPUT_RULES:
            self.assertIn(key, out)

    def test_height_zero_means_no_limit(self):
        out = plain(help_text(self.theme, DARK, 80, height=0))
        self.assertIn(COMMANDS[0].help, " ".join(out.split()))

    def test_an_explicit_height_beats_the_terminal(self):
        short = help_text(self.theme, TALL, 80, height=24)
        self.assertEqual(short, help_text(self.theme, DARK, 80))

    def test_the_group_and_topic_views_are_never_shortened(self):
        # They already fit, and shortening the answer to a direct question
        # would be the one place a summary is no use at all.
        for topic in ("session", "conversation", "model", "interface", "input",
                      "resume", "export"):
            out = plain(help_text(self.theme, DARK, 80, topic))
            self.assertNotIn("/help <topic> for what one does", out, topic)
            self.assertLessEqual(len(out.split("\n")), 24, topic)

    def test_an_unknown_topic_is_still_reported_on_a_short_screen(self):
        out = plain(help_text(self.theme, DARK, 80, "wat"))
        self.assertIn("No help topic", out)
        self.assertIn("/quit", out)

    def test_the_short_form_obeys_the_width_contract_too(self):
        for uni in (True, False):
            for w in range(8, 121, 3):
                caps = Caps(color=24, unicode=uni, is_tty=True, width=w, height=24)
                for line in help_text(self.theme, caps, w).split("\n"):
                    self.assertLessEqual(display_width(strip_ansi(line)), w,
                                         (w, uni, line))


class TabCompletionOfHelpTopicsTests(unittest.TestCase):
    """Tab has to offer what /help actually accepts, and each thing once."""

    def test_every_offered_topic_is_a_topic_help_understands(self):
        theme = get_theme("aurora")
        for topic in arg_values("help"):
            out = plain(help_text(theme, TALL_ASCII, 80, topic))
            self.assertNotIn("No help topic", out, topic)

    def test_the_typing_rules_are_offered(self):
        self.assertIn("input", arg_values("help"))
        self.assertIn("rules", arg_values("help"))

    def test_no_topic_is_offered_twice(self):
        topics = arg_values("help")
        self.assertEqual(len(topics), len(set(topics)), topics)
        self.assertEqual(topics.count("model"), 1)

    def test_tab_after_help_offers_them(self):
        self.assertIn("input", suggest("/help in"))
        self.assertEqual(suggest("/help mod"), ["model", "models"])


if __name__ == "__main__":
    unittest.main()
