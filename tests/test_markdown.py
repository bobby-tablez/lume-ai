"""Tests for lume.markdown — streaming equivalence, block/inline rendering, highlighting."""

import random
import re
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lume.ansi import RESET, Caps, display_width, strip_ansi
from lume.markdown import (
    MarkdownStream,
    highlight,
    render_markdown,
    supported_languages,
)
from lume.theme import THEMES, get_theme

SGR = re.compile(r"\x1b\[[0-9;]*m")

RICH = Caps(color=24, unicode=True, is_tty=True, width=72, hyperlinks=False, animation=True)
LINKS = Caps(color=8, unicode=True, is_tty=True, width=48, hyperlinks=True)
ASCII = Caps(color=0, unicode=False, is_tty=False, width=60)
MONO_TTY = Caps(color=0, unicode=True, is_tty=True, width=60)
NARROW = Caps(color=4, unicode=False, is_tty=True, width=24)


def theme_for(caps):
    return get_theme("aurora", caps)


def render(text, caps=ASCII, **kw):
    return render_markdown(text, theme_for(caps), caps, **kw)


def plain(text, caps=ASCII, **kw):
    return strip_ansi(render(text, caps, **kw))


def lines(text, caps=ASCII, **kw):
    return plain(text, caps, **kw).rstrip("\n").split("\n")


#: Documents that between them exercise every construct and every awkward edge.
DOCS = [
    "",
    "\n\n\n",
    "just one line",
    "no trailing newline",
    "a\n\nb\n\nc",
    "# h1\n## h2\n### h3\n#### h4\n##### h5\n###### h6\n# \n#nope",
    "**bold** *it* _it_ ~~strike~~ `code` ***both*** __bold__",
    "unclosed **bold and *italic and `code",
    "a\\*not em\\* b \\\\ backslash \\` tick",
    "`code with **stars** and #hash` after",
    "[label](http://example.com) <http://auto.link> https://bare.url/x, ![i](p.png)",
    "[broken](unclosed and [nolink] text",
    "- a\n- b\n\n- c after blank\n",
    "* star\n+ plus\n- dash\n",
    "1. one\n2. two\n10. ten\n",
    "- top\n  - mid\n    - deep\n      - deeper\n- back",
    "- [ ] todo\n- [x] done\n- [X] DONE\n",
    "- item\n  continued lazily\nlazy too\n\n  second para of item\n",
    "- item with code\n\n  ```py\n  x = 1\n  ```\n\n- next",
    "> quote\n> **bold** quote\n>\n> second para\n> > nested\nlazy quote line\n\nafter",
    "> - list in quote\n> - two\n",
    "```\nplain code\n```\n",
    "```python\ndef f(x):\n    '''doc\n    string'''\n    return x  # it's fine\n```",
    '```js\nconst s = "has # not comment"; // real\n```',
    "~~~sql\nSELECT * FROM t WHERE a = 'x'; -- note\n~~~",
    "```unknownlang\nsome text\n```",
    "```\nunclosed fence\nmore",
    "    indented code\n    line two\n\n    after blank\n\nparagraph",
    "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "| left | center | right |\n|:--|:-:|--:|\n| a | b | c |\n| much longer cell | x | y |\n",
    "| a | b |\n|---|---|\n",
    "| not | a table\nnext line",
    "| a | b |\n|---|---|\n| 1 | 2 |\nparagraph right after",
    "---\n***\n___\n- - -\n",
    "text\n---\nmore",
    "para one\n# heading interrupts\npara two\n- list interrupts\n",
    "hard break here  \nnext line\nand backslash\\\nafter",
    "a very long paragraph " + "word " * 40,
    "wide 中文字符 and emoji mixed with text " * 3,
    "# Title\n\nIntro.\n\n```python\nclass A:\n    @property\n    def x(self): return 1\n```\n\n"
    "| k | v |\n|---|---|\n| a | 1 |\n\n> note\n\n- one\n- two\n",
    "trailing spaces   \n\nand tabs\there",
    "nested > in text and | pipe in paragraph",
    "1) paren ordered\n2) second\n",
    "12. start at twelve\n13. next\n",
]


def stream_render(text, chunks, caps, **kw):
    st = MarkdownStream(theme_for(caps), caps, **kw)
    out = [st.feed(c) for c in chunks]
    out.append(st.close())
    return "".join(out)


def split_at(text, points):
    """Split `text` at explicit indices."""
    pts = [0] + sorted(p for p in points if 0 < p < len(text)) + [len(text)]
    return [text[a:b] for a, b in zip(pts, pts[1:]) if b > a]


class TestStreamingEquivalence(unittest.TestCase):
    """The headline invariant: chunking must not change a single byte."""

    def assert_same(self, doc, chunks, caps, msg="", **kw):
        want = render_markdown(doc, theme_for(caps), caps, **kw)
        got = stream_render(doc, chunks, caps, **kw)
        if got != want:
            a, b = want.split("\n"), got.split("\n")
            for i in range(max(len(a), len(b))):
                x = a[i] if i < len(a) else "<eof>"
                y = b[i] if i < len(b) else "<eof>"
                if x != y:
                    self.fail(f"{msg} line {i}\n  want {x!r}\n  got  {y!r}")
            self.fail(f"{msg}: outputs differ in length")

    def test_one_character_at_a_time(self):
        for caps in (RICH, ASCII, LINKS):
            for doc in DOCS:
                self.assert_same(doc, list(doc), caps, msg=f"1char {doc[:24]!r}")

    def test_single_chunk_and_empty_chunks(self):
        for doc in DOCS:
            self.assert_same(doc, [doc], RICH, msg="whole")
            self.assert_same(doc, ["", doc, ""], RICH, msg="padded")

    def test_random_chunkings(self):
        rnd = random.Random(20260817)
        for caps, width, indent in ((RICH, None, ""), (ASCII, 30, "  "), (NARROW, None, "")):
            for doc in DOCS:
                for _ in range(6):
                    i, chunks = 0, []
                    while i < len(doc):
                        k = rnd.randint(1, 11)
                        chunks.append(doc[i:i + k])
                        i += k
                    self.assert_same(doc, chunks, caps, msg=f"rand {doc[:24]!r}",
                                     width=width, indent=indent)

    def test_split_inside_emphasis(self):
        doc = "a **bold** b *it* c ~~s~~ d ***both*** e\n"
        for i in range(1, len(doc)):
            self.assert_same(doc, [doc[:i], doc[i:]], RICH, msg=f"emph@{i}")

    def test_split_inside_fence_marker(self):
        doc = "text\n\n```python\nx = 1\n```\n\nafter\n"
        for i in range(1, len(doc)):
            self.assert_same(doc, [doc[:i], doc[i:]], RICH, msg=f"fence@{i}")

    def test_split_inside_table_rows(self):
        doc = "| a | b |\n|---|--:|\n| 1 | 2 |\n| 3 | 4 |\n\nafter\n"
        for i in range(1, len(doc)):
            self.assert_same(doc, [doc[:i], doc[i:]], RICH, msg=f"table@{i}")

    def test_split_mid_word_and_mid_link(self):
        doc = "the quick brownfox jumps [over](http://x/y) the lazy dog\n"
        for i in range(1, len(doc)):
            self.assert_same(doc, [doc[:i], doc[i:]], RICH, msg=f"word@{i}")

    def test_split_inside_nested_containers(self):
        doc = "> - **deep** item\n>   more text\n>\n> - second\n\n- [ ] task **x**\n"
        for i in range(1, len(doc)):
            self.assert_same(doc, [doc[:i], doc[i:]], RICH, msg=f"nest@{i}")

    def test_split_at_every_pair_of_points(self):
        doc = "# H\n\npara **b** `c`\n\n- l1\n- l2\n\n```py\nx=1\n```\n"
        rnd = random.Random(5)
        for _ in range(120):
            pts = [rnd.randrange(len(doc)) for _ in range(3)]
            self.assert_same(doc, split_at(doc, pts), RICH, msg=f"pts {pts}")

    def test_crlf_input(self):
        doc = "# H\r\n\r\npara **b**\r\n\r\n- item\r\n"
        self.assert_same(doc, list(doc), RICH, msg="crlf")

    def test_link_landing_on_a_wrap_point(self):
        # The break before a link is where a partially-rendered line and the
        # finished one are most likely to disagree: the escapes that open the
        # link fall *on* the break.
        doc = "Body with `code`, [a link](http://x.y/z) and https://bare.io/p.\n"
        for caps in (RICH, LINKS):
            for width in (17, 24, 30, 48):
                self.assert_same(doc, list(doc), caps, msg=f"link w{width}",
                                 width=width, indent=" ")

    def test_tab_indented_line_inside_a_list(self):
        # A tab does not count toward a list's content indent, so this is an
        # indented code block inside the item -- in both paths.
        for doc in ("-\n\t *\n", "- a\n\t b\n", "- a\n\t- b\n", "1.\n\t x\n"):
            for caps in (RICH, ASCII):
                self.assert_same(doc, list(doc), caps, msg=f"tab {doc!r}",
                                 width=30, indent="  ")

    def test_randomized_documents(self):
        """Generated documents, randomly chunked: the invariant at scale."""
        rnd = random.Random(4242)
        inline = ["**{w}**", "*{w}*", "`{w}`", "[{w}](http://e.co/{w})", "~~{w}~~",
                  "https://bare.io/{w}", "![{w}](p.png)", "{w}", "{w},", "\\*{w}\\*"]
        words = "alpha beta gamma 中文 snake_case x".split()

        def phrase(k=5):
            return " ".join(rnd.choice(inline).format(w=rnd.choice(words))
                            for _ in range(rnd.randint(1, k)))

        def block():
            r = rnd.random()
            if r < 0.2:
                return phrase(9) + "\n"
            if r < 0.35:
                return "#" * rnd.randint(1, 6) + " " + phrase(3) + "\n"
            if r < 0.55:
                return "".join(rnd.choice(["", "  ", "\t", "    "])
                               + rnd.choice(["-", "*", "1.", "- [ ]", "- [x]"])
                               + " " + phrase(3) + "\n" for _ in range(rnd.randint(1, 3)))
            if r < 0.7:
                return "".join(rnd.choice(["> ", "> > ", ">"]) + phrase(3) + "\n"
                               for _ in range(rnd.randint(1, 3)))
            if r < 0.85:
                lang = rnd.choice(["python", "bash", "sql", "", "nope"])
                return f"```{lang}\nx = \"a # b\"\n" + rnd.choice(["```\n", ""])
            if r < 0.95:
                return ("| a | b |\n|:--|--:|\n"
                        + "".join(f"| {phrase(2)} | x |\n" for _ in range(rnd.randint(0, 2))))
            return rnd.choice(["---\n", "\n", "    indented\n"])

        for _ in range(60):
            doc = "".join(block() + ("\n" if rnd.random() < 0.6 else "")
                          for _ in range(rnd.randint(1, 5)))
            if rnd.random() < 0.2:
                doc = doc.replace("\n", "\r\n")
            caps = rnd.choice((RICH, LINKS, ASCII, MONO_TTY, NARROW))
            kw = {"width": rnd.choice((None, 17, 30, 48)), "indent": rnd.choice(("", " ", "  "))}
            i, chunks = 0, []
            while i < len(doc):
                k = rnd.randint(1, 9)
                chunks.append(doc[i:i + k])
                i += k
            self.assert_same(doc, chunks, caps, msg=f"gen {doc[:24]!r}", **kw)
            self.assert_same(doc, list(doc), caps, msg=f"gen1 {doc[:24]!r}", **kw)

    def test_every_theme(self):
        doc = DOCS[-6]
        for name in THEMES:
            caps = RICH
            th = THEMES[name]
            want = render_markdown(doc, th, caps)
            st = MarkdownStream(th, caps)
            got = "".join(st.feed(c) for c in doc) + st.close()
            self.assertEqual(got, want, name)


class TestAppendOnly(unittest.TestCase):
    def test_every_intermediate_output_is_a_prefix(self):
        doc = DOCS[-6] + "\n" + DOCS[19] + "\n" + DOCS[30]
        final = render(doc, RICH)
        st = MarkdownStream(theme_for(RICH), RICH)
        acc = ""
        for ch in doc:
            acc += st.feed(ch)
            self.assertTrue(final.startswith(acc), f"retracted at {acc[-40:]!r}")
        acc += st.close()
        self.assertEqual(acc, final)

    def test_close_is_idempotent(self):
        st = MarkdownStream(theme_for(ASCII), ASCII)
        st.feed("hello **world**")
        first = st.close()
        self.assertTrue(first.strip())
        self.assertEqual(st.close(), "")
        self.assertEqual(st.close(), "")

    def test_feed_after_close_is_inert(self):
        st = MarkdownStream(theme_for(ASCII), ASCII)
        st.feed("text")
        st.close()
        self.assertEqual(st.feed("more"), "")

    def test_output_is_whole_lines(self):
        st = MarkdownStream(theme_for(RICH), RICH)
        for ch in DOCS[-6]:
            piece = st.feed(ch)
            self.assertTrue(piece == "" or piece.endswith("\n"), repr(piece))
        self.assertTrue(st.close().endswith("\n"))


class TestHeadings(unittest.TestCase):
    def test_atx_levels_and_rules(self):
        out = lines("# One\n\n## Two\n\n### Three\n")
        self.assertEqual(out[0], "One")
        self.assertEqual(out[1], "=" * 3)
        self.assertIn("Two", out)
        self.assertIn("-" * 3, out)
        self.assertIn("Three", out)

    def test_hash_without_space_is_paragraph(self):
        self.assertEqual(lines("#nope"), ["#nope"])

    def test_closing_hashes_are_dropped(self):
        self.assertEqual(lines("## Title ##")[0], "Title")

    def test_unicode_rule_glyph(self):
        out = lines("# T", RICH)
        self.assertEqual(out[1], "━" * 1 * len("T") if False else "━" * 3)

    def test_heading_inline_markup(self):
        self.assertEqual(lines("# a **b** c")[0], "a **b** c" if False else "a b c")


class TestLists(unittest.TestCase):
    def test_bullets_and_nesting(self):
        out = lines("- a\n  - b\n    - c\n", RICH)
        self.assertEqual(out, ["• a", "  ◦ b", "    ▪ c"])

    def test_ascii_bullets(self):
        self.assertEqual(lines("- a\n  - b\n"), ["- a", "  * b"])

    def test_ordered_keeps_source_numbers(self):
        self.assertEqual(lines("3. three\n4. four\n"), ["3. three", "4. four"])

    def test_task_list(self):
        out = lines("- [ ] todo\n- [x] done\n", RICH)
        self.assertEqual(out, ["[ ] todo", "[✓] done"])
        self.assertEqual(lines("- [x] done\n"), ["[x] done"])

    def test_lazy_continuation_and_wrapping(self):
        out = lines("- one two three four five six seven\n  eight nine\n", width=20)
        self.assertEqual(out[0], "- one two three four")
        self.assertTrue(all(l.startswith("  ") for l in out[1:]))

    def test_loose_list_gets_blank_between_items(self):
        out = lines("- a\n\n- b\n")
        self.assertEqual(out, ["- a", "", "- b"])

    def test_tight_list_after_paragraph_has_no_blank(self):
        self.assertEqual(lines("text\n- a\n"), ["text", "- a"])

    def test_block_inside_item_is_indented(self):
        out = lines("1. step\n\n   ```\n   code\n   ```\n")
        self.assertEqual(out[0], "1. step")
        self.assertTrue(out[-1].startswith("   +"))

    def test_switching_marker_type_starts_a_new_list(self):
        out = lines("- a\n1. b\n")
        self.assertIn("- a", out)
        self.assertIn("1. b", out)

    def test_empty_item_still_shows_marker(self):
        self.assertEqual(lines("-\n- b\n"), ["-", "- b"])


class TestQuotes(unittest.TestCase):
    def test_bar_and_nesting(self):
        out = lines("> a\n>\n> > b\n", RICH)
        self.assertEqual(out, ["▌ a", "▌", "▌ ▌ b"])

    def test_ascii_bar(self):
        self.assertEqual(lines("> a\n"), ["| a"])

    def test_lazy_continuation(self):
        self.assertEqual(lines("> a\nb\n"), ["| a b"])

    def test_blank_line_closes_quote(self):
        out = lines("> a\n\nb\n")
        self.assertEqual(out, ["| a", "", "b"])

    def test_list_inside_quote(self):
        self.assertEqual(lines("> - x\n> - y\n"), ["| - x", "| - y"])


class TestCodeBlocks(unittest.TestCase):
    def test_fenced_box_and_label(self):
        out = lines("```python\nx = 1\n```\n", RICH)
        self.assertTrue(out[0].startswith("╭─ python "))
        self.assertTrue(out[0].endswith("╮"))
        self.assertIn("x = 1", out[1])
        self.assertTrue(out[-1].startswith("╰") and out[-1].endswith("╯"))

    def test_ascii_box(self):
        out = lines("```\ncode\n```\n")
        self.assertTrue(out[0].startswith("+-"))
        self.assertTrue(out[1].startswith("| code"))
        self.assertTrue(out[-1].startswith("+-"))

    def test_code_is_not_inline_parsed(self):
        out = plain("```\n**not bold** `x` [a](b)\n```\n")
        self.assertIn("**not bold** `x` [a](b)", out)

    def test_tildes_and_info_string(self):
        out = lines("~~~ruby extra\nputs 1\n~~~\n", RICH)
        self.assertIn("ruby", out[0])
        self.assertIn("puts 1", out[1])

    def test_unclosed_fence_is_closed_on_flush(self):
        out = lines("```\ncode\n")
        self.assertTrue(out[-1].startswith("+-"))

    def test_long_lines_are_truncated_not_wrapped(self):
        out = lines("```\n" + "x" * 200 + "\n```\n", width=40)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(len(l) <= 40 for l in out))
        self.assertIn("...", out[1])

    def test_indented_code_block(self):
        out = lines("    one\n    two\n")
        self.assertTrue(out[0].startswith("+-"))
        self.assertIn("one", out[1])
        self.assertIn("two", out[2])

    def test_blank_lines_inside_fence_are_kept(self):
        out = lines("```\na\n\nb\n```\n")
        self.assertEqual(len(out), 5)

    def test_box_width_matches_terminal(self):
        for w in (20, 40, 72):
            out = lines("```py\nx=1\n```\n", RICH, width=w)
            self.assertEqual({display_width(l) for l in out}, {w})


class TestTables(unittest.TestCase):
    def test_borders_and_alignment(self):
        out = lines("| a | bb |\n|:--|---:|\n| 1 | 2 |\n", RICH)
        self.assertTrue(out[0].startswith("╭") and out[0].endswith("╮"))
        self.assertIn("┼", out[2])
        self.assertTrue(out[-1].startswith("╰"))
        self.assertIn("│ 1 │  2 │", out[3])

    def test_center_alignment(self):
        out = lines("| head |\n|:----:|\n| x |\n")
        self.assertIn("|  x   |", out[3])

    def test_ascii_table(self):
        out = lines("| a |\n|---|\n| 1 |\n")
        self.assertTrue(out[0].startswith("+-"))
        self.assertIn("| 1 |", out[3])

    def test_not_a_table_without_delimiter_row(self):
        out = lines("| a | b |\nplain\n")
        self.assertEqual(out, ["| a | b | plain"])

    def test_cells_wrap_when_narrow(self):
        doc = "| col | other |\n|---|---|\n| a very long cell value here | x |\n"
        out = lines(doc, width=30)
        self.assertTrue(all(display_width(l) <= 30 for l in out))
        self.assertGreater(len(out), 5)

    def test_escaped_pipe_stays_in_cell(self):
        out = plain("| a |\n|---|\n| x \\| y |\n")
        self.assertIn("x | y", out)

    def test_inline_markup_in_cells(self):
        self.assertIn("bold", plain("| a |\n|---|\n| **bold** |\n"))


class TestInline(unittest.TestCase):
    def test_emphasis_strips_markers(self):
        out = plain("**b** *i* _i_ ~~s~~", MONO_TTY)
        self.assertNotIn("**", out)
        self.assertIn("b", out)

    def test_code_span_is_not_reparsed(self):
        self.assertIn("a **b** c", plain("`a **b** c`"))

    def test_backslash_escapes(self):
        self.assertEqual(lines(r"\*not em\* and \_this\_")[0], "*not em* and _this_")

    def test_intraword_underscore_is_literal(self):
        self.assertIn("snake_case_name", plain("snake_case_name"))

    def test_link_shows_url_when_no_hyperlink_support(self):
        out = plain("[docs](http://example.com/x)")
        self.assertIn("docs", out)
        self.assertIn("http://example.com/x", out)

    def test_link_uses_osc8_when_supported(self):
        out = render("[docs](http://example.com/x)", LINKS)
        self.assertIn("\x1b]8;;http://example.com/x", out)
        self.assertNotIn("(http://example.com/x)", strip_ansi(out))

    def test_bare_url_is_linkified(self):
        out = plain("see https://x.io/a, ok")
        self.assertIn("https://x.io/a", out)
        self.assertNotIn("https://x.io/a,", out.replace("https://x.io/a,", "@"))

    def test_autolink_angle_brackets(self):
        self.assertIn("http://a.b", plain("<http://a.b>"))

    def test_image_marker(self):
        self.assertIn("[img]", plain("![alt](p.png)"))

    def test_unmatched_markers_stay_literal(self):
        self.assertIn("**unclosed", plain("**unclosed bold"))

    def test_hard_break_splits_lines(self):
        self.assertEqual(lines("one  \ntwo"), ["one", "two"])
        self.assertEqual(lines("one\\\ntwo"), ["one", "two"])

    def test_soft_break_joins_lines(self):
        self.assertEqual(lines("one\ntwo"), ["one two"])


class TestCapsAndLayout(unittest.TestCase):
    def test_ascii_only_when_unicode_is_off(self):
        for doc in DOCS:
            if not doc.isascii():
                continue
            out = render(doc, ASCII)
            self.assertTrue(out.isascii(), f"non-ascii glyph for {doc[:30]!r}")

    def test_no_colour_uses_attributes_only(self):
        # A colourless tty still has the whole SGR attribute vocabulary, which is
        # where the `mono` theme keeps its meaning: bold, dim, italic, underline,
        # reverse and strike are all fair game -- colour parameters are not.
        out = render("# H\n\n**b** *i* ~~s~~ `c` text\n", MONO_TTY)
        self.assertNotIn("38;2;", out)
        self.assertNotIn("38;5;", out)
        allowed = {"0", "1", "2", "3", "4", "7", "9"}
        found = SGR.findall(out)
        for code in found:
            self.assertLessEqual(set(code[2:-1].split(";")), allowed, code)
        # ... and it must actually still say something.
        self.assertTrue(set(found) - {RESET}, "mono rendering lost every attribute")

    def test_pipe_output_has_no_escapes_at_all(self):
        out = render("# H\n\n**b** `c` [l](u)\n", ASCII)
        self.assertNotIn("\x1b", out)

    def test_width_and_indent_are_respected(self):
        for caps in (RICH, ASCII, NARROW):
            for width in (20, 33, 72):
                for doc in DOCS:
                    out = render(doc, caps, width=width, indent="  ")
                    for ln in out.split("\n")[:-1]:
                        self.assertLessEqual(display_width(ln), width,
                                             f"{ln!r} in {doc[:20]!r}")
                        self.assertTrue(ln == "" or ln.startswith("  ") or not ln.strip())

    def test_wide_characters_are_measured(self):
        out = lines("中文字符 " * 20, RICH, width=30)
        for ln in out:
            self.assertLessEqual(display_width(ln), 30)

    def test_no_trailing_whitespace_and_no_double_blanks(self):
        for caps in (RICH, ASCII):
            for doc in DOCS:
                out = render(doc, caps)
                prev_blank = False
                for ln in out.split("\n")[:-1]:
                    self.assertEqual(ln, ln.rstrip(), repr(ln))
                    self.assertFalse(prev_blank and ln == "", f"double blank in {doc[:20]!r}")
                    prev_blank = ln == ""
                self.assertFalse(out.startswith("\n"), repr(doc[:20]))

    def test_styles_are_closed_at_every_line_end(self):
        for caps in (RICH, LINKS, MONO_TTY):
            for doc in DOCS:
                for ln in render(doc, caps).split("\n"):
                    codes = SGR.findall(ln)
                    if codes:
                        self.assertEqual(codes[-1], RESET, repr(ln[:60]))
                    self.assertNotIn("\x1b", strip_ansi(ln))

    def test_default_width_comes_from_caps(self):
        caps = Caps(color=0, unicode=False, is_tty=False, width=25)
        for ln in render("word " * 30, caps).split("\n"):
            self.assertLessEqual(len(ln), 25)


class TestHighlight(unittest.TestCase):
    caps = RICH

    def setUp(self):
        self.theme = theme_for(self.caps)

    def hl(self, code, lang):
        return highlight(code, lang, self.theme, self.caps)

    def styled(self, code, lang, needle):
        """Return the style opener that `needle` is painted with.

        A style is one *or more* SGR sequences — attributes first, then colour —
        so the opener is the whole run of escapes sitting immediately before the
        text, minus the reset that closed the previous token.
        """
        out = self.hl(code, lang)
        idx = out.index(needle)
        runs = SGR.finditer(out, 0, idx)
        codes, end = [], None
        for m in runs:
            if end is not None and m.start() != end:
                codes = []
            codes.append(m.group(0))
            end = m.end()
        while codes and codes[0] == RESET:
            codes.pop(0)
        return "".join(codes)

    def test_text_is_preserved_exactly(self):
        for lang in ("python", "javascript", "json", "bash", "html", "css", "sql",
                     "go", "rust", "c", "cpp", "java", "yaml", "toml", "markdown",
                     "diff", None, "nonsense"):
            code = 'a = "x"  # 1\nb(1, 2)\n'
            self.assertEqual(strip_ansi(self.hl(code, lang)), code, lang)

    def test_required_languages_are_supported(self):
        names = supported_languages()
        for lang in ("python", "javascript", "typescript", "json", "bash", "sh",
                     "html", "css", "sql", "go", "rust", "c", "c++", "java",
                     "yaml", "toml", "markdown", "diff"):
            self.assertIn(lang, names)

    def test_unknown_language_renders_plainly(self):
        out = self.hl("def x(): pass", "brainfuck")
        self.assertEqual(strip_ansi(out), "def x(): pass")
        # Exactly one style, `syn.plain`, over the whole block: no keyword, no
        # string, no comment colour leaks in from a language we do not know.
        plain_style = set(SGR.findall(self.theme["syn.plain"].codes(self.caps)))
        self.assertEqual(set(SGR.findall(out)), plain_style | {RESET})

    def test_hash_inside_string_is_not_a_comment(self):
        comment = self.theme["syn.comment"].codes(self.caps)
        string = self.theme["syn.string"].codes(self.caps)
        self.assertEqual(self.styled('x = "a # b"', "python", '"a # b"'), string)
        self.assertNotEqual(string, comment)

    def test_quote_inside_comment_does_not_open_a_string(self):
        code = "# it's fine\nx = 1\n"
        out = self.hl(code, "python")
        comment = self.theme["syn.comment"].codes(self.caps)
        number = self.theme["syn.number"].codes(self.caps)
        self.assertTrue(out.startswith(comment))
        self.assertIn(number + "1", out)

    def test_triple_quoted_string_spans_lines(self):
        code = 'a = """one\ntwo # not comment\n"""\nb = 2\n'
        string = self.theme["syn.string"].codes(self.caps)
        self.assertEqual(self.styled(code, "python", "two # not comment"), string)

    def test_block_comment_spans_lines(self):
        code = "/* one\n   two \"quote\"\n*/\nint x;\n"
        comment = self.theme["syn.comment"].codes(self.caps)
        self.assertEqual(self.styled(code, "c", 'two "quote"'), comment)

    def test_keywords_builtins_numbers_and_functions(self):
        kw = self.theme["syn.keyword"].codes(self.caps)
        fn = self.theme["syn.func"].codes(self.caps)
        num = self.theme["syn.number"].codes(self.caps)
        dec = self.theme["syn.decorator"].codes(self.caps)
        code = "@deco\ndef greet(n):\n    return 0x1f\n"
        self.assertEqual(self.styled(code, "python", "def"), kw)
        self.assertEqual(self.styled(code, "python", "greet"), fn)
        self.assertEqual(self.styled(code, "python", "0x1f"), num)
        self.assertEqual(self.styled(code, "python", "@deco"), dec)

    def test_escaped_quote_does_not_end_string(self):
        code = 'a = "x \\" still" + b\n'
        string = self.theme["syn.string"].codes(self.caps)
        self.assertEqual(self.styled(code, "python", '"x \\" still"'), string)

    def test_sql_is_case_insensitive(self):
        kw = self.theme["syn.keyword"].codes(self.caps)
        self.assertEqual(self.styled("select 1 from t", "sql", "select"), kw)
        self.assertEqual(self.styled("SELECT 1 FROM t", "sql", "SELECT"), kw)

    def test_json_keys_differ_from_values(self):
        code = '{"k": "v"}'
        self.assertNotEqual(self.styled(code, "json", '"k"'),
                            self.styled(code, "json", '"v"'))

    def test_diff_marks_additions_and_deletions(self):
        code = "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n"
        self.assertNotEqual(self.styled(code, "diff", "-old"),
                            self.styled(code, "diff", "+new"))

    def test_bash_variables_and_html_tags(self):
        # `syn.variable` is byte-identical to `syn.plain` in every theme, so a
        # variable painted with it looks like the string ended mid-token.
        var = self.theme["syn.decorator"].codes(self.caps)
        self.assertEqual(self.styled('echo "$HOME"', "bash", "$HOME"), var)
        self.assertNotEqual(var, self.theme["syn.plain"].codes(self.caps))
        self.assertNotEqual(var, self.theme["syn.string"].codes(self.caps))
        kw = self.theme["syn.keyword"].codes(self.caps)
        self.assertEqual(self.styled("<div class='a'>hi</div>", "html", "div"), kw)

    def test_aliases_resolve(self):
        for a, b in (("py", "python"), ("js", "javascript"), ("sh", "bash"),
                     ("c++", "cpp"), ("yml", "yaml"), ("md", "markdown")):
            code = "x = 1\n"
            self.assertEqual(self.hl(code, a), self.hl(code, b), (a, b))

    def test_no_colour_highlighting_is_still_plain_text(self):
        caps = ASCII
        out = highlight("def f(): pass", "python", theme_for(caps), caps)
        self.assertEqual(out, "def f(): pass")

    def test_multiline_state_is_per_block_not_global(self):
        doc = "```python\ns = '''open\n```\n\n```python\nx = 1\n```\n"
        out = plain(doc, RICH)
        self.assertIn("x = 1", out)


class TestRobustness(unittest.TestCase):
    def test_pathological_inputs_do_not_raise(self):
        weird = [
            "*" * 200,
            "`" * 50,
            "[" * 40 + "]" * 40,
            "|" * 30 + "\n" + "-|" * 30,
            ">" * 40 + " deep",
            "- " * 60,
            "#" * 10 + " h",
            "\t\ttabbed\n",
            "a" * 500,
            "\x00 nul \x07 bel",
            "```" + "`" * 20 + "\ncode\n",
            "    \n   \n  \n \n",
        ]
        for doc in weird:
            for caps in (RICH, ASCII, NARROW):
                out = render(doc, caps, width=17)
                st = MarkdownStream(theme_for(caps), caps, width=17)
                got = "".join(st.feed(c) for c in doc) + st.close()
                self.assertEqual(got, out, repr(doc[:20]))

    def test_deep_nesting_is_bounded(self):
        doc = "".join("  " * i + "- level %d\n" % i for i in range(12))
        out = plain(doc, RICH, width=60)
        self.assertIn("level 11", out)

    def test_very_long_word_is_hard_split(self):
        out = lines("x" * 90, width=20)
        self.assertTrue(all(len(l) <= 20 for l in out))
        self.assertEqual("".join(out), "x" * 90)

    def test_render_markdown_equals_stream(self):
        doc = "# a\n\nb **c**\n"
        st = MarkdownStream(theme_for(RICH), RICH)
        self.assertEqual(st.feed(doc) + st.close(), render(doc, RICH))


# --------------------------------------------------------------------------
# Regression tests for the findings of the 2026-08 review. Every one of these
# was watched failing against the code as it stood before the matching fix.
# --------------------------------------------------------------------------

#: Everything a reply could try to do to the terminal it is printed on.
ATTACKS = {
    "osc52_clipboard": "\x1b]52;c;cHduZWQ=\x07",
    "osc0_window_title": "\x1b]0;pwned\x07",
    "clear_screen_and_home": "\x1b[2J\x1b[H",
    "alternate_buffer": "\x1b[?1049h",
    "dcs_string": "\x1bP0;1|xx\x1b\\",
    "eight_bit_csi": "\x9b31m",
    "carriage_return_overwrite": "visible\rOVERWRITTEN",
}

#: Every block context the payload can be smuggled through.
CONTEXTS = {
    "body": "{p}",
    "fence": "```\n{p}\n```",
    "fence_info": "```{p}\ncode\n```",
    "indented_code": "    {p}",
    "table_cell": "| head | two |\n|---|---|\n| {p} | x |",
    "heading": "# {p}",
    "link_label": "[{p}](http://example.com/ok)",
    "link_url": "[label](http://example.com/{p})",
    "bare_url": "http://example.com/{p}",
    "autolink": "<http://example.com/{p}>",
    "quote": "> {p}",
    "list_item": "- {p}",
    "task": "- [x] {p}",
    "linkdef": "[ref]: http://example.com/{p}",
}

#: The renderer's *own* escapes: SGR, OSC 8 hyperlinks and their terminators.
_OURS = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;[^\x1b\x07]*(?:\x1b\\|\x07)")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def terminal_payload(rendered):
    """What is left after removing the escapes the renderer legitimately emits."""
    return _CONTROL.findall(_OURS.sub("", rendered))


class TestEscapeInjection(unittest.TestCase):
    """Untrusted model output must not be able to drive the terminal."""

    def all_caps(self):
        for hyper in (True, False):
            for color in (24, 0):
                yield Caps(color=color, unicode=True, is_tty=True, width=60,
                           hyperlinks=hyper, animation=True)

    def test_no_attack_reaches_the_terminal_from_any_context(self):
        for name, payload in ATTACKS.items():
            for where, tmpl in CONTEXTS.items():
                doc = tmpl.format(p=payload) + "\n"
                for caps in self.all_caps():
                    out = render_markdown(doc, theme_for(caps), caps, 60)
                    left = terminal_payload(out)
                    self.assertEqual(
                        left, [],
                        f"{name} survived in {where} (hyperlinks={caps.hyperlinks}): "
                        f"{left!r} in {out!r}")

    def test_the_same_holds_when_the_payload_is_streamed(self):
        caps = Caps(color=24, unicode=True, is_tty=True, width=40,
                    hyperlinks=True, animation=True)
        for payload in ATTACKS.values():
            doc = "para\n\n```\n%s\n```\n\n| a |\n|---|\n| %s |\n" % (payload, payload)
            st = MarkdownStream(theme_for(caps), caps, 40)
            out = "".join(st.feed(c) for c in doc) + st.close()
            self.assertEqual(terminal_payload(out), [], repr(payload))

    def test_carriage_return_cannot_overwrite_what_was_printed(self):
        for caps in self.all_caps():
            out = strip_ansi(render_markdown("visible\rOVERWRITTEN\n",
                                             theme_for(caps), caps, 60))
            self.assertNotIn("\r", out)
            self.assertIn("visible", out)

    def test_highlight_sanitises_its_own_input(self):
        caps = Caps(color=24, unicode=True, is_tty=True, width=60, animation=True)
        for payload in ATTACKS.values():
            out = highlight("x = '%s'" % payload, "python", theme_for(caps), caps)
            self.assertEqual(terminal_payload(out), [], repr(payload))
        self.assertEqual(terminal_payload(highlight("a\x1b[2Jb", None, theme_for(caps), caps)), [])

    def test_hyperlink_payload_is_filtered_not_just_the_text(self):
        # A tab is the one character `sanitize_text` keeps and `sanitize_url`
        # removes, so it isolates the URL filter from the text filter.
        caps = Caps(color=8, unicode=True, is_tty=True, width=60,
                    hyperlinks=True, animation=True)
        out = render_markdown("[x](<http://e.example/a\tb>)\n", theme_for(caps), caps, 60)
        osc = re.search(r"\x1b\]8;;([^\x1b\x07]*)", out)
        self.assertIsNotNone(osc, out)
        self.assertNotIn("\t", osc.group(1))

    def test_the_layer_that_prints_a_url_filters_it_itself(self):
        # When OSC 8 is unavailable the URL is printed as dimmed text, a channel
        # `ansi.hyperlink()` never sees. Fed straight to the node renderer, with
        # the input filter bypassed, that path must still be clean on its own.
        from lume.markdown import _Glyphs, _render_nodes

        for hyper in (True, False):
            caps = Caps(color=8, unicode=True, is_tty=True, width=60,
                        hyperlinks=hyper, animation=True)
            theme = theme_for(caps)
            node = ("link", [("text", "label")],
                    "http://e.example/\x1b]52;c;cHduZWQ=\x07a", False)
            out = _render_nodes([node], theme, caps, theme["app.text"], True,
                                _Glyphs(True))
            self.assertEqual(terminal_payload(out), [], (hyper, out))
            self.assertIn("label", strip_ansi(out))


class TestHighlightByteFidelity(unittest.TestCase):
    """Code in a chat client gets copy-pasted; a highlighter may not eat bytes."""

    CAPS = Caps(color=24, unicode=True, is_tty=True, width=100, animation=True)

    SAMPLES = [
        "AT&T rocks",
        "5 < 3",
        "a < b && c > d",
        "if (x<y) { p(&z); }",
        "<div class='a'>hi &amp; bye</div>",
        "x = 1  # trailing",
        "'unterminated",
        '"also unterminated',
        "/* open block",
        "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+++ b/added",
        "$HOME/${X}/$1 $?",
        "r#\"raw \" string\"#",
        "fn f<'a>(s: &'a str) -> &'a str { s }",
        "key: value # note",
        "[table]\nk = \"v\"",
        "| a | b |",
        "\t tabs\tand  spaces ",
        "unicode 中文 é́ ok",
        "&&&<<<>>>;;;",
        "",
    ]

    def test_every_supported_language_round_trips_byte_for_byte(self):
        theme = theme_for(self.CAPS)
        for lang in supported_languages() + [None, "nonsense"]:
            for sample in self.SAMPLES:
                got = strip_ansi(highlight(sample, lang, theme, self.CAPS))
                self.assertEqual(got, sample, f"{lang}: {sample!r} -> {got!r}")

    def test_the_html_family_keeps_ampersands_and_comparisons(self):
        theme = theme_for(self.CAPS)
        for lang in ("html", "xml", "svg", "vue", "htm"):
            for sample in ("AT&T rocks", "5 < 3", "a & b < c"):
                self.assertEqual(strip_ansi(highlight(sample, lang, theme, self.CAPS)),
                                 sample, lang)

    def test_fenced_code_keeps_every_byte_too(self):
        caps = Caps(color=24, unicode=True, is_tty=True, width=100, animation=True)
        body = "AT&T rocks\n5 < 3\n"
        out = strip_ansi(render_markdown("```html\n" + body + "```\n",
                                         theme_for(caps), caps, 100))
        for line in body.rstrip("\n").split("\n"):
            self.assertIn(line, out)


class TestRenderingCost(unittest.TestCase):
    """A model emits prose as one long paragraph; that must not be quadratic."""

    CAPS = Caps(color=24, unicode=True, is_tty=True, width=80, animation=True)

    def elapsed(self, fn):
        best = None
        for _ in range(2):
            t = time.perf_counter()
            fn()
            d = time.perf_counter() - t
            best = d if best is None else min(best, d)
        return best

    def growth(self, make, small, big):
        """Time ratio for a `big / small` size ratio — quadratic shows up as its square."""
        a = self.elapsed(lambda: make(small))
        b = self.elapsed(lambda: make(big))
        return b / max(a, 1e-4)

    def test_feeding_a_paragraph_one_character_at_a_time_is_not_quadratic(self):
        theme = theme_for(self.CAPS)

        def run(n):
            text = ("word " * n).strip()
            st = MarkdownStream(theme, self.CAPS, 80)
            for ch in text:
                st.feed(ch)
            st.close()

        # 4x the input; linear predicts ~4x the time, quadratic ~16x.
        self.assertLess(self.growth(run, 400, 1600), 10.0)

    def test_rendering_a_long_paragraph_in_one_call_is_not_quadratic(self):
        theme = theme_for(self.CAPS)

        def run(n):
            render_markdown("\n".join("word " * 12 for _ in range(n)),
                            theme, self.CAPS, 80)

        self.assertLess(self.growth(run, 200, 800), 10.0)

    def test_a_streamed_blockquote_is_not_quadratic_either(self):
        theme = theme_for(self.CAPS)

        def run(n):
            doc = "\n".join("> word word word word word word" for _ in range(n))
            st = MarkdownStream(theme, self.CAPS, 80)
            for i in range(0, len(doc), 20):
                st.feed(doc[i:i + 20])
            st.close()

        self.assertLess(self.growth(run, 100, 400), 10.0)

    def test_a_wall_of_open_brackets_is_cheap(self):
        # Every '[' used to rescan the whole rest of the line for a ']'.
        theme = theme_for(self.CAPS)
        run = lambda n: render_markdown("[" * n, theme, self.CAPS, 80)
        self.assertLess(self.growth(run, 2000, 8000), 10.0)


class TestNestingIsBounded(unittest.TestCase):
    """Untrusted input may not overflow the terminal or the interpreter stack."""

    def test_deeply_nested_lists_never_exceed_the_width(self):
        doc = "".join("  " * i + "- level %d\n" % i for i in range(39))
        for w in (20, 40, 80, 200):
            for caps in (RICH, ASCII):
                caps = caps.with_size(w, 24)
                out = render_markdown(doc, theme_for(caps), caps, w)
                for line in out.split("\n"):
                    self.assertLessEqual(display_width(strip_ansi(line)), w,
                                         f"w={w} {strip_ansi(line)!r}")

    def test_two_hundred_quote_markers_do_not_raise(self):
        doc = ">" * 200 + " x\n"
        for caps in (RICH, ASCII):
            want = render_markdown(doc, theme_for(caps), caps, 80)
            st = MarkdownStream(theme_for(caps), caps, 80)
            got = "".join(st.feed(c) for c in doc) + st.close()
            self.assertEqual(got, want)
            self.assertLessEqual(
                max(display_width(strip_ansi(l)) for l in want.split("\n")), 80)
            self.assertIn("x", strip_ansi(want))

    def test_deep_nesting_of_every_container_kind(self):
        for unit in ("> ", "- ", "1. "):
            doc = "".join(unit * i + "text %d\n" % i for i in range(1, 40))
            for w in (24, 80):
                out = render_markdown(doc, theme_for(RICH), RICH, w)
                for line in out.split("\n"):
                    self.assertLessEqual(display_width(strip_ansi(line)), w, repr(line))

    def test_an_indent_wider_than_the_terminal_is_fitted(self):
        out = render_markdown("# h\n\npara text here\n\n```py\nx=1\n```\n",
                              theme_for(RICH), RICH, 10, indent=" " * 30)
        for line in out.split("\n"):
            self.assertLessEqual(display_width(strip_ansi(line)), 10, repr(line))
        self.assertIn("para", strip_ansi(out))


class TestStreamingSafetyAtNarrowWidths(unittest.TestCase):
    """Wide test paragraphs never wrap, so they never exercise `_stable_cut`."""

    WORDS = ["word", "`code", "span`", "**bold**", "[lab](http://x.y/z)", "_it_",
             "a*b", "~~s~~", "longerword", "中文字", "x", "**un",
             "closed**", "`tick", "tick`", "[ref]", "(paren)", "http://u.rl/p"]

    def check(self, doc, caps, width):
        want = render_markdown(doc, theme_for(caps), caps, width)
        st = MarkdownStream(theme_for(caps), caps, width)
        got = "".join(st.feed(c) for c in doc) + st.close()
        if got == want:
            return
        a, b = want.split("\n"), got.split("\n")
        for i in range(max(len(a), len(b))):
            x = a[i] if i < len(a) else "<eof>"
            y = b[i] if i < len(b) else "<eof>"
            if x != y:
                self.fail(f"w={width} {doc[:60]!r} line {i}\n"
                          f"  want {x!r}\n  got  {y!r}")
        self.fail("outputs differ in length")

    def test_paragraphs_that_wrap_many_times(self):
        rng = random.Random(20260817)
        for _ in range(120):
            width = rng.randint(24, 40)
            caps = Caps(color=rng.choice([0, 4, 24]), unicode=rng.choice([True, False]),
                        is_tty=True, width=width, hyperlinks=rng.choice([True, False]),
                        animation=True)
            doc = " ".join(rng.choice(self.WORDS) for _ in range(rng.randint(30, 90)))
            if rng.random() < 0.4:
                doc = "\n".join(doc[i:i + 40] for i in range(0, len(doc), 40))
            self.check(doc, caps, width)

    def test_wrapping_paragraphs_inside_containers(self):
        rng = random.Random(4242)
        for prefix, join in (("> ", "\n> "), ("- ", "\n  "), ("1. ", "\n   ")):
            for width in range(24, 41, 4):
                caps = Caps(color=24, unicode=True, is_tty=True, width=width,
                            hyperlinks=False, animation=True)
                body = " ".join(rng.choice(self.WORDS) for _ in range(60))
                doc = prefix + join.join(body[i:i + 50] for i in range(0, len(body), 50))
                self.check(doc, caps, width)

    def test_a_construct_that_straddles_every_wrap_point(self):
        for marker in ("`code`", "[lab](http://x.y)", "**bold**", "_it_", "~~s~~"):
            for pad_words in range(1, 30):
                doc = "word " * pad_words + marker + " tail tail tail tail tail"
                for width in (24, 31, 40):
                    caps = Caps(color=24, unicode=True, is_tty=True, width=width,
                                hyperlinks=False, animation=True)
                    self.check(doc, caps, width)


class TestInvariantsWithoutWhichNothingFails(unittest.TestCase):
    """Direct cover for the checks a whole-suite run happened not to exercise."""

    def test_a_shorter_run_of_backticks_does_not_close_a_fence(self):
        out = strip_ansi(render_markdown("````\n```\nstill code\n````\nafter\n",
                                         theme_for(RICH), RICH, 72))
        self.assertIn("```", out)
        self.assertIn("still code", out)
        body = out.split("\n")
        box = [l for l in body if l.startswith("│")]
        self.assertEqual(len(box), 2, out)          # both lines inside one box
        self.assertIn("after", out)

    def test_a_tilde_fence_is_not_closed_by_backticks(self):
        out = strip_ansi(render_markdown("~~~~\n~~~\ncode\n~~~~\n", theme_for(RICH), RICH, 72))
        self.assertEqual(len([l for l in out.split("\n") if l.startswith("│")]), 2, out)

    def test_a_delimiter_row_must_have_the_header_column_count(self):
        out = strip_ansi(render_markdown("| a | b | c |\n|---|---|\n| 1 | 2 | 3 |\n",
                                         theme_for(RICH), RICH, 72))
        self.assertNotIn("╭", out)             # no table was drawn
        self.assertIn("|---|---|", out)
        good = strip_ansi(render_markdown("| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |\n",
                                          theme_for(RICH), RICH, 72))
        self.assertIn("╭", good)

    def test_never_two_blank_lines_even_when_a_block_prints_nothing(self):
        # An empty blockquote asks for its separator and then emits nothing.
        for doc in ("para\n\n> \n\n---\n", "para\n\n>\n\n# head\n", "1. one\n> \n---\n"):
            for caps in (RICH, ASCII):
                out = strip_ansi(render_markdown(doc, theme_for(caps), caps, 60))
                self.assertNotIn("\n\n\n", out, repr(doc))

    def test_the_space_between_two_words_is_never_inside_a_style(self):
        # `wrap()` defers whitespace until the next word fits, which leaves it
        # inside the *next* word's SGR codes; `_tidy` moves it back out.
        inside = re.compile(r"\x1b\[[0-9;]*m(?<!\x1b\[0m) ")
        docs = ["word **bold** more words here and there",
                "a [link](http://x.example) tail text",
                "start `code` middle *it* end",
                "| a | b |\n|---|---|\n| **x** y | z |",
                "> quote **b** c and more text to wrap around the line",
                "# heading **b** c",
                "- item **b** c"]
        for doc in docs:
            for width in (24, 40, 72):
                out = render_markdown(doc, theme_for(RICH), RICH, width)
                self.assertIsNone(inside.search(out), f"{doc[:30]!r} w={width}: {out!r}")


class TestSetextAndReferenceDefinitions(unittest.TestCase):
    def test_an_equals_underline_makes_a_heading(self):
        out = lines("Title\n=====\n\nbody\n", RICH)
        self.assertEqual(out[0], "Title")
        self.assertEqual(set(out[1]), {"━"})
        self.assertNotIn("Title =====", "\n".join(out))

    def test_the_underline_never_reprints_committed_text(self):
        # Once a line is out it cannot be promoted, so a wrapped paragraph keeps
        # the current (documented) behaviour instead of breaking append-only.
        doc = "word " * 40 + "\n===\n"
        for caps in (RICH, ASCII):
            want = render_markdown(doc, theme_for(caps), caps, 40)
            st = MarkdownStream(theme_for(caps), caps, 40)
            got = "".join(st.feed(c) for c in doc) + st.close()
            self.assertEqual(got, want)

    def test_dashes_are_still_a_thematic_break(self):
        out = lines("text\n---\nmore\n", ASCII)
        self.assertIn("-" * 10, "\n".join(out))

    def test_reference_definitions_get_one_line_each(self):
        out = lines("[1]: http://x.example/one\n[2]: http://y.example/two\n\nsee [1]\n", ASCII)
        self.assertEqual(out[0], "[1] http://x.example/one")
        self.assertEqual(out[1], "[2] http://y.example/two")
        self.assertIn("see [1]", "\n".join(out))

    def test_a_definition_after_prose_stays_prose(self):
        out = lines("para text\n[1]: http://x.example\n", ASCII)
        self.assertEqual(out, ["para text [1]: http://x.example"])

    def test_definitions_stream_identically(self):
        doc = "[1]: http://x.example/one \"Title\"\n[2]: http://y.example/two\n\nbody\n"
        for caps in (RICH, ASCII, LINKS):
            want = render_markdown(doc, theme_for(caps), caps, 48)
            st = MarkdownStream(theme_for(caps), caps, 48)
            self.assertEqual("".join(st.feed(c) for c in doc) + st.close(), want)


class TestHighlighterDetails(unittest.TestCase):
    CAPS = Caps(color=24, unicode=True, is_tty=True, width=100, animation=True)

    def styled_of(self, code, lang, fragment):
        """The SGR codes in force over `fragment`."""
        out = highlight(code, lang, theme_for(self.CAPS), self.CAPS)
        plain_out = strip_ansi(out)
        at = plain_out.index(fragment)
        seen, codes, pos = [], [], 0
        for m in re.finditer(r"\x1b\[[0-9;]*m|[^\x1b]+", out):
            tok = m.group(0)
            if tok.startswith("\x1b"):
                codes = [] if tok == RESET else codes + [tok]
                continue
            if pos <= at < pos + len(tok):
                seen = list(codes)
                break
            pos += len(tok)
        return "".join(seen)

    def test_a_rust_lifetime_is_not_a_string(self):
        code = "fn f<'a>(s: &'a str) -> &'a str { s }"
        string = theme_for(self.CAPS)["syn.string"].codes(self.CAPS)
        self.assertNotEqual(self.styled_of(code, "rust", "'a"), string)
        # ... and the rest of the signature keeps its own colours.
        self.assertEqual(self.styled_of(code, "rust", "str"),
                         theme_for(self.CAPS)["syn.type"].codes(self.CAPS))

    def test_a_rust_char_literal_is_still_a_string(self):
        string = theme_for(self.CAPS)["syn.string"].codes(self.CAPS)
        self.assertEqual(self.styled_of("let c = 'a';", "rust", "'a'"), string)
        self.assertEqual(self.styled_of("let c = '\\n';", "rust", "'\\n'"), string)

    def test_a_rust_raw_hash_string_is_one_string(self):
        string = theme_for(self.CAPS)["syn.string"].codes(self.CAPS)
        code = 'let s = r#"raw " string"#;'
        self.assertEqual(self.styled_of(code, "rust", 'r#"raw " string"#'), string)
        self.assertEqual(self.styled_of(code, "rust", ";"),
                         theme_for(self.CAPS)["syn.punct"].codes(self.CAPS))

    def test_a_diff_of_a_diff_keeps_its_content_lines(self):
        code = "--- a\n+++ b\n@@ -1 +1 @@\n+++ b/inner.py\n--- a/inner.py\n"
        header = self.styled_of(code, "diff", "--- a\n")
        added = self.styled_of(code, "diff", "+++ b/inner.py")
        removed = self.styled_of(code, "diff", "--- a/inner.py")
        self.assertNotEqual(added, header)
        self.assertNotEqual(removed, header)
        self.assertNotEqual(added, removed)


class TestNoRegressionSweep(unittest.TestCase):
    """The properties the whole renderer is judged on, over the whole corpus."""

    EXTRA = [
        "# Head with 中文字符 and emoji \U0001f600️ combining á and RTL אבג",
        "| a | 中文字符 |\n|---|---|\n| \U0001f600️ | אבג x́ |\n| long cell here | y |",
        "```python\nx = 'very long line " + "y" * 120 + "'\n```",
        "".join("  " * i + "- L%d\n" % i for i in range(39)),
        ">" * 200 + " deep",
        "1234567890. huge marker item text here",
        "Title\n=====\n\n[1]: http://example.com/one\n[2]: http://example.com/two",
        "中" * 200,
    ]

    def test_no_overflow_no_trailing_space_no_double_blank(self):
        for doc in DOCS + self.EXTRA:
            for w in range(20, 201, 9):
                for uni in (True, False):
                    for ind in ("", "  "):
                        caps = Caps(color=24, unicode=uni, is_tty=True, width=w,
                                    hyperlinks=False, animation=True)
                        out = render_markdown(doc, theme_for(caps), caps, w, indent=ind)
                        self.assertNotIn("\n\n\n", out, (doc[:20], w))
                        for line in out.split("\n"):
                            p = strip_ansi(line)
                            self.assertLessEqual(display_width(p), w, (doc[:20], w, p))
                            self.assertEqual(p, p.rstrip(), (doc[:20], w, p))

    def test_code_boxes_and_tables_stay_rectangular(self):
        code = "```py\nx = 1\nlonger line of code\n```\n"
        table = "| a | bb |\n|---|---|\n| 1 | 2 |\n| much longer cell | y |\n"
        for w in range(20, 81):
            caps = Caps(color=0, unicode=True, is_tty=True, width=w, animation=True)
            box = [l for l in strip_ansi(render_markdown(code, theme_for(caps), caps, w)
                                         ).split("\n") if l.strip()]
            # The code box fills the terminal: its width is fixed before the
            # first code line has arrived, so it cannot be sized to the content.
            self.assertEqual({display_width(l) for l in box}, {w}, (w, box))
            grid = [l for l in strip_ansi(render_markdown(table, theme_for(caps), caps, w)
                                          ).split("\n") if l.strip()]
            # A table *is* buffered whole, so it is sized to its content — but it
            # still has to be a rectangle that fits.
            widths = {display_width(l) for l in grid}
            self.assertEqual(len(widths), 1, (w, grid))
            self.assertLessEqual(max(widths), w, (w, grid))


if __name__ == "__main__":
    unittest.main()
