"""Tests for lume.store — durability, safety, and the awkward cases."""

import contextlib
import errno
import json
import os
import pathlib
import random
import re
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lume import store as store_mod
from lume.ansi import display_width
from lume.store import (
    AmbiguousRefError, Message, SessionMeta, Store, StoreWarning,
    auto_title, default_root, new_id,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name) / "data"
        self.store = Store(self.root)

    # -- helpers

    def make(self, **kw):
        kw.setdefault("model", "claude-opus-5")
        return self.store.create(**kw)

    def path(self, sid):
        return self.store.sessions_dir / (sid + ".jsonl")

    def raw(self, sid):
        return self.path(sid).read_bytes()

    def write_raw(self, sid, data: bytes):
        self.path(sid).write_bytes(data)

    def quiet_load(self, sid):
        """load() ignoring the recovery warnings (asserted separately)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            return self.store.load(sid)


# --------------------------------------------------------------------- ids/titles


class TestIds(unittest.TestCase):
    def test_ids_are_unique_sortable_and_filename_safe(self):
        ids = [new_id() for _ in range(2000)]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, sorted(ids))
        for i in ids[:50]:
            self.assertRegex(i, r"^[0-9a-z]{26}$")

    def test_id_encodes_time_order(self):
        a = new_id(1000.0)
        b = new_id(2000.0)
        self.assertLess(a, b)


class TestTitles(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(auto_title("Hello there"), "Hello there")

    def test_newlines_and_tabs_collapse(self):
        t = auto_title("first line\nsecond line\t\tthird")
        self.assertNotIn("\n", t)
        self.assertEqual(t, "first line second line third")

    def test_very_long_input_is_clipped_to_display_columns(self):
        t = auto_title("word " * 400)
        self.assertLessEqual(display_width(t), 48)
        self.assertTrue(t)

    def test_emoji_counted_by_display_width(self):
        t = auto_title("🚀" * 200)
        self.assertLessEqual(display_width(t), 48)
        self.assertNotEqual(t, "Untitled")

    def test_mixed_emoji_and_cjk(self):
        t = auto_title("🎉 中文标题测试 " * 30)
        self.assertLessEqual(display_width(t), 48)

    def test_empty_and_whitespace_and_control(self):
        for bad in ("", "   ", "\n\n\t", "\x00\x00", "\x1b[31m\x1b[0m"):
            self.assertEqual(auto_title(bad), "Untitled")

    def test_markdown_lead_in_is_dropped(self):
        self.assertEqual(auto_title("### How do I sort a list?"), "How do I sort a list?")
        self.assertEqual(auto_title("- buy milk"), "buy milk")
        self.assertEqual(auto_title("1. buy milk"), "buy milk")

    def test_ansi_is_stripped(self):
        self.assertEqual(auto_title("\x1b[1;31mred\x1b[0m text"), "red text")

    def test_non_string_input(self):
        self.assertEqual(auto_title(12345), "12345")
        self.assertEqual(auto_title(None), "Untitled")

    def test_custom_width(self):
        self.assertLessEqual(display_width(auto_title("x" * 100, 10)), 10)


class TestDefaultRoot(unittest.TestCase):
    def test_lume_home_wins(self):
        self.assertEqual(default_root({"LUME_HOME": "/srv/lume", "XDG_DATA_HOME": "/x"}),
                         pathlib.Path("/srv/lume"))

    def test_xdg(self):
        self.assertEqual(default_root({"XDG_DATA_HOME": "/x"}), pathlib.Path("/x/lume"))

    def test_relative_xdg_is_ignored(self):
        self.assertNotEqual(default_root({"XDG_DATA_HOME": "relative"}),
                            pathlib.Path("relative/lume"))

    def test_fallback(self):
        self.assertEqual(default_root({}).name, "lume")


# ------------------------------------------------------------------------- paths


class TestPathSafety(Base):
    HOSTILE = [
        "../escape", "../../etc/passwd", "..", ".", "./x", "a/../../b",
        "/etc/passwd", "/tmp/x", "\\windows\\system32", "a\\b", "C:/x", "C:\\x",
        "nul\x00.jsonl", "a\x00b", "\x00", "a\nb", "a\tb", "\x1b[0m",
        "", " ", "x" * 65, "con", "COM1", "aux.jsonl", "sub/dir", "..hidden",
        "trailing.", "trailing ", "-leading", ".hidden", "\u200b", "ａｂｃ",
    ]

    def test_path_for_accepts_a_real_id(self):
        meta = self.make()
        p = self.store.path_for(meta.id)
        self.assertEqual(p.parent, self.store.sessions_dir)
        self.assertTrue(p.is_file())

    def test_path_for_rejects_traversal(self):
        for bad in self.HOSTILE:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.store.path_for(bad)

    def test_non_string_ids_rejected(self):
        for bad in (None, 5, b"abc", ["a"], pathlib.Path("/etc/passwd")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.store.path_for(bad)

    def test_public_methods_reject_hostile_ids(self):
        msg = Message(role="user", content="hi")
        for bad in ("../escape", "/etc/passwd", "a\x00b", ".."):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.store.load(bad)
                with self.assertRaises(ValueError):
                    self.store.append(bad, msg)
                with self.assertRaises(ValueError):
                    self.store.delete(bad)
                with self.assertRaises(ValueError):
                    self.store.update(bad, title="x")
                with self.assertRaises(ValueError):
                    self.store.export(bad)

    def test_nothing_escaped_the_root(self):
        outside = pathlib.Path(self.tmp.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        for bad in self.HOSTILE:
            with self.assertRaises(ValueError):
                self.store.path_for(bad)
        self.assertEqual(outside.read_text(encoding="utf-8"), "secret")

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_symlinked_session_file_is_refused(self):
        target = pathlib.Path(self.tmp.name) / "target.jsonl"
        target.write_bytes(b'{"type":"meta","id":"evil"}\n')
        link = self.store.sessions_dir / "evil.jsonl"
        try:
            os.symlink(str(target), str(link))
        except (OSError, NotImplementedError):  # pragma: no cover
            self.skipTest("symlinks unavailable")
        with self.assertRaises(ValueError):
            self.store.path_for("evil")
        with self.assertRaises(ValueError):
            self.store.load("evil")
        with self.assertRaises(ValueError):
            self.store.append("evil", Message(role="user", content="x"))
        self.assertFalse(self.store.exists("evil"))
        # ... and it never shows up in a listing.
        self.assertEqual([m.id for m in self.store.list()], [])
        self.assertEqual(target.read_bytes(), b'{"type":"meta","id":"evil"}\n')

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_symlinked_directory_escape_is_refused(self):
        elsewhere = pathlib.Path(self.tmp.name) / "elsewhere"
        elsewhere.mkdir()
        link = self.store.sessions_dir / "away.jsonl"
        os.symlink(str(elsewhere), str(link))
        with self.assertRaises(ValueError):
            self.store.path_for("away")


# -------------------------------------------------------------------- round trip


class TestRoundTrip(Base):
    def test_create_then_load_empty(self):
        meta = self.make(title="Hello", system="be nice")
        loaded, messages = self.store.load(meta.id)
        self.assertEqual(messages, [])
        self.assertEqual(loaded.id, meta.id)
        self.assertEqual(loaded.title, "Hello")
        self.assertEqual(loaded.system, "be nice")
        self.assertEqual(loaded.model, "claude-opus-5")
        self.assertEqual(loaded.message_count, 0)

    def test_full_message_round_trip(self):
        meta = self.make(title="t")
        sent = [
            Message(role="user", content="hello 🌍\nsecond line\twith tab"),
            Message(role="assistant", content="hi", model="claude-opus-5",
                    usage={"input_tokens": 10, "output_tokens": 3},
                    thinking="pondering…"),
            Message(role="system", content="note"),
            Message(role="assistant", content="", error="overloaded"),
        ]
        for m in sent:
            self.store.append(meta.id, m)
        loaded, got = self.store.load(meta.id)
        self.assertEqual(len(got), 4)
        for a, b in zip(sent, got):
            self.assertEqual((a.role, a.content, a.id), (b.role, b.content, b.id))
            self.assertAlmostEqual(a.ts, b.ts, places=6)
            self.assertEqual(a.model, b.model)
            self.assertEqual(a.usage, b.usage)
            self.assertEqual(a.thinking, b.thinking)
            self.assertEqual(a.error, b.error)
        self.assertEqual(loaded.message_count, 4)

    def test_pathological_content_survives(self):
        meta = self.make(title="t")
        bodies = [
            '{"type":"meta","id":"spoof"}',          # a record inside a record
            "line\nbreak\r\nand \\ backslash",
            "\x00\x01\x02 control bytes",
            "\ud83d",                                 # lone surrogate
            "🧬" * 500,
            "]" * 100 + "[",
        ]
        for b in bodies:
            self.store.append(meta.id, Message(role="user", content=b))
        _, got = self.store.load(meta.id)
        self.assertEqual([m.content for m in got], bodies)

    def test_one_line_per_message(self):
        meta = self.make(title="t")
        for i in range(5):
            self.store.append(meta.id, Message(role="user", content="a\nb\nc %d" % i))
        lines = [l for l in self.raw(meta.id).split(b"\n") if l.strip()]
        self.assertEqual(len(lines), 6)  # header + 5
        for line in lines:
            json.loads(line.decode("utf-8"))

    def test_two_stores_share_state(self):
        meta = self.make(title="t")
        other = Store(self.root)
        other.append(meta.id, Message(role="user", content="from elsewhere"))
        _, got = self.store.load(meta.id)
        self.assertEqual(got[0].content, "from elsewhere")
        self.assertEqual([m.id for m in other.list()], [meta.id])

    def test_permissions(self):
        meta = self.make(title="t")
        if os.name == "nt":  # pragma: no cover
            self.skipTest("POSIX modes only")
        self.assertEqual(stat.S_IMODE(os.stat(self.path(meta.id)).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.store.root).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(self.store.sessions_dir).st_mode), 0o700)
        self.store.list()
        self.assertEqual(stat.S_IMODE(os.stat(self.store.index_path).st_mode), 0o600)

    def test_append_to_unknown_session(self):
        with self.assertRaises(KeyError):
            self.store.append("aaaaaaaaaaaaaaaaaaaaaaaaaa", Message(role="user", content="x"))

    def test_load_unknown_session(self):
        with self.assertRaises(KeyError):
            self.store.load("aaaaaaaaaaaaaaaaaaaaaaaaaa")

    def test_message_with_junk_field_types_still_writes(self):
        meta = self.make(title="t")
        m = Message(role="user", content="ok")
        m.ts = "not a number"
        m.id = None
        self.store.append(meta.id, m)
        _, got = self.store.load(meta.id)
        self.assertEqual(got[0].content, "ok")
        self.assertTrue(got[0].id)

    def test_append_type_check(self):
        meta = self.make()
        with self.assertRaises(TypeError):
            self.store.append(meta.id, {"role": "user", "content": "x"})

    def test_no_temp_files_left_behind(self):
        meta = self.make(title="t")
        for i in range(3):
            self.store.append(meta.id, Message(role="user", content=str(i)))
        self.store.rename(meta.id, "new")
        self.store.record_usage(meta.id, {"input_tokens": 1}, 0.1)
        self.store.list()
        leftovers = [n for n in os.listdir(self.store.sessions_dir) if n.startswith(".tmp-")]
        self.assertEqual(leftovers, [])


# -------------------------------------------------------------------- corruption


class TestCorruption(Base):
    def _seed(self, n=3):
        meta = self.make(title="seed")
        for i in range(n):
            self.store.append(meta.id, Message(role="user", content="msg %d" % i))
        return meta

    def test_truncated_trailing_line_is_skipped(self):
        meta = self._seed()
        data = self.raw(meta.id)
        self.write_raw(meta.id, data + b'{"type":"message","role":"user","cont')
        with self.assertWarns(StoreWarning):
            loaded, got = self.store.load(meta.id)
        self.assertEqual([m.content for m in got], ["msg 0", "msg 1", "msg 2"])
        self.assertEqual(loaded.title, "seed")

    def test_mid_write_kill_loses_only_the_last_record(self):
        meta = self._seed()
        data = self.raw(meta.id)
        self.write_raw(meta.id, data[:-12])          # a write cut off mid-flight
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            _, got = self.store.load(meta.id)
        self.assertEqual([m.content for m in got], ["msg 0", "msg 1"])

    def test_append_repairs_a_torn_tail(self):
        meta = self._seed()
        self.write_raw(meta.id, self.raw(meta.id) + b'{"type":"messa')
        with self.assertWarns(StoreWarning):
            self.store.append(meta.id, Message(role="user", content="after crash"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            _, got = self.store.load(meta.id)
        self.assertEqual([m.content for m in got], ["msg 0", "msg 1", "msg 2", "after crash"])
        # The repair is idempotent: a further append needs no repair.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.store.append(meta.id, Message(role="user", content="later"))
        self.assertEqual([w for w in caught if issubclass(w.category, StoreWarning)], [])

    def test_garbage_in_the_middle(self):
        meta = self._seed()
        lines = self.raw(meta.id).split(b"\n")
        lines.insert(2, b"\x00\x01 not json at all \xff\xfe")
        self.write_raw(meta.id, b"\n".join(lines))
        with self.assertWarns(StoreWarning):
            _, got = self.store.load(meta.id)
        self.assertEqual([m.content for m in got], ["msg 0", "msg 1", "msg 2"])

    def test_valid_json_but_wrong_shape_is_skipped(self):
        meta = self._seed()
        junk = [b'[1,2,3]', b'"a string"', b'null', b'{"type":"message"}',
                b'{"type":"message","role":5,"content":"x"}', b'{"nope":1}']
        self.write_raw(meta.id, self.raw(meta.id) + b"\n".join(junk) + b"\n")
        with self.assertWarns(StoreWarning):
            _, got = self.store.load(meta.id)
        self.assertEqual(len(got), 3)

    def test_lost_header_is_reconstructed(self):
        meta = self._seed()
        lines = self.raw(meta.id).split(b"\n")
        lines[0] = b"{{{ not json"
        self.write_raw(meta.id, b"\n".join(lines))
        with self.assertWarns(StoreWarning):
            loaded, got = self.store.load(meta.id)
        self.assertEqual(len(got), 3)
        self.assertEqual(loaded.id, meta.id)
        self.assertEqual(loaded.title, "msg 0")   # re-derived from the first user message
        self.assertGreater(loaded.created, 0)
        # and it still lists.
        self.assertEqual([m.id for m in self.store.list()], [meta.id])

    def test_empty_file(self):
        meta = self.make(title="t")
        self.write_raw(meta.id, b"")
        with self.assertWarns(StoreWarning):
            loaded, got = self.store.load(meta.id)
        self.assertEqual(got, [])
        self.assertEqual(loaded.id, meta.id)
        self.assertEqual(self.store.list()[0].id, meta.id)

    def test_all_binary_file(self):
        meta = self.make(title="t")
        self.write_raw(meta.id, os.urandom(4096))
        with self.assertWarns(StoreWarning):
            loaded, got = self.store.load(meta.id)
        self.assertEqual(got, [])
        self.assertEqual(loaded.id, meta.id)

    def test_no_trailing_newline_on_last_good_line(self):
        meta = self._seed()
        self.write_raw(meta.id, self.raw(meta.id).rstrip(b"\n"))
        _, got = self.store.load(meta.id)          # still parses; no warning needed
        self.assertEqual(len(got), 3)

    def test_blank_lines_are_ignored(self):
        meta = self._seed()
        self.write_raw(meta.id, self.raw(meta.id).replace(b"\n", b"\n\n"))
        _, got = self.store.load(meta.id)
        self.assertEqual(len(got), 3)

    def test_header_rewrite_never_loses_messages(self):
        meta = self._seed(n=25)
        before = self.raw(meta.id).split(b"\n")[1:]
        self.store.rename(meta.id, "renamed")
        after = self.raw(meta.id).split(b"\n")[1:]
        self.assertEqual(before, after)            # messages byte-for-byte identical
        loaded, got = self.store.load(meta.id)
        self.assertEqual(loaded.title, "renamed")
        self.assertEqual(len(got), 25)

    def test_interrupted_header_rewrite_leaves_the_old_file(self):
        meta = self._seed()
        original = self.raw(meta.id)

        boom = OSError("disk full")
        with mock.patch.object(store_mod, "_write_all", side_effect=boom):
            with self.assertRaises(OSError):
                self.store.rename(meta.id, "never lands")
        self.assertEqual(self.raw(meta.id), original)
        self.assertEqual(
            [n for n in os.listdir(self.store.sessions_dir) if n.startswith(".tmp-")], [])
        _, got = self.store.load(meta.id)
        self.assertEqual(len(got), 3)

    def test_crash_between_write_and_replace(self):
        meta = self._seed()
        original = self.raw(meta.id)
        with mock.patch.object(store_mod.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.store.rename(meta.id, "nope")
        self.assertEqual(self.raw(meta.id), original)
        self.assertEqual(
            [n for n in os.listdir(self.store.sessions_dir) if n.startswith(".tmp-")], [])


# --------------------------------------------------------------------- resolution


class TestResolve(Base):
    def setUp(self):
        super().setUp()
        ids = iter(["aaaa11111111", "aaaa22222222", "bbbb33333333"])
        with mock.patch.object(store_mod, "new_id", lambda: next(ids)):
            self.a1 = self.make(title="alpha one")
            self.a2 = self.make(title="alpha two")
            self.b3 = self.make(title="beta")

    def test_exact_id(self):
        self.assertEqual(self.store.resolve("aaaa11111111"), "aaaa11111111")

    def test_unique_prefix(self):
        self.assertEqual(self.store.resolve("aaaa1"), "aaaa11111111")
        self.assertEqual(self.store.resolve("b"), "bbbb33333333")

    def test_prefix_is_case_insensitive(self):
        self.assertEqual(self.store.resolve("AAAA2"), "aaaa22222222")

    def test_ambiguous_prefix(self):
        with self.assertRaises(AmbiguousRefError) as ctx:
            self.store.resolve("aaaa")
        self.assertNotIsInstance(ctx.exception, KeyError)
        self.assertIsInstance(ctx.exception, LookupError)
        self.assertIn("aaaa11111111", str(ctx.exception))

    def test_unknown_ref(self):
        for bad in ("zzz", "nope", "  ", "", "../x", "\x00"):
            with self.subTest(bad=bad):
                with self.assertRaises(KeyError):
                    self.store.resolve(bad)

    def test_non_string_ref(self):
        with self.assertRaises(KeyError):
            self.store.resolve(None)

    def test_index_reference(self):
        rows = self.store.list()
        for i, row in enumerate(rows, 1):
            self.assertEqual(self.store.resolve(str(i)), row.id)
            self.assertEqual(self.store.resolve("#%d" % i), row.id)

    def test_out_of_range_index(self):
        with self.assertRaises(KeyError):
            self.store.resolve("99")

    def test_last(self):
        self.store.update(self.a1.id, created=time.time() + 500)
        self.assertEqual(self.store.resolve("last"), self.a1.id)
        self.assertEqual(self.store.resolve("LAST"), self.a1.id)
        self.assertEqual(self.store.latest().id, self.a1.id)

    def test_last_with_no_sessions(self):
        empty = Store(pathlib.Path(self.tmp.name) / "empty")
        self.assertIsNone(empty.latest())
        with self.assertRaises(KeyError):
            empty.resolve("last")

    def test_exact_id_beats_index_interpretation(self):
        ids = iter(["1abcdefgh"])
        with mock.patch.object(store_mod, "new_id", lambda: next(ids)):
            numeric = self.make(title="numeric-ish id")
        self.assertEqual(self.store.resolve("1abcdefgh"), numeric.id)

    def test_deleted_session_stops_resolving(self):
        self.store.delete(self.b3.id)
        with self.assertRaises(KeyError):
            self.store.resolve(self.b3.id)


# -------------------------------------------------------------------- list/query


class TestListing(Base):
    def setUp(self):
        super().setUp()
        base = time.time()
        self.s1 = self.make(title="Python questions")
        self.s2 = self.make(title="Rust borrow checker")
        self.s3 = self.make(title="Dinner ideas")
        # `updated` is derived, so ordering is staged through `created`, which
        # is stored: _meta_from() takes updated = max(stored, created, last ts).
        self.store.update(self.s1.id, created=base + 10)
        self.store.update(self.s2.id, created=base + 20)
        self.store.update(self.s3.id, created=base + 30)
        self.store.append(self.s1.id, Message(role="user", content="How do I sort a DICT?"))
        self.store.append(self.s2.id, Message(role="assistant", content="lifetimes are fine"))

    def test_newest_first(self):
        self.assertEqual([m.id for m in self.store.list()], [self.s3.id, self.s2.id, self.s1.id])

    def test_pinned_first(self):
        self.store.update(self.s1.id, pinned=True)
        rows = self.store.list()
        self.assertEqual(rows[0].id, self.s1.id)
        self.assertTrue(rows[0].pinned)
        self.assertEqual([m.id for m in rows[1:]], [self.s3.id, self.s2.id])

    def test_limit(self):
        self.assertEqual(len(self.store.list(limit=2)), 2)
        self.assertEqual(self.store.list(limit=0), [])
        self.assertEqual(len(self.store.list(limit=None)), 3)
        self.assertEqual(self.store.list(limit=-3), [])

    def test_query_matches_title_case_insensitively(self):
        self.assertEqual([m.id for m in self.store.list(query="rUsT")], [self.s2.id])

    def test_query_matches_message_text(self):
        self.assertEqual([m.id for m in self.store.list(query="sort a dict")], [self.s1.id])
        self.assertEqual([m.id for m in self.store.list(query="lifetimes")], [self.s2.id])

    def test_query_no_match(self):
        self.assertEqual(self.store.list(query="quantum knitting"), [])

    def test_counts_and_timestamps(self):
        rows = {m.id: m for m in self.store.list()}
        self.assertEqual(rows[self.s1.id].message_count, 1)
        self.assertEqual(rows[self.s3.id].message_count, 0)
        self.assertGreaterEqual(rows[self.s2.id].updated, rows[self.s1.id].updated)

    def test_empty_store(self):
        empty = Store(pathlib.Path(self.tmp.name) / "nothing")
        self.assertEqual(empty.list(), [])
        self.assertIsNone(empty.latest())

    def test_stray_files_are_ignored(self):
        (self.store.sessions_dir / "notes.txt").write_text("hi", encoding="utf-8")
        (self.store.sessions_dir / "..jsonl").write_text("hi", encoding="utf-8")
        (self.store.sessions_dir / ".tmp-x.jsonl").write_text("hi", encoding="utf-8")
        self.assertEqual(len(self.store.list()), 3)


# -------------------------------------------------------------------- index cache


class TestIndexCache(Base):
    def setUp(self):
        super().setUp()
        self.metas = [self.make(title="session %d" % i) for i in range(4)]
        for i, m in enumerate(self.metas):
            for j in range(i + 1):
                self.store.append(m.id, Message(role="user", content="m%d-%d" % (i, j)))

    def counts(self, st=None):
        st = st or self.store
        return {m.id: m.message_count for m in st.list()}

    def expected(self):
        return {m.id: i + 1 for i, m in enumerate(self.metas)}

    def test_index_is_written_and_reused(self):
        self.assertEqual(self.counts(), self.expected())
        self.assertTrue(self.store.index_path.is_file())
        doc = json.loads(self.store.index_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], 1)
        self.assertEqual(set(doc["entries"]), {m.id for m in self.metas})
        self.assertEqual(self.counts(Store(self.root)), self.expected())

    def test_missing_index(self):
        self.store.list()
        os.unlink(self.store.index_path)
        self.assertEqual(self.counts(Store(self.root)), self.expected())
        self.assertTrue(self.store.index_path.is_file())

    def test_corrupt_index_json(self):
        for junk in (b"", b"not json", b"[]", b"null", b'{"version":99}',
                     b'{"version":1,"entries":"nope"}', os.urandom(512)):
            with self.subTest(junk=junk[:12]):
                self.store.index_path.write_bytes(junk)
                self.assertEqual(self.counts(Store(self.root)), self.expected())

    def test_index_entry_for_a_deleted_session_does_not_resurrect_it(self):
        self.store.list()
        victim = self.metas[0]
        os.unlink(self.path(victim.id))
        rows = self.counts(Store(self.root))
        self.assertNotIn(victim.id, rows)
        self.assertEqual(len(rows), 3)

    def test_fabricated_index_entries_are_ignored(self):
        doc = {"version": 1, "entries": {
            "ghostsession": {"size": 10, "mtime_ns": 1, "count": 7, "updated": 9e9},
            self.metas[0].id: {"size": 999999, "mtime_ns": 1, "count": 4242,
                               "updated": 9e9},
        }}
        self.store.index_path.write_text(json.dumps(doc), encoding="utf-8")
        rows = {m.id: m for m in Store(self.root).list()}
        self.assertNotIn("ghostsession", rows)
        self.assertEqual({k: v.message_count for k, v in rows.items()}, self.expected())

    def test_impossible_count_with_matching_stat_is_distrusted(self):
        self.store.list()
        st = os.stat(self.path(self.metas[0].id))
        doc = json.loads(self.store.index_path.read_text(encoding="utf-8"))
        doc["entries"][self.metas[0].id] = {
            "size": st.st_size, "mtime_ns": st.st_mtime_ns, "count": 99999, "updated": 0,
        }
        self.store.index_path.write_text(json.dumps(doc), encoding="utf-8")
        rows = {m.id: m.message_count for m in Store(self.root).list()}
        self.assertEqual(rows, self.expected())

    def test_header_fields_are_always_read_fresh(self):
        self.store.list()                       # populate the cache
        self.store.rename(self.metas[1].id, "renamed out of band")
        self.store.update(self.metas[1].id, pinned=True, tags=["x"])
        rows = {m.id: m for m in Store(self.root).list()}
        self.assertEqual(rows[self.metas[1].id].title, "renamed out of band")
        self.assertTrue(rows[self.metas[1].id].pinned)
        self.assertEqual(rows[self.metas[1].id].tags, ["x"])

    def test_stale_after_an_append_from_another_store(self):
        self.store.list()                       # cache says session 0 has one message
        Store(self.root).append(self.metas[0].id, Message(role="user", content="extra"))
        self.assertEqual(self.counts(Store(self.root))[self.metas[0].id], 2)

    def test_index_is_rewritten_after_being_corrupted(self):
        self.store.index_path.write_bytes(b"garbage")
        Store(self.root).list()
        doc = json.loads(self.store.index_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], 1)
        self.assertEqual(set(doc["entries"]), {m.id for m in self.metas})

    def test_unwritable_index_is_not_fatal(self):
        if os.name == "nt" or os.geteuid() == 0:  # pragma: no cover
            self.skipTest("cannot make a directory unwritable here")
        self.store.list()
        os.chmod(self.store.sessions_dir, 0o500)
        self.addCleanup(os.chmod, self.store.sessions_dir, 0o700)
        self.assertEqual(self.counts(Store(self.root)), self.expected())

    def test_orphaned_temp_files_are_swept_when_old(self):
        recent = self.store.sessions_dir / ".tmp-recent.jsonl"
        stale = self.store.sessions_dir / ".tmp-stale.jsonl"
        recent.write_bytes(b"half written")
        stale.write_bytes(b"half written")
        old_time = time.time() - 7200
        os.utime(stale, (old_time, old_time))
        self.assertEqual(self.counts(), self.expected())
        self.assertFalse(stale.exists())
        self.assertTrue(recent.exists())      # may belong to a live writer

    def test_index_is_never_mistaken_for_a_session(self):
        self.store.list()
        self.assertNotIn("index", {m.id for m in self.store.list()})


# ------------------------------------------------------------------ meta updates


class TestUpdates(Base):
    def test_auto_title_from_first_user_message(self):
        meta = self.make()
        self.assertEqual(meta.title, "")
        self.store.append(meta.id, Message(role="system", content="you are helpful"))
        self.store.append(meta.id, Message(role="user", content="  ###  Why is the sky blue?\nreally  "))
        self.store.append(meta.id, Message(role="user", content="second question"))
        loaded, _ = self.store.load(meta.id)
        self.assertEqual(loaded.title, "Why is the sky blue? really")
        header = json.loads(self.raw(meta.id).split(b"\n")[0].decode("utf-8"))
        self.assertEqual(header["title"], "Why is the sky blue? really")

    def test_auto_title_handles_hostile_first_messages(self):
        for body in ("🚀" * 300, "\n\n\n", "x" * 5000, "\x00\x01"):
            meta = self.make()
            self.store.append(meta.id, Message(role="user", content=body))
            loaded, _ = self.store.load(meta.id)
            self.assertLessEqual(display_width(loaded.title), 48)
            self.assertNotIn("\n", loaded.title)

    def test_explicit_title_is_not_overwritten(self):
        meta = self.make(title="Keep me")
        self.store.append(meta.id, Message(role="user", content="something else"))
        loaded, _ = self.store.load(meta.id)
        self.assertEqual(loaded.title, "Keep me")

    def test_rename(self):
        meta = self.make(title="old")
        self.store.append(meta.id, Message(role="user", content="body"))
        out = self.store.rename(meta.id, "  New \n title  ")
        self.assertEqual(out.title, "New title")
        self.assertEqual(self.store.load(meta.id)[0].title, "New title")

    def test_rename_clips_long_titles(self):
        meta = self.make(title="x")
        out = self.store.rename(meta.id, "🌟" * 100)
        self.assertLessEqual(display_width(out.title), 48)

    def test_update_fields(self):
        meta = self.make(title="t")
        out = self.store.update(meta.id, model="claude-sonnet-5", system="sys",
                                pinned=True, tags=["a", "b"], cost_usd=1.5)
        self.assertEqual(out.model, "claude-sonnet-5")
        self.assertEqual(out.system, "sys")
        self.assertTrue(out.pinned)
        self.assertEqual(out.tags, ["a", "b"])
        again = self.store.load(meta.id)[0]
        self.assertEqual((again.model, again.system, again.pinned, again.tags, again.cost_usd),
                         ("claude-sonnet-5", "sys", True, ["a", "b"], 1.5))

    def test_update_rejects_unknown_and_derived_fields(self):
        meta = self.make(title="t")
        for bad in ({"nonsense": 1}, {"message_count": 99}, {"id": "other"}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.store.update(meta.id, **bad)

    def test_update_unknown_session(self):
        with self.assertRaises(KeyError):
            self.store.update("aaaaaaaaaaaaaaaaaaaaaaaaaa", title="x")

    def test_rename_does_not_reorder_the_list(self):
        base = time.time()
        a = self.make(title="a")
        b = self.make(title="b")
        self.store.update(a.id, created=base + 1)
        self.store.update(b.id, created=base + 2)
        self.assertEqual([m.id for m in self.store.list()], [b.id, a.id])
        self.store.rename(a.id, "a renamed")
        self.assertEqual([m.id for m in self.store.list()], [b.id, a.id])

    def test_record_usage_accumulates(self):
        meta = self.make(title="t")
        self.store.record_usage(meta.id, {"input_tokens": 100, "output_tokens": 20,
                                          "cache_creation_input_tokens": 7,
                                          "cache_read_input_tokens": 3}, 0.01)
        out = self.store.record_usage(meta.id, {"input_tokens": 5, "output_tokens": 1}, 0.02)
        self.assertEqual((out.input_tokens, out.output_tokens), (105, 21))
        self.assertEqual((out.cache_write_tokens, out.cache_read_tokens), (7, 3))
        self.assertAlmostEqual(out.cost_usd, 0.03)
        self.assertAlmostEqual(self.store.load(meta.id)[0].cost_usd, 0.03)

    def test_record_usage_ignores_junk(self):
        meta = self.make(title="t")
        out = self.store.record_usage(meta.id, {"input_tokens": "lots", "output_tokens": None},
                                      float("nan"))
        self.assertEqual((out.input_tokens, out.output_tokens, out.cost_usd), (0, 0, 0.0))
        out = self.store.record_usage(meta.id, None, -5)
        self.assertEqual(out.cost_usd, 0.0)

    def test_record_usage_preserves_messages(self):
        meta = self.make(title="t")
        for i in range(4):
            self.store.append(meta.id, Message(role="user", content=str(i)))
        self.store.record_usage(meta.id, {"input_tokens": 1}, 0.5)
        _, got = self.store.load(meta.id)
        self.assertEqual([m.content for m in got], ["0", "1", "2", "3"])

    def test_delete(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="x"))
        self.store.delete(meta.id)
        self.assertFalse(self.path(meta.id).exists())
        self.assertEqual(self.store.list(), [])
        with self.assertRaises(KeyError):
            self.store.load(meta.id)
        with self.assertRaises(KeyError):
            self.store.delete(meta.id)

    def test_delete_removes_the_lock_sidecar(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="x"))
        self.store.delete(meta.id)
        leftovers = [n for n in os.listdir(self.store.sessions_dir) if n.startswith(meta.id)]
        self.assertEqual(leftovers, [])


# ------------------------------------------------------------------------ export


class TestExport(Base):
    def setUp(self):
        super().setUp()
        self.meta = self.make(title="Exportable", system="be brief")
        self.store.append(self.meta.id, Message(role="user", content="hello ✨"))
        self.store.append(self.meta.id, Message(role="assistant", content="# hi\n\nthere",
                                                thinking="hmm", model="claude-opus-5"))
        self.store.record_usage(self.meta.id, {"input_tokens": 9, "output_tokens": 4}, 0.001)

    def test_markdown(self):
        out = self.store.export(self.meta.id)
        self.assertIn("# Exportable", out)
        self.assertIn("hello ✨", out)
        self.assertIn("there", out)
        self.assertIn("be brief", out)
        self.assertIn("You", out)
        self.assertIn("Claude", out)
        self.assertTrue(out.endswith("\n"))

    def test_json(self):
        doc = json.loads(self.store.export(self.meta.id, "json"))
        self.assertEqual(doc["meta"]["title"], "Exportable")
        self.assertEqual(len(doc["messages"]), 2)
        self.assertEqual(doc["messages"][0]["content"], "hello ✨")

    def test_text(self):
        out = self.store.export(self.meta.id, "text")
        self.assertIn("Exportable", out)
        self.assertIn("hello ✨", out)
        self.assertNotIn("**", out)

    def test_bad_format(self):
        with self.assertRaises(ValueError):
            self.store.export(self.meta.id, "pdf")

    def test_export_unknown_session(self):
        with self.assertRaises(KeyError):
            self.store.export("aaaaaaaaaaaaaaaaaaaaaaaaaa")

    def test_a_session_that_vanishes_mid_export_is_a_missing_session(self):
        # export() streams the messages in a second pass, so the file can go
        # away between the two. A caller of a session API expects a KeyError,
        # not a stray FileNotFoundError from inside a generator.
        real = store_mod.Store._iter_messages

        def vanish(path):
            os.unlink(str(path))
            return real(path)

        with mock.patch.object(Store, "_iter_messages", staticmethod(vanish)):
            with self.assertRaises(KeyError):
                self.store.export(self.meta.id)

    def test_export_of_damaged_session_still_works(self):
        self.write_raw(self.meta.id, self.raw(self.meta.id) + b'{"broken')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            out = self.store.export(self.meta.id)
        self.assertIn("hello ✨", out)


# ------------------------------------------------------- cost of a single append


class _PassCounter:
    """How many times something walked the whole transcript."""

    def __init__(self):
        self.count = 0


@contextlib.contextmanager
def counting_full_passes():
    """Count every full parse of a session file, however it is spelled.

    Both entry points are patched: a fix that swaps one for the other is not a
    fix, and a test that watches only one of them cannot tell the difference.
    """
    counter = _PassCounter()
    real = {name: getattr(store_mod, name) for name in ("_read_records", "_scan_meta")}

    def wrap(func):
        def counted(*a, **kw):
            counter.count += 1
            return func(*a, **kw)
        return counted

    with contextlib.ExitStack() as stack:
        for name, func in real.items():
            stack.enter_context(mock.patch.object(store_mod, name, wrap(func)))
        yield counter


class TestAppendCost(Base):
    """An append to an append-only log must not depend on the log's length.

    ``append()`` used to re-read and ``json.loads()`` the whole session on every
    user turn just to ask whether the header already had a title, which made a
    conversation quadratic (13.7s for 4000 appends; 0.17s once fixed).
    """

    def titled(self, n=10):
        meta = self.make()
        self.store.append(meta.id, Message(role="user", content="a real question"))
        for i in range(n):
            self.store.append(meta.id, Message(role="assistant", content="x" * 2000))
        return meta

    def test_append_never_parses_the_transcript(self):
        meta = self.titled()
        with counting_full_passes() as passes:
            for i in range(20):
                self.store.append(meta.id, Message(role="user", content="q%d" % i))
                self.store.append(meta.id, Message(role="assistant", content="a%d" % i))
        self.assertEqual(passes.count, 0,
                         "append() parsed the transcript %d time(s)" % passes.count)

    def test_first_user_message_parses_it_exactly_once(self):
        meta = self.make()
        with counting_full_passes() as passes:
            self.store.append(meta.id, Message(role="user", content="the question"))
            self.store.append(meta.id, Message(role="user", content="another"))
        self.assertEqual(passes.count, 1)   # once, to persist the derived title
        self.assertEqual(self.store.load(meta.id)[0].title, "the question")

    def test_bytes_read_per_append_do_not_grow_with_the_file(self):
        meta = self.titled(n=2)
        small = self.bytes_read_by_one_append(meta.id)
        for i in range(2000):                    # ~4 MB of transcript
            self.store.append(meta.id, Message(role="assistant", content="y" * 2000))
        size = self.path(meta.id).stat().st_size
        self.assertGreater(size, 4_000_000)
        big = self.bytes_read_by_one_append(meta.id)
        self.assertLess(big, 128 * 1024,
                        "an append read %d bytes of a %d byte file" % (big, size))
        self.assertLess(big, size // 4)
        self.assertLess(abs(big - small), 128 * 1024)

    def bytes_read_by_one_append(self, sid):
        total = [0]
        real = os.read

        def counting_read(fd, n):
            data = real(fd, n)
            total[0] += len(data)
            return data

        with mock.patch("os.read", counting_read):
            self.store.append(sid, Message(role="user", content="measure me"))
        return total[0]


# ------------------------------------------------------------------ lock failure


class FailingLock:
    """Stand-in for a filesystem where ``flock()`` itself fails (NFS, some FUSE)."""

    def __init__(self, err=errno.ENOLCK):
        self.err = err

    def __call__(self, fd):
        raise OSError(self.err, os.strerror(self.err))


class TestLockFailure(Base):
    def open_fds(self):
        return len(os.listdir("/proc/self/fd"))

    @unittest.skipUnless(os.path.isdir("/proc/self/fd"), "needs /proc")
    def test_a_failing_flock_does_not_leak_a_descriptor_per_operation(self):
        meta = self.make(title="t")
        with mock.patch.object(store_mod, "_lock_acquire", FailingLock()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", StoreWarning)
                self.store.append(meta.id, Message(role="user", content="warm up"))
                before = self.open_fds()
                for i in range(200):
                    self.store.append(meta.id, Message(role="user", content=str(i)))
                after = self.open_fds()
        self.assertLessEqual(after - before, 2,
                             "leaked %d descriptors over 200 appends" % (after - before))
        self.assertEqual(len(self.quiet_load(meta.id)[1]), 201)

    def test_a_failing_lock_says_so_instead_of_pretending(self):
        meta = self.make(title="t")
        with mock.patch.object(store_mod, "_lock_acquire", FailingLock()):
            with self.assertWarns(StoreWarning) as caught:
                self.store.append(meta.id, Message(role="user", content="x"))
        self.assertIn("could not lock", str(caught.warning))

    def test_the_lock_table_does_not_grow_without_bound(self):
        metas = [self.make(title="t%d" % i) for i in range(20)]
        for m in metas:
            self.store.append(m.id, Message(role="user", content="x"))
            self.store.rename(m.id, "renamed")
        self.assertEqual(store_mod._locks, {},
                         "the process-global lock table kept %d entries"
                         % len(store_mod._locks))

    def test_the_session_lock_is_not_re_entrant(self):
        # flock() is per open file description, so a nested acquire opens a second
        # fd and blocks on itself forever. The nesting happens in a worker thread
        # so that the deadlock this guards against shows up as a failed assertion
        # rather than as a hung test run.
        import threading
        meta = self.make(title="t")
        path = self.store.path_for(meta.id)
        outcome = []

        def nest():
            with store_mod._file_lock(path):
                try:
                    with store_mod._file_lock(path):
                        outcome.append("granted twice")
                except RuntimeError:
                    outcome.append("refused")

        worker = threading.Thread(target=nest, daemon=True)
        worker.start()
        worker.join(30)
        self.assertFalse(worker.is_alive(), "a nested acquire deadlocked on itself")
        self.assertEqual(outcome, ["refused"])
        self.assertEqual(store_mod._locks, {})
        self.store.append(meta.id, Message(role="user", content="still works"))


class TestGhostSessions(Base):
    """Operations on an id that does not exist must not leave anything behind."""

    MISSING = "aaaaaaaaaaaaaaaaaaaaaaaaaa"

    def attempts(self):
        msg = Message(role="user", content="x")
        for op in (lambda: self.store.append(self.MISSING, msg),
                   lambda: self.store.update(self.MISSING, title="x"),
                   lambda: self.store.rename(self.MISSING, "x"),
                   lambda: self.store.record_usage(self.MISSING, {"input_tokens": 1}, 0.1),
                   lambda: self.store.delete(self.MISSING),
                   lambda: self.store.load(self.MISSING)):
            with self.assertRaises(KeyError):
                op()

    def test_no_lock_sidecar_is_created_for_a_session_that_does_not_exist(self):
        for _ in range(50):
            self.attempts()
        leftovers = sorted(os.listdir(self.store.sessions_dir))
        self.assertEqual(leftovers, [], "ghost ids left files behind: %r" % leftovers)
        self.assertEqual(store_mod._locks, {})


# ------------------------------------------------------------- damaged headers


class TestDamagedHeaderIsNotOverwritten(Base):
    def damaged(self):
        meta = self.make(title="Important thread", system="a long system prompt")
        self.store.append(meta.id, Message(role="user", content="body"))
        self.store.record_usage(meta.id, {"input_tokens": 900000, "output_tokens": 5}, 12.34)
        original = self.raw(meta.id)
        lines = original.split(b"\n")
        lines[0] = lines[0][:40] + b" <-- damaged"      # header no longer parses
        self.write_raw(meta.id, b"\n".join(lines))
        return meta, self.raw(meta.id)

    def test_a_rename_keeps_the_bytes_it_cannot_read(self):
        meta, before = self.damaged()
        with self.assertWarns(StoreWarning) as caught:
            self.store.rename(meta.id, "same title")
        self.assertIn("no readable meta header", str(caught.warning))
        backup = pathlib.Path(str(self.path(meta.id)) + ".bak")
        self.assertTrue(backup.is_file(), "the only copy of the header was destroyed")
        self.assertEqual(backup.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)

    def test_the_first_rescue_copy_is_never_replaced_by_a_later_one(self):
        meta, before = self.damaged()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            self.store.rename(meta.id, "one")
            self.store.rename(meta.id, "two")       # header is readable again now
            self.write_raw(meta.id, b"{ broken again\n")
            self.store.rename(meta.id, "three")
        backup = pathlib.Path(str(self.path(meta.id)) + ".bak")
        self.assertEqual(backup.read_bytes(), before)

    def test_an_empty_session_is_not_worth_rescuing(self):
        meta = self.make(title="t")
        self.write_raw(meta.id, b"")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.store.rename(meta.id, "fresh start")
        self.assertFalse(pathlib.Path(str(self.path(meta.id)) + ".bak").exists())
        self.assertEqual([w for w in caught if issubclass(w.category, StoreWarning)], [])
        self.assertEqual(self.store.load(meta.id)[0].title, "fresh start")

    def test_delete_takes_the_rescue_copy_with_it(self):
        meta, _ = self.damaged()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            self.store.rename(meta.id, "x")
        self.assertTrue(pathlib.Path(str(self.path(meta.id)) + ".bak").exists())
        self.store.delete(meta.id)
        self.assertEqual(sorted(os.listdir(self.store.sessions_dir)), [])

    def test_the_rescue_copy_is_not_a_session(self):
        meta, _ = self.damaged()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            self.store.rename(meta.id, "x")
            self.assertEqual([m.id for m in self.store.list()], [meta.id])


# ----------------------------------------------------------- unusable file modes


class TestUnusableSessionFiles(Base):
    """A session path that is not a readable regular file is 'no such session',
    not a raw ``IsADirectoryError`` / ``PermissionError`` in the caller's face."""

    def a_directory(self):
        meta = self.make(title="t")
        os.unlink(self.path(meta.id))
        os.mkdir(self.path(meta.id))
        return meta.id

    def unreadable(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="x"))
        os.chmod(self.path(meta.id), 0o000)
        self.addCleanup(os.chmod, self.path(meta.id), 0o600)
        return meta.id

    def assert_all_report_missing(self, sid, ops):
        for name, op in ops:
            with self.subTest(op=name):
                with self.assertRaises(KeyError):
                    op(sid)

    def readers_and_writers(self):
        msg = Message(role="user", content="x")
        return [
            ("load", self.store.load),
            ("export", self.store.export),
            ("append", lambda s: self.store.append(s, msg)),
            ("update", lambda s: self.store.update(s, title="x")),
            ("rename", lambda s: self.store.rename(s, "x")),
            ("record_usage", lambda s: self.store.record_usage(s, {"input_tokens": 1}, 0.1)),
        ]

    def test_a_directory_where_a_session_should_be(self):
        sid = self.a_directory()
        self.assert_all_report_missing(sid, self.readers_and_writers())
        with self.assertRaises(KeyError):
            self.store.delete(sid)
        self.assertFalse(self.store.exists(sid))
        self.assertEqual(self.store.list(), [])

    @unittest.skipIf(os.name == "nt", "POSIX modes only")
    def test_a_session_file_with_no_permissions(self):
        if os.geteuid() == 0:  # pragma: no cover
            self.skipTest("root reads anything")
        sid = self.unreadable()
        self.assert_all_report_missing(sid, self.readers_and_writers())
        self.assertEqual(self.store.list(), [])   # already tolerant, and stays so

    def test_the_reason_survives_in_the_exception(self):
        sid = self.a_directory()
        with self.assertRaises(KeyError) as caught:
            self.store.load(sid)
        self.assertIn(sid, str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, OSError)


# ------------------------------------------------------------------- hard links


class TestHardLinks(Base):
    @unittest.skipUnless(hasattr(os, "link"), "no hard links")
    def test_a_hardlinked_session_file_is_refused(self):
        outside = pathlib.Path(self.tmp.name) / "outside.txt"
        outside.write_bytes(b"someone else's file\n")
        meta = self.make(title="t")
        os.unlink(self.path(meta.id))
        try:
            os.link(str(outside), str(self.path(meta.id)))
        except (OSError, NotImplementedError):  # pragma: no cover
            self.skipTest("hard links unavailable here")
        with self.assertRaises(ValueError):
            self.store.append(meta.id, Message(role="user", content="written through"))
        self.assertEqual(outside.read_bytes(), b"someone else's file\n")


# ------------------------------------------------- layered symlink refusal


class TestSymlinkDefenceLayers(Base):
    """The symlink refusal has three layers; each is tested where it is the only
    one that can fire, because a test that any single layer passes is a test that
    proves nothing about that layer."""

    def a_symlink(self):
        target = pathlib.Path(self.tmp.name) / "target.jsonl"
        target.write_bytes(b'{"type":"meta","id":"x"}\n')
        link = self.store.sessions_dir / "linked.jsonl"
        try:
            os.symlink(str(target), str(link))
        except (OSError, NotImplementedError):  # pragma: no cover
            self.skipTest("symlinks unavailable")
        return target, link

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    @unittest.skipUnless(getattr(os, "O_NOFOLLOW", 0), "no O_NOFOLLOW")
    def test_the_open_itself_refuses_to_follow_a_symlink(self):
        _, link = self.a_symlink()
        with self.assertRaises(ValueError):
            store_mod._open_read(link)          # no islink() check involved

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_the_islink_guard_covers_platforms_without_o_nofollow(self):
        _, link = self.a_symlink()
        with mock.patch.object(store_mod, "_O_NOFOLLOW", 0):    # i.e. Windows
            with self.assertRaises(ValueError):
                store_mod._read_bytes(link)
            with self.assertRaises(ValueError):
                store_mod._read_header(link)

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    @unittest.skipUnless(getattr(os, "O_NOFOLLOW", 0), "no O_NOFOLLOW")
    def test_append_still_refuses_if_the_symlink_appears_after_the_check(self):
        # A link whose realpath stays inside the store and keeps the same
        # basename: path_for() has nothing to object to, so the open is the only
        # layer left. Losing the check/open race is simulated by disabling the
        # pre-flight lstat, which is exactly what a TOCTOU attacker arranges.
        hidden = self.store.sessions_dir / "sub"
        hidden.mkdir()
        target = hidden / "linked.jsonl"
        target.write_bytes(b'{"type":"meta","id":"x"}\n')
        try:
            os.symlink(str(target), str(self.store.sessions_dir / "linked.jsonl"))
        except (OSError, NotImplementedError):  # pragma: no cover
            self.skipTest("symlinks unavailable")
        self.store.path_for("linked")           # accepted: it looks like a session
        with mock.patch.object(Store, "_require_session", lambda *a, **k: None):
            with self.assertRaises(ValueError):
                self.store.append("linked", Message(role="user", content="through"))
        self.assertEqual(target.read_bytes(), b'{"type":"meta","id":"x"}\n')


# ------------------------------------------------------- durability primitives


class TestDurabilityPrimitives(Base):
    """fsync is invisible until the power goes out, so it is asserted directly:
    nothing else in the suite can tell a store that fsyncs from one that does not.
    """

    def record_fsyncs(self):
        seen = []
        real = os.fsync

        def spy(fd):
            try:
                seen.append(os.fstat(fd).st_ino)
            except OSError:  # pragma: no cover
                pass
            return real(fd)

        return seen, spy

    def test_append_fsyncs_the_session_file_before_returning(self):
        meta = self.make(title="t")
        ino = os.stat(self.path(meta.id)).st_ino
        seen, spy = self.record_fsyncs()
        with mock.patch("os.fsync", spy):
            self.store.append(meta.id, Message(role="user", content="durable"))
        self.assertIn(ino, seen, "append() returned before the data was on disk")

    def test_create_fsyncs_the_new_file(self):
        seen, spy = self.record_fsyncs()
        with mock.patch("os.fsync", spy):
            meta = self.make(title="t")
        self.assertIn(os.stat(self.path(meta.id)).st_ino, seen)

    def test_fsync_dir_really_syncs_the_directory(self):
        seen, spy = self.record_fsyncs()
        with mock.patch("os.fsync", spy):
            store_mod._fsync_dir(self.store.sessions_dir)
        if not hasattr(os, "O_DIRECTORY"):  # pragma: no cover
            self.skipTest("directory fsync unsupported")
        self.assertIn(os.stat(self.store.sessions_dir).st_ino, seen,
                      "_fsync_dir() did not sync the directory")

    def test_the_directory_is_synced_when_files_appear_and_vanish(self):
        calls = []
        with mock.patch.object(store_mod, "_fsync_dir", lambda p: calls.append(str(p))):
            meta = self.make(title="t")
            self.assertTrue(calls, "create() left the new name unsynced")
            calls.clear()
            self.store.rename(meta.id, "renamed")
            self.assertTrue(calls, "the header rewrite left the rename unsynced")
            calls.clear()
            self.store.delete(meta.id)
            self.assertTrue(calls, "delete() left the unlink unsynced")


# --------------------------------------------------------------- API edge cases


class TestLimitCoercion(Base):
    def setUp(self):
        super().setUp()
        for i in range(4):
            self.make(title="s%d" % i)

    def test_a_numeric_string_is_a_number(self):
        # argv values arrive as strings; "2" used to mean "show nothing".
        self.assertEqual(len(self.store.list(limit="2")), 2)
        self.assertEqual(len(self.store.list(limit=" 3 ")), 3)
        self.assertEqual(self.store.list(limit="0"), [])

    def test_junk_is_refused_rather_than_read_as_zero(self):
        for bad in ("two", "", "1.5", "0x2", None.__class__, object(), True, False,
                    float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.store.list(limit=bad)

    def test_floats_and_ints_still_work(self):
        self.assertEqual(len(self.store.list(limit=2.0)), 2)
        self.assertEqual(len(self.store.list(limit=99)), 4)

    def test_a_limit_is_read_the_way_a_list_index_is(self):
        # int() is not the same question as "is this a number the user typed":
        # "1_0" is ten to int() and "\u0661" is one, and resolve() refuses both,
        # out loud and for the documented reason. A limit arrives from the same
        # command line as a ref does.
        for bad in ("1_0", "\u0661", "\u0662\u0663", " 1 0 ", "1e2", "+ 3", "١٠"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.store.list(limit=bad)
        self.assertEqual(len(self.store.list(limit="+2")), 2)
        self.assertEqual(self.store.list(limit="-2"), [])


class TestResolveNumericRefs(Base):
    def setUp(self):
        super().setUp()
        self.first = self.make(title="first")

    def test_unicode_digits_are_not_list_indexes(self):
        # '²'.isdigit() is True but int('²') raises; '١'.isdigit() is True and
        # int('١') is 1, which silently resolved to session number one.
        for ref in ("²", "١", "٣", "½", "Ⅷ", "#٢"):
            with self.subTest(ref=ref):
                with self.assertRaises(KeyError):
                    self.store.resolve(ref)

    def test_ascii_indexes_still_work(self):
        self.assertEqual(self.store.resolve("1"), self.first.id)
        self.assertEqual(self.store.resolve("#1"), self.first.id)

    def test_an_absurdly_long_number_is_just_unknown(self):
        with self.assertRaises(KeyError):
            self.store.resolve("9" * 5000)


class TestExportSafety(Base):
    def test_a_stored_role_cannot_write_its_own_markup(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="\x1b[31m**FAKE**\nrole", content="body"))
        for fmt in ("markdown", "text"):
            with self.subTest(fmt=fmt):
                out = self.store.export(meta.id, fmt)
                self.assertNotIn("\x1b", out)
                self.assertNotIn("FAKE**\nrole", out)
        self.assertIn("body", self.store.export(meta.id))

    def test_known_roles_are_unaffected(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="assistant", content="hi"))
        self.assertIn("**Claude**", self.store.export(meta.id))


class TestHeaderPassthrough(Base):
    def test_fields_a_later_version_added_survive_a_rewrite(self):
        meta = self.make(title="t")
        raw = self.raw(meta.id)
        header = json.loads(raw.split(b"\n")[0].decode("utf-8"))
        header["starred_by"] = ["ada"]
        header["schema"] = 7
        rest = b"\n".join(raw.split(b"\n")[1:])
        self.write_raw(meta.id, json.dumps(header).encode("utf-8") + b"\n" + rest)

        self.store.rename(meta.id, "renamed by an older build")
        self.store.record_usage(meta.id, {"input_tokens": 5}, 0.5)
        after = json.loads(self.raw(meta.id).split(b"\n")[0].decode("utf-8"))
        self.assertEqual(after["starred_by"], ["ada"])
        self.assertEqual(after["schema"], 7)
        self.assertEqual(after["title"], "renamed by an older build")
        self.assertEqual(after["type"], "meta")

    def test_a_hostile_extra_field_cannot_shadow_a_real_one(self):
        meta = self.make(title="t")
        raw = self.raw(meta.id)
        rest = b"\n".join(raw.split(b"\n")[1:])
        header = json.loads(raw.split(b"\n")[0].decode("utf-8"))
        self.write_raw(meta.id, json.dumps(header).encode("utf-8") + b"\n" + rest)
        loaded = self.store.load(meta.id)[0]
        loaded.extra = {"type": "message", "title": "spoofed", "id": "spoofed"}
        loaded.title = "real"
        d = loaded.to_dict()
        self.assertEqual(d["title"], "real")
        self.assertEqual(d["id"], meta.id)
        self.assertNotIn("type", d)


class TestUpdateHonesty(Base):
    def test_updated_is_not_settable_because_it_is_derived(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="x", ts=time.time()))
        with self.assertRaises(ValueError) as caught:
            self.store.update(meta.id, updated=1.0)
        self.assertIn("derived", str(caught.exception))

    def test_created_is_settable_and_is_believed(self):
        meta = self.make(title="t")
        when = time.time() - 86400
        out = self.store.update(meta.id, created=when)
        self.assertAlmostEqual(out.created, when, places=3)
        self.assertAlmostEqual(self.store.load(meta.id)[0].created, when, places=3)


class TestIndexBounds(Base):
    def test_a_count_the_file_is_too_small_to_hold_is_distrusted(self):
        meta = self.make(title="titled")
        for i in range(6):
            self.store.append(meta.id, Message(role="user", content="m%d" % i))
        self.store.list()
        st = os.stat(self.path(meta.id))
        # 35 records cannot fit in this file, but 35 * 24 could.
        self.assertLess(st.st_size, 35 * 42)
        self.assertGreater(st.st_size, 35 * 24)
        doc = json.loads(self.store.index_path.read_text(encoding="utf-8"))
        doc["entries"][meta.id] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                   "count": 35, "updated": 0}
        self.store.index_path.write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(Store(self.root).list()[0].message_count, 6)

    def test_a_cached_timestamp_cannot_postdate_the_file_that_records_it(self):
        # An index entry that claims activity in the year 33000 would otherwise
        # pin that session to the top of every listing forever.
        meta = self.make(title="titled")
        self.store.append(meta.id, Message(role="user", content="m"))
        self.store.list()
        st = os.stat(self.path(meta.id))
        doc = json.loads(self.store.index_path.read_text(encoding="utf-8"))
        doc["entries"][meta.id] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                   "count": 1, "updated": 1e18}
        self.store.index_path.write_text(json.dumps(doc), encoding="utf-8")
        row = Store(self.root).list()[0]
        self.assertLessEqual(row.updated, st.st_mtime)
        self.assertEqual(row.message_count, 1)

    def test_an_untitled_session_does_not_rewrite_the_index_every_time(self):
        meta = self.make()
        for i in range(3):      # no user message, so nothing derives a title
            self.store.append(meta.id, Message(role="assistant", content="a%d" % i))
        self.store.list()
        before = self.store.index_path.stat().st_mtime_ns
        for _ in range(3):
            self.assertEqual(self.store.list()[0].message_count, 3)
        self.assertEqual(self.store.index_path.stat().st_mtime_ns, before,
                         "list() rewrote an unchanged index")

    def test_a_real_change_still_rewrites_it(self):
        meta = self.make()
        self.store.list()
        before = self.store.index_path.stat().st_mtime_ns
        self.store.append(meta.id, Message(role="assistant", content="new"))
        self.store.list()
        self.assertNotEqual(self.store.index_path.stat().st_mtime_ns, before)

    def test_a_temp_file_from_a_dead_process_goes_sooner(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        dead = self.store.sessions_dir / (".tmp-%s-%d-abcd.jsonl"
                                          % (store_mod._HOST, proc.pid))
        mine = self.store.sessions_dir / (store_mod._tmp_prefix() + "efgh.jsonl")
        for p in (dead, mine):
            p.write_bytes(b"half a rewrite")
            old = time.time() - 60
            os.utime(p, (old, old))
        self.store.list()
        self.assertFalse(dead.exists(), "litter from a dead writer was kept")
        self.assertTrue(mine.exists(), "a live writer's temp file was deleted")

    def test_a_live_writers_temp_file_is_not_swept_however_slow_it_is(self):
        # "Old enough that no plausible writer is still mid-rewrite" was a guess,
        # and one fsync of a very large transcript on a slow disk outlives it.
        # Sweeping then deletes a live rewrite's temp file and its os.replace
        # comes back as a raw FileNotFoundError out of update().
        mine = self.store.sessions_dir / (store_mod._tmp_prefix() + "slow.jsonl")
        mine.write_bytes(b"a rewrite this process is still inside")
        slow = time.time() - 7200
        os.utime(mine, (slow, slow))
        self.store.list()
        self.assertTrue(mine.exists(),
                        "swept a temp file whose writer is alive on this machine")
        ancient = time.time() - 3 * 86400      # by now the pid has been recycled
        os.utime(mine, (ancient, ancient))
        self.store.list()
        self.assertFalse(mine.exists(), "kept litter for ever because a pid was reused")

    def test_a_temp_file_from_another_machine_is_left_alone(self):
        # os.kill() answers a question about *this* machine: a pid written by
        # another host on a shared home directory reads back as "definitely
        # dead", and deleting that file destroys a live rewrite on that host.
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        theirs = self.store.sessions_dir / (".tmp-otherbox-%d-abcd.jsonl" % proc.pid)
        legacy = self.store.sessions_dir / (".tmp-%d-abcd.jsonl" % proc.pid)
        for p in (theirs, legacy):
            p.write_bytes(b"half a rewrite from elsewhere")
            old = time.time() - 60
            os.utime(p, (old, old))
        self.store.list()
        self.assertTrue(theirs.exists(),
                        "deleted another machine's in-flight rewrite")
        self.assertTrue(legacy.exists(),
                        "a name with no host in it is not a pid to trust")
        for p in (theirs, legacy):       # still swept once it is plainly stale
            old = time.time() - 7200
            os.utime(p, (old, old))
        self.store.list()
        self.assertFalse(theirs.exists())
        self.assertFalse(legacy.exists())


# ------------------------------------------------------------------- concurrency


APPENDER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[1])
    from lume.store import Store, Message

    root, sid, tag, n, size = sys.argv[2:7]
    store = Store(root)
    for i in range(int(n)):
        body = "%s:%d:%s" % (tag, i, tag * int(size))
        store.append(sid, Message(role="user", content=body))
    """
)

RENAMER = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, sys.argv[1])
    from lume.store import Store

    root, sid, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
    store = Store(root)
    for i in range(n):
        store.rename(sid, "title %d" % i)
        time.sleep(0.002)
    """
)


#: Appends a stream of >PIPE_BUF records and only records one in a witness log
#: *after* ``append()`` has returned, i.e. after its fsync. Everything in that
#: log is a message the store promised was durable.
KILLABLE = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    from lume.store import Store, Message
    import warnings
    warnings.simplefilter("ignore")

    root, sid, tag, witness, mode = sys.argv[2:7]
    store = Store(root)
    wfd = os.open(witness, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    pad = "P" * 9000                      # far past PIPE_BUF and one page
    i = 0
    while True:
        i += 1
        body = "%s-%06d:%s" % (tag, i, pad)
        role = "user" if (mode == "user" and i % 4 == 0) else "assistant"
        store.append(sid, Message(role=role, content=body))
        os.write(wfd, ("%s\\n" % body).encode())
        os.fsync(wfd)                     # only now is this message *confirmed*
        if mode == "usage" and i % 3 == 0:
            store.record_usage(sid, {"input_tokens": 3, "output_tokens": 4}, 0.001)
        if mode == "rewrite" and i % 3 == 0:
            store.rename(sid, "renamed %d" % i)
    """
)


class TestCrashDurability(Base):
    """SIGKILL in the middle of concurrent writes must not lose a confirmed message.

    This is the test the store exists to pass. It is deliberately end-to-end: real
    processes, real fsyncs, a kill at an arbitrary instant, and a witness log that
    knows nothing about the store's internals. A message is only "confirmed" once
    ``append()`` has returned, so anything in the witness log must be in the file
    afterwards — no losses, no duplicates, no half-records that parse as something
    else, and the session must still be usable.

    The `rewrite` writer is what makes this sensitive to the *cross-process*
    lock: ``rename`` rebuilds the file and swaps it in with ``os.replace``, so
    without the lock every append another process made in between is silently
    dropped. (``record_usage`` used to be a rewrite too and used to be that
    writer; it appends now, which is why one had to be added back here.)
    """

    ROUNDS = 2
    RECORD_LEN = len("A-000001:") + 9000

    def _script(self, name, body):
        p = pathlib.Path(self.tmp.name) / name
        p.write_text(body, encoding="utf-8")
        return str(p)

    def witnessed(self, path):
        lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").split("\n")
        # Only the final line can be short: the kill landed mid-write of the log.
        return [l for l in lines if len(l) == self.RECORD_LEN]

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "needs SIGKILL")
    @unittest.skipUnless(hasattr(os, "fork") or sys.platform != "win32", "POSIX only")
    def test_sigkill_during_concurrent_writes_loses_nothing_confirmed(self):
        script = self._script("killable.py", KILLABLE)
        rnd = random.Random(20250817)
        for rd in range(self.ROUNDS):
            with self.subTest(round=rd):
                self.one_round(script, rnd)

    def one_round(self, script, rnd):
        meta = self.make()               # untitled: the first user turn rewrites it
        witness = str(pathlib.Path(self.tmp.name) / ("witness-%s.log" % meta.id))
        procs = [
            subprocess.Popen([sys.executable, script, REPO_ROOT, str(self.root),
                              meta.id, tag, witness, mode],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for tag, mode in (("A", "plain"), ("B", "user"), ("C", "usage"),
                              ("D", "rewrite"))
        ]
        time.sleep(rnd.uniform(0.5, 0.9))
        for p in procs:
            os.kill(p.pid, signal.SIGKILL)
        for p in procs:
            p.communicate(timeout=60)

        confirmed = self.witnessed(witness)
        self.assertGreater(len(confirmed), 20, "the writers never got going")

        loaded, messages = self.quiet_load(meta.id)
        seen = {}
        for m in messages:
            seen[m.content] = seen.get(m.content, 0) + 1

        lost = [c for c in confirmed if c not in seen]
        self.assertEqual(lost[:3], [], "%d of %d confirmed messages were lost"
                         % (len(lost), len(confirmed)))
        dups = [c for c in confirmed if seen[c] > 1]
        self.assertEqual(dups[:3], [], "%d confirmed messages were duplicated" % len(dups))

        # Nothing in the file may be a mangled record: a torn write must be
        # skipped, never parsed as a different message.
        pattern = re.compile(r"^([ABCD])-(\d{6}):P{9000}$")
        for m in messages:
            self.assertRegex(m.content, pattern, "a torn record parsed as a message")

        # Each writer's own messages must be in order and without gaps.
        for tag in "ABCD":
            nums = [int(pattern.match(m.content).group(2)) for m in messages
                    if m.content.startswith(tag + "-")]
            self.assertEqual(nums, sorted(nums), "%s's appends were reordered" % tag)
            self.assertEqual(nums, list(range(1, len(nums) + 1)),
                             "a gap in %s's appends" % tag)

        self.assertEqual(loaded.message_count, len(messages))
        self.store.append(meta.id, Message(role="user", content="after the crash"))
        self.store.list()
        self.store.export(meta.id)
        self.assertEqual(self.quiet_load(meta.id)[1][-1].content, "after the crash")
        self.store.delete(meta.id)


class TestConcurrency(Base):
    def _script(self, name, body):
        p = pathlib.Path(self.tmp.name) / name
        p.write_text(body, encoding="utf-8")
        return str(p)

    def _run(self, args, timeout=240):
        return subprocess.Popen([sys.executable] + args,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _reap(self, procs):
        for p in procs:
            out, err = p.communicate(timeout=240)
            self.assertEqual(p.returncode, 0, err.decode("utf-8", "replace"))

    def test_two_processes_append_whole_records(self):
        """Every record from every writer arrives complete and exactly once.

        Honest scope: on Linux this passes with the cross-process lock removed,
        because ``write(2)`` under ``O_APPEND`` is already atomic for records this
        size. It is kept as an end-to-end check that nothing *else* in the write
        path splits or drops a record. What the lock is actually load-bearing for
        is tested by ``test_appends_survive_concurrent_header_rewrites`` and by
        ``TestCrashDurability`` — both fail when it is removed.
        """
        meta = self.make(title="race")
        script = self._script("appender.py", APPENDER)
        n, size = 40, 3000       # ~9 KB per record: far past one page
        procs = [self._run([script, REPO_ROOT, str(self.root), meta.id, tag, str(n), str(size)])
                 for tag in ("A", "B", "C")]
        self._reap(procs)

        raw = self.raw(meta.id)
        lines = [l for l in raw.split(b"\n") if l.strip()]
        self.assertEqual(len(lines), 1 + 3 * n, "lines were lost or split")
        seen = set()
        pattern = re.compile(r"^([ABC]):(\d+):\1+$")
        for line in lines[1:]:
            obj = json.loads(line.decode("utf-8"))   # a torn line fails right here
            self.assertEqual(obj["type"], "message")
            m = pattern.match(obj["content"])
            self.assertIsNotNone(m, "interleaved content: %r" % obj["content"][:80])
            self.assertEqual(len(obj["content"]), len(m.group(1)) + len(m.group(2)) + 2 + size)
            seen.add((m.group(1), int(m.group(2))))
        self.assertEqual(seen, {(t, i) for t in "ABC" for i in range(n)})

        loaded, messages = self.store.load(meta.id)
        self.assertEqual(len(messages), 3 * n)
        self.assertEqual(loaded.message_count, 3 * n)

    def test_appends_survive_concurrent_header_rewrites(self):
        meta = self.make(title="race")
        appender = self._script("appender.py", APPENDER)
        renamer = self._script("renamer.py", RENAMER)
        n = 60
        procs = [
            self._run([appender, REPO_ROOT, str(self.root), meta.id, "A", str(n), "200"]),
            self._run([renamer, REPO_ROOT, str(self.root), meta.id, "40"]),
        ]
        self._reap(procs)
        loaded, messages = self.store.load(meta.id)
        self.assertEqual(len(messages), n, "a header rewrite dropped messages")
        self.assertEqual([m.content.split(":")[1] for m in messages],
                         [str(i) for i in range(n)])
        self.assertTrue(loaded.title.startswith("title "))

    def test_threads_in_one_process(self):
        import threading
        meta = self.make(title="threads")
        errors = []

        def worker(tag):
            try:
                for i in range(30):
                    self.store.append(meta.id, Message(role="user", content="%s-%d" % (tag, i)))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in "xyz"]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        _, messages = self.store.load(meta.id)
        self.assertEqual(len(messages), 90)
        self.assertEqual(len({m.id for m in messages}), 90)


# ------------------------------------------------- cost of recording one turn


class TestRecordUsageCost(Base):
    """The write app.py makes after *every* assistant reply.

    ``record_usage`` used to read the whole transcript, ``json.loads`` every
    line of it and rewrite every byte through a temp file and ``os.replace``.
    Turn 600 of a 2.7 MB session therefore cost 8.7 ms against 1.6 ms for turn
    1, and 600 turns wrote 809 MB — a 302x write amplification, once per reply,
    for four integers and a float. It now appends a usage checkpoint, which is
    an append at any file size.
    """

    def big(self, turns=200):
        meta = self.make(title="fixed")
        for i in range(turns):
            self.store.append(meta.id, Message(role="user", content="ask %d" % i))
            self.store.append(meta.id, Message(role="assistant", content="y" * 4000))
            self.store.record_usage(meta.id, {"input_tokens": 10, "output_tokens": 5}, 0.001)
        return meta

    def counted(self, fn):
        """(bytes read, bytes written) by one call.

        `os.pread` is POSIX-only, so patch it only where it exists — the store
        reaches it through its own `_pread` shim on platforms that lack it, and
        that shim is built on `os.read`, which is counted either way.
        """
        read, wrote = [0], [0]
        real_read, real_write = os.read, os.write
        real_pread = getattr(os, "pread", None)

        def r(fd, n):
            data = real_read(fd, n)
            read[0] += len(data)
            return data

        def pr(fd, n, off):
            data = real_pread(fd, n, off)
            read[0] += len(data)
            return data

        def w(fd, data):
            wrote[0] += len(data)
            return real_write(fd, data)

        patches = [mock.patch("os.read", r), mock.patch("os.write", w)]
        if real_pread is not None:
            patches.append(mock.patch("os.pread", pr))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            fn()
        return read[0], wrote[0]

    def test_recording_usage_never_parses_or_rewrites_the_transcript(self):
        meta = self.big(turns=30)
        with counting_full_passes() as passes:
            with mock.patch.object(Store, "_rewrite_header",
                                   side_effect=AssertionError("rewrote the header")):
                for i in range(20):
                    self.store.record_usage(meta.id, {"input_tokens": 1}, 0.01)
        self.assertEqual(passes.count, 0,
                         "record_usage parsed the transcript %d time(s)" % passes.count)

    def test_the_transcript_is_never_replaced(self):
        meta = self.make(title="fixed")
        self.store.append(meta.id, Message(role="user", content="hello"))
        before = self.path(meta.id).stat().st_ino
        for i in range(5):
            self.store.record_usage(meta.id, {"input_tokens": 1}, 0.01)
        self.assertEqual(self.path(meta.id).stat().st_ino, before,
                         "recording usage swapped the file every message was in")

    def test_bytes_written_per_turn_do_not_grow_with_the_file(self):
        meta = self.make(title="fixed")
        self.store.append(meta.id, Message(role="user", content="hello"))
        _, small = self.counted(
            lambda: self.store.record_usage(meta.id, {"input_tokens": 1}, 0.01))
        for i in range(1000):                    # ~4 MB of transcript
            self.store.append(meta.id, Message(role="assistant", content="y" * 4000))
            if i % 3 == 0:
                self.store.record_usage(meta.id, {"input_tokens": 1}, 0.01)
        size = self.path(meta.id).stat().st_size
        self.assertGreater(size, 4_000_000)
        read, wrote = self.counted(
            lambda: self.store.record_usage(meta.id, {"input_tokens": 1}, 0.01))
        self.assertLess(wrote, 4 * 1024,
                        "one turn wrote %d bytes into a %d byte session" % (wrote, size))
        self.assertLess(wrote, small + 1024)
        self.assertLess(read, 512 * 1024,
                        "one turn read %d bytes of a %d byte session" % (read, size))

    def test_a_turn_reads_kilobytes_to_find_a_hundred_byte_checkpoint(self):
        # 800 turns of a 3.76 MB transcript used to pread 52 MB — a fixed 64 KB
        # window per turn plus a fixed 64 KB gulp for line 1 — to read back one
        # ~150 byte record and one ~300 byte header. Flat in the file size, but
        # 14x more of the file than it ever looked at.
        meta = self.big(turns=60)
        size = self.path(meta.id).stat().st_size
        self.assertGreater(size, 240_000)
        read, wrote = self.counted(
            lambda: self.store.record_usage(meta.id, {"input_tokens": 1}, 0.001))
        self.assertLess(read, 48 * 1024,
                        "one turn read %d bytes to find its predecessor's checkpoint" % read)

    def test_a_turn_costs_the_same_at_the_end_of_a_long_session(self):
        # Bytes, not seconds: the shape of the cost is the claim, and a wall
        # clock measures the machine as much as the code.
        meta = self.big(turns=100)
        early_size = self.path(meta.id).stat().st_size
        early = sum(self.counted(
            lambda: self.store.record_usage(meta.id, {"input_tokens": 1}, 0.001)))
        for i in range(900):
            self.store.append(meta.id, Message(role="assistant", content="y" * 4000))
            if i % 3 == 0:
                self.store.record_usage(meta.id, {"input_tokens": 1}, 0.001)
        late_size = self.path(meta.id).stat().st_size
        late = sum(self.counted(
            lambda: self.store.record_usage(meta.id, {"input_tokens": 1}, 0.001)))
        self.assertGreater(late_size, early_size * 5)
        self.assertLess(late, early * 2,
                        "a turn moved %d bytes at %d MB and %d bytes at %d MB"
                        % (early, early_size // 10**6, late, late_size // 10**6))


class TestUsageCheckpoints(Base):
    """Totals live in checkpoint records folded over the header at read time."""

    def totals(self, meta):
        return (meta.input_tokens, meta.output_tokens, meta.cache_read_tokens,
                meta.cache_write_tokens, round(meta.cost_usd, 6))

    def record(self, sid, n=1):
        for _ in range(n):
            self.store.record_usage(sid, {"input_tokens": 100, "output_tokens": 20,
                                          "cache_read_input_tokens": 3,
                                          "cache_creation_input_tokens": 7}, 0.01)

    def test_a_checkpoint_is_not_a_message(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="only me"))
        self.record(meta.id, 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded, messages = self.store.load(meta.id)
        self.assertEqual([m.content for m in messages], ["only me"])
        self.assertEqual(loaded.message_count, 1)
        self.assertEqual([w for w in caught if issubclass(w.category, StoreWarning)], [],
                         "a record this build writes was read back as damage")

    def test_a_header_rewrite_folds_them_in_without_double_counting(self):
        meta = self.make(title="t")
        self.record(meta.id, 3)
        before = self.totals(self.store.load(meta.id)[0])
        self.assertEqual(before[:4], (300, 60, 9, 21))
        self.store.update(meta.id, pinned=True)          # rewrites the header
        self.assertEqual(self.totals(self.store.load(meta.id)[0]), before)
        self.assertNotIn(b'"type":"usage"', self.raw(meta.id),
                         "folded checkpoints were left in the file to be counted twice")
        self.record(meta.id, 1)
        self.assertEqual(self.totals(self.store.load(meta.id)[0])[:4], (400, 80, 12, 28))

    def test_an_explicit_total_wins_over_the_checkpoints_it_replaces(self):
        meta = self.make(title="t")
        self.record(meta.id, 2)
        self.store.update(meta.id, input_tokens=5, cost_usd=0.5)
        loaded = self.store.load(meta.id)[0]
        self.assertEqual(loaded.input_tokens, 5)
        self.assertAlmostEqual(loaded.cost_usd, 0.5)

    def test_a_damaged_checkpoint_costs_one_turn_not_the_history(self):
        meta = self.make(title="t")
        self.record(meta.id, 3)
        raw = self.raw(meta.id)
        lines = raw.split(b"\n")
        while not lines[-1]:
            lines.pop()
        lines[-1] = lines[-1][:20]           # a crash mid-checkpoint
        self.write_raw(meta.id, b"\n".join(lines) + b"\n")
        loaded = self.quiet_load(meta.id)[0]
        self.assertEqual(self.totals(loaded)[:4], (200, 40, 6, 14))

    def test_a_checkpoint_survives_a_reopened_store(self):
        meta = self.make(title="t")
        self.record(meta.id, 2)
        self.assertEqual(self.totals(Store(self.root).load(meta.id)[0])[:4], (200, 40, 6, 14))

    def test_list_shows_totals_that_no_header_ever_held(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.record(meta.id, 2)
        for store in (self.store, Store(self.root), self.store):   # cold, then cached
            row = [m for m in store.list() if m.id == meta.id][0]
            self.assertEqual((row.input_tokens, row.output_tokens), (200, 40))
            self.assertAlmostEqual(row.cost_usd, 0.02)

    def test_the_fast_path_is_still_a_fast_path(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.record(meta.id, 1)
        self.store.list()                    # warm the cache
        with counting_full_passes() as passes:
            row = [m for m in self.store.list() if m.id == meta.id][0]
        self.assertEqual(passes.count, 0, "list() re-read a session it had cached")
        self.assertEqual(row.input_tokens, 100)

    def test_an_index_written_before_checkpoints_existed_is_not_believed(self):
        # The entries an older build wrote carry no totals. Believing the header
        # for those would report a session's cost as zero for as long as its
        # size and mtime happened to match.
        meta = self.make(title="t")
        self.record(meta.id, 1)
        self.store.list()
        doc = json.loads(self.store.index_path.read_text(encoding="utf-8"))
        entry = doc["entries"][meta.id]
        for broken in ({k: v for k, v in entry.items() if k != "usage"},
                       dict(entry, usage="nope"),
                       dict(entry, usage=[1, 2, 3]),
                       dict(entry, usage=[1, 2, 3, 4, float("nan")])):
            with self.subTest(usage=broken.get("usage")):
                doc["entries"][meta.id] = broken
                self.store.index_path.write_text(json.dumps(doc), encoding="utf-8")
                row = [m for m in Store(self.root).list() if m.id == meta.id][0]
                self.assertEqual(row.input_tokens, 100)

    def test_recording_usage_repairs_a_truncated_line_first(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        with open(str(self.path(meta.id)), "ab") as fh:
            fh.write(b'{"type":"message","role":"user","content":"cut')
        with self.assertWarns(StoreWarning):
            self.store.record_usage(meta.id, {"input_tokens": 4}, 0.0)
        loaded, messages = self.quiet_load(meta.id)
        self.assertEqual([m.content for m in messages], ["q"])
        self.assertEqual(loaded.input_tokens, 4)

    def test_recording_usage_refuses_the_files_append_refuses(self):
        meta = self.make(title="t")
        outside = pathlib.Path(self.tmp.name) / "outside.jsonl"
        outside.write_bytes(b"someone else's file\n")
        victim = self.make(title="v")
        os.unlink(self.path(victim.id))
        os.link(str(outside), str(self.path(victim.id)))
        with self.assertRaises(ValueError):
            self.store.record_usage(victim.id, {"input_tokens": 1}, 0.0)
        self.assertEqual(outside.read_bytes(), b"someone else's file\n")
        with self.assertRaises(KeyError):
            self.store.record_usage("nosuchsession", {"input_tokens": 1}, 0.0)


# --------------------------------------------------- interrupting a lock (F1)


class TestInterruptedLock(Base):
    """Ctrl-C is how a TUI user cancels, and it lands wherever the code is.

    ``fcntl.flock(LOCK_EX)`` blocks, so "wherever the code is" is routinely
    *inside the acquire*, and only ``OSError`` was caught there: the descriptor
    the store had just opened was never closed. One per cancelled write, in a
    process that runs all day.
    """

    def open_fds(self):
        return len(os.listdir("/proc/self/fd"))

    @unittest.skipUnless(os.path.isdir("/proc/self/fd"), "needs /proc")
    def test_a_cancelled_acquire_does_not_leak_a_descriptor(self):
        meta = self.make(title="t")
        path = self.store.path_for(meta.id)

        def interrupt(fd):
            raise KeyboardInterrupt()

        with mock.patch.object(store_mod, "_lock_acquire", interrupt):
            with contextlib.suppress(KeyboardInterrupt):
                with store_mod._file_lock(path):
                    pass
            before = self.open_fds()
            for _ in range(50):
                with self.assertRaises(KeyboardInterrupt):
                    with store_mod._file_lock(path):
                        pass          # pragma: no cover
            after = self.open_fds()
        self.assertLessEqual(after - before, 2,
                             "leaked %d descriptors over 50 cancelled acquires"
                             % (after - before))
        self.assertEqual(store_mod._locks, {})
        self.store.append(meta.id, Message(role="user", content="still works"))

    def test_a_cancelled_acquire_is_not_reported_as_a_lock_failure(self):
        # Swallowing it would turn Ctrl-C into "cross-process locking is off now".
        meta = self.make(title="t")

        def interrupt(fd):
            raise KeyboardInterrupt()

        with mock.patch.object(store_mod, "_lock_acquire", interrupt):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaises(KeyboardInterrupt):
                    self.store.append(meta.id, Message(role="user", content="x"))
        self.assertEqual([w for w in caught if issubclass(w.category, StoreWarning)], [])

    def test_running_out_of_descriptors_is_not_a_reason_to_stop_locking(self):
        # The degradation path exists for a filesystem that cannot lock. EMFILE
        # is not that: it is a passing condition, and continuing without the
        # lock is the exact state the SIGKILL test proves loses messages.
        meta = self.make(title="t")
        with mock.patch.object(store_mod, "_lock_acquire", FailingLock(errno.EMFILE)):
            with self.assertRaises(OSError) as caught:
                self.store.append(meta.id, Message(role="user", content="x"))
        self.assertEqual(caught.exception.errno, errno.EMFILE)

        real_open = os.open

        def no_more(path, *a, **kw):
            if str(path).endswith(".lock"):
                raise OSError(errno.EMFILE, "too many open files")
            return real_open(path, *a, **kw)

        with mock.patch("os.open", no_more):
            with self.assertRaises(OSError):
                self.store.append(meta.id, Message(role="user", content="x"))

    def test_a_filesystem_that_cannot_lock_still_degrades(self):
        meta = self.make(title="t")
        with mock.patch.object(store_mod, "_lock_acquire", FailingLock(errno.ENOLCK)):
            with self.assertWarns(StoreWarning):
                self.store.append(meta.id, Message(role="user", content="x"))
        self.assertEqual(len(self.quiet_load(meta.id)[1]), 1)


# ------------------------------------------------------- rescue copies (F2/F3)


class TestRescueCopies(Base):
    """Every damage event's bytes are kept, and none of them silently is not."""

    def damage(self, sid, marker=b" <-- damaged"):
        lines = self.raw(sid).split(b"\n")
        lines[0] = lines[0][:40] + marker
        self.write_raw(sid, b"\n".join(lines))
        return self.raw(sid)

    def rescues(self, sid):
        base = self.path(sid).name
        return sorted(n for n in os.listdir(self.store.sessions_dir)
                      if n.startswith(base + ".bak"))

    def busy_session(self, marker=b" <-- damaged"):
        meta = self.make(title="Important thread", system="a long system prompt")
        self.store.append(meta.id, Message(role="user", content="body"))
        self.store.record_usage(meta.id, {"input_tokens": 900000}, 12.34)
        return meta, self.damage(meta.id, marker)

    def rename_quietly(self, sid, title):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.store.rename(sid, title)
        return [str(w.message) for w in caught if issubclass(w.category, StoreWarning)]

    def test_a_second_damage_event_keeps_its_own_bytes(self):
        meta, first_bytes = self.busy_session()
        self.rename_quietly(meta.id, "recovered")
        for i in range(50):
            self.store.append(meta.id, Message(role="assistant", content="reply %d" % i))
        self.store.record_usage(meta.id, {"input_tokens": 4000}, 123.45)
        second_bytes = self.damage(meta.id, b" <-- damaged again")

        messages = self.rename_quietly(meta.id, "recovered twice")

        kept = {(self.store.sessions_dir / n).read_bytes() for n in self.rescues(meta.id)}
        self.assertIn(first_bytes, kept, "the first rescue copy was overwritten")
        self.assertIn(second_bytes, kept,
                      "the second damage event was dropped and the user was told "
                      "it had been kept: %s" % messages)
        named = [n for n in self.rescues(meta.id)
                 if any(m.endswith("at %s." % (self.store.sessions_dir / n))
                        for m in messages)]
        self.assertEqual(len(named), 1,
                         "the warning did not name the copy it made: %s" % messages)
        self.assertEqual((self.store.sessions_dir / named[0]).read_bytes(), second_bytes)
        for name in self.rescues(meta.id):
            self.assertEqual(stat.S_IMODE((self.store.sessions_dir / name).stat().st_mode),
                             0o600)

    def test_the_same_bytes_are_not_copied_twice(self):
        meta, damaged = self.busy_session()
        self.rename_quietly(meta.id, "one")
        self.assertEqual(self.rescues(meta.id), [self.path(meta.id).name + ".bak"])
        self.write_raw(meta.id, damaged)              # the very same damage again
        messages = self.rename_quietly(meta.id, "two")
        self.assertEqual(self.rescues(meta.id), [self.path(meta.id).name + ".bak"],
                         "kept a second copy of bytes that were already safe")
        self.assertTrue(any("already kept" in m for m in messages),
                        "claimed to have made a copy it did not make: %s" % messages)

    def test_a_useless_bak_does_not_disable_the_rescue(self):
        # lexists() is satisfied by all of these, and any of them used to mean
        # "already rescued, nothing to do" for the rest of the session's life.
        for kind in ("empty file", "directory", "symlink", "dangling symlink"):
            with self.subTest(kind=kind):
                self.setUp()
                meta, damaged = self.busy_session()
                bak = pathlib.Path(str(self.path(meta.id)) + ".bak")
                if kind == "empty file":
                    bak.write_bytes(b"")
                elif kind == "directory":
                    bak.mkdir()
                elif kind == "symlink":
                    other = pathlib.Path(self.tmp.name) / "elsewhere"
                    other.write_bytes(b"not a transcript")
                    os.symlink(str(other), bak)
                else:
                    os.symlink(str(bak) + "-nowhere", bak)
                self.rename_quietly(meta.id, "rescued anyway")
                kept = []
                for name in self.rescues(meta.id):
                    p = self.store.sessions_dir / name
                    if p.is_file() and not p.is_symlink():
                        kept.append(p.read_bytes())
                self.assertIn(damaged, kept,
                              "%s on the rescue name threw the only copy away" % kind)

    def test_delete_takes_every_rescue_copy_with_it(self):
        meta, _ = self.busy_session()
        self.rename_quietly(meta.id, "one")
        self.write_raw(meta.id, self.damage(meta.id, b" <-- and again"))
        self.rename_quietly(meta.id, "two")
        self.assertGreater(len(self.rescues(meta.id)), 1)
        self.store.delete(meta.id)
        self.assertEqual(sorted(os.listdir(self.store.sessions_dir)), [])


# --------------------------------------------------- reaching an orphan (F9)


class TestOrphanedSidecars(Base):
    """A ``.bak`` is a full plaintext transcript. It must be reachable."""

    def orphan(self):
        meta = self.make(title="t", system="private thing")
        self.store.append(meta.id, Message(role="user", content="private thing"))
        bak = pathlib.Path(str(self.path(meta.id)) + ".bak")
        bak.write_bytes(self.raw(meta.id))
        os.unlink(self.path(meta.id))
        return meta, bak

    def test_delete_sweeps_a_rescue_copy_whose_session_is_gone(self):
        meta, bak = self.orphan()
        with self.assertRaises(KeyError):       # still "no such session"
            self.store.delete(meta.id)
        self.assertFalse(bak.exists(),
                         "the one file a user would most want gone had no API path")

    def test_list_sweeps_a_rescue_copy_whose_session_is_gone(self):
        meta, bak = self.orphan()
        extra = pathlib.Path(str(self.path(meta.id)) + ".bak.3")
        extra.write_bytes(b"more of the same\n")
        self.store.list()
        self.assertFalse(bak.exists())
        self.assertFalse(extra.exists())

    def test_a_rescue_copy_with_a_session_beside_it_is_left_alone(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="live"))
        bak = pathlib.Path(str(self.path(meta.id)) + ".bak")
        bak.write_bytes(b"rescued once\n")
        self.store.list()
        self.assertTrue(bak.exists(), "swept the rescue copy of a live session")

    def test_a_stale_orphaned_lock_goes_but_a_fresh_one_stays(self):
        meta = self.make(title="t")
        fresh = pathlib.Path(str(self.path(meta.id)) + ".lock")
        fresh.write_bytes(b"")
        os.unlink(self.path(meta.id))
        self.store.list()
        self.assertTrue(fresh.exists(), "took a lock somebody may still hold")
        old = time.time() - 7200
        os.utime(fresh, (old, old))
        self.store.list()
        self.assertFalse(fresh.exists())


# ------------------------------------------------------ unopenable paths (F4)


class TestFifoSessions(Base):
    """A FIFO under a session name must be refused, not waited on."""

    def call_with_deadline(self, fn, seconds=20):
        import threading
        outcome = []

        def run():
            try:
                outcome.append(("ok", fn()))
            except BaseException as exc:                # pragma: no cover
                outcome.append(("raised", exc))

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(seconds)
        self.assertFalse(worker.is_alive(),
                         "blocked for %ds with no writer on the other end" % seconds)
        return outcome[0]

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs mkfifo")
    def test_load_and_export_do_not_wait_for_a_writer(self):
        meta = self.make(title="t")
        os.unlink(self.path(meta.id))
        os.mkfifo(str(self.path(meta.id)))
        for name, call in (("load", lambda: self.store.load(meta.id)),
                           ("export", lambda: self.store.export(meta.id)),
                           ("append", lambda: self.store.append(
                               meta.id, Message(role="user", content="x"))),
                           ("record_usage", lambda: self.store.record_usage(
                               meta.id, {"input_tokens": 1}, 0.0))):
            with self.subTest(op=name):
                kind, value = self.call_with_deadline(call)
                self.assertEqual(kind, "raised")
                self.assertIsInstance(value, KeyError)
        self.assertEqual(self.store.list(), [])
        self.assertFalse(self.store.exists(meta.id))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs mkfifo")
    def test_the_open_itself_does_not_wait_either(self):
        # Belt and braces: _require_session is the guard, O_NONBLOCK is what
        # stops the raw read path parking if anything ever gets past it.
        fifo = pathlib.Path(self.tmp.name) / "fifo"
        os.mkfifo(str(fifo))
        kind, value = self.call_with_deadline(lambda: store_mod._read_header(fifo))
        self.assertEqual(kind, "ok")


# ------------------------------------------------- one title, one message (F5)


class TestTitleAgreement(Base):
    """The persisted title and the derived title are the same title.

    ``_maybe_autotitle`` stamped the message it happened to be called for;
    ``_meta_from`` derives from the *first user message in the file*. They agree
    only until something clears the stored title — which ``rename(sid, "")`` is
    documented to do — after which the header disagreed with every derived view
    of the same session, permanently.
    """

    def header_title(self, sid):
        return store_mod._read_header(self.path(sid))["title"]

    def test_clearing_a_title_and_carrying_on_restores_the_derived_one(self):
        meta = self.make()
        self.store.append(meta.id, Message(role="user", content="the first question"))
        self.assertEqual(self.header_title(meta.id), "the first question")
        self.store.rename(meta.id, "")                  # documented: restore the derived one
        self.store.append(meta.id, Message(role="user", content="a much later question"))
        self.assertEqual(self.store.load(meta.id)[0].title, "the first question")
        self.assertEqual(self.header_title(meta.id), "the first question",
                         "the stored title and the derived title parted company")
        self.assertEqual(self.store.list()[0].title, "the first question")

    def test_an_opening_message_with_no_text_in_it_is_skipped_by_both(self):
        meta = self.make()
        self.store.append(meta.id, Message(role="user", content="   \n\t "))
        self.store.append(meta.id, Message(role="user", content="second question"))
        self.assertEqual(self.store.load(meta.id)[0].title, "second question")
        self.assertEqual(self.header_title(meta.id), "second question")
        self.assertEqual(self.store.list()[0].title, "second question")

    def test_a_transcript_with_no_title_in_it_is_not_re_read_every_turn(self):
        # Deriving from the file instead of from the message in hand means
        # "still no title" is an answer the file has to be asked for. Asking it
        # once per turn is the quadratic bug wearing a different hat.
        meta = self.make()
        self.store.append(meta.id, Message(role="user", content=" "))
        with counting_full_passes() as passes:
            for i in range(20):
                self.store.append(meta.id, Message(role="user", content="\x00\x01"))
        self.assertEqual(passes.count, 0,
                         "parsed the transcript %d time(s) looking for a title that "
                         "is not in it" % passes.count)
        self.assertEqual(self.store.load(meta.id)[0].title, "")


# ------------------------------------------------------ a header of any size (F7)


class TestLargeHeaders(Base):
    """A long ``/system`` prompt must not quietly cost the session its fast path."""

    def test_a_header_over_64k_still_lists_and_still_titles(self):
        meta = self.make()
        self.store.update(meta.id, system="S" * 100_000)
        self.store.append(meta.id, Message(role="user", content="a real question"))
        self.assertGreater(self.raw(meta.id).find(b"\n"), 1 << 16)
        self.assertEqual(store_mod._read_header(self.path(meta.id))["title"],
                         "a real question", "no title was ever persisted")
        self.store.list()                               # warm the cache
        with counting_full_passes() as passes:
            rows = self.store.list()
        self.assertEqual([m.title for m in rows], ["a real question"])
        self.assertEqual(passes.count, 0,
                         "a big header dropped the session off the fast path for good")

    def test_a_header_with_no_newline_at_all_is_still_not_a_header(self):
        meta = self.make(title="t")
        self.write_raw(meta.id, b"{" + b"x" * 200_000)
        self.assertIsNone(store_mod._read_header(self.path(meta.id)))


# ------------------------------------------------------------ mode and memory


class TestRewriteHygiene(Base):
    def test_a_rewritten_session_is_still_private(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="x"))
        self.store.update(meta.id, pinned=True)
        self.assertEqual(stat.S_IMODE(self.path(meta.id).stat().st_mode), 0o600)
        self.store.list()
        self.assertEqual(stat.S_IMODE(self.store.index_path.stat().st_mode), 0o600)

    def test_a_rewrite_does_not_read_the_transcript_into_memory(self):
        # One update() on a 200 MB transcript took RSS from 22 MB to 596 MB: the
        # file was read whole, split whole and joined whole to change line 1.
        # tracemalloc rather than RSS — a high-water mark is inherited across a
        # fork and never comes back down, so it answers a different question.
        import tracemalloc
        meta = self.make(title="t")
        blob = "y" * 100_000
        for i in range(160):                       # 16 MB
            self.store.append(meta.id, Message(role="assistant", content=blob))
        size = self.path(meta.id).stat().st_size
        self.assertGreater(size, 15_000_000)
        tracemalloc.start()
        try:
            self.store.update(meta.id, pinned=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, size // 4,
                        "rewriting a %d MB header held %d MB at once"
                        % (size // 10**6, peak // 10**6))
        self.assertEqual(len(self.quiet_load(meta.id)[1]), 160)


# ------------------------------------------------- the checkpoint format's backstops


class UsageBase(Base):
    """Shared helpers for the token/cost totals."""

    def totals(self, meta):
        return (meta.input_tokens, meta.output_tokens, meta.cache_read_tokens,
                meta.cache_write_tokens, round(meta.cost_usd, 6))

    def record(self, sid, n=1):
        for _ in range(n):
            self.store.record_usage(sid, {"input_tokens": 100, "output_tokens": 20,
                                          "cache_read_input_tokens": 3,
                                          "cache_creation_input_tokens": 7}, 0.01)

    def checkpoint(self, sid, **fields):
        """Append a raw usage checkpoint, the way a crash or an old build might."""
        record = {"type": "usage", "ts": time.time()}
        record.update(fields)
        with open(str(self.path(sid)), "ab") as fh:
            fh.write(json.dumps(record).encode("utf-8") + b"\n")


class TestUsageBackstops(UsageBase):
    """The one-line defences the checkpoint format rests on.

    Each test here was watched to fail with the single line it covers deleted,
    and only these tests failed: without them the whole suite stayed green while
    a session's history — or every assistant message in it — was destroyed.
    """

    def test_history_survives_a_turn_that_outruns_the_tail_window(self):
        # _counters_at walks back from the end looking for the newest checkpoint
        # and gives up after _TAIL_CAP. Giving up must mean a full pass, not
        # "there is no history": a compaction drops every checkpoint, so the very
        # next turn of a large transcript has none to find, and starting from
        # zero there rewrites the session's whole token and cost history as one
        # turn's worth — silently, and durably at the next compaction.
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.record(meta.id, 20)
        before = self.totals(self.store.load(meta.id)[0])
        self.assertEqual(before[:4], (2000, 400, 60, 140))
        self.store.update(meta.id, pinned=True)          # drops every checkpoint
        self.assertNotIn(b'"type":"usage"', self.raw(meta.id))
        blob = "z" * (1 << 20)
        for _ in range(9):                               # more than _TAIL_CAP
            self.store.append(meta.id, Message(role="assistant", content=blob))
        self.assertGreater(self.path(meta.id).stat().st_size,
                           store_mod._TAIL_CAP + (1 << 21))
        self.store.record_usage(meta.id, {"input_tokens": 1}, 0.0)
        after = self.totals(self.store.load(meta.id)[0])
        self.assertEqual(after[:4], (2001, 400, 60, 140),
                         "the history before the tail window was thrown away")
        self.assertAlmostEqual(after[4], before[4])
        self.assertEqual(self.store.load(meta.id)[0].message_count, 10)

    def test_compaction_keeps_assistant_messages_that_carry_usage(self):
        # app.py puts usage={...} on every assistant message, so every one of
        # those lines contains the substring '"usage"' and is handed to
        # _usage_record() during a compaction. Only its type test stops a rename
        # deleting all of them.
        meta = self.make(title="t")
        for i in range(4):
            self.store.append(meta.id, Message(role="user", content="q%d" % i))
            self.store.append(meta.id, Message(
                role="assistant", content="a%d" % i,
                usage={"input_tokens": 10, "output_tokens": 3}))
            self.record(meta.id, 1)
        lines = [l for l in self.raw(meta.id).split(b"\n") if l]
        self.assertEqual(sum(1 for l in lines if b'"usage"' in l), 8,
                         "the fixture stopped exercising the substring test")
        before = [(m.role, m.content, m.usage) for m in self.store.load(meta.id)[1]]
        self.assertEqual(len(before), 8)
        self.store.rename(meta.id, "a new title")
        after = [(m.role, m.content, m.usage) for m in self.store.load(meta.id)[1]]
        self.assertEqual(after, before, "compaction ate the assistant messages")
        self.assertEqual(self.store.load(meta.id)[0].message_count, 8)

    def test_a_checkpoints_own_count_carries_the_messages_before_it(self):
        # A checkpoint records what its writer knew about the messages behind it
        # ("n"), so the next writer does not have to re-read them. Dropping that
        # accumulation makes every turn's meta claim the session is one turn old.
        meta = self.make(title="t")
        for i in range(6):
            self.store.append(meta.id, Message(role="user", content="q%d" % i))
        self.record(meta.id, 1)
        self.store.append(meta.id, Message(role="assistant", content="a"))
        out = self.store.record_usage(meta.id, {"input_tokens": 1}, 0.0)
        self.assertEqual(out.message_count, 7)
        self.assertEqual(self.store.load(meta.id)[0].message_count, 7)

    def test_junk_in_a_checkpoint_cannot_reach_the_reader(self):
        # A negative token count or a NaN/string cost used to sail through into
        # export()'s "%.4f" and into list()'s sort key.
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.checkpoint(meta.id, input_tokens=-5, output_tokens=[1], cost_usd="free")
        loaded = self.quiet_load(meta.id)[0]
        self.assertEqual(self.totals(loaded), (0, 0, 0, 0, 0.0))
        self.assertIn("$0.0000", self.store.export(meta.id))
        self.checkpoint(meta.id, input_tokens=3, cost_usd=float("nan"))
        self.assertEqual(self.quiet_load(meta.id)[0].cost_usd, 0.0)
        self.assertEqual(self.quiet_load(meta.id)[0].input_tokens, 3)
        self.assertTrue(Store(self.root).list())

    def test_a_rewrite_fsyncs_the_new_file_before_it_replaces_the_old(self):
        # os.replace is atomic against a crash; it is not durable against one.
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        order = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            order.append("fsync")
            return real_fsync(fd)

        def replace(src, dst, **kw):
            order.append("replace")
            return real_replace(src, dst, **kw)

        with mock.patch("os.fsync", fsync), mock.patch("os.replace", replace):
            self.store.update(meta.id, pinned=True)
        self.assertIn("replace", order)
        self.assertEqual(order[:order.index("replace")].count("fsync") > 0, True,
                         "replaced in a transcript that had never been fsynced")


#: A writer that records usage after every turn and can be killed inside the
#: checkpoint write or inside a compaction's os.replace. A tag is written to the
#: witness log only *after* record_usage() has returned, so every line in that
#: log is one turn's counters the store promised were durable.
USAGE_KILLABLE = textwrap.dedent(
    """
    import os, random, signal, sys, warnings
    sys.path.insert(0, sys.argv[1])
    import lume.store as store_mod
    from lume.store import Store, Message
    warnings.simplefilter("ignore")

    root, sid, tag, witness, mode, seed = sys.argv[2:8]
    rnd = random.Random(int(seed))
    store = Store(root)
    wfd = os.open(witness, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    real_write, real_replace = store_mod._write_all, os.replace

    def die(p):
        if rnd.random() < p:
            os.kill(os.getpid(), signal.SIGKILL)

    if mode == "killckpt":
        def write_all(fd, data):
            if b'"type":"usage"' in data and len(data) < 400:
                die(0.03)
                if rnd.random() < 0.03:        # half a checkpoint, then death
                    real_write(fd, data[:len(data) // 2])
                    os.kill(os.getpid(), signal.SIGKILL)
            out = real_write(fd, data)
            if b'"type":"usage"' in data:
                die(0.03)
            return out
        store_mod._write_all = write_all
    elif mode == "killcompact":
        def replace(src, dst, **kw):
            die(0.15)
            out = real_replace(src, dst, **kw)
            die(0.15)
            return out
        os.replace = replace

    i = 0
    while True:
        i += 1
        store.append(sid, Message(role="user", content="%s-%06d:%s" % (tag, i, "P" * 200)))
        if mode != "rename":
            store.record_usage(sid, {"input_tokens": 1, "output_tokens": 1,
                                     "cache_read_input_tokens": 1,
                                     "cache_creation_input_tokens": 1}, 0.01)
            os.write(wfd, b"%s\\n" % tag.encode())   # only now is it confirmed
            os.fsync(wfd)
        if mode in ("rename", "killcompact") and i % 3 == 0:
            store.rename(sid, "t%d" % i)
    """
)


class TestUsageCrashDurability(UsageBase):
    """SIGKILL around a checkpoint must not lose confirmed token or cost history.

    TestCrashDurability asks the same question about messages. It cannot ask it
    about the counters, because the counters are not messages: they are a
    read-modify-write of the file's tail, and what protects them is the session
    lock ``record_usage`` takes. Remove that lock and every store test still
    passes while roughly a fifth of the confirmed history is lost — two writers
    read the same checkpoint, add their own turn to it, and the second one's
    write erases the first one's.

    So this is deliberately end-to-end: real processes, real fsyncs, kills placed
    inside the checkpoint write (including one that writes half a line and dies)
    and inside the compaction's ``os.replace``, and a witness log that knows
    nothing about the format. Everything in that log is a turn ``record_usage``
    returned from, so the folded totals must be at least as large as the log,
    and no larger than the log plus the one turn each writer could have had in
    flight.
    """

    ROUNDS = 2
    #: tag -> mode. Four of the five record usage; the fifth only compacts.
    WRITERS = (("A", "killckpt"), ("B", "usage"), ("C", "killcompact"),
               ("D", "usage"), ("E", "rename"))

    def _script(self, name, body):
        path = pathlib.Path(self.tmp.name) / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "needs SIGKILL")
    @unittest.skipUnless(hasattr(os, "fork") or sys.platform != "win32", "POSIX only")
    def test_sigkill_around_a_checkpoint_loses_no_confirmed_usage(self):
        script = self._script("usage_killable.py", USAGE_KILLABLE)
        rnd = random.Random(20250818)
        for rd in range(self.ROUNDS):
            with self.subTest(round=rd):
                self.one_round(script, rnd)

    def one_round(self, script, rnd):
        meta = self.make(title="fixed")
        witness = str(pathlib.Path(self.tmp.name) / ("usage-%s.log" % meta.id))
        pathlib.Path(witness).write_bytes(b"")
        procs = [
            subprocess.Popen([sys.executable, script, REPO_ROOT, str(self.root),
                              meta.id, tag, witness, mode, str(rnd.randrange(1 << 30))],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for tag, mode in self.WRITERS
        ]
        time.sleep(rnd.uniform(0.6, 0.9))
        for proc in procs:
            with contextlib.suppress(ProcessLookupError):
                os.kill(proc.pid, signal.SIGKILL)
        for proc in procs:
            proc.communicate(timeout=60)

        tags = {tag for tag, mode in self.WRITERS if mode != "rename"}
        log = pathlib.Path(witness).read_text(encoding="utf-8", errors="replace")
        confirmed = sum(1 for line in log.split("\n") if line.strip() in tags)
        self.assertGreater(confirmed, 20, "the writers never got going")
        in_flight = len(tags)          # at most one unconfirmed turn per writer

        loaded = self.quiet_load(meta.id)[0]
        observed = self.totals(loaded)
        for name, value in zip(store_mod._USAGE_KEYS, observed):
            self.assertGreaterEqual(
                value, confirmed,
                "%s lost %d of %d confirmed turns" % (name, confirmed - value, confirmed))
            self.assertLessEqual(value, confirmed + in_flight,
                                 "%s counted turns nobody confirmed" % name)
        self.assertGreaterEqual(round(loaded.cost_usd, 6), round(confirmed * 0.01, 6))
        self.assertLessEqual(loaded.cost_usd, (confirmed + in_flight) * 0.01 + 1e-9)

        # Folding the survivors into the header must not change them, and the
        # listing must agree with the file.
        self.store.update(meta.id, pinned=True)
        self.assertEqual(self.totals(self.quiet_load(meta.id)[0]), observed,
                         "a compaction moved the totals it was folding")
        row = [m for m in Store(self.root).list() if m.id == meta.id][0]
        self.assertEqual(self.totals(row), observed, "list() disagreed with the file")

        # And the session still works.
        self.store.record_usage(meta.id, {"input_tokens": 1}, 0.0)
        self.assertEqual(self.quiet_load(meta.id)[0].input_tokens, observed[0] + 1)
        self.store.delete(meta.id)


# --------------------------------------------------- totals only ever go up


class TestTotalsNeverGoBackwards(UsageBase):
    """A checkpoint below the header is always wrong, so it never wins.

    Compaction is the only thing that writes totals into a header, and it only
    ever folds checkpoints *in*, so a file's true totals can never be below what
    its header says. Believing a lower checkpoint was silent and permanent: the
    next compaction baked the smaller number in and deleted the checkpoint it
    came from.
    """

    def build(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.record(meta.id, 50)
        self.store.update(meta.id, pinned=True)      # fold the history in
        return meta

    def test_a_single_flipped_digit_cannot_erase_the_history(self):
        meta = self.build()
        high = self.totals(self.store.load(meta.id)[0])
        self.assertEqual(high[:4], (5000, 1000, 150, 350))
        self.checkpoint(meta.id, input_tokens=1, output_tokens=1, cache_read_tokens=1,
                        cache_write_tokens=1, cost_usd=0.01)
        self.assertEqual(self.totals(self.quiet_load(meta.id)[0]), high,
                         "one bad checkpoint erased the token and cost history")
        self.store.update(meta.id, pinned=False)     # ... irrecoverably, if believed
        self.assertEqual(self.totals(self.store.load(meta.id)[0]), high)
        row = [m for m in Store(self.root).list() if m.id == meta.id][0]
        self.assertEqual(self.totals(row), high)

    def test_the_next_turn_counts_on_from_the_true_total(self):
        meta = self.build()
        self.checkpoint(meta.id, input_tokens=1, cost_usd=0.01)
        out = self.store.record_usage(meta.id, {"input_tokens": 100}, 0.01)
        self.assertEqual(out.input_tokens, 5100)
        self.assertAlmostEqual(out.cost_usd, 0.51, places=6)
        self.assertEqual(self.store.load(meta.id)[0].input_tokens, 5100)

    def test_a_stale_checkpoint_left_by_an_older_build_does_not_win(self):
        # A mixed-version store: the old build rewrote the header on every turn
        # and knew nothing about checkpoints, so one of ours can survive a
        # rewrite and then claim a total from long before it.
        meta = self.make(title="t")
        self.record(meta.id, 3)
        raw = self.raw(meta.id).split(b"\n")
        header = json.loads(raw[0].decode("utf-8"))
        header.update({"input_tokens": 9000, "output_tokens": 900, "cost_usd": 9.0})
        stale = [l for l in raw if l and b'"type":"usage"' in l][0]
        self.write_raw(meta.id, json.dumps(header).encode("utf-8") + b"\n" + stale + b"\n")
        loaded = self.store.load(meta.id)[0]
        self.assertEqual((loaded.input_tokens, loaded.output_tokens), (9000, 900))
        self.assertAlmostEqual(loaded.cost_usd, 9.0)

    def test_an_explicit_update_can_still_lower_a_total(self):
        # The floor is over checkpoints, not over the user: update() writes the
        # header and drops the checkpoints, so it stays authoritative.
        meta = self.build()
        self.store.update(meta.id, input_tokens=7, cost_usd=0.07)
        loaded = self.store.load(meta.id)[0]
        self.assertEqual(loaded.input_tokens, 7)
        self.assertAlmostEqual(loaded.cost_usd, 0.07)


# --------------------------------------------- the index cannot invent totals


class TestIndexTotalsAreNotTrusted(UsageBase):
    """``list()`` derives the totals it shows; it does not take them on trust.

    The cached ``count`` has been bounded against the file size since it was
    first cached. The money figure was bounded by nothing at all, so a
    well-formed row — one bad write, one stale entry, one edit — made ``list()``
    report a number ``load()`` disagreed with, in either direction, with nothing
    to notice it by.
    """

    def forge(self, sid, **fields):
        doc = json.loads(self.store.index_path.read_text(encoding="utf-8"))
        doc["entries"][sid].update(fields)
        self.store.index_path.write_text(json.dumps(doc), encoding="utf-8")

    def listed(self, sid):
        return [m for m in Store(self.root).list() if m.id == sid][0]

    def test_a_forged_row_cannot_inflate_what_a_listing_reports(self):
        meta = self.make(title="Real Title")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.record(meta.id, 5)
        self.store.list()                       # warm the cache
        self.forge(meta.id, usage=[999999, 888888, 777777, 666666, 12345.678])
        truth = self.totals(self.store.load(meta.id)[0])
        self.assertEqual(truth[:4], (500, 100, 15, 35))
        self.assertEqual(self.totals(self.listed(meta.id)), truth,
                         "the index invented a token and cost history")

    def test_a_zeroed_row_cannot_hide_a_folded_history(self):
        meta = self.make(title="Real Title")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.record(meta.id, 5)
        self.store.update(meta.id, pinned=True)  # totals now live in the header
        self.store.list()
        self.forge(meta.id, usage=[0, 0, 0, 0, 0.0])
        self.assertEqual(self.totals(self.listed(meta.id))[:4], (500, 100, 15, 35))

    def test_a_row_below_the_header_is_distrusted_out_of_reach_of_the_tail(self):
        # The cheap check is a tail read; a message bigger than the tail window
        # puts the newest checkpoint out of its reach. What is left is the
        # invariant: a cached row can only ever be a fold of checkpoints over
        # the header, so a row below the header is stale or forged.
        meta = self.make(title="Real Title")
        self.record(meta.id, 5)
        self.store.update(meta.id, pinned=True)
        self.store.append(meta.id, Message(
            role="assistant", content="y" * (store_mod._INDEX_TAIL * 2)))
        self.store.list()
        self.forge(meta.id, usage=[0, 0, 0, 0, 0.0])
        row = self.listed(meta.id)
        self.assertEqual(self.totals(row)[:4], (500, 100, 15, 35))
        self.assertEqual(row.message_count, 1)

    def test_the_cheap_path_is_still_cheap(self):
        meta = self.make(title="Real Title")
        self.store.append(meta.id, Message(role="user", content="q"))
        self.record(meta.id, 2)
        self.store.list()
        with counting_full_passes() as passes:
            row = self.listed(meta.id)
        self.assertEqual(passes.count, 0, "list() re-read a session it had cached")
        self.assertEqual(row.input_tokens, 200)


# ------------------------------------------- compaction keeps every message


class TestCompactionKeepsEveryMessage(Base):
    """A rewrite drops the header and the checkpoints. Nothing else."""

    def headers_in(self, sid):
        return sum(1 for line in self.raw(sid).split(b"\n")
                   if store_mod._meta_record(line) is not None)

    def test_a_message_on_line_one_survives_a_compaction(self):
        # Line 1 is *usually* the header. A file where it is not — a crash
        # between two writers, a hand-repaired transcript — lost a real message
        # to every rename, with no warning and no rescue copy, because as far as
        # the rewrite was concerned the header was perfectly readable.
        meta = self.make(title="t")
        for i in range(3):
            self.store.append(meta.id, Message(role="user", content="KEEPME-%d" % i))
        lines = [l for l in self.raw(meta.id).split(b"\n") if l]
        self.write_raw(meta.id, b"\n".join([lines[1], lines[0]] + lines[2:]) + b"\n")
        before = [m.content for m in self.quiet_load(meta.id)[1]]
        self.assertEqual(before, ["KEEPME-0", "KEEPME-1", "KEEPME-2"])

        self.store.rename(meta.id, "renamed")
        self.assertEqual([m.content for m in self.quiet_load(meta.id)[1]], before,
                         "a compaction deleted a message that was on line 1")
        self.assertEqual(self.store.load(meta.id)[0].title, "renamed")
        self.assertEqual(self.headers_in(meta.id), 1,
                         "the rewrite left two headers in the file")

    def test_a_headerless_transcript_keeps_its_only_message(self):
        meta = self.make(title="t")
        # No header, and no newline either: a crash mid-write of the first line.
        self.write_raw(meta.id, b'{"type":"message","role":"user","content":"only line",'
                                b'"ts":1,"id":"aaaa"}')
        with self.assertWarns(StoreWarning):
            self.store.rename(meta.id, "recovered")
        self.assertEqual([m.content for m in self.quiet_load(meta.id)[1]], ["only line"])
        self.assertEqual(self.headers_in(meta.id), 1)

    def test_an_unterminated_last_line_is_still_carried_across(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="whole"))
        with open(str(self.path(meta.id)), "ab") as fh:
            fh.write(b'{"type":"message","role":"user","content":"cut off')
        self.store.rename(meta.id, "renamed")
        self.assertIn(b'"content":"cut off', self.raw(meta.id),
                      "a rewrite threw away the bytes a crash left behind")
        self.assertEqual([m.content for m in self.quiet_load(meta.id)[1]], ["whole"])

    def test_a_record_this_build_does_not_understand_is_kept(self):
        meta = self.make(title="t")
        self.store.append(meta.id, Message(role="user", content="q"))
        with open(str(self.path(meta.id)), "ab") as fh:
            fh.write(b'{"type":"annotation","by":"a later build","note":"keep me"}\n')
        self.store.rename(meta.id, "renamed")
        self.assertIn(b'"note":"keep me"', self.raw(meta.id))


# ------------------------------------------------ rescue names running out


class TestRescueNamesRunOut(Base):
    """A hundredth damage event must not end the session.

    ``_meta_for_rewrite`` will not rewrite a header it could not copy aside
    first — rightly — so when the rescue names ran out, ``update``, ``rename``,
    pinning and re-modelling all began raising a ``KeyError`` that reads like a
    missing session, for ever, while ``load`` and ``append`` carried on working.
    """

    def damaged(self):
        meta = self.make(title="t")
        self.write_raw(meta.id, b'{"type":"message","role":"user","content":"orphan",'
                                b'"ts":1,"id":"aaaa"}\n')
        return meta

    def fill_rescue_names(self, meta):
        base = str(self.path(meta.id))
        for i, name in enumerate([base + ".bak"] + ["%s.bak.%d" % (base, n)
                                                    for n in range(1, 100)]):
            pathlib.Path(name).write_bytes(b"an older rescue" + b"x" * (200 + i))

    def test_a_hundred_and_first_damage_event_can_still_be_renamed(self):
        meta = self.damaged()
        original = self.raw(meta.id)
        self.fill_rescue_names(meta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoreWarning)
            self.store.rename(meta.id, "still renameable")
        self.assertEqual(self.quiet_load(meta.id)[0].title, "still renameable")
        self.assertEqual([m.content for m in self.quiet_load(meta.id)[1]], ["orphan"])
        kept = [n for n in os.listdir(str(self.store.sessions_dir))
                if (self.store.sessions_dir / n).is_file()
                and (self.store.sessions_dir / n).read_bytes() == original
                and n != os.path.basename(str(self.path(meta.id)))]
        self.assertTrue(kept, "the damaged transcript was replaced with no rescue copy")
        # A rescue copy is a full plaintext transcript, whatever it is called.
        self.store.delete(meta.id)
        left = [n for n in os.listdir(str(self.store.sessions_dir)) if ".bak" in n]
        self.assertEqual(left, [], "delete() left rescue copies behind")

    def test_a_rescue_does_not_hash_the_transcript_once_per_old_copy(self):
        # Under the session lock, on a file that may be hundreds of megabytes.
        meta = self.damaged()
        self.fill_rescue_names(meta)
        digests = []
        real = store_mod._file_digest

        def counted(path):
            digests.append(str(path))
            return real(path)

        with mock.patch.object(store_mod, "_file_digest", counted):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", StoreWarning)
                self.store.rename(meta.id, "renamed")
        self.assertLess(len(digests), 5,
                        "hashed %d files to find a free rescue name" % len(digests))


# ------------------------------------------------------------ export memory


class TestExportMemory(Base):
    def test_export_does_not_hold_three_copies_of_the_transcript(self):
        # export() built the whole document as a list of pieces and joined it,
        # on top of the messages load() had already parsed: 610 MB of resident
        # memory to write out a 200 MB session. tracemalloc rather than RSS —
        # a high-water mark never comes back down, so it answers a different
        # question.
        import tracemalloc
        meta = self.make(title="t", system="be brief")
        blob = "y" * 100_000
        for i in range(80):                       # 8 MB
            self.store.append(meta.id, Message(role="assistant", content=blob))
        size = self.path(meta.id).stat().st_size
        self.assertGreater(size, 7_500_000)
        for fmt in ("text", "markdown", "json"):
            with self.subTest(fmt=fmt):
                tracemalloc.start()
                try:
                    out = self.store.export(meta.id, fmt)
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertIn(blob, out)
                self.assertLess(peak, size * 2.5,
                                "exporting %d MB as %s held %d MB at once"
                                % (size // 10 ** 6, fmt, peak // 10 ** 6))

    def test_export_reads_the_messages_it_needs_but_never_keeps_them(self):
        meta = self.make(title="t")
        for i in range(3):
            self.store.append(meta.id, Message(role="user", content="m%d" % i))
        with mock.patch.object(store_mod, "_read_records",
                               side_effect=AssertionError("parsed the file whole")):
            out = self.store.export(meta.id, "text")
        self.assertIn("m2", out)


# -------------------------------------------- one session, one lock, one key


class TestLockIdentity(Base):
    def test_two_names_for_one_sessions_directory_are_one_lock(self):
        # $LUME_HOME reached through a symlink, or /var vs /private/var. The
        # .lock sidecar is the same inode either way, so keying the table by an
        # unresolved path put two open file descriptions on it: the second
        # acquire blocked on the first for ever instead of being refused, and a
        # lock that had degraded to in-process exclusion excluded nobody.
        link = pathlib.Path(self.tmp.name) / "link-to-data"
        try:
            os.symlink(str(self.root), str(link), target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):  # pragma: no cover
            self.skipTest("cannot create a symlink here")
        meta = self.make(title="t")
        other = Store(link)
        other.append(meta.id, Message(role="user", content="through the link"))
        self.assertEqual([m.content for m in self.store.load(meta.id)[1]],
                         ["through the link"])
        first, second = self.store.path_for(meta.id), other.path_for(meta.id)
        self.assertNotEqual(str(first), str(second))

        outcome = []

        def nest():
            try:
                with store_mod._file_lock(first):
                    with store_mod._file_lock(second):
                        outcome.append("nested")
            except BaseException as exc:        # pragma: no cover - the fix
                outcome.append(exc)

        worker = threading.Thread(target=nest, daemon=True)
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive(),
                         "a second name for one session deadlocked against the first")
        self.assertIsInstance(outcome[0], RuntimeError)



if __name__ == "__main__":
    unittest.main()
