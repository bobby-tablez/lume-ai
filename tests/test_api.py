"""Tests for lume.api — canned transports and a loopback http.server only.

Nothing here talks to api.anthropic.com; there is no key in this environment and
a real call would cost money. The default base_url is never used.
"""

import datetime
import json
import random
import re
import socket
import sys
import traceback
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from lume.api import (  # noqa: E402
    API_VERSION, CACHE_MIN_TOKENS, DEFAULT_MODEL, EFFORTS, MAX_CACHE_BREAKPOINTS, MODELS,
    APIError, AuthError, BadRequestError, CancelledError, Client, HTTPTransport,
    ModelSpec, NetworkError, OverloadedError, RateLimitError, SDKTransport,
    ServerError, SSEDecoder, StreamEvent, Usage,
    find_api_key, is_oauth_token, redact, resolve_model,
)
from lume.api import (  # noqa: E402
    _count_breakpoints, _estimate_tokens, _HTTPStream, _parse_retry_after,
    _transport_error,
)

KEY = "sk-ant-api03-TESTKEY-not-a-real-credential-0000"
OAUTH = "sk-ant-oat01-TESTTOKEN-not-a-real-credential-0000"


# --------------------------------------------------------------------------- helpers


def sse(records, eol="\n"):
    """Render (event_name, payload dict) pairs as an SSE byte stream."""
    out = []
    for name, payload in records:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        out.append(f"event: {name}{eol}")
        for line in body.split("\n"):
            out.append(f"data: {line}{eol}")
        out.append(eol)
    return "".join(out).encode("utf-8")


def message_start(model="claude-opus-5", usage=None):
    return ("message_start", {
        "type": "message_start",
        "message": {"id": "msg_01TEST", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "usage": usage or {"input_tokens": 1200, "output_tokens": 1,
                                       "cache_creation_input_tokens": 0,
                                       "cache_read_input_tokens": 0}},
    })


def text_stream(pieces=("Hel", "lo, ", "wörld…"), stop_reason="end_turn",
                out_tokens=42, thinking=("Let me think. ",)):
    records = [message_start(), ("ping", {"type": "ping"})]
    if thinking:
        records.append(("content_block_start", {"type": "content_block_start", "index": 0,
                                                "content_block": {"type": "thinking",
                                                                  "thinking": ""}}))
        for t in thinking:
            records.append(("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                    "delta": {"type": "thinking_delta",
                                                              "thinking": t}}))
        records.append(("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "signature_delta",
                                                          "signature": "abc=="}}))
        records.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
    idx = 1 if thinking else 0
    records.append(("content_block_start", {"type": "content_block_start", "index": idx,
                                            "content_block": {"type": "text", "text": ""}}))
    for p in pieces:
        records.append(("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                "delta": {"type": "text_delta", "text": p}}))
    records.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
    records.append(("message_delta", {"type": "message_delta",
                                      "delta": {"stop_reason": stop_reason,
                                                "stop_sequence": None},
                                      "usage": {"output_tokens": out_tokens}}))
    records.append(("message_stop", {"type": "message_stop"}))
    return records


class FakeStream:
    """A canned response: iterating yields the scripted chunks."""

    def __init__(self, status=200, reason="OK", headers=None, chunks=(), body=b""):
        self.status = status
        self.reason = reason
        self.headers = dict(headers or {})
        self._chunks = list(chunks) if chunks else ([body] if body else [])
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)

    def close(self):
        self.closed = True


class FakeTransport:
    """Injectable transport. `script` is a list of callables or FakeStream factories."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.streams = []
        self.closed = False

    def open(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers),
                           "body": json.loads(body.decode("utf-8")), "timeout": timeout})
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        stream = item() if callable(item) else item
        if isinstance(stream, BaseException):
            raise stream
        self.streams.append(stream)
        return stream

    def close(self):
        self.closed = True


def ok_transport(chunk_size=None, records=None):
    raw = sse(records if records is not None else text_stream())
    if chunk_size:
        chunks = [raw[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]
    else:
        chunks = [raw]
    return FakeTransport([lambda: FakeStream(chunks=list(chunks))])


def client(transport, **kw):
    kw.setdefault("base_url", "http://127.0.0.1:1/never-used")
    c = Client(KEY, transport=transport, **kw)
    c.backoff_base = 0.0
    c.backoff_max = 0.0
    c._sleep = lambda d: None
    return c


def drain(events):
    text = "".join(e.text for e in events if e.kind == "text")
    thinking = "".join(e.text for e in events if e.kind == "thinking")
    return text, thinking


# ---------------------------------------------------------------------------- models


class TestModels(unittest.TestCase):
    def test_default_is_opus_5(self):
        self.assertEqual(DEFAULT_MODEL, "claude-opus-5")
        self.assertIn("claude-opus-5", MODELS)

    def test_aliases(self):
        for alias, expect in [("opus", "claude-opus-5"), ("sonnet", "claude-sonnet-5"),
                              ("haiku", "claude-haiku-4-5"), ("fable", "claude-fable-5"),
                              ("OPUS", "claude-opus-5"), (" Sonnet ", "claude-sonnet-5"),
                              ("sonnet-4.6", "claude-sonnet-4-6"),
                              ("claude-opus-4.8", "claude-opus-4-8"),
                              ("claude-sonnet-5", "claude-sonnet-5")]:
            self.assertEqual(resolve_model(alias).id, expect, alias)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            resolve_model("gpt-9")

    def test_spec_passthrough_and_prices(self):
        spec = resolve_model("opus")
        self.assertIs(resolve_model(spec), spec)
        self.assertIsInstance(spec, ModelSpec)
        self.assertEqual((spec.price_in, spec.price_out), (5.0, 25.0))
        self.assertEqual(spec.price_cache_write, 6.25)
        self.assertEqual(spec.price_cache_read, 0.5)
        self.assertFalse(spec.supports_temperature)
        self.assertEqual(spec.thinking, "adaptive")

    def test_price_table(self):
        want = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (3.0, 15.0),
                "claude-haiku-4-5": (1.0, 5.0), "claude-fable-5": (10.0, 50.0)}
        for mid, (pin, pout) in want.items():
            spec = MODELS[mid]
            self.assertEqual((spec.price_in, spec.price_out), (pin, pout), mid)
            self.assertAlmostEqual(spec.price_cache_write, pin * 1.25)
            self.assertAlmostEqual(spec.price_cache_read, pin * 0.1)


class TestUsage(unittest.TestCase):
    def test_cost_arithmetic(self):
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(u.cost("claude-opus-5"), 30.0)
        self.assertAlmostEqual(u.cost("haiku"), 6.0)
        self.assertAlmostEqual(u.cost("fable"), 60.0)
        # sonnet-5 is on introductory pricing until 2026-08-31; both sides of the
        # expiry are pinned so neither number can quietly become the other's.
        self.assertAlmostEqual(u.cost("sonnet", when="2026-08-17"), 12.0)
        self.assertAlmostEqual(u.cost("sonnet", when="2026-09-01"), 18.0)

    def test_cache_pricing(self):
        u = Usage(cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
        self.assertAlmostEqual(u.cost("opus"), 6.25 + 0.5)

    def test_small_numbers(self):
        u = Usage(input_tokens=1500, output_tokens=300, cache_read_input_tokens=10_000)
        expect = (1500 * 5 + 300 * 25 + 10_000 * 0.5) / 1e6
        self.assertAlmostEqual(u.cost("opus"), expect)

    def test_add_and_sum(self):
        a = Usage(1, 2, 3, 4)
        b = Usage(10, 20, 30, 40)
        self.assertEqual(a + b, Usage(11, 22, 33, 44))
        self.assertEqual(sum([a, b], Usage()), Usage(11, 22, 33, 44))
        self.assertEqual((a + b).total_tokens, 110)
        with self.assertRaises(TypeError):
            a + 3

    def test_total_tokens_counts_every_bucket(self):
        u = Usage(input_tokens=1, output_tokens=2,
                  cache_creation_input_tokens=4, cache_read_input_tokens=8)
        self.assertEqual(u.total_tokens, 15)
        self.assertEqual(Usage().total_tokens, 0)

    def test_as_dict_is_the_wire_shape(self):
        u = Usage(input_tokens=3, output_tokens=4,
                  cache_creation_input_tokens=5, cache_read_input_tokens=6)
        self.assertEqual(u.as_dict(), {"input_tokens": 3, "output_tokens": 4,
                                       "cache_creation_input_tokens": 5,
                                       "cache_read_input_tokens": 6})
        self.assertEqual(Usage.from_dict(u.as_dict()), u)
        self.assertEqual(json.loads(json.dumps(Usage().as_dict())), Usage().as_dict())

    def test_fallback_iterations_are_kept(self):
        u = Usage.from_dict({"input_tokens": 1,
                             "iterations": [{"type": "refusal_message"},
                                            {"type": "fallback_message"}]})
        self.assertTrue(u.served_by_fallback)
        self.assertEqual(u.as_dict()["iterations"][1], {"type": "fallback_message"})
        self.assertFalse(Usage(input_tokens=1).served_by_fallback)
        self.assertNotIn("iterations", Usage(input_tokens=1).as_dict())

    def test_from_dict_tolerates_junk(self):
        u = Usage.from_dict({"input_tokens": 5, "service_tier": "standard", "x": None})
        self.assertEqual(u, Usage(input_tokens=5))
        self.assertEqual(Usage.from_dict(None), Usage())
        self.assertEqual(Usage.from_dict({}).as_dict()["output_tokens"], 0)


# -------------------------------------------------------------------------- sse parse


FUZZ_RECORDS = [
    message_start(),
    ("ping", {"type": "ping"}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "héllo 🌊 “quoted”"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "\nsecond line"}}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                       "usage": {"output_tokens": 9}}),
    ("message_stop", {"type": "message_stop"}),
]


class TestRetryAfterParsing(unittest.TestCase):
    def test_seconds_and_dates(self):
        import email.utils
        self.assertEqual(_parse_retry_after("2.5"), 2.5)
        self.assertEqual(_parse_retry_after(" 30 "), 30.0)
        self.assertEqual(_parse_retry_after("-5"), 0.0)
        when = email.utils.formatdate(time.time() + 3, usegmt=True)
        self.assertTrue(0 < _parse_retry_after(when) <= 4)

    def test_garbage_is_none_and_the_past_is_zero(self):
        import email.utils
        for junk in ("soon", "", "  ", "later, maybe", "NaN-ish!!"):
            self.assertIsNone(_parse_retry_after(junk), junk)
        self.assertIsNone(_parse_retry_after(None))
        past = email.utils.formatdate(time.time() - 600, usegmt=True)
        self.assertEqual(_parse_retry_after(past), 0.0)


class TestSSEDecoder(unittest.TestCase):
    def parse_all(self, data, chunker):
        dec = SSEDecoder()
        out = []
        for chunk in chunker(data):
            out.extend(dec.feed(chunk))
        out.extend(dec.close())
        return out

    def test_basic(self):
        raw = b"event: ping\ndata: {}\n\n"
        self.assertEqual(SSEDecoder().feed(raw), [("ping", "{}")])

    def test_crlf_and_lone_cr(self):
        for eol in ("\n", "\r\n", "\r"):
            # The trailing comment line keeps the record's blank line unambiguous:
            # a lone CR at the very end of a stream is held back (it may be half a
            # CRLF) and then discarded, never guessed at.
            raw = f"event: ping{eol}data: 1{eol}{eol}:bye{eol}".encode()
            dec = SSEDecoder()
            got = dec.feed(raw) + dec.close()
            self.assertEqual(got, [("ping", "1")], repr(eol))

    def test_crlf_split_across_chunks(self):
        dec = SSEDecoder()
        self.assertEqual(dec.feed(b"event: ping\r"), [])
        self.assertEqual(dec.feed(b"\ndata: 1\r\n\r"), [])
        self.assertEqual(dec.feed(b"\n"), [("ping", "1")])

    def test_multiline_data_joined_with_newline(self):
        raw = b"event: e\ndata: {\ndata:  \"a\": 1}\n\n"
        self.assertEqual(SSEDecoder().feed(raw), [("e", '{\n "a": 1}')])

    def test_comments_and_keepalives_ignored(self):
        raw = b": ping keepalive\n\n: another\nevent: x\ndata: 1\n\n"
        self.assertEqual(SSEDecoder().feed(raw), [("x", "1")])

    def test_blank_line_without_data_dispatches_nothing(self):
        self.assertEqual(SSEDecoder().feed(b"\n\n\nevent: x\n\n"), [])

    def test_default_event_name(self):
        self.assertEqual(SSEDecoder().feed(b"data: hi\n\n"), [("message", "hi")])

    def test_field_without_colon_and_no_space(self):
        self.assertEqual(SSEDecoder().feed(b"event:x\ndata:1\nid\n\n"), [("x", "1")])

    def test_bom_stripped(self):
        self.assertEqual(SSEDecoder().feed("﻿event: x\ndata: 1\n\n".encode()),
                         [("x", "1")])

    def test_utf8_split_across_chunks(self):
        raw = "data: 🌊é\n\n".encode()
        dec = SSEDecoder()
        out = []
        for i in range(len(raw)):
            out.extend(dec.feed(raw[i:i + 1]))
        self.assertEqual(out, [("message", "🌊é")])

    def test_unterminated_record_is_discarded_on_close(self):
        # A record whose terminating blank line never arrived is NOT an event: the
        # stream was cut inside it, and pretending otherwise hands the caller a
        # truncated answer labelled complete.
        dec = SSEDecoder()
        self.assertEqual(dec.feed(b"event: x\ndata: 1\n"), [])
        self.assertEqual(dec.close(), [])

    def test_record_cut_inside_its_terminator_is_discarded(self):
        raw = sse([("message_stop", {"type": "message_stop"})])
        dec = SSEDecoder()
        self.assertEqual(dec.feed(raw[:-1]) + dec.close(), [])
        dec = SSEDecoder()
        self.assertEqual(dec.feed(raw) + dec.close(),
                         [("message_stop", '{"type": "message_stop"}')])

    def test_str_and_bytes_chunks_may_be_mixed(self):
        dec = SSEDecoder()
        self.assertEqual(dec.feed("data: hé\n") + dec.feed(b"\n"),
                         [("message", "hé")])
        # A str chunk must go through the same incremental decoder as the bytes
        # around it. Bypassing it strands the lead byte pending here, which then
        # corrupts a later chunk — so the mixed feed must agree with the pure-bytes
        # feed of the very same bytes.
        mixed = SSEDecoder()
        self.assertEqual(mixed.feed(b"data: h\xc3"), [])
        pure = SSEDecoder()
        self.assertEqual(pure.feed(b"data: h\xc3"), [])
        self.assertEqual(mixed.feed("x\n\n"), pure.feed(b"x\n\n"))
        self.assertEqual(mixed.feed(b"data: 2\n\n"), pure.feed(b"data: 2\n\n"))

    def test_every_single_split_point(self):
        raw = sse(FUZZ_RECORDS).replace(b"event: ping", b": keep-alive\n\nevent: ping", 1)
        expect = self.parse_all(raw, lambda d: [d])
        self.assertEqual(len(expect), 7)  # the comment record carries no data
        for i in range(len(raw) + 1):
            got = self.parse_all(raw, lambda d, i=i: [d[:i], d[i:]])
            self.assertEqual(got, expect, f"split at byte {i}")

    def test_one_byte_at_a_time(self):
        raw = sse(FUZZ_RECORDS, eol="\r\n").replace(b"data: {", b": comment\r\ndata: {", 1)
        expect = self.parse_all(raw, lambda d: [d])
        got = self.parse_all(raw, lambda d: [d[i:i + 1] for i in range(len(d))])
        self.assertEqual(got, expect)

    def test_random_chunkings(self):
        rng = random.Random(1234)
        raw = sse(FUZZ_RECORDS, eol="\r\n")
        expect = self.parse_all(raw, lambda d: [d])
        for _ in range(200):
            def chunker(d, rng=rng):
                out, i = [], 0
                while i < len(d):
                    n = rng.randint(1, 9)
                    out.append(d[i:i + n])
                    i += n
                return out
            self.assertEqual(self.parse_all(raw, chunker), expect)

    def test_mixed_line_endings_in_one_stream(self):
        raw = (b"event: a\r\ndata: 1\r\n\r\n"
               b"event: b\ndata: 2\n\n"
               b"event: c\rdata: 3\r\r")
        dec = SSEDecoder()
        # The third record's terminator is a lone CR at the very end. `feed` holds
        # it back — it could be the first half of a CRLF still in flight — but at
        # end of stream nothing more is coming, so it terminates the line and the
        # record is complete. Discarding it lost the `message_stop` of a whole
        # answer and reported a broken stream instead (WHATWG § 9.2.6).
        self.assertEqual(dec.feed(raw), [("a", "1"), ("b", "2")])
        self.assertEqual(dec.close(), [("c", "3")])
        dec = SSEDecoder()
        self.assertEqual(dec.feed(raw + b":\n") + dec.close(),
                         [("a", "1"), ("b", "2"), ("c", "3")])


# ------------------------------------------------------------------------- streaming


class TestStream(unittest.TestCase):
    def test_happy_path(self):
        t = ok_transport()
        events = list(client(t).stream(model="opus", messages=[{"role": "user",
                                                               "content": "hi"}]))
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "done")
        self.assertIn("ping", kinds)
        text, thinking = drain(events)
        self.assertEqual(text, "Hello, wörld…")
        self.assertEqual(thinking, "Let me think. ")
        done = events[-1]
        self.assertEqual(done.stop_reason, "end_turn")
        self.assertEqual(done.model, "claude-opus-5")
        self.assertEqual(done.usage.input_tokens, 1200)
        self.assertEqual(done.usage.output_tokens, 42)
        self.assertAlmostEqual(done.usage.cost("opus"), (1200 * 5 + 42 * 25) / 1e6)

    def test_usage_is_cumulative_not_additive(self):
        records = text_stream(out_tokens=7)
        records.insert(-1, ("message_delta", {"type": "message_delta", "delta": {},
                                              "usage": {"output_tokens": 11}}))
        events = list(client(ok_transport(records=records)).stream(
            model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(events[-1].usage.output_tokens, 11)
        self.assertEqual(events[-1].usage.input_tokens, 1200)

    def test_chunk_boundaries_never_change_the_text(self):
        want = "Hello, wörld…"
        for size in list(range(1, 20)) + [37, 128, 4096]:
            events = list(client(ok_transport(chunk_size=size)).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(drain(events)[0], want, f"chunk size {size}")
            self.assertEqual(events[-1].kind, "done")

    def test_random_chunkings_over_the_client(self):
        rng = random.Random(99)
        raw = sse(text_stream())
        for _ in range(60):
            chunks, i = [], 0
            while i < len(raw):
                n = rng.randint(1, 11)
                chunks.append(raw[i:i + n])
                i += n
            t = FakeTransport([lambda c=chunks: FakeStream(chunks=list(c))])
            events = list(client(t).stream(model="opus",
                                           messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(drain(events)[0], "Hello, wörld…")

    def test_refusal_is_a_done_event_not_an_exception(self):
        records = text_stream(pieces=(), stop_reason="refusal", thinking=())
        records[-2] = ("message_delta", {"type": "message_delta",
                                         "delta": {"stop_reason": "refusal",
                                                   "stop_details": {"type": "refusal",
                                                                    "category": "cyber"}},
                                         "usage": {"output_tokens": 0}})
        events = list(client(ok_transport(records=records)).stream(
            model="opus", messages=[{"role": "user", "content": "hi"}]))
        done = events[-1]
        self.assertEqual(done.kind, "done")
        self.assertEqual(done.stop_reason, "refusal")
        self.assertEqual(done.stop_details["category"], "cyber")

    def test_fallback_content_block(self):
        records = text_stream(thinking=())
        records.insert(2, ("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "fallback", "from": {"model": "claude-opus-5"},
                              "to": {"model": "claude-opus-4-8"}}}))
        events = list(client(ok_transport(records=records)).stream(
            model="opus", messages=[{"role": "user", "content": "hi"}]))
        fb = [e for e in events if e.kind == "fallback"]
        self.assertEqual(len(fb), 1)
        self.assertEqual(fb[0].model, "claude-opus-4-8")
        self.assertEqual(fb[0].stop_details["from"]["model"], "claude-opus-5")
        self.assertEqual(drain(events)[0], "Hello, wörld…")

    def test_input_json_and_signature_deltas_emit_nothing(self):
        records = [message_start(),
                   ("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "tool_use", "id": "t1",
                                                              "name": "x", "input": {}}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                            "delta": {"type": "input_json_delta",
                                                      "partial_json": '{"a":'}}),
                   ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                   ("message_stop", {"type": "message_stop"})]
        events = list(client(ok_transport(records=records)).stream(
            model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(drain(events), ("", ""))
        self.assertEqual(events[-1].kind, "done")

    def test_unknown_event_types_are_ignored(self):
        records = [message_start(), ("weird_new_event", {"type": "weird_new_event"}),
                   ("message_stop", {"type": "message_stop"})]
        events = list(client(ok_transport(records=records)).stream(
            model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual([e.kind for e in events], ["start", "done"])

    def test_truncated_stream_raises(self):
        raw = sse(text_stream())[:-40]
        t = FakeTransport([lambda: FakeStream(chunks=[raw])])
        with self.assertRaises(NetworkError):
            list(client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}]))

    def test_stream_cut_inside_the_last_terminator_is_not_a_clean_finish(self):
        # One byte short of the final blank line: message_stop never lands, so this
        # is a truncated answer and must be reported as one, not handed over with
        # stop_reason="end_turn" as though the model had finished.
        raw = sse(text_stream())
        t = FakeTransport([lambda: FakeStream(chunks=[raw[:-1]])])
        seen = []
        with self.assertRaises(NetworkError) as ctx:
            for event in client(t, max_retries=0).stream(
                    model="opus", messages=[{"role": "user", "content": "hi"}]):
                seen.append(event)
        self.assertIn("message_stop", ctx.exception.message)
        self.assertEqual([e.kind for e in seen if e.kind == "done"], [])

    def test_fallback_reprices_the_rest_of_the_turn(self):
        # A fable-5 turn rescued by opus-4-8 is billed at opus rates; if the run
        # keeps naming fable the caller reports twice the real cost.
        records = [message_start(model="claude-fable-5"),
                   ("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {
                                                "type": "fallback",
                                                "from": {"model": "claude-fable-5"},
                                                "to": {"model": "claude-opus-4-8"}}}),
                   ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                   ("content_block_start", {"type": "content_block_start", "index": 1,
                                            "content_block": {"type": "text", "text": ""}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                            "delta": {"type": "text_delta", "text": "ok"}}),
                   ("message_delta", {"type": "message_delta",
                                      "delta": {"stop_reason": "end_turn"},
                                      "usage": {"output_tokens": 1000,
                                                "iterations": [
                                                    {"type": "refusal_message"},
                                                    {"type": "fallback_message"}]}}),
                   ("message_stop", {"type": "message_stop"})]
        t = FakeTransport([lambda: FakeStream(chunks=[sse(records)])])
        events = list(client(t, max_retries=0).stream(
            model="fable", messages=[{"role": "user", "content": "hi"}]))
        kinds = {e.kind: e for e in events}
        self.assertEqual(kinds["fallback"].model, "claude-opus-4-8")
        self.assertEqual(kinds["usage"].model, "claude-opus-4-8")
        self.assertEqual(kinds["done"].model, "claude-opus-4-8")
        usage = kinds["done"].usage
        self.assertTrue(usage.served_by_fallback)
        # The usage records who served it, so the turn prices correctly whichever
        # model the caller names — and both agree with opus-4-8's rates.
        self.assertEqual(usage.served_model, "claude-opus-4-8")
        self.assertAlmostEqual(usage.cost(kinds["done"].model),
                               usage.cost("claude-fable-5"))
        bare = Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        self.assertAlmostEqual(usage.cost("claude-fable-5"), bare.cost("claude-opus-4-8"))
        self.assertLess(usage.cost("claude-fable-5"), bare.cost("claude-fable-5"))

    def test_stream_closes_the_connection(self):
        t = ok_transport()
        list(client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertTrue(t.streams[0].closed)

    def test_early_generator_close_does_not_leak(self):
        t = ok_transport(chunk_size=8)
        gen = client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}])
        next(gen)
        gen.close()
        self.assertTrue(t.streams[0].closed)

    def test_client_close_closes_transport(self):
        t = ok_transport()
        c = client(t)
        c.close()
        c.close()
        self.assertTrue(t.closed)

    def test_argument_errors_raise_eagerly(self):
        c = client(ok_transport())
        with self.assertRaises(ValueError):
            c.stream(model="opus", messages=[])
        with self.assertRaises(ValueError):
            c.stream(model="nope", messages=[{"role": "user", "content": "hi"}])
        with self.assertRaises(ValueError):
            c.stream(model="opus", messages=[{"role": "user", "content": "x"}], effort="turbo")
        self.assertEqual(len(ok_transport().calls), 0)

    def test_client_requires_a_key(self):
        with self.assertRaises(ValueError):
            Client("")


# ---------------------------------------------------------------------------- retries


def err_body(kind="rate_limit_error", message="slow down"):
    return json.dumps({"type": "error", "error": {"type": kind, "message": message},
                       "request_id": "req_123"}).encode()


class TestRetries(unittest.TestCase):
    def test_429_then_success(self):
        t = FakeTransport([
            lambda: FakeStream(status=429, reason="Too Many Requests",
                               headers={"retry-after": "0", "request-id": "req_1"},
                               body=err_body()),
            lambda: FakeStream(chunks=[sse(text_stream())]),
        ])
        c = client(t)
        events = list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(drain(events)[0], "Hello, wörld…")

    def test_retry_after_is_honoured(self):
        slept = []
        t = FakeTransport([
            lambda: FakeStream(status=429, headers={"retry-after": "2.5"}, body=err_body()),
            lambda: FakeStream(chunks=[sse(text_stream())]),
        ])
        c = client(t)
        c._sleep = slept.append
        list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(slept, [2.5])

    def test_retry_after_http_date(self):
        import email.utils
        when = email.utils.formatdate(time.time() + 3, usegmt=True)
        slept = []
        t = FakeTransport([
            lambda: FakeStream(status=429, headers={"Retry-After": when}, body=err_body()),
            lambda: FakeStream(chunks=[sse(text_stream())]),
        ])
        c = client(t)
        c._sleep = slept.append
        list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(slept), 1)
        self.assertTrue(0 < slept[0] <= 4, slept)

    def test_backoff_is_exponential_and_jittered(self):
        slept = []
        t = FakeTransport([lambda: FakeStream(status=503, body=err_body("api_error"))])
        c = client(t, max_retries=4)
        c.backoff_base, c.backoff_max = 0.5, 16.0
        c._sleep = slept.append
        with self.assertRaises(ServerError):
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(slept), 4)
        self.assertEqual(len(t.calls), 5)
        for i, d in enumerate(slept):
            self.assertTrue(0.0 <= d <= 0.5 * 2 ** i, (i, d))
        self.assertGreater(len(set(slept)), 1)  # jitter, not a fixed ladder

    def test_retryable_statuses(self):
        for status, cls in [(429, RateLimitError), (500, ServerError), (502, ServerError),
                            (503, ServerError), (504, ServerError), (529, OverloadedError)]:
            t = FakeTransport([
                lambda s=status: FakeStream(status=s, body=err_body("api_error")),
                lambda: FakeStream(chunks=[sse(text_stream())]),
            ])
            events = list(client(t).stream(model="opus",
                                           messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(len(t.calls), 2, status)
            self.assertEqual(drain(events)[0], "Hello, wörld…")
            self.assertTrue(issubclass(cls, APIError))

    def test_no_retry_on_client_errors(self):
        for status, cls in [(400, BadRequestError), (401, AuthError), (403, AuthError),
                            (404, BadRequestError), (413, BadRequestError)]:
            t = FakeTransport([lambda s=status: FakeStream(
                status=s, body=err_body("invalid_request_error", "bad"))])
            with self.assertRaises(cls) as ctx:
                list(client(t).stream(model="opus",
                                      messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(len(t.calls), 1, status)
            self.assertEqual(ctx.exception.status, status)
            self.assertFalse(ctx.exception.retryable)
            self.assertEqual(ctx.exception.request_id, "req_123")

    def test_connection_error_is_retried(self):
        t = FakeTransport([
            NetworkError("connection refused"),
            lambda: FakeStream(chunks=[sse(text_stream())]),
        ])
        events = list(client(t).stream(model="opus",
                                       messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(drain(events)[0], "Hello, wörld…")

    def test_retryable_sse_error_before_output_is_retried_silently(self):
        first = sse([("error", {"type": "error",
                                "error": {"type": "overloaded_error", "message": "busy"}})])
        t = FakeTransport([
            lambda: FakeStream(chunks=[first]),
            lambda: FakeStream(chunks=[sse(text_stream())]),
        ])
        events = list(client(t).stream(model="opus",
                                       messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 2)
        self.assertEqual([e.kind for e in events if e.kind == "error"], [])
        self.assertEqual(drain(events)[0], "Hello, wörld…")

    def test_no_retry_once_text_has_been_emitted(self):
        partial = sse(text_stream(pieces=("Hel",), thinking=())[:4])
        t = FakeTransport([
            lambda: FakeStream(chunks=[partial]),
            lambda: FakeStream(chunks=[sse(text_stream())]),
        ])
        seen = []
        with self.assertRaises(NetworkError):
            for e in client(t).stream(model="opus",
                                      messages=[{"role": "user", "content": "hi"}]):
                seen.append(e)
        self.assertEqual(len(t.calls), 1, "must not replay a partially delivered turn")
        self.assertEqual("".join(e.text for e in seen if e.kind == "text"), "Hel")

    def test_mid_stream_error_after_output_surfaces_once(self):
        records = text_stream(pieces=("Hel",), thinking=())[:5]
        records.append(("error", {"type": "error",
                                  "error": {"type": "overloaded_error", "message": "busy"}}))
        t = FakeTransport([lambda: FakeStream(chunks=[sse(records)])])
        seen = []
        with self.assertRaises(OverloadedError):
            for e in client(t).stream(model="opus",
                                      messages=[{"role": "user", "content": "hi"}]):
                seen.append(e)
        self.assertEqual(len(t.calls), 1)
        self.assertEqual([e.kind for e in seen].count("error"), 1)
        self.assertEqual("".join(e.text for e in seen if e.kind == "text"), "Hel")

    def test_max_retries_zero(self):
        t = FakeTransport([lambda: FakeStream(status=529, body=err_body("overloaded_error"))])
        with self.assertRaises(OverloadedError):
            list(client(t, max_retries=0).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 1)

    def test_wall_clock_cap_stops_retrying(self):
        t = FakeTransport([lambda: FakeStream(status=429, headers={"retry-after": "30"},
                                              body=err_body())])
        c = client(t, max_retries=10)
        c.max_retry_wall = 45.0
        c._sleep = lambda d: None
        with self.assertRaises(RateLimitError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(ctx.exception.retry_after, 30.0)

    def test_retry_after_beyond_the_budget_says_so(self):
        t = FakeTransport([lambda: FakeStream(status=429, headers={"retry-after": "3600"},
                                              body=err_body()),
                           lambda: FakeStream(chunks=[sse(text_stream())])])
        c = client(t, max_retries=4)
        c.max_retry_wall = 60.0
        slept = []
        c._sleep = slept.append
        with self.assertRaises(RateLimitError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 1)
        self.assertEqual(slept, [], "must not sit through an hour-long retry-after")
        self.assertEqual(ctx.exception.retry_after, 3600.0)
        self.assertIn("retry budget", ctx.exception.message)
        self.assertIn("3600", ctx.exception.message)

    def test_a_plain_transport_exception_becomes_an_api_error(self):
        class Rude:
            def open(self, url, headers, body, timeout):
                raise RuntimeError("upstream rejected headers %r" % (headers,))

        token = "glpat-" "abcdefghijklmnop"
        c = Client(token, base_url="http://127.0.0.1:1", transport=Rude(), max_retries=2)
        c._sleep = lambda d: None
        with self.assertRaises(APIError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        exc = ctx.exception
        self.assertNotIn(token, str(exc))
        self.assertNotIn(token, repr(exc))
        self.assertFalse(exc.retryable, "a broken transport is not worth replaying")
        self.assertIn("RuntimeError", exc.message)

    def test_ping_alone_does_not_block_a_retry(self):
        first = sse([("ping", {"type": "ping"}),
                     ("error", {"type": "error", "error": {"type": "overloaded_error",
                                                           "message": "busy"}})])
        t = FakeTransport([lambda: FakeStream(chunks=[first]),
                           lambda: FakeStream(chunks=[sse(text_stream())])])
        events = list(client(t).stream(model="opus",
                                       messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(drain(events)[0], "Hello, wörld…")


# ----------------------------------------------------------------------- cancellation


class BlockingStream(FakeStream):
    """Yields its scripted chunks, then parks on a read — like a real socket.

    The park ends the way a live connection's really does when another thread
    tears it down mid-read: not with a tidy StopIteration but with the teardown
    race http.client produces, `AttributeError: 'NoneType' object has no attribute
    'close'`, raised from inside the read against a half-detached response. A fake
    that returns cleanly instead only ever tests itself.
    """

    def __init__(self, chunks):
        super().__init__(chunks=chunks)
        self.gate = threading.Event()
        self.blocked = threading.Event()

    def __next__(self):
        if self._chunks and not self.closed:
            return self._chunks.pop(0)
        self.blocked.set()
        if self.gate.wait(10):
            raise AttributeError("'NoneType' object has no attribute 'close'")
        raise AssertionError("blocking read was never interrupted")

    def close(self):
        self.closed = True
        self.gate.set()


class _FakeResponse:
    status = 200
    reason = "OK"
    headers = {}

    def __init__(self, log, exc=None):
        self._log = log
        self._exc = exc

    def read1(self, size):
        if self._exc is not None:
            raise self._exc
        return b""

    def close(self):
        self._log.append("close-response")


class _FakeConn:
    def __init__(self, log, sock=None):
        self._log = log
        self.sock = sock

    def close(self):
        self._log.append("close-conn")


class _FakeSocket:
    def __init__(self, log):
        self._log = log

    def shutdown(self, how):
        self._log.append(("shutdown", how))


class TestHTTPStreamTeardown(unittest.TestCase):
    """The socket-level half of cancellation, without a socket."""

    def test_close_shuts_down_before_closing(self):
        # close() alone does not interrupt a recv() already parked in the C layer;
        # shutdown() does. Order matters, so it is asserted.
        log = []
        stream = _HTTPStream(_FakeConn(log, _FakeSocket(log)), _FakeResponse(log))
        stream.close()
        self.assertEqual(log[0], ("shutdown", socket.SHUT_RDWR))
        self.assertIn("close-response", log)
        self.assertIn("close-conn", log)
        stream.close()  # idempotent
        self.assertEqual(log.count("close-conn"), 1)

    def test_close_survives_a_socketless_connection(self):
        log = []
        stream = _HTTPStream(_FakeConn(log, None), _FakeResponse(log))
        stream.close()  # must not raise
        self.assertIn("close-conn", log)

    def test_teardown_race_under_a_live_read_ends_the_stream(self):
        # The real sequence: the watchdog closes the response while a read is
        # parked, and the read finishes against the half-detached object with
        # AttributeError. Caught, that is a clean end of stream; uncaught it sails
        # past every `except APIError` in the app and kills the TUI.
        log = []
        holder = {}

        class Racing(_FakeResponse):
            def read1(self, size):
                holder["stream"].close()  # another thread, mid-read
                raise AttributeError("'NoneType' object has no attribute 'close'")

        stream = _HTTPStream(_FakeConn(log, _FakeSocket(log)), Racing(log))
        holder["stream"] = stream
        with self.assertRaises(StopIteration):
            next(stream)

    def test_a_read_failure_while_open_is_a_network_error(self):
        log = []
        boom = AttributeError("'NoneType' object has no attribute 'close'")
        stream = _HTTPStream(_FakeConn(log, _FakeSocket(log)), _FakeResponse(log, boom))
        with self.assertRaises(NetworkError):
            next(stream)


class TestCancellation(unittest.TestCase):
    def test_cancel_before_the_request(self):
        t = FakeTransport([lambda: FakeStream(chunks=[sse(text_stream())])])
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            list(client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                                  cancel=cancel))
        self.assertEqual(len(t.calls), 0)

    def test_cancel_mid_stream_is_prompt_and_closes_the_socket(self):
        head = sse(text_stream(pieces=("Hel",), thinking=())[:5])
        stream = BlockingStream([head])
        t = FakeTransport([lambda: stream])
        cancel = threading.Event()
        gen = client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                               cancel=cancel)
        seen, box = [], {}

        def consume():
            try:
                for event in gen:
                    seen.append(event)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                box["exc"] = exc

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        # The reader is now parked on a blocking read, exactly like a live socket.
        self.assertTrue(stream.blocked.wait(5), "did not reach the blocking read")
        started = time.monotonic()
        cancel.set()
        worker.join(5)
        elapsed = time.monotonic() - started
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("exc"), CancelledError)
        self.assertLess(elapsed, 2.0, f"cancellation took {elapsed:.2f}s")
        self.assertTrue(stream.closed)
        self.assertEqual("".join(e.text for e in seen if e.kind == "text"), "Hel")

    def test_cancel_during_backoff(self):
        t = FakeTransport([lambda: FakeStream(status=529, body=err_body("overloaded_error"))])
        c = client(t, max_retries=3)
        c.backoff_base, c.backoff_max = 30.0, 30.0
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        threading.Timer(0.05, cancel.set).start()
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            list(gen)
        self.assertLess(time.monotonic() - started, 3.0)

    def test_cancelled_error_is_not_retryable(self):
        self.assertFalse(CancelledError("x").retryable)


# --------------------------------------------------------------------- request shape


class TestRequestBody(unittest.TestCase):
    def body(self, **kw):
        kw.setdefault("messages", [{"role": "user", "content": "hi"}])
        return Client(KEY).build_body(**kw)

    def test_default_turn(self):
        self.assertEqual(self.body(), {
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32000,
            "stream": True,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"},
            "fallbacks": "default",
        })

    def test_no_budget_tokens_anywhere(self):
        raw = json.dumps(self.body())
        self.assertNotIn("budget_tokens", raw)
        self.assertNotIn("temperature", raw)

    def test_effort_is_nested_not_top_level(self):
        body = self.body(effort="max")
        self.assertEqual(body["output_config"], {"effort": "max"})
        self.assertNotIn("effort", body)

    def test_all_efforts_accepted(self):
        for effort in EFFORTS:
            self.assertEqual(self.body(effort=effort)["output_config"]["effort"], effort)

    def test_thinking_disabled_is_said_out_loud(self):
        # Omitting `thinking` runs adaptive thinking on these models: the user pays
        # for reasoning they turned off and max_tokens silently covers it.
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-6",
                      "claude-sonnet-4-6"):
            body = self.body(model=model, thinking=False)
            self.assertEqual(body["thinking"], {"type": "disabled"}, model)

    def test_thinking_disabled_by_omission_where_that_works(self):
        # On 4.8/4.7 an absent field really is off, and haiku only thinks when it
        # is handed a budget.
        for model in ("claude-opus-4-8", "claude-opus-4-7", "claude-haiku-4-5"):
            self.assertNotIn("thinking", self.body(model=model, thinking=False), model)

    def test_fable_cannot_turn_thinking_off(self):
        # fable-5 always thinks; an explicit {"type": "disabled"} is a 400 at any
        # effort, so the field is left out rather than sent and rejected.
        body = self.body(model="fable", thinking=False)
        self.assertNotIn("thinking", body)
        self.assertEqual(body["output_config"], {"effort": "high"})

    def test_disabled_thinking_clamps_effort_on_opus_5(self):
        # {"type": "disabled"} above effort `high` is a 400 on opus-5.
        for effort in ("xhigh", "max"):
            body = self.body(model="opus", thinking=False, effort=effort)
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(body["output_config"], {"effort": "high"}, effort)
        for effort in ("low", "medium", "high"):
            self.assertEqual(self.body(model="opus", thinking=False,
                                       effort=effort)["output_config"]["effort"], effort)
        # Thinking on, effort untouched.
        self.assertEqual(self.body(model="opus", effort="max")["output_config"],
                         {"effort": "max"})
        # And the clamp is opus-5's rule, not everyone's.
        self.assertEqual(self.body(model="sonnet", thinking=False,
                                   effort="max")["output_config"], {"effort": "max"})

    def test_temperature_dropped_on_models_that_reject_it(self):
        for model in ("claude-opus-5", "claude-fable-5", "claude-sonnet-5",
                      "claude-opus-4-8", "claude-opus-4-7"):
            self.assertNotIn("temperature", self.body(model=model, temperature=0.7), model)

    def test_temperature_sent_where_supported(self):
        for model in ("claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"):
            self.assertEqual(self.body(model=model, temperature=0.2)["temperature"], 0.2)

    def test_haiku_uses_budget_thinking_and_no_effort(self):
        body = self.body(model="haiku", max_tokens=8000)
        self.assertEqual(body["thinking"]["type"], "enabled")
        self.assertLess(body["thinking"]["budget_tokens"], body["max_tokens"])
        self.assertGreaterEqual(body["thinking"]["budget_tokens"], 1024)
        self.assertNotIn("output_config", body)
        self.assertNotIn("fallbacks", body)

    def test_fallbacks_only_on_refusing_models(self):
        self.assertEqual(self.body(model="opus")["fallbacks"], "default")
        self.assertEqual(self.body(model="fable")["fallbacks"], "default")
        for model in ("sonnet", "haiku", "claude-opus-4-8"):
            self.assertNotIn("fallbacks", self.body(model=model), model)

    def test_max_tokens_clamped_to_the_model(self):
        self.assertEqual(self.body(max_tokens=999999)["max_tokens"], 128000)
        self.assertEqual(self.body(model="haiku", max_tokens=999999)["max_tokens"], 64000)
        self.assertEqual(self.body(max_tokens=0)["max_tokens"], 1)

    def test_stream_is_always_true(self):
        self.assertIs(self.body()["stream"], True)

    def test_system_string_passthrough(self):
        self.assertEqual(self.body(system="Be brief.")["system"], "Be brief.")
        self.assertNotIn("system", self.body(system=None))
        self.assertNotIn("system", self.body(system=""))

    def test_system_blocks_are_copied(self):
        blocks = [{"type": "text", "text": "x"}]
        body = self.body(system=blocks)
        self.assertEqual(body["system"], blocks)
        self.assertIsNot(body["system"][0], blocks[0])

    def test_caller_messages_are_not_mutated(self):
        msgs = [{"role": "user", "content": "a" * 8000},
                {"role": "assistant", "content": "b" * 8000},
                {"role": "user", "content": "c"}]
        snapshot = json.dumps(msgs)
        Client(KEY).build_body(messages=msgs)
        self.assertEqual(json.dumps(msgs), snapshot)

    def test_bad_message_raises(self):
        with self.assertRaises(ValueError):
            self.body(messages=[{"content": "no role"}])


class TestCacheControl(unittest.TestCase):
    def body(self, **kw):
        kw.setdefault("messages", [{"role": "user", "content": "hi"}])
        return Client(KEY).build_body(**kw)

    def count(self, body):
        return json.dumps(body).count('"cache_control"')

    def test_no_breakpoint_on_a_short_prefix(self):
        self.assertEqual(self.count(self.body(system="Be brief.")), 0)

    def test_system_breakpoint_when_large(self):
        system = "x" * (CACHE_MIN_TOKENS * 4 + 10)
        body = self.body(system=system)
        self.assertEqual(body["system"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["system"][-1]["text"], system)
        self.assertEqual(self.count(body), 1)

    def test_breakpoint_on_second_to_last_message(self):
        big = "x" * (CACHE_MIN_TOKENS * 4 + 10)
        msgs = [{"role": "user", "content": big},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow up"}]
        body = self.body(messages=msgs)
        self.assertEqual(body["messages"][1]["content"][-1]["cache_control"],
                         {"type": "ephemeral"})
        self.assertNotIn("cache_control", json.dumps(body["messages"][2]))
        self.assertEqual(self.count(body), 1)

    def test_last_message_is_never_a_breakpoint(self):
        big = "x" * (CACHE_MIN_TOKENS * 8)
        body = self.body(messages=[{"role": "user", "content": big}])
        self.assertEqual(self.count(body), 0)

    def test_at_most_four_breakpoints(self):
        big = "x" * (CACHE_MIN_TOKENS * 4 + 10)
        msgs = [{"role": "user", "content": big}, {"role": "assistant", "content": big},
                {"role": "user", "content": big}, {"role": "assistant", "content": big},
                {"role": "user", "content": "now"}]
        body = self.body(system=big, messages=msgs)
        self.assertLessEqual(self.count(body), 4)
        self.assertEqual(self.count(body), 2)

    def test_per_model_minimum_prefix(self):
        # The minimum cacheable prefix is not monotonic across generations: 512 on
        # opus-5/fable-5, 4096 on opus-4-6/haiku-4-5. A flat 1024 both misses
        # cacheable prefixes and marks prefixes the server will ignore.
        want = {"claude-opus-5": 512, "claude-fable-5": 512, "claude-opus-4-8": 1024,
                "claude-sonnet-5": 1024, "claude-sonnet-4-6": 1024,
                "claude-opus-4-7": 2048, "claude-opus-4-6": 4096,
                "claude-haiku-4-5": 4096}
        for model, minimum in want.items():
            self.assertEqual(MODELS[model].cache_min, minimum, model)
            for tokens in (minimum // 2, minimum + 8):
                system = "x" * (tokens * 4)
                body = self.body(model=model, system=system)
                placed = self.count(body) == 1
                self.assertEqual(placed, tokens >= minimum,
                                 f"{model} at ~{tokens} tokens")

    def test_caller_breakpoints_count_against_the_budget(self):
        # More than four cache_control markers is a hard 400, and the caller's own
        # markers are markers too.
        big = "x" * 8000
        marked = {"type": "text", "text": big, "cache_control": {"type": "ephemeral"}}
        system = [dict(marked), dict(marked)]
        msgs = [{"role": "user", "content": [dict(marked)]},
                {"role": "assistant", "content": [dict(marked)]},
                {"role": "user", "content": "now?"}]
        body = self.body(model="opus", system=system, messages=msgs)
        self.assertEqual(self.count(body), MAX_CACHE_BREAKPOINTS)

    def test_five_breakpoints_are_never_built(self):
        big = "x" * 8000
        marked = {"type": "text", "text": big, "cache_control": {"type": "ephemeral"}}
        msgs = [{"role": "user", "content": [dict(marked), dict(marked)]},
                {"role": "assistant", "content": big},
                {"role": "user", "content": "q"}]
        body = self.body(model="opus", system=[dict(marked), dict(marked)], messages=msgs)
        self.assertLessEqual(self.count(body), MAX_CACHE_BREAKPOINTS)

    def test_an_already_marked_block_is_not_counted_as_a_new_one(self):
        # Finding the system block already marked places nothing, so it must not
        # spend a slot: with three markers in play the fourth is still ours to use.
        big = "x" * 8000
        marked = {"type": "text", "text": big, "cache_control": {"type": "ephemeral"}}
        system = [dict(marked)]
        msgs = [{"role": "user", "content": [dict(marked)]},
                {"role": "assistant", "content": [dict(marked)]},
                {"role": "user", "content": big},
                {"role": "user", "content": "q"}]
        body = self.body(model="opus", system=system, messages=msgs)
        self.assertEqual(self.count(body), MAX_CACHE_BREAKPOINTS)
        self.assertEqual(body["messages"][-2]["content"][-1]["cache_control"],
                         {"type": "ephemeral"})

    def test_cjk_prefixes_are_not_undercounted(self):
        # ~1 token per character for CJK, not 4: a 900-character Japanese system
        # prompt is well past opus-5's 512-token minimum and must cache.
        system = "日本語のシステムプロンプト。" * 70
        self.assertLess(len(system) // 4, MODELS["claude-opus-5"].cache_min)
        self.assertGreater(_estimate_tokens(system), MODELS["claude-opus-5"].cache_min)
        self.assertEqual(self.count(self.body(model="opus", system=system)), 1)
        self.assertEqual(_estimate_tokens("x" * 400), 100)

    def test_cache_false_places_nothing(self):
        big = "x" * (CACHE_MIN_TOKENS * 8)
        body = self.body(system=big, messages=[{"role": "user", "content": big},
                                               {"role": "assistant", "content": big},
                                               {"role": "user", "content": "q"}], cache=False)
        self.assertEqual(self.count(body), 0)

    def test_breakpoint_skips_a_thinking_block(self):
        big = "x" * (CACHE_MIN_TOKENS * 4 + 10)
        msgs = [{"role": "user", "content": big},
                {"role": "assistant", "content": [{"type": "text", "text": "seen"},
                                                  {"type": "thinking", "thinking": "hm"}]},
                {"role": "user", "content": "q"}]
        body = self.body(messages=msgs)
        blocks = body["messages"][1]["content"]
        self.assertNotIn("cache_control", blocks[1])
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})


class TestHeaders(unittest.TestCase):
    def test_api_key_form(self):
        h = Client(KEY)._build_headers()
        self.assertEqual(h["x-api-key"], KEY)
        self.assertEqual(h["anthropic-version"], API_VERSION)
        self.assertEqual(h["content-type"], "application/json")
        self.assertNotIn("authorization", h)
        self.assertNotIn("anthropic-beta", h)

    def test_oauth_form(self):
        h = Client(OAUTH)._build_headers()
        self.assertEqual(h["authorization"], f"Bearer {OAUTH}")
        self.assertNotIn("x-api-key", h)
        self.assertIn("oauth-2025-04-20", h["anthropic-beta"])

    def test_beta_headers_are_comma_joined(self):
        h = Client(OAUTH)._build_headers(betas=["server-side-fallback-2026-07-01"])
        self.assertEqual(h["anthropic-beta"],
                         "oauth-2025-04-20,server-side-fallback-2026-07-01")

    def test_fallback_beta_is_sent_with_the_fallback_parameter(self):
        t = ok_transport()
        list(client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        sent = t.calls[0]
        self.assertEqual(sent["headers"]["anthropic-beta"],
                         "server-side-fallback-2026-07-01")
        self.assertEqual(sent["body"]["fallbacks"], "default")
        self.assertEqual(sent["url"], "http://127.0.0.1:1/never-used/v1/messages")

    def test_no_fallback_beta_without_the_parameter(self):
        t = ok_transport()
        list(client(t).stream(model="sonnet", messages=[{"role": "user", "content": "hi"}]))
        self.assertNotIn("anthropic-beta", t.calls[0]["headers"])

    def test_is_oauth_token(self):
        self.assertTrue(is_oauth_token(OAUTH))
        self.assertFalse(is_oauth_token(KEY))
        self.assertFalse(is_oauth_token(None))


# --------------------------------------------------------------------------- secrets


class TestSecrets(unittest.TestCase):
    def test_find_api_key_precedence(self):
        self.assertEqual(find_api_key({"ANTHROPIC_API_KEY": KEY}), KEY)
        self.assertEqual(find_api_key({"ANTHROPIC_AUTH_TOKEN": OAUTH}), OAUTH)
        self.assertEqual(find_api_key({"ANTHROPIC_API_KEY": KEY,
                                       "ANTHROPIC_AUTH_TOKEN": OAUTH}), KEY)
        self.assertIsNone(find_api_key({}))
        self.assertIsNone(find_api_key({"ANTHROPIC_API_KEY": "   "}))
        self.assertEqual(find_api_key({"ANTHROPIC_API_KEY": " " + KEY + "\n"}), KEY)

    def test_redact(self):
        self.assertEqual(redact(f"auth failed for {KEY}"), "auth failed for sk-ant-***")
        self.assertNotIn("TESTKEY", redact(KEY))
        self.assertEqual(redact(""), "")

    def test_key_never_appears_in_a_repr(self):
        c = Client(KEY, base_url="http://127.0.0.1:1")
        for text in (repr(c), str(c)):
            self.assertNotIn(KEY, text)
            self.assertNotIn("TESTKEY", text)
        self.assertNotIn(KEY, repr(c.transport))

    def test_key_never_appears_in_an_exception(self):
        body = json.dumps({"type": "error",
                           "error": {"type": "authentication_error",
                                     "message": f"invalid x-api-key: {KEY}"}}).encode()
        t = FakeTransport([lambda: FakeStream(status=401, body=body)])
        with self.assertRaises(AuthError) as ctx:
            list(client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        exc = ctx.exception
        for text in (str(exc), repr(exc), exc.message, str(exc.args)):
            self.assertNotIn(KEY, text)
            self.assertNotIn("TESTKEY", text)
        self.assertIn("sk-ant-***", exc.message)

    def test_non_standard_token_is_scrubbed_too(self):
        token = "gateway-token-abcdef123456"
        body = json.dumps({"type": "error",
                           "error": {"type": "authentication_error",
                                     "message": f"bad token {token}"}}).encode()
        t = FakeTransport([lambda: FakeStream(status=401, body=body)])
        c = Client(token, base_url="http://127.0.0.1:1", transport=t, max_retries=0)
        with self.assertRaises(AuthError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertNotIn(token, str(ctx.exception))
        self.assertNotIn(token, repr(ctx.exception))
        self.assertIn("***", ctx.exception.message)

    def test_header_values_hide_the_key_from_a_repr(self):
        # The value still goes on the wire; only its repr is blanked.
        for key in (KEY, OAUTH):
            headers = Client(key, base_url="http://127.0.0.1:1")._build_headers()
            value = headers.get("x-api-key") or headers["authorization"]
            self.assertIn(key, str(value))
            self.assertNotIn("TESTKEY", repr(headers))
            self.assertNotIn("TESTTOKEN", repr(headers))
            self.assertEqual(json.loads(json.dumps({"h": str(value)}))["h"], str(value))

    def test_a_traceback_that_captures_locals_cannot_print_the_key(self):
        # rich, better-exceptions and cgitb all format frame locals by default, and
        # `headers` is a live local in six frames between stream() and the socket.
        class Rude:
            def open(self, url, headers, body, timeout):
                raise OSError("connection reset")

        c = Client(KEY, base_url="http://127.0.0.1:1", transport=Rude(), max_retries=0)
        try:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        except APIError as exc:
            report = "".join(traceback.TracebackException.from_exception(
                exc, capture_locals=True).format())
        else:  # pragma: no cover - the transport always raises
            self.fail("expected an APIError")
        self.assertNotIn("TESTKEY", report)
        self.assertNotIn(KEY, report)
        self.assertIn("x-api-key", report, "the header itself should still be visible")

    def test_client_state_does_not_print_the_key(self):
        c = Client(KEY, base_url="http://127.0.0.1:1")
        self.assertNotIn("TESTKEY", repr(vars(c)))
        self.assertEqual(str(c._api_key), KEY)

    def test_error_taxonomy_attributes(self):
        exc = APIError("boom", status=500, type="api_error", request_id="req_9",
                       retryable=True)
        self.assertEqual((exc.status, exc.type, exc.request_id, exc.retryable),
                         (500, "api_error", "req_9", True))
        self.assertIn("boom", str(exc))
        self.assertIn("req_9", str(exc))
        for cls in (AuthError, BadRequestError, RateLimitError, OverloadedError,
                    ServerError, NetworkError, CancelledError):
            self.assertTrue(issubclass(cls, APIError))
        self.assertIsNone(RateLimitError("x").retry_after)


# ------------------------------------------------------------------- real http server


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length)
        self.server.requests.append({"headers": dict(self.headers), "body": raw})
        script = self.server.script
        item = script[min(len(self.server.requests) - 1, len(script) - 1)]
        status, headers, chunks = item
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        for i, chunk in enumerate(chunks):
            self.wfile.write(chunk)
            self.wfile.flush()
            if i == 0 and self.server.gate is not None:
                self.server.gate.wait(5)

    def log_message(self, *args):
        pass


class LiveServerCase(unittest.TestCase):
    def serve(self, script, gate=None):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.script = script
        server.requests = []
        server.gate = gate
        # A cancelled client hangs up mid-response; that is the scenario, not an
        # error worth a traceback on stderr.
        server.handle_error = lambda *args: None
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_address[1]}"


class TestHTTPTransport(LiveServerCase):
    def test_end_to_end_over_a_real_socket(self):
        raw = sse(text_stream())
        server, url = self.serve([(200, {"content-type": "text/event-stream",
                                         "request-id": "req_live"},
                                   [raw[:20], raw[20:]])])
        c = Client(KEY, base_url=url, timeout=10)
        events = list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        c.close()
        self.assertEqual(drain(events)[0], "Hello, wörld…")
        sent = server.requests[0]
        self.assertEqual(sent["headers"]["x-api-key"], KEY)
        self.assertEqual(sent["headers"]["anthropic-version"], API_VERSION)
        self.assertEqual(sent["headers"]["anthropic-beta"],
                         "server-side-fallback-2026-07-01")
        body = json.loads(sent["body"])
        self.assertIs(body["stream"], True)
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertNotIn("budget_tokens", sent["body"].decode())

    def test_oauth_headers_over_a_real_socket(self):
        server, url = self.serve([(200, {}, [sse(text_stream(thinking=()))])])
        c = Client(OAUTH, base_url=url, timeout=10)
        list(c.stream(model="sonnet", messages=[{"role": "user", "content": "hi"}]))
        c.close()
        headers = server.requests[0]["headers"]
        self.assertEqual(headers["authorization"], f"Bearer {OAUTH}")
        self.assertNotIn("x-api-key", {k.lower() for k in headers})
        self.assertEqual(headers["anthropic-beta"], "oauth-2025-04-20")

    def test_retry_over_a_real_socket(self):
        server, url = self.serve([
            (429, {"retry-after": "0", "content-type": "application/json"}, [err_body()]),
            (200, {}, [sse(text_stream(thinking=()))]),
        ])
        c = Client(KEY, base_url=url, timeout=10, max_retries=2)
        c.backoff_base = 0.0
        events = list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        c.close()
        self.assertEqual(len(server.requests), 2)
        self.assertEqual(drain(events)[0], "Hello, wörld…")

    def test_http_error_body_is_parsed(self):
        server, url = self.serve([(400, {"request-id": "req_bad"},
                                   [err_body("invalid_request_error", "max_tokens too big")])])
        c = Client(KEY, base_url=url, timeout=10)
        with self.assertRaises(BadRequestError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        c.close()
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.type, "invalid_request_error")
        self.assertIn("max_tokens", ctx.exception.message)
        self.assertEqual(len(server.requests), 1)

    def test_events_arrive_before_the_response_completes(self):
        gate = threading.Event()
        raw = sse(text_stream())
        head = raw.split(b"\n\n")[0] + b"\n\n"
        server, url = self.serve([(200, {}, [head, raw[len(head):]])], gate=gate)
        c = Client(KEY, base_url=url, timeout=10)
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}])
        started = time.monotonic()
        first = next(gen)
        elapsed = time.monotonic() - started
        self.assertEqual(first.kind, "start")
        self.assertLess(elapsed, 3.0, "the transport buffered the whole response")
        gate.set()
        self.assertEqual(drain(list(gen))[0], "Hello, wörld…")
        c.close()

    def test_cancel_on_a_quiet_socket_is_prompt(self):
        # The reader is parked in recv() with nothing coming: exactly the shape of
        # a long "thinking" pause. Closing the fd from the watchdog thread does not
        # wake it — only shutdown() does — and until it wakes the app is frozen for
        # up to `timeout` seconds and then dies on an AttributeError.
        gate = threading.Event()
        raw = sse(text_stream())
        head = raw.split(b"\n\n")[0] + b"\n\n"
        server, url = self.serve([(200, {}, [head, raw[len(head):]])], gate=gate)
        self.addCleanup(gate.set)
        c = Client(KEY, base_url=url, timeout=60, max_retries=0)
        self.addCleanup(c.close)
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        self.assertEqual(next(gen).kind, "start")
        threading.Timer(0.2, cancel.set).start()
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            for _ in gen:
                pass
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.0, f"cancellation took {elapsed:.2f}s on a quiet socket")

    def test_connection_refused_is_a_network_error(self):
        server, url = self.serve([(200, {}, [b""])])
        port = server.server_address[1]
        server.shutdown()
        server.server_close()
        c = Client(KEY, base_url=f"http://127.0.0.1:{port}", timeout=2, max_retries=0)
        with self.assertRaises(NetworkError):
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        c.close()

    def test_a_refused_connection_does_not_carry_the_key_in_its_traceback(self):
        # http.client builds the whole request — auth header and all — into a
        # local before it connects, so the exception chain that reaches that frame
        # is cut rather than re-raised `from` it.
        server, url = self.serve([(200, {}, [b""])])
        port = server.server_address[1]
        server.shutdown()
        server.server_close()
        c = Client(KEY, base_url=f"http://127.0.0.1:{port}", timeout=2, max_retries=0)
        self.addCleanup(c.close)
        with self.assertRaises(NetworkError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        failure = ctx.exception
        report = "".join(traceback.TracebackException.from_exception(
            failure, capture_locals=True).format())
        self.assertNotIn("TESTKEY", report)
        self.assertNotIn(KEY, report)
        # And the chain is cut, not merely uninteresting today: `from exc` here
        # would keep working until the frame it chains to happens to hold the
        # headers again, and nothing would say so.
        self.assertIsNone(failure.__cause__, "the transport error still carries a cause")
        self.assertIsNone(failure.__context__,
                          "the transport error still carries a context")

    def test_bad_scheme(self):
        with self.assertRaises(NetworkError):
            HTTPTransport().open("ftp://example.invalid/x", {}, b"", 1)


# ------------------------------------------------------------------------ sdk bridge


class _FakeSDKEvent:
    def __init__(self, payload):
        self._payload = payload
        self.type = payload.get("type")

    def to_dict(self):
        return dict(self._payload)


class _FakeSDKStream:
    def __init__(self, events, owner):
        self._events = events
        self._owner = owner

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._owner.exited = True
        return False

    def __iter__(self):
        for payload in self._events:
            yield _FakeSDKEvent(payload)


class _FakeMessages:
    def __init__(self, owner):
        self._owner = owner

    def stream(self, **kwargs):
        self._owner.calls.append(kwargs)
        return _FakeSDKStream([p for _, p in text_stream()], self._owner)


class _FakeAnthropic:
    instances = []

    def __init__(self, api_key=None, base_url=None, max_retries=None, **kw):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []
        self.exited = False
        self.messages = _FakeMessages(self)
        _FakeAnthropic.instances.append(self)


class TestSDKTransport(unittest.TestCase):
    def setUp(self):
        _FakeAnthropic.instances = []
        module = types.ModuleType("anthropic")
        module.Anthropic = _FakeAnthropic
        self._saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = module
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = self._saved

    def test_auto_selects_the_sdk_and_maps_events(self):
        c = Client(KEY, base_url="https://example.invalid", transport="auto")
        self.assertIsInstance(c.transport, SDKTransport)
        events = list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(drain(events)[0], "Hello, wörld…")
        self.assertEqual(events[-1].kind, "done")
        self.assertEqual(events[-1].stop_reason, "end_turn")
        sdk = _FakeAnthropic.instances[0]
        self.assertEqual(sdk.api_key, KEY)
        call = sdk.calls[0]
        self.assertEqual(call["model"], "claude-opus-5")
        self.assertEqual(call["max_tokens"], 32000)
        self.assertEqual(call["extra_body"]["thinking"],
                         {"type": "adaptive", "display": "summarized"})
        self.assertEqual(call["extra_body"]["output_config"], {"effort": "high"})
        self.assertNotIn("stream", call["extra_body"])
        self.assertEqual(call["extra_headers"]["anthropic-beta"],
                         "server-side-fallback-2026-07-01")
        self.assertTrue(sdk.exited)
        c.close()

    def test_events_stop_when_the_cancel_event_is_set(self):
        transport = SDKTransport(KEY, "https://example.invalid")
        cancel = threading.Event()
        cancel.set()
        events = transport.events({"model": "claude-opus-5", "max_tokens": 16,
                                   "messages": [{"role": "user", "content": "hi"}],
                                   "stream": True}, {}, 10.0, cancel)
        with self.assertRaises(CancelledError):
            next(events)
        # The SDK stream context must still be exited, or the connection leaks.
        self.assertTrue(_FakeAnthropic.instances[-1].exited)

    def test_cancel_mid_stream_through_the_sdk_transport(self):
        c = Client(KEY, base_url="https://example.invalid", transport="auto",
                   max_retries=0)
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        self.assertEqual(next(gen).kind, "start")
        cancel.set()
        with self.assertRaises(CancelledError):
            list(gen)
        self.assertTrue(_FakeAnthropic.instances[-1].exited)
        c.close()

    def test_available(self):
        self.assertTrue(SDKTransport.available())

    def test_auto_falls_back_when_the_sdk_is_missing(self):
        sys.modules["anthropic"] = None  # import raises ImportError for a None entry
        c = Client(KEY, transport="auto")
        self.assertIsInstance(c.transport, HTTPTransport)

    def test_sdk_exceptions_become_api_errors(self):
        class _Boom(Exception):
            status_code = 429
            request_id = "req_sdk"

        def explode(**kwargs):
            raise _Boom("too fast")

        c = Client(KEY, transport="auto", max_retries=0)
        c.transport._sdk.messages.stream = explode
        with self.assertRaises(RateLimitError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(ctx.exception.request_id, "req_sdk")



# ------------------------------------------------------------- cancel before headers


class SilentSocketServer:
    """A loopback listener that reads a request and then says nothing at all.

    The shape of every real cancel that matters: the connection is up, the
    request is sent, and the user hits Ctrl-C while waiting for the first token.
    `http.server` cannot express it — it answers before any hook runs.
    """

    def __init__(self, reply=b"", delay=30.0):
        self.reply = reply
        self.delay = delay
        self.requests = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self.stopped = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def _serve(self):
        while not self.stopped.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
            self.requests.append(buf)
            self.stopped.wait(self.delay)
            if self.reply:
                conn.sendall(self.reply)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        self.stopped.set()
        try:
            self._sock.close()
        except OSError:
            pass


class TestCancelBeforeTheFirstByte(unittest.TestCase):
    """Cancelling while the server is still thinking — the common case.

    The watchdog used to be started only after `transport.open()` returned, i.e.
    only once response headers had arrived. Until then a cancel was not observed
    at all and the ceiling was `Client.timeout` — 600 s in lume.
    """

    def server(self, **kw):
        server = SilentSocketServer(**kw)
        self.addCleanup(server.close)
        return server

    def test_cancel_while_waiting_for_response_headers(self):
        server = self.server()
        c = Client(KEY, base_url=server.url, timeout=60.0, max_retries=3)
        self.addCleanup(c.close)
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError) as ctx:
            list(gen)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0,
                        f"cancel waited {elapsed:.2f}s for response headers")
        # And it is a cancel, not a retryable network failure: a cancelled
        # request that comes back retryable is promptly sent again.
        self.assertFalse(ctx.exception.retryable)
        time.sleep(0.2)
        self.assertEqual(len(server.requests), 1, "the cancelled request was retried")

    def test_cancel_while_the_connect_itself_is_outstanding(self):
        # No socket exists yet during a connect, so nothing can be shut down: the
        # connect has to be polled. `select` is stubbed to never report the socket
        # writable, which is exactly what a blackholed address looks like.
        import lume.api as api

        listener = self.server()
        real_select = api.select

        class _NeverReady:
            @staticmethod
            def select(readable, writable, failing, timeout):
                time.sleep(min(timeout or 0.01, 0.01))
                return ([], [], [])

        api.select = _NeverReady
        self.addCleanup(setattr, api, "select", real_select)

        c = Client(KEY, base_url=listener.url, timeout=60.0, max_retries=3)
        self.addCleanup(c.close)
        cancel = threading.Event()
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                          cancel=cancel))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0, f"cancel waited {elapsed:.2f}s on a stuck connect")
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(server_requests := len(listener.requests), 0, server_requests)

    def test_a_stuck_connect_still_times_out_without_a_cancel(self):
        import lume.api as api

        real_select = api.select

        class _NeverReady:
            @staticmethod
            def select(readable, writable, failing, timeout):
                time.sleep(min(timeout or 0.01, 0.01))
                return ([], [], [])

        api.select = _NeverReady
        self.addCleanup(setattr, api, "select", real_select)
        listener = self.server()
        c = Client(KEY, base_url=listener.url, timeout=0.3, max_retries=0)
        self.addCleanup(c.close)
        started = time.monotonic()
        with self.assertRaises(NetworkError):
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertLess(time.monotonic() - started, 5.0)


class SlowTransport:
    """A cancel-unaware transport: `open()` blocks, exactly as the seam allows."""

    def __init__(self, stream, delay=5.0):
        self.stream = stream
        self.delay = delay
        self.released = threading.Event()
        self.entered = threading.Event()

    def open(self, url, headers, body, timeout):  # no `cancel` parameter
        self.entered.set()
        self.released.wait(self.delay)
        return self.stream

    def close(self):
        self.released.set()


class TestCancelWithACancelUnawareTransport(unittest.TestCase):
    """The seam still takes a four-argument `open()`, and cancel still wins."""

    def test_a_transport_without_a_cancel_argument_still_streams(self):
        t = FakeTransport([lambda: FakeStream(chunks=[sse(text_stream())])])
        self.assertNotIn("cancel", str(FakeTransport.open.__doc__ or ""))
        events = list(client(t).stream(model="opus",
                                       messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(drain(events)[0], "Hello, wörld…")

    def test_a_blocking_open_is_still_cancellable(self):
        stream = FakeStream(chunks=[sse(text_stream())])
        t = SlowTransport(stream, delay=5.0)
        c = client(t, max_retries=3)
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        box = {}

        def consume():
            try:
                list(gen)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                box["exc"] = exc

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        self.assertTrue(t.entered.wait(5), "open() was never called")
        started = time.monotonic()
        cancel.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("exc"), CancelledError)
        self.assertLess(time.monotonic() - started, 1.0)
        # The abandoned call is reaped, not leaked: whatever it finally returns
        # gets closed.
        t.released.set()
        for _ in range(100):
            if stream.closed:
                break
            time.sleep(0.02)
        self.assertTrue(stream.closed, "the abandoned connection was never closed")


class TestCancelLatency(unittest.TestCase):
    def test_cancel_latency_is_bounded_not_merely_eventual(self):
        # The watchdog polls; nothing else asserts how often. At a 5 s poll every
        # other cancellation test still passes and the app still freezes for five
        # seconds on every Ctrl-C.
        head = sse(text_stream(pieces=("Hel",), thinking=())[:5])
        stream = BlockingStream([head])
        t = FakeTransport([lambda: stream])
        cancel = threading.Event()
        gen = client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                               cancel=cancel)
        box = {}

        def consume():
            try:
                list(gen)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                box["exc"] = exc

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        self.assertTrue(stream.blocked.wait(5), "did not reach the blocking read")
        started = time.monotonic()
        cancel.set()
        worker.join(5)
        elapsed = time.monotonic() - started
        self.assertIsInstance(box.get("exc"), CancelledError)
        self.assertLess(elapsed, 0.5,
                        f"cancel took {elapsed:.2f}s; the watchdog poll is too coarse")

    def test_the_cancel_watchdog_does_not_outlive_the_stream(self):
        # The watchdog polls a "we are done here" flag while it waits on the
        # cancel event. Polling it every few seconds costs nothing in latency —
        # `Event.wait` returns the moment the event is set — but leaves one live
        # thread per turn hanging around after each reply, all session.
        before = {t.name for t in threading.enumerate()}
        t = ok_transport()
        cancel = threading.Event()
        list(client(t).stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                              cancel=cancel))
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            leftover = [x for x in threading.enumerate()
                        if x.name.startswith("lume-cancel") and x.name not in before]
            if not leftover:
                break
            time.sleep(0.02)
        self.assertEqual(leftover, [], "the cancel watchdog outlived its stream")

    def test_cancel_during_backoff_does_not_wait_out_the_delay(self):
        # `_pause` waits on the cancel event rather than sleeping. Sleeping
        # instead leaves Ctrl-C dead for the whole backoff — up to 16 s — and the
        # retry then goes out anyway.
        t = FakeTransport([lambda: FakeStream(status=529, body=err_body("overloaded_error"))])
        c = client(t, max_retries=3)
        c.backoff_base = c.backoff_max = 30.0
        c._sleep = time.sleep  # a real sleep, so only the event can cut it short
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        timer = threading.Timer(0.05, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            list(gen)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.0, f"the backoff ignored the cancel for {elapsed:.2f}s")
        self.assertEqual(len(t.calls), 1, "a cancelled request was retried anyway")


class TestTransportBookkeeping(unittest.TestCase):
    def test_a_closed_stream_is_forgotten_by_its_transport(self):
        # `_live` retains a connection, a response and a socket each. Never
        # forgetting a finished one leaks all three for the life of the process:
        # one per turn, for a session that runs all day.
        server, url = LiveServerCase.serve(self, [(200, {}, [sse(text_stream())])] * 5)
        transport = HTTPTransport()
        c = Client(KEY, base_url=url, timeout=10, transport=transport)
        self.addCleanup(c.close)
        for _ in range(3):
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(len(transport._live), 0, transport._live)

    def test_close_releases_a_stream_that_is_still_open(self):
        server, url = LiveServerCase.serve(self, [(200, {}, [sse(text_stream())])])
        transport = HTTPTransport()
        stream = transport.open(f"{url}/v1/messages", {"content-type": "application/json"},
                                b"{}", 5)
        self.assertEqual(len(transport._live), 1)
        transport.close()
        self.assertEqual(len(transport._live), 0)
        self.assertTrue(stream._closed)


# ------------------------------------------------------------------ wire correctness


class TestEffortLadder(unittest.TestCase):
    def body(self, **kw):
        kw.setdefault("messages", [{"role": "user", "content": "hi"}])
        return Client(KEY).build_body(**kw)

    def test_xhigh_is_not_sent_to_models_that_predate_it(self):
        # `xhigh` arrived with opus-4-7. On the 4.6 family it is a hard 400, so
        # `lume --effort xhigh -m sonnet-4-6` used to fail outright.
        for model in ("claude-opus-4-6", "claude-sonnet-4-6"):
            self.assertNotIn("xhigh", MODELS[model].efforts, model)
            self.assertEqual(self.body(model=model, effort="xhigh")["output_config"],
                             {"effort": "high"}, model)
            self.assertEqual(self.body(model=model, effort="max")["output_config"],
                             {"effort": "max"}, model)

    def test_models_that_have_xhigh_still_get_it(self):
        for model in ("claude-opus-5", "claude-fable-5", "claude-opus-4-8",
                      "claude-opus-4-7", "claude-sonnet-5"):
            self.assertEqual(self.body(model=model, effort="xhigh")["output_config"],
                             {"effort": "xhigh"}, model)

    def test_a_model_with_no_effort_at_all_sends_none(self):
        self.assertEqual(MODELS["claude-haiku-4-5"].effort_ladder, ())
        for effort in EFFORTS:
            self.assertNotIn("output_config", self.body(model="haiku", effort=effort))

    def test_an_unknown_effort_is_still_a_hard_error(self):
        with self.assertRaises(ValueError):
            self.body(effort="ludicrous")
        with self.assertRaises(ValueError):
            resolve_model("sonnet-4-6").clamp_effort("ludicrous")

    def test_clamping_picks_the_best_level_at_or_below(self):
        spec = resolve_model("sonnet-4-6")
        self.assertEqual(spec.clamp_effort("xhigh"), "high")
        self.assertEqual(spec.clamp_effort("low"), "low")
        self.assertEqual(spec.clamp_effort("max"), "max")
        self.assertIsNone(spec.clamp_effort(None))
        self.assertIsNone(resolve_model("haiku").clamp_effort("high"))


class TestThinkingDisplay(unittest.TestCase):
    def body(self, **kw):
        kw.setdefault("messages", [{"role": "user", "content": "hi"}])
        return Client(KEY).build_body(**kw)

    def test_display_is_only_sent_where_it_exists(self):
        # `thinking.display` arrived with opus-4-7; the 4.6 family does not know
        # the field, and its old behaviour is `summarized` anyway.
        for model in ("claude-opus-5", "claude-fable-5", "claude-opus-4-8",
                      "claude-opus-4-7", "claude-sonnet-5"):
            self.assertEqual(self.body(model=model)["thinking"],
                             {"type": "adaptive", "display": "summarized"}, model)
        for model in ("claude-opus-4-6", "claude-sonnet-4-6"):
            self.assertEqual(self.body(model=model)["thinking"], {"type": "adaptive"},
                             model)
            self.assertFalse(MODELS[model].supports_thinking_display, model)


class TestBreakpointOverflow(unittest.TestCase):
    def test_five_caller_breakpoints_raise_instead_of_400ing(self):
        marked = {"type": "text", "text": "x" * 4000,
                  "cache_control": {"type": "ephemeral"}}
        msgs = [{"role": "user", "content": [dict(marked), dict(marked), dict(marked)]},
                {"role": "user", "content": "q"}]
        with self.assertRaises(ValueError) as ctx:
            Client(KEY).build_body(model="opus", system=[dict(marked), dict(marked)],
                                   messages=msgs)
        self.assertIn("cache_control", str(ctx.exception))
        # Even with our own placement switched off: the markers are the caller's.
        with self.assertRaises(ValueError):
            Client(KEY).build_body(model="opus", system=[dict(marked), dict(marked)],
                                   messages=msgs, cache=False)

    def test_exactly_four_is_still_allowed(self):
        marked = {"type": "text", "text": "x" * 4000,
                  "cache_control": {"type": "ephemeral"}}
        body = Client(KEY).build_body(
            model="opus", system=[dict(marked), dict(marked)],
            messages=[{"role": "user", "content": [dict(marked), dict(marked)]},
                      {"role": "user", "content": "q"}])
        self.assertEqual(_count_breakpoints(body["system"])
                         + _count_breakpoints(body["messages"]), 4)


class TestNonSSEResponse(unittest.TestCase):
    def test_a_200_that_is_not_an_event_stream_says_what_it_got(self):
        # A proxy or captive portal answering for the API: 200, HTML body. Parsed
        # as SSE that is "stream ended before message_stop" with the one useful
        # thing thrown away.
        html = b"<html><body>Sign in to the corporate proxy to continue</body></html>"
        t = FakeTransport([lambda: FakeStream(headers={"content-type": "text/html"},
                                              chunks=[html])])
        with self.assertRaises(APIError) as ctx:
            list(client(t, max_retries=2).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))
        message = ctx.exception.message
        self.assertIn("text/html", message)
        self.assertIn("corporate proxy", message)
        self.assertNotIn("message_stop", message)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(len(t.calls), 1, "a misrouted 200 was retried")

    def test_a_json_error_served_with_status_200_is_readable(self):
        body = json.dumps({"error": {"message": "gateway rejected the upstream"}}).encode()
        t = FakeTransport([lambda: FakeStream(headers={"content-type": "application/json"},
                                              chunks=[body])])
        with self.assertRaises(APIError) as ctx:
            list(client(t, max_retries=0).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertIn("gateway rejected the upstream", ctx.exception.message)

    def test_an_event_stream_content_type_is_untouched(self):
        t = FakeTransport([lambda: FakeStream(
            headers={"content-type": "text/event-stream; charset=utf-8"},
            chunks=[sse(text_stream())])])
        events = list(client(t).stream(model="opus",
                                       messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(drain(events)[0], "Hello, wörld…")


class TestCRTerminatedStream(unittest.TestCase):
    def test_a_stream_whose_last_terminator_is_a_lone_cr_finishes_cleanly(self):
        # Every line ending is legal in SSE, CR included. Holding the final CR
        # back past end of stream loses `message_stop` and turns a complete answer
        # into NetworkError("stream ended before message_stop").
        raw = sse(text_stream(), eol="\r")
        t = FakeTransport([lambda: FakeStream(chunks=[raw])])
        events = list(client(t, max_retries=0).stream(
            model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(drain(events)[0], "Hello, wörld…")
        self.assertEqual(events[-1].kind, "done")
        self.assertEqual(events[-1].stop_reason, "end_turn")

    def test_a_genuinely_cut_stream_is_still_reported(self):
        raw = sse(text_stream())
        t = FakeTransport([lambda: FakeStream(chunks=[raw[:-1]])])
        with self.assertRaises(NetworkError) as ctx:
            list(client(t, max_retries=0).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertIn("message_stop", ctx.exception.message)


# --------------------------------------------------------------------------- pricing


class TestFallbackPricing(unittest.TestCase):
    """A rescued turn is billed per attempt, at each attempt's own model."""

    def usage(self):
        records = [
            message_start(model="claude-fable-5",
                          usage={"input_tokens": 1000, "output_tokens": 0}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta",
                                               "text": "partial from fable"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("content_block_start", {"type": "content_block_start", "index": 1,
                                     "content_block": {
                                         "type": "fallback",
                                         "from": {"model": "claude-fable-5"},
                                         "to": {"model": "claude-opus-4-8"}}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            ("content_block_start", {"type": "content_block_start", "index": 2,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 2,
                                     "delta": {"type": "text_delta",
                                               "text": "the rest, from opus"}}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn"},
                               "usage": {"input_tokens": 1000, "output_tokens": 10000,
                                         "iterations": [
                                             {"type": "refusal_message",
                                              "usage": {"input_tokens": 1000,
                                                        "output_tokens": 2000}},
                                             {"type": "fallback_message",
                                              "usage": {"input_tokens": 1000,
                                                        "output_tokens": 8000}}]}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        t = FakeTransport([lambda: FakeStream(chunks=[sse(records)])])
        events = list(client(t, max_retries=0).stream(
            model="fable", messages=[{"role": "user", "content": "hi"}]))
        return [e for e in events if e.kind == "done"][0].usage

    def test_each_attempt_is_priced_at_the_model_that_ran_it(self):
        usage = self.usage()
        declined = Usage(input_tokens=1000, output_tokens=2000)
        rescued = Usage(input_tokens=1000, output_tokens=8000)
        truth = declined.cost("claude-fable-5") + rescued.cost("claude-opus-4-8")
        self.assertAlmostEqual(truth, 0.3150)
        # Whichever end the caller names, the answer is what the server billed.
        self.assertAlmostEqual(usage.cost("claude-fable-5"), truth)
        self.assertAlmostEqual(usage.cost("claude-opus-4-8"), truth)
        self.assertAlmostEqual(usage.cost("claude-fable-5", served="claude-opus-4-8"),
                               truth)
        # Pricing the cumulative total at one model is what used to happen.
        flat = Usage(input_tokens=1000, output_tokens=10000)
        self.assertLess(flat.cost("claude-opus-4-8"), truth)
        self.assertGreater(flat.cost("claude-fable-5"), truth)

    def test_the_served_model_travels_with_the_usage(self):
        usage = self.usage()
        self.assertEqual(usage.served_model, "claude-opus-4-8")
        self.assertTrue(usage.served_by_fallback)
        restored = Usage.from_dict(json.loads(json.dumps(usage.as_dict())))
        self.assertEqual(restored, usage)
        self.assertAlmostEqual(restored.cost("claude-fable-5"), usage.cost("claude-fable-5"))

    def test_an_unknown_fallback_model_does_not_crash_the_footer(self):
        usage = Usage(input_tokens=10, output_tokens=10, served_model="claude-opus-9",
                      iterations=({"type": "fallback_message", "model": "claude-opus-9",
                                   "usage": {"input_tokens": 10, "output_tokens": 10}},))
        self.assertAlmostEqual(usage.cost("claude-opus-5"),
                               Usage(input_tokens=10, output_tokens=10).cost("claude-opus-5"))

    def test_summing_turns_keeps_the_fallback_signal(self):
        # `served_by_fallback` is read off `iterations`; dropping them in __add__
        # makes a conversation's running total quietly forget it ever fell back,
        # and takes the per-attempt pricing with it.
        a = Usage(input_tokens=1, iterations=({"type": "fallback_message"},))
        b = Usage(input_tokens=2)
        self.assertTrue((a + b).served_by_fallback)
        self.assertTrue((b + a).served_by_fallback)
        self.assertTrue(sum([b, a], Usage()).served_by_fallback)
        self.assertEqual(len((a + a).iterations), 2)


class TestUsageArithmeticEdges(unittest.TestCase):
    def test_bare_sum_starts_at_zero(self):
        # sum() begins with 0, so `0 + Usage` has to work; __radd__ = __add__
        # returns NotImplemented and the whole call raises TypeError.
        a = Usage(1, 2, 3, 4)
        b = Usage(10, 20, 30, 40)
        self.assertEqual(sum([a, b]), Usage(11, 22, 33, 44))
        self.assertEqual(sum([]), 0)
        self.assertEqual(sum([a]), a)
        with self.assertRaises(TypeError):
            a + 3
        with self.assertRaises(TypeError):
            3 + a


class TestPricingCalendar(unittest.TestCase):
    def test_sonnet_5_intro_pricing_expires_by_itself(self):
        spec = resolve_model("claude-sonnet-5")
        self.assertEqual(spec.prices("2026-08-17"), (2.0, 10.0))
        self.assertEqual(spec.prices("2026-08-31"), (2.0, 10.0))
        self.assertEqual(spec.prices("2026-09-01"), (3.0, 15.0))
        # The list price stays on the spec so the table does not go stale.
        self.assertEqual((spec.price_in, spec.price_out), (3.0, 15.0))

    def test_models_without_intro_pricing_are_flat(self):
        for model in ("claude-opus-5", "claude-fable-5", "claude-haiku-4-5"):
            spec = resolve_model(model)
            self.assertEqual(spec.prices("2026-01-01"), spec.prices("2027-01-01"))
            self.assertEqual(spec.prices(), (spec.price_in, spec.price_out))

    def test_a_turn_is_priced_at_the_rate_in_force_that_day(self):
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(u.cost("sonnet", when="2026-08-31"), 12.0)
        self.assertAlmostEqual(u.cost("sonnet", when="2026-09-01"), 18.0)
        self.assertAlmostEqual(u.cost("sonnet", when=datetime.date(2026, 8, 31)), 12.0)


class TestCacheWritePricing(unittest.TestCase):
    def test_a_one_hour_write_costs_twice_input_not_1_25x(self):
        u = Usage(cache_creation_input_tokens=1_000_000)
        self.assertAlmostEqual(u.cost("opus"), 6.25)
        self.assertAlmostEqual(u.cost("opus", cache_ttl="1h"), 10.0)

    def test_a_per_ttl_breakdown_from_the_server_wins(self):
        u = Usage(cache_creation_input_tokens=1_000_000,
                  cache_creation={"ephemeral_5m_input_tokens": 400_000,
                                  "ephemeral_1h_input_tokens": 600_000})
        self.assertAlmostEqual(u.cost("opus"), (0.4 * 6.25) + (0.6 * 10.0))
        # And it survives the round trip through the store.
        again = Usage.from_dict(json.loads(json.dumps(u.as_dict())))
        self.assertEqual(again, u)
        self.assertAlmostEqual(again.cost("opus"), u.cost("opus"))

    def test_the_breakdown_is_merged_off_the_wire(self):
        records = [message_start(usage={"input_tokens": 5, "output_tokens": 0,
                                        "cache_creation_input_tokens": 1000,
                                        "cache_creation": {
                                            "ephemeral_1h_input_tokens": 1000}}),
                   ("message_delta", {"type": "message_delta",
                                      "delta": {"stop_reason": "end_turn"},
                                      "usage": {"output_tokens": 1}}),
                   ("message_stop", {"type": "message_stop"})]
        t = FakeTransport([lambda: FakeStream(chunks=[sse(records)])])
        events = list(client(t).stream(model="opus",
                                       messages=[{"role": "user", "content": "hi"}]))
        usage = events[-1].usage
        self.assertEqual(usage.cache_creation, {"ephemeral_1h_input_tokens": 1000})
        self.assertAlmostEqual(usage.cost("opus"),
                               (5 * 5.0 + 1 * 25.0 + 1000 * 10.0) / 1e6)


# ---------------------------------------------------------------- errors and secrets


class TestInternalErrors(unittest.TestCase):
    def test_a_bug_in_the_translator_is_not_a_network_error(self):
        # A TypeError in lume's own code reported as a NetworkError sends the user
        # to check their wifi, and gets retried four times on the way.
        import lume.api as api

        original = api._translate
        self.addCleanup(setattr, api, "_translate", original)

        def broken(state, data):
            raise TypeError("programming bug in _translate")

        api._translate = broken
        t = FakeTransport([lambda: FakeStream(chunks=[sse(text_stream())])])
        with self.assertRaises(TypeError):
            list(client(t, max_retries=3).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 1, "a programming error was retried")

    def test_a_network_shaped_failure_is_still_an_api_error(self):
        t = FakeTransport([OSError("connection reset")])
        with self.assertRaises(NetworkError):
            list(client(t, max_retries=0).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))

    def test_transport_error_tags_a_bug_as_internal(self):
        self.assertTrue(_transport_error(TypeError("x")).internal)
        self.assertFalse(_transport_error(OSError("x")).internal)
        self.assertFalse(_transport_error(NetworkError("x")).internal)
        self.assertFalse(NetworkError("x").internal)


class TestExceptionChainSecrecy(unittest.TestCase):
    def test_the_key_does_not_survive_in_the_exception_context(self):
        # A third-party transport that formats its headers into an error message.
        # `raise ... from None` clears __cause__ only; __context__ still points at
        # the original, and a formatter that walks the chain prints the key.
        key = KEY

        class LeakyTransport:
            def open(self, url, headers, body, timeout):
                raise RuntimeError(f"auth rejected for {headers['x-api-key']}")

        c = Client(key, base_url="http://127.0.0.1:1", transport=LeakyTransport(),
                   max_retries=0)
        with self.assertRaises(NetworkError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        exc = ctx.exception
        self.assertIsNone(exc.__cause__)
        self.assertIsNone(exc.__context__)
        self.assertNotIn("TESTKEY", exc.message)
        report = "".join(traceback.TracebackException.from_exception(
            exc, capture_locals=True).format())
        self.assertNotIn("TESTKEY", report)
        self.assertNotIn(key, report)

    def test_a_non_sk_ant_token_is_cut_out_of_the_chain_too(self):
        token = "hunter2-corporate-gateway-token"

        class LeakyTransport:
            def open(self, url, headers, body, timeout):
                raise RuntimeError(f"rejected: {headers['x-api-key']}")

        c = Client(token, base_url="http://127.0.0.1:1", transport=LeakyTransport(),
                   max_retries=0)
        with self.assertRaises(NetworkError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertNotIn(token, ctx.exception.message)
        self.assertIsNone(ctx.exception.__context__)
        report = "".join(traceback.TracebackException.from_exception(
            ctx.exception, capture_locals=True).format())
        self.assertNotIn(token, report)


class TestChainCutFromTheCallersFrame(unittest.TestCase):
    """`__context__` is attached at *raise* time, from whatever is being handled.

    Detaching the error before raising it is not enough: the raise re-attaches the
    exception live in the *caller's* frame, and the app drives `stream()` from
    inside its own `except` blocks. So a chain-walking crash reporter still ends up
    in the app's frames — with the request, the headers and the key among the
    locals it prints.
    """

    MARKER = "outer-frame-marker-9f3a"

    def failing_client(self):
        class LeakyTransport:
            def open(self, url, headers, body, timeout):
                raise RuntimeError("upstream refused the connection")

        return Client(KEY, base_url="http://127.0.0.1:1", transport=LeakyTransport(),
                      max_retries=0)

    def error_raised_while_handling(self):
        # The key is never a local of this frame — only the module constant — so a
        # failure here is the chain, not the test tripping over its own fixture.
        c = self.failing_client()
        try:
            raise ValueError(self.MARKER)
        except ValueError:
            try:
                list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
            except NetworkError as exc:
                return exc
        raise AssertionError("the stream did not fail")

    def test_the_callers_own_exception_is_not_chained_on(self):
        exc = self.error_raised_while_handling()
        self.assertIsNone(exc.__cause__)
        self.assertIsNone(exc.__context__,
                          "the caller's exception is still on the chain")
        self.assertTrue(exc.__suppress_context__)

    def test_a_chain_walking_crash_report_finds_nothing_to_walk(self):
        exc = self.error_raised_while_handling()
        report = "".join(traceback.TracebackException.from_exception(
            exc, capture_locals=True).format())
        self.assertNotIn(self.MARKER, report)
        self.assertNotIn(KEY, report)
        self.assertNotIn("TESTKEY", report)

    def test_the_same_holds_for_a_transport_level_failure(self):
        # The connect path builds its error in the handler and raises it past —
        # the same cut has to happen there.
        transport = HTTPTransport()
        self.addCleanup(transport.close)
        try:
            raise ValueError(self.MARKER)
        except ValueError:
            with self.assertRaises(NetworkError) as ctx:
                transport.open("http://127.0.0.1:1/v1/messages",
                               {"x-api-key": KEY}, b"{}", 1.0)
        # Asserted here, at the seam, and not only on what the Client re-raises:
        # `Client._run` detaches everything on its way out, so a transport that
        # chains its own failure to `http.client`'s request serialiser — headers,
        # key and all — is invisible one layer up.
        self.assertIsNone(ctx.exception.__cause__,
                          "the transport chained its own failure to the connect")
        self.assertIsNone(ctx.exception.__context__)
        report = "".join(traceback.TracebackException.from_exception(
            ctx.exception, capture_locals=True).format())
        self.assertNotIn(self.MARKER, report)
        self.assertNotIn("TESTKEY", report)


class TestBearerRedaction(unittest.TestCase):
    def test_an_oauth_bearer_header_is_redacted(self):
        # OAuth tokens do not match the sk-ant- shape once they are behind
        # "Bearer ", and the whole OAuth path went through this rule alone.
        line = "authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature"
        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", redact(line))
        self.assertIn("Bearer ***", redact(line))
        self.assertEqual(redact("bearer abcdefgh"), "Bearer ***")

    def test_an_oauth_token_never_reaches_the_message(self):
        class LeakyTransport:
            def open(self, url, headers, body, timeout):
                raise RuntimeError(f"rejected {headers['authorization']}")

        c = Client(OAUTH, base_url="http://127.0.0.1:1", transport=LeakyTransport(),
                   max_retries=0)
        with self.assertRaises(NetworkError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertNotIn(OAUTH, ctx.exception.message)
        self.assertNotIn("TESTTOKEN", ctx.exception.message)

    def test_a_bearer_token_in_a_server_error_body_is_redacted(self):
        body = json.dumps({"error": {"type": "authentication_error",
                                     "message": "Bearer sk-ant-oat01-LEAKED-0000 is bad"}})
        t = FakeTransport([lambda: FakeStream(status=401, body=body.encode())])
        with self.assertRaises(AuthError) as ctx:
            list(client(t, max_retries=0).stream(
                model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertNotIn("LEAKED", ctx.exception.message)



class _QuietSDKStream:
    """An SDK stream that produces nothing — a long think before the first token."""

    def __init__(self, owner):
        self._owner = owner
        self.gate = threading.Event()
        self.blocked = threading.Event()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._owner.exited = True
        return False

    def __iter__(self):
        self.blocked.set()
        if not self.gate.wait(10):
            raise AssertionError("the quiet SDK stream was never interrupted")
        return iter(())

    def close(self):
        self.closed = True
        self.gate.set()


class TestSDKTransportCancellation(unittest.TestCase):
    """`transport="auto"` must not silently downgrade cancellation.

    Checking the cancel flag between events cannot fire on a stream that has not
    produced an event yet, which is exactly when a user gives up and hits Ctrl-C.
    """

    def setUp(self):
        _FakeAnthropic.instances = []
        module = types.ModuleType("anthropic")
        module.Anthropic = _FakeAnthropic
        self._saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = module
        self.addCleanup(self._restore)
        self.streams = []

        def quiet(_self, **kwargs):
            stream = _QuietSDKStream(_self._owner)
            self.streams.append(stream)
            return stream

        self._saved_stream = _FakeMessages.stream
        _FakeMessages.stream = quiet
        self.addCleanup(setattr, _FakeMessages, "stream", self._saved_stream)

    def _restore(self):
        if self._saved is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = self._saved

    def test_a_silent_sdk_stream_is_still_cancellable(self):
        c = Client(KEY, base_url="https://example.invalid", transport="auto",
                   max_retries=3)
        self.addCleanup(c.close)
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        box = {}

        def consume():
            try:
                list(gen)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                box["exc"] = exc

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        for _ in range(500):
            if self.streams and self.streams[0].blocked.is_set():
                break
            time.sleep(0.01)
        self.assertTrue(self.streams and self.streams[0].blocked.is_set(),
                        "the SDK stream never started")
        started = time.monotonic()
        cancel.set()
        worker.join(5)
        elapsed = time.monotonic() - started
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("exc"), CancelledError,
                              f"got {box.get('exc')!r}")
        self.assertLess(elapsed, 1.0, f"the SDK cancel took {elapsed:.2f}s")
        self.assertTrue(self.streams[0].closed)
        self.assertTrue(_FakeAnthropic.instances[-1].exited, "the SDK stream leaked")



# ------------------------------------------------ round-3 regression defences
#
# Every test below was watched failing with its fix reverted; the reverts live in
# scratchpad/r3_reverts.py. They exist because the whole suite stayed green
# through the corresponding sabotage.


class CancelAwareTransport(FakeTransport):
    """A transport that takes the `cancel` argument, exactly as HTTPTransport does.

    It deliberately ignores the event: the point is that `Client` must not hand it
    a request the user has already cancelled in the first place.
    """

    def open(self, url, headers, body, timeout, cancel=None):
        return FakeTransport.open(self, url, headers, body, timeout)


class TestPreRequestCancel(unittest.TestCase):
    """A request the user cancelled before it went out must never be sent.

    Not a latency question: an outbound request is billed whatever the client does
    with the answer, so "we cancelled it on the way back" is not the same thing.
    """

    def test_a_cancel_aware_transport_is_never_called(self):
        t = CancelAwareTransport([lambda: FakeStream(chunks=[sse(text_stream())])])
        cancel = threading.Event()
        cancel.set()
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            list(client(t).stream(model="opus",
                                  messages=[{"role": "user", "content": "hi"}],
                                  cancel=cancel))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(t.calls, [], "a cancelled request was sent anyway")

    def test_a_cancel_unaware_transport_is_never_called_either(self):
        # The same guarantee through the supervised path, where `open()` runs on a
        # helper thread and would otherwise have fired before anyone looked.
        t = FakeTransport([lambda: FakeStream(chunks=[sse(text_stream())])])
        cancel = threading.Event()
        cancel.set()
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            list(client(t).stream(model="opus",
                                  messages=[{"role": "user", "content": "hi"}],
                                  cancel=cancel))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(t.calls, [], "a cancelled request was sent anyway")

    def test_an_already_cancelled_open_never_reaches_the_server(self):
        # Straight at the transport, over a real socket. `timeout` is short so
        # that a transport which ignores the flag fails the latency bound instead
        # of hanging the suite.
        server = SilentSocketServer()
        self.addCleanup(server.close)
        transport = HTTPTransport()
        self.addCleanup(transport.close)
        cancel = threading.Event()
        cancel.set()
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            transport.open(f"{server.url}/v1/messages",
                           {"content-type": "application/json"}, b"{}", 3.0,
                           cancel=cancel)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5, f"a cancelled connect took {elapsed:.2f}s")
        time.sleep(0.2)
        self.assertEqual(server.requests, [],
                         "a cancelled request still went out on the wire")
        self.assertEqual(len(transport._live), 0,
                         "the abandoned attempt was retained by the transport")


class TestCancelledConnectIsNotRetryable(unittest.TestCase):
    """Torn down by our own watchdog, a socket error is a cancel wearing a disguise.

    Straight at the transport seam, because `Client._pause` masks this end to end:
    it refuses the retry on the way past, so a retryable error looks harmless from
    outside while the classification underneath is wrong.
    """

    def test_a_connect_we_tore_down_comes_back_as_a_cancel(self):
        server = SilentSocketServer()
        self.addCleanup(server.close)
        transport = HTTPTransport()
        self.addCleanup(transport.close)
        cancel = threading.Event()
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError) as ctx:
            transport.open(f"{server.url}/v1/messages",
                           {"content-type": "application/json"}, b"{}", 30.0,
                           cancel=cancel)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0, f"the cancel took {elapsed:.2f}s to be noticed")
        self.assertFalse(ctx.exception.retryable,
                         "a cancelled request came back retryable and would be resent")
        self.assertEqual(len(transport._live), 0)

    def test_and_the_client_above_it_does_not_resend(self):
        server = SilentSocketServer()
        self.addCleanup(server.close)
        c = Client(KEY, base_url=server.url, timeout=30.0, max_retries=4)
        self.addCleanup(c.close)
        c.backoff_base = c.backoff_max = 0.0
        cancel = threading.Event()
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                          cancel=cancel))
        self.assertLess(time.monotonic() - started, 2.0)
        time.sleep(0.3)
        self.assertEqual(len(server.requests), 1,
                         f"the cancelled request went out {len(server.requests)} times")


class TestBackoffCancel(unittest.TestCase):
    """`_pause` must *raise*, not merely stop waiting.

    Deleting the raise leaves every end-to-end cancellation test green — the next
    attempt's own pre-request check picks the cancel up a moment later — while the
    retry that was supposed to be abandoned has already been decided on. These
    tests pin the raise to `_pause` itself.
    """

    def paused(self, delay, cancel):
        c = client(ok_transport())
        started = time.monotonic()
        with self.assertRaises(CancelledError) as ctx:
            c._pause(delay, cancel)
        return time.monotonic() - started, ctx.exception

    def test_an_already_set_event_raises_at_once(self):
        cancel = threading.Event()
        cancel.set()
        elapsed, exc = self.paused(30.0, cancel)
        self.assertLess(elapsed, 0.5, f"_pause sat on a set cancel for {elapsed:.2f}s")
        self.assertIn("backing off", exc.message)
        self.assertFalse(exc.retryable)

    def test_an_event_set_mid_wait_raises_without_serving_out_the_delay(self):
        cancel = threading.Event()
        timer = threading.Timer(0.05, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        elapsed, _ = self.paused(30.0, cancel)
        self.assertLess(elapsed, 1.0, f"_pause waited {elapsed:.2f}s past the cancel")

    def test_the_backoff_still_happens_when_nothing_is_cancelling(self):
        # The other direction: _pause must not quietly become a no-op.
        c = client(ok_transport())
        slept = []
        c._sleep = slept.append
        c._pause(2.5, None)
        self.assertEqual(slept, [2.5])

    def test_the_cancel_comes_from_the_backoff_not_from_the_next_attempt(self):
        # End to end. The message says which guard fired, so a `_pause` that only
        # stops waiting cannot pass by letting `_attempt` raise instead.
        t = FakeTransport([lambda: FakeStream(status=529,
                                              body=err_body("overloaded_error"))])
        c = client(t, max_retries=3)
        c.backoff_base = c.backoff_max = 30.0
        c._sleep = time.sleep  # a real sleep, so only the event can cut it short
        cancel = threading.Event()
        gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}],
                       cancel=cancel)
        timer = threading.Timer(0.05, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError) as ctx:
            list(gen)
        elapsed = time.monotonic() - started
        self.assertIn("backing off", ctx.exception.message)
        self.assertLess(elapsed, 1.0, f"the backoff ignored the cancel for {elapsed:.2f}s")
        self.assertEqual(len(t.calls), 1, "a cancelled request was retried anyway")


class TestEffortLadderAcrossTheTable(unittest.TestCase):
    """Not two hand-picked models: every model in MODELS, at every effort."""

    def body(self, **kw):
        kw.setdefault("messages", [{"role": "user", "content": "hi"}])
        return Client(KEY).build_body(**kw)

    def test_no_model_is_ever_sent_an_effort_it_does_not_have(self):
        for mid, spec in MODELS.items():
            for asked in EFFORTS:
                config = self.body(model=mid, effort=asked).get("output_config")
                if not spec.supports_effort:
                    self.assertIsNone(config, mid)
                    continue
                sent = config["effort"]
                self.assertIn(sent, spec.efforts, f"{mid} was sent {sent!r}")
                self.assertLessEqual(EFFORTS.index(sent), EFFORTS.index(asked),
                                     f"{mid}: asked {asked!r}, sent the higher {sent!r}")

    def test_xhigh_lands_on_high_wherever_xhigh_does_not_exist(self):
        for mid, spec in MODELS.items():
            if not spec.supports_effort:
                continue
            sent = self.body(model=mid, effort="xhigh")["output_config"]["effort"]
            self.assertEqual(sent, "xhigh" if "xhigh" in spec.efforts else "high", mid)

    def test_the_ladder_is_per_model_not_one_global_list(self):
        ladders = {tuple(spec.efforts) for spec in MODELS.values() if spec.supports_effort}
        self.assertGreater(len(ladders), 1,
                           "every model shares one ladder; the per-model one is gone")
        self.assertIn("xhigh", MODELS["claude-opus-4-7"].efforts)
        self.assertNotIn("xhigh", MODELS["claude-opus-4-6"].efforts)


class TestLoneCRAtEndOfStream(unittest.TestCase):
    """A record whose only terminator is a CR at EOF is complete, not truncated."""

    def test_close_flushes_a_record_the_feed_held_back(self):
        decoder = SSEDecoder()
        # feed() holds the trailing CR back: it may be the first half of a CRLF.
        self.assertEqual(decoder.feed(b"event: message_stop\rdata: {}\r\r"), [])
        self.assertEqual(decoder.close(), [("message_stop", "{}")])

    def test_earlier_records_still_arrive_during_the_feed(self):
        decoder = SSEDecoder()
        raw = b"event: ping\rdata: {}\r\revent: message_stop\rdata: {}\r\r"
        self.assertEqual(decoder.feed(raw), [("ping", "{}")])
        self.assertEqual(decoder.close(), [("message_stop", "{}")])

    def test_close_is_idempotent_and_releases_nothing_twice(self):
        decoder = SSEDecoder()
        decoder.feed(b"event: message_stop\rdata: {}\r\r")
        self.assertEqual(len(decoder.close()), 1)
        self.assertEqual(decoder.close(), [])

    def test_a_cr_terminated_stream_completes_over_the_client(self):
        raw = sse(text_stream(), eol="\r")
        t = FakeTransport([lambda: FakeStream(chunks=[raw])])
        events = list(client(t, max_retries=0).stream(
            model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(drain(events)[0], "Hello, wörld…")
        self.assertEqual(events[-1].kind, "done")

    def test_an_unterminated_final_record_is_still_dropped(self):
        # The other half of the same rule: close() flushes a *terminated* record,
        # never one the connection cut in half.
        decoder = SSEDecoder()
        decoder.feed(b"event: message_stop\ndata: {}\n")
        self.assertEqual(decoder.close(), [])


class TestSummedUsagePricing(unittest.TestCase):
    """`cost()` on a summed Usage must not lose the turns that carry no iterations."""

    def rescued(self):
        return Usage(input_tokens=1000, output_tokens=10000,
                     served_model="claude-opus-4-8",
                     iterations=({"type": "refusal_message",
                                  "model": "claude-fable-5",
                                  "usage": {"input_tokens": 1000,
                                            "output_tokens": 2000}},
                                 {"type": "fallback_message",
                                  "model": "claude-opus-4-8",
                                  "usage": {"input_tokens": 1000,
                                            "output_tokens": 8000}}))

    def test_one_rescued_turn_is_still_priced_per_hop(self):
        self.assertAlmostEqual(self.rescued().cost("claude-fable-5"), 0.3150)

    def test_an_ordinary_turn_added_to_it_is_not_dropped(self):
        plain = Usage(input_tokens=500, output_tokens=1000)
        total = (self.rescued() + plain).cost("claude-fable-5")
        self.assertAlmostEqual(total, 0.3150 + plain.cost("claude-opus-4-8"))
        self.assertGreater(total, self.rescued().cost("claude-fable-5"))

    def test_app_style_per_turn_accounting_agrees_with_the_summed_one(self):
        # app.py adds up `usage.cost(served)` per turn. The two roads have to meet.
        turns = [self.rescued(), Usage(input_tokens=500, output_tokens=1000),
                 Usage(input_tokens=20, output_tokens=30)]
        per_turn = sum(u.cost(u.served_model or "claude-opus-4-8") for u in turns)
        self.assertAlmostEqual(sum(turns, Usage()).cost("claude-opus-4-8"), per_turn)


class TestUsageIterationsPricing(unittest.TestCase):
    """The critic measured $0.3150 billed against $0.2550 reported."""

    def test_the_flat_price_that_used_to_be_reported_is_the_wrong_one(self):
        flat = Usage(input_tokens=1000, output_tokens=10000)
        self.assertAlmostEqual(flat.cost("claude-opus-4-8"), 0.2550)
        self.assertNotAlmostEqual(flat.cost("claude-opus-4-8"), 0.3150)

    def test_naming_either_end_of_the_fallback_gives_the_billed_figure(self):
        usage = Usage(input_tokens=1000, output_tokens=10000,
                      served_model="claude-opus-4-8",
                      iterations=({"type": "refusal_message",
                                   "model": "claude-fable-5",
                                   "usage": {"input_tokens": 1000,
                                             "output_tokens": 2000}},
                                  {"type": "fallback_message",
                                   "model": "claude-opus-4-8",
                                   "usage": {"input_tokens": 1000,
                                             "output_tokens": 8000}}))
        for named in ("claude-fable-5", "claude-opus-4-8"):
            self.assertAlmostEqual(usage.cost(named), 0.3150, msg=named)


class TestIntroductoryPricingFrozenClock(unittest.TestCase):
    """Never `date.today()`: a test that only passes this month is not a test."""

    def test_the_boundary_is_the_last_day_inclusive(self):
        spec = resolve_model("claude-sonnet-5")
        self.assertEqual(spec.intro_until, "2026-08-31")
        for day in ("2026-01-01", "2026-08-30", "2026-08-31"):
            self.assertEqual(spec.prices(day), (2.0, 10.0), day)
        for day in ("2026-09-01", "2027-01-01"):
            self.assertEqual(spec.prices(day), (3.0, 15.0), day)

    def test_a_turn_costs_what_it_costs_on_the_day_it_ran(self):
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        before = u.cost("sonnet", when=datetime.date(2026, 8, 31))
        after = u.cost("sonnet", when=datetime.date(2026, 9, 1))
        self.assertAlmostEqual(before, 12.0)
        self.assertAlmostEqual(after, 18.0)
        self.assertLess(before, after)

    def test_today_follows_the_same_rule_whatever_today_is(self):
        spec = resolve_model("claude-sonnet-5")
        today = datetime.date.today()
        expected = (2.0, 10.0) if today <= datetime.date(2026, 8, 31) else (3.0, 15.0)
        self.assertEqual(spec.prices(), expected)

    def test_cache_prices_follow_the_rate_in_force_too(self):
        # `spec.price_cache_write` is derived from the list rate and is only ever
        # informational; what is charged is 1.25x (or 2x at an hour) of whatever
        # input costs *that day*.
        write = Usage(cache_creation_input_tokens=1_000_000)
        read = Usage(cache_read_input_tokens=1_000_000)
        self.assertAlmostEqual(write.cost("sonnet", when="2026-08-31"), 2.5)
        self.assertAlmostEqual(write.cost("sonnet", when="2026-09-01"), 3.75)
        self.assertAlmostEqual(write.cost("sonnet", when="2026-08-31", cache_ttl="1h"), 4.0)
        self.assertAlmostEqual(read.cost("sonnet", when="2026-08-31"), 0.2)
        self.assertAlmostEqual(read.cost("sonnet", when="2026-09-01"), 0.3)

    def test_no_other_model_carries_an_intro_rate(self):
        for mid, spec in MODELS.items():
            if mid == "claude-sonnet-5":
                continue
            self.assertIsNone(spec.intro_until, mid)
            self.assertEqual(spec.prices("2026-01-01"), (spec.price_in, spec.price_out), mid)


class TestTransportLiveSetIsBounded(unittest.TestCase):
    """`_live` retains a connection, a response and a socket for each entry."""

    def test_a_session_of_turns_leaves_nothing_behind(self):
        server, url = LiveServerCase.serve(self, [(200, {}, [sse(text_stream())])] * 8)
        transport = HTTPTransport()
        c = Client(KEY, base_url=url, timeout=10, transport=transport)
        self.addCleanup(c.close)
        for _ in range(6):
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(transport._live), 0,
                         f"{len(transport._live)} connections retained after 6 turns")

    def test_a_stream_abandoned_early_is_forgotten_too(self):
        server, url = LiveServerCase.serve(self, [(200, {}, [sse(text_stream())])] * 4)
        transport = HTTPTransport()
        c = Client(KEY, base_url=url, timeout=10, transport=transport)
        self.addCleanup(c.close)
        for _ in range(3):
            gen = c.stream(model="opus", messages=[{"role": "user", "content": "hi"}])
            next(gen)
            gen.close()
        self.assertEqual(len(transport._live), 0, transport._live)


class TestBearerRedactionIsReachable(unittest.TestCase):
    """The OAuth path goes through `_BEARER_RE` alone — `sk-ant-` never matches it."""

    def test_a_bearer_value_with_no_sk_ant_shape_is_still_blanked(self):
        for token in ("eyJhbGciOiJIUzI1NiJ9.payload.signature",
                      "abcdefghijklmnop", "A1b2C3d4~+/=-_."):
            line = f"authorization: Bearer {token}"
            self.assertNotIn(token, redact(line), token)
            self.assertIn("Bearer ***", redact(line))

    def test_the_sk_ant_rule_alone_would_not_have_caught_it(self):
        # Pins *why* the Bearer rule exists: the OAuth token shape is not the key
        # shape, so a suite that only checks sk-ant- proves nothing about it.
        token = "eyJhbGciOiJIUzI1NiJ9.payload.signature"
        self.assertIsNone(re.search(r"sk-ant-", token))
        self.assertNotIn(token, redact(f"Bearer {token}"))

    def test_an_oauth_client_never_leaks_its_header(self):
        class LeakyTransport:
            def open(self, url, headers, body, timeout):
                raise RuntimeError(f"upstream rejected {headers['authorization']}")

        c = Client(OAUTH, base_url="http://127.0.0.1:1", transport=LeakyTransport(),
                   max_retries=0)
        with self.assertRaises(NetworkError) as ctx:
            list(c.stream(model="opus", messages=[{"role": "user", "content": "hi"}]))
        report = "".join(traceback.TracebackException.from_exception(
            ctx.exception, capture_locals=True).format())
        self.assertNotIn(OAUTH, report)
        self.assertNotIn("TESTTOKEN", report)


class TestFallbackSignalSurvivesArithmetic(unittest.TestCase):
    """`served_by_fallback` is read off `iterations`; __add__ must carry them."""

    def test_two_turns_summed_still_remember_the_fallback(self):
        rescued = Usage(input_tokens=1, iterations=({"type": "fallback_message"},))
        plain = Usage(input_tokens=2)
        for total in (rescued + plain, plain + rescued,
                      sum([plain, rescued], Usage()), sum([plain, rescued])):
            self.assertTrue(total.served_by_fallback, total)

    def test_a_whole_conversation_of_turns_keeps_it(self):
        turns = [Usage(input_tokens=1) for _ in range(5)]
        turns[2] = Usage(input_tokens=1, iterations=({"type": "fallback_message"},))
        self.assertTrue(sum(turns, Usage()).served_by_fallback)
        self.assertEqual(len(sum(turns, Usage()).iterations), 1)

    def test_it_survives_the_store_round_trip(self):
        rescued = Usage(input_tokens=1, iterations=({"type": "fallback_message"},))
        total = rescued + Usage(input_tokens=2)
        again = Usage.from_dict(json.loads(json.dumps(total.as_dict())))
        self.assertEqual(again, total)
        self.assertTrue(again.served_by_fallback)


if __name__ == "__main__":
    unittest.main()
