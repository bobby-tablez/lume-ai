"""Tests for lume.providers.openai — canned transports only, never a socket.

Nothing here talks to api.openai.com (or to Groq, or to anything else): there is
no key in this environment and a real call would cost money. Every test injects a
transport, and the one test that looks at the default transport only checks that
it was constructed, never opened.
"""

import json
import os
import re
import threading
import time
import unittest

import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))   # the repo root, for `lume`
sys.path.insert(0, _TESTS)                    # this directory, for `test_api`

from test_api import FakeStream, FakeTransport, sse  # noqa: E402

from lume import providers  # noqa: E402
from lume.api import (APIError, AuthError, BadRequestError, CancelledError,  # noqa: E402
                      HTTPTransport, ModelSpec, NetworkError, RateLimitError,
                      ServerError, StreamEvent, Usage)
from lume.providers.openai import (ALIASES, DEFAULT_BASE_URL, DEFAULT_MODEL,  # noqa: E402
                                   FINISH_REASONS, MODELS, OpenAIClient,
                                   model_names, provider, resolve_model,
                                   _flatten, _off_effort, _uses_max_completion_tokens,
                                   _usage_from_payload)

KEY = "sk-proj-TESTKEY-not-a-real-credential-0000"


# --------------------------------------------------------------------------- helpers


def wire(payloads, done=True, eol="\n"):
    """Render chat-completion chunks the way the API really sends them.

    No `event:` field — OpenAI sends bare `data:` records — and the `[DONE]`
    sentinel that terminates the stream.
    """
    out = []
    for payload in payloads:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        out.append(f"data: {body}{eol}{eol}")
    if done:
        out.append(f"data: [DONE]{eol}{eol}")
    return "".join(out).encode("utf-8")


def chunk(delta=None, *, finish=None, model="gpt-5.1", usage=None, index=0):
    """One `chat.completion.chunk` record."""
    payload = {"id": "chatcmpl-TEST", "object": "chat.completion.chunk",
               "created": 1770000000, "model": model, "choices": []}
    if delta is not None or finish is not None:
        payload["choices"] = [{"index": index, "delta": dict(delta or {}),
                               "finish_reason": finish}]
    if usage is not None:
        payload["usage"] = usage
    return payload


USAGE = {"prompt_tokens": 1200, "completion_tokens": 300,
         "prompt_tokens_details": {"cached_tokens": 1000},
         "completion_tokens_details": {"reasoning_tokens": 128}}


def turn(pieces=("Hel", "lo, ", "wörld…"), thinking=("Let me ", "think. "),
         finish="stop", usage=USAGE, model="gpt-5.1"):
    """A whole assistant turn: reasoning, text, finish_reason, usage."""
    records = [chunk({"role": "assistant", "content": ""}, model=model)]
    for t in thinking:
        records.append(chunk({"reasoning_content": t}, model=model))
    for p in pieces:
        records.append(chunk({"content": p}, model=model))
    records.append(chunk({}, finish=finish, model=model))
    if usage is not None:
        records.append(chunk(model=model, usage=usage))
    return records


def transport(records=None, chunk_size=None, raw=None):
    """A FakeTransport that replays one scripted stream."""
    body = raw if raw is not None else wire(records if records is not None else turn())
    if chunk_size:
        chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)]
    else:
        chunks = [body]
    return FakeTransport([lambda: FakeStream(chunks=list(chunks))])


def client(t, **kw):
    """A client wired to a canned transport, with the backoff taken out."""
    kw.setdefault("base_url", "http://127.0.0.1:1/never-used/v1")
    c = OpenAIClient(KEY, transport=t, **kw)
    c.backoff_base = 0.0
    c.backoff_max = 0.0
    c.slept = []
    c._sleep = c.slept.append
    return c


def run(t, **kw):
    """Drain one stream into a list of events."""
    kw.setdefault("model", "gpt-5.1")
    kw.setdefault("messages", [{"role": "user", "content": "hi"}])
    return list(client(t).stream(**kw))


def body_of(t):
    return t.calls[0]["body"]


def err_body(message="boom", type="invalid_request_error", code=None):
    return json.dumps({"error": {"message": message, "type": type,
                                 "code": code, "param": None}}).encode("utf-8")


def texts(events, kind="text"):
    return "".join(e.text for e in events if e.kind == kind)


# ---------------------------------------------------------------------------- models


class TestModels(unittest.TestCase):
    def test_the_table_has_a_flagship_a_mid_tier_and_a_small_model(self):
        self.assertEqual(DEFAULT_MODEL, "gpt-5.1")
        for mid in ("gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano",
                    "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"):
            self.assertIn(mid, MODELS, mid)
        self.assertEqual(model_names()[0], DEFAULT_MODEL)

    def test_every_spec_is_complete(self):
        for mid, spec in MODELS.items():
            self.assertIsInstance(spec, ModelSpec, mid)
            self.assertEqual(spec.id, mid)
            self.assertGreater(spec.context, 0, mid)
            self.assertGreater(spec.max_output, 0, mid)
            self.assertGreater(spec.price_in, 0, mid)
            self.assertGreater(spec.price_out, 0, mid)
            self.assertGreater(spec.price_cache_read, 0, mid)
            self.assertLess(spec.price_cache_read, spec.price_in, mid)
            self.assertTrue(spec.label, mid)

    def test_prices(self):
        want = {"gpt-5.1": (1.25, 10.0, 0.125), "gpt-5": (1.25, 10.0, 0.125),
                "gpt-5-mini": (0.25, 2.0, 0.025), "gpt-5-nano": (0.05, 0.40, 0.005),
                "gpt-4.1": (2.0, 8.0, 0.50), "gpt-4.1-mini": (0.40, 1.60, 0.10),
                "gpt-4o-mini": (0.15, 0.60, 0.075)}
        for mid, (pin, pout, cached) in want.items():
            spec = MODELS[mid]
            self.assertEqual((spec.price_in, spec.price_out), (pin, pout), mid)
            self.assertAlmostEqual(spec.price_cache_read, cached, msg=mid)
            # Writing to OpenAI's cache is free: the write rate is the input rate.
            self.assertEqual(spec.price_cache_write, spec.price_in, mid)

    def test_reasoning_and_chat_models_are_marked_apart(self):
        for mid in ("gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano"):
            spec = MODELS[mid]
            self.assertTrue(spec.supports_effort, mid)
            self.assertFalse(spec.supports_temperature, mid)
            self.assertTrue(_uses_max_completion_tokens(spec), mid)
        for mid in ("gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"):
            spec = MODELS[mid]
            self.assertFalse(spec.supports_effort, mid)
            self.assertTrue(spec.supports_temperature, mid)
            self.assertFalse(_uses_max_completion_tokens(spec), mid)

    def test_off_effort_comes_off_the_models_own_ladder(self):
        self.assertEqual(_off_effort(MODELS["gpt-5.1"]), "none")
        self.assertEqual(_off_effort(MODELS["gpt-5"]), "minimal")
        self.assertIsNone(_off_effort(MODELS["gpt-4.1"]))

    def test_aliases(self):
        for alias, expect in [("gpt", "gpt-5.1"), ("openai", "gpt-5.1"),
                              ("gpt5", "gpt-5"), ("mini", "gpt-5-mini"),
                              ("nano", "gpt-5-nano"), ("gpt4.1", "gpt-4.1"),
                              ("GPT", "gpt-5.1"), (" gpt-5 ", "gpt-5")]:
            self.assertEqual(resolve_model(alias).id, expect, alias)
        for alias, target in ALIASES.items():
            self.assertIn(target, MODELS, alias)

    def test_a_spec_passes_through_and_none_means_default(self):
        spec = MODELS["gpt-5"]
        self.assertIs(resolve_model(spec), spec)
        self.assertEqual(resolve_model(None).id, DEFAULT_MODEL)

    def test_an_unknown_model_gets_a_permissive_generic_spec(self):
        # The whole point of the compatible-endpoint story: Groq's llama id is
        # not in this table and must still work.
        spec = resolve_model("llama-3.3-70b-versatile")
        self.assertEqual(spec.id, "llama-3.3-70b-versatile")
        self.assertTrue(spec.supports_temperature)
        self.assertFalse(spec.supports_effort)
        self.assertFalse(_uses_max_completion_tokens(spec))
        self.assertEqual((spec.price_in, spec.price_out), (0.0, 0.0))


class TestCostMaths(unittest.TestCase):
    def test_a_turns_cost(self):
        spec = MODELS["gpt-5"]
        usage = _usage_from_payload(USAGE, "gpt-5")
        self.assertEqual((usage.input_tokens, usage.output_tokens,
                          usage.cache_read_input_tokens), (200, 300, 1000))
        expect = (200 * 1.25 + 300 * 10.0 + 1000 * 1.25 * 0.1) / 1e6
        self.assertAlmostEqual(usage.cost(spec), expect)

    def test_cached_tokens_are_taken_out_of_the_prompt_total(self):
        # prompt_tokens includes the cached prefix; input_tokens must not, or
        # every cached token is billed twice.
        usage = _usage_from_payload({"prompt_tokens": 1000, "completion_tokens": 0,
                                     "prompt_tokens_details": {"cached_tokens": 1000}})
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.cache_read_input_tokens, 1000)
        self.assertEqual(usage.total_tokens, 1000)

    def test_round_millions(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(usage.cost(MODELS["gpt-5.1"]), 11.25)
        self.assertAlmostEqual(usage.cost(MODELS["gpt-5-nano"]), 0.45)
        self.assertAlmostEqual(usage.cost(MODELS["gpt-4.1"]), 10.0)

    def test_missing_or_junk_usage_fields_cost_nothing(self):
        self.assertEqual(_usage_from_payload({}), Usage())
        usage = _usage_from_payload({"prompt_tokens": 10, "prompt_tokens_details": None})
        self.assertEqual((usage.input_tokens, usage.cache_read_input_tokens), (10, 0))
        # A server that reports more cached than prompt tokens must not go negative.
        usage = _usage_from_payload({"prompt_tokens": 5,
                                     "prompt_tokens_details": {"cached_tokens": 50}})
        self.assertEqual((usage.input_tokens, usage.cache_read_input_tokens), (0, 5))

    def test_usage_from_the_stream_is_costable(self):
        events = run(transport())
        usage = [e.usage for e in events if e.kind == "done"][0]
        self.assertEqual(usage.output_tokens, 300)
        self.assertGreater(usage.cost(MODELS["gpt-5.1"]), 0.0)


# ----------------------------------------------------------------------- the request


class TestRequestBody(unittest.TestCase):
    def test_endpoint_and_streaming_options(self):
        t = transport()
        c = client(t)
        self.assertEqual(c.url, "http://127.0.0.1:1/never-used/v1/chat/completions")
        list(c.stream(model="gpt-5.1", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(t.calls[0]["url"], c.url)
        body = body_of(t)
        self.assertTrue(body["stream"])
        # Without include_usage there is no usage record and no cost at all.
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_base_url_override_for_compatible_endpoints(self):
        t = transport()
        c = OpenAIClient(KEY, base_url="https://api.groq.com/openai/v1/", transport=t)
        list(c.stream(model="llama-3.3-70b-versatile",
                      messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(t.calls[0]["url"],
                         "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(c.base_url, "https://api.groq.com/openai/v1")

    def test_reasoning_models_get_max_completion_tokens_and_effort(self):
        t = transport()
        list(client(t).stream(model="gpt-5.1", max_tokens=4096, effort="medium",
                              temperature=0.7,
                              messages=[{"role": "user", "content": "hi"}]))
        body = body_of(t)
        self.assertEqual(body["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["reasoning_effort"], "medium")
        # temperature is a 400 on the reasoning models even when asked for.
        self.assertNotIn("temperature", body)

    def test_chat_models_get_max_tokens_and_temperature(self):
        t = transport()
        list(client(t).stream(model="gpt-4.1", max_tokens=1000, temperature=0.2,
                              messages=[{"role": "user", "content": "hi"}]))
        body = body_of(t)
        self.assertEqual(body["max_tokens"], 1000)
        self.assertNotIn("max_completion_tokens", body)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["temperature"], 0.2)

    def test_an_unknown_model_takes_the_old_field(self):
        t = transport()
        list(client(t).stream(model="qwen2.5-coder", max_tokens=800, temperature=0.5,
                              messages=[{"role": "user", "content": "hi"}]))
        body = body_of(t)
        self.assertEqual(body["model"], "qwen2.5-coder")
        self.assertEqual(body["max_tokens"], 800)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["temperature"], 0.5)

    def test_max_tokens_is_clamped_to_the_model(self):
        t = transport()
        list(client(t).stream(model="gpt-4o-mini", max_tokens=999_999,
                              messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(body_of(t)["max_tokens"], MODELS["gpt-4o-mini"].max_output)

    def test_effort_is_clamped_to_the_models_own_ladder(self):
        for asked, expect in [("high", "high"), ("low", "low"), ("max", "high"),
                              ("xhigh", "high"), ("minimal", "minimal")]:
            t = transport()
            list(client(t).stream(model="gpt-5", effort=asked,
                                  messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(body_of(t)["reasoning_effort"], expect, asked)

    def test_thinking_off_dials_the_effort_down_instead_of_omitting_it(self):
        for model, expect in [("gpt-5.1", "none"), ("gpt-5", "minimal")]:
            t = transport()
            list(client(t).stream(model=model, thinking=False,
                                  messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(body_of(t)["reasoning_effort"], expect, model)

    def test_message_translation(self):
        t = transport()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "text", "text": "one "},
                                              {"type": "text", "text": "two"}]},
            {"role": "user", "content": "last"},
        ]
        list(client(t).stream(model="gpt-5.1", messages=messages, system="be brief"))
        self.assertEqual(body_of(t)["messages"], [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one two"},
            {"role": "user", "content": "last"},
        ])

    def test_a_block_list_system_prompt_becomes_one_leading_message(self):
        t = transport()
        list(client(t).stream(model="gpt-5.1", system=[{"type": "text", "text": "a"},
                                                       {"type": "text", "text": "b"}],
                              messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(body_of(t)["messages"][0],
                         {"role": "system", "content": "ab"})

    def test_no_system_message_when_there_is_no_system_prompt(self):
        t = transport()
        list(client(t).stream(model="gpt-5.1",
                              messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual([m["role"] for m in body_of(t)["messages"]], ["user"])

    def test_flatten_handles_every_shape(self):
        self.assertEqual(_flatten(None), "")
        self.assertEqual(_flatten("x"), "x")
        self.assertEqual(_flatten({"type": "text", "text": "x"}), "x")
        self.assertEqual(_flatten([{"type": "text", "text": "a"}, "b"]), "ab")
        self.assertEqual(_flatten([{"type": "image", "url": "u"}]), "")

    def test_argument_errors_raise_immediately(self):
        c = client(transport())
        with self.assertRaises(ValueError):
            c.stream(model="gpt-5.1", messages=[])
        with self.assertRaises(ValueError):
            c.stream(model="gpt-5.1", messages=[{"content": "no role"}])
        with self.assertRaises(ValueError):
            c.stream(model="gpt-5.1", effort="turbo",
                     messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(transport().calls), 0)

    def test_no_cache_control_leaks_into_an_openai_body(self):
        # OpenAI caches automatically; Anthropic's breakpoints would be a 400.
        t = transport()
        list(client(t).stream(model="gpt-5.1", system="s" * 8000, cache=True,
                              messages=[{"role": "user", "content": "x" * 8000},
                                        {"role": "assistant", "content": "y"},
                                        {"role": "user", "content": "z"}]))
        self.assertNotIn("cache_control", json.dumps(body_of(t)))


class TestHeaders(unittest.TestCase):
    def test_headers(self):
        t = transport()
        list(client(t).stream(model="gpt-5.1",
                              messages=[{"role": "user", "content": "hi"}]))
        headers = t.calls[0]["headers"]
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["authorization"], f"Bearer {KEY}")
        self.assertIn("event-stream", headers["accept"])
        # Anthropic's headers must not come along for the ride.
        self.assertNotIn("x-api-key", headers)
        self.assertNotIn("anthropic-version", headers)

    def test_the_key_is_not_in_the_url_or_the_body(self):
        t = transport()
        list(client(t).stream(model="gpt-5.1",
                              messages=[{"role": "user", "content": "hi"}]))
        self.assertNotIn(KEY, t.calls[0]["url"])
        self.assertNotIn(KEY, json.dumps(t.calls[0]["body"]))


# ------------------------------------------------------------------------ streaming


class TestStreaming(unittest.TestCase):
    def test_happy_path(self):
        events = run(transport())
        self.assertEqual([e.kind for e in events][0], "start")
        self.assertEqual([e.kind for e in events][-1], "done")
        self.assertEqual(texts(events), "Hello, wörld…")
        self.assertEqual(texts(events, "thinking"), "Let me think. ")
        kinds = [e.kind for e in events]
        self.assertEqual(kinds.count("usage"), 1)
        self.assertLess(kinds.index("usage"), kinds.index("done"))
        done = events[-1]
        self.assertEqual(done.stop_reason, "end_turn")
        self.assertEqual(done.model, "gpt-5.1")
        self.assertEqual(done.usage.output_tokens, 300)
        self.assertEqual(done.usage.served_model, "gpt-5.1")
        self.assertEqual(events[0].model, "gpt-5.1")

    def test_a_record_split_across_chunk_boundaries_still_parses(self):
        whole = run(transport())
        for size in (1, 3, 7, 13, 64, 512):
            events = run(transport(chunk_size=size))
            self.assertEqual(texts(events), texts(whole), size)
            self.assertEqual(texts(events, "thinking"), texts(whole, "thinking"), size)
            self.assertEqual([e.kind for e in events], [e.kind for e in whole], size)

    def test_crlf_and_named_events_parse_the_same(self):
        # `sse()` from test_api writes `event:` lines and is the Anthropic shape;
        # the decoder is shared, so an OpenAI payload inside it must behave the same.
        records = [("message", json.dumps(p)) for p in turn()]
        raw = sse(records, eol="\r\n") + b"data: [DONE]\r\n\r\n"
        events = run(transport(raw=raw))
        self.assertEqual(texts(events), "Hello, wörld…")
        self.assertEqual(events[-1].kind, "done")

    def test_thinking_deltas_under_every_name(self):
        for field, value in [("reasoning_content", "a"), ("reasoning", "a"),
                             ("reasoning", {"content": "a"})]:
            events = run(transport([chunk({field: value}),
                                    chunk({"content": "hi"}, finish="stop"),
                                    chunk(usage=USAGE)]))
            self.assertEqual(texts(events, "thinking"), "a", field)
            self.assertEqual(texts(events), "hi", field)

    def test_finish_reason_mapping(self):
        for wire_reason, expect in FINISH_REASONS.items():
            events = run(transport(turn(finish=wire_reason)))
            self.assertEqual(events[-1].stop_reason, expect, wire_reason)
        # An unknown reason is passed through rather than swallowed.
        events = run(transport(turn(finish="something_new")))
        self.assertEqual(events[-1].stop_reason, "something_new")

    def test_refusal_text_is_shown_and_the_turn_is_marked(self):
        events = run(transport([chunk({"refusal": "I can't help with that."},
                                      finish="content_filter"),
                                chunk(usage=USAGE)]))
        self.assertEqual(texts(events), "I can't help with that.")
        self.assertEqual(events[-1].stop_reason, "refusal")

    def test_a_heartbeat_record_is_a_ping(self):
        events = run(transport([{"id": "chatcmpl-TEST"}] + turn()))
        self.assertEqual(events[0].kind, "ping")
        self.assertEqual(texts(events), "Hello, wörld…")

    def test_extra_choices_are_ignored(self):
        events = run(transport([chunk({"content": "one"}),
                                chunk({"content": "two"}, index=1),
                                chunk({}, finish="stop"), chunk(usage=USAGE)]))
        self.assertEqual(texts(events), "one")

    def test_a_stream_without_done_but_with_a_finish_reason_still_finishes(self):
        # Some compatible servers just hang up after the last chunk.
        events = run(transport(raw=wire(turn(), done=False)))
        self.assertEqual(events[-1].kind, "done")
        self.assertEqual(events[-1].stop_reason, "end_turn")
        self.assertEqual(texts(events), "Hello, wörld…")

    def test_a_truncated_stream_is_an_error(self):
        raw = wire(turn(finish=None, usage=None), done=False)
        with self.assertRaises(NetworkError):
            run(transport(raw=raw))

    def test_an_error_record_inside_a_started_stream_reaches_the_caller(self):
        records = [chunk({"content": "partial"}),
                   {"error": {"message": "upstream exploded", "type": "server_error",
                              "code": 500}}]
        gen = client(transport(raw=wire(records, done=False))).stream(
            model="gpt-5.1", messages=[{"role": "user", "content": "hi"}])
        seen = []
        with self.assertRaises(ServerError):
            for event in gen:
                seen.append(event)
        self.assertEqual(texts(seen), "partial")
        self.assertEqual(seen[-1].kind, "error")

    def test_usage_arriving_without_a_finish_reason_is_still_reported(self):
        events = run(transport([chunk({"content": "x"}), chunk(usage=USAGE)]))
        self.assertEqual([e.kind for e in events], ["start", "text", "usage", "done"])
        self.assertIsNone(events[-1].stop_reason)

    def test_close_closes_the_transport(self):
        t = transport()
        c = client(t)
        list(c.stream(model="gpt-5.1", messages=[{"role": "user", "content": "hi"}]))
        c.close()
        self.assertTrue(t.closed)
        c.close()  # idempotent


# -------------------------------------------------------------------------- failures


class TestRetries(unittest.TestCase):
    def test_a_500_is_retried_and_the_answer_is_not_duplicated(self):
        good = wire(turn())
        t = FakeTransport([lambda: FakeStream(status=500, reason="Server Error",
                                              body=err_body("oops", "server_error")),
                           lambda: FakeStream(chunks=[good])])
        events = list(client(t).stream(model="gpt-5.1",
                                       messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(texts(events), "Hello, wörld…")

    def test_backoff_grows_and_is_jittered(self):
        t = FakeTransport([lambda: FakeStream(status=503, body=err_body("busy"))])
        c = client(t, max_retries=3)
        c.backoff_base, c.backoff_max = 1.0, 60.0
        c._sleep = c.slept.append
        with self.assertRaises(APIError):
            list(c.stream(model="gpt-5.1", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 4)
        self.assertEqual(len(c.slept), 3)
        # Full jitter: each wait is somewhere in a window that doubles.
        for i, delay in enumerate(c.slept):
            self.assertGreaterEqual(delay, 0.0)
            self.assertLessEqual(delay, 1.0 * 2 ** i)
        self.assertNotEqual(len(set(c.slept)), 1)  # not a fixed ladder

    def test_retry_after_is_honoured(self):
        t = FakeTransport([lambda: FakeStream(status=429, headers={"retry-after": "2.5"},
                                              body=err_body("slow down",
                                                            "rate_limit_exceeded")),
                           lambda: FakeStream(chunks=[wire(turn())])])
        c = client(t)
        c.backoff_base, c.backoff_max = 30.0, 30.0  # would be much longer than 2.5
        events = list(c.stream(model="gpt-5.1",
                               messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(c.slept, [2.5])
        self.assertEqual(texts(events), "Hello, wörld…")

    def test_a_rate_limit_error_carries_the_delay(self):
        t = FakeTransport([lambda: FakeStream(status=429, headers={"retry-after": "9"},
                                              body=err_body("slow down"))])
        with self.assertRaises(RateLimitError) as caught:
            list(client(t, max_retries=0).stream(
                model="gpt-5.1", messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(caught.exception.retry_after, 9.0)

    def test_no_retry_on_400_or_401(self):
        for status, cls in [(400, BadRequestError), (401, AuthError),
                            (403, AuthError), (404, BadRequestError),
                            (413, BadRequestError)]:
            t = FakeTransport([lambda: FakeStream(status=status,
                                                  body=err_body("nope"))])
            with self.assertRaises(cls):
                list(client(t, max_retries=4).stream(
                    model="gpt-5.1", messages=[{"role": "user", "content": "hi"}]))
            self.assertEqual(len(t.calls), 1, status)

    def test_a_connection_failure_is_retried(self):
        t = FakeTransport([ConnectionResetError("reset by peer"),
                           lambda: FakeStream(chunks=[wire(turn())])])
        events = list(client(t).stream(model="gpt-5.1",
                                       messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(texts(events), "Hello, wörld…")

    def test_a_failure_after_the_first_token_is_never_retried(self):
        # Retrying here would print the opening of the answer twice.
        head = wire([chunk({"content": "Hel"})], done=False)
        t = FakeTransport([lambda: FakeStream(chunks=[head]),
                           lambda: FakeStream(chunks=[wire(turn())])])
        seen = []
        with self.assertRaises(NetworkError):
            for event in client(t).stream(model="gpt-5.1",
                                          messages=[{"role": "user", "content": "hi"}]):
                seen.append(event)
        self.assertEqual(len(t.calls), 1)
        self.assertEqual(texts(seen), "Hel")


class TestRedaction(unittest.TestCase):
    def test_an_error_body_echoing_the_key_is_scrubbed(self):
        body = err_body(f"Incorrect API key provided: {KEY}. You can find your key at "
                        "https://platform.openai.com/account/api-keys",
                        "invalid_request_error", "invalid_api_key")
        t = FakeTransport([lambda: FakeStream(status=401, reason="Unauthorized",
                                              body=body)])
        with self.assertRaises(AuthError) as caught:
            list(client(t).stream(model="gpt-5.1",
                                  messages=[{"role": "user", "content": "hi"}]))
        exc = caught.exception
        for text in (str(exc), repr(exc), exc.message, "".join(map(str, exc.args))):
            self.assertNotIn(KEY, text)
        self.assertIn("***", exc.message)

    def test_the_client_never_reprs_the_key(self):
        c = client(transport())
        for text in (repr(c), str(c), repr(c.__dict__), repr(vars(c))):
            self.assertNotIn(KEY, text)
        self.assertIn("key=***", repr(c))

    def test_the_authorization_header_reprs_as_stars(self):
        c = client(transport())
        headers = c._build_headers()
        self.assertNotIn(KEY, repr(headers))
        self.assertEqual(headers["authorization"], f"Bearer {KEY}")  # still real

    def test_a_transport_exception_carrying_the_key_is_scrubbed(self):
        t = FakeTransport([RuntimeError(f"POST failed with authorization: Bearer {KEY}")])
        with self.assertRaises(APIError) as caught:
            list(client(t, max_retries=0).stream(
                model="gpt-5.1", messages=[{"role": "user", "content": "hi"}]))
        self.assertNotIn(KEY, str(caught.exception))

    def test_the_module_never_prints_and_touches_the_key_in_one_place_only(self):
        # A cheap guard against a debugging print sneaking back in, and against a
        # second place learning to format the credential.
        path = os.path.join(os.path.dirname(_TESTS), "lume", "providers", "openai.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIsNone(re.search(r"^\s*print\(", source, re.M))
        self.assertEqual(source.count("_api_key"), 1)


# ---------------------------------------------------------------------- cancellation


class BlockingStream(FakeStream):
    """Yields its scripted chunks, then parks on a read — like a real socket."""

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


class TestCancellation(unittest.TestCase):
    def test_cancel_before_the_request_never_opens_a_connection(self):
        t = transport()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            list(client(t).stream(model="gpt-5.1", cancel=cancel,
                                  messages=[{"role": "user", "content": "hi"}]))
        self.assertEqual(len(t.calls), 0)

    def test_cancel_mid_stream_is_prompt_and_closes_the_socket(self):
        head = wire([chunk({"content": "Hel"})], done=False)
        stream = BlockingStream([head])
        t = FakeTransport([lambda: stream])
        cancel = threading.Event()
        gen = client(t).stream(model="gpt-5.1", cancel=cancel,
                               messages=[{"role": "user", "content": "hi"}])
        seen, box = [], {}

        def consume():
            try:
                for event in gen:
                    seen.append(event)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                box["exc"] = exc

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        self.assertTrue(stream.blocked.wait(5), "did not reach the blocking read")
        started = time.monotonic()
        cancel.set()
        worker.join(5)
        elapsed = time.monotonic() - started
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("exc"), CancelledError)
        self.assertLess(elapsed, 2.0, f"cancellation took {elapsed:.2f}s")
        self.assertTrue(stream.closed)
        self.assertEqual(texts(seen), "Hel")

    def test_a_cancel_after_the_finish_reason_is_still_a_cancel(self):
        # The missing-[DONE] tolerance must forgive a truncation, not a cancel:
        # reporting "done" here would show the turn the user just stopped.
        head = wire([chunk({"content": "Hel"}), chunk({}, finish="stop")], done=False)
        stream = BlockingStream([head])
        t = FakeTransport([lambda: stream])
        cancel = threading.Event()
        gen = client(t).stream(model="gpt-5.1", cancel=cancel,
                               messages=[{"role": "user", "content": "hi"}])
        box = {}

        def consume():
            try:
                box["events"] = list(gen)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                box["exc"] = exc

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        self.assertTrue(stream.blocked.wait(5), "did not reach the blocking read")
        cancel.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("exc"), CancelledError)
        self.assertNotIn("events", box)

    def test_cancel_during_backoff(self):
        t = FakeTransport([lambda: FakeStream(status=503, body=err_body("busy"))])
        c = client(t, max_retries=3)
        c.backoff_base, c.backoff_max = 30.0, 30.0
        cancel = threading.Event()
        gen = c.stream(model="gpt-5.1", cancel=cancel,
                       messages=[{"role": "user", "content": "hi"}])
        timer = threading.Timer(0.05, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            list(gen)
        self.assertLess(time.monotonic() - started, 3.0)

    def test_a_cancelled_turn_is_never_retried(self):
        self.assertFalse(CancelledError("x").retryable)


# ---------------------------------------------------------------------- the registry


class TestProvider(unittest.TestCase):
    def test_shape(self):
        p = provider()
        self.assertEqual(p.name, "openai")
        self.assertEqual(p.label, "OpenAI")
        self.assertEqual(p.env_keys, ("OPENAI_API_KEY",))
        self.assertEqual(p.base_url, DEFAULT_BASE_URL)
        self.assertIs(p.models, MODELS)
        self.assertIs(p.factory, OpenAIClient)
        self.assertEqual(p.aliases, ALIASES)
        self.assertEqual(p.doc_url, "https://platform.openai.com/api-keys")

    def test_openai_base_url_redirects_the_whole_provider(self):
        saved = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1/"
        try:
            self.assertEqual(provider().base_url, "http://localhost:11434/v1")
        finally:
            if saved is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = saved

    def test_find_key_reads_the_environment_and_returns_nothing_otherwise(self):
        p = provider()
        self.assertEqual(p.find_key({"OPENAI_API_KEY": " " + KEY + " "}), KEY)
        self.assertIsNone(p.find_key({}))
        self.assertIsNone(p.find_key({"OPENAI_API_KEY": "  "}))

    def test_the_registry_finds_openai_models(self):
        self.assertIn("openai", providers.provider_names())
        for name in ("gpt-5.1", "gpt-5", "gpt", "openai:gpt-5-mini"):
            p, spec = providers.resolve(name)
            self.assertEqual(p.name, "openai", name)
            self.assertIn(spec.id, MODELS, name)
        # Anthropic still resolves first for its own names.
        self.assertEqual(providers.resolve("opus")[0].name, "anthropic")

    def test_make_client_builds_an_openai_client_pointed_at_the_provider(self):
        c, spec = providers.make_client("gpt-5", env={"OPENAI_API_KEY": KEY},
                                        transport=transport())
        self.addCleanup(c.close)
        self.assertIsInstance(c, OpenAIClient)
        self.assertEqual(spec.id, "gpt-5")
        self.assertEqual(c.url, f"{DEFAULT_BASE_URL}/chat/completions")

    def test_without_a_key_make_client_explains_itself_without_leaking_one(self):
        with self.assertRaises(ValueError) as caught:
            providers.make_client("gpt-5", env={})
        self.assertIn("OPENAI_API_KEY", str(caught.exception))

    def test_a_default_client_builds_the_stdlib_transport_but_never_opens_it(self):
        c = OpenAIClient(KEY)
        self.addCleanup(c.close)
        self.assertIsInstance(c.transport, HTTPTransport)
        self.assertEqual(c.url, "https://api.openai.com/v1/chat/completions")

    def test_a_client_needs_a_key(self):
        with self.assertRaises(ValueError):
            OpenAIClient("")

    def test_the_stream_signature_matches_the_anthropic_client(self):
        import inspect

        from lume.api import Client as AnthropicClient
        mine = inspect.signature(OpenAIClient.stream)
        theirs = inspect.signature(AnthropicClient.stream)
        self.assertEqual(list(mine.parameters), list(theirs.parameters))
        for name, param in theirs.parameters.items():
            if name in ("self", "model"):   # the default model differs, of course
                continue
            self.assertEqual(mine.parameters[name].default, param.default, name)

    def test_events_are_the_shared_stream_event_type(self):
        for event in run(transport()):
            self.assertIsInstance(event, StreamEvent)
            self.assertIn(event.kind, ("start", "text", "thinking", "usage",
                                       "done", "error", "ping"))


if __name__ == "__main__":
    unittest.main()
