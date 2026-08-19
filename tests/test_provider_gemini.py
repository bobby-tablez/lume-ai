"""Tests for lume.providers.gemini — canned transports only.

Nothing here talks to generativelanguage.googleapis.com. There is no key in this
environment and a real call would cost money; every request goes through an
injected transport, and the default base_url is never opened.
"""

import json
import os
import sys
import threading
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from test_api import FakeStream, FakeTransport, sse  # noqa: E402

from lume.api import (  # noqa: E402
    APIError, AuthError, BadRequestError, CancelledError, ModelSpec, NetworkError,
    RateLimitError, ServerError, StreamEvent, Usage,
)
from lume.providers import gemini  # noqa: E402
from lume.providers.gemini import (  # noqa: E402
    ALIASES, DEFAULT_BASE_URL, DEFAULT_MODEL, FINISH_REASONS, MODELS, THINKING_BUDGET,
    GeminiClient, provider, resolve_model,
)

KEY = "AIzaSyTESTKEY-not-a-real-credential-000000"


# --------------------------------------------------------------------------- helpers


def gsse(records, eol="\n"):
    """Render payload dicts as Gemini's SSE: `data:` lines only, no `event:`."""
    out = []
    for payload in records:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        for line in body.split("\n"):
            out.append(f"data: {line}{eol}")
        out.append(eol)
    return "".join(out).encode("utf-8")


def chunk(payload, *, parts=(("Hello", False),), finish=None, usage=None,
          model="gemini-2.5-flash"):
    """One GenerateContentResponse. `parts` is a list of (text, is_thought)."""
    body = {"modelVersion": model, "responseId": "resp-1"}
    content_parts = []
    for text, thought in parts:
        part = {"text": text}
        if thought:
            part["thought"] = True
        content_parts.append(part)
    candidate = {"index": 0, "content": {"role": "model", "parts": content_parts}}
    if finish:
        candidate["finishReason"] = finish
    body["candidates"] = [candidate]
    if usage is not None:
        body["usageMetadata"] = usage
    body.update(payload or {})
    return body


def usage_meta(prompt=100, candidates=10, thoughts=0, cached=0):
    meta = {"promptTokenCount": prompt, "candidatesTokenCount": candidates,
            "totalTokenCount": prompt + candidates + thoughts}
    if thoughts:
        meta["thoughtsTokenCount"] = thoughts
    if cached:
        meta["cachedContentTokenCount"] = cached
    return meta


def script(pieces=("Hel", "lo"), thoughts=("thinking… ",), finish="STOP",
           model="gemini-2.5-flash"):
    """A complete, realistic response split over several SSE records."""
    records = []
    for i, t in enumerate(thoughts):
        records.append(chunk(None, parts=((t, True),), model=model,
                             usage=usage_meta(100, 0, thoughts=5 * (i + 1))))
    for i, p in enumerate(pieces):
        records.append(chunk(None, parts=((p, False),), model=model,
                             usage=usage_meta(100, 4 * (i + 1),
                                              thoughts=5 * len(thoughts))))
    records.append(chunk(None, parts=(), finish=finish, model=model,
                         usage=usage_meta(100, 4 * len(pieces),
                                          thoughts=5 * len(thoughts))))
    return records


def transport_for(records, chunk_size=None, renderer=gsse):
    raw = renderer(records)
    if chunk_size:
        chunks = [raw[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]
    else:
        chunks = [raw]
    headers = {"content-type": "text/event-stream"}
    return FakeTransport([lambda: FakeStream(headers=headers, chunks=list(chunks))])


def client(transport, **kw):
    kw.setdefault("base_url", "http://127.0.0.1:1/never-used")
    c = GeminiClient(KEY, transport=transport, **kw)
    c.backoff_base = 0.0
    c.backoff_max = 0.0
    c._sleep = lambda d: None
    return c


def run(transport, **kw):
    cancel = kw.pop("cancel", None)
    client_kw = {k: kw.pop(k) for k in ("base_url", "max_retries", "timeout")
                 if k in kw}
    kw.setdefault("model", "gemini-2.5-flash")
    kw.setdefault("messages", [{"role": "user", "content": "hi"}])
    return list(client(transport, **client_kw).stream(cancel=cancel, **kw))


def drain(events):
    text = "".join(e.text for e in events if e.kind == "text")
    thinking = "".join(e.text for e in events if e.kind == "thinking")
    return text, thinking


def err_body(code=429, status="RESOURCE_EXHAUSTED", message="quota exceeded",
             wrap=False):
    doc = {"error": {"code": code, "message": message, "status": status}}
    return json.dumps([doc] if wrap else doc).encode("utf-8")


# ---------------------------------------------------------------------------- models


class TestModels(unittest.TestCase):
    def test_table_has_a_pro_a_flash_and_a_small_model(self):
        self.assertIn("gemini-3.1-pro-preview", MODELS)
        self.assertIn("gemini-3.7-flash", MODELS)
        self.assertIn("gemini-2.5-flash-lite", MODELS)
        for spec in MODELS.values():
            self.assertIsInstance(spec, ModelSpec)
            self.assertGreater(spec.context, 100_000)
            self.assertGreater(spec.max_output, 1000)
            self.assertGreater(spec.price_in, 0.0)
            self.assertGreater(spec.price_out, spec.price_in)
            self.assertGreaterEqual(spec.price_cache_read, 0.0)
            self.assertLess(spec.price_cache_read, spec.price_in)

    def test_prices_are_the_documented_ones(self):
        want = {"gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
                "gemini-3.7-flash": (1.50, 7.50, 0.075),
                "gemini-2.5-pro": (1.25, 10.00, 0.125),
                "gemini-2.5-flash": (0.30, 2.50, 0.03),
                "gemini-2.5-flash-lite": (0.10, 0.40, 0.01)}
        for mid, (pin, pout, cache) in want.items():
            spec = MODELS[mid]
            self.assertEqual((spec.price_in, spec.price_out), (pin, pout), mid)
            self.assertAlmostEqual(spec.price_cache_read, cache, msg=mid)

    def test_flash_intro_pricing_expires_itself(self):
        spec = MODELS["gemini-3.7-flash"]
        self.assertEqual(spec.prices("2026-08-18"), (0.75, 3.75))
        self.assertEqual(spec.prices("2027-01-01"), (1.50, 7.50))

    def test_cost_maths_through_lume_usage(self):
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(u.cost(MODELS["gemini-2.5-flash"]), 0.30 + 2.50)
        self.assertAlmostEqual(u.cost(MODELS["gemini-3.1-pro-preview"]), 2.0 + 12.0)

    def test_aliases_resolve(self):
        self.assertEqual(ALIASES["gemini"], DEFAULT_MODEL)
        for alias, expect in [("gemini", DEFAULT_MODEL), ("flash", "gemini-3.7-flash"),
                              ("FLASH", "gemini-3.7-flash"), (" lite ",
                                                              "gemini-2.5-flash-lite"),
                              ("gemini-2.5-pro", "gemini-2.5-pro"),
                              ("google:flash", "gemini-3.7-flash")]:
            self.assertEqual(resolve_model(alias).id, expect, alias)
        self.assertIs(resolve_model(MODELS["gemini-2.5-flash"]),
                      MODELS["gemini-2.5-flash"])
        self.assertEqual(resolve_model(None).id, DEFAULT_MODEL)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            resolve_model("claude-opus-5")


class TestProviderEntry(unittest.TestCase):
    def test_provider_shape(self):
        p = provider()
        self.assertEqual(p.name, "google")
        self.assertEqual(p.label, "Google Gemini")
        self.assertEqual(p.env_keys, ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        self.assertEqual(p.base_url, DEFAULT_BASE_URL)
        self.assertIs(p.factory, GeminiClient)
        self.assertEqual(p.doc_url, "https://aistudio.google.com/apikey")
        self.assertEqual(set(p.models), set(MODELS))
        self.assertEqual(p.aliases["gemini"], DEFAULT_MODEL)

    def test_env_override_of_the_base_url(self):
        old = os.environ.get("GEMINI_BASE_URL")
        os.environ["GEMINI_BASE_URL"] = "https://gw.example/v1beta"
        try:
            self.assertEqual(provider().base_url, "https://gw.example/v1beta")
        finally:
            if old is None:
                os.environ.pop("GEMINI_BASE_URL", None)
            else:
                os.environ["GEMINI_BASE_URL"] = old

    def test_find_key_prefers_gemini_then_google(self):
        p = provider()
        self.assertEqual(p.find_key({"GEMINI_API_KEY": " a "}), "a")
        self.assertEqual(p.find_key({"GOOGLE_API_KEY": "b"}), "b")
        self.assertEqual(p.find_key({"GEMINI_API_KEY": "a", "GOOGLE_API_KEY": "b"}), "a")
        self.assertIsNone(p.find_key({}))

    def test_registered_in_the_package(self):
        from lume import providers
        self.assertIn("google", providers.provider_names())
        p, spec = providers.resolve("google:gemini-2.5-flash")
        self.assertEqual(p.name, "google")
        self.assertEqual(spec.id, "gemini-2.5-flash")

    def test_stream_signature_matches_the_anthropic_client(self):
        import inspect
        from lume.api import Client
        ours = inspect.signature(GeminiClient.stream).parameters
        theirs = inspect.signature(Client.stream).parameters
        self.assertEqual(list(ours), list(theirs))
        for name in theirs:
            if name == "self":
                continue
            self.assertEqual(ours[name].kind, theirs[name].kind, name)


# --------------------------------------------------------------------- request shape


class TestRequestBody(unittest.TestCase):
    def body(self, **kw):
        t = transport_for(script())
        run(t, **kw)
        return t.calls[0]["body"]

    def test_contents_role_mapping_and_flattening(self):
        body = self.body(messages=[
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": [{"type": "text", "text": "two"},
                                              {"type": "text", "text": "-three"}]},
            {"role": "user", "content": "four"},
        ])
        self.assertEqual(body["contents"], [
            {"role": "user", "parts": [{"text": "one"}]},
            {"role": "model", "parts": [{"text": "two-three"}]},
            {"role": "user", "parts": [{"text": "four"}]},
        ])

    def test_consecutive_same_role_turns_merge_into_one_content(self):
        body = self.body(messages=[
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ])
        self.assertEqual([c["role"] for c in body["contents"]],
                         ["user", "model", "user"])
        self.assertEqual(body["contents"][0]["parts"],
                         [{"text": "a"}, {"text": "b"}])
        self.assertEqual(body["contents"][1]["parts"],
                         [{"text": "c"}, {"text": "d"}])

    def test_non_text_blocks_are_dropped_not_mistranslated(self):
        body = self.body(messages=[{"role": "user", "content": [
            {"type": "image", "source": {}}, {"type": "text", "text": "look"}]}])
        self.assertEqual(body["contents"][0]["parts"], [{"text": "look"}])

    def test_unknown_roles_become_user(self):
        body = self.body(messages=[{"role": "system", "content": "x"},
                                   {"role": "user", "content": "y"}])
        self.assertEqual(body["contents"],
                         [{"role": "user", "parts": [{"text": "x"}, {"text": "y"}]}])

    def test_system_instruction(self):
        body = self.body(system="be terse")
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "be terse"}]})
        body = self.body(system=[{"type": "text", "text": "a"},
                                 {"type": "text", "text": "b"}])
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "ab"}]})
        self.assertNotIn("systemInstruction", self.body())
        self.assertNotIn("systemInstruction", self.body(system=""))

    def test_generation_config(self):
        body = self.body(max_tokens=1234, temperature=0.25)
        cfg = body["generationConfig"]
        self.assertEqual(cfg["maxOutputTokens"], 1234)
        self.assertEqual(cfg["temperature"], 0.25)
        # max_tokens is clamped to what the model can emit.
        self.assertEqual(self.body(max_tokens=10 ** 9)["generationConfig"]
                         ["maxOutputTokens"], MODELS["gemini-2.5-flash"].max_output)

    def test_temperature_only_when_the_spec_allows_it(self):
        c = GeminiClient(KEY)
        spec = MODELS["gemini-2.5-flash"]
        no_temp = ModelSpec(**{**spec.__dict__, "supports_temperature": False})
        body = c.build_body(model=no_temp, messages=[{"role": "user", "content": "x"}],
                            temperature=0.5)
        self.assertNotIn("temperature", body["generationConfig"])

    def test_effort_maps_to_a_thinking_budget(self):
        for effort, want in THINKING_BUDGET.items():
            cfg = self.body(effort=effort)["generationConfig"]["thinkingConfig"]
            self.assertTrue(cfg["includeThoughts"])
            self.assertEqual(cfg["thinkingBudget"], want, effort)

    def test_budget_never_exceeds_max_tokens(self):
        cfg = self.body(effort="xhigh", max_tokens=2048)["generationConfig"]["thinkingConfig"]
        self.assertEqual(cfg["thinkingBudget"], 2048)
        # Below the API's own floor a budget is a 400; ask for none instead.
        cfg = self.body(effort="high", max_tokens=100)["generationConfig"]["thinkingConfig"]
        self.assertEqual(cfg["thinkingBudget"], 0)

    def test_gemini_3_models_take_a_thinking_level(self):
        cfg = self.body(model="gemini-3.7-flash",
                        effort="medium")["generationConfig"]["thinkingConfig"]
        self.assertEqual(cfg, {"includeThoughts": True, "thinkingLevel": "medium"})
        # The level ladder tops out at "high": xhigh/max clamp onto it.
        for effort in ("xhigh", "max"):
            cfg = self.body(model="gemini-3.7-flash",
                            effort=effort)["generationConfig"]["thinkingConfig"]
            self.assertEqual(cfg["thinkingLevel"], "high", effort)

    def test_thinking_false_turns_it_off_per_model(self):
        cfg = self.body(thinking=False)["generationConfig"]["thinkingConfig"]
        self.assertEqual(cfg, {"includeThoughts": False, "thinkingBudget": 0})
        cfg = self.body(model="gemini-3.7-flash",
                        thinking=False)["generationConfig"]["thinkingConfig"]
        self.assertEqual(cfg, {"includeThoughts": False, "thinkingLevel": "minimal"})
        # A Pro model cannot stop thinking; the most we can do is stop streaming it.
        cfg = self.body(model="gemini-2.5-pro",
                        thinking=False)["generationConfig"]["thinkingConfig"]
        self.assertEqual(cfg, {"includeThoughts": False})
        cfg = self.body(model="gemini-3.1-pro-preview",
                        thinking=False)["generationConfig"]["thinkingConfig"]
        self.assertEqual(cfg, {"includeThoughts": False})

    def test_cache_flag_sends_nothing(self):
        # Gemini's implicit caching has no request field; the argument exists so
        # the two clients stay call-compatible.
        self.assertEqual(self.body(cache=True), self.body(cache=False))

    def test_bad_arguments_raise_before_any_request(self):
        t = transport_for(script())
        c = client(t)
        with self.assertRaises(ValueError):
            c.stream(model="gemini-2.5-flash", messages=[], effort="high")
        with self.assertRaises(ValueError):
            c.stream(model="gemini-2.5-flash",
                     messages=[{"role": "user", "content": "x"}], effort="turbo")
        with self.assertRaises(ValueError):
            c.stream(model="gemini-2.5-flash", messages=[{"content": "x"}])
        self.assertEqual(t.calls, [])

    def test_no_api_key_is_a_value_error(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                GeminiClient(bad)


class TestUrlAndHeaders(unittest.TestCase):
    def test_endpoint_and_header_construction(self):
        t = transport_for(script())
        run(t, model="gemini-2.5-flash")
        call = t.calls[0]
        self.assertEqual(
            call["url"],
            "http://127.0.0.1:1/never-used/models/"
            "gemini-2.5-flash:streamGenerateContent?alt=sse")
        self.assertEqual(call["headers"]["content-type"], "application/json")
        self.assertEqual(call["headers"]["x-goog-api-key"], KEY)
        self.assertEqual(call["headers"]["accept"], "text/event-stream")

    def test_the_key_is_never_in_the_url(self):
        t = transport_for(script())
        run(t)
        url = t.calls[0]["url"]
        self.assertNotIn(KEY, url)
        self.assertNotIn("key=", url)
        self.assertNotIn("AIza", url)

    def test_base_url_override(self):
        t = transport_for(script())
        run(t, base_url="https://proxy.example/v1beta/", model="gemini-2.5-pro")
        self.assertEqual(
            t.calls[0]["url"],
            "https://proxy.example/v1beta/models/"
            "gemini-2.5-pro:streamGenerateContent?alt=sse")
        self.assertEqual(GeminiClient(KEY).base_url, DEFAULT_BASE_URL)
        self.assertEqual(GeminiClient(KEY, base_url=None).base_url, DEFAULT_BASE_URL)

    def test_the_default_base_url_is_googles(self):
        self.assertEqual(DEFAULT_BASE_URL,
                         "https://generativelanguage.googleapis.com/v1beta")


class TestKeyRedaction(unittest.TestCase):
    def test_repr_hides_the_key(self):
        c = GeminiClient(KEY, base_url="https://x.example")
        self.assertNotIn(KEY, repr(c))
        self.assertNotIn(KEY, str(c))
        self.assertIn("key=***", repr(c))

    def test_headers_repr_hides_the_key(self):
        # A traceback formatter that captures locals prints header dicts.
        headers = GeminiClient(KEY)._build_headers()
        self.assertEqual(headers["x-goog-api-key"], KEY)  # still correct on the wire
        self.assertNotIn(KEY, repr(headers))

    def test_redact_key_blanks_google_shapes(self):
        self.assertNotIn(KEY, gemini.redact_key(f"bad key {KEY} sorry"))
        self.assertIn("AIza***", gemini.redact_key(f"bad key {KEY}"))
        self.assertIn("***", gemini.redact_key("GET /v1beta/models?key=AIzaSecretValue"))
        self.assertEqual(gemini.redact_key(""), "")

    def test_the_key_never_reaches_an_exception_message(self):
        body = json.dumps({"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                     "message": f"API key {KEY} is invalid"}}).encode()
        t = FakeTransport([lambda: FakeStream(status=400, reason="Bad Request",
                                              body=body)])
        with self.assertRaises(BadRequestError) as ctx:
            run(t)
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertNotIn(KEY, repr(ctx.exception))
        self.assertNotIn(KEY, ctx.exception.message)

    def test_a_transport_that_leaks_the_key_is_scrubbed(self):
        t = FakeTransport([RuntimeError(f"connect failed with header {KEY}")])
        with self.assertRaises(APIError) as ctx:
            run(t, max_retries=0)
        self.assertNotIn(KEY, str(ctx.exception))


# -------------------------------------------------------------------------- streaming


class TestStreaming(unittest.TestCase):
    def test_happy_path(self):
        t = transport_for(script(pieces=("Hel", "lo, ", "wörld…"),
                                 thoughts=("Let me think. ",)))
        events = run(t)
        text, thinking = drain(events)
        self.assertEqual(text, "Hello, wörld…")
        self.assertEqual(thinking, "Let me think. ")
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(events[0].model, "gemini-2.5-flash")
        self.assertEqual(events[-1].stop_reason, "end_turn")
        self.assertIn("usage", kinds)

    def test_event_objects_are_lume_stream_events(self):
        events = run(transport_for(script()))
        for e in events:
            self.assertIsInstance(e, StreamEvent)
            self.assertIn(e.kind, ("start", "text", "thinking", "usage", "done",
                                   "error", "ping"))

    def test_thought_parts_and_text_parts_in_one_record(self):
        records = [chunk(None, parts=(("reasoning", True), ("answer", False)),
                         finish="STOP", usage=usage_meta(10, 2, thoughts=3))]
        events = run(transport_for(records))
        self.assertEqual([(e.kind, e.text) for e in events if e.kind in
                          ("text", "thinking")],
                         [("thinking", "reasoning"), ("text", "answer")])

    def test_a_part_without_the_thought_flag_is_answer_text(self):
        records = [chunk(None, parts=(("plain", False),), finish="STOP",
                         usage=usage_meta(1, 1))]
        events = run(transport_for(records))
        self.assertEqual(drain(events), ("plain", ""))

    def test_a_record_split_across_chunk_boundaries(self):
        want = "Hello, wörld… " * 12
        records = script(pieces=tuple(want[i:i + 5] for i in range(0, len(want), 5)),
                         thoughts=("thinking hard about it", ))
        for size in (1, 2, 3, 7, 13, 64, 1024):
            t = transport_for(records, chunk_size=size)
            text, thinking = drain(run(t))
            self.assertEqual(text, want, f"chunk size {size}")
            self.assertEqual(thinking, "thinking hard about it")

    def test_crlf_line_endings_parse(self):
        t = transport_for(script(), renderer=lambda r: gsse(r, eol="\r\n"))
        self.assertEqual(drain(run(t))[0], "Hello")

    def test_the_test_api_sse_helper_also_parses(self):
        # Real Gemini sends bare `data:` records; an `event:` name in front is
        # still valid SSE and must not change the outcome.
        t = transport_for(script(), renderer=lambda r: sse([("message", p) for p in r]))
        self.assertEqual(drain(run(t))[0], "Hello")

    def test_generator_is_lazy_until_iterated(self):
        t = transport_for(script())
        gen = client(t).stream(model="gemini-2.5-flash",
                               messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(t.calls, [])
        list(gen)
        self.assertEqual(len(t.calls), 1)

    def test_close_closes_the_transport(self):
        t = transport_for(script())
        c = client(t)
        list(c.stream(model="flash", messages=[{"role": "user", "content": "hi"}]))
        c.close()
        c.close()  # idempotent
        self.assertTrue(t.closed)

    def test_a_stream_that_stops_before_a_finish_reason_is_an_error(self):
        records = [chunk(None, parts=(("half an ans", False),),
                         usage=usage_meta(10, 3))]
        t = transport_for(records)
        with self.assertRaises(NetworkError) as ctx:
            run(t, max_retries=0)
        self.assertIn("finishReason", str(ctx.exception))

    def test_a_200_that_is_not_an_event_stream(self):
        t = FakeTransport([lambda: FakeStream(
            headers={"content-type": "text/html"}, body=b"<html>captive portal</html>")])
        with self.assertRaises(NetworkError) as ctx:
            run(t, max_retries=0)
        self.assertIn("event stream", str(ctx.exception))
        self.assertFalse(ctx.exception.retryable)


class TestUsage(unittest.TestCase):
    def test_cumulative_usage_is_not_double_counted(self):
        # Every record restates the running totals; adding them would triple the bill.
        records = [
            chunk(None, parts=(("a", False),), usage=usage_meta(1000, 5)),
            chunk(None, parts=(("b", False),), usage=usage_meta(1000, 9)),
            chunk(None, parts=(), finish="STOP", usage=usage_meta(1000, 12)),
        ]
        events = run(transport_for(records))
        final = events[-1].usage
        self.assertEqual(final.input_tokens, 1000)
        self.assertEqual(final.output_tokens, 12)
        for e in events:
            if e.usage is not None:
                self.assertLessEqual(e.usage.input_tokens, 1000)
                self.assertLessEqual(e.usage.output_tokens, 12)

    def test_thought_tokens_count_as_output(self):
        records = [chunk(None, parts=(("x", False),), finish="STOP",
                         usage=usage_meta(50, 20, thoughts=30))]
        final = run(transport_for(records))[-1].usage
        self.assertEqual(final.output_tokens, 50)
        self.assertEqual(final.input_tokens, 50)

    def test_cached_tokens_are_billed_in_their_own_bucket(self):
        # promptTokenCount includes the cached part; lume prices them separately.
        records = [chunk(None, parts=(("x", False),), finish="STOP",
                         usage=usage_meta(1000, 10, cached=800))]
        final = run(transport_for(records))[-1].usage
        self.assertEqual(final.cache_read_input_tokens, 800)
        self.assertEqual(final.input_tokens, 200)
        self.assertEqual(final.cache_creation_input_tokens, 0)
        spec = MODELS["gemini-2.5-flash"]
        want = (200 * spec.price_in + 10 * spec.price_out
                + 800 * spec.price_in * 0.1) / 1e6
        self.assertAlmostEqual(final.cost(spec), want)

    def test_a_partial_usage_record_keeps_the_fields_it_omits(self):
        records = [
            chunk(None, parts=(("a", False),), usage=usage_meta(400, 7, thoughts=3)),
            chunk(None, parts=(), finish="STOP",
                  usage={"promptTokenCount": 400, "totalTokenCount": 999}),
        ]
        final = run(transport_for(records))[-1].usage
        self.assertEqual(final.input_tokens, 400)
        self.assertEqual(final.output_tokens, 10)

    def test_usage_events_carry_a_snapshot_not_a_live_object(self):
        events = run(transport_for(script(pieces=("a", "b"))))
        snapshots = [e.usage for e in events if e.kind == "usage"]
        self.assertGreater(len(snapshots), 1)
        self.assertNotEqual(snapshots[0].output_tokens, snapshots[-1].output_tokens)


class TestFinishReasons(unittest.TestCase):
    def test_mapping_table(self):
        cases = [("STOP", "end_turn"), ("MAX_TOKENS", "max_tokens"),
                 ("SAFETY", "refusal"), ("RECITATION", "refusal"),
                 ("PROHIBITED_CONTENT", "refusal"), ("OTHER", "other")]
        for wire, want in cases:
            records = [chunk(None, parts=(("x", False),), finish=wire,
                             usage=usage_meta(1, 1))]
            events = run(transport_for(records))
            self.assertEqual(events[-1].kind, "done", wire)
            self.assertEqual(events[-1].stop_reason, want, wire)
            self.assertEqual(events[-1].stop_details["reason"], wire)
        self.assertEqual(FINISH_REASONS["SAFETY"], "refusal")

    def test_a_safety_stop_is_a_done_event_not_an_exception(self):
        records = [chunk({"candidates": [{"index": 0, "finishReason": "SAFETY",
                                          "safetyRatings": [{"category": "HARM",
                                                             "probability": "HIGH"}]}]},
                         parts=(), usage=usage_meta(20, 0))]
        events = run(transport_for(records))  # must not raise
        done = events[-1]
        self.assertEqual(done.kind, "done")
        self.assertEqual(done.stop_reason, "refusal")
        self.assertTrue(done.stop_details["refusal"])
        self.assertEqual(done.stop_details["reason"], "SAFETY")
        self.assertEqual(done.stop_details["safety_ratings"][0]["category"], "HARM")

    def test_an_unspecified_finish_reason_does_not_end_the_turn(self):
        records = [chunk(None, parts=(("x", False),),
                         finish="FINISH_REASON_UNSPECIFIED", usage=usage_meta(1, 1))]
        with self.assertRaises(NetworkError):
            run(transport_for(records), max_retries=0)

    def test_a_blocked_prompt_is_a_refusal(self):
        records = [{"promptFeedback": {"blockReason": "SAFETY",
                                       "safetyRatings": [{"category": "HARM"}]},
                    "usageMetadata": usage_meta(12, 0)}]
        events = run(transport_for(records))
        self.assertEqual(events[-1].kind, "done")
        self.assertEqual(events[-1].stop_reason, "refusal")
        self.assertEqual(events[-1].stop_details["source"], "prompt")


# ----------------------------------------------------------------------------- errors


class TestErrors(unittest.TestCase):
    def test_status_to_exception_family(self):
        cases = [(400, "INVALID_ARGUMENT", BadRequestError),
                 (401, "UNAUTHENTICATED", AuthError),
                 (403, "PERMISSION_DENIED", AuthError),
                 (404, "NOT_FOUND", BadRequestError),
                 (429, "RESOURCE_EXHAUSTED", RateLimitError),
                 (500, "INTERNAL", ServerError),
                 (503, "UNAVAILABLE", ServerError)]
        for code, status, cls in cases:
            t = FakeTransport([lambda code=code, status=status: FakeStream(
                status=code, body=err_body(code, status, "nope"))])
            with self.assertRaises(cls, msg=status) as ctx:
                run(t, max_retries=0)
            self.assertEqual(ctx.exception.status, code)
            self.assertEqual(ctx.exception.type, status)
            self.assertIn("nope", str(ctx.exception))

    def test_resource_exhausted_is_a_rate_limit_whatever_the_envelope(self):
        # The RPC name outranks a missing/odd HTTP code.
        t = FakeTransport([lambda: FakeStream(status=200, headers={
            "content-type": "application/json"}, body=err_body(None,
                                                               "RESOURCE_EXHAUSTED"))])
        with self.assertRaises(RateLimitError) as ctx:
            run(t, max_retries=0)
        self.assertEqual(ctx.exception.status, 429)

    def test_a_streaming_failure_arrives_as_a_json_array(self):
        t = FakeTransport([lambda: FakeStream(status=503, body=err_body(
            503, "UNAVAILABLE", "overloaded", wrap=True))])
        with self.assertRaises(ServerError) as ctx:
            run(t, max_retries=0)
        self.assertIn("overloaded", str(ctx.exception))

    def test_an_error_inside_a_200_event_stream(self):
        records = [chunk(None, parts=(("partial", False),), usage=usage_meta(5, 1)),
                   [{"error": {"code": 500, "status": "INTERNAL", "message": "boom"}}]]
        t = transport_for(records)
        gen = client(t, max_retries=0).stream(
            model="gemini-2.5-flash", messages=[{"role": "user", "content": "hi"}])
        seen = []
        with self.assertRaises(ServerError):
            for e in gen:
                seen.append(e)
        # Text already delivered stays delivered, and the error is announced.
        self.assertEqual("".join(e.text for e in seen if e.kind == "text"), "partial")
        self.assertEqual(seen[-1].kind, "error")

    def test_an_unparseable_error_body_still_classifies(self):
        t = FakeTransport([lambda: FakeStream(status=502, reason="Bad Gateway",
                                              body=b"<html>nginx</html>")])
        with self.assertRaises(ServerError) as ctx:
            run(t, max_retries=0)
        self.assertEqual(ctx.exception.status, 502)

    def test_retry_after_is_read(self):
        t = FakeTransport([lambda: FakeStream(status=429, headers={"retry-after": "7"},
                                              body=err_body())])
        with self.assertRaises(RateLimitError) as ctx:
            run(t, max_retries=0)
        self.assertEqual(ctx.exception.retry_after, 7.0)


# ---------------------------------------------------------------------------- retries


class TestRetries(unittest.TestCase):
    def test_429_then_success(self):
        t = FakeTransport([
            lambda: FakeStream(status=429, body=err_body()),
            lambda: FakeStream(headers={"content-type": "text/event-stream"},
                               chunks=[gsse(script())]),
        ])
        text, _ = drain(run(t))
        self.assertEqual(text, "Hello")
        self.assertEqual(len(t.calls), 2)

    def test_retries_every_retryable_status(self):
        for code in (429, 500, 502, 503, 504):
            t = FakeTransport([
                lambda code=code: FakeStream(status=code, body=err_body(code, "X")),
                lambda: FakeStream(headers={"content-type": "text/event-stream"},
                                   chunks=[gsse(script())]),
            ])
            drain(run(t))
            self.assertEqual(len(t.calls), 2, code)

    def test_connection_errors_are_retried(self):
        t = FakeTransport([
            ConnectionResetError("reset by peer"),
            lambda: FakeStream(headers={"content-type": "text/event-stream"},
                               chunks=[gsse(script())]),
        ])
        self.assertEqual(drain(run(t))[0], "Hello")
        self.assertEqual(len(t.calls), 2)

    def test_no_retry_on_400_or_401(self):
        for code, status, cls in ((400, "INVALID_ARGUMENT", BadRequestError),
                                  (401, "UNAUTHENTICATED", AuthError),
                                  (403, "PERMISSION_DENIED", AuthError),
                                  (404, "NOT_FOUND", BadRequestError)):
            t = FakeTransport([lambda code=code, status=status: FakeStream(
                status=code, body=err_body(code, status, "wrong"))])
            with self.assertRaises(cls):
                run(t, max_retries=4)
            self.assertEqual(len(t.calls), 1, code)

    def test_retries_are_bounded(self):
        t = FakeTransport([lambda: FakeStream(status=503, body=err_body(503, "X"))])
        with self.assertRaises(ServerError):
            run(t, max_retries=2)
        self.assertEqual(len(t.calls), 3)

    def test_backoff_has_jitter_and_grows(self):
        c = GeminiClient(KEY)
        c.backoff_base, c.backoff_max = 0.5, 16.0
        for attempt in range(4):
            window = min(16.0, 0.5 * (2 ** attempt))
            samples = {round(c._delay(attempt, None), 6) for _ in range(40)}
            self.assertGreater(len(samples), 1, "no jitter")
            self.assertLessEqual(max(samples), window)
            self.assertGreaterEqual(min(samples), 0.0)
        self.assertGreater(max(c._delay(3, None) for _ in range(200)),
                           max(c._delay(0, None) for _ in range(200)))
        self.assertEqual(c._delay(0, 12.5), 12.5)   # retry-after wins outright

    def test_a_retry_can_never_duplicate_text(self):
        # The first attempt delivers text and then dies; retrying would print
        # "Hello" twice, so the failure is surfaced instead.
        head = gsse([chunk(None, parts=(("Hello", False),), usage=usage_meta(9, 2))])
        t = FakeTransport([
            lambda: FakeStream(headers={"content-type": "text/event-stream"},
                               chunks=[head]),
            lambda: FakeStream(headers={"content-type": "text/event-stream"},
                               chunks=[gsse(script())]),
        ])
        seen = []
        with self.assertRaises(NetworkError):
            for e in client(t, max_retries=3).stream(
                    model="gemini-2.5-flash",
                    messages=[{"role": "user", "content": "hi"}]):
                seen.append(e)
        self.assertEqual(len(t.calls), 1)
        self.assertEqual("".join(e.text for e in seen if e.kind == "text"), "Hello")

    def test_a_retry_budget_that_cannot_be_met_gives_up_with_a_reason(self):
        t = FakeTransport([lambda: FakeStream(
            status=429, headers={"retry-after": "3600"}, body=err_body())])
        with self.assertRaises(RateLimitError) as ctx:
            run(t, max_retries=4)
        self.assertIn("retry budget", str(ctx.exception))
        self.assertEqual(len(t.calls), 1)


# ------------------------------------------------------------------------ cancelling


class BlockingStream(FakeStream):
    """Yields its scripted chunks, then parks on a read — like a real socket."""

    def __init__(self, chunks):
        super().__init__(headers={"content-type": "text/event-stream"}, chunks=chunks)
        self.gate = threading.Event()
        self.blocked = threading.Event()

    def __next__(self):
        if self._chunks and not self.closed:
            return self._chunks.pop(0)
        self.blocked.set()
        if self.gate.wait(10):
            # How a live connection really ends when another thread tears it
            # down mid-read: not a tidy StopIteration but http.client's race.
            raise AttributeError("'NoneType' object has no attribute 'close'")
        raise AssertionError("blocking read was never interrupted")

    def close(self):
        self.closed = True
        self.gate.set()


class SlowOpenTransport:
    """`open()` blocks, exactly as a transport waiting for headers would."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def open(self, url, headers, body, timeout):  # no `cancel` parameter
        self.entered.set()
        self.release.wait(10)
        return FakeStream(headers={"content-type": "text/event-stream"},
                          chunks=[gsse(script())])

    def close(self):
        self.release.set()


class TestCancellation(unittest.TestCase):
    def test_cancel_before_the_request(self):
        t = transport_for(script())
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            run(t, cancel=cancel)
        self.assertEqual(t.calls, [])

    def test_cancel_mid_stream_is_prompt_and_closes_the_socket(self):
        head = gsse([chunk(None, parts=(("Hel", False),), usage=usage_meta(9, 1))])
        stream = BlockingStream([head])
        t = FakeTransport([lambda: stream])
        cancel = threading.Event()
        gen = client(t).stream(model="gemini-2.5-flash",
                               messages=[{"role": "user", "content": "hi"}],
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

    def test_cancel_while_the_connect_is_outstanding(self):
        t = SlowOpenTransport()
        cancel = threading.Event()
        gen = client(t).stream(model="gemini-2.5-flash",
                               messages=[{"role": "user", "content": "hi"}],
                               cancel=cancel)
        box = {}

        def consume():
            try:
                list(gen)
            except BaseException as exc:  # noqa: BLE001
                box["exc"] = exc

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        self.assertTrue(t.entered.wait(5), "open() was never called")
        started = time.monotonic()
        cancel.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("exc"), CancelledError)
        self.assertLess(time.monotonic() - started, 2.0)
        t.close()

    def test_cancel_during_backoff(self):
        t = FakeTransport([lambda: FakeStream(status=503, body=err_body(503, "X"))])
        c = client(t, max_retries=3)
        c.backoff_base, c.backoff_max = 30.0, 30.0
        cancel = threading.Event()
        gen = c.stream(model="gemini-2.5-flash",
                       messages=[{"role": "user", "content": "hi"}], cancel=cancel)
        timer = threading.Timer(0.05, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(CancelledError):
            list(gen)
        self.assertLess(time.monotonic() - started, 3.0)

    def test_a_cancel_is_never_retried(self):
        self.assertFalse(CancelledError("x").retryable)
        t = transport_for(script())
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            run(t, cancel=cancel, max_retries=4)
        self.assertEqual(len(t.calls), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
