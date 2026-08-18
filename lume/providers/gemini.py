"""Google Gemini, translated into lume's vocabulary.

A provider is a translation layer and nothing more: this module speaks the
`generateContent` wire format to `generativelanguage.googleapis.com` and hands
back the very same :class:`lume.api.StreamEvent` objects the Anthropic client
yields, so the renderer, the store and the cost maths cannot tell who answered.

Everything reusable is reused from :mod:`lume.api` — the error family, the SSE
decoder, the transport, the retry classifier, the cancel watchdog. What is
genuinely different lives here: the request shape (`contents` / `parts` /
`generationConfig`), the `x-goog-api-key` header, cumulative `usageMetadata`,
and `finishReason`.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.parse
from dataclasses import replace

from ..api import (
    EFFORTS, APIError, CancelledError, HTTPTransport, ModelSpec, NetworkError,
    RateLimitError, SSEDecoder, StreamEvent, Usage, redact,
    _accepts_cancel, _class_for_status, _close_quietly, _drain_body, _header,
    _parse_retry_after, _raise_detached, _reap, _transport_error, _watch_cancel,
    _INTERNAL_BUGS, _Secret,
)

__all__ = [
    "MODELS", "ALIASES", "DEFAULT_MODEL", "DEFAULT_BASE_URL", "FINISH_REASONS",
    "THINKING_BUDGET", "GeminiClient", "provider", "resolve_model", "model_names",
    "redact_key",
]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

_READ_LIMIT = 64 * 1024


# --------------------------------------------------------------------------- models


def _spec(mid, label, price_in, price_out, price_cache_read, *, context=1_048_576,
          max_output=65_536, thinking="budget", thinking_off="disabled",
          efforts=EFFORTS, intro=None) -> ModelSpec:
    # `price_cache_write` is 0: Gemini bills context caching by *storage per hour*,
    # not by a per-token write, and implicit caching (the only kind this client
    # uses) is free to populate. Nothing here ever reports cache-write tokens, so
    # a zero rate multiplied by zero tokens is the honest answer.
    return ModelSpec(
        id=mid, label=label, context=context, max_output=max_output,
        price_in=price_in, price_out=price_out,
        price_cache_write=0.0, price_cache_read=price_cache_read,
        supports_temperature=True, supports_effort=True, thinking=thinking,
        efforts=tuple(efforts), supports_thinking_display=True,
        thinking_off=thinking_off, cache_min=1024,
        intro_price_in=(intro or (None, None, None))[0],
        intro_price_out=(intro or (None, None, None))[1],
        intro_until=(intro or (None, None, None))[2],
    )


# One editable table. Ids, limits and USD-per-1M prices read off
# https://ai.google.dev/gemini-api/docs/pricing and .../docs/models on
# **2026-08-18** (paid tier, standard — not Batch, not the free tier). Re-check
# them when that date feels old; Google reprices more often than Anthropic.
#
# Two Gemini quirks the table has to carry:
#   * Pro pricing is *tiered* by prompt length ($2/$12 up to 200k tokens, $4/$18
#     above). ModelSpec has one rate, so the sub-200k rate is stored — the tier
#     almost every chat turn is in — and a very long turn is under-reported.
#   * `thinking` selects the knob, not the vendor's word: "budget" means the 2.5
#     family's `thinkingBudget` (a token count); "adaptive" means the Gemini 3
#     family's `thinkingLevel` (minimal|low|medium|high).
MODELS: dict = {
    m.id: m for m in (
        # Flagship. Still `-preview`; the id changes when it graduates.
        # Thinking cannot be switched off on a Pro model.
        _spec("gemini-3.1-pro-preview", "Gemini 3.1 Pro", 2.00, 12.00, 0.20,
              thinking="adaptive", thinking_off="unsupported",
              efforts=("low", "medium", "high")),
        # Intro pricing through 2026-12-31, then $1.50/$7.50 — carried as `intro`
        # so it expires itself with no code change. The cache-read rate below is
        # likewise the intro one ($0.15 from 2027-01-01); ModelSpec's intro
        # fields cover input/output only, so that one needs a manual bump.
        _spec("gemini-3.7-flash", "Gemini 3.7 Flash", 1.50, 7.50, 0.075,
              thinking="adaptive", thinking_off="disabled",
              efforts=("low", "medium", "high"),
              intro=(0.75, 3.75, "2026-12-31")),
        _spec("gemini-2.5-pro", "Gemini 2.5 Pro", 1.25, 10.00, 0.125,
              thinking="budget", thinking_off="unsupported"),
        _spec("gemini-2.5-flash", "Gemini 2.5 Flash", 0.30, 2.50, 0.03),
        _spec("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", 0.10, 0.40, 0.01),
    )
}

ALIASES: dict = {
    "default": DEFAULT_MODEL,
    "gemini": DEFAULT_MODEL,
    "gemini-pro": DEFAULT_MODEL,
    "pro": DEFAULT_MODEL,
    "gemini-3-pro": DEFAULT_MODEL,
    "gemini-3.1-pro": DEFAULT_MODEL,
    "flash": "gemini-3.7-flash",
    "gemini-flash": "gemini-3.7-flash",
    "gemini-3-flash": "gemini-3.7-flash",
    "gemini-3.7": "gemini-3.7-flash",
    "flash-lite": "gemini-2.5-flash-lite",
    "lite": "gemini-2.5-flash-lite",
    "mini": "gemini-2.5-flash-lite",
}


def model_names() -> list:
    """Canonical Gemini model ids, in menu order."""
    return list(MODELS)


def resolve_model(name) -> ModelSpec:
    """Look up a ModelSpec by id or alias ('gemini', 'flash', a ModelSpec).

    Raises ValueError for anything unknown — guessing a model means guessing a
    price, and a wrong price is worse than an error.
    """
    if isinstance(name, ModelSpec):
        return name
    if not name:
        return MODELS[DEFAULT_MODEL]
    key = str(name).strip().lower()
    if key.startswith("google:") or key.startswith("gemini:"):
        key = key.partition(":")[2].strip()
    if key in MODELS:
        return MODELS[key]
    if key in ALIASES:
        return MODELS[ALIASES[key]]
    # Google writes both `gemini-2.5-flash` and `gemini-2-5-flash` in places.
    dotted = key.replace("-5-", "-5.").replace("-1-", "-1.").replace("-7-", "-7.")
    if dotted in MODELS:
        return MODELS[dotted]
    raise ValueError(f"unknown Gemini model {name!r}; known: {', '.join(MODELS)}")


# ------------------------------------------------------------------------- thinking

#: lume's effort ladder mapped onto `thinkingConfig.thinkingBudget`, the token
#: allowance the Gemini 2.5 family spends on reasoning. `max` sends -1, Google's
#: "dynamic thinking" sentinel: the model picks its own depth, which is the
#: closest thing the API has to "as much as it takes". Every positive value here
#: sits inside the narrowest per-model window (512..24576 on Flash-Lite), so no
#: level is a 400 on any model in the table.
THINKING_BUDGET: dict = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": -1,
}

# Below this the API rejects a budget outright (Flash-Lite's floor), so a caller
# who squeezed `max_tokens` under it gets no thinking rather than a 400.
_MIN_BUDGET = 512


def _thinking_config(spec: ModelSpec, thinking: bool, effort, max_tokens: int):
    """Build `generationConfig.thinkingConfig`, or None to send no such field.

    `thinking=False` is per-model, exactly as it is on Anthropic:

    * ``thinking_off="disabled"`` — the model can genuinely stop. The 2.5 family
      is told ``thinkingBudget: 0``; the Gemini 3 family has no zero budget, so
      it gets ``thinkingLevel: "minimal"``.
    * ``thinking_off="unsupported"`` — a Pro model always reasons and rejects an
      attempt to disable it. The most that can be done is to stop asking for the
      summaries, so only ``includeThoughts: false`` goes out.
    * ``thinking_off="omit"`` — absence really means off; nothing is sent.
    """
    if spec.thinking == "none":
        return None

    if not thinking:
        if spec.thinking_off == "omit":
            return None
        if spec.thinking_off == "unsupported":
            return {"includeThoughts": False}
        if spec.thinking == "budget":
            return {"includeThoughts": False, "thinkingBudget": 0}
        return {"includeThoughts": False, "thinkingLevel": "minimal"}

    level = spec.clamp_effort(effort) if spec.supports_effort else None
    if spec.thinking != "budget":
        return {"includeThoughts": True, "thinkingLevel": level or "high"}

    budget = THINKING_BUDGET.get(level or "high", THINKING_BUDGET["high"])
    if budget > 0:
        # Thinking tokens are drawn from the same output allowance, so a budget
        # larger than `max_tokens` would leave nothing for the answer.
        budget = min(budget, int(max_tokens))
        if budget < _MIN_BUDGET:
            budget = 0
    if budget == 0 and spec.thinking_off == "unsupported":
        # This model cannot be told zero; let it choose its own depth instead.
        return {"includeThoughts": True}
    return {"includeThoughts": True, "thinkingBudget": budget}


# --------------------------------------------------------------------------- errors

# Google keys are `AIza…`; lume's own `redact` only knows Anthropic shapes.
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{8,}")
_KEY_QUERY_RE = re.compile(r"(?i)([?&](?:key|api_key)=)[^&\s]+")


def redact_key(text) -> str:
    """Blank out anything shaped like a credential, Google's shapes included."""
    if not text:
        return ""
    return _KEY_QUERY_RE.sub(r"\1***", _GOOGLE_KEY_RE.sub("AIza***", redact(text)))


#: Google's RPC status names that are not simply implied by the HTTP code.
_GOOGLE_STATUS = {
    "RESOURCE_EXHAUSTED": 429,
    "UNAUTHENTICATED": 401,
    "PERMISSION_DENIED": 403,
    "INVALID_ARGUMENT": 400,
    "FAILED_PRECONDITION": 400,
    "NOT_FOUND": 404,
    "UNAVAILABLE": 503,
    "DEADLINE_EXCEEDED": 504,
    "INTERNAL": 500,
    "UNKNOWN": 500,
}


def _error_doc(body):
    """Pull the `{"error": {...}}` object out of a response body.

    Streaming failures come back as a JSON *array* of one such object rather
    than the bare object a unary call returns, so both shapes are unwrapped.
    """
    if isinstance(body, (bytes, bytearray)):
        try:
            body = json.loads(bytes(body).decode("utf-8", "replace"))
        except ValueError:
            return None
    if isinstance(body, list):
        for item in body:
            found = _error_doc(item)
            if found is not None:
                return found
        return None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return err
        if isinstance(err, str):
            return {"message": err}
        if "code" in body and "message" in body:
            return dict(body)
    return None


def _error_from_google(err, *, status=None, request_id=None, retry_after=None) -> APIError:
    """Map a Google error object onto lume's APIError family."""
    err = err or {}
    name = str(err.get("status") or "").upper()
    code = err.get("code")
    http = None
    if isinstance(code, int) and code >= 100:
        http = code
    # The RPC name outranks the HTTP code: a quota failure is a rate limit even
    # when it arrives inside a 200 event stream, where there is no code to read.
    if name in _GOOGLE_STATUS:
        http = _GOOGLE_STATUS[name]
    if http is None:
        http = int(status) if status else 500
    cls, retryable = _class_for_status(int(http))
    message = err.get("message") or f"Gemini error {name or http}"
    if cls is RateLimitError:
        return RateLimitError(message, status=http, type=name or None,
                              request_id=request_id, retry_after=retry_after)
    return cls(message, status=http, type=name or None, request_id=request_id,
               retryable=retryable)


def _error_from_response(status, reason, headers, body: bytes) -> APIError:
    """Build the right exception from a non-200 HTTP response."""
    request_id = _header(headers, "x-request-id") or _header(headers, "request-id")
    retry_after = _parse_retry_after(_header(headers, "retry-after"))
    err = _error_doc(body)
    if err is not None:
        return _error_from_google(err, status=status, request_id=request_id,
                                  retry_after=retry_after)
    text = bytes(body or b"")[:200].decode("utf-8", "replace").strip()
    cls, retryable = _class_for_status(int(status))
    message = text or str(reason or "HTTP error")
    if cls is RateLimitError:
        return RateLimitError(message, status=status, request_id=request_id,
                              retry_after=retry_after)
    return cls(message, status=status, request_id=request_id, retryable=retryable)


# ---------------------------------------------------------------------- translation

#: `finishReason` -> lume's `stop_reason`. The blocked reasons all collapse onto
#: "refusal", the same word `lume.api` surfaces for an Anthropic refusal, and
#: they arrive as a `done` event rather than an exception: a model that declined
#: is an answer the app should render, not a transport failure.
FINISH_REASONS: dict = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "refusal",
    "RECITATION": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "BLOCKLIST": "refusal",
    "SPII": "refusal",
    "IMAGE_SAFETY": "refusal",
    "LANGUAGE": "refusal",
    "MALFORMED_FUNCTION_CALL": "error",
    "OTHER": "other",
}

_REFUSALS = frozenset(k for k, v in FINISH_REASONS.items() if v == "refusal")


class _StreamState:
    """Per-attempt accumulator: model, cumulative usage, stop reason, progress."""

    def __init__(self) -> None:
        self.model = None
        self.request_id = None
        self.stop_reason = None
        self.stop_details = None
        self.started = False
        self.saw_finish = False
        self.emitted = False
        # Kept as raw wire counters because `usageMetadata` is *cumulative* and
        # partial: each record restates the totals so far, and any field may be
        # absent from any record. Adding them up would multiply the bill.
        self.prompt_tokens = 0
        self.candidate_tokens = 0
        self.thought_tokens = 0
        self.cached_tokens = 0

    def absorb(self, meta: dict) -> None:
        """Replace the counters this record restates; leave the rest alone."""
        for field, name in (("prompt_tokens", "promptTokenCount"),
                            ("candidate_tokens", "candidatesTokenCount"),
                            ("thought_tokens", "thoughtsTokenCount"),
                            ("cached_tokens", "cachedContentTokenCount")):
            value = meta.get(name)
            if value is not None:
                try:
                    setattr(self, field, int(value))
                except (TypeError, ValueError):
                    pass

    @property
    def usage(self) -> Usage:
        """The counters as a lume Usage.

        `cachedContentTokenCount` is the part of the prompt served from cache,
        and Google counts it *inside* `promptTokenCount` — lume bills the two
        buckets separately, so it is subtracted out here rather than charged at
        full rate and again at the cache rate.
        """
        return Usage(
            input_tokens=max(0, self.prompt_tokens - self.cached_tokens),
            output_tokens=self.candidate_tokens + self.thought_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=self.cached_tokens,
        )


def _translate(state: _StreamState, data: dict) -> list:
    """Map one decoded `GenerateContentResponse` onto zero or more StreamEvents."""
    if not isinstance(data, dict):
        return []
    out = []

    err = _error_doc(data)
    if err is not None:
        return [StreamEvent(kind="error",
                            error=_error_from_google(err, request_id=state.request_id))]

    state.request_id = data.get("responseId") or state.request_id
    if not state.started:
        state.started = True
        state.model = data.get("modelVersion") or state.model
        out.append(StreamEvent(kind="start", model=state.model, usage=state.usage))
    elif data.get("modelVersion"):
        state.model = data["modelVersion"]

    feedback = data.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        # The *prompt* was blocked: there is no candidate and no finishReason, so
        # without this the turn would look like a truncated stream.
        state.stop_reason = "refusal"
        state.stop_details = {"reason": feedback["blockReason"], "source": "prompt"}
        if feedback.get("safetyRatings"):
            state.stop_details["safety_ratings"] = feedback["safetyRatings"]
        state.saw_finish = True

    candidates = data.get("candidates") or []
    if candidates and isinstance(candidates[0], dict):
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if not text:
                continue
            # A part is a thought summary only when it says so; everything else
            # with text is the answer.
            out.append(StreamEvent(kind="thinking" if part.get("thought") else "text",
                                   text=str(text)))
        finish = candidate.get("finishReason")
        if finish and finish != "FINISH_REASON_UNSPECIFIED":
            state.stop_reason = FINISH_REASONS.get(finish, str(finish).lower())
            details = {"reason": finish}
            if finish in _REFUSALS:
                details["refusal"] = True
            if candidate.get("safetyRatings"):
                details["safety_ratings"] = candidate["safetyRatings"]
            if candidate.get("finishMessage"):
                details["message"] = redact_key(candidate["finishMessage"])
            state.stop_details = details
            state.saw_finish = True

    meta = data.get("usageMetadata")
    if isinstance(meta, dict):
        state.absorb(meta)
        out.append(StreamEvent(kind="usage", usage=state.usage,
                               stop_reason=state.stop_reason,
                               stop_details=state.stop_details, model=state.model))
    return out


# ------------------------------------------------------------------- request bodies


def _flatten(content) -> str:
    """lume content -> one string.

    Content is either a plain string or a list of `{"type": "text", ...}` blocks.
    This client speaks text only, so a block of any other type is dropped rather
    than mistranslated into a `part` Gemini would reject.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    pieces = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") in (None, "text") and block.get("text"):
                pieces.append(str(block["text"]))
        elif block:
            pieces.append(str(block))
    return "".join(pieces)


def _to_contents(messages) -> list:
    """lume messages -> Gemini `contents`, roles renamed and runs merged.

    Two differences from the Anthropic shape. Gemini calls the assistant
    ``"model"``. And while the REST endpoint tolerates consecutive same-role
    turns, every documented example and the SDK's own chat session alternate, so
    a run of same-role messages is merged into one `Content` carrying several
    `parts` — which preserves the turn boundaries without betting on the
    server's tolerance. Anything that is not the assistant is sent as `"user"`,
    including a stray `"system"` message: the system prompt has its own field.
    """
    contents = []
    for msg in messages or ():
        if not isinstance(msg, dict) or "role" not in msg:
            raise ValueError("each message needs a 'role' and 'content'")
        role = "model" if msg["role"] in ("assistant", "model") else "user"
        text = _flatten(msg.get("content", ""))
        if not text:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    if not contents:
        raise ValueError("messages must not be empty")
    return contents


# --------------------------------------------------------------------------- client


class GeminiClient:
    """Streaming Google Gemini client over the same transport seam as `lume.api`.

    `transport` may be None or "http" for plain HTTPS, or any object exposing
    ``open(url, headers, body, timeout)`` — optionally with a fifth ``cancel``
    argument — returning an iterable of bytes with ``.status``, ``.reason`` and
    ``.headers``. Tests inject canned SSE through it; nothing else reaches the
    network.
    """

    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 600.0, max_retries: int = 4, transport=None) -> None:
        if not api_key or not str(api_key).strip():
            raise ValueError("an API key is required")
        # Goes on the wire as an ordinary string but reprs as '***', so a
        # traceback formatter that captures locals cannot print the header dict.
        self._api_key = _Secret(str(api_key).strip())
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        # Backoff knobs are attributes so tests (and impatient users) can shrink them.
        self.backoff_base = 0.5
        self.backoff_max = 16.0
        self.max_retry_wall = 60.0
        self._sleep = time.sleep
        self.transport = HTTPTransport() if transport in (None, "http") else transport

    def __repr__(self) -> str:
        # No key material, not even a prefix.
        return f"<lume.providers.gemini.GeminiClient base_url={self.base_url!r} key=***>"

    __str__ = __repr__

    def url_for(self, model) -> str:
        """The streaming endpoint for one model. Never carries the key."""
        spec = resolve_model(model)
        return (f"{self.base_url}/models/"
                f"{urllib.parse.quote(spec.id, safe='')}:streamGenerateContent?alt=sse")

    # ------------------------------------------------------------------ requests

    def build_body(self, *, model=DEFAULT_MODEL, messages, system=None,
                   max_tokens: int = 32000, thinking: bool = True, effort: str = "high",
                   temperature=None, cache: bool = True) -> dict:
        """Build the exact JSON body for a streaming turn. No secrets in the result.

        `cache` is accepted for signature compatibility and has nothing to send:
        Gemini's implicit context caching is automatic on every model in the
        table and has no request field, and explicit caching needs a
        server-side cache resource created ahead of time, which a chat client
        has nowhere to put. Cache *hits* still come back in the usage.
        """
        spec = resolve_model(model)
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        max_tokens = max(1, min(int(max_tokens), spec.max_output))

        config: dict = {"maxOutputTokens": max_tokens}
        # Gemini accepts sampling parameters, but the spec still gates them so a
        # model that stops accepting them needs one table edit and no code.
        if temperature is not None and spec.supports_temperature:
            config["temperature"] = float(temperature)
        thinking_config = _thinking_config(spec, thinking, effort, max_tokens)
        if thinking_config is not None:
            config["thinkingConfig"] = thinking_config

        body: dict = {"contents": _to_contents(messages), "generationConfig": config}
        text = _flatten(system)
        if text:
            body["systemInstruction"] = {"parts": [{"text": text}]}
        return body

    def _build_headers(self) -> dict:
        # The key goes in a header, never in the query string: a URL ends up in
        # proxy logs, crash reports and shell history, and a header does not.
        return {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "x-goog-api-key": _Secret(self._api_key),
        }

    # ------------------------------------------------------------------- streaming

    def stream(self, *, model: str = DEFAULT_MODEL, messages, system=None,
               max_tokens: int = 32000, thinking: bool = True, effort: str = "high",
               temperature=None, cache: bool = True, cancel=None):
        """Stream one assistant turn as `lume.api.StreamEvent`s.

        Signature-identical to :meth:`lume.api.Client.stream`, so the app can
        hold either without knowing which. Argument errors raise immediately;
        transport errors raise from the generator. `cancel` is a
        `threading.Event`: setting it aborts the request, closes the socket and
        raises `CancelledError` — including while the connect is still
        outstanding and while the server is thinking with nothing on the wire.
        A cancel never surfaces as a retryable error, so a cancelled turn is
        never quietly sent again.
        """
        spec = resolve_model(model)
        body = self.build_body(model=spec, messages=messages, system=system,
                               max_tokens=max_tokens, thinking=thinking, effort=effort,
                               temperature=temperature, cache=cache)
        return self._run(self.url_for(spec), body, self._build_headers(), cancel)

    def _run(self, url, body: dict, headers: dict, cancel):
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        started = time.monotonic()
        planned = 0.0
        attempt = 0
        while True:
            state = _StreamState()
            try:
                yield from self._attempt(url, raw, headers, cancel, state)
                return
            except (CancelledError, GeneratorExit):
                raise
            except _INTERNAL_BUGS:
                # A TypeError from this translator is a bug, not a network
                # failure: laundering it into a NetworkError hides it, retries it
                # four times and blames the user's connection.
                raise
            except Exception as exc:  # noqa: BLE001 - normalised on the next line
                failure = self._scrub(_transport_error(exc))
            # Outside the handler on purpose: raising inside it would re-attach
            # the original exception as `__context__`, and the frames behind it
            # hold the request headers — key included.
            #
            # A retry may only happen while the caller has seen nothing at all,
            # or the reply would be duplicated on screen.
            if state.emitted or not failure.retryable or attempt >= self.max_retries:
                _raise_detached(failure)
            delay = self._delay(attempt, getattr(failure, "retry_after", None))
            used = max(time.monotonic() - started, planned)
            if used + delay > self.max_retry_wall:
                _raise_detached(self._out_of_budget(failure, delay, used))
            self._pause(delay, cancel)
            planned += delay
            attempt += 1

    def _delay(self, attempt: int, retry_after) -> float:
        if retry_after is not None:
            return max(0.0, float(retry_after))
        window = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        # Full jitter over the window: synchronised clients must not resonate.
        return random.uniform(0.0, window)

    def _out_of_budget(self, exc, delay: float, used: float):
        """Annotate an error we are giving up on because the wait is too long."""
        note = (f"not retried: the next attempt would wait {delay:.0f}s, past the "
                f"{self.max_retry_wall:.0f}s retry budget")
        exc.message = redact_key(f"{exc.message} ({note})" if exc.message else note)
        exc.args = (exc.message,)
        return exc

    def _pause(self, delay: float, cancel) -> None:
        if cancel is not None:
            if cancel.wait(delay):
                raise CancelledError("cancelled while backing off")
            return
        if delay > 0:
            self._sleep(delay)

    def _open(self, url, raw, headers, cancel):
        """Open the response, with `cancel` observed *while* it blocks.

        Until this returns there is no stream to close, so a cancel during the
        handshake is raised here or nowhere.
        """
        opener = self.transport.open
        if cancel is None:
            return opener(url, headers, raw, self.timeout)
        if _accepts_cancel(opener):
            return opener(url, headers, raw, self.timeout, cancel=cancel)
        return self._open_supervised(opener, url, raw, headers, cancel)

    def _open_supervised(self, opener, url, raw, headers, cancel):
        """Run a cancel-unaware transport's `open()` where a cancel can escape it.

        Whatever the abandoned call eventually produces is closed, not leaked.
        """
        box: dict = {}

        def run():
            try:
                box["stream"] = opener(url, headers, raw, self.timeout)
            except BaseException as exc:  # noqa: BLE001 - re-raised on this thread
                box["error"] = exc

        worker = threading.Thread(target=run, name="lume-gemini-open", daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(0.02)
            if worker.is_alive() and cancel.is_set():
                threading.Thread(target=_reap, args=(worker, box),
                                 name="lume-gemini-open-reap", daemon=True).start()
                raise CancelledError("cancelled while opening the connection")
        if "error" in box:
            raise box["error"]
        stream = box.get("stream")
        if cancel.is_set():
            _close_quietly(stream)
            raise CancelledError("cancelled while opening the connection")
        return stream

    def _attempt(self, url, raw, headers, cancel, state):
        if cancel is not None and cancel.is_set():
            raise CancelledError("cancelled before the request was made")

        stream = self._open(url, raw, headers, cancel)
        watchdog_done = threading.Event()
        watcher = None
        if cancel is not None:
            # Same mechanism as lume.api: a side thread closes the socket, so a
            # read parked in the C layer returns at once instead of sitting there
            # until the server says something.
            watcher = threading.Thread(target=_watch_cancel,
                                       args=(stream, cancel, watchdog_done),
                                       name="lume-gemini-cancel", daemon=True)
            watcher.start()
        try:
            status = int(getattr(stream, "status", 200) or 200)
            hdrs = getattr(stream, "headers", None)
            state.request_id = (_header(hdrs, "x-request-id")
                                or _header(hdrs, "request-id") or state.request_id)
            if status != 200:
                raise _error_from_response(status, getattr(stream, "reason", ""),
                                           hdrs, _drain_body(stream, _READ_LIMIT))
            ctype = _header(hdrs, "content-type")
            if ctype and "event-stream" not in str(ctype).lower():
                # A 200 that is not an event stream is a proxy or a captive
                # portal answering for the API. Report the body, not "empty
                # stream" with the one useful thing thrown away.
                raise self._not_sse(ctype, _drain_body(stream, _READ_LIMIT), state)

            decoder = SSEDecoder()
            for chunk in stream:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("cancelled mid-stream")
                for _name, payload in decoder.feed(chunk):
                    yield from self._emit(payload, state)
            if cancel is not None and cancel.is_set():
                raise CancelledError("cancelled mid-stream")
            for _name, payload in decoder.close():
                yield from self._emit(payload, state)
            if not state.saw_finish:
                # Gemini has no `message_stop`; a finished answer is one that
                # carried a finishReason. Without one the connection was cut.
                raise NetworkError("stream ended before a finishReason",
                                   request_id=state.request_id,
                                   retryable=not state.emitted)
            yield StreamEvent(kind="done", usage=state.usage,
                              stop_reason=state.stop_reason,
                              stop_details=state.stop_details, model=state.model)
        except Exception:  # noqa: BLE001 - re-raised unless we caused it
            # Once the cancel event is set we are tearing the connection down, so
            # whatever the read raises on the way out is cancellation wearing
            # someone else's exception type.
            if cancel is not None and cancel.is_set():
                raise CancelledError("cancelled mid-stream") from None
            raise
        finally:
            watchdog_done.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            try:
                stream.close()
            except Exception:
                pass

    def _not_sse(self, content_type, body: bytes, state) -> APIError:
        err = _error_doc(body)
        if err is not None:
            return _error_from_google(err, status=200, request_id=state.request_id)
        text = " ".join(bytes(body or b"").decode("utf-8", "replace").split())[:200]
        message = f"expected an event stream, got {str(content_type).strip()!r}"
        if text:
            message += f": {text}"
        # Not retryable: whatever answered instead of the API answers the same
        # way again, and four more round trips only delay the explanation.
        return NetworkError(message, status=200, request_id=state.request_id,
                            retryable=False)

    def _emit(self, payload, state):
        try:
            data = json.loads(payload)
        except ValueError:
            return  # a keep-alive or a truncated tail: nothing decodable
        if isinstance(data, list):
            # A mid-stream failure arrives as a one-element array, not an object.
            err = _error_doc(data)
            if err is None:
                return
            data = {"error": err}
        if not isinstance(data, dict):
            return
        for event in _translate(state, data):
            yield from self._deliver(event, state)

    def _scrub(self, exc):
        """Last line of defence: redact the literal key, whatever shape it has."""
        if exc is None:
            return exc
        message = redact_key(exc.message)
        if self._api_key and self._api_key in message:
            message = message.replace(str(self._api_key), "***")
        if message != exc.message:
            exc.message = message
            exc.args = (message,)
        return exc

    def _deliver(self, event, state):
        if event.kind == "error":
            exc = self._scrub(event.error)
            # Before any output a retryable failure is invisible: retry silently.
            if exc is not None and exc.retryable and not state.emitted:
                raise exc
            yield event
            state.emitted = True
            if exc is not None:
                raise exc
            return
        yield event
        if event.kind != "ping":
            state.emitted = True

    def close(self) -> None:
        """Release transport resources. Safe to call twice."""
        closer = getattr(self.transport, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                pass


# ------------------------------------------------------------------------- registry


def provider():
    """The registry entry, built once when `lume.providers` imports this module."""
    from . import Provider

    override = (os.environ.get("GEMINI_BASE_URL") or "").strip()
    return Provider(
        name="google",
        label="Google Gemini",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        base_url=override or DEFAULT_BASE_URL,
        models=dict(MODELS),
        factory=GeminiClient,
        aliases=dict(ALIASES),
        doc_url="https://aistudio.google.com/apikey",
    )
