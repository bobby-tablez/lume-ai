"""OpenAI (and OpenAI-compatible) provider: model table, chat-completions streaming.

Stdlib only, and not one line of network code of its own: the transport, the SSE
decoder, the retry loop, the cancellation watchdog and the error hierarchy all
come from :mod:`lume.api`. :class:`OpenAIClient` is that client with a different
wire dialect bolted on — a different URL, a different request body and a
different event translator — so anything the Anthropic client learns about
sockets, cancels and redaction is inherited here for free, and the app cannot
tell which of the two it is holding.

**OpenAI-compatible endpoints.** Everything here speaks plain
``POST {base_url}/chat/completions`` with a bearer token, so the same client
drives Groq, xAI, DeepSeek, Together, Fireworks, OpenRouter, Ollama and LM Studio.
Point it somewhere else with ``OPENAI_BASE_URL`` (honoured by :func:`provider`)
or ``OpenAIClient(base_url=...)``::

    OPENAI_BASE_URL=https://api.groq.com/openai/v1
    OPENAI_BASE_URL=https://api.deepseek.com/v1
    OPENAI_BASE_URL=http://localhost:11434/v1        # Ollama
    OPENAI_BASE_URL=http://localhost:1234/v1         # LM Studio

A model id this table has never heard of is not an error — it gets a permissive
generic spec (temperature allowed, no ``reasoning_effort``, plain ``max_tokens``,
zero prices), which is exactly what a third-party endpoint wants.

Two honest caveats about money, both inherited from :class:`lume.api.Usage`:
``cost()`` prices a cache *read* at a fixed 0.1x of input, which is exact for the
gpt-5 family but cheap for the gpt-4.1/4o families (0.25x/0.5x — see
``price_cache_read`` in the table below); and ``Usage.cost("gpt-5")`` with a bare
string raises, because that resolver only knows Anthropic ids. Pass the
:class:`~lume.api.ModelSpec`.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from ..api import (EFFORTS, Client, ModelSpec, NetworkError, StreamEvent, Usage,
                   _class_for_status, _Secret)

__all__ = [
    "MODELS", "ALIASES", "DEFAULT_MODEL", "DEFAULT_BASE_URL", "EFFORT_LEVELS",
    "OpenAIClient", "provider", "resolve_model", "model_names",
]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.1"

# OpenAI's own ladder. `minimal` (gpt-5) and `none` (gpt-5.1) sit below `low` and
# are not in lume's shared vocabulary, so they are reachable only by asking for
# them by name — or by turning thinking off, which is what they mean.
EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high")


# --------------------------------------------------------------------------- models


def _spec(mid, label, context, max_output, price_in, price_out, cached, *,
          effort=True, efforts=("minimal", "low", "medium", "high"),
          temperature=False) -> ModelSpec:
    # `price_cache_write` is the *input* rate, not a premium: OpenAI's prompt
    # caching is automatic and writing to it costs nothing extra.
    return ModelSpec(
        id=mid, label=label, context=context, max_output=max_output,
        price_in=price_in, price_out=price_out,
        price_cache_write=price_in, price_cache_read=cached,
        supports_temperature=temperature, supports_effort=effort,
        thinking="adaptive" if effort else "none",
        efforts=tuple(efforts) if effort else (),
        supports_thinking_display=False,
        # Reasoning is never "omitted" here — it is dialled to `none`/`minimal`;
        # the chat models have nothing to turn off.
        thinking_off="omit",
        cache_min=1024,
    )


# Prices are USD per 1M tokens, as published on platform.openai.com/pricing.
# **Figures as of 2026-08-18** — re-check them before quoting anyone a bill; a
# stale table here is a wrong number in the footer, not a crash.
#
#   id             ctx        max out    in     out    cached in
#   gpt-5.1        400k       128k       1.25   10.00  0.125   (0.10x)
#   gpt-5          400k       128k       1.25   10.00  0.125   (0.10x)
#   gpt-5-mini     400k       128k       0.25    2.00  0.025   (0.10x)
#   gpt-5-nano     400k       128k       0.05    0.40  0.005   (0.10x)
#   gpt-4.1        1,047,576   32k       2.00    8.00  0.50    (0.25x)
#   gpt-4.1-mini   1,047,576   32k       0.40    1.60  0.10    (0.25x)
#   gpt-4o-mini    128k        16k       0.15    0.60  0.075   (0.50x)
MODELS: dict = {
    m.id: m for m in (
        # Reasoning models: `reasoning_effort`, no `temperature`, and the API
        # wants `max_completion_tokens` (plain `max_tokens` is a 400).
        _spec("gpt-5.1", "GPT-5.1", 400_000, 128_000, 1.25, 10.0, 0.125,
              efforts=("none", "low", "medium", "high")),
        _spec("gpt-5", "GPT-5", 400_000, 128_000, 1.25, 10.0, 0.125),
        _spec("gpt-5-mini", "GPT-5 mini", 400_000, 128_000, 0.25, 2.0, 0.025),
        _spec("gpt-5-nano", "GPT-5 nano", 400_000, 128_000, 0.05, 0.40, 0.005),
        # Chat models: sampling parameters work, there is no reasoning to ask
        # for, and `max_tokens` is the field they have always taken.
        _spec("gpt-4.1", "GPT-4.1", 1_047_576, 32_768, 2.0, 8.0, 0.50,
              effort=False, temperature=True),
        _spec("gpt-4.1-mini", "GPT-4.1 mini", 1_047_576, 32_768, 0.40, 1.60, 0.10,
              effort=False, temperature=True),
        _spec("gpt-4o-mini", "GPT-4o mini", 128_000, 16_384, 0.15, 0.60, 0.075,
              effort=False, temperature=True),
    )
}

ALIASES: dict = {
    "gpt": DEFAULT_MODEL,
    "openai": DEFAULT_MODEL,
    "gpt5": "gpt-5",
    "gpt-5.1": "gpt-5.1",
    "gpt51": "gpt-5.1",
    "gpt-5-1": "gpt-5.1",
    "mini": "gpt-5-mini",
    "gpt5-mini": "gpt-5-mini",
    "nano": "gpt-5-nano",
    "gpt5-nano": "gpt-5-nano",
    "gpt4.1": "gpt-4.1",
    "gpt41": "gpt-4.1",
    "4.1": "gpt-4.1",
    "4o-mini": "gpt-4o-mini",
}


def model_names() -> list:
    """Canonical model ids, in menu order."""
    return list(MODELS)


def resolve_model(name) -> ModelSpec:
    """Look up a :class:`~lume.api.ModelSpec` by id or alias.

    Unlike the Anthropic resolver this never raises: an id it does not know is a
    model on some OpenAI-compatible endpoint, and refusing to talk to it would
    defeat the point. Such a model gets a permissive spec — sampling allowed, no
    reasoning parameters, plain ``max_tokens``, and zero prices, so a cost of
    $0.00 marks "we do not know this vendor's rates" rather than inventing one.
    """
    if isinstance(name, ModelSpec):
        return name
    if not name:
        return MODELS[DEFAULT_MODEL]
    key = str(name).strip()
    if key in MODELS:
        return MODELS[key]
    low = key.lower()
    if low in MODELS:
        return MODELS[low]
    if low in ALIASES:
        return MODELS[ALIASES[low]]
    return _generic(key)


def _generic(mid: str) -> ModelSpec:
    """A spec for a model only the far end knows about."""
    return ModelSpec(
        id=mid, label=mid, context=128_000, max_output=128_000,
        price_in=0.0, price_out=0.0, price_cache_write=0.0, price_cache_read=0.0,
        supports_temperature=True, supports_effort=False, thinking="none",
        efforts=(), supports_thinking_display=False, thinking_off="omit",
    )


def _uses_max_completion_tokens(spec: ModelSpec) -> bool:
    """True where the output cap is ``max_completion_tokens``, not ``max_tokens``.

    The reasoning models reject the old field outright (400: "Unsupported
    parameter: 'max_tokens'"), while plenty of compatible servers have never
    heard of the new one — so the choice follows the model rather than being a
    single global guess. In this table that line falls exactly on
    ``supports_effort``: gpt-5* take the new field, gpt-4.1/4o and every unknown
    model take the old one.
    """
    return spec.supports_effort


def _off_effort(spec: ModelSpec):
    """The effort level that means "do not think", or None if there is none.

    gpt-5.1 spells it ``none``; gpt-5 and its minis spell it ``minimal``. Both
    come straight off the model's own ladder, so a new model is described by its
    ``efforts`` tuple alone.
    """
    for level in ("none", "minimal"):
        if level in spec.efforts:
            return level
    return None


# ---------------------------------------------------------------------- translation


# OpenAI's finish reasons, said in lume's (Anthropic's) vocabulary, so the app
# renders "hit the length cap" the same way whoever answered.
FINISH_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


def _flatten(content) -> str:
    """Message content -> plain text, whether it arrived as a string or blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    out = []
    for block in content:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("type") in (None, "text"):
            out.append(str(block.get("text") or ""))
    return "".join(out)


def _reasoning_text(delta: dict) -> str:
    """Thinking text out of a delta, under any of the three names in the wild.

    OpenAI's own summaries and DeepSeek use ``reasoning_content``; OpenRouter and
    several proxies use ``reasoning``, sometimes as an object rather than a string.
    """
    for name in ("reasoning_content", "reasoning"):
        value = delta.get(name)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            text = _flatten(value.get("content") or value.get("text"))
            if text:
                return text
    return ""


def _usage_from_payload(payload: dict, model=None) -> Usage:
    """OpenAI's ``usage`` object as a :class:`~lume.api.Usage`.

    ``prompt_tokens`` *includes* the cached prefix, where Anthropic's
    ``input_tokens`` excludes it — so the cached part is subtracted out here.
    Leaving it in would bill every cached token twice: once at full rate and once
    at the cache rate.
    """
    prompt = int(payload.get("prompt_tokens") or 0)
    completion = int(payload.get("completion_tokens") or 0)
    details = payload.get("prompt_tokens_details")
    cached = 0
    if isinstance(details, dict):
        cached = max(0, min(prompt, int(details.get("cached_tokens") or 0)))
    return Usage(input_tokens=prompt - cached, output_tokens=completion,
                 cache_read_input_tokens=cached, served_model=model or None)


def _stream_error(payload: dict, request_id=None):
    """An ``error`` object inside a 200 stream -> the right APIError subclass."""
    code = payload.get("code")
    status = payload.get("status") or payload.get("status_code")
    try:
        status = int(status if status is not None else code)
    except (TypeError, ValueError):
        status = 500
    cls, retryable = _class_for_status(status)
    message = payload.get("message") or "stream error"
    exc = cls(message, status=status, type=payload.get("type"),
              request_id=request_id, retryable=retryable)
    return exc


def translate(state, data: dict) -> list:
    """Map one decoded ``data:`` payload onto zero or more :class:`StreamEvent`s.

    `state` is :class:`lume.api._StreamState`, carried across the whole attempt.
    The shape of a turn on the wire is: chunks of ``choices[0].delta``, then a
    chunk carrying ``finish_reason``, then (thanks to ``stream_options``) a
    choice-less chunk carrying ``usage``, then ``[DONE]`` — which lands here as
    ``start``, ``thinking``/``text``…, ``usage``, ``done``.
    """
    if not isinstance(data, dict):
        return []
    out = []

    error = data.get("error")
    if isinstance(error, dict):
        return [StreamEvent(kind="error", error=_stream_error(error, state.request_id))]

    if data.get("id"):
        state.request_id = state.request_id or data["id"]
    choices = data.get("choices") or []
    usage = data.get("usage")

    if not choices and not isinstance(usage, dict):
        return [StreamEvent(kind="ping")]  # a heartbeat, or a chunk with nothing in it

    if not getattr(state, "started", False):
        state.started = True
        state.model = data.get("model") or state.model
        state.usage.served_model = state.model
        out.append(StreamEvent(kind="start", model=state.model,
                               usage=replace(state.usage)))

    for choice in choices:
        # `n > 1` is not something a chat UI can show; only the first completion
        # is rendered, and asking for more is not a thing this client does.
        if choice.get("index", 0):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        thinking = _reasoning_text(delta)
        if thinking:
            out.append(StreamEvent(kind="thinking", text=thinking))
        text = _flatten(delta.get("content"))
        if text:
            out.append(StreamEvent(kind="text", text=text))
        refusal = delta.get("refusal")
        if isinstance(refusal, str) and refusal:
            # A refusal is prose the user should see, not a silent empty turn.
            out.append(StreamEvent(kind="text", text=refusal))
        finish = choice.get("finish_reason")
        if finish:
            state.stop_reason = FINISH_REASONS.get(finish, finish)

    if isinstance(usage, dict):
        state.usage = _usage_from_payload(usage, state.model)
        out.append(StreamEvent(kind="usage", usage=replace(state.usage),
                               stop_reason=state.stop_reason, model=state.model))
    return out


# --------------------------------------------------------------------------- client


class OpenAIClient(Client):
    """Streaming chat-completions client, in :class:`lume.api.Client`'s clothing.

    Everything structural is inherited: the transport seam, retries with jittered
    backoff and ``retry-after``, the cancel watchdog that shuts the socket down
    mid-connect, and the redaction that keeps the key out of every error. What is
    overridden is only the dialect — the URL, the headers, the request body and
    the SSE translation.

    ``base_url`` may point at any OpenAI-compatible endpoint; it is used verbatim
    with ``/chat/completions`` appended, so it should include the vendor's own
    version segment (``.../v1``).
    """

    def __init__(self, api_key, *, base_url=DEFAULT_BASE_URL, timeout=600.0,
                 max_retries=4, transport=None) -> None:
        super().__init__(api_key, base_url=base_url or DEFAULT_BASE_URL,
                         timeout=timeout, max_retries=max_retries, transport=transport)

    def __repr__(self) -> str:
        # No key material, not even a prefix.
        return f"<lume.providers.openai.OpenAIClient base_url={self.base_url!r} key=***>"

    __str__ = __repr__

    @property
    def url(self) -> str:
        """The one endpoint this client posts to."""
        return f"{self.base_url}/chat/completions"

    # ------------------------------------------------------------------ requests

    def build_body(self, *, model=DEFAULT_MODEL, messages, system=None,
                   max_tokens: int = 32000, thinking: bool = True, effort: str = "high",
                   temperature=None, cache: bool = True) -> dict:
        """Build the exact JSON body for a streaming turn. No secrets in the result.

        lume's internal message shape (``role`` plus a string or a list of text
        blocks) is flattened to the strings chat-completions wants, and `system`
        becomes a leading ``system`` message. ``cache`` is accepted for signature
        compatibility and sent nowhere: OpenAI's prompt caching is automatic above
        ~1024 tokens, free to write, and cannot be asked for or declined.
        """
        spec = resolve_model(model)
        if not messages:
            raise ValueError("messages must not be empty")
        if effort is not None and effort not in EFFORTS and effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}")

        prepared = []
        if system is not None and system != "":
            text = _flatten(system)
            if text:
                # `developer` is what the reasoning models call this role now, but
                # `system` is still accepted by every one of them and is the only
                # spelling most compatible servers understand. Compatibility wins.
                prepared.append({"role": "system", "content": text})
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg:
                raise ValueError("each message needs a 'role' and 'content'")
            prepared.append({"role": msg["role"], "content": _flatten(msg.get("content"))})

        body: dict = {
            "model": spec.id,
            "messages": prepared,
            "stream": True,
            # Without this the final usage chunk is never sent and a turn cannot
            # be costed at all.
            "stream_options": {"include_usage": True},
        }

        cap = max(1, min(int(max_tokens), spec.max_output))
        body["max_completion_tokens" if _uses_max_completion_tokens(spec)
             else "max_tokens"] = cap

        if spec.supports_effort:
            level = spec.clamp_effort(effort) if thinking else _off_effort(spec)
            if level:
                body["reasoning_effort"] = level

        # Sampling parameters are a 400 on the reasoning models.
        if temperature is not None and spec.supports_temperature:
            body["temperature"] = float(temperature)
        return body

    def _build_headers(self, *, betas=()) -> dict:
        # `betas` exists only to match the base class; OpenAI has no such header.
        return {
            "content-type": "application/json",
            "accept": "text/event-stream",
            # The credential goes on the wire as an ordinary string but reprs as
            # '***', so a traceback that captures locals cannot print it.
            "authorization": _Secret(f"Bearer {self._api_key}"),
        }

    # ------------------------------------------------------------------- streaming

    def stream(self, *, model: str = DEFAULT_MODEL, messages, system=None,
               max_tokens: int = 32000, thinking: bool = True, effort: str = "high",
               temperature=None, cache: bool = True, cancel=None):
        """Stream one assistant turn as StreamEvents.

        Signature-identical to :meth:`lume.api.Client.stream`, on purpose: the app
        holds one or the other and never asks which. Argument errors raise
        immediately; transport errors raise from the generator. `cancel` is a
        ``threading.Event``, honoured while the connect is outstanding and while
        the server is quiet, and it never surfaces as a retryable error.
        """
        body = self.build_body(model=model, messages=messages, system=system,
                               max_tokens=max_tokens, thinking=thinking, effort=effort,
                               temperature=temperature, cache=cache)
        return self._run(body, self._build_headers(), cancel)

    def _attempt(self, body, raw, headers, cancel, state):
        """The inherited attempt, plus tolerance for a missing ``[DONE]``.

        The base client treats a stream that ends without its terminator as a
        truncated answer, which is right for Anthropic and right here too — except
        that a handful of compatible servers close the connection straight after
        the ``finish_reason`` chunk. A finished turn is a finished turn.

        Only a truncation is forgiven, and only when nobody cancelled: a
        `CancelledError` dressed up as a completed answer would leave the app
        showing a turn the user just stopped.
        """
        try:
            yield from super()._attempt(body, raw, headers, cancel, state)
        except NetworkError:
            if (state.saw_stop or state.stop_reason is None
                    or (cancel is not None and cancel.is_set())):
                raise
            state.saw_stop = True
            yield from self._deliver(
                StreamEvent(kind="done", usage=replace(state.usage),
                            stop_reason=state.stop_reason, model=state.model), state)

    def _emit(self, name, payload, state):
        """One SSE record -> delivered StreamEvents. ``[DONE]`` ends the turn."""
        if payload.strip() == "[DONE]":
            state.saw_stop = True
            yield from self._deliver(
                StreamEvent(kind="done", usage=replace(state.usage),
                            stop_reason=state.stop_reason, model=state.model), state)
            return
        try:
            data = json.loads(payload)
        except ValueError:
            return  # a keep-alive or a truncated tail: nothing decodable
        for event in translate(state, data):
            yield from self._deliver(event, state)


# ------------------------------------------------------------------------- registry


def provider():
    """This module as a :class:`lume.providers.Provider`. Called once, at import.

    ``OPENAI_BASE_URL`` is read here rather than in the client so that one
    environment variable redirects the whole app — model list, key lookup and
    streaming — at Groq, DeepSeek, Ollama or anything else that speaks this API.
    """
    from . import Provider

    base = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    return Provider(
        name="openai",
        label="OpenAI",
        env_keys=("OPENAI_API_KEY",),
        base_url=(base or DEFAULT_BASE_URL).rstrip("/"),
        models=MODELS,
        factory=OpenAIClient,
        aliases=ALIASES,
        doc_url="https://platform.openai.com/api-keys",
    )
