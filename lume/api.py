"""Claude API client: model registry, cost maths, SSE streaming, retries.

Stdlib only. The wire format here follows the current Messages API reference —
adaptive thinking, `output_config.effort`, no sampling parameters on the current
models — and deliberately differs from older, widely-copied examples.

Everything that touches the network goes through a *transport*, so the whole
client is exercisable with canned bytes and no socket at all.
"""

from __future__ import annotations

import codecs
import datetime
import errno
import http.client
import inspect
import json
import os
import random
import re
import select
import socket
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass, replace
from email.utils import parsedate_to_datetime

__all__ = [
    "ModelSpec", "MODELS", "DEFAULT_MODEL", "ALIASES", "resolve_model", "model_names",
    "Usage", "StreamEvent",
    "APIError", "AuthError", "BadRequestError", "RateLimitError", "OverloadedError",
    "ServerError", "NetworkError", "CancelledError",
    "find_api_key", "is_oauth_token", "redact",
    "SSEDecoder", "HTTPTransport", "SDKTransport", "Client",
    "API_VERSION", "EFFORTS", "CACHE_MIN_TOKENS", "MAX_CACHE_BREAKPOINTS",
]

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-opus-5"

OAUTH_BETA = "oauth-2025-04-20"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Every effort level the API has, weakest first. It is *not* a per-model ladder:
# `xhigh` arrived with opus-4-7, so sending it to a 4.6-family model is a 400.
# The ladder a given model actually has is `ModelSpec.efforts`.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
EFFORTS_NO_XHIGH = ("low", "medium", "high", "max")

# Cache writes cost 1.25x input at the default 5-minute TTL and 2x at 1 hour.
CACHE_WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0}
CACHE_READ_MULTIPLIER = 0.1

# Prompt caching only kicks in above a minimum prefix; below it a breakpoint is
# pure overhead, so we do not place one. The real minimum is per-model (see
# ModelSpec.cache_min); this is only the fallback for a spec that does not say.
CACHE_MIN_TOKENS = 1024
MAX_CACHE_BREAKPOINTS = 4

_READ_SIZE = 8192
_MAX_ERROR_BODY = 64 * 1024


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelSpec:
    """One model's identity, limits, prices (USD per 1M tokens) and quirks."""

    id: str
    label: str
    context: int
    max_output: int
    price_in: float
    price_out: float
    price_cache_write: float
    price_cache_read: float
    supports_temperature: bool
    supports_effort: bool
    thinking: str  # "adaptive" | "budget" | "none"
    # The effort levels *this* model has. `xhigh` is new on opus-4-7; the 4.6
    # family is low|medium|high|max and rejects `xhigh` with a 400, so there is
    # no one global ladder to validate against.
    efforts: tuple = EFFORTS
    # `thinking.display` is new on opus-4-7 too. On the 4.6 family the field is
    # not understood, and its old behaviour ("summarized") is the default anyway.
    supports_thinking_display: bool = True
    # How this model is told *not* to think. Omitting `thinking` is not "off"
    # everywhere: opus-5 and the sonnets run adaptive thinking when the field is
    # absent, so "off" has to be said out loud.
    #   "disabled"    -> send {"type": "disabled"}
    #   "omit"        -> leaving the field out genuinely means no thinking
    #   "unsupported" -> the model always thinks; an explicit disable is a 400
    thinking_off: str = "disabled"
    # Shortest prefix this model will actually cache; below it a breakpoint is
    # silently ignored by the server (no error, no cache entry).
    cache_min: int = CACHE_MIN_TOKENS
    # Introductory pricing, when the model is on any. `intro_until` is the last
    # day (inclusive, ISO) it applies; after that the list price above is what is
    # charged, with no code change and nothing to remember.
    intro_price_in: "float | None" = None
    intro_price_out: "float | None" = None
    intro_until: "str | None" = None

    def prices(self, when=None) -> tuple:
        """(input, output) $/1M actually charged on `when` (default: today).

        A model on introductory pricing bills the intro rate up to and including
        `intro_until` and the list rate from the next day. Reporting the list
        rate today would show the user a number they are not charged; hard-coding
        the intro rate would show a stale one from the 1st of the month.
        """
        if (self.intro_until and self.intro_price_in is not None
                and _as_date(when) <= _as_date(self.intro_until)):
            return (self.intro_price_in,
                    self.intro_price_out if self.intro_price_out is not None
                    else self.price_out)
        return (self.price_in, self.price_out)

    @property
    def effort_ladder(self) -> tuple:
        """The effort levels this model accepts (empty when it takes none)."""
        return tuple(self.efforts) if self.supports_effort else ()

    def clamp_effort(self, effort):
        """The nearest level this model has, at or below `effort`; None if none.

        `xhigh` on a 4.6-family model is a hard 400, so `lume --effort xhigh -m
        sonnet-4-6` has to mean "as high as this model goes below xhigh" rather
        than a failed request.
        """
        ladder = self.effort_ladder
        if not effort or not ladder:
            return None
        if effort in ladder:
            return effort
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        wanted = EFFORTS.index(effort)
        below = [name for name in ladder
                 if name in EFFORTS and EFFORTS.index(name) <= wanted]
        return max(below, key=EFFORTS.index) if below else ladder[0]

    @property
    def supports_fallbacks(self) -> bool:
        """Server-side refusal fallbacks are offered on the models that can refuse."""
        return self.id in ("claude-opus-5", "claude-fable-5")

    @property
    def disabled_thinking_effort_cap(self):
        """Highest effort that may accompany `thinking: disabled`, or None.

        On opus-5 `{"type": "disabled"}` is a 400 above effort `high`; the pair is
        clamped rather than sent and rejected.
        """
        return "high" if self.id == "claude-opus-5" else None


def _as_date(value=None) -> datetime.date:
    """Coerce None/str/date/datetime to a date; None means today."""
    if value is None:
        return datetime.date.today()
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value))


def _spec(mid, label, context, max_output, price_in, price_out, *,
          temperature=False, effort=True, efforts=EFFORTS, thinking="adaptive",
          thinking_display=True, thinking_off="disabled", cache_min=CACHE_MIN_TOKENS,
          intro=None) -> ModelSpec:
    # `price_cache_write` is the 5-minute-TTL rate (1.25x input); a 1-hour write
    # is 2x, which `Usage.cost` applies from the per-TTL breakdown.
    return ModelSpec(
        id=mid, label=label, context=context, max_output=max_output,
        price_in=price_in, price_out=price_out,
        price_cache_write=round(price_in * CACHE_WRITE_MULTIPLIER["5m"], 6),
        price_cache_read=round(price_in * CACHE_READ_MULTIPLIER, 6),
        supports_temperature=temperature, supports_effort=effort, thinking=thinking,
        efforts=tuple(efforts), supports_thinking_display=thinking_display,
        thinking_off=thinking_off, cache_min=cache_min,
        intro_price_in=(intro or (None, None, None))[0],
        intro_price_out=(intro or (None, None, None))[1],
        intro_until=(intro or (None, None, None))[2],
    )


# `price_in`/`price_out` are the list rates. claude-sonnet-5 is on introductory
# pricing of $2/$10 per Mtok through 2026-08-31: that is carried as `intro=` and
# applied by `Usage.cost`, so a cost shown today is the one actually billed and
# the intro rate expires itself on 2026-09-01 with no code change.
MODELS: dict = {
    m.id: m for m in (
        _spec("claude-opus-5", "Opus 5", 1_000_000, 128_000, 5.0, 25.0,
              thinking_off="disabled", cache_min=512),
        _spec("claude-fable-5", "Fable 5", 1_000_000, 128_000, 10.0, 50.0,
              thinking_off="unsupported", cache_min=512),
        _spec("claude-opus-4-8", "Opus 4.8", 1_000_000, 128_000, 5.0, 25.0,
              thinking_off="omit", cache_min=1024),
        _spec("claude-opus-4-7", "Opus 4.7", 1_000_000, 128_000, 5.0, 25.0,
              thinking_off="omit", cache_min=2048),
        # 4.6 family: no `xhigh` (it arrived with 4.7) and no `thinking.display`.
        _spec("claude-opus-4-6", "Opus 4.6", 1_000_000, 128_000, 5.0, 25.0, temperature=True,
              efforts=EFFORTS_NO_XHIGH, thinking_display=False,
              thinking_off="disabled", cache_min=4096),
        _spec("claude-sonnet-5", "Sonnet 5", 1_000_000, 128_000, 3.0, 15.0,
              thinking_off="disabled", cache_min=1024,
              intro=(2.0, 10.0, "2026-08-31")),
        _spec("claude-sonnet-4-6", "Sonnet 4.6", 1_000_000, 128_000, 3.0, 15.0,
              temperature=True, efforts=EFFORTS_NO_XHIGH, thinking_display=False,
              thinking_off="disabled", cache_min=1024),
        _spec("claude-haiku-4-5", "Haiku 4.5", 200_000, 64_000, 1.0, 5.0,
              temperature=True, effort=False, efforts=(), thinking="budget",
              thinking_display=False, thinking_off="omit", cache_min=4096),
    )
}

ALIASES: dict = {
    "default": DEFAULT_MODEL,
    "opus": "claude-opus-5",
    "opus5": "claude-opus-5",
    "opus-5": "claude-opus-5",
    "opus-4-8": "claude-opus-4-8",
    "opus-4-7": "claude-opus-4-7",
    "opus-4-6": "claude-opus-4-6",
    "sonnet": "claude-sonnet-5",
    "sonnet5": "claude-sonnet-5",
    "sonnet-5": "claude-sonnet-5",
    "sonnet-4-6": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "haiku-4-5": "claude-haiku-4-5",
    "fable": "claude-fable-5",
    "fable5": "claude-fable-5",
    "fable-5": "claude-fable-5",
}


def model_names() -> list:
    """Canonical model ids, in menu order."""
    return list(MODELS)


def resolve_model(name) -> ModelSpec:
    """Look up a ModelSpec by id or alias ('opus', 'sonnet-4.6', a ModelSpec).

    Raises ValueError for anything unknown — guessing would mean guessing a price.
    """
    if isinstance(name, ModelSpec):
        return name
    if not name:
        return MODELS[DEFAULT_MODEL]
    key = str(name).strip().lower().replace(".", "-").replace("_", "-")
    while "--" in key:
        key = key.replace("--", "-")
    if key in MODELS:
        return MODELS[key]
    if key in ALIASES:
        return MODELS[ALIASES[key]]
    if key.startswith("claude-") and key[7:] in ALIASES:
        return MODELS[ALIASES[key[7:]]]
    raise ValueError(f"unknown model {name!r}; known: {', '.join(MODELS)}")


def _resolve_or(name, default: ModelSpec) -> ModelSpec:
    """resolve_model, but a model we have never heard of prices at `default`.

    The server may route a turn to a model newer than this table; a cost figure
    computed at the requested model's rates beats a crash in the footer.
    """
    try:
        return resolve_model(name)
    except ValueError:
        return default


# ---------------------------------------------------------------------------- usage


@dataclass
class Usage:
    """Token counts for one exchange, in the API's own field names."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # `usage.iterations` — one entry per attempt the server made for this turn.
    # A "fallback_message" entry is the only signal that a fallback model served
    # the turn when routing was sticky (those turns carry no `fallback` block).
    iterations: tuple = ()
    # Per-TTL split of the cache writes, when the server reports one (keys along
    # the lines of "ephemeral_5m_input_tokens" / "ephemeral_1h_input_tokens").
    # A 1-hour write is 2x input, not 1.25x.
    cache_creation: "dict | None" = None
    # The model that actually produced the message — `message_start`'s model,
    # or the fallback that rescued the turn. Recorded here so a turn can be
    # priced correctly from the usage alone, including after a reload.
    served_model: "str | None" = None

    @property
    def served_by_fallback(self) -> bool:
        """True when some attempt in this turn was served by a fallback model."""
        return any(isinstance(it, dict) and it.get("type") == "fallback_message"
                   for it in self.iterations)

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_creation_input_tokens + self.cache_read_input_tokens)

    def cost(self, model, *, served=None, when=None, cache_ttl="5m") -> float:
        """USD actually charged for these tokens.

        `model` is the model the turn was *requested* on; `served` is the model
        that finished it (a `fallback`/`done` event's `.model`). When the turn
        carries `iterations` — one entry per attempt the server made — each
        attempt is priced at the rates of the model that ran it, because that is
        how the server bills it: the declining partial at the requested model's
        rates and the rescue at the fallback's. Pricing the cumulative total at
        one model under-reports a rescued turn by around 20%.

        `served` defaults to `served_model`, which the stream records, so
        `usage.cost(config.model)` alone prices a rescued turn correctly.

        `when` (default today) selects introductory versus list pricing, and
        `cache_ttl` says which TTL a cache write used when the server sent no
        per-TTL breakdown ("5m", the API's own default, or "1h" at 2x).
        """
        requested = resolve_model(model)
        if served is not None:
            final = resolve_model(served)
        elif self.served_model:
            final = _resolve_or(self.served_model, requested)
        else:
            final = requested
        entries = [it for it in self.iterations
                   if isinstance(it, dict) and isinstance(it.get("usage"), dict)]
        if not entries:
            return self._price(final, when, cache_ttl)
        total = 0.0
        counted = Usage()
        for entry in entries:
            spec = requested
            if entry.get("model"):
                spec = _resolve_or(entry["model"], requested)
            elif entry.get("type") == "fallback_message":
                spec = final
            part = Usage.from_dict(entry["usage"])
            total += part._price(spec, when, cache_ttl)
            # Output is produced once per attempt and adds up; the prompt is
            # re-read by each attempt but reported once, so the cumulative figure
            # matches the largest attempt rather than their sum.
            counted = Usage(
                max(counted.input_tokens, part.input_tokens),
                counted.output_tokens + part.output_tokens,
                max(counted.cache_creation_input_tokens,
                    part.cache_creation_input_tokens),
                max(counted.cache_read_input_tokens, part.cache_read_input_tokens),
            )
        rest = Usage(
            max(0, self.input_tokens - counted.input_tokens),
            max(0, self.output_tokens - counted.output_tokens),
            max(0, self.cache_creation_input_tokens - counted.cache_creation_input_tokens),
            max(0, self.cache_read_input_tokens - counted.cache_read_input_tokens),
        )
        # Zero for a single turn, where the attempts account for all of it. It is
        # non-zero only for a *summed* usage, where one rescued turn contributes
        # iterations and the ordinary turns contribute none: pricing the entries
        # alone would silently drop every one of those turns from the total.
        if rest.total_tokens:
            total += rest._price(final, when, cache_ttl)
        return total

    def _price(self, spec: ModelSpec, when, cache_ttl) -> float:
        price_in, price_out = spec.prices(when)
        five, hour = self._cache_write_split(cache_ttl)
        return (
            self.input_tokens * price_in
            + self.output_tokens * price_out
            + five * price_in * CACHE_WRITE_MULTIPLIER["5m"]
            + hour * price_in * CACHE_WRITE_MULTIPLIER["1h"]
            + self.cache_read_input_tokens * price_in * CACHE_READ_MULTIPLIER
        ) / 1_000_000.0

    def _cache_write_split(self, default_ttl="5m") -> tuple:
        """(5-minute tokens, 1-hour tokens) written this turn."""
        five = hour = 0
        for key, value in (self.cache_creation or {}).items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            name = str(key).lower()
            if "1h" in name or "60m" in name or "hour" in name:
                hour += count
            elif "5m" in name or "min" in name or "ephemeral" in name:
                five += count
        rest = self.cache_creation_input_tokens - (five + hour)
        if rest > 0:
            if str(default_ttl) == "1h":
                hour += rest
            else:
                five += rest
        return five, hour

    def as_dict(self) -> dict:
        # `iterations` is omitted when empty so the stored shape of an ordinary
        # turn stays exactly what it has always been.
        out = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }
        if self.iterations:
            out["iterations"] = [dict(it) if isinstance(it, dict) else it
                                 for it in self.iterations]
        if self.cache_creation:
            out["cache_creation"] = dict(self.cache_creation)
        if self.served_model:
            out["served_model"] = self.served_model
        return out

    @classmethod
    def from_dict(cls, d) -> "Usage":
        """Tolerant constructor: unknown keys ignored, missing keys zero."""
        d = d or {}
        return cls(
            input_tokens=int(d.get("input_tokens") or 0),
            output_tokens=int(d.get("output_tokens") or 0),
            cache_creation_input_tokens=int(d.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(d.get("cache_read_input_tokens") or 0),
            iterations=tuple(d.get("iterations") or ()),
            cache_creation=(dict(d["cache_creation"])
                            if isinstance(d.get("cache_creation"), dict) else None),
            served_model=d.get("served_model") or None,
        )

    def __add__(self, other) -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        creation = None
        if self.cache_creation or other.cache_creation:
            creation = dict(self.cache_creation or {})
            for key, value in (other.cache_creation or {}).items():
                try:
                    creation[key] = int(creation.get(key, 0)) + int(value)
                except (TypeError, ValueError):
                    creation[key] = value
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            # Keeping `iterations` is what keeps `served_by_fallback` true across
            # a summed conversation, and what lets `cost()` price each attempt.
            tuple(self.iterations) + tuple(other.iterations),
            creation,
            # Summed usage is a token total; the most recent turn's model is kept
            # so the running total still prices at something real, but a turn's
            # own cost should be taken per turn, before the sum.
            other.served_model or self.served_model,
        )

    def __radd__(self, other) -> "Usage":
        # `sum()` starts at 0, and `0 + Usage` must not be a TypeError.
        if not other:
            return self
        return self.__add__(other)


def _merge_usage(prev: Usage, payload) -> Usage:
    """Stream usage is *cumulative*, not incremental: later fields replace earlier."""
    if not payload:
        return prev
    out = replace(prev)
    for name in ("input_tokens", "output_tokens",
                 "cache_creation_input_tokens", "cache_read_input_tokens"):
        v = payload.get(name)
        if v is not None:
            setattr(out, name, int(v))
    iterations = payload.get("iterations")
    if iterations:
        out.iterations = tuple(iterations)
    creation = payload.get("cache_creation")
    if isinstance(creation, dict):
        out.cache_creation = dict(creation)
    return out


# --------------------------------------------------------------------------- events


@dataclass
class StreamEvent:
    """One thing that happened on the wire, already decoded.

    kind: "start" | "text" | "thinking" | "usage" | "fallback" | "ping" | "done" | "error"
    """

    kind: str
    text: str = ""
    usage: "Usage | None" = None
    stop_reason: "str | None" = None
    stop_details: "dict | None" = None
    model: "str | None" = None
    error: "APIError | None" = None


# --------------------------------------------------------------------------- errors


_SECRET_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{4,}")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{4,}")


def redact(text) -> str:
    """Blank out anything shaped like a credential. Applied to every error message."""
    if not text:
        return ""
    return _BEARER_RE.sub("Bearer ***", _SECRET_RE.sub("sk-ant-***", str(text)))


class _Secret(str):
    """A string that refuses to show itself in a repr.

    Header dicts are live locals in half a dozen frames, and any traceback
    formatter that captures locals (rich, better-exceptions, cgitb,
    `TracebackException(capture_locals=True)`) prints them. The value still goes
    on the wire — `str`/`%s`/f-strings are untouched — but nothing that formats a
    container can spill it.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "'***'"


class APIError(Exception):
    """Base class. `.message` is always redacted — a key must never reach a log."""

    def __init__(self, message="", *, status=None, type=None, request_id=None,
                 retryable=False):
        self.message = redact(message)
        self.status = status
        self.type = type
        self.request_id = request_id
        self.retryable = bool(retryable)
        # True when the failure is a bug in this process rather than anything the
        # network did, so a UI can say so instead of blaming the connection.
        self.internal = False
        super().__init__(self.message)

    def __str__(self) -> str:
        bits = [self.message or self.__class__.__name__]
        if self.status:
            bits.append(f"(HTTP {self.status})")
        if self.request_id:
            bits.append(f"[{self.request_id}]")
        return " ".join(bits)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(status={self.status!r}, type={self.type!r}, "
                f"request_id={self.request_id!r}, message={self.message!r})")


class AuthError(APIError):
    """401 / 403 — bad, missing, or unprivileged credentials."""


class BadRequestError(APIError):
    """400 / 404 / 413 — the request itself is wrong; retrying cannot help."""


class RateLimitError(APIError):
    """429. `.retry_after` is the server's requested delay in seconds, if given."""

    def __init__(self, message="", *, retry_after=None, **kw):
        kw.setdefault("retryable", True)
        super().__init__(message, **kw)
        self.retry_after = retry_after


class OverloadedError(APIError):
    """529 — the service is temporarily saturated."""


class ServerError(APIError):
    """5xx other than 529."""


class NetworkError(APIError):
    """DNS / TLS / socket / timeout / truncated stream. Retryable by default."""

    def __init__(self, message="", **kw):
        kw.setdefault("retryable", True)
        super().__init__(message, **kw)


class CancelledError(APIError):
    """The caller's cancel event fired."""


_ERROR_TYPES = {
    "invalid_request_error": (BadRequestError, 400),
    "authentication_error": (AuthError, 401),
    "permission_error": (AuthError, 403),
    "not_found_error": (BadRequestError, 404),
    "request_too_large": (BadRequestError, 413),
    "rate_limit_error": (RateLimitError, 429),
    "api_error": (ServerError, 500),
    "timeout_error": (ServerError, 504),
    "overloaded_error": (OverloadedError, 529),
}


def _class_for_status(status: int):
    if status in (401, 403):
        return AuthError, False
    if status in (400, 404, 413, 422):
        return BadRequestError, False
    if status == 429:
        return RateLimitError, True
    if status == 529:
        return OverloadedError, True
    if status >= 500:
        return ServerError, True
    if status in (408, 409):
        return APIError, True
    return APIError, False


def _header(headers, name: str):
    """Case-insensitive header lookup that works for dicts and email.Message."""
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if get is not None:
        v = get(name)
        if v is None:
            v = get(name.lower())
        if v is None:
            v = get(name.title())
        if v is not None:
            return v
    try:
        for k, v in (headers.items() if hasattr(headers, "items") else headers):
            if str(k).lower() == name.lower():
                return v
    except Exception:
        pass
    return None


def _parse_retry_after(value) -> "float | None":
    """`retry-after` is either seconds or an HTTP date."""
    if value is None:
        return None
    s = str(value).strip()
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(s)
    except Exception:
        return None
    if when is None:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _error_from_payload(err_type, message, *, status=None, request_id=None,
                        retry_after=None) -> APIError:
    cls, default_status = _ERROR_TYPES.get(err_type, (APIError, status or 0))
    status = status or default_status or None
    retryable = cls in (RateLimitError, OverloadedError, ServerError)
    if cls is RateLimitError:
        return RateLimitError(message, status=status, type=err_type,
                              request_id=request_id, retry_after=retry_after)
    return cls(message, status=status, type=err_type, request_id=request_id,
               retryable=retryable)


def _error_from_response(status, reason, headers, body: bytes) -> APIError:
    """Build the right exception from a non-200 HTTP response."""
    request_id = _header(headers, "request-id") or _header(headers, "x-request-id")
    err_type = None
    message = ""
    try:
        doc = json.loads(body.decode("utf-8", "replace")) if body else {}
        err = doc.get("error") if isinstance(doc, dict) else None
        if isinstance(err, dict):
            err_type = err.get("type")
            message = err.get("message") or ""
        request_id = (doc.get("request_id") if isinstance(doc, dict) else None) or request_id
    except Exception:
        message = body[:200].decode("utf-8", "replace") if body else ""
    if not message:
        message = f"{reason or 'HTTP error'}"
    cls, retryable = _class_for_status(int(status))
    if cls is RateLimitError:
        return RateLimitError(message, status=status, type=err_type, request_id=request_id,
                              retry_after=_parse_retry_after(_header(headers, "retry-after")))
    return cls(message, status=status, type=err_type, request_id=request_id,
               retryable=retryable)


# ------------------------------------------------------------------------- api keys


def find_api_key(env=None) -> "str | None":
    """ANTHROPIC_API_KEY, else ANTHROPIC_AUTH_TOKEN, else None.

    The value is returned, never logged, never echoed and never stored elsewhere.
    """
    env = os.environ if env is None else env
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def is_oauth_token(key) -> bool:
    """OAuth access tokens go on `authorization:`, not `x-api-key:`."""
    return bool(key) and str(key).startswith("sk-ant-oat")


# ------------------------------------------------------------------------------ sse


class SSEDecoder:
    """Incremental Server-Sent Events decoder.

    Correctness rests on never assuming a read boundary means anything: bytes are
    decoded through an incremental UTF-8 decoder (a code point may straddle two
    TCP segments), a trailing lone CR is held back (it may be the first half of
    CRLF), and records are only dispatched on a blank line.
    """

    _EOL = re.compile("\r\n|\n|\r")

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buf = ""
        self._event = ""
        self._data: list = []
        self._seen_any = False

    def feed(self, chunk) -> list:
        """Consume bytes (or str); return a list of (event_name, data) records.

        A `str` chunk is re-encoded and pushed through the same incremental
        decoder rather than short-circuiting it: bypassing it would strand any
        half-finished code point from a previous bytes chunk and corrupt the
        character that straddles the two.
        """
        if isinstance(chunk, str):
            text = self._decoder.decode(chunk.encode("utf-8"))
        else:
            text = self._decoder.decode(bytes(chunk))
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def close(self) -> list:
        """Flush. Anything not terminated by a blank line is *discarded*.

        The one thing that *is* released here is a record whose final terminator
        was a lone CR at the very end of the stream: `feed` holds a trailing CR
        back because it may be the first half of a CRLF still in flight, but at
        end of stream nothing more is coming, so per WHATWG it terminates the
        line. Discarding it instead loses the `message_stop` of a complete
        answer and reports a broken stream.

        Per the SSE spec a trailing incomplete line is dropped, and dropping it is
        what makes a truncated stream detectable: a connection cut inside the
        final record terminator leaves `message_stop` undelivered, so the client
        reports a broken stream instead of a complete-looking answer that stops
        mid-sentence.
        """
        try:
            self._decoder.decode(b"", True)
        except Exception:
            pass
        out = self._drain(final=True)
        self._buf = ""
        self._data = []
        self._event = ""
        return out

    def _drain(self, final: bool) -> list:
        out = []
        pos = 0
        buf = self._buf
        while True:
            m = self._EOL.search(buf, pos)
            if m is None:
                break
            if not final and m.group() == "\r" and m.end() == len(buf):
                break  # could be the CR of a CRLF still in flight
            line = buf[pos:m.start()]
            pos = m.end()
            rec = self._line(line)
            if rec is not None:
                out.append(rec)
        self._buf = buf[pos:]
        return out

    def _line(self, line: str):
        if not self._seen_any:
            self._seen_any = True
            if line.startswith("﻿"):
                line = line[1:]
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None  # comment / keep-alive
        field, sep, value = line.partition(":")
        if not sep:
            field, value = line, ""
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        # "id" and "retry" carry no meaning for this API.
        return None

    def _dispatch(self):
        if not self._data:
            self._event = ""
            return None
        payload = "\n".join(self._data)
        name = self._event or "message"
        self._event = ""
        self._data = []
        return (name, payload)


# ------------------------------------------------------------------------ transports


def _stream_socket(conn, resp):
    """The live socket behind a response, wherever http.client left it.

    On a connection the server intends to close (HTTP/1.0, or `connection:
    close`) `getresponse()` has already detached `conn.sock`, and the only
    remaining handle is the one the response's file object reads through.
    """
    for candidate in (getattr(conn, "sock", None),
                      getattr(getattr(resp, "fp", None), "raw", None),
                      getattr(resp, "fp", None)):
        sock = getattr(candidate, "_sock", candidate)
        if sock is not None and hasattr(sock, "shutdown"):
            return sock
    return None


class _HTTPStream:
    """A live response: iterating yields body bytes as they arrive."""

    def __init__(self, conn, resp, owner=None):
        self._conn = conn
        self._resp = resp
        self._owner = owner
        self._sock = _stream_socket(conn, resp)
        self._closed = False
        self.status = resp.status
        self.reason = resp.reason
        self.headers = resp.headers
        # read1() returns as soon as *any* bytes are available; read() would block
        # until the buffer is full and defeat streaming entirely.
        self._read = getattr(resp, "read1", None) or resp.read

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        try:
            chunk = self._read(_READ_SIZE)
        except (OSError, http.client.HTTPException, ValueError, AttributeError) as exc:
            # AttributeError belongs here: when another thread tears the response
            # down under a parked read, http.client finishes the read against a
            # half-detached object (`_close_conn` on a None fp). That is a
            # teardown race, not a bug in the caller, and it must not escape as a
            # non-APIError and kill the app.
            if self._closed:
                raise StopIteration
            self.close()
            raise NetworkError(f"stream read failed: {exc}") from exc
        if not chunk:
            raise StopIteration
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # shutdown() before close(): closing a file descriptor does NOT interrupt a
        # recv() already parked in the C layer, so a cancel that only closed would
        # sit there until the server sent something (up to `timeout`, 600s by
        # default). shutdown() wakes the parked reader immediately.
        sock = self._sock or _stream_socket(self._conn, self._resp)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError, ValueError):
                pass  # already dead, or never connected
        for obj in (self._resp, self._conn):
            try:
                obj.close()
            except Exception:
                pass
        if self._owner is not None:
            self._owner._forget(self)


class _Pending:
    """A connection that is being opened: closeable before any stream exists.

    Cancellation used to begin only once `open()` returned, i.e. once the server
    sent response headers — so Ctrl-C while waiting for the first token, the most
    common cancel there is, did nothing at all for up to `Client.timeout`. This
    handle is registered *before* the blocking calls, so the same watchdog that
    tears down a live stream can tear down a connect, a request write, or a wait
    for headers.
    """

    def __init__(self, conn) -> None:
        self._conn = conn
        self._sock = None
        self._lock = threading.Lock()
        self.closed = False

    def attach(self, sock) -> None:
        with self._lock:
            self._sock = sock
            if self.closed:  # cancelled during the connect: do not leak it
                _shutdown(sock)

    def close(self) -> None:
        with self._lock:
            self.closed = True
            sock = self._sock or getattr(self._conn, "sock", None)
        _shutdown(sock)
        try:
            self._conn.close()
        except Exception:
            pass


def _shutdown(sock) -> None:
    """Wake anything parked on this socket, then let it be closed normally."""
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except (OSError, AttributeError, ValueError):
        pass
    try:
        sock.close()
    except (OSError, AttributeError, ValueError):
        pass


_INPROGRESS = {errno.EINPROGRESS, errno.EALREADY, errno.EWOULDBLOCK,
               getattr(errno, "WSAEWOULDBLOCK", 10035), getattr(errno, "WSAEINVAL", 10022)}


def _connect(host, port, timeout, cancel=None, poll: float = 0.02):
    """Connect, observing `cancel` while the handshake is outstanding.

    A blocking `connect()` cannot be interrupted — there is no socket to shut
    down until it returns — so a blackholed address used to hold a cancelled
    request for the whole `timeout` and then raise a *retryable* NetworkError,
    which got the cancelled request retried. Polling a non-blocking connect keeps
    the cancel latency at `poll` regardless of where the packets went.
    """
    deadline = None if not timeout else time.monotonic() + float(timeout)
    last = None
    for family, socktype, proto, _canon, address in socket.getaddrinfo(
            host, port, 0, socket.SOCK_STREAM):
        if cancel is not None and cancel.is_set():
            raise CancelledError("cancelled while connecting")
        sock = socket.socket(family, socktype, proto)
        try:
            sock.setblocking(False)
            code = sock.connect_ex(address)
            while code not in (0, errno.EISCONN):
                if code not in _INPROGRESS:
                    raise OSError(code, os.strerror(code))
                if cancel is not None and cancel.is_set():
                    raise CancelledError("cancelled while connecting")
                if deadline is not None and time.monotonic() >= deadline:
                    raise socket.timeout("connection timed out")
                wait = poll
                if deadline is not None:
                    wait = max(0.0, min(poll, deadline - time.monotonic()))
                _r, writable, failed = select.select([], [sock], [sock], wait)
                if not writable and not failed:
                    continue
                code = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if code:
                    raise OSError(code, os.strerror(code))
                break
            sock.setblocking(True)
            if timeout:
                sock.settimeout(float(timeout))
            return sock
        except CancelledError:
            _shutdown(sock)
            raise
        except OSError as exc:
            _shutdown(sock)
            last = exc
    raise last or OSError("could not connect")


class HTTPTransport:
    """Default transport: one `http.client` connection per request, TLS for https.

    `open()` takes an optional `cancel` event and honours it from before the
    connect is attempted. A transport that does not accept the argument still
    works — `Client` detects the signature — but cannot be interrupted until it
    returns a stream.
    """

    def __init__(self, *, ssl_context=None) -> None:
        self._ctx = ssl_context
        self._lock = threading.Lock()
        self._live = set()

    def open(self, url, headers, body, timeout, cancel=None):
        """POST `body` to `url`; return an iterable of response bytes."""
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise NetworkError(f"unsupported scheme {parts.scheme!r}")
        if not parts.hostname:
            raise NetworkError("missing host in base_url")
        if cancel is not None and cancel.is_set():
            raise CancelledError("cancelled before the request was made")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if parts.scheme == "https":
            ctx = self._ctx or ssl.create_default_context()
            conn = http.client.HTTPSConnection(parts.hostname, parts.port,
                                               timeout=timeout, context=ctx)
        else:
            ctx = None
            conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query

        pending = _Pending(conn)
        failure = None
        with self._lock:
            self._live.add(pending)
        done = threading.Event()
        watcher = None
        if cancel is not None:
            watcher = threading.Thread(target=_watch_cancel, args=(pending, cancel, done),
                                       name="lume-cancel-open", daemon=True)
            watcher.start()
        try:
            # Connect by hand rather than letting `conn.request()` do it, so the
            # socket exists (and is therefore closeable by the watchdog) from the
            # earliest possible moment.
            sock = _connect(parts.hostname, port, timeout, cancel)
            pending.attach(sock)
            if ctx is not None:
                sock = ctx.wrap_socket(sock, server_hostname=parts.hostname)
                pending.attach(sock)
            conn.sock = sock
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
        except CancelledError:
            pending.close()
            raise
        except (OSError, socket.timeout, http.client.HTTPException) as exc:
            pending.close()
            if cancel is not None and cancel.is_set():
                # The failure is our own teardown wearing a socket error's type.
                # It must not surface as a *retryable* NetworkError, or the
                # cancelled request is promptly tried again.
                failure = CancelledError("cancelled while opening the connection")
            else:
                failure = NetworkError(f"connection failed: {exc}")
        finally:
            done.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            with self._lock:
                self._live.discard(pending)
        if failure is not None:
            # Raised out here, past the handler, so the new error inherits no
            # `__context__`: the frames behind it are http.client's request
            # serialiser, whose locals hold the whole request *including the auth
            # header* as raw bytes, and `raise ... from None` inside the handler
            # clears only `__cause__`.
            _raise_detached(failure)
        stream = _HTTPStream(conn, resp, self)
        with self._lock:
            self._live.add(stream)
        if cancel is not None and cancel.is_set():
            stream.close()
            raise CancelledError("cancelled while opening the connection")
        return stream

    def _forget(self, stream) -> None:
        with self._lock:
            self._live.discard(stream)

    def close(self) -> None:
        """Close every connection this transport still owns."""
        with self._lock:
            live = list(self._live)
            self._live.clear()
        for stream in live:
            stream.close()


class _Closer:
    """Closes whichever of the given objects knows how. Never raises."""

    def __init__(self, *targets) -> None:
        self._targets = targets

    def close(self) -> None:
        for target in self._targets:
            closer = getattr(target, "close", None)
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                pass


class SDKTransport:
    """Optional bridge to the official `anthropic` SDK, if it happens to be installed.

    It sits at a different seam from `HTTPTransport`: instead of `open()` returning
    bytes it exposes `events()` yielding raw stream events already parsed by the
    SDK, which `Client` maps onto `StreamEvent` with the same translator.
    """

    def __init__(self, api_key, base_url=DEFAULT_BASE_URL, *, module=None) -> None:
        self._module = module or self.load()
        if self._module is None:
            raise NetworkError("the anthropic SDK is not importable")
        self._sdk = self._module.Anthropic(api_key=api_key, base_url=base_url,
                                           max_retries=0)

    @staticmethod
    def load():
        """Return the `anthropic` module, or None when it is not installed."""
        import importlib
        try:
            return importlib.import_module("anthropic")
        except Exception:
            return None

    @classmethod
    def available(cls) -> bool:
        return cls.load() is not None

    def events(self, payload, headers, timeout, cancel=None):
        """Yield raw event dicts, in the same shape as the SSE `data:` payloads.

        `cancel` is watched from a side thread that closes the SDK's stream, not
        only between events: a quiet stream (a long think before the first token)
        yields nothing to check the flag in, and checking only per event made a
        silent stream uncancellable.
        """
        body = dict(payload)
        body.pop("stream", None)
        core = {k: body.pop(k) for k in ("model", "messages", "max_tokens", "system")
                if k in body}
        extra_headers = {k: v for k, v in headers.items()
                         if k in ("anthropic-beta",)}
        stream = self._sdk.messages.stream(
            extra_headers=extra_headers or None, extra_body=body or None,
            timeout=timeout, **core)
        with stream as live:
            done = threading.Event()
            watcher = None
            if cancel is not None:
                watcher = threading.Thread(
                    target=_watch_cancel, args=(_Closer(live, stream), cancel, done),
                    name="lume-cancel-sdk", daemon=True)
                watcher.start()
            try:
                for event in live:
                    if cancel is not None and cancel.is_set():
                        raise CancelledError("cancelled")
                    yield _as_dict(event)
                if cancel is not None and cancel.is_set():
                    # The watchdog closed the stream under us; a stream that ends
                    # because we killed it is a cancellation, not a short read.
                    raise CancelledError("cancelled")
            except CancelledError:
                raise
            except Exception:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("cancelled") from None
                raise
            finally:
                done.set()
                if watcher is not None:
                    watcher.join(timeout=1.0)

    def close(self) -> None:
        closer = getattr(self._sdk, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                pass


def _as_dict(event) -> dict:
    """Normalise an SDK event object (pydantic or otherwise) to a plain dict."""
    if isinstance(event, dict):
        return event
    for name in ("to_dict", "model_dump", "dict"):
        fn = getattr(event, name, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    out = dict(getattr(event, "__dict__", {}) or {})
    if "type" not in out and hasattr(event, "type"):
        out["type"] = event.type
    return out


# ------------------------------------------------------------------- request bodies


def _estimate_tokens(value) -> int:
    """Rough character-based estimate; only ever used to decide cache placement.

    Latin script runs about four characters per token, but CJK, Cyrillic, Greek
    and friends are closer to one token per character. A flat len//4 under-counts
    them roughly fourfold, which meant caching never engaged for anyone not
    writing in English — so non-ASCII characters are weighted at 1.
    """
    plain, wide = _text_weight(value)
    return plain // 4 + wide


def _text_weight(value):
    """(ascii character count, non-ascii character count) over nested content."""
    if value is None:
        return (0, 0)
    if isinstance(value, dict):
        parts = [_text_weight(v) for k, v in value.items() if k != "cache_control"]
    elif isinstance(value, (list, tuple)):
        parts = [_text_weight(v) for v in value]
    else:
        text = value if isinstance(value, str) else str(value)
        plain = len(text.encode("ascii", "ignore"))
        return (plain, len(text) - plain)
    return (sum(p[0] for p in parts), sum(p[1] for p in parts))


_CACHEABLE_BLOCKS = ("text", "image", "document", "tool_result", "tool_use", "search_result")


def _blocks(content) -> list:
    """Normalise message/system content to a list of block dicts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [dict(content)]
    return [dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)}
            for b in content]


def _mark_cache(blocks) -> bool:
    """Put one ephemeral breakpoint on the last cacheable block.

    True only when a *new* marker was added. A block the caller already marked is
    not an extra breakpoint of ours, and reporting it as one used to make the
    budget of four look spent in one place while another breakpoint was still
    being added elsewhere.
    """
    for block in reversed(blocks):
        if block.get("type") in _CACHEABLE_BLOCKS:
            if "cache_control" in block:
                return False
            block["cache_control"] = {"type": "ephemeral"}
            return True
    return False


def _count_breakpoints(value) -> int:
    """Count `cache_control` markers anywhere in already-built content."""
    if isinstance(value, dict):
        return ((1 if "cache_control" in value else 0)
                + sum(_count_breakpoints(v) for k, v in value.items()
                      if k != "cache_control"))
    if isinstance(value, (list, tuple)):
        return sum(_count_breakpoints(v) for v in value)
    return 0


# ---------------------------------------------------------------------------- client


class _StreamState:
    """Per-attempt accumulator: model, cumulative usage, block types, stop reason."""

    def __init__(self) -> None:
        self.model = None
        # The model the turn started on. After a mid-output fallback `model` is
        # the rescuer, but the declining attempt still bills at this one.
        self.first_model = None
        self.usage = Usage()
        self.blocks: dict = {}
        self.stop_reason = None
        self.stop_details = None
        self.request_id = None
        self.saw_stop = False
        self.emitted = False


def _label_iterations(entries, requested, served) -> tuple:
    """Name the model behind each attempt, without touching the caller's dicts."""
    out = []
    for entry in entries:
        if isinstance(entry, dict) and not entry.get("model"):
            entry = dict(entry)
            entry["model"] = (served if entry.get("type") == "fallback_message"
                              else requested)
        out.append(entry)
    return tuple(out)


def _translate(state: _StreamState, data: dict) -> list:
    """Map one decoded event payload onto zero or more StreamEvents."""
    kind = data.get("type") if isinstance(data, dict) else None
    if not kind:
        return []
    out = []

    if kind == "message_start":
        msg = data.get("message") or {}
        state.model = msg.get("model") or state.model
        state.first_model = state.first_model or state.model
        state.request_id = msg.get("id") or state.request_id
        state.usage = _merge_usage(state.usage, msg.get("usage"))
        state.usage.served_model = state.model
        out.append(StreamEvent(kind="start", model=state.model, usage=replace(state.usage)))

    elif kind == "content_block_start":
        index = data.get("index", 0)
        block = data.get("content_block") or {}
        btype = block.get("type") or "text"
        state.blocks[index] = btype
        if btype == "fallback":
            # From here on the turn is produced *and billed* by the fallback
            # model, so the run's model changes with it — otherwise every later
            # usage/done event names the model that declined and the turn is
            # priced at the wrong rate (2x out for a fable-5 turn rescued by
            # opus-4-8).
            to = (block.get("to") or {})
            state.model = to.get("model") or state.model
            # The usage carries the model with it, so a turn can be priced from
            # the usage alone — the declining partial at the model that was
            # asked for, the rescue at the fallback's rates.
            state.usage.served_model = state.model
            out.append(StreamEvent(kind="fallback", model=state.model,
                                   stop_details=dict(block)))
        elif btype == "thinking" and block.get("thinking"):
            out.append(StreamEvent(kind="thinking", text=block["thinking"]))
        elif btype == "text" and block.get("text"):
            out.append(StreamEvent(kind="text", text=block["text"]))

    elif kind == "content_block_delta":
        delta = data.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            text = delta.get("text") or ""
            if text:
                out.append(StreamEvent(kind="text", text=text))
        elif dtype == "thinking_delta":
            text = delta.get("thinking") or ""
            if text:
                out.append(StreamEvent(kind="thinking", text=text))
        # signature_delta and input_json_delta carry nothing a chat UI can show.

    elif kind == "content_block_stop":
        state.blocks.pop(data.get("index", 0), None)

    elif kind == "message_delta":
        delta = data.get("delta") or {}
        if "stop_reason" in delta:
            state.stop_reason = delta.get("stop_reason")
        if delta.get("stop_details") is not None:
            state.stop_details = delta.get("stop_details")
        if delta.get("model"):
            state.model = delta["model"]
        state.usage = _merge_usage(state.usage, data.get("usage"))
        state.usage.served_model = state.model or state.usage.served_model
        # Each attempt bills at the rates of the model that ran it. The wire
        # entries name neither model, so they are labelled here, while both are
        # still known, rather than left for the caller to guess later.
        state.usage.iterations = _label_iterations(
            state.usage.iterations, state.first_model or state.model, state.model)
        out.append(StreamEvent(kind="usage", usage=replace(state.usage),
                               stop_reason=state.stop_reason,
                               stop_details=state.stop_details, model=state.model))

    elif kind == "message_stop":
        state.saw_stop = True
        out.append(StreamEvent(kind="done", usage=replace(state.usage),
                               stop_reason=state.stop_reason,
                               stop_details=state.stop_details, model=state.model))

    elif kind == "ping":
        out.append(StreamEvent(kind="ping"))

    elif kind == "error":
        err = data.get("error") or {}
        exc = _error_from_payload(err.get("type"), err.get("message") or "stream error",
                                  request_id=state.request_id)
        out.append(StreamEvent(kind="error", error=exc))

    return out


class Client:
    """Streaming Claude client over a pluggable transport.

    `transport` may be None (plain HTTPS), "auto" (the official SDK when it is
    importable, else HTTPS), "http", or any object exposing `open(url, headers,
    body, timeout)` returning an iterable of bytes with `.status`, `.reason` and
    `.headers` — or `events(payload, headers, timeout, cancel)` yielding decoded
    event dicts.

    A transport's `open()` may also take a fifth argument, `cancel`; when it does
    it is handed the caller's event and is expected to abort a connect or a wait
    for headers itself. One that does not is run on a helper thread instead, so
    the seam keeps working unchanged and a cancel still returns promptly — the
    abandoned call is closed when it eventually finishes rather than leaked.
    """

    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 600.0, max_retries: int = 4, transport=None) -> None:
        if not api_key or not str(api_key).strip():
            raise ValueError("an API key is required")
        self._api_key = _Secret(str(api_key).strip())
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        # Backoff knobs are attributes so tests (and impatient users) can shrink them.
        self.backoff_base = 0.5
        self.backoff_max = 16.0
        self.max_retry_wall = 60.0
        self._sleep = time.sleep

        if transport is None or transport == "http":
            self.transport = HTTPTransport()
        elif transport == "auto":
            module = SDKTransport.load()
            self.transport = (SDKTransport(self._api_key, self.base_url, module=module)
                              if module is not None else HTTPTransport())
        else:
            self.transport = transport

    def __repr__(self) -> str:
        # No key material, not even a prefix.
        return f"<lume.api.Client base_url={self.base_url!r} key=***>"

    __str__ = __repr__

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/messages"

    # ------------------------------------------------------------------ requests

    def build_body(self, *, model=DEFAULT_MODEL, messages, system=None,
                   max_tokens: int = 32000, thinking: bool = True, effort: str = "high",
                   temperature=None, cache: bool = True) -> dict:
        """Build the exact JSON body for a streaming turn. No secrets in the result.

        `thinking=False` is per-model: an explicit `{"type": "disabled"}` where the
        model thinks by default, nothing at all where absence means off, and
        nothing on fable-5, which cannot be told to stop.
        """
        spec = resolve_model(model)
        if not messages:
            raise ValueError("messages must not be empty")
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        # `xhigh` exists only from opus-4-7 on; asking for it on a 4.6-family
        # model is a hard 400, so it is clamped to the best level that model has.
        effort = spec.clamp_effort(effort)
        max_tokens = max(1, min(int(max_tokens), spec.max_output))

        prepared = []
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg:
                raise ValueError("each message needs a 'role' and 'content'")
            prepared.append({"role": msg["role"], "content": msg.get("content", "")})

        body: dict = {
            "model": spec.id,
            "messages": prepared,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if system is not None and system != "":
            body["system"] = system if isinstance(system, str) else _blocks(system)

        if thinking:
            if spec.thinking == "adaptive":
                # budget_tokens is a 400 on these models; adaptive replaces it.
                body["thinking"] = {"type": "adaptive"}
                if spec.supports_thinking_display:
                    # `display` arrived with opus-4-7. The 4.6 family does not
                    # know the field and already summarises by default.
                    body["thinking"]["display"] = "summarized"
            elif spec.thinking == "budget":
                budget = max(1024, min(max_tokens - 1, max_tokens // 2))
                if budget >= 1024 and budget < max_tokens:
                    body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif spec.thinking_off == "disabled":
            # Leaving `thinking` out does not turn it off here: the model thinks
            # by default, bills for it, and eats the max_tokens budget the user
            # meant for the answer. Say "off" explicitly.
            body["thinking"] = {"type": "disabled"}
            cap = spec.disabled_thinking_effort_cap
            if cap and effort and EFFORTS.index(effort) > EFFORTS.index(cap):
                effort = cap  # disabled thinking above `high` is a 400
        # "omit": absence really is off. "unsupported" (fable-5): thinking is
        # always on and an explicit disable is rejected, so send nothing.

        if spec.supports_effort and effort:
            body["output_config"] = {"effort": effort}

        # Sampling parameters are rejected outright by the current models.
        if temperature is not None and spec.supports_temperature:
            body["temperature"] = float(temperature)

        if spec.supports_fallbacks:
            body["fallbacks"] = "default"

        # More than four `cache_control` markers is a hard 400. Ours are budgeted
        # in `_place_cache`, but a caller who supplied five of their own would
        # otherwise have the rejection handed to them by the server, one round
        # trip and one confusing error later.
        supplied = (_count_breakpoints(body.get("system"))
                    + _count_breakpoints(body.get("messages")))
        if supplied > MAX_CACHE_BREAKPOINTS:
            raise ValueError(
                f"{supplied} cache_control breakpoints in this request; the API "
                f"allows at most {MAX_CACHE_BREAKPOINTS}")

        if cache:
            self._place_cache(body, spec)
        return body

    def _place_cache(self, body: dict, spec: ModelSpec) -> None:
        """Breakpoints on the stable prefix only, and only when it can actually cache.

        The minimum prefix is per-model — 512 tokens on opus-5/fable-5 but 4096 on
        opus-4-6/haiku-4-5 — so the spec has to come in with the body.
        """
        minimum = spec.cache_min
        # Markers the caller placed spend the same budget of four; counting only
        # our own placements is how a request reaches five and gets a hard 400.
        breakpoints = (_count_breakpoints(body.get("system"))
                       + _count_breakpoints(body.get("messages")))
        if breakpoints >= MAX_CACHE_BREAKPOINTS:
            return

        prefix = 0
        system = body.get("system")
        if system is not None:
            prefix += _estimate_tokens(system)
            if prefix >= minimum:
                blocks = _blocks(system)
                if _mark_cache(blocks):
                    body["system"] = blocks
                    breakpoints += 1

        messages = body["messages"]
        # The last message is the volatile one; the breakpoint goes before it.
        for msg in messages[:-1]:
            prefix += _estimate_tokens(msg.get("content"))
        if len(messages) >= 2 and prefix >= minimum and breakpoints < MAX_CACHE_BREAKPOINTS:
            blocks = _blocks(messages[-2].get("content"))
            if _mark_cache(blocks):
                messages[-2] = dict(messages[-2], content=blocks)
                breakpoints += 1

    def _build_headers(self, *, betas=()) -> dict:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "anthropic-version": API_VERSION,
        }
        betas = list(betas)
        # The credential goes on the wire as an ordinary string but reprs as
        # '***', so a traceback that captures locals cannot print it.
        if is_oauth_token(self._api_key):
            headers["authorization"] = _Secret(f"Bearer {self._api_key}")
            if OAUTH_BETA not in betas:
                betas.insert(0, OAUTH_BETA)
        else:
            headers["x-api-key"] = _Secret(self._api_key)
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    # ------------------------------------------------------------------- streaming

    def stream(self, *, model: str = DEFAULT_MODEL, messages, system=None,
               max_tokens: int = 32000, thinking: bool = True, effort: str = "high",
               temperature=None, cache: bool = True, cancel=None):
        """Stream one assistant turn as StreamEvents.

        Argument errors raise immediately; transport errors raise from the
        generator. `cancel` is a `threading.Event`: setting it aborts the
        request, closes the socket and raises `CancelledError` — including while
        the connect is still outstanding and while the server is thinking with
        nothing yet on the wire, which is where most cancels actually land. A
        cancel never surfaces as a retryable error, so a cancelled turn is never
        quietly sent again.
        """
        spec = resolve_model(model)
        body = self.build_body(model=spec, messages=messages, system=system,
                               max_tokens=max_tokens, thinking=thinking, effort=effort,
                               temperature=temperature, cache=cache)
        betas = [FALLBACK_BETA] if body.get("fallbacks") else []
        headers = self._build_headers(betas=betas)
        return self._run(body, headers, cancel)

    def _run(self, body: dict, headers: dict, cancel):
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        started = time.monotonic()
        planned = 0.0
        attempt = 0
        while True:
            state = _StreamState()
            try:
                yield from self._attempt(body, raw, headers, cancel, state)
                return
            except (CancelledError, GeneratorExit):
                raise
            except _INTERNAL_BUGS:
                # A TypeError from lume's own translator is a bug, not a network
                # failure: laundering it into a NetworkError hides it, retries it
                # four times, and tells the user their connection is flaky.
                raise
            except Exception as exc:  # noqa: BLE001 - normalised on the next line
                # Everything the transport can throw funnels through here so it
                # arrives as an APIError with a scrubbed message; a third-party
                # transport raising a plain RuntimeError used to reach the caller
                # verbatim, credential and all.
                failure = self._scrub(_transport_error(exc))
            # Deliberately outside the handler. Raising in it would re-attach the
            # original exception as `__context__` — whose message can hold the raw
            # key (a third-party transport formatting headers with %s) and whose
            # frames' locals certainly do. `raise ... from None` clears only
            # `__cause__`; nothing clears an implicit `__context__` but not
            # raising inside the handler at all.
            #
            # A retry may only happen while the caller has seen nothing at all,
            # otherwise the reply would be duplicated on screen.
            if state.emitted or not failure.retryable or attempt >= self.max_retries:
                _raise_detached(failure)
            delay = self._delay(attempt, getattr(failure, "retry_after", None))
            # Count both real elapsed time and time we have promised to wait.
            used = max(time.monotonic() - started, planned)
            if used + delay > self.max_retry_wall:
                _raise_detached(self._out_of_budget(failure, delay, used))
            self._pause(delay, cancel)
            planned += delay
            attempt += 1

    def _delay(self, attempt: int, retry_after) -> float:
        if retry_after is not None:
            # Not clamped to the wall-clock budget: an hour-long retry-after has
            # to read as "longer than we will wait" so the caller is told why we
            # stopped, rather than being silently trimmed to the cap.
            return max(0.0, float(retry_after))
        window = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        # Full jitter over the window: synchronised clients must not resonate.
        return random.uniform(0.0, window)

    def _out_of_budget(self, exc, delay: float, used: float):
        """Annotate an error we are giving up on because the wait is too long."""
        note = (f"not retried: the next attempt would wait {delay:.0f}s, past the "
                f"{self.max_retry_wall:.0f}s retry budget")
        exc.message = redact(f"{exc.message} ({note})" if exc.message else note)
        exc.args = (exc.message,)
        return exc

    def _pause(self, delay: float, cancel) -> None:
        if cancel is not None:
            if cancel.wait(delay):
                raise CancelledError("cancelled while backing off")
            return
        if delay > 0:
            self._sleep(delay)

    def _open(self, raw, headers, cancel):
        """Open the response, with the cancel event observed *while* it blocks.

        The transport does the waiting — for the TCP handshake, for the request
        to go out, and for the server to send response headers. Until this
        returns there is no stream to close, so a cancel raised here or nowhere.
        """
        opener = self.transport.open
        if cancel is None:
            return opener(self.url, headers, raw, self.timeout)
        if _accepts_cancel(opener):
            return opener(self.url, headers, raw, self.timeout, cancel=cancel)
        return self._open_supervised(opener, raw, headers, cancel)

    def _open_supervised(self, opener, raw, headers, cancel):
        """Run a cancel-unaware transport's `open()` where a cancel can escape it.

        Third-party transports predate the `cancel` argument and may block for as
        long as they like. Running one on a helper thread keeps the seam working
        unchanged while still bounding cancel latency; whatever the abandoned
        call eventually produces is closed rather than leaked.
        """
        box: dict = {}

        def run():
            try:
                box["stream"] = opener(self.url, headers, raw, self.timeout)
            except BaseException as exc:  # noqa: BLE001 - re-raised on this thread
                box["error"] = exc

        worker = threading.Thread(target=run, name="lume-open", daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(0.02)
            if worker.is_alive() and cancel.is_set():
                threading.Thread(target=_reap, args=(worker, box),
                                 name="lume-open-reap", daemon=True).start()
                raise CancelledError("cancelled while opening the connection")
        if "error" in box:
            raise box["error"]
        stream = box.get("stream")
        if cancel.is_set():
            _close_quietly(stream)
            raise CancelledError("cancelled while opening the connection")
        return stream

    def _attempt(self, body, raw, headers, cancel, state):
        if cancel is not None and cancel.is_set():
            raise CancelledError("cancelled before the request was made")

        events = getattr(self.transport, "events", None)
        if events is not None:
            yield from self._attempt_sdk(events, body, headers, cancel, state)
            return

        stream = self._open(raw, headers, cancel)
        watchdog_done = threading.Event()
        watcher = None
        if cancel is not None:
            watcher = threading.Thread(target=_watch_cancel,
                                       args=(stream, cancel, watchdog_done),
                                       name="lume-cancel", daemon=True)
            watcher.start()
        try:
            status = int(getattr(stream, "status", 200) or 200)
            hdrs = getattr(stream, "headers", None)
            state.request_id = (_header(hdrs, "request-id")
                                or _header(hdrs, "x-request-id") or state.request_id)
            if status != 200:
                raise _error_from_response(status, getattr(stream, "reason", ""),
                                           hdrs, _drain_body(stream))
            # A 200 that is not an event stream is a proxy, a captive portal or a
            # gateway answering for the API. Parsing it as SSE yields nothing and
            # used to end as "stream ended before message_stop" with the one
            # useful thing — the body — thrown away.
            ctype = _header(hdrs, "content-type")
            if ctype and "event-stream" not in str(ctype).lower():
                raise _not_sse_error(ctype, _drain_body(stream), state.request_id)

            decoder = SSEDecoder()
            for chunk in stream:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("cancelled mid-stream")
                for name, payload in decoder.feed(chunk):
                    yield from self._emit(name, payload, state)
                if state.saw_stop:
                    break
            if not state.saw_stop:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("cancelled mid-stream")
                # close() releases only a record whose last terminator was a lone
                # CR at end of stream — complete, just held back in case a CRLF
                # was still in flight. A record the connection cut in half is
                # dropped, not delivered, which is why the error below is
                # reachable and a truncated answer never passes for a finish.
                for name, payload in decoder.close():
                    yield from self._emit(name, payload, state)
            if not state.saw_stop:
                raise NetworkError("stream ended before message_stop",
                                   request_id=state.request_id,
                                   retryable=not state.emitted)
        except Exception:  # noqa: BLE001 - re-raised unless we caused it
            # Once the cancel event is set the connection is being torn down by
            # us, so whatever the read raises on the way out (a reset, a
            # half-detached response object, a truncated record) is cancellation
            # wearing someone else's exception type.
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

    def _attempt_sdk(self, events, body, headers, cancel, state):
        failure = None
        try:
            source = events(body, headers, self.timeout, cancel)
            for data in source:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("cancelled mid-stream")
                for event in _translate(state, _as_dict(data)):
                    yield from self._deliver(event, state)
        except (APIError, GeneratorExit):
            raise
        except _INTERNAL_BUGS:
            raise  # our own bug: it must not read as a network failure
        except Exception as exc:  # SDK exceptions are not ours; normalise them.
            failure = self._scrub(_sdk_error(exc))
        if failure is not None:
            # Outside the handler: an SDK exception's own chain runs back through
            # its request builder, whose locals hold the key.
            _raise_detached(failure)
        if not state.saw_stop:
            if cancel is not None and cancel.is_set():
                raise CancelledError("cancelled mid-stream")
            raise NetworkError("stream ended before message_stop",
                               retryable=not state.emitted)

    def _emit(self, name, payload, state):
        try:
            data = json.loads(payload)
        except ValueError:
            return  # a keep-alive or a truncated tail: nothing decodable
        if not isinstance(data, dict):
            return
        data.setdefault("type", name)
        for event in _translate(state, data):
            yield from self._deliver(event, state)

    def _scrub(self, exc):
        """Last line of defence: redact a token that is not `sk-ant-*` shaped."""
        if exc is not None and exc.message and self._api_key in exc.message:
            exc.message = exc.message.replace(self._api_key, "***")
            exc.args = (exc.message,)
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


def _detach(exc):
    """Cut an exception loose from whatever it was raised inside.

    `raise ... from None` clears `__cause__` and sets the suppress flag, but
    `__context__` still points at the original exception — and formatters that
    capture locals (rich, better-exceptions, `TracebackException(capture_locals
    =True)`) walk it anyway. The frames behind it hold the request headers.
    """
    if isinstance(exc, BaseException):
        exc.__cause__ = None
        exc.__context__ = None
        exc.__suppress_context__ = True
    return exc


def _raise_detached(exc):
    """Raise `exc` chained to nothing at all — not even the caller's own frame.

    Detaching before the `raise` is not enough: the raise itself re-attaches
    whatever exception is *currently being handled*, and this generator is
    resumed from inside the app's `except` blocks. So the chain is cut once more
    on the way out, where nothing can put it back.
    """
    try:
        raise _detach(exc)
    finally:
        _detach(exc)


def _accepts_cancel(fn) -> bool:
    """True when `fn` has a `cancel` parameter we can pass the event to."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    param = params.get("cancel")
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _close_quietly(obj) -> None:
    closer = getattr(obj, "close", None)
    if closer is None:
        return
    try:
        closer()
    except Exception:
        pass


def _reap(worker, box) -> None:
    """Close whatever an abandoned `open()` eventually returns."""
    worker.join()
    _close_quietly(box.get("stream"))


def _snippet(body: bytes, limit: int = 200) -> str:
    text = bytes(body or b"").decode("utf-8", "replace")
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _not_sse_error(content_type, body: bytes, request_id) -> APIError:
    """A 200 whose body is not an event stream — usually something in the middle."""
    detail = _snippet(body)
    message = f"expected an event stream, got {str(content_type).strip()!r}"
    if detail:
        message += f": {detail}"
    # Not retryable: a proxy answering for the API answers the same way again,
    # and four more round trips only delay the explanation.
    return NetworkError(message, status=200, request_id=request_id, retryable=False)


def _drain_body(stream, limit: int = _MAX_ERROR_BODY) -> bytes:
    out = bytearray()
    try:
        for chunk in stream:
            out += chunk
            if len(out) >= limit:
                break
    except Exception:
        pass
    return bytes(out)


def _watch_cancel(stream, cancel, done) -> None:
    """Close the socket from the side, so a blocked read returns at once."""
    while not done.is_set():
        if cancel.wait(0.02):
            try:
                stream.close()
            except Exception:
                pass
            return


_RETRYABLE_EXC = (OSError, http.client.HTTPException, ssl.SSLError, TimeoutError)
# Programming errors. These are never network failures, never retried, and never
# dressed up as one: they reach the caller as themselves so the traceback points
# at the line that is actually wrong.
_INTERNAL_BUGS = (TypeError, AttributeError, NameError)


def _transport_error(exc) -> APIError:
    """Normalise anything a transport raised into an APIError.

    Only genuinely network-shaped failures come back retryable. A programming
    error is tagged `internal` and left non-retryable so the UI can say "internal
    error" rather than blaming the network; `Client._run` re-raises those
    unchanged before they ever get here.
    """
    if isinstance(exc, APIError):
        return exc
    error = NetworkError(f"{exc.__class__.__name__}: {exc}",
                         retryable=isinstance(exc, _RETRYABLE_EXC))
    error.internal = isinstance(exc, _INTERNAL_BUGS)
    return error


def _sdk_error(exc) -> APIError:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    request_id = getattr(exc, "request_id", None)
    if status:
        cls, retryable = _class_for_status(int(status))
        if cls is RateLimitError:
            return RateLimitError(str(exc), status=int(status), request_id=request_id)
        return cls(str(exc), status=int(status), request_id=request_id, retryable=retryable)
    return NetworkError(str(exc) or exc.__class__.__name__, retryable=True,
                        request_id=request_id)
