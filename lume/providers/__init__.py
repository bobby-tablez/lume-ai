"""Model providers: Anthropic, plus the other foundation model APIs.

One registry, one seam. Every provider hands back the same ``StreamEvent``
vocabulary that :mod:`lume.api` defines, so the renderer, the session store and
the cost maths never learn which company answered — a provider is a translation
layer and nothing more.

Adding one means: a ``ModelSpec`` per model, a client exposing ``stream()`` and
``close()``, and a line in :data:`PROVIDERS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..api import (DEFAULT_MODEL, MODELS as ANTHROPIC_MODELS, Client as AnthropicClient,
                   ModelSpec, resolve_model as _resolve_anthropic)

__all__ = [
    "Provider", "PROVIDERS", "providers", "provider_names", "all_models",
    "model_names", "resolve", "provider_for", "find_key", "make_client",
    "missing_key_hint", "available", "LOAD_ERRORS",
]


@dataclass(frozen=True)
class Provider:
    """One vendor's API: where it lives, what it can run, how to reach it."""

    name: str
    label: str
    env_keys: tuple
    base_url: str
    models: dict
    factory: object                      # (key, **kw) -> client with .stream/.close
    aliases: dict = field(default_factory=dict)
    doc_url: str = ""

    def find_key(self, env=None):
        """First of `env_keys` that is set, stripped. Never logged or stored."""
        env = os.environ if env is None else env
        for name in self.env_keys:
            value = env.get(name)
            if value and value.strip():
                return value.strip()
        return None


#: Provider modules that failed to import, name -> exception. A broken optional
#: provider must not stop lume from starting, but it must not vanish silently
#: either — `lume --config` and `/models` can say what went wrong.
LOAD_ERRORS = {}


def _load(module: str, attr: str, default=None):
    """Import a provider lazily: a broken optional provider must not break lume."""
    try:
        mod = __import__(f"{__name__}.{module}", fromlist=[attr])
        return getattr(mod, attr)
    except Exception as exc:             # pragma: no cover - a provider module is optional
        LOAD_ERRORS[module] = exc
        return default


def _anthropic() -> Provider:
    return Provider(
        name="anthropic",
        label="Anthropic",
        env_keys=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        base_url="https://api.anthropic.com",
        models=dict(ANTHROPIC_MODELS),
        factory=AnthropicClient,
        aliases={"opus": "claude-opus-5", "sonnet": "claude-sonnet-5",
                 "haiku": "claude-haiku-4-5", "fable": "claude-fable-5",
                 "claude": "claude-opus-5"},
        doc_url="https://console.anthropic.com/settings/keys",
    )


def _openai() -> "Provider | None":
    build = _load("openai", "provider")
    return build() if build else None


def _google() -> "Provider | None":
    build = _load("gemini", "provider")
    return build() if build else None


#: Registry, in the order they are offered. Anthropic first: it is the one lume
#: was built around, and the only one whose thinking blocks stream natively.
PROVIDERS = tuple(p for p in (_anthropic(), _openai(), _google()) if p is not None)
_BY_NAME = {p.name: p for p in PROVIDERS}


def providers() -> tuple:
    return PROVIDERS


def provider_names() -> list:
    return [p.name for p in PROVIDERS]


def all_models() -> dict:
    """Every model from every provider, provider order preserved."""
    out = {}
    for p in PROVIDERS:
        for mid, spec in p.models.items():
            out.setdefault(mid, spec)
    return out


def model_names() -> list:
    return list(all_models())


def resolve(name: str = None) -> tuple:
    """``'gpt-5'`` / ``'openai:gpt-5'`` / ``'sonnet'`` -> ``(Provider, ModelSpec)``.

    Anthropic keeps its own resolver (it knows about dated ids and families);
    everyone else matches on id, then alias, then a unique prefix. A
    ``provider:model`` prefix pins the search to one vendor.
    """
    name = (name or DEFAULT_MODEL).strip()
    scope = None
    if ":" in name:
        head, _, tail = name.partition(":")
        if head.lower() in _BY_NAME:
            scope, name = _BY_NAME[head.lower()], tail.strip()
    key = name.lower()

    for p in ([scope] if scope else PROVIDERS):
        if p.name == "anthropic" and not scope:
            try:
                return p, _resolve_anthropic(name)
            except Exception:
                continue
        if key in p.models:
            return p, p.models[key]
        if key in p.aliases and p.aliases[key] in p.models:
            return p, p.models[p.aliases[key]]
    if scope is not None and scope.name == "anthropic":
        return scope, _resolve_anthropic(name)

    for p in ([scope] if scope else PROVIDERS):
        hits = [m for m in p.models if m.startswith(key)]
        if len(hits) == 1:
            return p, p.models[hits[0]]
    raise ValueError(f"unknown model: {name!r}")


def provider_for(model: str) -> Provider:
    return resolve(model)[0]


def find_key(provider, env=None):
    """The API key for a provider (name or Provider), or None."""
    p = _BY_NAME.get(provider) if isinstance(provider, str) else provider
    return p.find_key(env) if p is not None else None


def available(env=None) -> list:
    """Providers that have a key set — what the user can actually talk to."""
    return [p for p in PROVIDERS if p.find_key(env)]


def missing_key_hint(provider) -> str:
    """One line telling the user which variable to set, and where to get it."""
    p = _BY_NAME.get(provider) if isinstance(provider, str) else provider
    if p is None:
        return "No such provider."
    # ASCII only: this line has to survive a terminal with no unicode.
    where = f", from {p.doc_url}" if p.doc_url else ""
    return f"No API key for {p.label}. Set {p.env_keys[0]}{where}."


def make_client(model: str = None, env=None, **kw):
    """Build the client that serves `model`. Raises ValueError without a key."""
    p, spec = resolve(model)
    key = p.find_key(env)
    if not key:
        raise ValueError(missing_key_hint(p))
    if p.base_url and "base_url" not in kw:
        kw["base_url"] = p.base_url
    return p.factory(key, **kw), spec
