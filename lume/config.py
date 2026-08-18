"""User configuration: defaults, JSON file, environment, and CLI overrides.

Resolution order, lowest priority first: dataclass defaults -> config file ->
environment -> command-line flags. Unknown keys in the file are preserved as
warnings rather than errors, so a config written by a newer version still loads.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

__all__ = ["Config", "config_path", "data_home", "load_config", "save_config"]

EFFORTS = ("low", "medium", "high", "xhigh", "max")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def config_path(env=None) -> Path:
    """Where config.json lives: $LUME_CONFIG > $XDG_CONFIG_HOME/lume > ~/.config/lume."""
    env = os.environ if env is None else env
    explicit = env.get("LUME_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    # LUME_HOME moves everything lume owns, config included: a user who points it
    # at a USB stick means all of it.
    home = env.get("LUME_HOME")
    if home:
        return Path(home).expanduser() / "config.json"
    xdg = env.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else _home() / ".config"
    return base / "lume" / "config.json"


def data_home(env=None) -> Path:
    """Where sessions and history live: $LUME_HOME > $XDG_DATA_HOME/lume > ~/.local/share/lume."""
    env = os.environ if env is None else env
    explicit = env.get("LUME_HOME")
    if explicit:
        return Path(explicit).expanduser()
    xdg = env.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "lume"
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / "lume"
    return _home() / ".local" / "share" / "lume"


@dataclass
class Config:
    """Every knob the app exposes. Values here are already validated."""

    model: str = "claude-opus-5"
    theme: str = "auto"        # "auto" = pick from the terminal background
    system: str = ""
    max_tokens: int = 32000
    effort: str = "high"
    thinking: bool = True
    show_thinking: bool = True
    cache: bool = True
    animation: bool = True
    show_cost: bool = True
    max_width: int = 100          # hard cap on text column width, 0 = use full terminal
    history_size: int = 2000
    timeout: float = 600.0
    max_retries: int = 4
    unknown: dict = field(default_factory=dict, repr=False, compare=False)
    warnings: list = field(default_factory=list, repr=False, compare=False)

    # ------------------------------------------------------------------ validate
    def validate(self) -> "Config":
        """Clamp and normalise in place; append a note for anything corrected."""
        if not isinstance(self.model, str) or not self.model.strip():
            self.warnings.append("model was empty; using the default")
            self.model = Config.model
        self.model = self.model.strip()
        # Store the resolved id, not an alias: aliases are a convenience for the
        # command line, and a saved config should not depend on one surviving.
        from .providers import resolve
        try:
            self.model = resolve(self.model)[1].id
        except ValueError:
            self.warnings.append(f"unknown model {self.model!r}; using the default")
            self.model = Config.model

        from .theme import THEMES  # imported late to keep this module dependency-free
        self.theme = str(self.theme).strip().lower()
        if self.theme and self.theme not in THEMES and self.theme != "auto":
            self.warnings.append(f"unknown theme {self.theme!r}; choosing automatically")
            self.theme = "auto"
        if not self.theme:
            self.theme = "auto"

        if self.effort not in EFFORTS:
            self.warnings.append(f"unknown effort {self.effort!r}; using 'high'")
            self.effort = "high"

        self.max_tokens = _clamp_int(self.max_tokens, 256, 128000, 32000, "max_tokens", self.warnings)
        self.history_size = _clamp_int(self.history_size, 0, 100000, 2000, "history_size", self.warnings)
        self.max_retries = _clamp_int(self.max_retries, 0, 10, 4, "max_retries", self.warnings)
        self.max_width = _clamp_int(self.max_width, 0, 400, 100, "max_width", self.warnings)
        try:
            self.timeout = float(self.timeout)
        except (TypeError, ValueError):
            self.warnings.append("timeout was not a number; using 600")
            self.timeout = 600.0
        self.timeout = min(max(self.timeout, 5.0), 3600.0)

        for name in ("thinking", "show_thinking", "cache", "animation", "show_cost"):
            value = getattr(self, name)
            if isinstance(value, str):
                # `"thinking": "no"` in a config file means no, not "non-empty".
                coerced = _coerce(value, bool)
                if coerced is None:
                    self.warnings.append(f"{name} was not a boolean; using the default")
                    coerced = getattr(Config, name)
                value = coerced
            setattr(self, name, bool(value))
        if not isinstance(self.system, str):
            self.system = ""
        seen, unique = set(), []
        for warning in self.warnings:
            if warning not in seen:
                seen.add(warning)
                unique.append(warning)
        self.warnings[:] = unique[-20:]
        return self

    # ---------------------------------------------------------------- overrides
    def apply_env(self, env=None) -> "Config":
        env = os.environ if env is None else env
        mapping = {
            "LUME_MODEL": "model", "LUME_THEME": "theme", "LUME_EFFORT": "effort",
            "LUME_SYSTEM": "system", "LUME_MAX_TOKENS": "max_tokens",
        }
        for key, attr in mapping.items():
            raw = env.get(key)
            if raw is None or raw == "":
                continue
            current = getattr(self, attr)
            value = _coerce(raw, type(current))
            if value is not None:
                setattr(self, attr, value)
        # Same predicate as ansi.detect_caps, so LUME_NO_MOTION=0 does not
        # mysteriously mean the opposite of LUME_NO_MOTION=1.
        from .ansi import _env_flag
        if _env_flag(env, "LUME_NO_MOTION"):
            self.animation = False
        return self.validate()

    def apply_overrides(self, **kw) -> "Config":
        """Apply non-None values (used for command-line flags)."""
        known = {f.name for f in fields(self)}
        for key, value in kw.items():
            if value is None or key not in known or key in ("unknown", "warnings"):
                continue
            setattr(self, key, value)
        return self.validate()

    # ------------------------------------------------------------------- to disk
    def to_dict(self) -> dict:
        d = asdict(self)
        extra = d.pop("unknown", {}) or {}
        d.pop("warnings", None)
        out = dict(extra)
        out.update(d)
        return out

    def width_for(self, terminal_width: int) -> int:
        """Text column width: the terminal, capped by max_width, with a sane floor."""
        w = max(20, terminal_width)
        if self.max_width:
            w = min(w, self.max_width)
        return w


def _clamp_int(value, low, high, default, name, warnings):
    try:
        v = int(value)
    except (TypeError, ValueError):
        warnings.append(f"{name} was not an integer; using {default}")
        return default
    if v < low or v > high:
        warnings.append(f"{name} out of range; clamped to [{low}, {high}]")
    return min(max(v, low), high)


def _coerce(raw: str, kind):
    if kind is bool:
        s = raw.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        return None                      # unparseable: leave the current value alone
    if kind is int:
        try:
            return int(raw)
        except ValueError:
            return raw
    if kind is float:
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def load_config(path: Path = None, env=None) -> Config:
    """Load config, never raising: a broken file degrades to defaults plus a warning."""
    path = config_path(env) if path is None else Path(path)
    cfg = Config()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config root must be a JSON object")
        except Exception as exc:  # malformed config must never block startup
            cfg.warnings.append(f"could not read {path}: {exc}")
            raw = {}
        known = {f.name for f in fields(Config)} - {"unknown", "warnings"}
        for key, value in raw.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                cfg.unknown[key] = value
                cfg.warnings.append(f"unknown config key {key!r} (kept, not used)")
    cfg.validate()
    cfg.apply_env(env)
    return cfg


def save_config(cfg: Config, path: Path = None, env=None) -> Path:
    """Write config atomically with private permissions (it can hold a system prompt)."""
    path = config_path(env) if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)      # only the leaf: parents are the user's
    data = json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        if hasattr(os, "fchmod"):         # POSIX only; mkstemp is already
            os.fchmod(fd, 0o600)          # owner-private on Windows
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def _fsync_dir(directory: Path) -> None:
    """Make the rename itself durable, not just the file contents."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
