# lume — architecture and interface contract

A dependency-free terminal chat client for the Claude API. Python 3.10+, **stdlib only**.
Tests run with `python3 -m unittest discover -s tests` (currently ~1200 tests, ~2.5 min).

This document is the contract between the modules. It described work to be done while the
project was being built; it now describes what was built. Where the shipped code and this
document disagree, the code is right and this document is a bug.

## Layering

Each module may use those above it and must not reach into a module below it:

| Layer | Module | Responsibility |
|---|---|---|
| 1 | `ansi.py` | terminal capabilities, colour quantisation, display width, ANSI-aware wrap/truncate/pad, `Console`, `sanitize_text`/`sanitize_url`, the exit/signal net |
| 1 | `theme.py` | named palettes; token → `Style`; contrast enforced at build time |
| 2 | `markdown.py` | incremental Markdown renderer and syntax highlighter |
| 2 | `motion.py` | the status animation, the wordmark, `rule` |
| 2 | `store.py` | append-only local session persistence |
| 2 | `api.py` | streaming Claude client, retries, cost accounting |
| 3 | `commands.py` | the command vocabulary: declare, parse, complete, render help |
| 3 | `input.py` | the prompt: paste framing, multi-line rules, history |
| 3 | `config.py` | defaults → file → environment → command line |
| 4 | `app.py` | the chat loop and every command handler |
| 5 | `cli.py` | argument parsing, then hand over to `App` |

Rules that hold across the whole tree, and that the tests enforce:

- **No third-party imports anywhere**, including in tests. No pytest — `unittest` only.
- **Untrusted text is sanitised before it reaches the terminal.** Model output arrives
  through `markdown.py` and `app.py`; both route it through `ansi.sanitize_text`, and every
  displayed URL through `ansi.sanitize_url`. Everything the renderer emits is added after.
- **Nothing hard-codes a colour outside `theme.py`**; render code asks for a token.
- **Nothing hard-codes a width**; `ansi.display_width` is the only measure of a column, and
  no rendered line may exceed the width it was given.
- **The only network egress is `api.py`**, and the only URL is the Anthropic endpoint.
- **Transcripts never leave the machine**: `0600` files in a `0700` directory.
- A command declared in `commands.py` must have a handler in `app.py`; a test asserts it.

## Commands

The shipped vocabulary is 21 commands: `help`, `new`, `resume`, `list`, `rename`, `delete`,
`export`, `system`, `retry`, `undo`, `edit`, `copy`, `clear`, `usage` (aliases `tokens`,
`cost`), `model`, `models`, `think`, `effort`, `theme`, `keys`, `quit`.

**`stream` was specified and then dropped:** replies always stream, so a toggle would have
described a feature the app does not have.

## The foundation API the other modules build on

```python
from lume.ansi import (
    Caps, detect_caps, Style, NULL_STYLE, Console,
    display_width, strip_ansi, wrap, truncate, pad, hyperlink,
    hex_rgb, blend, gradient, fg, bg,
    RESET, CSI, HIDE_CURSOR, SHOW_CURSOR, CLEAR_LINE, up, down, col,
)
from lume.theme import Theme, THEMES, TOKENS, get_theme, theme_names
```

- `Caps(color: int, unicode: bool, is_tty: bool, width: int, height: int, hyperlinks: bool, animation: bool)`
  — `color` is 0 / 4 / 8 / 24 bits. Construct one directly in tests, e.g.
  `Caps(color=24, unicode=True, is_tty=True, width=80, height=24, animation=True)`.
- `Style(fg=None, bg=None, bold=..., dim=..., italic=..., underline=..., strike=..., reverse=...)`;
  call it as `style(text, caps) -> str`; merge with `a + b` (right wins on colour).
- `Theme.__getitem__(token) -> Style`, `theme.render(text, token, caps) -> str`. Token names are
  fixed — see `lume.theme.TOKENS`. **Never hard-code a colour outside `theme.py`.**
- `wrap(text, width, initial_indent="", subsequent_indent=None) -> list[str]` is ANSI-aware and
  carries SGR state across breaks. `display_width` accounts for CJK/emoji/combining marks.
- `Console` owns the single output lock: `console.write(s)`, `console.print(s)`, `console.lock`,
  `console.caps`, `console.width`, `console.refresh()`, `hide_cursor()`, `show_cursor()`,
  `clear_line()`, `clear_screen(scrollback=False)`.

---

## `lume/markdown.py` — streaming Markdown renderer and syntax highlighter

The single most visible part of the product: assistant replies arrive as a token stream and must
render as beautiful Markdown *as they arrive*, never flickering or re-printing.

```python
class MarkdownStream:
    def __init__(self, theme: Theme, caps: Caps, width: int = None, indent: str = "") -> None: ...
    def feed(self, chunk: str) -> str:
        """Consume a chunk of Markdown. Return text ready to print *now* (append-only).
        Buffer anything whose meaning is not yet decidable."""
    def close(self) -> str:
        """Flush all buffered content. Idempotent: a second call returns ''."""

def render_markdown(text: str, theme: Theme, caps: Caps, width: int = None, indent: str = "") -> str: ...
def highlight(code: str, lang: str | None, theme: Theme, caps: Caps) -> str: ...
def supported_languages() -> list[str]: ...
```

Requirements it satisfies:
- **Append-only streaming.** The concatenation of every `feed()` result plus `close()` must equal
  `render_markdown(full_text, ...)` for the same inputs, for *any* chunk split — including splits
  inside `**bold**`, inside a fence marker, mid-word, and one character at a time. This is the
  headline invariant; test it with randomized chunkings.
- Never emit a partial escape sequence and never leave a style unclosed at a line end.
- Block constructs: ATX headings `#`–`######`, fenced code blocks (``` and ~~~, with info string),
  indented code blocks, unordered lists (`-`/`*`/`+`), ordered lists, task lists (`- [ ]`/`- [x]`),
  nested lists (indent-aware), blockquotes (with a left bar), thematic breaks, tables (aligned
  columns with box-drawing borders when `caps.unicode`), and paragraphs.
- Inline constructs: `**bold**`, `*italic*`/`_italic_`, `` `code` ``, `~~strike~~`, links
  `[label](url)` (use `hyperlink()` when `caps.hyperlinks`, else show the URL dimmed), bare URLs,
  and backslash escapes. Inline code must never be re-parsed for other inline markers.
- Wrap paragraphs to `width` (default `caps.width`) honouring `indent`; do **not** wrap inside
  fenced code — instead, let long code lines be truncated or horizontally clipped, your choice,
  but document it.
- ASCII fallbacks for every glyph when `caps.unicode` is False.
- `highlight()` must support at least: python, javascript/typescript, json, bash/sh, html, css,
  sql, go, rust, c/c++, java, yaml, toml, markdown, diff. Unknown languages render plainly.
  Highlighting is a small hand-written tokenizer — strings (incl. escapes and triple-quotes),
  comments (line + block), numbers, keywords, builtins, decorators, function names, operators.
  A string containing `#` must not become a comment; a comment containing a quote must not open
  a string. Use only `syn.*` theme tokens.
- Code blocks get a subtle bordered/gutter presentation with the language label; keep it tasteful.

Covered by `tests/test_markdown.py`. Include the randomized-chunking equivalence test.

---

## `lume/motion.py` — the status animation, the wordmark, rules

"Slight motion" — tasteful, never noisy, and **completely invisible on a non-tty**.

```python
class Animator:
    def __init__(self, console: Console, theme: Theme, fps: float = 18.0) -> None: ...
    def status(self, label: str, style: str = "orbit") -> "ContextManager[StatusHandle]":
        """Context manager. Renders a transient one-line animation while the block runs and
        leaves the line clean on exit (no leftover glyphs, cursor restored)."""
    def stop(self) -> None: ...

class StatusHandle:
    def update(self, label: str) -> None: ...
    def elapsed(self) -> float: ...

SPINNERS: dict[str, ...]          # named frame sets; must include ASCII-safe variants
def banner(console: Console, theme: Theme, subtitle: str = "", animate: bool = True) -> None: ...
def rule(width: int, theme: Theme, caps: Caps, label: str = "") -> str: ...
def fade_in(text: str, console: Console, theme: Theme, token: str = "app.text") -> None: ...
```

Requirements it satisfies:
- The animation runs on a daemon thread, writes **only** through `console.write` while holding
  `console.lock`, and confines itself to the current line (`\r` + clear-line; never `\n`).
- On exit — normal, exception, `KeyboardInterrupt`, or `SIGINT` — the line is erased and the
  cursor is shown again. A leaked hidden cursor is a failing bug.
- Nesting / double-stop / stop-before-start must all be safe. `Animator` must be reusable.
- When `caps.animation` is False (not a tty, `TERM=dumb`, `NO_COLOR`, `LUME_NO_MOTION`), `status()`
  prints nothing at all (or one static line to stderr — your call, documented) and costs no thread.
- The banner is a gradient wordmark ("lume") built from `theme.accent_stops()` via `gradient()`;
  the animated reveal must finish in well under a second and degrade to a single static frame
  when `animate=False` or motion is unavailable.
- Frame writes must be cheap: no per-frame allocation storms, and no busy-wait (use `Event.wait`).
- Must never dead-lock with a foreground writer that holds `console.lock`.

Covered by `tests/test_motion.py` — drive it with an in-memory `io.StringIO` console (`is_tty` faked
both ways) and assert on the emitted bytes: no stray newline, cursor restored, line cleared.

---

## `lume/store.py` — local session persistence

Local-only. Nothing ever leaves the machine.

```python
@dataclass
class Message:
    role: str                 # "user" | "assistant" | "system"
    content: str
    ts: float = ...
    id: str = ...
    model: str | None = None
    usage: dict | None = None
    thinking: str | None = None
    error: str | None = None

@dataclass
class SessionMeta:
    id: str; title: str; created: float; updated: float
    model: str; system: str | None
    message_count: int; input_tokens: int; output_tokens: int
    cache_read_tokens: int; cache_write_tokens: int; cost_usd: float
    pinned: bool; tags: list[str]

class Store:
    def __init__(self, root: pathlib.Path | None = None) -> None: ...
    def create(self, *, model: str, system: str | None = None, title: str | None = None) -> SessionMeta: ...
    def append(self, session_id: str, message: Message) -> None: ...
    def load(self, session_id: str) -> tuple[SessionMeta, list[Message]]: ...
    def list(self, limit: int | None = None, query: str | None = None) -> list[SessionMeta]: ...
    def latest(self) -> SessionMeta | None: ...
    def resolve(self, ref: str) -> str:
        """Accept a full id, a unique id prefix, '1'-based list index, or 'last'.
        Raise KeyError (not found) / LookupError (ambiguous prefix)."""
    def update(self, session_id: str, **fields) -> SessionMeta: ...
    def rename(self, session_id: str, title: str) -> SessionMeta: ...
    def delete(self, session_id: str) -> None: ...
    def export(self, session_id: str, fmt: str = "markdown") -> str:   # markdown | json | text
    def path_for(self, session_id: str) -> pathlib.Path: ...
    def record_usage(self, session_id: str, usage: dict, cost_usd: float) -> SessionMeta: ...
```

Requirements it satisfies:
- Storage root resolution: `$LUME_HOME`, else `$XDG_DATA_HOME/lume`, else `~/.local/share/lume`
  (macOS: `~/Library/Application Support/lume` is acceptable). Directory created with mode `0o700`;
  session files `0o600` — these transcripts are private.
- One append-only **JSONL** file per session: line 1 is the meta header (`{"type":"meta",...}`),
  every later line is one message (`{"type":"message",...}`). Appends must be atomic enough that a
  crash mid-write cannot corrupt earlier lines, and a truncated/garbled trailing line must be
  skipped on load with a warning rather than raising.
- Meta updates rewrite the header safely (temp file + `os.replace`), never losing messages, and
  never leaving a partial file behind if interrupted.
- Concurrent processes must not interleave a write mid-line (use `fcntl.flock` where available,
  degrade gracefully on Windows).
- A `sessions/index.json` cache speeds up `list()`, but it is only a *cache*: if it is missing,
  stale, or corrupt, `list()` rebuilds from the session files and still returns correct results.
- Auto-title: derive a short title from the first user message when none was given; never crash on
  emoji/newlines/very long input; keep it to ~48 display columns.
- IDs are collision-resistant, sortable, and filename-safe; reject path traversal in any id/ref
  reaching `path_for` (`../`, absolute paths, NUL) — a malicious id must not escape the root.
- `list()` sorted newest-first, pinned first. `query` matches title and message text, case-insensitive.

Covered by `tests/test_store.py` — use `tempfile.TemporaryDirectory()` as the root; cover corruption
recovery, prefix/ambiguity resolution, traversal rejection, round-trips, and index-cache staleness.

---

## `lume/api.py` — the Claude API client

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str; label: str; context: int; max_output: int
    price_in: float; price_out: float                 # USD per 1M tokens
    price_cache_write: float; price_cache_read: float
    supports_temperature: bool; supports_effort: bool; thinking: str  # "adaptive"|"budget"|"none"

MODELS: dict[str, ModelSpec]
DEFAULT_MODEL = "claude-opus-5"
def resolve_model(name: str) -> ModelSpec: ...      # accepts aliases: opus, sonnet, haiku, fable

@dataclass
class Usage:
    input_tokens: int = 0; output_tokens: int = 0
    cache_creation_input_tokens: int = 0; cache_read_input_tokens: int = 0
    def cost(self, model: str | ModelSpec) -> float: ...
    def __add__(self, other) -> "Usage": ...

@dataclass
class StreamEvent:
    kind: str        # "text" | "thinking" | "usage" | "start" | "done" | "error" | "ping" | "fallback"
    text: str = ""
    usage: Usage | None = None
    stop_reason: str | None = None
    stop_details: dict | None = None
    model: str | None = None
    error: "APIError | None" = None

class APIError(Exception): ...
class AuthError(APIError): ...          # 401/403
class BadRequestError(APIError): ...    # 400/404/413
class RateLimitError(APIError): ...     # 429   (.retry_after: float | None)
class OverloadedError(APIError): ...    # 529
class ServerError(APIError): ...        # 5xx
class NetworkError(APIError): ...       # DNS/TLS/socket/timeouts
class CancelledError(APIError): ...
Each carries .status, .type, .request_id, .message, and .retryable.

def find_api_key(env=None) -> str | None:
    """ANTHROPIC_API_KEY, else ANTHROPIC_AUTH_TOKEN, else None. Never log or echo the value."""

class Client:
    def __init__(self, api_key: str, *, base_url: str = "https://api.anthropic.com",
                 timeout: float = 600.0, max_retries: int = 4, transport=None) -> None: ...
    def stream(self, *, model: str, messages: list[dict], system: str | list | None = None,
               max_tokens: int = 32000, thinking: bool = True, effort: str = "high",
               temperature: float | None = None, cache: bool = True,
               cancel: threading.Event | None = None) -> Iterator[StreamEvent]: ...
    def close(self) -> None: ...
```

Wire-format requirements, taken from the current API reference:
- Endpoint `POST {base_url}/v1/messages`. Headers: `content-type: application/json`,
  `x-api-key: <key>`, `anthropic-version: 2023-06-01`, plus `anthropic-beta` when a beta is used.
  If the key looks like an OAuth token (`sk-ant-oat*`), send `authorization: Bearer <key>` and the
  `oauth-2025-04-20` beta instead of `x-api-key`.
- `"stream": true` always; parse Server-Sent Events: blank-line-delimited records, `event:` and
  `data:` fields, multi-line `data:` concatenated with `\n`, ignoring comment lines beginning `:`.
  Event types to handle: `message_start` (carries `message.model` and initial `usage`),
  `content_block_start` / `content_block_delta` / `content_block_stop`, `message_delta`
  (carries `delta.stop_reason`, `delta.stop_details`, and cumulative `usage.output_tokens`),
  `message_stop`, `ping`, and `error`. Delta types: `text_delta`, `thinking_delta`,
  `signature_delta`, `input_json_delta`. A `content_block` of type `thinking` yields
  `kind="thinking"` events; type `fallback` yields `kind="fallback"`.
- Thinking: on `claude-opus-5` / `claude-fable-5` / `claude-opus-4-8` / `claude-opus-4-7` /
  `claude-sonnet-5`, send `"thinking": {"type": "adaptive", "display": "summarized"}` when enabled.
  **`budget_tokens` is rejected with a 400 on these models — never send it.** Effort goes in
  `"output_config": {"effort": ...}` (`low|medium|high|xhigh|max`), never top-level.
- **`temperature` is rejected (400) by the current models** — only send it when
  `spec.supports_temperature` is true.
- Refusal fallbacks on `claude-opus-5`/`claude-fable-5`: send `"fallbacks": "default"` with beta
  header `server-side-fallback-2026-07-01`. Surface `stop_reason == "refusal"` (HTTP 200!) to the
  caller as a `done` event with the stop reason, not as an exception.
- Prompt caching: place `"cache_control": {"type": "ephemeral"}` on the last content block of the
  stable prefix (system prompt, and the last block of the second-to-last message) when `cache=True`
  and the prefix plausibly exceeds ~1024 tokens. Never put anything volatile (timestamps, ids)
  before a breakpoint. Max 4 breakpoints.
- Retries with exponential backoff **and jitter** on 429/500/502/503/504/529 and connection errors,
  honouring the `retry-after` header; never retry 400/401/403/404/413; cap total wall clock.
  A retry must not emit duplicate text to the caller — only retry before the first token arrives.
- `cancel` (a `threading.Event`) must abort a stream promptly, close the socket, and raise
  `CancelledError` from the generator; closing the generator early must not leak the connection.
- `Usage.cost()` uses the per-model price table (USD/1M): opus-5 5/25, sonnet-5 3/15,
  haiku-4-5 1/5, fable-5 10/50; cache write = 1.25x input, cache read = 0.1x input.
- Transport seam: `Client(transport=...)` accepts an object with
  `open(url, headers, body, timeout) -> iterator of bytes` (plus `status`, `reason`, `headers`)
  so tests can inject canned SSE with **no network**. Default transport uses `http.client` with TLS.
- `SDKTransport` exists behind `Client(transport="auto")` and maps SDK stream events onto
  `StreamEvent`. **Nothing in the app selects it** — `app.py` always builds the stdlib
  transport — so it is unreachable in the shipped product and is a candidate for removal.
- Never write the API key to a log, an exception message, or a repr.

Covered by `tests/test_api.py` — canned-SSE transport plus a local `http.server` fake for the real
transport path (bind 127.0.0.1:0). Cover: chunk boundaries splitting an SSE record, retry/backoff,
retry-after, cancellation, refusal stop reason, cost maths, header construction, key redaction.

---

## `lume/input.py` and `lume/commands.py` — the prompt and the command vocabulary

```python
# commands.py
@dataclass(frozen=True)
class Command:
    name: str; args: str; help: str; group: str
    aliases: tuple = ()

COMMANDS: tuple[Command, ...]
def find(name: str) -> Command | None: ...
def parse(line: str) -> tuple[str | None, str]:
    """('model', 'sonnet') for '/model sonnet'; (None, line) for ordinary prose.
    A lone '/' or an unknown '/xyz' still returns ('xyz', rest) so the app can say
    'unknown command'. Escape hatch: a line starting '//' is literal text."""
def suggest(prefix: str) -> list[str]: ...
def help_text(theme, caps, width: int, group: str | None = None) -> str: ...

# input.py
class Prompt:
    def __init__(self, console: Console, theme: Theme, history_path=None,
                 completer=None, multiline_key: str = "alt+enter") -> None: ...
    def read(self, prefix: str = "", placeholder: str = "") -> str:
        """Read one submission. Raises EOFError on ctrl-D at an empty prompt and
        KeyboardInterrupt on ctrl-C. Returns '' for an empty line."""
    def add_history(self, text: str) -> None: ...
    def close(self) -> None: ...
```

Requirements it satisfies:
- Built on stdlib `readline` when importable (history file, emacs bindings, tab completion of
  slash commands and their arguments), with a plain-`input()` fallback when it is not — and the
  fallback must still work (no crash, no missing prompt) when stdin is a pipe.
- Multiline: a trailing `\` continues the line; a `"""` fence opens a block that ends on `"""`;
  paste of multi-line text must not fire one submission per line. Document the exact rules and
  show them in `/help`.
- Ctrl-C at the prompt clears the current line and returns to a fresh prompt (it must **not** kill
  the app); Ctrl-C during a response cancels only the response; Ctrl-D on an empty line exits.
- The prompt marker is themed and its display width is computed with `display_width` so readline's
  cursor arithmetic stays correct — wrap non-printing escapes in `\001`/`\002` for readline.
- History file is per-user, mode `0o600`, size-capped, and must never record a line that begins
  with a secret-looking token (`sk-ant-`).
- Commands, at minimum: `help`, `new`, `resume`, `list`, `delete`, `rename`, `model`, `models`,
  `system`, `theme`, `think`, `effort`, `clear`, `retry`, `edit`, `copy`, `export`, `usage`
  (with `tokens`/`cost` as aliases), `undo`, `keys`, `quit`. **`stream` was dropped:**
  replies always stream, so the toggle described a feature that does not exist. `commands.py` only *declares and parses* them; the handlers
  live in `app.py`.
- `help_text` renders a grouped, aligned, themed table that fits `width` and reads beautifully.

Covered by `tests/test_input.py` and `tests/test_commands.py`. Drive `Prompt` with a piped stdin
(`io.StringIO` / a real pipe) so it is testable without a tty.
