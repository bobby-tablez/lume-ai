<div align="center">

```
  █   ▄  ▄  ▄▄▄▄▄   ▄▄
  █   █  █  █ █ █  █▄▄█
  █▄  ▀▄▄▀  █ █ █  ▀▄▄▄
```

**A small, beautiful terminal chat client for Claude, GPT and Gemini.**

*lume* — rhymes with **room**. Not "loo-may", not "lum". It's the old word for *light*.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-black)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-2ea043)](pyproject.toml)
[![PyPI](https://img.shields.io/badge/pypi-lume--ai-0b7285)](https://pypi.org/project/lume-ai/)

</div>

---

Replies stream in as **rendered Markdown** — syntax-highlighted code, aligned tables,
nested lists — under a calm animated status line. Conversations live on your own
disk, in plain JSONL you can read, grep and delete. Five themes. One binary-free
install.

**Zero dependencies.** Pure standard library: nothing to resolve, nothing to audit,
nothing to break on upgrade day. Python 3.10 or newer.

```
  █   ▄  ▄  ▄▄▄▄▄   ▄▄
  █   █  █  █ █ █  █▄▄█
  █▄  ▀▄▄▀  █ █ █  ▀▄▄▄
  Opus 5

  /help for commands · Ctrl-C to stop a reply · Ctrl-D to leave
❯ explain a bloom filter in two sentences

lume
  A Bloom filter is a compact bit array plus k hash functions: to add an item you
  set the k bits it hashes to, and to test one you check those same bits.

  It answers "definitely not present" or "probably present" — false positives are
  possible, false negatives are not.

  ─ opus-5 · 1.4s · 412 in / 96 out · $0.004
❯
```

## Install

```bash
pip install lume-ai          # provides the `lume` command
```

Or run it straight from a checkout — there is nothing to build:

```bash
git clone https://github.com/bobby-tablez/lume-ai && cd lume-ai
./lume-cli                   # or: python3 -m lume
```

## Providers

lume speaks to three vendors through one interface. Set the key for whichever you
want; anything you have a key for shows up in `/models`.

| Provider | Environment variable | Aliases | Models |
|---|---|---|---|
| **Anthropic** | `ANTHROPIC_API_KEY` | `opus` `sonnet` `haiku` `fable` | Opus 5, Opus 4.x, Sonnet 5, Sonnet 4.6, Haiku 4.5, Fable 5 |
| **OpenAI** | `OPENAI_API_KEY` | `gpt` `mini` `nano` | GPT‑5.1, GPT‑5, GPT‑5 mini/nano, GPT‑4.1, GPT‑4o mini |
| **Google** | `GEMINI_API_KEY` | `gemini` `flash` `lite` | Gemini 3.1 Pro, 3.7 Flash, 2.5 Pro/Flash/Flash‑Lite |

`/models` shows the live table with context windows and per‑million prices, and
greys out any provider you have not set a key for.

```bash
export ANTHROPIC_API_KEY=...
lume                         # starts a session on the default model
lume --model sonnet          # or pick one
```

Switch mid-conversation with `/model gpt-5.1`, and pin the vendor when an alias is
claimed by two of them: `/model google:mini`. Aliases resolve in provider order —
Anthropic, then OpenAI, then Google. Any OpenAI-compatible endpoint — Groq, xAI,
DeepSeek, Together, OpenRouter, Ollama, LM Studio — works by pointing
`OPENAI_BASE_URL` at it.

Keys are read from the environment, used, and never written anywhere: not to the
config file, not to a log, not into an error message, and never into a URL.

## Use

```bash
lume                              # interactive
lume "explain this regex: ^\d{3}-\d{4}$"   # one-shot, then exit
git diff | lume "review this"     # stdin is folded into the message
lume --resume last                # pick up where you left off
lume --model haiku --no-thinking  # cheap and fast
lume --list                       # what have I got saved?
```

### Commands

**Session**

| Command | |
|---|---|
| `/new [title]` | Start a fresh conversation, optionally with a title. |
| `/resume, /r [ref]` | Reopen a session by id, list number, or "last". |
| `/list, /ls [query]` | List saved sessions, newest first; filter with a query. |
| `/rename [ref] <title>` | Give a session a better title. |
| `/delete, /rm <ref>` | Delete a session and its transcript for good. |
| `/export [format] [path]` | Write the transcript out as markdown, json, or text. |

**Conversation**

| Command | |
|---|---|
| `/system [text]` | Show, set, or clear the system prompt. |
| `/retry [note]` | Send the last message again, optionally with a nudge. |
| `/undo [n]` | Drop the last exchange, or the last n, from the chat. |
| `/edit [n]` | Reopen an earlier message in `$EDITOR` and send it again. |
| `/copy, /y [n]` | Copy the last reply, or message n, to the clipboard. |
| `/clear, /cls [history]` | Clear the screen; add `history` to also forget the conversation. |
| `/usage, /tokens, /cost` | Show the tokens this session has used and what they cost. |

**Model**

| Command | |
|---|---|
| `/model, /m [name]` | Show the current model, or switch to another. |
| `/models` | List every model with its context window and price. |
| `/think [on\|off]` | Turn extended thinking on or off. |
| `/effort [level]` | How hard to think: low, medium, high, xhigh, or max. |

**Interface**

| Command | |
|---|---|
| `/theme [name]` | Show the current colour theme, or switch to another. |
| `/keys` | Show the key bindings and the multi-line input rules. |
| `/help, /h, /? [topic]` | Show this help, or the detail for one command. |
| `/quit, /exit, /q` | Leave lume. |

Anything that is not a command is sent to the model. To send a line that really
does start with a slash, double it: `//not a command`.

### Typing

| | |
|---|---|
| `Enter` | Send the message. |
| `\` at end of line | Continue on the next line; the backslash is dropped. |
| `"""` | Open a block. A line ending in `"""` closes it and sends. |
| `Alt+Enter` | Add a newline without sending. *(needs GNU readline — not Windows, not libedit builds; `\` and `"""` work everywhere)* |
| paste | A multi-line paste arrives whole, as one message — never one per line. |
| `Tab` | Complete a command or its argument. |
| `↑` `↓` | Walk back and forth through history. |
| `Ctrl-C` | Throw the line away — or stop a reply that is already running. |
| `Ctrl-D` | Exit. *(`Ctrl-Z Enter` on Windows)* |

## Configuration

`~/.config/lume/config.json`, written by `/model`, `/theme` and friends, or edited
by hand. Every field also has an environment override:

| Variable | |
|---|---|
| `LUME_MODEL` | Default model. |
| `LUME_THEME` | `aurora`, `solar`, `ember`, `mono`, `plain`, `auto`. |
| `LUME_EFFORT` | `low`, `medium`, `high`, `xhigh`, `max`. |
| `LUME_SYSTEM` | Default system prompt. |
| `LUME_MAX_TOKENS` | Reply cap. |
| `LUME_HOME` | Where sessions and history live. |
| `LUME_NO_MOTION` | Turn animation off. |
| `NO_COLOR` | Turn colour off (respected everywhere). |

## Privacy

Conversations never leave your machine except as a request to the model provider
you chose. Sessions are append-only JSONL under `$LUME_HOME` (or
`~/.local/share/lume`), directories `0700`, files `0600`. The prompt's history file
is capped, `0600`, and refuses to record any line that looks like a credential —
paste a key at the prompt by accident and it is not written to disk.

## Development

```bash
python3 -m unittest discover -s tests -v      # 1,400+ tests, no network, no fixtures
```

Every module is independently testable and each provider client takes an injected
transport, so the suite never opens a socket and never needs an API key.
`SPEC.md` is the interface contract the modules were built against.

Adding a provider is three things: a `ModelSpec` per model, a client exposing
`stream()` and `close()` that yields `lume.api.StreamEvent`, and a line in
`lume/providers/__init__.py`.

## The name

**lume** — one syllable, rhymes with *room*. From *lumen*: light. It's what a
terminal does when it's doing this well.

## License

MIT © 2026 bobby-tablez
