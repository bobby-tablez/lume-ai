"""The application shell: wiring, the chat loop, and the slash-command handlers.

Everything interesting lives in the other modules; this one only decides what to
call and when. `commands.py` declares the commands, `App` implements them.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

from . import commands as cmds
from . import motion
from . import providers
from .ansi import (Console, detect_background, detect_caps, display_width, pad,
                   sanitize_text, truncate, wrap)
from .api import (APIError, AuthError, CancelledError, Client, DEFAULT_MODEL,
                  Usage)
from .config import Config, config_path, load_config, save_config
from .input import (Prompt, default_completer, default_history_path,
                    interrupt_guard)
from .markdown import MarkdownStream, render_markdown
from .store import Message, Store, default_root
from .theme import get_theme, theme_names

QUIT = object()          # sentinel: a handler asking the loop to exit


@dataclass
class Session:
    """The conversation currently on screen."""

    id: str
    title: str
    model: str
    system: str
    messages: list


class App:
    def __init__(self, config: Config, *, console: Console = None, store: Store = None,
                 client: Client = None, prompt: Prompt = None, env=None):
        self.env = os.environ if env is None else env
        self.config = config
        self.console = console or Console()
        if not config.animation:
            self.console.caps = self.console.caps.__class__(
                **{**self.console.caps.__dict__, "animation": False})
        # A dark palette on a white terminal is unreadable, so honour the
        # terminal's own report when the user has not chosen a theme.
        self._background = detect_background(self.env)
        self.theme = get_theme(config.theme, self.console.caps, self._background)
        self.store = store or Store(default_root(self.env))
        self.client = client
        self.animator = motion.Animator(self.console, self.theme)
        self.prompt = prompt or Prompt(
            self.console, self.theme,
            history_path=default_history_path(),
            completer=self._complete,
            history_max=config.history_size,
        )
        self.session: Session | None = None
        #: True when the model came from the command line rather than a session.
        self._model_pinned = False
        #: Width fixed for the current turn, so a mid-reply resize cannot split it.
        self._turn_width = 0
        #: Cost of the turn just finished, or None if it cost nothing.
        self._turn_cost = None
        self.usage = Usage()
        self.cost = 0.0
        self._cancel = threading.Event()
        self._streaming = False

    # ------------------------------------------------------------------ helpers
    @property
    def caps(self):
        return self.console.caps

    @property
    def width(self) -> int:
        return self.config.width_for(self.console.refresh().width)

    def say(self, text: str, token: str = "app.text") -> None:
        """Print one styled line. Text is sanitised: some of it is server- or
        model-controlled, and the styling is applied after, so nothing is lost."""
        self.console.print(self.theme.render(sanitize_text(text, keep_newlines=False),
                                             token, self.caps))

    def note(self, text: str) -> None:
        self.say(f"  {text}", "app.muted")

    def warn(self, text: str) -> None:
        self.say(f"  {text}", "app.warn")

    def fail(self, text: str) -> None:
        self.say(f"  {text}", "app.error")

    def ok(self, text: str) -> None:
        self.say(f"  {text}", "app.success")

    def glyph(self, fancy: str, plain: str) -> str:
        """Chrome must never emit a glyph the terminal cannot draw."""
        return fancy if self.caps.unicode else plain

    def quote(self, text: str) -> str:
        return f"\u201c{text}\u201d" if self.caps.unicode else f'"{text}"'

    def rule(self, label: str = "") -> None:
        self.console.print(motion.rule(self.width, self.theme, self.caps, label))

    def _complete(self, text: str, line: str) -> list:
        """Complete commands, then their arguments.

        Static argument values live in `commands.arg_values` so the registry stays
        the single source of truth; only the values that depend on live state
        (models, themes, saved sessions) are supplied here.
        """
        head = line.lstrip()
        if head.startswith("/") and " " in head:
            name = head[1:].split(" ", 1)[0]
            cmd = cmds.find(name)
            if cmd is not None:
                values = list(cmds.arg_values(cmd.name))
                if cmd.name == "model":
                    values = providers.model_names() + [
                        a for p in providers.providers() for a in p.aliases]
                elif cmd.name == "theme":
                    values = ["auto"] + theme_names()
                elif cmd.name in ("resume", "delete", "rename"):
                    values = ["last"] + [s.id for s in self.store.list(limit=12)]
                matches = [v for v in values if v.startswith(text)]
                if matches:
                    return matches
        return default_completer(text, line)

    # ------------------------------------------------------------------ sessions
    def start_session(self, *, title: str = None, system: str = None) -> Session:
        meta = self.store.create(model=self.config.model,
                                 system=system if system is not None else (self.config.system or None),
                                 title=title)
        self.session = Session(id=meta.id, title=meta.title, model=meta.model,
                               system=meta.system or "", messages=[])
        self.usage, self.cost = Usage(), 0.0
        return self.session

    def resume_session(self, ref: str) -> Session:
        session_id = self._resolve_ref(ref)
        meta, messages = self.store.load(session_id)
        self.session = Session(id=meta.id, title=meta.title, model=meta.model or self.config.model,
                               system=meta.system or self.config.system or "",
                               messages=list(messages))
        # A model named on this command line beats the one the session was saved
        # with — otherwise `--resume last -m haiku` silently bills at opus rates.
        if not self._model_pinned:
            self.config.model = self.session.model
        elif self.session.model and self.session.model != self.config.model:
            self.session.model = self.config.model
        self.usage = Usage(input_tokens=meta.input_tokens, output_tokens=meta.output_tokens,
                           cache_creation_input_tokens=meta.cache_write_tokens,
                           cache_read_input_tokens=meta.cache_read_tokens)
        self.cost = meta.cost_usd
        return self.session

    def _resolve_ref(self, ref: str) -> str:
        """Resolve a session reference, skipping conversations with nothing in them.

        An empty session can still exist (a `/new` that was never used), and
        "last" should mean the last conversation the user actually had.
        """
        ref = (ref or "last").strip()
        if ref.lower() in ("last", ""):
            for meta in self.store.list():
                if meta.message_count and meta.id != (self.session.id if self.session else None):
                    return meta.id
            for meta in self.store.list():          # nothing but empties: take one
                if meta.id != (self.session.id if self.session else None):
                    return meta.id
        return self.store.resolve(ref)

    def _api_messages(self) -> list:
        out = []
        if self.session is None:
            return out
        for m in self.session.messages:
            if m.role not in ("user", "assistant") or not m.content.strip():
                continue
            if m.error and not m.content.strip():
                continue        # a failed turn with nothing to show carries nothing
            # The API rejects two consecutive turns from the same role.
            if out and out[-1]["role"] == m.role:
                out[-1]["content"] += "\n\n" + m.content
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    # -------------------------------------------------------------------- render
    def _print_user_echo(self, text: str) -> None:
        if not self.caps.is_tty:
            return
        label = self.theme.render("you", "user.label", self.caps)
        self.console.print(f"\n{label}")
        for line in sanitize_text(text).splitlines() or [""]:
            self.console.print(self.theme.render("  " + line, "user.text", self.caps))

    def _ensure_client(self):
        """The client that serves the current model, built on first use.

        Which vendor answers is a property of the model, so switching model can
        mean switching client; `_cmd_model` drops this one when that happens. A
        client handed in by a caller (tests, embedding) is always kept.
        """
        if self.client is not None:
            return self.client
        try:
            client, _ = providers.make_client(
                self.config.model, self.env,
                timeout=self.config.timeout, max_retries=self.config.max_retries)
        except ValueError as exc:
            raise AuthError(str(exc)) from None
        self.client = client
        self._client_provider = providers.provider_for(self.config.model).name
        return client

    _client_provider = None

    @staticmethod
    def _spec_for(name):
        """Resolve a model across every provider, not just Anthropic."""
        return providers.resolve(name)[1]

    @contextlib.contextmanager
    def _quiet_store(self):
        """Turn store warnings into ordinary app messages.

        The default `warnings` output prints a file path and the offending source
        line to stderr, which is jarring in the middle of a conversation.
        """
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            yield
        for entry in caught:
            self.warn(str(entry.message))

    def send(self, text: str) -> bool:
        """One conversational turn: user message in, streamed reply out.

        Returns False when the turn failed, so one-shot mode can exit non-zero.
        """
        if self.session is None:
            self.start_session()
        try:
            client = self._ensure_client()
        except AuthError as exc:
            self.fail(str(exc))
            return False

        user_msg = Message(role="user", content=text)
        self.session.messages.append(user_msg)
        self.store.append(self.session.id, user_msg)
        if not self.session.title:
            with contextlib.suppress(Exception):
                meta = self.store.load(self.session.id)[0]
                self.session.title = meta.title

        self._cancel.clear()
        reply, thinking, usage, stop_reason, err, served = self._stream_reply(client)
        # The API may answer on a different model than we asked for (a refusal
        # fallback), and that model's rates are what the turn is billed at.
        served = served or self.config.model

        # Input tokens are billed as soon as the request is accepted, so account
        # for them whatever happened next — a failed turn is not a free turn.
        self._record_usage(usage, served)

        if err is not None and reply:
            # A partial reply is still on screen; say why it stopped rather than
            # letting the user believe that was the whole answer.
            self.fail(f"{err}")
        if err is not None and not reply:
            self.fail(f"{err}")
            failed = Message(role="assistant", content="", error=str(err))
            self.store.append(self.session.id, failed)
            self.session.messages.append(failed)
            return False

        msg = Message(role="assistant", content=reply, model=served,
                      thinking=thinking or None,
                      usage=usage.as_dict() if usage else None,
                      error=str(err) if err else None)
        self.session.messages.append(msg)
        self.store.append(self.session.id, msg)

        if self._turn_cost is not None and self.config.show_cost and self.caps.is_tty:
            self._print_footer(usage, self._turn_cost, stop_reason)
        if stop_reason == "refusal":
            self.warn("The model declined this request.")
        elif served != self.config.model:
            self.note(f"answered by {served}")
        return err is None

    def _record_usage(self, usage, served: str) -> None:
        """Bill a turn, however it ended. Sets `_turn_cost` for the footer."""
        self._turn_cost = None
        if usage is None or not (usage.input_tokens or usage.output_tokens
                                 or usage.cache_read_input_tokens
                                 or usage.cache_creation_input_tokens):
            return
        self.usage = self.usage + usage
        # Price against the *spec*, not the id: `Usage.cost` resolves a bare
        # string through the Anthropic table only, which is a ValueError for a
        # GPT or Gemini turn. A model we cannot place at all bills at zero
        # rather than taking the turn down with it.
        try:
            spec = self._spec_for(served or self.config.model)
        except ValueError:
            spec = None
        self._turn_cost = usage.cost(spec) if spec is not None else 0.0
        self.cost += self._turn_cost
        with contextlib.suppress(Exception):
            with self._quiet_store():
                self.store.record_usage(self.session.id, usage.as_dict(), self._turn_cost)

    def _stream_reply(self, client: Client):
        """Drive the stream, rendering markdown as it arrives. Returns the pieces."""
        theme, caps = self.theme, self.caps
        # Into a pipe, the useful output is the text the model wrote. Rendering it
        # would put box-drawing and bullets into someone's commit message.
        # Re-read the terminal size at the start of each turn. A resize mid-reply
        # cannot re-flow text already printed, so the width is fixed for the turn
        # and the footer below uses the same one rather than the live value.
        turn_width = self._turn_width = self.width
        renderer = (MarkdownStream(theme, caps, width=turn_width, indent="  ")
                    if caps.is_tty else _Verbatim())
        thinking_stream = _Wrapped(self.console, theme, caps, turn_width, "thinking.text")
        reply_parts, thinking_parts = [], []
        usage = None
        stop_reason = None
        error = None
        served_model = None
        started = False
        thinking_open = False

        label = theme.render("lume", "assistant.label", caps) if caps.is_tty else ""
        stack = contextlib.ExitStack()
        # The Ctrl-C watch lives in its own stack: `stack` is closed early, the
        # moment the first token lands, to take the spinner down.
        watch = contextlib.ExitStack()

        try:
            watch.enter_context(interrupt_guard(self._cancel.set))
            self._streaming = True
            if label:
                self.console.print(f"\n{label}")
            handle = stack.enter_context(self.animator.status("thinking"))

            for event in client.stream(
                model=self.config.model,
                messages=self._api_messages(),
                system=self.session.system or None,
                max_tokens=self.config.max_tokens,
                thinking=self.config.thinking,
                effort=self.config.effort,
                cache=self.config.cache,
                cancel=self._cancel,
            ):
                if event.kind == "thinking":
                    thinking_parts.append(event.text)
                    if self.config.show_thinking and event.text and caps.is_tty:
                        stack.close()
                        if not thinking_open:
                            thinking_open = True
                            if caps.is_tty:
                                self.console.print(
                                    theme.render("  thinking", "thinking.label", caps))
                        thinking_stream.feed(sanitize_text(event.text))
                elif event.kind == "text":
                    if not started:
                        stack.close()
                        if thinking_open:
                            thinking_stream.close()
                            self.console.print()
                            thinking_open = False
                        started = True
                    reply_parts.append(event.text)
                    self.console.write(renderer.feed(sanitize_text(event.text)))
                elif event.kind == "usage" and event.usage:
                    usage = event.usage
                elif event.kind in ("start", "fallback"):
                    if event.model:
                        served_model = event.model
                    # Input tokens are billed the moment the request is accepted,
                    # so keep them even if the user cancels a second later.
                    if event.usage and usage is None:
                        usage = event.usage
                elif event.kind == "done":
                    stop_reason = event.stop_reason or stop_reason
                    if event.usage:
                        usage = event.usage
                    if event.model:
                        served_model = event.model
                elif event.kind == "error" and event.error is not None:
                    error = event.error
        except CancelledError:
            error = None
            stop_reason = "cancelled"
        except APIError as exc:
            error = exc
        except KeyboardInterrupt:
            stop_reason = "cancelled"
        except Exception as exc:
            # A transport or parser bug must cost the user one reply, never the
            # whole session and never the text already on screen.
            error = exc
        finally:
            self._streaming = False
            watch.close()
            stack.close()
            self.animator.stop()
            tail = renderer.close()
            if tail:
                self.console.write(tail)
            if thinking_open or (started and caps.is_tty):
                self.console.print()

        if stop_reason == "cancelled":
            self.note("stopped")
        return ("".join(reply_parts), "".join(thinking_parts), usage, stop_reason,
                error, served_model)

    def _print_footer(self, usage: Usage, cost: float, stop_reason) -> None:
        theme, caps = self.theme, self.caps
        bits = [
            ("in", f"{usage.input_tokens:,}"),
            ("out", f"{usage.output_tokens:,}"),
        ]
        if usage.cache_read_input_tokens:
            bits.append(("cached", f"{usage.cache_read_input_tokens:,}"))
        bits.append(("cost", _money(cost)))
        bits.append(("total", _money(self.cost)))
        sep = theme.render(self.glyph(" \u00b7 ", " | "), "status.sep", caps)
        pieces = [theme.render(k, "status.key", caps) + " "
                  + theme.render(v, "status.value", caps) for k, v in bits]
        # Drop the least interesting figures rather than wrapping raggedly.
        width = self._turn_width or self.width
        while len(pieces) > 2 and display_width(sep.join(pieces)) + 2 > width:
            pieces.pop(0)
        self.console.print("  " + truncate(sep.join(pieces), max(4, width - 2)))

    # ------------------------------------------------------------------ commands
    def dispatch(self, name: str, rest: str):
        cmd = cmds.find(name)
        if cmd is None:
            self.fail(f"Unknown command /{name}." if name else "Type /help for commands.")
            near = cmds.suggest(name)
            if near:
                self.note("Did you mean: " + ", ".join("/" + n for n in near[:4]))
            return None
        handler = getattr(self, "_cmd_" + cmd.name, None)
        if handler is None:
            # The registry and the handlers live in different modules and can
            # drift; say so rather than dying inside the run loop.
            self.fail(f"/{cmd.name} is declared but not implemented yet.")
            return None
        try:
            return handler(rest.strip())
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError as exc:
            # Storage and clipboard errors are the common case here.
            self.fail(f"/{cmd.name} failed: {exc}")
        except Exception as exc:                    # noqa: BLE001
            # A bug in one command must not end the conversation.
            self.fail(f"/{cmd.name} failed unexpectedly: {type(exc).__name__}: {exc}")
        return None

    def _cmd_help(self, rest):
        self.console.print(cmds.help_text(self.theme, self.caps, self.width, rest or None))

    def _cmd_keys(self, rest):
        self.console.print(cmds.help_text(self.theme, self.caps, self.width, "input"))

    def _cmd_quit(self, rest):
        return QUIT

    def _cmd_new(self, rest):
        self.start_session(title=rest or None)
        self.rule("new conversation")

    def _cmd_resume(self, rest):
        try:
            self.reopen(rest or "last")
        except (KeyError, LookupError, ValueError) as exc:
            self.fail(_reason(exc, "No such conversation."))

    def reopen(self, ref: str) -> None:
        """Resume a conversation and show the user what they got back.

        Both `/resume` and `--resume` go through here: two entry points to one
        feature should not behave differently.
        """
        self.resume_session(ref)
        if self.caps.is_tty:
            self.rule(self.session.title or self.session.id[:8])
        self._replay()

    def _replay(self, limit: int = 6):
        shown = [m for m in self.session.messages if m.content.strip()][-limit:]
        if not shown:
            self.note("(no messages yet)")
            return
        for m in shown:
            if m.role == "user":
                self._print_user_echo(m.content)
            else:
                self.console.print("\n" + self.theme.render("lume", "assistant.label", self.caps))
                if self.caps.is_tty:
                    self.console.write(render_markdown(sanitize_text(m.content), self.theme,
                                                       self.caps, self.width, indent="  "))
                else:
                    self.console.print(sanitize_text(m.content))
        self.console.print()

    def _cmd_list(self, rest):
        self.list_sessions(rest)

    def list_sessions(self, rest: str = "") -> None:
        sessions = self.store.list(limit=30, query=rest or None)
        if not sessions:
            self.note("No saved conversations yet." if not rest else "Nothing matched.")
            return
        theme, caps = self.theme, self.caps
        self.console.print()
        for i, s in enumerate(sessions, 1):
            when = _ago(s.updated)
            num = theme.render(f"{i:>3}", "md.number", caps)
            title = truncate(s.title or "(untitled)", max(16, self.width - 34))
            title = theme.render(pad(title, max(16, self.width - 34)), "app.text", caps)
            meta = theme.render(f"{s.message_count:>3} msg  {when:>8}", "app.dim", caps)
            self.console.print(f"{num}  {title} {meta}")
        self.console.print()
        self.note('Reopen with /resume <number>, or /resume last')

    def _cmd_delete(self, rest):
        if not rest:
            self.fail("Which one? /delete <number|id|last>")
            return
        try:
            session_id = self.store.resolve(rest)
            meta = self.store.load(session_id)[0]
            self.store.delete(session_id)
        except (KeyError, LookupError, ValueError) as exc:
            self.fail(_reason(exc, "No such conversation."))
            return
        self.ok(f"Deleted {self.quote(meta.title or session_id[:8])}.")
        if self.session and self.session.id == session_id:
            self.session = None

    def _cmd_rename(self, rest):
        if not rest:
            self.fail("Usage: /rename [ref] <title>")
            return
        # A leading word is a reference only if it really names a session; a digit
        # that indexes nothing is an error, not the first word of a title.
        parts = rest.split(None, 1)
        ref, title = (self.session.id if self.session else "last"), rest
        if len(parts) == 2:
            head = parts[0]
            looks_like_ref = head == "last" or head.isdigit() or self.store.exists(head)
            if looks_like_ref:
                try:
                    self.store.resolve(head)
                except (KeyError, LookupError, ValueError) as exc:
                    self.fail(_reason(exc, f"No conversation matches {head!r}."))
                    return
                ref, title = head, parts[1]
        try:
            meta = self.store.rename(self.store.resolve(ref), title)
        except (KeyError, LookupError, ValueError) as exc:
            self.fail(_reason(exc, "No such conversation."))
            return
        if self.session and self.session.id == meta.id:
            self.session.title = meta.title
        self.ok(f"Renamed to {self.quote(meta.title)}.")

    def _cmd_export(self, rest):
        if self.session is None:
            self.fail("Nothing to export yet.")
            return
        # Both arguments are optional and either may come first:
        #   /export            -> print markdown
        #   /export json       -> print json
        #   /export out.md     -> write markdown to out.md
        #   /export json a.json-> write json to a.json
        formats = ("markdown", "md", "json", "text", "txt")
        parts = rest.split(None, 1)
        if parts and parts[0].lower() in formats:
            fmt = parts[0].lower()
            target = parts[1].strip() if len(parts) > 1 else None
        else:
            fmt = "markdown"
            target = rest.strip() or None
        try:
            data = self.store.export(self.session.id, fmt)
        except ValueError as exc:
            self.fail(str(exc))
            return
        if not target:
            self.console.print(data)
            return
        path = os.path.expanduser(target)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(data)
        except OSError as exc:
            self.fail(f"Could not write {path}: {exc}")
            return
        self.ok(f"Wrote {path}")

    def _cmd_system(self, rest):
        if self.session is None:
            self.start_session()
        if not rest:
            current = self.session.system or self.config.system
            self.note(f"System prompt: {current}" if current else "No system prompt set.")
            return
        if rest.strip().lower() in ("off", "none", "clear"):
            self.session.system = ""
            with contextlib.suppress(OSError, KeyError, ValueError):
                self.store.update(self.session.id, system=None)
            self.ok("System prompt cleared.")
            return
        self.session.system = rest
        try:
            self.store.update(self.session.id, system=rest)
        except (OSError, KeyError, ValueError) as exc:
            self.warn(f"Set for this run, but not saved: {exc}")
        else:
            self.ok("System prompt set for this conversation.")

    def _cmd_retry(self, rest):
        if not self.session or not any(m.role == "user" for m in self.session.messages):
            self.fail("Nothing to retry yet.")
            return
        # Drop the trailing assistant turn so the model answers afresh.
        while self.session.messages and self.session.messages[-1].role == "assistant":
            self.session.messages.pop()
        last_user = next(m for m in reversed(self.session.messages) if m.role == "user")
        self.session.messages.pop(self.session.messages.index(last_user))
        text = f"{last_user.content}\n\n{rest}" if rest else last_user.content
        self._print_user_echo(text)
        self.send(text)

    def _cmd_edit(self, rest):
        if not self.session:
            self.fail("Nothing to edit yet.")
            return
        users = [m for m in self.session.messages if m.role == "user"]
        if not users:
            self.fail("No messages to edit.")
            return
        try:
            index = int(rest) if rest else len(users)
            seed = users[index - 1].content
        except (ValueError, IndexError):
            self.fail(f"Pick a message between 1 and {len(users)}.")
            return
        edited = _external_edit(seed, self.env)
        if edited is None:
            self.fail("No editor available. Set $EDITOR.")
            return
        if not edited.strip():
            self.note("Empty — nothing sent.")
            return
        self._print_user_echo(edited)
        self.send(edited)

    def _cmd_copy(self, rest):
        if not self.session:
            self.fail("Nothing to copy yet.")
            return
        msgs = [m for m in self.session.messages if m.content.strip()]
        try:
            text = msgs[int(rest) - 1].content if rest else next(
                m.content for m in reversed(msgs) if m.role == "assistant")
        except (ValueError, IndexError, StopIteration):
            self.fail("Nothing to copy.")
            return
        if _clipboard_write(text):
            self.ok(f"Copied {len(text):,} characters.")
        else:
            self.warn("No clipboard tool found; printing instead.")
            self.console.print(text)

    def _cmd_clear(self, rest):
        self.console.clear_screen(scrollback=True)
        if rest.strip().lower() in ("history", "all"):
            self.start_session()
            self.note("Started a fresh conversation.")

    def _cmd_usage(self, rest):
        """Tokens and money in one place. /tokens and /cost are aliases of this."""
        u = self.usage
        spec = self._spec_for(self.config.model)
        dot = self.glyph(" \u00b7 ", " | ")
        self.note(dot.join((
            f"in {u.input_tokens:,}", f"out {u.output_tokens:,}",
            f"cache write {u.cache_creation_input_tokens:,}",
            f"cache read {u.cache_read_input_tokens:,}")))
        # The window holds the *current* prompt, not the sum of every turn, and
        # cache reads count toward it. Estimate from the conversation we would
        # send next, so switching models reprices the same number honestly.
        prompt_chars = sum(len(m["content"]) for m in self._api_messages()) if self.session else 0
        prompt_chars += len(self.session.system or "") if self.session else 0
        estimate = prompt_chars // 4
        share = f"~{estimate / spec.context * 100:.1f}%" if spec.context else "?"
        self.note(dot.join((
            f"next prompt ~{estimate:,} of {spec.context:,} tokens ({share})",
            f"cost {_money(self.cost)}")))

    def _cmd_undo(self, rest):
        """Drop the last exchange from the live conversation.

        The transcript on disk is append-only, so this changes what the model is
        sent from here on, not what was recorded. Say so, rather than implying the
        history was rewritten.
        """
        if not self.session or not self.session.messages:
            self.fail("Nothing to undo yet.")
            return
        try:
            rounds = max(1, int(rest)) if rest.strip() else 1
        except ValueError:
            self.fail("Usage: /undo [how many exchanges]")
            return

        dropped = 0
        for _ in range(rounds):
            if not self.session.messages:
                break
            while self.session.messages and self.session.messages[-1].role != "user":
                self.session.messages.pop()
            if self.session.messages:
                self.session.messages.pop()
                dropped += 1
        if not dropped:
            self.fail("Nothing to undo yet.")
            return
        word = "exchange" if dropped == 1 else "exchanges"
        self.ok(f"Dropped the last {dropped} {word} from the conversation.")
        self.note("The saved transcript still has them; /export shows everything.")

    def _cmd_model(self, rest):
        if not rest:
            spec = self._spec_for(self.config.model)
            vendor = providers.provider_for(self.config.model)
            self.note(f"{spec.label} ({spec.id}) via {vendor.label}")
            return
        try:
            vendor, spec = providers.resolve(rest)
        except ValueError:
            self.fail(f"Unknown model {rest!r}. Try /models.")
            return
        if self._client_provider is not None and vendor.name != self._client_provider:
            with contextlib.suppress(Exception):       # a new vendor needs a new client
                self.client.close()
            self.client, self._client_provider = None, None
        self.config.model = spec.id
        if self.session:
            self.session.model = spec.id
            with contextlib.suppress(Exception):
                self.store.update(self.session.id, model=spec.id)
        self.ok(f"Now using {spec.label}.")
        # Switching is always allowed: a user may well set the key next.
        if not vendor.find_key(self.env):
            self.warn(providers.missing_key_hint(vendor))

    def _cmd_models(self, rest):
        theme, caps = self.theme, self.caps
        self.console.print()
        for n, vendor in enumerate(providers.providers()):
            if n:
                self.console.print()
            keyed = bool(vendor.find_key(self.env))
            head = theme.render(vendor.label, "cmd.group", caps)
            if not keyed:
                head += theme.render("  " + providers.missing_key_hint(vendor),
                                     "app.dim", caps)
            self.console.print(f" {head}")
            self._print_models(vendor.models.values(), dim=not keyed)
        self.console.print()

    def _print_models(self, specs, dim: bool = False):
        theme, caps = self.theme, self.caps
        specs = list(specs)
        for spec in specs:
            width = max(24, max((len(m.id) for m in specs), default=0) + 2)
            name = theme.render(pad(spec.id, width),
                                "cmd.name" if not dim else "app.dim", caps)
            window = (f"{spec.context // 1_000_000}M" if spec.context >= 1_000_000
                      else f"{spec.context // 1000}K")
            ctx = theme.render(pad(window, 6), "app.muted", caps)
            price_in, price_out = spec.prices()
            label = f"${price_in:g}/${price_out:g} per Mtok"
            if (price_in, price_out) != (spec.price_in, spec.price_out):
                label += " (introductory)"
            price = theme.render(label, "app.dim", caps)
            marker = (self.glyph("\u2192", ">") if spec.id == self.config.model else " ")
            self.console.print(f" {marker} {name}{ctx}{price}")

    def _cmd_think(self, rest):
        value = _on_off(rest, self.config.thinking)
        if value is None:
            self.fail("Usage: /think on|off")
            return
        self.config.thinking = value
        self.ok(f"Extended thinking {'on' if value else 'off'}.")

    def _cmd_effort(self, rest):
        if not rest:
            self.note(f"Effort is {self.config.effort}.")
            return
        if rest not in ("low", "medium", "high", "xhigh", "max"):
            self.fail("Pick one of: low, medium, high, xhigh, max.")
            return
        self.config.effort = rest
        self.ok(f"Effort set to {rest}.")

    def _cmd_theme(self, rest):
        if not rest:
            self.note(f"Theme is {self.theme.name}. Available: {', '.join(theme_names())}")
            return
        if rest != "auto" and rest not in theme_names():
            self.fail(f"Unknown theme {rest!r}. "
                      f"Try: auto, {', '.join(theme_names())}")
            return
        self.config.theme = rest
        self.theme = get_theme(rest, self.caps, self._background)
        self.animator = motion.Animator(self.console, self.theme)
        self.ok(f"Theme set to {self.theme.name}.")

    # ---------------------------------------------------------------------- loop
    _greeted = False

    def greet(self) -> None:
        if self._greeted or not self.caps.is_tty or self.config.theme == "plain":
            return
        self._greeted = True
        spec = self._spec_for(self.config.model)
        motion.banner(self.console, self.theme, subtitle=spec.label,
                      animate=self.config.animation and self.caps.animation)
        dot = self.glyph(" \u00b7 ", " | ")
        self.note(dot.join(("/help for commands", "Ctrl-C to stop a reply",
                            "%s to leave" % cmds.EOF_KEY)))
        self.report_warnings()

    def report_warnings(self) -> None:
        """Say what was ignored. One-shot mode never greets, but still needs this."""
        for warning in self.config.warnings:
            self.warn(warning)
        self.config.warnings.clear()

    def run(self) -> int:
        self.greet()
        while True:
            try:
                line = self.prompt.read()
            except EOFError:
                self.console.print()
                break
            except KeyboardInterrupt:
                self.console.print()
                break
            if not line.strip():
                continue
            name, rest = cmds.parse(line)
            if name is not None:
                if self.dispatch(name, rest) is QUIT:
                    break
                continue
            self.send(rest)
        self.shutdown()
        return 0

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self.animator.stop()
        with contextlib.suppress(Exception):
            self.prompt.close()
        with contextlib.suppress(Exception):
            if self.client is not None:
                self.client.close()
        self.console.show_cursor()


# ------------------------------------------------------------------- utilities


class _Verbatim:
    """A renderer that renders nothing: the model's own text, unchanged.

    Used when stdout is not a terminal, so `lume "…" > file` and
    `git diff | lume "write a commit message"` produce exactly what was written.
    """

    def feed(self, chunk: str) -> str:
        return chunk

    def close(self) -> str:
        return ""


class _Wrapped:
    """Stream plain text into a wrapped, indented block.

    Thinking output is prose, not markdown, but it still has to respect the same
    margin and width as everything else on screen.
    """

    def __init__(self, console, theme, caps, width, token, indent="  "):
        self._console, self._theme, self._caps = console, theme, caps
        self._width, self._token, self._indent = width, token, indent
        self._buf = ""

    def feed(self, text: str) -> None:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)

    def close(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, line: str) -> None:
        for wrapped in wrap(line, self._width, self._indent, self._indent):
            self._console.print(self._theme.render(wrapped, self._token, self._caps))


def _reason(exc: BaseException, fallback: str) -> str:
    """A human-readable reason from an exception.

    `str(KeyError("x"))` is `"'x'"` — truthy, and meaningless to a user — so a
    bare `str(exc) or fallback` never reaches the fallback.
    """
    text = str(exc).strip()
    if isinstance(exc, KeyError) or len(text) < 4 or text == repr(text.strip("'\"")):
        return fallback
    return text


def _money(amount: float) -> str:
    """Format a dollar amount for display.

    The running total is accumulated at full precision and only rounded here, so
    a column of per-turn figures can differ from the total beside it by up to
    half a display unit per turn. The total is the accurate one.
    """
    if amount and amount < 0.0001:
        return "<$0.0001"
    return f"${amount:.4f}"


def _on_off(rest: str, current: bool):
    value = rest.strip().lower()
    if value in ("on", "yes", "true", "1"):
        return True
    if value in ("off", "no", "false", "0"):
        return False
    if value == "":
        return not current
    return None


def _ago(when: float) -> str:
    delta = max(0, time.time() - (when or 0))
    for limit, unit, size in ((60, "s", 1), (3600, "m", 60), (86400, "h", 3600),
                              (86400 * 30, "d", 86400)):
        if delta < limit:
            return f"{int(delta // size)}{unit}"
    return f"{int(delta // (86400 * 30))}mo"


def _clipboard_write(text: str) -> bool:
    for argv in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"],
                 ["xsel", "--clipboard", "--input"], ["clip.exe"]):
        if shutil.which(argv[0]) is None:
            continue
        try:
            proc = subprocess.run(argv, input=text.encode("utf-8"), timeout=5)
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _external_edit(seed: str, env) -> str | None:
    editor = env.get("VISUAL") or env.get("EDITOR")
    if not editor:
        return None
    fd, path = tempfile.mkstemp(prefix="lume-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(seed)
        subprocess.call(shlex.split(editor) + [path])
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
