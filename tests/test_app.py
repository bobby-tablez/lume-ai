"""End-to-end tests for the application shell.

These drive the real App over a fake transport and a scripted stdin, so the whole
path — prompt, command dispatch, streaming, markdown rendering, persistence, cost
accounting — is exercised without a network or a terminal.
"""

import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lume import commands as cmds
from lume.ansi import Caps, Console, strip_ansi
from lume.api import Client, Usage, resolve_model
from lume.app import QUIT, App
from lume.config import Config
from lume.input import Prompt
from lume.store import Store
from lume.theme import get_theme

from test_api import FakeStream, FakeTransport, sse


def caps(tty=True, color=24):
    return Caps(color=color, unicode=True, is_tty=tty, width=80, height=24,
                hyperlinks=tty, animation=False)


def reply_records(text="Hello **there**.", *, stop_reason="end_turn", model="claude-opus-5"):
    return [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [], "usage": {"input_tokens": 12, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason},
                           "usage": {"output_tokens": 7}}),
        ("message_stop", {"type": "message_stop"}),
    ]


class AppHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def make_app(self, *, script=None, stdin="", tty=True, color=24, **cfg):
        console = Console(stream=self.out, caps=caps(tty=tty, color=color))
        config = Config(animation=False, **cfg).validate()
        transport = FakeTransport(script if script is not None else
                                  [lambda: FakeStream(chunks=[sse(reply_records())])])
        client = Client("sk-ant-test", transport=transport)
        prompt = Prompt(console, get_theme(config.theme, console.caps),
                        history_path="", stdin=io.StringIO(stdin))
        app = App(config, console=console, store=Store(self.root),
                  client=client, prompt=prompt, env={"LUME_HOME": str(self.root)})
        self.transport = transport
        return app

    @property
    def text(self):
        return strip_ansi(self.out.getvalue())


class TestOneTurn(AppHarness):
    def test_reply_is_rendered_and_persisted(self):
        app = self.make_app()
        app.start_session()
        self.assertTrue(app.send("hi"))
        self.assertIn("Hello there.", self.text)
        meta, messages = app.store.load(app.session.id)
        self.assertEqual([m.role for m in messages], ["user", "assistant"])
        self.assertEqual(messages[0].content, "hi")
        self.assertEqual(messages[1].content, "Hello **there**.")

    def test_markdown_is_actually_rendered_not_echoed(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        self.assertNotIn("**there**", self.text)

    def test_usage_and_cost_are_recorded(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        self.assertEqual(app.usage.input_tokens, 12)
        self.assertEqual(app.usage.output_tokens, 7)
        self.assertGreater(app.cost, 0)
        meta = app.store.load(app.session.id)[0]
        self.assertGreater(meta.cost_usd, 0)

    def test_request_carries_the_conversation(self):
        app = self.make_app()
        app.start_session()
        app.send("first")
        body = self.transport.calls[0]["body"]
        self.assertEqual(body["messages"], [{"role": "user", "content": "first"}])
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertTrue(body["stream"])

    def test_second_turn_includes_history(self):
        app = self.make_app(script=[lambda: FakeStream(chunks=[sse(reply_records("one"))]),
                                    lambda: FakeStream(chunks=[sse(reply_records("two"))])])
        app.start_session()
        app.send("a")
        app.send("b")
        roles = [m["role"] for m in self.transport.calls[1]["body"]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_system_prompt_is_sent(self):
        app = self.make_app()
        app.start_session(system="Be terse.")
        app.send("hi")
        body = self.transport.calls[0]["body"]
        self.assertIn("Be terse.", json.dumps(body["system"]))

    def test_no_escapes_when_output_is_not_a_terminal(self):
        app = self.make_app(tty=False, color=0)
        app.start_session()
        app.send("hi")
        self.assertNotIn("\x1b", self.out.getvalue())

    def test_refusal_is_reported_not_raised(self):
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records("", stop_reason="refusal"))])])
        app.start_session()
        app.send("hi")
        self.assertIn("declined", self.text)


class TestErrors(AppHarness):
    def test_api_error_is_shown_and_reported_as_failure(self):
        body = json.dumps({"type": "error", "error": {
            "type": "invalid_request_error", "message": "nope"}}).encode()
        app = self.make_app(script=[lambda: FakeStream(status=400, body=body)])
        app.start_session()
        self.assertFalse(app.send("hi"))
        self.assertIn("nope", self.text)

    def test_missing_key_is_a_clear_message(self):
        console = Console(stream=self.out, caps=caps())
        app = App(Config(animation=False).validate(), console=console,
                  store=Store(self.root),
                  prompt=Prompt(console, get_theme("aurora"), history_path="",
                                stdin=io.StringIO("")),
                  env={"LUME_HOME": str(self.root)})
        app.start_session()
        self.assertFalse(app.send("hi"))
        self.assertIn("No API key", self.text)

    def test_api_key_never_reaches_the_screen(self):
        body = json.dumps({"type": "error", "error": {
            "type": "authentication_error", "message": "bad key"}}).encode()
        app = self.make_app(script=[lambda: FakeStream(status=401, body=body)])
        app.start_session()
        app.send("hi")
        self.assertNotIn("sk-ant-test", self.out.getvalue())


class TestCommands(AppHarness):
    def test_help_lists_commands(self):
        app = self.make_app()
        app.dispatch("help", "")
        for name in ("resume", "model", "theme", "quit"):
            self.assertIn("/" + name, self.text)

    def test_quit_returns_the_sentinel(self):
        self.assertIs(self.make_app().dispatch("quit", ""), QUIT)

    def test_unknown_command_suggests(self):
        app = self.make_app()
        app.dispatch("mdoel", "")
        self.assertIn("Unknown command", self.text)

    def test_model_switch_changes_the_next_request(self):
        app = self.make_app()
        app.start_session()
        app.dispatch("model", "haiku")
        app.send("hi")
        self.assertEqual(self.transport.calls[0]["body"]["model"], "claude-haiku-4-5")

    def test_model_rejects_nonsense(self):
        app = self.make_app()
        app.dispatch("model", "gpt-9")
        self.assertIn("Unknown model", self.text)
        self.assertEqual(app.config.model, "claude-opus-5")

    def test_theme_switch(self):
        app = self.make_app()
        app.dispatch("theme", "ember")
        self.assertEqual(app.theme.name, "ember")

    def test_theme_rejects_nonsense(self):
        app = self.make_app()
        app.dispatch("theme", "neon")
        self.assertIn("Unknown theme", self.text)

    def test_new_starts_a_fresh_session(self):
        app = self.make_app()
        app.start_session()
        first = app.session.id
        app.dispatch("new", "Notes")
        self.assertNotEqual(app.session.id, first)
        self.assertEqual(app.session.title, "Notes")

    def test_system_set_and_clear(self):
        app = self.make_app()
        app.start_session()
        app.dispatch("system", "Be brief.")
        self.assertEqual(app.session.system, "Be brief.")
        app.dispatch("system", "off")
        self.assertEqual(app.session.system, "")

    def test_list_and_resume_round_trip(self):
        app = self.make_app()
        app.start_session(title="Alpha")
        app.send("hi")
        original = app.session.id
        app.dispatch("new", "Beta")
        app.dispatch("list", "")
        app.dispatch("resume", original)
        self.assertEqual(app.session.id, original)
        self.assertIn("Hello there.", self.text)

    def test_ambiguous_prefix_is_refused_rather_than_guessed(self):
        """Ids minted in the same millisecond share a long prefix; never guess."""
        app = self.make_app()
        app.start_session(title="Alpha")
        first = app.session.id
        app.dispatch("new", "Beta")
        second = app.session.id
        shared = 0
        for a, b in zip(first, second):
            if a != b:
                break
            shared += 1
        app.dispatch("resume", first[:shared])
        self.assertEqual(app.session.id, second)      # unchanged
        self.assertIn("matches 2 sessions", self.text)

    def test_resume_by_list_number(self):
        app = self.make_app()
        app.start_session(title="Alpha")
        app.send("hi")
        wanted = app.session.id
        app.dispatch("new", "Beta")
        app.dispatch("resume", "2")
        self.assertEqual(app.session.id, wanted)

    def test_resume_last(self):
        app = self.make_app()
        app.start_session(title="Alpha")
        app.send("hi")
        app.dispatch("resume", "last")
        self.assertEqual(app.session.title, "Alpha")

    def test_resume_unknown_is_a_message_not_a_crash(self):
        app = self.make_app()
        app.dispatch("resume", "nope-nope-nope")
        self.assertTrue(self.text.strip())

    def test_delete_removes_the_session(self):
        app = self.make_app()
        app.start_session(title="Doomed")
        app.send("hi")
        session_id = app.session.id
        app.dispatch("delete", session_id[:8])
        self.assertFalse(app.store.exists(session_id))

    def test_rename(self):
        app = self.make_app()
        app.start_session(title="Old")
        app.dispatch("rename", "Fresh Name")
        self.assertEqual(app.store.load(app.session.id)[0].title, "Fresh Name")

    def test_export_prints_markdown(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        app.dispatch("export", "")
        self.assertIn("hi", self.text)

    def test_export_to_a_file(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        target = self.root / "out.md"
        app.dispatch("export", f"markdown {target}")
        self.assertIn("hi", target.read_text(encoding="utf-8"))

    def test_effort_and_think_reach_the_request(self):
        app = self.make_app()
        app.start_session()
        app.dispatch("effort", "low")
        app.dispatch("think", "off")
        app.send("hi")
        body = self.transport.calls[0]["body"]
        self.assertEqual(body.get("output_config", {}).get("effort"), "low")
        self.assertNotEqual(body.get("thinking", {}).get("type"), "adaptive")

    def test_effort_rejects_nonsense(self):
        app = self.make_app()
        app.dispatch("effort", "ludicrous")
        self.assertEqual(app.config.effort, "high")

    def test_tokens_and_cost_report(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        app.dispatch("tokens", "")
        app.dispatch("cost", "")
        self.assertIn("$", self.text)

    def test_retry_resends_without_duplicating_history(self):
        app = self.make_app(script=[lambda: FakeStream(chunks=[sse(reply_records("one"))]),
                                    lambda: FakeStream(chunks=[sse(reply_records("two"))])])
        app.start_session()
        app.send("question")
        app.dispatch("retry", "")
        sent = self.transport.calls[1]["body"]["messages"]
        self.assertEqual(sent, [{"role": "user", "content": "question"}])


class TestLoop(AppHarness):
    def test_scripted_session_runs_and_exits(self):
        app = self.make_app(stdin="hello\n/quit\n")
        self.assertEqual(app.run(), 0)
        self.assertIn("Hello there.", self.text)

    def test_eof_ends_the_loop(self):
        app = self.make_app(stdin="")
        self.assertEqual(app.run(), 0)

    def test_blank_lines_are_ignored(self):
        app = self.make_app(stdin="\n   \n/quit\n")
        app.run()
        self.assertEqual(len(self.transport.calls), 0)

    def test_literal_slash_escape_is_sent_as_text(self):
        app = self.make_app(stdin="//help\n/quit\n")
        app.run()
        self.assertEqual(self.transport.calls[0]["body"]["messages"][0]["content"], "/help")

    def test_session_survives_a_restart(self):
        app = self.make_app()
        app.start_session(title="Persisted")
        app.send("remember this")
        session_id = app.session.id
        app.shutdown()

        again = self.make_app()
        again.resume_session(session_id)
        self.assertEqual([m.content for m in again.session.messages],
                         ["remember this", "Hello **there**."])




class TestAsciiChrome(AppHarness):
    """Chrome must stay inside ASCII when the terminal cannot draw more."""

    def make_ascii_app(self, **kw):
        app = self.make_app(**kw)
        app.console.caps = Caps(color=24, unicode=False, is_tty=True, width=80,
                                height=24, hyperlinks=False, animation=False)
        return app

    def _assert_ascii(self):
        for ch in self.text:
            self.assertLess(ord(ch), 128, f"non-ASCII chrome: {ch!r} in {self.text!r}")

    def test_greeting_is_ascii(self):
        app = self.make_ascii_app()
        app.greet()
        self._assert_ascii()

    def test_usage_footer_is_ascii(self):
        app = self.make_ascii_app()
        app.start_session()
        app.send("hi")
        self.out.truncate(0), self.out.seek(0)
        app.dispatch("tokens", "")
        app.dispatch("cost", "")
        self._assert_ascii()

    def test_models_table_is_ascii(self):
        app = self.make_ascii_app()
        app.dispatch("models", "")
        self._assert_ascii()

    def test_rename_and_delete_messages_are_ascii(self):
        app = self.make_ascii_app()
        app.start_session(title="Old")
        app.dispatch("rename", "New Title")
        app.dispatch("delete", app.session.id)
        self._assert_ascii()

    def test_unicode_chrome_when_supported(self):
        app = self.make_app()
        app.greet()
        self.assertIn("·", self.text)


class TestResilience(AppHarness):
    """An unexpected failure must cost one reply, not the session."""

    def test_unexpected_transport_exception_does_not_kill_the_loop(self):
        class Exploding:
            def open(self, url, headers, body, timeout):
                raise AttributeError("'NoneType' object has no attribute 'close'")

        app = self.make_app(stdin="hello\n/quit\n")
        app.client = Client("sk-ant-test", transport=Exploding())
        self.assertEqual(app.run(), 0)
        self.assertIn("NoneType", self.text)

    def test_partial_reply_is_kept_when_the_stream_dies_midway(self):
        class HalfStream:
            """Emits some text, then fails the way a dropped connection does."""

            status, reason, headers = 200, "OK", {}

            def __iter__(self):
                yield sse(reply_records("Partial answer")[:3])
                raise OSError("connection reset by peer")

            def close(self):
                pass

        app = self.make_app(script=[lambda: HalfStream()])
        app.start_session()
        app.send("hi")
        self.assertIn("Partial answer", self.text)
        messages = app.store.load(app.session.id)[1]
        self.assertEqual(messages[-1].content, "Partial answer")

    def test_a_failed_turn_does_not_corrupt_the_next_request(self):
        """A failed turn leaves no empty assistant message in the history."""
        body = json.dumps({"type": "error", "error": {
            "type": "overloaded_error", "message": "busy"}}).encode()
        app = self.make_app(script=[
            lambda: FakeStream(status=529, body=body),
            lambda: FakeStream(chunks=[sse(reply_records("recovered"))]),
        ])
        app.client.max_retries = 0
        app.start_session()
        self.assertFalse(app.send("first"))
        self.assertTrue(app.send("second"))
        sent = self.transport.calls[-1]["body"]["messages"]
        for message in sent:
            self.assertTrue(message["content"].strip(), f"empty turn sent: {sent}")
        self.assertNotIn("assistant", [m["role"] for m in sent])
        self.assertIn("recovered", self.text)


class TestExportParsing(AppHarness):
    """Both /export arguments are optional and either may come first."""

    def setUp(self):
        super().setUp()
        self.app = self.make_app()
        self.app.start_session(title="Session")
        self.app.send("hello")
        self.out.truncate(0), self.out.seek(0)

    def test_bare_export_prints_markdown(self):
        self.app.dispatch("export", "")
        self.assertIn("hello", self.text)

    def test_format_only_prints_that_format(self):
        self.app.dispatch("export", "json")
        self.assertIn('"role"', self.text)

    def test_path_only_writes_markdown(self):
        target = self.root / "a.md"
        self.app.dispatch("export", str(target))
        self.assertIn("hello", target.read_text(encoding="utf-8"))

    def test_format_and_path(self):
        target = self.root / "b.json"
        self.app.dispatch("export", f"json {target}")
        self.assertIn('"role"', target.read_text(encoding="utf-8"))

    def test_a_path_that_looks_like_a_format_is_still_a_format(self):
        self.app.dispatch("export", "TEXT")
        self.assertTrue(self.text.strip())

    def test_unwritable_path_reports_instead_of_raising(self):
        self.app.dispatch("export", str(self.root / "nope" / "c.md"))
        self.assertIn("Could not write", self.text)


class TestEditorInvocation(AppHarness):
    def test_editor_arguments_are_honoured_without_a_shell(self):
        import lume.app as app_mod

        seen = {}

        def fake_call(argv):
            seen["argv"] = argv
            with open(argv[-1], "w", encoding="utf-8") as fh:
                fh.write("edited text")
            return 0

        original = app_mod.subprocess.call
        app_mod.subprocess.call = fake_call
        try:
            result = app_mod._external_edit("seed", {"EDITOR": "code -w"})
        finally:
            app_mod.subprocess.call = original
        self.assertEqual(seen["argv"][:2], ["code", "-w"])
        self.assertEqual(result, "edited text")

    def test_no_editor_configured_returns_none(self):
        import lume.app as app_mod
        self.assertIsNone(app_mod._external_edit("seed", {}))


class TestServedModelPricing(AppHarness):
    """A turn is priced and recorded against the model that actually answered."""

    def test_normal_turn_uses_the_requested_model(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        self.assertEqual(app.store.load(app.session.id)[1][-1].model, "claude-opus-5")

    def test_a_turn_answered_by_another_model_is_priced_at_that_model(self):
        records = reply_records("rescued", model="claude-haiku-4-5")
        app = self.make_app(script=[lambda: FakeStream(chunks=[sse(records)])])
        app.start_session()
        app.send("hi")
        stored = app.store.load(app.session.id)[1][-1]
        self.assertEqual(stored.model, "claude-haiku-4-5")
        expected = Usage(input_tokens=12, output_tokens=7).cost("claude-haiku-4-5")
        self.assertAlmostEqual(app.cost, expected, places=9)
        self.assertIn("answered by claude-haiku-4-5", self.text)

    def test_the_substitution_is_not_announced_when_it_did_not_happen(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        self.assertNotIn("answered by", self.text)


class TestTerminalInjection(AppHarness):
    """Model and server output must never be able to drive the terminal."""

    ATTACKS = (
        "\x1b]52;c;cHduZWQ=\x07",      # clipboard write
        "\x1b]0;PWNED\x07",            # window title
        "\x1b[2J\x1b[H",               # clear screen
        "\x1b[?1049h",                 # alternate screen
        "\x9b31m",                     # 8-bit CSI
        "before\rAFTER",               # carriage-return overwrite
    )

    def _assert_clean(self):
        """No escape *sequence* from the payload may survive.

        The payload text itself may remain — with its ESC removed it is inert
        prose. What must not survive is anything the terminal would act on.
        """
        raw = self.out.getvalue()
        self.assertNotIn("\x1b]", raw, "an OSC sequence reached the terminal")
        self.assertNotIn("\x1b[?", raw, "a private-mode sequence reached the terminal")
        self.assertNotIn("\x1bP", raw, "a DCS sequence reached the terminal")
        self.assertNotIn("\x9b", raw, "an 8-bit CSI reached the terminal")
        self.assertNotIn("\r", raw, "a carriage return reached the terminal")
        for match in re.finditer(r"\x1b\[([0-9;]*)m", raw):
            self.assertRegex(match.group(1), r"^[0-9;]*$")
        self.assertNotIn("\x1b[2J", raw, "a clear-screen reached the terminal")

    def test_an_api_error_message_cannot_inject(self):
        for attack in self.ATTACKS:
            self.out.truncate(0), self.out.seek(0)
            body = json.dumps({"type": "error", "error": {
                "type": "invalid_request_error", "message": f"bad {attack} thing"}}).encode()
            app = self.make_app(script=[lambda b=body: FakeStream(status=400, body=b)])
            app.start_session()
            app.send("hi")
            self._assert_clean()

    def test_a_session_title_cannot_inject(self):
        app = self.make_app()
        app.start_session(title="\x1b]0;PWNED\x07evil")
        app.dispatch("list", "")
        self._assert_clean()

    def test_pasted_user_text_cannot_inject(self):
        app = self.make_app()
        app.start_session()
        app._print_user_echo("hello \x1b]52;c;cHduZWQ=\x07 world")
        self._assert_clean()

    def test_ordinary_text_still_renders(self):
        app = self.make_app()
        app.start_session()
        app._print_user_echo("日本語 and ⚠️ and café")
        self.assertIn("日本語", self.text)
        self.assertIn("⚠️", self.text)


class TestRegistryHandlerParity(AppHarness):
    """Every declared command must have a handler; drift here crashes the loop."""

    def test_every_command_has_a_handler(self):
        missing = [c.name for c in cmds.COMMANDS
                   if not hasattr(App, "_cmd_" + c.name)]
        self.assertEqual(missing, [])

    def test_every_command_and_alias_dispatches_without_raising(self):
        for command in cmds.COMMANDS:
            for name in (command.name,) + tuple(command.aliases):
                app = self.make_app()
                app.start_session()
                try:
                    app.dispatch(name, "")
                except Exception as exc:            # noqa: BLE001 - that is the point
                    self.fail(f"/{name} raised {type(exc).__name__}: {exc}")

    def test_a_declared_command_without_a_handler_is_reported_not_raised(self):
        app = self.make_app()
        fake = cmds.Command(name="nonesuch", args="", help="", group="Session")
        original = cmds._INDEX.get("nonesuch")
        cmds._INDEX["nonesuch"] = fake
        try:
            app.dispatch("nonesuch", "")
        finally:
            if original is None:
                cmds._INDEX.pop("nonesuch", None)
            else:
                cmds._INDEX["nonesuch"] = original
        self.assertIn("not implemented", self.text)


class TestUsageAndUndo(AppHarness):
    def test_usage_reports_tokens_context_and_cost(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        self.out.truncate(0), self.out.seek(0)
        app.dispatch("usage", "")
        for expected in ("in 12", "out 7", "next prompt", "of 1,000,000 tokens", "cost $"):
            self.assertIn(expected, self.text)

    def test_usage_context_is_about_the_next_prompt_not_the_running_total(self):
        """Summing every turn and dividing by the current model was meaningless."""
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records("x" * 400))])])
        app.start_session()
        app.send("y" * 400)
        self.out.truncate(0), self.out.seek(0)
        app.dispatch("usage", "")
        first = re.search(r"next prompt ~([\d,]+)", self.text).group(1)
        # Switching models must reprice the same prompt, not a different number.
        app.dispatch("model", "haiku")
        self.out.truncate(0), self.out.seek(0)
        app.dispatch("usage", "")
        second = re.search(r"next prompt ~([\d,]+)", self.text).group(1)
        self.assertEqual(first, second)
        self.assertIn("of 200,000 tokens", self.text)

    def test_models_shows_the_price_actually_billed(self):
        app = self.make_app()
        app.dispatch("models", "")
        billed_in, billed_out = resolve_model("claude-sonnet-5").prices()
        self.assertIn(f"${billed_in:g}/${billed_out:g}", self.text)
        if (billed_in, billed_out) != (3.0, 15.0):
            self.assertIn("introductory", self.text)

    def test_models_reports_a_million_token_window_as_1M(self):
        app = self.make_app()
        app.dispatch("models", "")
        self.assertIn("1M", self.text)
        self.assertNotIn("1000K", self.text)

    def test_the_footer_never_overflows_a_narrow_terminal(self):
        for width in (30, 40, 46, 60, 80):
            app = self.make_app()
            app.console.caps = Caps(color=24, unicode=True, is_tty=True, width=width,
                                    height=24, hyperlinks=False, animation=False)
            app.start_session()
            self.out.truncate(0), self.out.seek(0)
            app.send("hi")
            for line in strip_ansi(self.out.getvalue()).split("\n"):
                self.assertLessEqual(len(line), width, f"width={width}: {line!r}")

    def test_turn_costs_agree_with_the_total_within_display_precision(self):
        """No fixed precision can make rounded turns sum exactly to a rounded
        total, so assert the real property: the drift stays inside the rounding."""
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records("a"))])])
        app.start_session()
        turn_count = 5
        for _ in range(turn_count):
            app.send("q")
        turns = [float(m) for m in re.findall(r"cost \$([\d.]+)", self.text)]
        totals = [float(m) for m in re.findall(r"total \$([\d.]+)", self.text)]
        self.assertEqual(len(turns), turn_count)
        self.assertLessEqual(abs(sum(turns) - totals[-1]), turn_count * 0.00005)

    def test_the_total_itself_is_exact(self):
        """Whatever the display does, the accumulated figure must be right."""
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records("a"))])])
        app.start_session()
        for _ in range(3):
            app.send("q")
        one_turn = Usage(input_tokens=12, output_tokens=7).cost("claude-opus-5")
        self.assertAlmostEqual(app.cost, one_turn * 3, places=12)

    def test_undo_drops_the_last_exchange_from_what_is_sent(self):
        app = self.make_app(script=[
            lambda: FakeStream(chunks=[sse(reply_records("one"))]),
            lambda: FakeStream(chunks=[sse(reply_records("two"))]),
        ])
        app.start_session()
        app.send("first question")
        app.dispatch("undo", "")
        app.send("second question")
        sent = self.transport.calls[-1]["body"]["messages"]
        self.assertEqual(sent, [{"role": "user", "content": "second question"}])

    def test_undo_keeps_the_saved_transcript(self):
        app = self.make_app()
        app.start_session()
        app.send("remember me")
        app.dispatch("undo", "")
        stored = [m.content for m in app.store.load(app.session.id)[1]]
        self.assertIn("remember me", stored)
        self.assertIn("still has them", self.text)

    def test_undo_several_exchanges(self):
        app = self.make_app(script=[lambda: FakeStream(chunks=[sse(reply_records("r"))])])
        app.start_session()
        for text in ("one", "two", "three"):
            app.send(text)
        app.dispatch("undo", "2")
        self.assertEqual([m.content for m in app.session.messages
                          if m.role == "user"], ["one"])

    def test_undo_with_nothing_to_undo(self):
        app = self.make_app()
        app.start_session()
        app.dispatch("undo", "")
        self.assertIn("Nothing to undo", self.text)

    def test_undo_rejects_nonsense(self):
        app = self.make_app()
        app.start_session()
        app.send("hi")
        app.dispatch("undo", "lots")
        self.assertIn("Usage:", self.text)


class TestCommandFailureIsolation(AppHarness):
    """A failing command reports; it never ends the conversation."""

    def test_a_storage_error_is_reported_not_raised(self):
        app = self.make_app()
        app.start_session()

        def boom(*a, **k):
            raise OSError("disk is full")

        app.store.update = boom
        app.dispatch("system", "Be brief.")
        self.assertIn("not saved", self.text)
        self.assertEqual(app.session.system, "Be brief.")

    def test_an_unexpected_bug_in_a_command_does_not_kill_the_loop(self):
        app = self.make_app(stdin="/list\n/quit\n")

        def boom(*a, **k):
            raise ZeroDivisionError("bug")

        app.store.list = boom
        self.assertEqual(app.run(), 0)
        self.assertIn("failed unexpectedly", self.text)

    def test_keyboard_interrupt_still_propagates(self):
        app = self.make_app()

        def boom(*a, **k):
            raise KeyboardInterrupt

        app.store.list = boom
        with self.assertRaises(KeyboardInterrupt):
            app.dispatch("list", "")


class TestReplyInjection(AppHarness):
    """The reply stream bypasses say(); it needs its own filter."""

    PAYLOAD = ("Here is some text \x1b]52;c;cHduZWQ=\x07 and a fence:\n\n"
               "```\n\x1b[2J\x1b[H\x1b[?1049h\n```\n\nand \x9b31m done.")

    def _assert_clean(self):
        raw = self.out.getvalue()
        self.assertNotIn("\x1b]", raw, "an OSC sequence reached the terminal")
        self.assertNotIn("\x1b[?", raw, "a private-mode sequence reached the terminal")
        self.assertNotIn("\x1b[2J", raw, "a clear-screen reached the terminal")
        self.assertNotIn("\x9b", raw, "an 8-bit CSI reached the terminal")
        self.assertNotIn("\r", raw, "a carriage return reached the terminal")

    def test_a_streamed_reply_cannot_drive_the_terminal(self):
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records(self.PAYLOAD))])])
        app.start_session()
        app.send("hi")
        self._assert_clean()

    def test_a_reply_split_across_chunks_cannot_drive_the_terminal(self):
        """An escape split across two deltas must not reassemble."""
        records = reply_records("")[:2] + [
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": part}})
            for part in ("safe \x1b", "]52;c;cHduZWQ=\x07 tail")
        ] + reply_records("")[3:]
        app = self.make_app(script=[lambda: FakeStream(chunks=[sse(records)])])
        app.start_session()
        app.send("hi")
        self._assert_clean()

    def test_a_replayed_transcript_cannot_drive_the_terminal(self):
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records(self.PAYLOAD))])])
        app.start_session()
        app.send("hi")
        session_id = app.session.id
        self.out.truncate(0), self.out.seek(0)
        app.dispatch("resume", session_id)
        self._assert_clean()

    def test_the_reply_text_itself_still_arrives(self):
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records(self.PAYLOAD))])])
        app.start_session()
        app.send("hi")
        self.assertIn("Here is some text", self.text)
        self.assertIn("done.", self.text)

    def test_the_stored_transcript_keeps_what_the_model_sent(self):
        """Sanitising is a display concern; the record should stay faithful."""
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records(self.PAYLOAD))])])
        app.start_session()
        app.send("hi")
        self.assertEqual(app.store.load(app.session.id)[1][-1].content, self.PAYLOAD)


class TestCliFlags(unittest.TestCase):
    """The command line, exercised without starting a conversation."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cfg = Path(self.dir.name) / "config.json"
        self.env = {"LUME_CONFIG": str(self.cfg), "LUME_HOME": self.dir.name,
                    "TERM": "xterm-256color", "PATH": os.environ.get("PATH", "")}

    def tearDown(self):
        self.dir.cleanup()

    def _run(self, *argv):
        """Run the CLI, swallowing its stdout so the suite stays readable."""
        import contextlib
        from lume.cli import main
        with contextlib.redirect_stdout(io.StringIO()):
            return main(list(argv), env=self.env)

    def test_save_config_keeps_real_preferences(self):
        self.assertEqual(self._run("-m", "haiku", "--theme", "ember", "--save-config"), 0)
        saved = json.loads(self.cfg.read_text())
        self.assertEqual(saved["model"], "claude-haiku-4-5")
        self.assertEqual(saved["theme"], "ember")

    def test_save_config_does_not_bake_in_per_run_presentation_flags(self):
        """--plain describes one run's output; making it permanent is a trap."""
        self._run("--plain", "--save-config")
        saved = json.loads(self.cfg.read_text())
        self.assertTrue(saved["animation"])

    def test_config_prints_the_path(self):
        self.assertEqual(self._run("--config"), 0)

    def test_list_on_an_empty_store(self):
        self.assertEqual(self._run("--list"), 0)

    def test_resume_an_unknown_reference_fails_cleanly(self):
        self.assertEqual(self._run("--resume", "nope-nope-nope"), 1)


class TestPipedOutput(AppHarness):
    """Into a pipe, lume is a filter: what the model wrote is what you get."""

    MARKDOWN = ("Fix the parser crash\n\n"
                "- guard the empty case\n- add a **regression** test\n\n"
                "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
                "```python\nx = 1\n```\n")

    def piped(self, **cfg):
        app = self.make_app(tty=False, color=0, theme="plain", **cfg)
        app.start_session()
        app.send("go")
        return self.out.getvalue()

    def test_output_is_exactly_what_the_model_wrote(self):
        app = self.make_app(tty=False, color=0, theme="plain",
                            script=[lambda: FakeStream(
                                chunks=[sse(reply_records(self.MARKDOWN))])])
        app.start_session()
        app.send("go")
        self.assertEqual(self.out.getvalue().strip(), self.MARKDOWN.strip())

    def test_no_escapes_reach_a_pipe(self):
        self.assertNotIn("\x1b", self.piped())

    def test_no_box_drawing_reaches_a_pipe(self):
        app = self.make_app(tty=False, color=0, theme="plain",
                            script=[lambda: FakeStream(
                                chunks=[sse(reply_records(self.MARKDOWN))])])
        app.start_session()
        app.send("go")
        for glyph in "╭╮╰╯│─┬┼┤├•":
            self.assertNotIn(glyph, self.out.getvalue(), glyph)

    def test_no_chrome_reaches_a_pipe(self):
        text = self.piped()
        self.assertNotIn("lume", text)
        self.assertNotIn("cost", text)
        self.assertNotIn("thinking", text)

    def test_a_terminal_still_gets_the_rendered_version(self):
        app = self.make_app(script=[lambda: FakeStream(
            chunks=[sse(reply_records(self.MARKDOWN))])])
        app.start_session()
        app.send("go")
        rendered = self.text
        self.assertNotIn("**regression**", rendered)
        self.assertIn("│", rendered)

    def test_a_replayed_transcript_is_also_verbatim_in_a_pipe(self):
        app = self.make_app(tty=False, color=0, theme="plain",
                            script=[lambda: FakeStream(
                                chunks=[sse(reply_records(self.MARKDOWN))])])
        app.start_session()
        app.send("go")
        session_id = app.session.id
        self.out.truncate(0), self.out.seek(0)
        app.reopen(session_id)
        self.assertIn(self.MARKDOWN.strip(), self.out.getvalue())


class TestResumeIsHonest(AppHarness):
    """Launching lume must not create a conversation, and 'last' must mean one."""

    def test_a_session_is_not_created_until_there_is_something_to_save(self):
        app = self.make_app(stdin="/quit\n")
        app.run()
        self.assertEqual(app.store.list(), [])

    def test_an_empty_session_is_never_what_last_resumes(self):
        app = self.make_app()
        app.start_session(title="Real")
        app.send("a real question")
        real = app.session.id
        app.dispatch("new", "Empty shell")
        app.dispatch("resume", "last")
        self.assertEqual(app.session.id, real)

    def test_resume_from_the_command_line_shows_what_it_restored(self):
        app = self.make_app()
        app.start_session(title="Prior")
        app.send("earlier question")
        session_id = app.session.id
        self.out.truncate(0), self.out.seek(0)
        app.reopen(session_id)
        self.assertIn("earlier question", self.text)
        self.assertIn("Prior", self.text)


def thinking_records(text, thinking, model="claude-opus-5"):
    """A reply that thinks first, which every current model does by default."""
    return [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_t", "type": "message", "role": "assistant", "model": model,
            "content": [], "usage": {"input_tokens": 100, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "thinking", "thinking": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "thinking_delta", "thinking": thinking}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 50}}),
        ("message_stop", {"type": "message_stop"}),
    ]


class TestThinkingNeverReachesAPipe(AppHarness):
    """Chain of thought is a terminal affordance, not part of the answer."""

    THINK = "The user wants a commit message. I should be terse."
    ANSWER = "Fix the parser crash on empty input"

    def _run(self, tty):
        app = self.make_app(tty=tty, color=24 if tty else 0,
                            theme="aurora" if tty else "plain",
                            script=[lambda: FakeStream(chunks=[sse(
                                thinking_records(self.ANSWER, self.THINK))])])
        app.start_session()
        app.send("write a commit message")
        return self.out.getvalue()

    def test_a_pipe_gets_the_answer_and_nothing_else(self):
        out = self._run(tty=False)
        self.assertNotIn(self.THINK, out)
        self.assertEqual(out.strip(), self.ANSWER)

    def test_a_terminal_still_shows_the_thinking(self):
        out = strip_ansi(self._run(tty=True))
        self.assertIn(self.THINK, out)
        self.assertIn("thinking", out)

    def test_the_thinking_is_still_recorded(self):
        app = self.make_app(tty=False, color=0, theme="plain",
                            script=[lambda: FakeStream(chunks=[sse(
                                thinking_records(self.ANSWER, self.THINK))])])
        app.start_session()
        app.send("go")
        self.assertEqual(app.store.load(app.session.id)[1][-1].thinking, self.THINK)


class TestNoSessionIsSafe(AppHarness):
    """Right after launch there is no session; no command may crash there."""

    def test_every_command_survives_having_no_session(self):
        for command in cmds.COMMANDS:
            for name in (command.name,) + tuple(command.aliases):
                app = self.make_app()
                self.assertIsNone(app.session)
                try:
                    app.dispatch(name, "")
                except Exception as exc:                      # noqa: BLE001
                    self.fail(f"/{name} raised {type(exc).__name__}: {exc}")
                self.assertNotIn("failed unexpectedly", self.text, f"/{name}")

    def test_usage_reports_zero_rather_than_crashing(self):
        app = self.make_app()
        app.dispatch("usage", "")
        self.assertIn("next prompt ~0", self.text)


class TestResumeRespectsTheCommandLine(AppHarness):
    """`--resume last -m haiku` must not silently bill at opus rates."""

    def _saved(self):
        app = self.make_app()
        app.start_session(title="Earlier")
        app.send("q")
        return app.session.id

    def test_a_model_named_on_the_command_line_wins(self):
        session_id = self._saved()
        app = self.make_app()
        app.config.model = "claude-haiku-4-5"
        app._model_pinned = True
        app.reopen(session_id)
        self.assertEqual(app.config.model, "claude-haiku-4-5")

    def test_without_the_flag_the_session_model_is_restored(self):
        session_id = self._saved()
        app = self.make_app()
        app.config.model = "claude-haiku-4-5"
        app.reopen(session_id)
        self.assertEqual(app.config.model, "claude-opus-5")

    def test_a_system_prompt_from_the_command_line_is_actually_sent(self):
        session_id = self._saved()
        app = self.make_app()
        app.config.system = "YOU ARE A PIRATE"
        app.reopen(session_id)
        app.send("ahoy")
        body = self.transport.calls[-1]["body"]
        self.assertIn("PIRATE", json.dumps(body.get("system")))


class TestCancelledTurnsAreStillBilled(AppHarness):
    def test_input_tokens_are_kept_when_a_turn_is_cut_short(self):
        """They are billed the moment the request is accepted."""
        class Truncated:
            status, reason, headers = 200, "OK", {}

            def __iter__(self):
                yield sse(thinking_records("partial", "t")[:2])
                raise OSError("connection reset by peer")

            def close(self):
                pass

        app = self.make_app(script=[lambda: Truncated()])
        app.start_session()
        app.send("go")
        self.assertGreater(app.usage.input_tokens, 0)


class TestErrorsReadLikeEnglish(AppHarness):
    def test_an_unknown_reference_is_not_a_python_repr(self):
        app = self.make_app()
        app.dispatch("resume", "zzzz")
        self.assertNotIn("'zzzz'", self.text.strip())
        self.assertIn("No such conversation", self.text)

    def test_rename_with_a_numeric_reference_that_matches_nothing_says_so(self):
        app = self.make_app()
        app.start_session(title="Only")
        app.send("q")
        self.out.truncate(0), self.out.seek(0)
        app.dispatch("rename", "99 x")
        self.assertIn("No conversation matches", self.text)
        self.assertEqual(app.store.load(app.session.id)[0].title, "Only")


class TestConfigWarningsAreVisible(AppHarness):
    def test_a_bad_model_is_reported_in_one_shot_mode(self):
        app = self.make_app()
        app.config.warnings.append("unknown model 'nope'; using the default")
        app.report_warnings()
        self.assertIn("unknown model", self.text)

    def test_warnings_are_only_said_once(self):
        app = self.make_app()
        app.config.warnings.append("something")
        app.report_warnings()
        app.report_warnings()
        self.assertEqual(self.text.count("something"), 1)

if __name__ == "__main__":
    unittest.main()
