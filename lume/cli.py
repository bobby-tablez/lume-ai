"""Command-line entry point: argument parsing, then hand over to `App`."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .ansi import Console, detect_caps, install_signal_net
from .config import Config, config_path, load_config, save_config
from .theme import theme_names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lume",
        description="A small, beautiful terminal chat client for Claude, GPT and Gemini.",
        epilog="With no arguments, lume opens an interactive chat. "
               "Give it a message to get a single answer and exit.",
    )
    p.add_argument("message", nargs="*", help="ask one question and exit")
    p.add_argument("-m", "--model", help="model id or alias (opus, sonnet, haiku, fable, gpt, gemini)")
    p.add_argument("-r", "--resume", nargs="?", const="last", metavar="REF",
                   help="reopen a saved conversation (number, id, or 'last')")
    p.add_argument("-l", "--list", action="store_true", help="list saved conversations and exit")
    p.add_argument("-s", "--system", metavar="TEXT", help="system prompt for this run")
    p.add_argument("--theme", choices=tuple(theme_names()) + ("auto",),
                   help="colour theme (default: auto, from the terminal background)")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"),
                   help="how hard the model should think")
    p.add_argument("--max-tokens", type=int, metavar="N", help="cap on reply length")
    p.add_argument("--no-thinking", action="store_true", help="turn extended thinking off")
    p.add_argument("--no-cache", action="store_true", help="disable prompt caching")
    p.add_argument("--no-motion", action="store_true", help="disable animation")
    p.add_argument("--no-color", action="store_true", help="disable colour entirely")
    p.add_argument("--plain", action="store_true",
                   help="no colour, no motion, no banner, no styling escapes")
    p.add_argument("--config", action="store_true", help="show the config file path and exit")
    p.add_argument("--save-config", action="store_true",
                   help="save the current settings as defaults and exit "
                        "(--plain/--no-color/--no-motion are per-run and are not saved)")
    p.add_argument("-V", "--version", action="version", version=f"lume {__version__}")
    return p


def main(argv=None, env=None) -> int:
    env = dict(os.environ if env is None else env)
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    if args.plain:
        args.no_color = args.no_motion = True

    config = load_config(env=env)
    # Preferences that are worth remembering.
    config.apply_overrides(
        model=args.model,
        theme=args.theme,
        effort=args.effort,
        max_tokens=args.max_tokens,
        system=args.system,
        thinking=False if args.no_thinking else None,
        cache=False if args.no_cache else None,
    )

    if args.config:
        print(config_path(env))
        return 0
    if args.save_config:
        # Saved before the presentation flags are applied: --plain describes one
        # run's output, and silently making it permanent would be a trap.
        print(f"wrote {save_config(config, env=env)}")
        return 0

    # Presentation flags: this run only.
    if args.no_color:
        env["NO_COLOR"] = "1"
    if args.no_motion:
        env["LUME_NO_MOTION"] = "1"
        config.apply_overrides(animation=False)
    if args.plain:
        # Escape-free styling, and no wordmark: --plain is for pipes and logs.
        config.apply_overrides(theme="plain")

    console = Console(caps=detect_caps(sys.stdout, env))
    install_signal_net()      # explicit, never an import side effect

    from .app import App, _reason             # imported late: keeps --version fast
    app = App(config, console=console, env=env)

    try:
        if args.list:
            app.list_sessions()
            return 0
        if args.resume:
            app._model_pinned = args.model is not None
            try:
                app.greet()          # banner first, then what was restored
                app.reopen(args.resume)
            except (KeyError, LookupError, ValueError) as exc:
                app.fail(_reason(exc, "No such conversation."))
                return 1
        app.report_warnings()      # greet() never runs in one-shot mode
        message = " ".join(args.message).strip()
        if not message and not sys.stdin.isatty():
            piped = sys.stdin.read().strip()
            message = f"{message}\n\n{piped}".strip() if message else piped
        if message:
            # One-shot mode: answer and leave, so lume composes with other tools.
            if app.session is None:
                app.start_session()
            return 0 if app.send(message) else 1
        return app.run()
    except KeyboardInterrupt:
        console.print()
        return 130
    finally:
        app.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
