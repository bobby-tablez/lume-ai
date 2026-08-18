"""Allow `python -m lume`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
