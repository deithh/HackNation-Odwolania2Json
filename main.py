"""Entry point: reads input/<case>/*.pdf and writes output/<case>/*.json.

Usage:
  python main.py                                  # run full pipeline on ./input -> ./output
  python main.py --debug                          # also save step-by-step audit artifacts in ./debug
  python main.py --input some/dir --output out    # custom folders
  python main.py types                            # list configured document types
  python main.py --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from odwolania.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args = ["run", *args]
    sys.exit(main(args))
