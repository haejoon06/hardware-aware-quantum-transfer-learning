from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from article10.pipeline import run_cli  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["setup", "smoke", "screen", "ablate", "final", "shots", "noise", "transpile", "confirm", "report", "all"],
        default="all",
    )
    parser.add_argument("--dataset", choices=["cifar10", "pathmnist", "eurosat"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, help="Test-only epoch override.")
    args = parser.parse_args()
    run_cli(args)
