#!/usr/bin/env python3
"""Per-category top-line score reporter for a given model.

Usage:
    python score_split.py --model MODEL_NAME [--split {live,non-live,multi-turn}]

This is a thin wrapper around `bfcl_eval.score_split.report_scores`; the
same functionality is also available as `bfcl score ...`.
"""

import argparse
import sys
from pathlib import Path

from bfcl_eval.score_split import SPLIT_ALIASES, report_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model identifier as passed to bfcl evaluate")
    parser.add_argument(
        "--split",
        default=None,
        choices=sorted(SPLIT_ALIASES.keys()),
        help="If omitted, reports all available splits with a grand total.",
    )
    parser.add_argument(
        "--score-dir",
        default=str(Path(__file__).resolve().parent / "score"),
        help="Path to the score/ directory (default: ./score next to this script)",
    )
    args = parser.parse_args()
    return report_scores(args.model, args.split, Path(args.score_dir))


if __name__ == "__main__":
    sys.exit(main())
