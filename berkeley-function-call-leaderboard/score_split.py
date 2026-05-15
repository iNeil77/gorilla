#!/usr/bin/env python3
"""Print per-category top-line scores for a given model.

Usage:
    python score_split.py --model MODEL_NAME [--split {live,non-live,multi-turn}]

If --split is omitted, all available splits under the model's score
directory are reported, followed by a grand total across them.

The model name is the same identifier passed to `bfcl evaluate` (e.g.
`yantri-tool/Qwen3-8B-FC`); the slash is converted to an underscore to
match the on-disk folder layout under `score/`.
"""

import argparse
import json
import sys
from pathlib import Path

SPLIT_ALIASES = {
    "live": "live",
    "non-live": "non_live",
    "non_live": "non_live",
    "nonlive": "non_live",
    "multi-turn": "multi_turn",
    "multi_turn": "multi_turn",
    "multiturn": "multi_turn",
}

DEFAULT_SPLIT_ORDER = ["non_live", "live", "multi_turn"]


def load_split_rows(split_dir: Path):
    score_files = sorted(split_dir.glob("*_score.json"))
    rows = []
    for score_file in score_files:
        with score_file.open() as f:
            top_line = json.loads(f.readline())
        category = score_file.stem.removesuffix("_score")
        for prefix in ("BFCL_v4_", "BFCL_v3_"):
            if category.startswith(prefix):
                category = category[len(prefix):]
                break
        rows.append((category, top_line["accuracy"], top_line["correct_count"], top_line["total_count"]))
    return rows


def print_split(split_label: str, split_dir_name: str, rows, name_w: int):
    print(f"=== {split_label} ({split_dir_name}) ===")
    print(f"{'category'.ljust(name_w)}  {'accuracy':>9}  {'correct':>8}  {'total':>6}")
    print(f"{'-' * name_w}  {'-' * 9}  {'-' * 8}  {'-' * 6}")

    total_correct = 0
    total_count = 0
    accuracy_sum = 0.0
    for category, accuracy, correct, total in rows:
        print(f"{category.ljust(name_w)}  {accuracy * 100:>8.2f}%  {correct:>8}  {total:>6}")
        total_correct += correct
        total_count += total
        accuracy_sum += accuracy

    print(f"{'-' * name_w}  {'-' * 9}  {'-' * 8}  {'-' * 6}")
    unweighted = accuracy_sum / len(rows)
    weighted = total_correct / total_count if total_count else 0.0
    print(f"{'unweighted mean'.ljust(name_w)}  {unweighted * 100:>8.2f}%")
    print(f"{'weighted (micro)'.ljust(name_w)}  {weighted * 100:>8.2f}%  {total_correct:>8}  {total_count:>6}")
    return unweighted, total_correct, total_count


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

    model_root = Path(args.score_dir) / args.model.replace("/", "_")
    if not model_root.is_dir():
        print(f"error: no scores found at {model_root}", file=sys.stderr)
        return 1

    if args.split is not None:
        split_dirs = [SPLIT_ALIASES[args.split]]
    else:
        split_dirs = [s for s in DEFAULT_SPLIT_ORDER if (model_root / s).is_dir()]
        if not split_dirs:
            print(f"error: no split subdirectories under {model_root}", file=sys.stderr)
            return 1

    splits_data = []
    for split_dir_name in split_dirs:
        split_dir = model_root / split_dir_name
        if not split_dir.is_dir():
            print(f"error: no scores found at {split_dir}", file=sys.stderr)
            return 1
        rows = load_split_rows(split_dir)
        if not rows:
            print(f"error: no *_score.json files in {split_dir}", file=sys.stderr)
            return 1
        splits_data.append((split_dir_name, rows))

    name_w = max(len(category) for _, rows in splits_data for category, *_ in rows)
    name_w = max(name_w, len("weighted (micro)"))

    print(f"Model:  {args.model}")
    print(f"Source: {model_root}")
    print()

    grand_correct = 0
    grand_count = 0
    split_unweighted = []
    for i, (split_dir_name, rows) in enumerate(splits_data):
        if i > 0:
            print()
        unweighted, correct, total = print_split(split_dir_name, split_dir_name, rows, name_w)
        split_unweighted.append(unweighted)
        grand_correct += correct
        grand_count += total

    if len(splits_data) > 1:
        print()
        print("=== overall (across reported splits) ===")
        print(f"{'-' * name_w}  {'-' * 9}  {'-' * 8}  {'-' * 6}")
        macro = sum(split_unweighted) / len(split_unweighted)
        micro = grand_correct / grand_count if grand_count else 0.0
        print(f"{'macro (mean of split unweighted means)'.ljust(name_w)}  {macro * 100:>8.2f}%")
        print(f"{'micro (entry-weighted across splits)'.ljust(name_w)}  {micro * 100:>8.2f}%  {grand_correct:>8}  {grand_count:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
