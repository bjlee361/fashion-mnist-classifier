"""Summarize every trained run as a Markdown table (paste it into the README).

Example:
    python compare.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    runs = sorted(Path(args.out_dir).glob("*/metrics.json"))
    if not runs:
        raise SystemExit(f"No runs found in {args.out_dir}/. Train a model first with train.py.")

    print("| Model | Parameters | Best epoch | Validation acc. | Test acc. |")
    print("|---|---:|---:|---:|---:|")
    for path in runs:
        m = json.loads(path.read_text())
        print(f"| {m['model'].upper()} | {m['parameters']:,} | {m['best_epoch']} | "
              f"{100 * m['val_acc']:.2f}% | {100 * m['test_acc']:.2f}% |")


if __name__ == "__main__":
    main()
