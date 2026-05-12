#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from minimt3.eval.metrics import evaluate_directory
from minimt3.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted MIDI files against MAESTRO metadata.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--ref_meta", required=True)
    parser.add_argument("--out", default="outputs/logs/eval.json")
    args = parser.parse_args()
    results = evaluate_directory(args.pred_dir, args.ref_meta)
    write_json(args.out, results)
    print(f"Wrote evaluation to {Path(args.out)}")
    print(results.get("summary", {}))


if __name__ == "__main__":
    main()
