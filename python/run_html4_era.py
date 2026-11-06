"""Measure the html4-era DC rows (SidewalkWebpage#5920) in the committed data.

Usage (from the repo root):
    python python/run_html4_era.py            # print the summary
    python python/run_html4_era.py --write    # also write data/html4-era-summary.json

Offline and deterministic: the committed labels-dc-latlng.csv.gz in, one JSON out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html4_era import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--write", action="store_true", help="write data/html4-era-summary.json")
    args = ap.parse_args()
    summary = summarize(args.data_dir)
    print(json.dumps(summary, indent=2))
    if args.write:
        out = os.path.join(args.data_dir, "html4-era-summary.json")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
