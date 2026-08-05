"""Run the ported analysis on the committed reconstructed data and print the comparison table.

Usage (from the repo root):
    python python/run_analysis.py                 # R-fixture split if available, else random
    python python/run_analysis.py --no-fixtures   # force the numpy random split

Prints the seven-estimator error table (the Rmd's headline comparison), the winning est7
coefficients next to the published 2021 values, and, when the R baseline fixtures are present,
the maximum relative deviation from the R run as a quick cross-language check (the rigorous
version of that check lives in tests/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import run_analysis  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PUBLISHED_2021 = {  # from the frozen scripts/label-latlng-estimation.md output
    "dist": [
        {"(Intercept)": 18.6051843, "sv_image_y": 0.0138947, "canvas_y": 0.0011023},
        {"(Intercept)": 20.8794248, "sv_image_y": 0.0184087, "canvas_y": 0.0022135},
        {"(Intercept)": 25.2472682, "sv_image_y": 0.0264216, "canvas_y": 0.0011071},
    ],
    "heading": [
        {"(Intercept)": -51.2401711, "canvas_x": 0.1443374},
        {"(Intercept)": -27.5267447, "canvas_x": 0.0784357},
        {"(Intercept)": -13.5675945, "canvas_x": 0.0396061},
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--fixtures-dir", default=os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    ap.add_argument("--no-fixtures", action="store_true",
                    help="use the numpy random split instead of the R-exported split")
    ap.add_argument("--json-out", help="also dump the full results dict to this path")
    args = ap.parse_args()

    fixtures = None
    if not args.no_fixtures and os.path.exists(os.path.join(args.fixtures_dir, "split_train.csv.gz")):
        fixtures = args.fixtures_dir

    results = run_analysis(args.data_dir, fixtures)

    m = results["meta"]
    print(f"Rows: raw {m['rows_raw']} -> cleaned {m['rows_after_cleaning']} "
          f"(train {m['rows_train']} / test {m['rows_test']}, split: {m['split']})")

    print("\nTest-set error summary (m), sorted by median:")
    print(f"{'estimate':<12}{'mean':>8}{'median':>8}{'min':>10}{'max':>8}{'sd':>8}")
    for r in results["error_stats"]["summary"]:
        print(f"{r['estimate']:<12}{r['mean']:>8.3f}{r['median']:>8.3f}"
              f"{r['min']:>10.6f}{r['max']:>8.2f}{r['sd']:>8.3f}")

    print("\nest7 coefficients (this run vs published 2021):")
    for z in range(3):
        cd, ch = results["est7"]["dist"][z], results["est7"]["heading"][z]
        pd_, ph = PUBLISHED_2021["dist"][z], PUBLISHED_2021["heading"][z]
        print(f"zoom {z + 1}: dist {cd['(Intercept)']:.7f} + {cd['sv_image_y']:.7f}*sv_image_y"
              f" + {cd['canvas_y']:.7f}*canvas_y   (2021: {pd_['(Intercept)']:.7f}"
              f" + {pd_['sv_image_y']:.7f} + {pd_['canvas_y']:.7f})")
        print(f"        heading {ch['(Intercept)']:.7f} + {ch['canvas_x']:.7f}*canvas_x"
              f"           (2021: {ph['(Intercept)']:.7f} + {ph['canvas_x']:.7f})")

    if fixtures:
        with open(os.path.join(fixtures, "baseline.json"), encoding="utf-8") as f:
            baseline = json.load(f)
        worst = 0.0
        for z in range(3):
            for part in ("dist", "heading"):
                for k, v in baseline["est7"][part][z].items():
                    rel = abs(results["est7"][part][z][k] - v) / max(abs(v), 1e-12)
                    worst = max(worst, rel)
        print(f"\nMax relative deviation from R baseline (est7 coefficients): {worst:.2e}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull results written to {args.json_out}")


if __name__ == "__main__":
    main()
