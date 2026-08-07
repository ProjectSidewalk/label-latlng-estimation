"""Run the #5 comparison: est7's per-zoom linear heading fits vs the exact POV inversion.

Usage (from the repo root):
    python python/run_pov_inversion.py                # print the comparison
    python python/run_pov_inversion.py --write        # also write data/pov-inversion-summary.json

Everything is offline and deterministic: committed CSVs in, the R-exported train/test split,
zero network. The distance half of est7 is fit exactly as in the 2021 analysis; only the
heading half is swapped, so every reported difference is the heading swap alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    CITIES, add_heading_diff, clean_data, fit_models, load_city, split_from_fixtures,
)
from pov_inversion import (  # noqa: E402
    MODEL_NAMES, fidelity_report, score_heading_swap, summarize_heading_swap,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--fixtures-dir", default=os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    ap.add_argument("--write", action="store_true",
                    help="write data/pov-inversion-summary.json for the findings tests")
    args = ap.parse_args()

    print("Fidelity replay of evolution 179 vs stored current_pano_x/y, per city:")
    frames = []
    fidelity = {}
    for city in CITIES:
        df = load_city(args.data_dir, city)
        frames.append(df)
        fidelity[city] = fidelity_report(df)
        f = fidelity[city]
        if f["n_with_current_pano_xy"] == 0:
            print(f"  {city:<11} no current_pano_x/y (evolution 179 never ran)")
        else:
            print(f"  {city:<11} n={f['n_with_current_pano_xy']:>7}  "
                  f"y exact {f['pano_y_exact_match_rate']:.4f}  "
                  f"x exact {f['pano_x_exact_match_rate']:.4f} "
                  f"(pre-cutoff {f['pano_x_exact_match_rate_pre_cutoff']:.4f} / "
                  f"post {f['pano_x_exact_match_rate_post_cutoff']:.4f})")

    cleaned, _ = clean_data(pd.concat(frames, ignore_index=True))
    cleaned = add_heading_diff(cleaned)
    train, test = split_from_fixtures(cleaned, args.fixtures_dir)
    models = fit_models(train, include_est6=False)

    scored = score_heading_swap(models, train, test)
    summary = summarize_heading_swap(scored)
    summary["fidelity"] = fidelity

    names = {"est7": "est7 linear fits (6 params)", "exact": "exact inversion (0 params)",
             "era": "era-faithful exact (0 params)", "era_cal": "era + 1 global const"}
    hm = summary["heading_error_median_deg"]
    lm = summary["latlng_error_median_m"]
    print(f"\nTest set (n={summary['n_test']}); the fitted constant is "
          f"{summary['era_cal_delta_deg']:.4f} deg "
          f"(one depth-grid column = {summary['depth_grid_column_deg']:.4f} deg):")
    print(f"  {'model':<30}{'heading err med (deg)':>22}{'latlng err med (m)':>20}")
    for m in MODEL_NAMES:
        print(f"  {names[m]:<30}{hm[m]:>22.4f}{lm[m]:>20.4f}")

    print("\nBy zoom (heading error median, deg):")
    for z, v in sorted(summary["heading_error_median_deg_by_zoom"].items()):
        print(f"  zoom {z}: " + "  ".join(f"{m} {v[m]:.4f}" for m in MODEL_NAMES))

    print("\nBy |canvas_x - 360| (where the linear approximation loses):")
    for row in summary["by_canvas_x_offset"]:
        h = row["heading_error_median_deg"]
        print(f"  {row['bin']:<16} n={row['n']:>6}  " +
              "  ".join(f"{m} {h[m]:.4f}" for m in MODEL_NAMES))

    print("\nBy |pitch|:")
    for row in summary["by_abs_pitch"]:
        h = row["heading_error_median_deg"]
        print(f"  {row['bin']:<16} n={row['n']:>6}  " +
              "  ".join(f"{m} {h[m]:.4f}" for m in MODEL_NAMES))

    ppc = summary["photographer_pitch_residual_check"]
    print(f"\nphotographer_pitch residual check: r={ppc['pearson_r']:.4f}, "
          f"slope {ppc['slope_deg_per_deg']:.5f} deg/deg "
          f"(photographer_pitch p5..p95: {ppc['photographer_pitch_p5_p95'][0]:.2f}"
          f"..{ppc['photographer_pitch_p5_p95'][1]:.2f})")

    if args.write:
        out = os.path.join(args.data_dir, "pov-inversion-summary.json")
        # newline="\n" or a Windows rerun writes CRLF and stops matching a fresh checkout
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
