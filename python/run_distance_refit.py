"""Run the #3 distance-refit ladder: geometry-shaped candidates vs the 2021 distance half.

Usage (from the repo root):
    python python/run_distance_refit.py               # print the comparison
    python python/run_distance_refit.py --write       # also write data/distance-refit-summary.json

Everything is offline and deterministic: committed CSVs in, the R-exported train/test split,
zero network. The heading half is held identical across every rung (the era-faithful exact
inversion from #5 plus its one train-fitted constant) and the destination is turf-spherical,
so every difference between rungs is the distance half alone; the unmodified est7 pipeline is
kept as the continuity row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    add_heading_diff, clean_data, fit_models, load_data, split_from_fixtures,
)
import distance_refit as dr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--fixtures-dir", default=os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    ap.add_argument("--write", action="store_true",
                    help="write data/distance-refit-summary.json for the findings tests")
    args = ap.parse_args()
    t0 = time.time()

    def step(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("loading and cleaning (the exact 2021 pipeline)...")
    cleaned, _ = clean_data(load_data(args.data_dir))
    cleaned = dr.add_depression(add_heading_diff(cleaned))
    train, test = split_from_fixtures(cleaned, args.fixtures_dir)
    models = fit_models(train, include_est6=False)

    step("fitting the ladder (both losses)...")
    fits = dr.fit_all_rungs(train, models, args.data_dir)
    chosen = dr.choose_candidate(fits, train)
    step(f"chosen on train: {chosen['rung']}")

    step("scoring on the published test split...")
    scored = dr.score_rungs(fits, models, train, test)

    step("candidate B and the #4765 apply path...")
    fixed_frame = dr.fixed_frame_check(cleaned)
    cand_b = dr.candidate_b_checks(train, test)
    apply_path = dr.apply_path_check(models, test)

    headline = ["est7", "A_ols", "anchor", "C_l1", chosen["rung"], "E_l1"]
    step("click-noise sweep...")
    noise = dr.noise_sweep(fits, models, train, test, keys=headline)

    step("riders and quantile bands...")
    riders = dr.rider_checks(scored, cleaned, fits, chosen["rung"], args.data_dir)
    quantiles = dr.quantile_bands(train, test)

    checks = {"meta": {"n_train": int(len(train))}, "fixed_frame": fixed_frame,
              "candidate_b": cand_b, "apply_path": apply_path, "noise": noise,
              "riders": riders, "quantiles": quantiles}
    summary = dr.build_summary(
        scored, fits, chosen, checks, keys=headline,
        near_horizon_keys=["est7", "A_ols", "anchor", "C_l1", "D_soft_l1", chosen["rung"],
                           "E_l1"])

    # ------------------------------------------------------------------ report to stdout
    m = summary["matrix"]
    c = summary["continuity"]
    print(f"\nContinuity: est7 legacy {c['est7_legacy_median_m']:.4f} m; under the shared "
          f"heading+spherical convention {c['est7_spherical_median_m']:.4f} m "
          f"(convention delta {c['scoring_convention_delta_m']:+.4f} m)")

    print(f"\nThe ladder (test n={summary['meta']['n_test']}; latlng under the shared "
          f"heading; params = distance-half only, est7 = full pipeline):")
    print(f"  {'rung':<18}{'params':>7}{'latlng med':>12}{'latlng p90':>12}"
          f"{'dist med':>10}{'dist p90':>10}{'n':>8}")
    for k in dr.KEYS:
        r = m[k]
        mark = " <- chosen" if k == chosen["rung"] else ""
        print(f"  {k:<18}{r['n_params']:>7}{r['latlng_median_m']:>12.4f}"
              f"{r['latlng_p90_m']:>12.4f}{r['dist_median_m']:>10.4f}"
              f"{r['dist_p90_m']:>10.4f}{r['n']:>8}{mark}")

    p = fixed_frame["pooled"]
    print(f"\nFixed-frame check: implied/exact depression ratio {p['ratio_median_6656']:.4f} "
          f"(6656 px) vs {p['ratio_median_8192']:.4f} (8192 px); a real-pixel frame would "
          f"put their ratio at {p['if_real_pixel_frame']:.3f}, measured "
          f"{p['ratio_8192_over_6656']:.4f}")

    ap_ = summary["candidate_b"]["apply_path"]
    print(f"\n#4765 apply path (n={ap_['n']} test rows with current_pano_y):")
    for v in ("raw", "normalized"):
        r = ap_[v]
        print(f"  {v:<11} dist med {r['dist_median_m']:.4f} m; signed bias "
              f"h6656 {r['h6656']['signed_median_m']:+.4f} m (n={r['h6656']['n']}), "
              f"h8192 {r['h8192']['signed_median_m']:+.4f} m (n={r['h8192']['n']})")

    print("\nNear-horizon bins (latlng med / p95 / max; max predicted dist in that bin):")
    for row in summary["near_horizon"]:
        print(f"  dep {row['bin_deg']:<14} n={row['n']:>6}")
        for k, v in row["per_rung"].items():
            print(f"    {k:<16}{v['latlng_median_m']:>9.2f}{v['latlng_p95_m']:>9.2f}"
                  f"{v['latlng_max_m']:>9.2f}   pred<= {v['dist_pred_max_m']:.1f} m")

    print("\nStructural bounds (largest answer each form can EVER return, swept over the full "
          "depression domain):")
    for k, v in summary["bounds"].items():
        if v is not None:
            print(f"  {k:<18}{v:>8.2f} m" + ("  <- the 50 m cap, not saturation"
                                             if v >= dr.DIST_CAP_M - 1e-9 else ""))

    print("\nClick-noise sweep (delta median latlng error vs unperturbed, m):")
    print(f"  {'rung':<18}" + "".join(f"{s:>10}" for s in noise["sigmas_px"]))
    for k in headline:
        row = noise["per_rung"][k]
        print(f"  {k:<18}" + "".join(f"{row[str(s)]['delta_median_m']:>10.4f}"
                                     for s in noise["sigmas_px"]))

    r = riders
    print(f"\nRiders: photographer_pitch r={r['photographer_pitch']['pearson_r']:+.4f} "
          f"({r['photographer_pitch']['slope_m_per_deg']:+.4f} m/deg); "
          f"tilt sinusoid n={r['tilt_sinusoid']['n']} "
          f"coef={r['tilt_sinusoid'].get('coef_deg_per_deg')} "
          f"r={r['tilt_sinusoid'].get('pearson_r')}; camera height fitted "
          f"{r['camera_height']['fitted_C_l1_m']:.3f} m (L1) / "
          f"{r['camera_height']['fitted_C_ols_m']:.3f} m (OLS) vs served median "
          f"{r['camera_height']['served_median_m_excl_pin']:.3f} m "
          f"(n={r['camera_height']['n_served']})")

    q = quantiles
    print(f"Quantile bands (tau 0.1/0.9 in disparity space): interval width median "
          f"{q['interval_width_median_m']:.2f} m, p90 {q['interval_width_p90_m']:.2f} m")

    zr = summary["zoom_residual_chosen"]
    print("Zoom residual of the chosen rung (signed median, m): "
          + "  ".join(f"z{z} {v['signed_median_m']:+.4f}" for z, v in sorted(zr.items())))

    if args.write:
        out = os.path.join(args.data_dir, "distance-refit-summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nSummary written to {out}")
    step("done")


if __name__ == "__main__":
    main()
