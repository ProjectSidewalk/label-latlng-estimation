"""Score the frozen #6 boosters on modern measured-plane truth: does the ceiling transfer?

Usage (from the repo root):
    python python/run_gbm_transfer.py            # print the comparison
    python python/run_gbm_transfer.py --write    # also write data/gbm-transfer-summary.json

The boosters are refitted in-process by ``run_gbm_ceiling.fit_gbm`` -- the same code, same
seed, same two-pass early stopping -- and then REQUIRED to reproduce the committed
``data/gbm-ceiling-summary.json`` era-test medians to float precision before a single
modern row is scored. That assertion is what makes "the #6 model" a meaningful phrase
here: if it fails, this run is not benchmarking the model the report describes.

Nothing is refitted on modern data. The one parameter this runner ever fits is a global
scale per booster, on a train half of panoramas, scored on the disjoint half -- the same
treatment (and the same split) ``modern_truth.remedy_check`` gave the blend, so the
held-out table's closed-form rows are asserted equal to the committed
``modern-truth-summary.json`` remedies block.

Offline, and this is the complete input list: the era CSVs, the R-fixture split,
``data/modern-truth-labels.csv.gz``, ``distance-refit-summary.json`` (the era blend
coefficients, via ``modern_truth.load_blend_params``), ``gbm-ceiling-summary.json`` and
``modern-truth-summary.json``. No network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    add_heading_diff, clean_data, load_data, split_from_fixtures,
)
import distance_refit as dr  # noqa: E402
import gbm_transfer as gt  # noqa: E402
import modern_truth as mt  # noqa: E402
import run_gbm_ceiling as gc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The four boosters this test needs: the headline model, the era-best variant, the l2
# objective, and the single-signal control that landed on top of the blend in #6.
VARIANTS = {
    "gbm_l1": (gc.FEATURES_FULL, "regression_l1"),
    "gbm_dep_l1": (gc.FEATURES_FULL + ["depression_deg"], "regression_l1"),
    "gbm_l2": (gc.FEATURES_FULL, "regression_l2"),
    "only_sv_image_y": (["sv_image_y"], "regression_l1"),
}
SCALED_KEYS = [f"{k}_scaled" for k in VARIANTS]

# The models handed the two richer recalibrations, and the recalibrations themselves. Every
# key roster below is derived from these three lines, so an arm cannot be added to the run
# and then quietly missed by a scoring table, a bootstrap or the best-arm search.
RECAL_MODELS = ["gbm_l1", "gbm_dep_l1", "D_flat"]
RECAL_FORMS = ["affine", "quantile"]
RECAL_KEYS = {m: [f"{m}_{f}" for f in RECAL_FORMS] for m in RECAL_MODELS}
# Every booster arm that got a modern parameter: one scale each, then the richer maps, then
# the modern-trained control. This is the set "the best recalibrated GBM" is chosen from.
RECALIBRATED_GBM_KEYS = (SCALED_KEYS
                         + [k for m in RECAL_MODELS if m.startswith("gbm")
                            for k in RECAL_KEYS[m]]
                         + ["gbm_modern"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--fixtures-dir",
                    default=os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    ap.add_argument("--write", action="store_true",
                    help="write data/gbm-transfer-summary.json for the findings tests")
    args = ap.parse_args()
    t0 = time.time()

    def step(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    # ------------------------------------------------------------------ the frozen models
    step("loading the era split and refitting the #6 boosters...")
    cleaned, _ = clean_data(load_data(args.data_dir))
    cleaned = dr.add_depression(add_heading_diff(cleaned))
    era_train, era_test = split_from_fixtures(cleaned, args.fixtures_dir)

    gbms = {}
    for key, (cols, objective) in VARIANTS.items():
        gbms[key] = gc.fit_gbm(era_train, cols, objective)
        step(f"  {key}: {gbms[key]['best_iteration']} rounds")

    step("asserting the boosters reproduce the committed #6 era-test numbers...")
    era_scored = gc.score_models(gbms, {}, era_train, era_test)
    era_matrix = {k: gc.metrics(era_scored, k) for k in gbms}
    with open(os.path.join(args.data_dir, "gbm-ceiling-summary.json"), encoding="utf-8") as f:
        ceiling = json.load(f)
    for key in gbms:
        for metric in ("latlng_median_m", "latlng_p90_m", "dist_median_m", "dist_p90_m"):
            got, want = era_matrix[key][metric], ceiling["matrix"][key][metric]
            assert abs(got - want) < 1e-9, (key, metric, got, want)
    era_reference = {
        "source": "data/gbm-ceiling-summary.json (reproduced in-process, asserted equal)",
        "n_test": ceiling["meta"]["n_test"],
        "models": {k: {m: ceiling["matrix"][k][m]
                       for m in ("dist_median_m", "dist_p90_m", "latlng_median_m")}
                   for k in list(VARIANTS) + ["A_ols", "D_blend_type_l1"]},
        "d_over_gbm_gap_pct": ceiling["ceiling"]["d_over_gbm_gap_pct"],
    }

    # ------------------------------------------------------------------ the modern rows
    step("loading modern truth and mapping it into the era feature frame...")
    labels = gt.load_modern_labels(args.data_dir)
    human = gt.to_era_frame(gt.gated_human(labels))
    truth = human["truth_m"].to_numpy(float)
    pano = human["pano_id"].to_numpy()

    preds = pd.concat([gt.closed_form_predictions(human, args.data_dir),
                       gt.gbm_predictions(gbms, human)], axis=1)

    population = {
        "n_gated_human": int(len(human)),
        "n_panos": int(human["pano_id"].nunique()),
        "n_cities": int(human["city"].nunique()),
        "time_created_range": [str(human["time_created"].min()),
                               str(human["time_created"].max())],
        "by_era_column": {str(k): int(v) for k, v in human["era"].value_counts().items()},
        "truth_median_m": float(np.median(truth)),
        "era_train_pano_dist_median_m": float(era_train["pano_dist"].median()),
        "disjoint_from_era_split": "the era reconstruction stops at the 2021-01-01 cutoff "
                                   "and every row here is post-2021, so the boosters are "
                                   "fully held out on this population",
        "label_types": gt.label_type_census(human),
        "support_shift": gt.support_shift(era_train, human),
    }
    frame = gt.frame_mapping_evidence(human)

    # Why the booster arrives on modern truth almost unbiased while the era blend arrives
    # 1.07 m long: the era truth's own implied scale is not constant across resolutions.
    scale_diagnostic = {
        "era_truth": gt.implied_height_by_resolution(cleaned, "pano_dist", "depression_deg"),
        "modern_truth": gt.implied_height_by_resolution(human, "truth_m", "depression_deg"),
        "shipped_flat_height_m": gt.flat_params(args.data_dir)["height_m"],
        "reading": "the era fit pooled these into one per-type table, so the shipped era "
                   "blend answers on the pooled scale everywhere; a booster given "
                   "pano_height can answer on each subpopulation's own scale, which is "
                   "worth a great deal inside the era truth and nothing on a population "
                   "with a different resolution mix. Whether the split is a real rig "
                   "difference or a truth-pipeline artifact is NOT settled here -- #7 "
                   "section 6 measures 2.337-2.559 m across GSV cities by triangulation, "
                   "so both remain live",
    }

    # ------------------------------------------------------- pooled: no fitted parameter
    step("scoring the pooled population (every era-trained model, fully held out)...")
    pooled_keys = ["A_deployed", "D_blend"] + list(VARIANTS)
    pooled = gt.score_frame(preds, truth, pooled_keys + ["D_flat_shipped"])
    pooled_boot = gt.bootstrap_medians(preds, truth, pano, pooled_keys, reference="D_blend")

    # --------------------------------------------- held out: one modern parameter each
    step("fitting one scale per booster on a train half of panoramas...")
    in_train, in_test = gt.pano_half_split(human)
    dep = human["depression_deg"].to_numpy(float)
    scales = {k: gt.scale_factor(preds[k].to_numpy(float)[in_train], truth[in_train],
                                 dep[in_train]) for k in VARIANTS}
    for k, s in scales.items():
        preds[f"{k}_scaled"] = preds[k].to_numpy(float) * s

    # remedy_check's own remedies, from remedy_check itself rather than a second copy of
    # its arithmetic: same seed, same depression floor, so its train half IS the half the
    # scales above were fitted on, and the two parameters below are the ones the committed
    # Stage 4 table was built from. The assertions further down hold that to float precision.
    blend_params = mt.load_blend_params(args.data_dir)
    remedy = mt.remedy_check(labels, blend_params, seed=gt.SEED,
                             min_dep_deg=gt.SCALE_MIN_DEP_DEG)
    k_rescale, flat_h = remedy["k_rescale"], remedy["flat_height_m"]
    preds["D_rescaled"] = dr.predict_dist(
        mt.rescaled_blend_params(blend_params, k_rescale), human)
    preds["D_flat"] = dr.predict_dist(
        {"form": "blend", "blend_deg": blend_params["blend_deg"], "height_m": flat_h}, human)

    # Answering "you only gave the booster one parameter" before it is asked: the same two
    # richer recalibrations, fitted on the same train half, given to the ceiling candidates
    # AND to the shipped closed form, so every rung of generosity is like-for-like.
    step("richer recalibrations (affine, monotone quantile map) on the same train half...")
    recal = {}
    for key in RECAL_MODELS:
        p, t = preds[key].to_numpy(float), truth
        a, b = gt.affine_l1(p[in_train], t[in_train])
        fitted = {"affine": a + b * p,
                  "quantile": gt.quantile_map(p[in_train], t[in_train])(p)}
        for form in RECAL_FORMS:
            preds[f"{key}_{form}"] = fitted[form]
        recal[key] = {"affine_a": a, "affine_b": b}

    # The underpowered control a reader will ask for: a booster that sees modern truth
    # itself. 1,293 training rows against the era split's 316,118 — this is a FLOOR on what
    # modern data supports, not the modern ceiling, and it is labelled that way everywhere.
    #
    # It carries a SECOND handicap, and it is not the sample size: gc.fit_gbm builds its
    # features with gc.build_features, which pins label_type to the era's seven categories
    # (dr.LABEL_TYPES). Crosswalk and Signal — 433 rows, 16.3% of this population — are
    # therefore a missing category in this booster's own TRAINING data, not just at
    # prediction time as they are for the era boosters. Both handicaps push the same way
    # (a fairer control would score better, and this one already loses to the closed form),
    # so the reading is conservative; §6 of the report says so rather than leaving the
    # arm's weakness attributed entirely to its row count.
    step("control: a booster trained on the modern train half (underpowered by design)...")
    modern_fit = human.assign(pano_dist=truth)
    gbm_modern = gc.fit_gbm(modern_fit[in_train], gc.FEATURES_FULL, "regression_l1")
    preds["gbm_modern"] = gc.predict_gbm(gbm_modern, human)

    held_keys = (["A_deployed", "D_blend", "D_rescaled", "D_flat"] + RECAL_KEYS["D_flat"]
                 + list(VARIANTS) + RECALIBRATED_GBM_KEYS)
    test_preds = preds[in_test]
    held = gt.score_frame(test_preds, truth[in_test], held_keys)
    held_boot = gt.bootstrap_medians(test_preds, truth[in_test], pano[in_test],
                                     ["D_flat", "D_rescaled"] + RECAL_KEYS["gbm_l1"]
                                     + ["gbm_modern"] + SCALED_KEYS,
                                     reference="D_flat")

    step("asserting the closed-form rows equal the committed Stage 4 remedy table...")
    with open(os.path.join(args.data_dir, "modern-truth-summary.json"), encoding="utf-8") as f:
        remedies = json.load(f)["remedies"]
    assert abs(k_rescale - remedies["k_rescale"]) < 1e-12, (k_rescale, remedies["k_rescale"])
    assert abs(flat_h - remedies["flat_height_m"]) < 1e-12
    assert int(in_train.sum()) == remedies["split"]["n_train_rows"]
    assert int(in_test.sum()) == remedies["split"]["n_test_rows"]
    for ours, theirs in (("A_deployed", "A_deployed"), ("D_blend", "D_blend_as_shipped"),
                         ("D_rescaled", "D_rescaled"), ("D_flat", "D_flat")):
        for metric in ("median_abs_m", "signed_median_m", "p90_abs_m"):
            got, want = held[ours][metric], remedies["test_half"][theirs][metric]
            assert abs(got - want) < 1e-9, (ours, metric, got, want)

    # ------------------------------------------------------------------------- cuts
    step("error vs true distance, and the unseen-label-type cut...")
    shape_keys = ["D_blend", "D_flat", "gbm_l1", "gbm_l1_scaled"]
    by_distance = gt.error_vs_distance(test_preds, truth[in_test], shape_keys)

    seen = human["label_type"].isin(dr.LABEL_TYPES).to_numpy()
    by_seen = {}
    for name, sel in (("seen_types", in_test & seen), ("unseen_types", in_test & ~seen)):
        if sel.sum() >= 25:
            by_seen[name] = {"n": int(sel.sum()),
                             **gt.score_frame(preds[sel], truth[sel],
                                              ["D_flat", "gbm_l1", "gbm_l1_scaled"])}

    # ---------------------------------------------------------------------- headline
    era_gap = (era_reference["models"]["D_blend_type_l1"]["dist_median_m"]
               / era_reference["models"]["gbm_l1"]["dist_median_m"] - 1.0) * 100.0
    modern_raw_gap = (pooled["D_blend"]["median_abs_m"]
                      / pooled["gbm_l1"]["median_abs_m"] - 1.0) * 100.0
    modern_cal_gap = (held["D_flat"]["median_abs_m"]
                      / held["gbm_l1_scaled"]["median_abs_m"] - 1.0) * 100.0
    best_gbm_key = min(RECALIBRATED_GBM_KEYS, key=lambda k: held[k]["median_abs_m"])
    # What the interaction structure is worth, measured the same way in both truth frames:
    # the full booster's edge over the single-signal booster. This is the load-bearing
    # comparison — it is internal to the GBM family, so no closed form's calibration enters.
    structure = {
        "era_frame": {
            "only_sv_image_y_dist_median_m":
                era_reference["models"]["only_sv_image_y"]["dist_median_m"],
            "gbm_l1_dist_median_m": era_reference["models"]["gbm_l1"]["dist_median_m"],
            "worth_m": (era_reference["models"]["only_sv_image_y"]["dist_median_m"]
                        - era_reference["models"]["gbm_l1"]["dist_median_m"]),
        },
        "modern_calibrated": {
            "only_sv_image_y_scaled_median_abs_m": held["only_sv_image_y_scaled"]["median_abs_m"],
            "gbm_l1_scaled_median_abs_m": held["gbm_l1_scaled"]["median_abs_m"],
            "worth_m": (held["only_sv_image_y_scaled"]["median_abs_m"]
                        - held["gbm_l1_scaled"]["median_abs_m"]),
        },
        "reading": "in the era frame the eight extra inputs buy the booster a large margin "
                   "over the same booster given only the vertical pixel; on modern truth, "
                   "with each side carrying the same one-parameter scale, they buy nothing",
    }
    headline = {
        "era_frame_gap_pct_D_over_gbm_l1_dist": era_gap,
        "modern_raw_gap_pct_D_blend_over_gbm_l1": modern_raw_gap,
        "modern_calibrated_gap_pct_D_flat_over_gbm_l1_scaled": modern_cal_gap,
        "best_recalibrated_gbm_key": best_gbm_key,
        "modern_gap_pct_D_flat_over_best_recalibrated_gbm":
            (held["D_flat"]["median_abs_m"] / held[best_gbm_key]["median_abs_m"] - 1.0) * 100.0,
        "structure_worth": structure,
        "reading": "positive = the closed form is that much worse than the booster. The era "
                   "column is #6's headline restated in distance space; the modern columns "
                   "are the same comparison against truth the booster's target never "
                   "touched. Raw holds the era scale handicap constant across both models; "
                   "calibrated gives each side the same single modern parameter, and the "
                   "best-recalibrated column gives the booster the best of every richer "
                   "recalibration too.",
    }

    summary = {
        "meta": {
            "generated_by": "python/run_gbm_transfer.py",
            "lightgbm_version": lgb.__version__,
            "seed": gt.SEED,
            "n_boot": gt.N_BOOT,
            "scale_min_dep_deg": gt.SCALE_MIN_DEP_DEG,
            "calibration_height_px": gt.CALIBRATION_HEIGHT,
            "best_iterations": {k: v["best_iteration"] for k, v in gbms.items()},
            "boosters_match_committed_ceiling": True,
            "closed_forms_match_committed_remedies": True,
            "nothing_refitted_on_modern_data": "the boosters are the #6 boosters; the only "
                                               "modern parameter is one scale per model, "
                                               "fitted on the train half of panoramas",
        },
        "era_reference": era_reference,
        "population": population,
        "frame_mapping": frame,
        "truth_scale_by_resolution": scale_diagnostic,
        "pooled": {
            "note": "all gated human rows. Every era-trained model is fully held out here; "
                    "D_flat_shipped is NOT (its height was fitted on these rows), which is "
                    "why the calibrated comparison lives in held_out_half",
            "models": pooled,
            "bootstrap": pooled_boot,
        },
        "held_out_half": {
            "note": "remedy_check's disjoint pano half; every fitted parameter on either "
                    "side comes from the other half",
            "split": {"n_train_rows": int(in_train.sum()), "n_test_rows": int(in_test.sum()),
                      "by": "pano", "seed": gt.SEED},
            "k_rescale": k_rescale, "flat_height_m": flat_h,
            "gbm_scales": scales,
            "recalibrations": recal,
            "gbm_modern": {
                "best_iteration": gbm_modern["best_iteration"],
                "n_train_rows": int(in_train.sum()),
                "caveat": "trained on 1,293 modern rows against the era split's 316,118: a "
                          "FLOOR on what modern data supports, never a modern ceiling",
            },
            "models": held,
            "bootstrap": held_boot,
        },
        "by_distance": by_distance,
        "by_label_type_seen": by_seen,
        "headline": headline,
    }

    # ------------------------------------------------------------------- print report
    def row(name, d):
        print(f"  {name:<22}{d['median_abs_m']:>10.4f}{d['signed_median_m']:>11.4f}"
              f"{d['p90_abs_m']:>9.3f}{d['range_slope_m_per_m']:>9.3f}{d['n']:>7d}")

    print(f"\nPooled modern gated human rows (n={len(human)}, {population['n_panos']} panos, "
          f"{population['n_cities']} cities) — median |err| m:")
    print(f"  {'model':<22}{'median':>10}{'signed':>11}{'p90':>9}{'slope':>9}{'n':>7}")
    for k in pooled_keys + ["D_flat_shipped"]:
        row(k, pooled[k])

    print(f"\nHeld-out pano half (n={int(in_test.sum())}; scales fitted on the other half):")
    print(f"  {'model':<22}{'median':>10}{'signed':>11}{'p90':>9}{'slope':>9}{'n':>7}")
    for k in held_keys:
        row(k, held[k])
    print("  scales:", {k: round(v, 4) for k, v in scales.items()},
          f"(blend k_rescale {k_rescale:.4f})")

    print("\nThe gap, three ways (positive = closed form worse):")
    print(f"  era frame     D_blend vs gbm_l1          {era_gap:+7.1f}%")
    print(f"  modern raw    D_blend vs gbm_l1          {modern_raw_gap:+7.1f}%")
    print(f"  modern cal.   D_flat  vs gbm_l1_scaled   {modern_cal_gap:+7.1f}%")
    print(f"  modern best   D_flat  vs {best_gbm_key:<18}"
          f"{headline['modern_gap_pct_D_flat_over_best_recalibrated_gbm']:+7.1f}%")
    print("\nImplied camera height by panorama height, median(truth x tan(dep)) at dep>=5:")
    for name, block in (("era truth", scale_diagnostic["era_truth"]),
                        ("modern truth", scale_diagnostic["modern_truth"])):
        cells = "  ".join(f"{k}: {v['implied_height_m']:.3f} (n={v['n']})"
                          for k, v in block["by_pano_height"].items())
        print(f"  {name:<13} pooled {block['pooled']['implied_height_m']:.3f}   {cells}")
    print(f"  shipped flat height {scale_diagnostic['shipped_flat_height_m']:.4f} m")

    print("\nWhat the eight extra inputs buy the booster (full minus single-signal):")
    print(f"  era frame          {structure['era_frame']['worth_m']:+7.3f} m")
    print(f"  modern calibrated  {structure['modern_calibrated']['worth_m']:+7.3f} m")
    ref = held_boot["paired_diff_vs_reference"]
    print("\nPaired bootstrap vs D_flat (held-out half, 95% CI on the median difference):")
    for k, v in ref.items():
        print(f"  {k:<24}[{v['delta_median_abs_m_lo']:+7.3f}, "
              f"{v['delta_median_abs_m_hi']:+7.3f}] m   "
              f"P(better) {v['frac_draws_better_than_reference']:.3f}")

    print("\nBy true-distance bin (held-out half, median |err| m):")
    print(f"  {'bin':<14}{'n':>6}" + "".join(f"{k:>16}" for k in shape_keys))
    for r in by_distance:
        print(f"  {r['bin_m']:<14}{r['n']:>6}"
              + "".join(f"{r['per_model'][k]['median_abs_m']:>16.3f}" for k in shape_keys))

    if args.write:
        out = os.path.join(args.data_dir, "gbm-transfer-summary.json")
        # newline="\n" so a Windows rerun matches the committed LF bytes exactly
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nSummary written to {out}")
    step("done")


if __name__ == "__main__":
    main()
