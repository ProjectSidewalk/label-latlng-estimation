"""Fit the #6 gradient-boosted benchmark: an accuracy ceiling over the #3 closed forms.

Usage (from the repo root):
    python python/run_gbm_ceiling.py            # print the comparison
    python python/run_gbm_ceiling.py --write    # also write data/gbm-ceiling-summary.json

LightGBM on the published train split, scored on the published test split — identical rows,
cleaning, scoring geometry (turf-style spherical destination), error definition, and
click-noise sweep as python/run_distance_refit.py, all reused by import rather than copied
(the sweep included: distance_refit.noise_sweep takes the GBMs as extra predictors, so the
A/D rows match the committed #3 summary structurally, not by hand-syncing two copies). The GBM predicts DISTANCE only (that is what #6 bounds) and is paired with the same
heading half as every #3 rung (exact POV inversion + the one era constant), so its lat/lng
numbers sit directly alongside the #3 Stage-2 matrix. The A_ols and D_blend_type_l1 baselines
are refit in-process and asserted equal to data/distance-refit-summary.json, which locks the
comparison to the same rows and the same geometry.

Explicitly NOT a production candidate: no JS runtime, no interpretable coefficients. The
point is to measure how much accuracy the shipped closed form leaves on the table (modeling
regret), and which features carry whatever gap exists (the ablation).

Handling of pano_height: null on every DC row (58% of the data — the column postdates the DC
schema), so the height-normalized feature sv_norm = sv_image_y * 6656/pano_height exists only
for the six modern cities. Both are passed to LightGBM as NaN and routed natively
(use_missing); no rows are dropped and no imputation is invented.

Deterministic: fixed seeds everywhere, deterministic=true + force_row_wise=true LightGBM
params, no bagging or feature subsampling. Early stopping uses a seeded 90/10 carve of the
TRAIN split only (the test split is never consulted during fitting); the final booster is
refit on the full train split for exactly the early-stopped round count. Offline: committed
CSVs and fixtures only.
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
    add_heading_diff, clean_data, fit_models, haversine_m, load_data, spherical_dest,
    split_from_fixtures,
)
import distance_refit as dr  # noqa: E402
from pov_inversion import pov_if_centered  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 666  # the repo-wide seed (same one the #3 noise sweep uses)

# The issue #6 feature set. sv_norm is #4765's height-normalized vertical pixel; NaN on DC
# (no pano_height) — LightGBM takes NaN natively. label_type is a native categorical.
FEATURES_FULL = ["sv_image_y", "sv_norm", "canvas_x", "canvas_y", "zoom",
                 "label_type", "heading", "pitch", "pano_height"]
# Drop-one ablation groups (correlated features move together, e.g. both canvas axes).
ABLATION_GROUPS = {
    "sv_image_y": ["sv_image_y"],
    "sv_norm": ["sv_norm"],
    "canvas": ["canvas_x", "canvas_y"],
    "zoom": ["zoom"],
    "label_type": ["label_type"],
    "heading_pitch": ["heading", "pitch"],
    "pano_height": ["pano_height"],
}
# Single-input reference models: how far does each vertical signal get on its own?
ONLY_MODELS = {
    "only_sv_image_y": ["sv_image_y"],
    "only_canvas": ["canvas_x", "canvas_y"],
    "only_depression": ["depression_deg"],
}

LGB_PARAMS = {
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
    "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
    "deterministic": True, "force_row_wise": True, "seed": SEED, "verbose": -1,
}
MAX_ROUNDS = 3000
STOPPING_ROUNDS = 100
INNER_VALID_FRAC = 0.1  # seeded carve of the TRAIN split for early stopping


# ---------------------------------------------------------------------------- features / fit

def build_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Feature frame for LightGBM. All numeric columns as float64 (nullable Int64 becomes
    NaN-carrying float), label_type as a fixed-category categorical so train/test/perturbed
    frames encode identically."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c == "sv_norm":
            h = df["pano_height"].astype(float).to_numpy(float)
            out[c] = df["sv_image_y"].to_numpy(float) * (6656.0 / h)
        elif c == "label_type":
            out[c] = pd.Categorical(df["label_type"].astype(str), categories=dr.LABEL_TYPES)
        elif c == "pano_height":
            out[c] = df["pano_height"].astype(float).to_numpy(float)
        else:
            out[c] = df[c].to_numpy(float)
    return out


def fit_gbm(train: pd.DataFrame, cols: list[str], objective: str = "regression_l1") -> dict:
    """Two-pass deterministic fit: early-stop on a seeded 90/10 carve of the train split,
    then refit on the full train split for exactly best_iteration rounds. The test split is
    never touched. objective regression_l1 aligns the fit with the published median metric
    (the ladder's l1 column); regression_l2 is the ols analogue."""
    X = build_features(train, cols)
    y = train["pano_dist"].to_numpy(float)
    params = {**LGB_PARAMS, "objective": objective,
              "metric": "l1" if objective == "regression_l1" else "l2"}

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train))
    n_fit = round(len(train) * (1.0 - INNER_VALID_FRAC))
    d_fit = lgb.Dataset(X.iloc[idx[:n_fit]], label=y[idx[:n_fit]])
    d_val = lgb.Dataset(X.iloc[idx[n_fit:]], label=y[idx[n_fit:]], reference=d_fit)
    probe = lgb.train(params, d_fit, num_boost_round=MAX_ROUNDS, valid_sets=[d_val],
                      callbacks=[lgb.early_stopping(STOPPING_ROUNDS, verbose=False)])
    best = probe.best_iteration
    booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=best)
    return {"booster": booster, "best_iteration": int(best), "cols": list(cols),
            "objective": objective}


def predict_gbm(model: dict, df: pd.DataFrame) -> np.ndarray:
    """Predicted distance, clipped to the same [0, DIST_CAP_M] bound every #3 rung respects."""
    d = model["booster"].predict(build_features(df, model["cols"]))
    return np.clip(d, 0.0, dr.DIST_CAP_M)


# --------------------------------------------------------------------------------- scoring

def score_models(gbms: dict, fits: dict, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per-label error columns, exactly score_rungs' convention: heading half = era_cal exact
    inversion (identical across models), destination spherical, error = haversine meters to
    the depth-truth latlng."""
    heading_pred, delta = dr.heading_for_scoring(train, test)

    def latlng_err(d: np.ndarray) -> np.ndarray:
        lng_e, lat_e = spherical_dest(test["pano_lng"], test["pano_lat"],
                                      test["heading"].to_numpy(float) + heading_pred, d)
        return haversine_m(test["lng"], test["lat"], lng_e, lat_e)

    out = pd.DataFrame({
        "label_id": test["label_id"].to_numpy(),
        "city": test["city"].to_numpy(),
        "pano_dist": test["pano_dist"].to_numpy(float),
        "depression_deg": test["depression_deg"].to_numpy(float),
    })
    for key, params in fits.items():
        d = dr.predict_dist(params, test)
        out[f"dist_pred_{key}"] = d
        out[f"error_{key}"] = latlng_err(d)
        out[f"dist_error_{key}"] = np.abs(out["pano_dist"] - d)
    for key, model in gbms.items():
        d = predict_gbm(model, test)
        out[f"dist_pred_{key}"] = d
        out[f"error_{key}"] = latlng_err(d)
        out[f"dist_error_{key}"] = np.abs(out["pano_dist"] - d)
    out.attrs["era_cal_delta_deg"] = float(delta)
    return out


def metrics(scored: pd.DataFrame, key: str) -> dict:
    """distance_refit._metrics' convention, including its notna guard: n counts the rows that
    actually scored, so a silent NaN cannot inflate the denominator while the median skips it."""
    err, derr = scored[f"error_{key}"], scored[f"dist_error_{key}"]
    ok = err.notna()
    return {"n": int(ok.sum()),
            "latlng_median_m": float(err[ok].median()),
            "latlng_p90_m": float(err[ok].quantile(0.9)),
            "dist_median_m": float(derr[ok].median()),
            "dist_p90_m": float(derr[ok].quantile(0.9))}


DIST_BIN_EDGES = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]


def error_vs_distance(scored: pd.DataFrame, keys: list[str]) -> list[dict]:
    """Median lat/lng error by TRUE distance bin — where does the GBM's advantage live?"""
    bins = pd.cut(scored["pano_dist"], DIST_BIN_EDGES, right=False)
    rows = []
    for interval, g in scored.groupby(bins, observed=True):
        rows.append({"bin_m": str(interval), "n": int(len(g)), "per_model": {
            k: {"latlng_median_m": float(g[f"error_{k}"].median()),
                "dist_median_m": float(g[f"dist_error_{k}"].median())} for k in keys}})
    return rows


def noise_sweep(gbms: dict, fits: dict, models: dict, train: pd.DataFrame, test: pd.DataFrame,
                gbm_keys: list[str], fit_keys: list[str], seed: int = SEED) -> dict:
    """#3's sweep itself, with the GBMs handed in as extra predictors — not a copy of it.

    ``distance_refit.noise_sweep`` owns the perturbation design (Gaussian click noise on
    canvas_x/y, every click-dependent input re-derived: depression via the exact projection,
    sv_image_y via the fixed-frame px/deg scale, and sv_norm downstream of that; heading half
    unperturbed) and the rng recipe. Calling it rather than mirroring it is what makes the
    A/D rows equal data/distance-refit-summary.json *structurally* — a findings test still
    checks the equality, but it can no longer be broken by the two implementations drifting.
    Only the output is renamed (``per_rung`` -> ``per_model``) to match this summary's schema.
    """
    extra = {k: (lambda frame, m=gbms[k]: predict_gbm(m, frame)) for k in gbm_keys}
    sweep = dr.noise_sweep(fits, models, train, test, keys=list(fit_keys),
                           seed=seed, extra_predictors=extra)
    return {"sigmas_px": sweep["sigmas_px"], "n_draws": sweep["n_draws"], "seed": seed,
            "baseline_median_m": sweep["baseline_median_m"], "per_model": sweep["per_rung"]}


# ------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--fixtures-dir", default=os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    ap.add_argument("--write", action="store_true",
                    help="write data/gbm-ceiling-summary.json for the findings tests")
    args = ap.parse_args()
    t0 = time.time()

    def step(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("loading and cleaning (the exact 2021 pipeline)...")
    cleaned, _ = clean_data(load_data(args.data_dir))
    cleaned = dr.add_depression(add_heading_diff(cleaned))
    train, test = split_from_fixtures(cleaned, args.fixtures_dir)
    models = fit_models(train, include_est6=False)

    step("refitting the #3 baselines (A_ols, D_blend_type_l1) for in-process comparison...")
    fits = {"A_ols": dr.fit_linear(train, "ols"),
            "D_blend_type_l1": dr.fit_blend(train, "l1", per_type=True)}
    for z in (1, 2, 3):  # A_ols must equal est7's distance half, as in fit_all_rungs
        for term, got in fits["A_ols"]["coef"][z - 1].items():
            want = models["est7"]["dist"][z - 1][term]
            assert abs(got - want) < 1e-9, (term, z)

    step("fitting the GBM variants (early stop on a seeded train carve, refit on full train)...")
    gbms: dict = {}
    gbms["gbm_l1"] = fit_gbm(train, FEATURES_FULL, "regression_l1")
    step(f"  gbm_l1: {gbms['gbm_l1']['best_iteration']} rounds")
    gbms["gbm_l2"] = fit_gbm(train, FEATURES_FULL, "regression_l2")
    step(f"  gbm_l2: {gbms['gbm_l2']['best_iteration']} rounds")
    gbms["gbm_dep_l1"] = fit_gbm(train, FEATURES_FULL + ["depression_deg"], "regression_l1")
    step(f"  gbm_dep_l1 (+exact depression): {gbms['gbm_dep_l1']['best_iteration']} rounds")
    for name, cols in ONLY_MODELS.items():
        gbms[name] = fit_gbm(train, cols, "regression_l1")
        step(f"  {name}: {gbms[name]['best_iteration']} rounds")

    step("drop-one ablations...")
    for gname, dropped in ABLATION_GROUPS.items():
        cols = [c for c in FEATURES_FULL if c not in dropped]
        gbms[f"drop_{gname}"] = fit_gbm(train, cols, "regression_l1")
        step(f"  drop_{gname}: {gbms[f'drop_{gname}']['best_iteration']} rounds")

    step("scoring on the published test split...")
    scored = score_models(gbms, fits, train, test)
    matrix = {k: metrics(scored, k) for k in list(fits) + list(gbms)}

    # Lock the comparison: the recomputed baselines must equal the committed #3 summary.
    with open(os.path.join(args.data_dir, "distance-refit-summary.json"), encoding="utf-8") as f:
        ref = json.load(f)
    for key in ("A_ols", "D_blend_type_l1"):
        for m in ("latlng_median_m", "latlng_p90_m", "dist_median_m", "dist_p90_m"):
            assert abs(matrix[key][m] - ref["matrix"][key][m]) < 1e-9, (key, m)
    assert abs(scored.attrs["era_cal_delta_deg"] - ref["meta"]["era_cal_delta_deg"]) < 1e-9

    step("error vs true distance...")
    evd_keys = ["A_ols", "D_blend_type_l1", "gbm_l1", "gbm_dep_l1"]
    evd = error_vs_distance(scored, evd_keys)

    step("click-noise sweep (#3's own sweep, GBMs handed in as extra predictors)...")
    noise = noise_sweep(gbms, fits, models, train, test,
                        gbm_keys=["gbm_l1", "gbm_dep_l1"],
                        fit_keys=["A_ols", "D_blend_type_l1"])

    # ------------------------------------------------------------------------- assemble
    ablation = {"reference_key": "gbm_l1",
                "reference_dist_median_m": matrix["gbm_l1"]["dist_median_m"],
                "reference_latlng_median_m": matrix["gbm_l1"]["latlng_median_m"],
                "drop": {}}
    for gname in ABLATION_GROUPS:
        k = f"drop_{gname}"
        ablation["drop"][gname] = {
            **matrix[k],
            "delta_dist_median_m": matrix[k]["dist_median_m"] - matrix["gbm_l1"]["dist_median_m"],
            "delta_latlng_median_m": (matrix[k]["latlng_median_m"]
                                      - matrix["gbm_l1"]["latlng_median_m"]),
            "best_iteration": gbms[k]["best_iteration"]}

    booster = gbms["gbm_l1"]["booster"]
    gain = dict(zip(booster.feature_name(),
                    map(float, booster.feature_importance(importance_type="gain"))))
    total_gain = sum(gain.values())
    importance = {k: {"gain": v, "gain_share": v / total_gain}
                  for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}

    gbm_keys = ["gbm_l1", "gbm_l2", "gbm_dep_l1"]
    best_key = min(gbm_keys, key=lambda k: matrix[k]["latlng_median_m"])
    d_med = matrix["D_blend_type_l1"]["latlng_median_m"]
    gbm_med = matrix[best_key]["latlng_median_m"]
    ceiling = {
        "blend_d_latlng_median_m": d_med,
        "gbm_best_key": best_key,
        "gbm_best_latlng_median_m": gbm_med,
        "d_over_gbm_gap_pct": 100.0 * (d_med / gbm_med - 1.0),
        "a_over_gbm_gap_pct": 100.0 * (matrix["A_ols"]["latlng_median_m"] / gbm_med - 1.0),
        "note": "best-of-variants ON TEST, so the quoted ceiling is optimistic and D's regret "
                "is an UPPER bound — the anti-conservative direction for this report's own "
                "'the gap is large' conclusion, which is why it is stated rather than buried. "
                "d_over_gbm_gap_pct_by_variant shows what the selection is worth: the answer "
                "to #6 is the same under every variant, including the one that loses",
        "d_over_gbm_gap_pct_by_variant": {
            k: 100.0 * (d_med / matrix[k]["latlng_median_m"] - 1.0) for k in gbm_keys},
    }

    summary = {
        "meta": {
            "n_train": int(len(train)), "n_test": int(len(test)),
            "era_cal_delta_deg": scored.attrs["era_cal_delta_deg"],
            "dist_cap_m": dr.DIST_CAP_M,
            "lightgbm_version": lgb.__version__,
            "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
            "max_rounds": MAX_ROUNDS, "stopping_rounds": STOPPING_ROUNDS,
            "inner_valid_frac": INNER_VALID_FRAC, "seed": SEED,
            "features_full": FEATURES_FULL,
            "best_iterations": {k: v["best_iteration"] for k, v in gbms.items()},
            "n_pano_height_missing_train": int(train["pano_height"].isna().sum()),
            "n_pano_height_missing_test": int(test["pano_height"].isna().sum()),
            "baselines_match_committed_summary": True,
            "not_a_production_candidate": "benchmark only: no JS runtime, no interpretable "
                                          "coefficients; bounds the #3 closed forms' regret",
        },
        "matrix": matrix,
        "ablation": ablation,
        "feature_importance_gain_gbm_l1": importance,
        "error_vs_distance": evd,
        "noise_sweep": noise,
        "ceiling": ceiling,
    }

    # ---------------------------------------------------------------------- print report
    print(f"\nThe matrix (test n={len(test)}; heading half identical; distance clipped to "
          f"[0, {dr.DIST_CAP_M:.0f}] m):")
    print(f"  {'model':<22}{'rounds':>7}{'latlng med':>12}{'latlng p90':>12}"
          f"{'dist med':>10}{'dist p90':>10}")
    show = ["A_ols", "D_blend_type_l1", "gbm_l2", "gbm_l1", "gbm_dep_l1",
            "only_sv_image_y", "only_canvas", "only_depression"]
    for k in show:
        r = matrix[k]
        rounds = gbms[k]["best_iteration"] if k in gbms else "-"
        print(f"  {k:<22}{rounds:>7}{r['latlng_median_m']:>12.4f}{r['latlng_p90_m']:>12.4f}"
              f"{r['dist_median_m']:>10.4f}{r['dist_p90_m']:>10.4f}")

    print(f"\nCeiling: blend D {d_med:.4f} m vs best GBM ({best_key}) {gbm_med:.4f} m -> "
          f"D is {ceiling['d_over_gbm_gap_pct']:+.1f}% above the ceiling "
          f"(A_ols {ceiling['a_over_gbm_gap_pct']:+.1f}%)")

    print("\nDrop-one ablation (delta test dist median vs gbm_l1, m; + = worse without it):")
    for gname, row in sorted(ablation["drop"].items(),
                             key=lambda kv: -kv[1]["delta_dist_median_m"]):
        print(f"  -{gname:<16}{row['delta_dist_median_m']:>+9.4f}"
              f"   (latlng {row['delta_latlng_median_m']:>+8.4f})")

    print("\nGain share (gbm_l1):")
    for k, v in importance.items():
        print(f"  {k:<16}{v['gain_share']:>8.1%}")

    print("\nError vs true distance (latlng median, m):")
    print(f"  {'bin':<14}{'n':>7}" + "".join(f"{k:>18}" for k in evd_keys))
    for row in evd:
        print(f"  {row['bin_m']:<14}{row['n']:>7}"
              + "".join(f"{row['per_model'][k]['latlng_median_m']:>18.4f}" for k in evd_keys))

    print("\nClick-noise sweep (delta median latlng error vs unperturbed, m; same draws as #3):")
    keys = ["A_ols", "D_blend_type_l1", "gbm_l1", "gbm_dep_l1"]
    print(f"  {'model':<18}" + "".join(f"{s:>10}" for s in noise["sigmas_px"]))
    for k in keys:
        row = noise["per_model"][k]
        print(f"  {k:<18}" + "".join(f"{row[str(s)]['delta_median_m']:>10.4f}"
                                     for s in noise["sigmas_px"]))

    if args.write:
        out = os.path.join(args.data_dir, "gbm-ceiling-summary.json")
        # newline="\n" so a Windows rerun matches the committed LF bytes exactly
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nSummary written to {out}")
    step("done")


if __name__ == "__main__":
    main()
