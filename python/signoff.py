"""Production-adoption sign-off for the geometric lat/lng estimator (SidewalkWebpage#5084).

SidewalkWebpage ships the geometric estimator as ``computation_method = 'approximation3'``
(PanoDataService.toLatLng / Label.js#toLatLng, PR #4819; evolution 352 recomputed every stored
'approximation2' row with it). This module scores exactly that shipped path -- constants from
``final_coefficients``, spherical destination on the app's 6371 km sphere -- against the two
truth frames the repo holds, and answers the three questions the issue asks:

1. accuracy head-to-head with the 2021 per-zoom regression, on the regression's own
   720x480-era held-out split *and* on modern fresh-depth truth, sliced by zoom, label type,
   distance from the camera, city and panorama resolution;
2. the geodesy decision (sphere vs WGS84 ellipsoid, and the three sphere radii in play),
   quantified at label-placement distances and pinned by a cross-implementation fixture;
3. the frame contract the Immersive Explore work (SidewalkWebpage#5085) needs: the estimator
   consumes angles, so a viewport of any size reproduces the same position *provided the
   click is projected through its own frame* -- and what goes wrong when it is not.

Everything is offline and deterministic except the ``fetch`` stage of ``run_signoff.py``, which
pulls imagery tiles for the worked examples and commits them verbatim.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    EARTH_RADIUS_M, add_heading_diff, clean_data, fit_models, haversine_m, latlng_error_m,
    load_data, predict_dist_heading, spherical_dest, split_from_fixtures,
)
import distance_refit as dr  # noqa: E402
from distance_refit import DIST_CAP_M, _predict_blend  # noqa: E402
from pov_inversion import (  # noqa: E402
    CANVAS_H, CANVAS_W, exact_heading_diff, get_3d_fov, pov_if_centered,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ------------------------------------------------------------------ the shipped constants
# PanoDataService.LatLngEstimation (SidewalkWebpage) and 352.sql, verbatim.
R_PRODUCTION_M = 6371000.0    # CommonUtils.EARTH_RADIUS_KM * 1000 (Scala + SQL)
R_TURF_M = 6371008.8          # turf's earthRadius: what the Explore client's destination uses
R_HARNESS_M = EARTH_RADIUS_M  # 6378137: geosphere's sphere, what every report here scored with
MAX_DISTANCE_M = 50.0


def load_shipped(data_dir: str = DATA) -> dict:
    """``final_coefficients`` as a predict_dist params dict: the production constants."""
    with open(os.path.join(data_dir, "modern-truth-summary.json"), encoding="utf-8") as f:
        fc = json.load(f)["final_coefficients"]
    return {"form": "blend", "height_m": float(fc["params"]["height_m"]),
            "blend_deg": float(fc["params"]["blend_deg"])}


def load_era_blend(data_dir: str = DATA) -> dict:
    """The era fit's per-type blend (``era_fit_coefficients``), the refit report's chosen rung."""
    with open(os.path.join(data_dir, "distance-refit-summary.json"), encoding="utf-8") as f:
        return json.load(f)["era_fit_coefficients"]["params"]


# ------------------------------------------------------------- the production path, ported
# Statement-for-statement ports of the Scala (PanoDataService) the sign-off is about. The
# harness's own helpers are used for scoring; these exist so the fixture and the record-path
# check exercise the deployed formula, not a harness convention.

def pov_from_pano_xy(pano_x, pano_y, width, height, camera_heading):
    """PanoDataService.calculatePovFromPanoXY: (heading, pitch) of a pano pixel.

    Column zero sits at ``camera_heading - 180``; y is linear in elevation with the horizon at
    ``height / 2``. Heading is returned unwrapped the way Scala's ``%`` leaves it (it can be
    negative); the destination formula is periodic so nothing downstream cares."""
    x = np.asarray(pano_x, float)
    y = np.asarray(pano_y, float)
    w = np.asarray(width, float)
    h = np.asarray(height, float)
    heading = np.fmod(np.asarray(camera_heading, float) - 180.0 + x / w * 360.0, 360.0)
    pitch = 90.0 - 180.0 * y / h
    return heading, pitch


def estimate_distance_m(depression_deg, params: dict) -> np.ndarray:
    """PanoDataService.estimateDistanceFromPanoM: the saturating-cotangent blend."""
    dep = np.asarray(depression_deg, float)
    h = params["height_m"]
    a = params["blend_deg"]
    a_rad = np.radians(a)
    with np.errstate(divide="ignore"):
        cot = h / np.tan(np.radians(dep))
    tail = (h / np.tan(a_rad)
            + h * (np.pi / 180.0) / np.sin(a_rad) ** 2 * (a - np.maximum(dep, 0.0)))
    return np.where(dep >= a, cot, np.minimum(tail, MAX_DISTANCE_M))


def destination(lat, lng, dist_m, bearing_deg, radius_m: float = R_PRODUCTION_M):
    """CommonUtils.calculateDestination: spherical destination point, (lat, lng) in degrees."""
    lat1 = np.radians(np.asarray(lat, float))
    lng1 = np.radians(np.asarray(lng, float))
    b = np.radians(np.asarray(bearing_deg, float))
    d = np.asarray(dist_m, float) / radius_m
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(b))
    lng2 = lng1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lng2)


def production_to_latlng(pano_lat, pano_lng, pano_x, pano_y, width, height, camera_heading,
                         params: dict, radius_m: float = R_PRODUCTION_M):
    """PanoDataService.toLatLng end to end: pano pixel -> POV -> distance -> destination."""
    heading, pitch = pov_from_pano_xy(pano_x, pano_y, width, height, camera_heading)
    dist = estimate_distance_m(-pitch, params)
    lat, lng = destination(pano_lat, pano_lng, dist, heading, radius_m)
    return lat, lng, dist, heading, pitch


# --------------------------------------------------------------------------- metrics

DIST_BINS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
DEP_BINS = [-90.0, 2.0, 5.0, 11.25, 20.0, 90.0]


def _bin_labels(edges: list[float]) -> list[str]:
    return [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]


def _stats(err: np.ndarray, signed: np.ndarray | None = None) -> dict:
    ok = np.isfinite(err)
    e = err[ok]
    out = {"n": int(ok.sum()),
           "median_m": float(np.median(e)) if e.size else None,
           "p90_m": float(np.percentile(e, 90)) if e.size else None,
           "mean_m": float(np.mean(e)) if e.size else None}
    if signed is not None:
        s = signed[ok]
        out["signed_median_m"] = float(np.median(s)) if s.size else None
    return out


def slice_table(frame: pd.DataFrame, models: dict[str, str], by: str, min_n: int = 25,
                signed: dict[str, str] | None = None, reference: str | None = None) -> list[dict]:
    """Per-slice metrics for every model column; ``models`` maps a model key to its error
    column. With ``reference`` set, adds each model's paired win rate against that model."""
    rows = []
    for value, g in frame.groupby(by, sort=True, observed=True):
        if len(g) < min_n:
            continue
        row = {by: str(value), "n": int(len(g))}
        for key, col in models.items():
            err = g[col].to_numpy(float)
            sgn = g[signed[key]].to_numpy(float) if signed and key in signed else None
            row[key] = _stats(err, sgn)
            if reference and key != reference:
                ref = g[models[reference]].to_numpy(float)
                ok = np.isfinite(err) & np.isfinite(ref)
                row[key]["win_rate_vs_ref"] = float(np.mean(err[ok] < ref[ok])) if ok.any() else None
        rows.append(row)
    return rows


def cluster_bootstrap_median_diff(err_a: np.ndarray, err_b: np.ndarray, cluster: np.ndarray,
                                  n_boot: int = 1000, seed: int = 666) -> dict:
    """Bootstrap CI for median(err_a) - median(err_b), resampling clusters (panos) so that
    the several labels on one panorama are not counted as independent draws."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    starts = np.searchsorted(inv_sorted, np.arange(len(codes)))
    ends = np.append(starts[1:], len(inv_sorted))
    a_s, b_s = err_a[order], err_b[order]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(codes), len(codes))
        idx = np.concatenate([np.arange(starts[c], ends[c]) for c in pick])
        diffs[i] = np.median(a_s[idx]) - np.median(b_s[idx])
    return {"n_clusters": int(len(codes)), "n_boot": n_boot,
            "point_m": float(np.median(err_a) - np.median(err_b)),
            "ci95_m": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


# ------------------------------------------------------------------------ era frame

# The three production methods, in production vocabulary (label_point.computation_method):
#   'depth'          -- 2017-2020, read from Google's depth map at label time: the era TRUTH, not a candidate;
#   'approximation1' -- evolution 93 (2020-11-13): a fixed 10 m along the viewport heading, flat-earth offsets.
#                       The 2021 analysis's est1 (4.84 m). Retired by evolution 98 two months later;
#   'approximation2' -- evolution 98 (2021-01-12): the 2021 per-zoom regression (est7 / A_deployed below);
#   'approximation3' -- evolution 349/352 (2026-08): the shipped geometric estimator (approx3 below).
APPROX1_DISTANCE_M = 10.0
APPROX1_M_PER_DEG = 111111.0  # evolution 93's flat-earth metres per degree


def approximation1_latlng(pano_lat, pano_lng, viewport_heading_deg):
    """Evolution 93 verbatim: lat/lng 10 m along the viewport heading on a flat-earth grid."""
    lat = np.asarray(pano_lat, float)
    lng = np.asarray(pano_lng, float)
    h = np.radians(np.asarray(viewport_heading_deg, float))
    return (lat + APPROX1_DISTANCE_M * np.cos(h) / APPROX1_M_PER_DEG,
            lng + APPROX1_DISTANCE_M * np.sin(h) / (APPROX1_M_PER_DEG * np.cos(np.radians(lat))))


ERA_MODELS = {
    "approx1": "err_approx1",                  # approximation1: 10 m along the viewport heading (evolution 93)
    "est7": "err_est7",                        # the 2021 pipeline as published (own heading, legacy geodesy)
    "est7_sph": "err_est7_sph",                # same distances, the shared spherical/era-cal heading
    "approx3": "err_approx3",                  # SHIPPED: modern height, exact heading, no era constant
    "approx3_eracal": "err_approx3_eracal",    # shipped distance, heading with the era truth's +0.72 deg removed
    "approx3_eraflat": "err_approx3_eraflat",  # same form, one height fitted on the era train split
    "blend_type_era": "err_blend_type_era",    # the refit's chosen 8-parameter era rung, for continuity
    "anchor": "err_anchor",                    # 2.6 m / tan(dep), zero parameters
}
ERA_DIST_MODELS = {k: v.replace("err_", "derr_") for k, v in ERA_MODELS.items()}
ERA_SIGNED = {k: v.replace("err_", "sderr_") for k, v in ERA_MODELS.items()}


def era_flat_height(train: pd.DataFrame, min_dep_deg: float = 5.0) -> float:
    """The shipped form's one parameter, fitted the shipped way but on the era TRAIN split:
    median(truth x tan(depression)) at depression >= 5 deg. Equal-budget comparator."""
    tr = train[train["depression_deg"] >= min_dep_deg]
    return float(np.median(tr["pano_dist"].to_numpy(float)
                           * np.tan(np.radians(tr["depression_deg"].to_numpy(float)))))


def era_frame(shipped: dict, data_dir: str = DATA,
              fixtures_dir: str = os.path.join(ROOT, "tests", "fixtures", "r-baseline")
              ) -> tuple[pd.DataFrame, dict]:
    """Score the shipped estimator on the 2021 regression's own held-out split.

    Truth is the 2017-2020 client's depth-derived positions (the frame the regression was
    fitted in). The regression is scored exactly as published (continuity row 1.4621 m) and
    under the shared conventions; the shipped estimator is scored as shipped (exact heading,
    no era constant) and with the era truth's documented +0.72 deg bearing bias removed, which
    is the like-for-like comparison since est7's fitted heading absorbed that bias."""
    cleaned, _ = clean_data(load_data(data_dir))
    cleaned = dr.add_depression(add_heading_diff(cleaned))
    train, test = split_from_fixtures(cleaned, fixtures_dir)
    models = fit_models(train, include_est6=False)
    era_blend = load_era_blend(data_dir)

    dist7, head7 = predict_dist_heading(models, test, "est7")
    heading_eracal, delta = dr.heading_for_scoring(train, test)
    heading_exact = exact_heading_diff(test)
    dep = test["depression_deg"].to_numpy(float)
    truth = test["pano_dist"].to_numpy(float)
    h_era = era_flat_height(train)
    eraflat = {"form": "blend", "height_m": h_era, "blend_deg": shipped["blend_deg"]}

    def latlng_err(d: np.ndarray, heading_diff: np.ndarray) -> np.ndarray:
        lng_e, lat_e = spherical_dest(test["pano_lng"], test["pano_lat"],
                                      test["heading"].to_numpy(float) + heading_diff, d)
        return haversine_m(test["lng"], test["lat"], lng_e, lat_e)

    out = pd.DataFrame({
        "label_id": test["label_id"].to_numpy(), "city": test["city"].to_numpy(str),
        "pano_id": test["pano_id"].to_numpy(str), "zoom": test["zoom"].to_numpy(int),
        "label_type": test["label_type"].to_numpy(str),
        "pano_height": test["pano_height"].to_numpy(float),
        "pano_dist": truth, "depression_deg": dep,
    })
    with np.errstate(divide="ignore"):
        anchor = np.where(dep > 0, 2.6 / np.tan(np.radians(np.maximum(dep, 1e-9))), np.inf)
    dists = {
        "est7": dist7, "est7_sph": dist7,
        "approx3": _predict_blend(shipped, dep), "approx3_eracal": _predict_blend(shipped, dep),
        "approx3_eraflat": _predict_blend(eraflat, dep),
        "blend_type_era": _predict_blend(era_blend, dep, test["label_type"].to_numpy(str)),
        "anchor": np.clip(anchor, 0.0, DIST_CAP_M),
    }
    headings = {"est7_sph": heading_eracal, "approx3": heading_exact,
                "approx3_eracal": heading_eracal, "approx3_eraflat": heading_eracal,
                "blend_type_era": heading_eracal, "anchor": heading_eracal}
    out["err_est7"] = latlng_error_m(test, dist7, head7, crude=False)
    # approximation1 (evolution 93): no pitch input at all, and the viewport heading rather than
    # the label's, so both halves of the position are wrong by construction. Scored as written.
    lat_1, lng_1 = approximation1_latlng(test["pano_lat"], test["pano_lng"], test["heading"])
    out["dist_approx1"] = np.full(len(test), APPROX1_DISTANCE_M)
    out["derr_approx1"] = np.abs(APPROX1_DISTANCE_M - truth)
    out["sderr_approx1"] = APPROX1_DISTANCE_M - truth
    out["err_approx1"] = haversine_m(test["lng"], test["lat"], lng_1, lat_1)
    for key, d in dists.items():
        out[f"dist_{key}"] = d
        out[f"derr_{key}"] = np.abs(d - truth)
        out[f"sderr_{key}"] = d - truth
        if key != "est7":
            out[f"err_{key}"] = latlng_err(d, headings[key])

    # The production RECORD path: evolution 179's pano_x/pano_y through calculatePovFromPanoXY,
    # i.e. what 352.sql would compute for these rows had they not been 'depth' rows.
    rec = test[["current_pano_x", "current_pano_y", "pano_width", "pano_height",
                "photographer_heading"]].notna().all(axis=1).to_numpy()
    lat_r, lng_r, d_r, h_r, p_r = production_to_latlng(
        test["pano_lat"], test["pano_lng"], test["current_pano_x"].fillna(0),
        test["current_pano_y"].fillna(0), test["pano_width"].fillna(1), test["pano_height"].fillna(1),
        test["photographer_heading"].fillna(0), shipped)
    lng_h, lat_h = spherical_dest(test["pano_lng"], test["pano_lat"],
                                  test["heading"].to_numpy(float) + heading_exact, dists["approx3"])
    out["err_record"] = np.where(rec, haversine_m(test["lng"], test["lat"], lng_r, lat_r), np.nan)
    out["record_vs_harness_m"] = np.where(rec, haversine_m(lng_r, lat_r, lng_h, lat_h), np.nan)
    out["record_dep_minus_exact_deg"] = np.where(rec, -p_r - dep, np.nan)

    out["dist_bin"] = pd.cut(out["pano_dist"], DIST_BINS, labels=_bin_labels(DIST_BINS), right=False)
    out["dep_bin"] = pd.cut(out["depression_deg"], DEP_BINS, labels=_bin_labels(DEP_BINS), right=False)
    out["pano_height_px"] = out["pano_height"].fillna(0).astype(int).astype(str)

    overall = {k: _stats(out[v].to_numpy(float), out[ERA_SIGNED[k]].to_numpy(float))
               for k, v in ERA_MODELS.items()}
    for k in ERA_MODELS:
        if k != "est7":
            overall[k]["win_rate_vs_est7"] = float(np.mean(out[ERA_MODELS[k]] < out["err_est7"]))
        overall[k]["dist"] = _stats(out[ERA_DIST_MODELS[k]].to_numpy(float))

    # The truth's own scale along the axis the shipped constant leans on: the camera height
    # each subpopulation's truth implies, median(truth x tan(dep)) at dep >= 5 deg. Where this
    # sits far from the shipped 2.34 m, the era truth -- not the click geometry -- is what the
    # shipped estimator disagrees with (modern-truth report SS7).
    steep = out[out["depression_deg"] >= 5.0].copy()
    steep["implied"] = (steep["pano_dist"].to_numpy(float)
                        * np.tan(np.radians(steep["depression_deg"].to_numpy(float))))

    def implied_by(col: str) -> list[dict]:
        rows = []
        for value, g in steep.groupby(col, sort=True, observed=True):
            if len(g) >= 100:
                rows.append({col: str(value), "n": int(len(g)),
                             "implied_height_m": float(np.median(g["implied"]))})
        return rows

    summary = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "era_cal_delta_deg": float(delta), "era_flat_height_m": h_era,
        "shipped": shipped,
        "overall": overall,
        "implied_height_overall_m": float(np.median(steep["implied"])),
        "implied_height_by_city": implied_by("city"),
        "implied_height_by_pano_height": implied_by("pano_height_px"),
        "implied_height_by_zoom": implied_by("zoom"),
        "bootstrap_median_diff_vs_est7": {
            k: cluster_bootstrap_median_diff(out[v].to_numpy(float), out["err_est7"].to_numpy(float),
                                             out["pano_id"].to_numpy(str))
            for k, v in ERA_MODELS.items() if k in ("approx3", "approx3_eracal", "approx3_eraflat")},
        "by_zoom": slice_table(out, ERA_MODELS, "zoom", signed=ERA_SIGNED, reference="est7"),
        "by_label_type": slice_table(out, ERA_MODELS, "label_type", signed=ERA_SIGNED, reference="est7"),
        "by_city": slice_table(out, ERA_MODELS, "city", signed=ERA_SIGNED, reference="est7"),
        "by_pano_height": slice_table(out, ERA_MODELS, "pano_height_px", min_n=100,
                                      signed=ERA_SIGNED, reference="est7"),
        "by_true_distance": slice_table(out, ERA_MODELS, "dist_bin", signed=ERA_SIGNED, reference="est7"),
        "by_depression": slice_table(out, ERA_MODELS, "dep_bin", signed=ERA_SIGNED, reference="est7"),
        "record_path": {
            "n_with_record": int(rec.sum()),
            "err_record": _stats(out["err_record"].to_numpy(float)),
            "record_vs_harness_m": _stats(out["record_vs_harness_m"].to_numpy(float)),
            "record_dep_minus_exact_deg": {
                "median": float(np.nanmedian(out["record_dep_minus_exact_deg"])),
                "p90_abs": float(np.nanpercentile(np.abs(out["record_dep_minus_exact_deg"]), 90))},
        },
    }
    return out, summary


# --------------------------------------------------------------------- modern frame

MODERN_MODELS = {"approx1": "err_approx1", "A_deployed": "err_A", "approx3": "err_approx3",
                 "C_anchor": "err_C", "D_blend_era": "err_D"}
MODERN_SIGNED = {"approx1": "sderr_approx1", "A_deployed": "sderr_A", "approx3": "sderr_approx3",
                 "C_anchor": "sderr_C", "D_blend_era": "sderr_D"}


def load_modern(data_dir: str = DATA) -> pd.DataFrame:
    labels = pd.read_csv(os.path.join(data_dir, "modern-truth-labels.csv.gz"), dtype={"pano_id": str})
    human = labels[labels["gate_ok"] & ~labels["is_ai"].astype(bool)].copy()
    human["capture_year"] = human["capture_date"].astype(str).str[:4]
    human["zoom_i"] = np.clip(np.round(human["zoom"].astype(float)).astype(int), 1, 3)
    return human.reset_index(drop=True)


def modern_predictions(human: pd.DataFrame, shipped: dict) -> pd.DataFrame:
    out = human.copy()
    truth = out["truth_m"].to_numpy(float)
    dep = out["depression_deg"].to_numpy(float)
    out["dist_approx3"] = _predict_blend(shipped, dep)
    # approximation1's distance half; its bearing half (the viewport heading) cannot be scored
    # in this frame, whose truth is a range along the label's own ray.
    out["dist_approx1"] = np.full(len(out), APPROX1_DISTANCE_M)
    for key, col in (("approx1", "dist_approx1"), ("A", "A_deployed"), ("approx3", "dist_approx3"),
                     ("C", "C_anchor"), ("D", "D_blend")):
        out[f"sderr_{key}"] = out[col].to_numpy(float) - truth
        out[f"err_{key}"] = np.abs(out[f"sderr_{key}"])
    out["dist_bin"] = pd.cut(out["truth_m"], DIST_BINS, labels=_bin_labels(DIST_BINS), right=False)
    out["dep_bin"] = pd.cut(out["depression_deg"], DEP_BINS, labels=_bin_labels(DEP_BINS), right=False)
    out["pano_height_px"] = out["pano_height"].fillna(0).astype(int).astype(str)
    return out


def fit_flat_height(rows: pd.DataFrame, min_dep_deg: float = 5.0) -> float:
    sub = rows[rows["depression_deg"] >= min_dep_deg]
    return float(np.median(sub["truth_m"].to_numpy(float)
                           * np.tan(np.radians(sub["depression_deg"].to_numpy(float)))))


def repeated_holdout(human: pd.DataFrame, shipped: dict, n_rep: int = 200, seed: int = 666) -> dict:
    """The modern-truth remedy check, repeated: split the panos in half, fit the one height on
    one half, score the other. The shipped constant is fitted on ALL these rows, so its
    in-sample number is optimistic by construction; this is the honest error estimate."""
    rng = np.random.default_rng(seed)
    panos = np.sort(human["pano_id"].unique())
    truth = human["truth_m"].to_numpy(float)
    dep = human["depression_deg"].to_numpy(float)
    a_err = np.abs(human["A_deployed"].to_numpy(float) - truth)
    meds, p90s, heights, a_meds = [], [], [], []
    for _ in range(n_rep):
        train_ids = set(rng.choice(panos, len(panos) // 2, replace=False))
        in_train = human["pano_id"].isin(train_ids).to_numpy()
        h = fit_flat_height(human[in_train])
        params = {"form": "blend", "height_m": h, "blend_deg": shipped["blend_deg"]}
        err = np.abs(_predict_blend(params, dep[~in_train]) - truth[~in_train])
        meds.append(float(np.median(err)))
        p90s.append(float(np.percentile(err, 90)))
        heights.append(h)
        a_meds.append(float(np.median(a_err[~in_train])))
    meds, p90s, heights, a_meds = map(np.asarray, (meds, p90s, heights, a_meds))

    def band(v):
        return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}

    return {"n_rep": n_rep, "seed": seed, "n_panos": int(len(panos)),
            "approx3_median_m": band(meds), "approx3_p90_m": band(p90s),
            "fitted_height_m": band(heights), "A_deployed_median_m": band(a_meds),
            "shipped_beats_deployed_in_every_split": bool(np.all(meds < a_meds))}


def leave_one_city_out(human: pd.DataFrame, shipped: dict, min_n: int = 50) -> list[dict]:
    """Calibrate the height on every other city, score the held-out city: does one height
    transfer across rigs and street geometry? Cities with >= min_n gated human rows."""
    truth = human["truth_m"].to_numpy(float)
    dep = human["depression_deg"].to_numpy(float)
    rows = []
    for city, n in human["city"].value_counts().items():
        if n < min_n:
            continue
        held = (human["city"] == city).to_numpy()
        h = fit_flat_height(human[~held])
        params = {"form": "blend", "height_m": h, "blend_deg": shipped["blend_deg"]}
        err_loco = np.abs(_predict_blend(params, dep[held]) - truth[held])
        err_ship = np.abs(_predict_blend(shipped, dep[held]) - truth[held])
        err_a = np.abs(human["A_deployed"].to_numpy(float)[held] - truth[held])
        rows.append({"city": city, "n": int(n), "height_fitted_elsewhere_m": h,
                     "approx3_loco_median_m": float(np.median(err_loco)),
                     "approx3_shipped_median_m": float(np.median(err_ship)),
                     "A_deployed_median_m": float(np.median(err_a)),
                     "latitude": float(human.loc[held, "pano_lat"].median())})
    return rows


def _wrap_deg(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def rig_tilt_rider(human: pd.DataFrame, data_dir: str = DATA) -> dict:
    """Does the rig's tilt reach the shipped estimator?

    The 2020-2022 undergraduate crop work saw curved horizon lines on hilly panoramas and
    blamed photographer pitch, then roll (gis.stackexchange 422656); SidewalkWebpage#4784
    diagnoses the same thing as a tilt term missing from the POV -> pano_y projection. The
    stored pano_y treats the panorama as level, so if the depth raster (and the imagery) were
    rig-aligned rather than gravity-rectified, the depression angle read from pano_y would be
    off by the rig's tilt projected onto the label's bearing,
    ``T = pitch * cos(bearing - heading) + roll * sin(bearing - heading)``, and the camera
    height each label implies (``truth * tan(depression)``) would track T with a slope near
    ``dh/dT = truth * sec^2(depression) * pi/180``. Both pose components come from the modern
    truth's fresh metadata fetch (``modern-truth-panos.csv.gz``: ``fresh_pitch_deg``,
    ``fresh_roll_deg``, the latter served unwrapped in [0, 360)); the database's own
    ``camera_roll`` is empty for every GSV row. Google's sign conventions for the two angles
    are undocumented, so the rider fits both projections jointly (a sign flip only flips a
    coefficient's sign) and reports the joint R^2 alongside each slope.

    A second, pano-level check needs no labels at all: the depth payload's ground-plane
    normal is tilted (``ground_tilt_deg``) by road camber and slope in a rectified frame, but
    by the rig's whole tilt in a rig-aligned one, so its correlation with the rig tilt
    magnitude says which frame the raster is in."""
    panos = pd.read_csv(os.path.join(data_dir, "modern-truth-panos.csv.gz"), dtype={"pano_id": str})
    panos = panos[panos["fresh_pitch_deg"].notna()].copy()
    panos["roll_wrapped_deg"] = _wrap_deg(panos["fresh_roll_deg"])
    panos["tilt_mag_deg"] = np.hypot(panos["fresh_pitch_deg"], panos["roll_wrapped_deg"])
    cols = ["pano_id", "fresh_pitch_deg", "roll_wrapped_deg", "fresh_heading_deg", "tilt_mag_deg",
            "ground_tilt_deg"]
    j = human.merge(panos[cols], on="pano_id", how="inner")
    j = j[j["depression_deg"] >= 5.0].copy()
    bearing, _ = pov_from_pano_xy(j["pano_x"], j["pano_y"], j["pano_width"], j["pano_height"],
                                  j["camera_heading"])
    rel = np.radians(_wrap_deg(bearing - j["fresh_heading_deg"].to_numpy(float)))
    t_pitch = j["fresh_pitch_deg"].to_numpy(float) * np.cos(rel)
    t_roll = j["roll_wrapped_deg"].to_numpy(float) * np.sin(rel)
    dep = j["depression_deg"].to_numpy(float)
    truth = j["truth_m"].to_numpy(float)
    implied = truth * np.tan(np.radians(dep))
    sderr = j["sderr_approx3"].to_numpy(float)
    expected_dh_dT = truth / np.cos(np.radians(dep)) ** 2 * (np.pi / 180.0)  # m per degree of tilt
    # ... and the matching sensitivity of the DISTANCE: d(h / tan dep)/d(dep) = -h / sin^2(dep).
    h_ship = float(np.median(implied))
    expected_dd_dT = h_ship * (np.pi / 180.0) / np.sin(np.radians(dep)) ** 2

    def joint(y: np.ndarray) -> dict:
        X = np.column_stack([np.ones_like(t_pitch), t_pitch, t_roll])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((y - y.mean()) ** 2))
        return {"slope_pitch_m_per_deg": float(beta[1]), "slope_roll_m_per_deg": float(beta[2]),
                "r2": r2,
                "pearson_r_pitch_term": float(np.corrcoef(t_pitch, y)[0, 1]),
                "pearson_r_roll_term": float(np.corrcoef(t_roll, y)[0, 1])}

    # Bin by the projected tilt magnitude so a sign-symmetric effect cannot hide in a slope.
    t_mag = np.abs(t_pitch + t_roll)
    bins = [0.0, 0.5, 1.0, 2.0, 4.0, 90.0]
    cut = pd.cut(t_mag, bins, right=False)
    by_abs = []
    for iv, g in pd.DataFrame({"implied": implied, "sderr": sderr}).groupby(cut, observed=True):
        by_abs.append({"abs_projected_tilt_deg": f"{iv.left:g}-{iv.right:g}", "n": int(len(g)),
                       "implied_height_median_m": float(np.median(g["implied"])),
                       "approx3_signed_median_m": float(np.median(g["sderr"]))})
    pl = panos[panos["ground_tilt_deg"].notna()]
    return {
        "n_labels": int(len(j)), "n_panos": int(j["pano_id"].nunique()),
        "db_camera_roll_available": bool(human["camera_roll"].notna().any()),
        "abs_pitch_p50_p90_deg": [float(np.percentile(np.abs(panos["fresh_pitch_deg"]), 50)),
                                  float(np.percentile(np.abs(panos["fresh_pitch_deg"]), 90))],
        "abs_roll_p50_p90_deg": [float(np.percentile(np.abs(panos["roll_wrapped_deg"]), 50)),
                                 float(np.percentile(np.abs(panos["roll_wrapped_deg"]), 90))],
        "projected_tilt_sd_deg": float(np.std(t_pitch + t_roll)),
        "expected_slope_if_tilt_entered_m_per_deg": float(np.median(expected_dh_dT)),
        "expected_signed_error_slope_if_tilt_entered_m_per_deg": float(np.median(expected_dd_dT)),
        "implied_height": joint(implied),
        "approx3_signed_error": joint(sderr),
        "by_abs_projected_tilt": by_abs,
        "pano_level": {
            "n": int(len(pl)),
            "ground_tilt_median_deg": float(np.median(pl["ground_tilt_deg"])),
            "rig_tilt_magnitude_median_deg": float(np.median(pl["tilt_mag_deg"])),
            "pearson_r_ground_tilt_vs_rig_tilt": float(
                np.corrcoef(pl["ground_tilt_deg"], pl["tilt_mag_deg"])[0, 1]),
        },
    }


def modern_frame(shipped: dict, data_dir: str = DATA) -> tuple[pd.DataFrame, dict]:
    human = modern_predictions(load_modern(data_dir), shipped)
    head = human[human["stratum"] == "representative"]

    def block(frame: pd.DataFrame) -> dict:
        o = {k: _stats(frame[v].to_numpy(float), frame[MODERN_SIGNED[k]].to_numpy(float))
             for k, v in MODERN_MODELS.items()}
        for k in MODERN_MODELS:
            if k != "A_deployed":
                o[k]["win_rate_vs_A"] = float(np.mean(frame[MODERN_MODELS[k]] < frame["err_A"]))
        return o

    summary = {
        "n_human_gated": int(len(human)), "n_representative": int(len(head)),
        "n_cities": int(human["city"].nunique()),
        "representative": block(head), "pooled": block(human),
        "bootstrap_median_diff_vs_A": {
            "representative": cluster_bootstrap_median_diff(
                head["err_approx3"].to_numpy(float), head["err_A"].to_numpy(float),
                head["pano_id"].to_numpy(str)),
            "pooled": cluster_bootstrap_median_diff(
                human["err_approx3"].to_numpy(float), human["err_A"].to_numpy(float),
                human["pano_id"].to_numpy(str))},
        "by_zoom": slice_table(human, MODERN_MODELS, "zoom_i", signed=MODERN_SIGNED,
                               reference="A_deployed"),
        "by_label_type": slice_table(human, MODERN_MODELS, "label_type", signed=MODERN_SIGNED,
                                     reference="A_deployed"),
        "by_city": slice_table(human, MODERN_MODELS, "city", min_n=50, signed=MODERN_SIGNED,
                               reference="A_deployed"),
        "by_capture_year": slice_table(human, MODERN_MODELS, "capture_year", min_n=50,
                                       signed=MODERN_SIGNED, reference="A_deployed"),
        "by_pano_height": slice_table(human, MODERN_MODELS, "pano_height_px", min_n=50,
                                      signed=MODERN_SIGNED, reference="A_deployed"),
        "by_true_distance": slice_table(human, MODERN_MODELS, "dist_bin", signed=MODERN_SIGNED,
                                        reference="A_deployed"),
        "by_depression": slice_table(human, MODERN_MODELS, "dep_bin", min_n=10, signed=MODERN_SIGNED,
                                     reference="A_deployed"),
        "repeated_holdout": repeated_holdout(human, shipped),
        "leave_one_city_out": leave_one_city_out(human, shipped),
        "rig_tilt_rider": rig_tilt_rider(human),
    }
    return human, summary


# --------------------------------------------------------------------------- geodesy

GEODESY_DISTANCES_M = [1.0, 2.0, 5.0, 10.0, 11.770106120938644, 15.0, 20.0,
                       23.848261259830384, 30.0, 50.0]
MAX_ANSWER_M = 23.848261259830384


def geodesy_displacements(latitudes: dict[str, float], distances=GEODESY_DISTANCES_M,
                          bearing_step_deg: float = 5.0) -> dict:
    """How far the production sphere's destination point sits from the alternatives.

    For every city latitude, distance and bearing: the WGS84 geodesic destination (pyproj) vs
    the 6371 km sphere the Scala/SQL paths use; the client's turf sphere (6371.0088 km) vs
    that; and the harness's geosphere sphere (6378.137 km) vs that. Separations are measured
    as geodesic distances between the two destination points, in meters."""
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    bearings = np.arange(0.0, 360.0, bearing_step_deg)
    lng0 = 0.0
    per_city = []
    for city, lat0 in sorted(latitudes.items(), key=lambda kv: kv[1]):
        rows = []
        for d in distances:
            lat_p, lng_p = destination(lat0, lng0, d, bearings, R_PRODUCTION_M)
            lng_g, lat_g, _ = geod.fwd(np.full_like(bearings, lng0), np.full_like(bearings, lat0),
                                       bearings, np.full_like(bearings, d))
            lat_t, lng_t = destination(lat0, lng0, d, bearings, R_TURF_M)
            lat_h, lng_h = destination(lat0, lng0, d, bearings, R_HARNESS_M)
            _, _, sep_g = geod.inv(lng_p, lat_p, lng_g, lat_g)
            _, _, sep_t = geod.inv(lng_p, lat_p, lng_t, lat_t)
            _, _, sep_h = geod.inv(lng_p, lat_p, lng_h, lat_h)
            rows.append({"distance_m": d,
                         "ellipsoid_vs_production_max_m": float(np.max(sep_g)),
                         "ellipsoid_vs_production_bearing_of_max": float(bearings[int(np.argmax(sep_g))]),
                         "turf_vs_production_max_m": float(np.max(sep_t)),
                         "harness_vs_production_max_m": float(np.max(sep_h))})
        per_city.append({"city": city, "latitude": lat0, "rows": rows})
    # Closed-form reading: the sphere's radius error against the local radii of curvature.
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    curvature = []
    for city, lat0 in sorted(latitudes.items(), key=lambda kv: kv[1]):
        s = np.sin(np.radians(lat0))
        m = a * (1 - e2) / (1 - e2 * s * s) ** 1.5
        n = a / np.sqrt(1 - e2 * s * s)
        curvature.append({"city": city, "latitude": lat0, "meridional_radius_m": float(m),
                          "prime_vertical_radius_m": float(n),
                          "north_south_scale_error": float(R_PRODUCTION_M / m - 1),
                          "east_west_scale_error": float(R_PRODUCTION_M / n - 1)})
    worst = max(r["ellipsoid_vs_production_max_m"] for c in per_city for r in c["rows"]
                if abs(r["distance_m"] - MAX_ANSWER_M) < 1e-9)
    return {"radii_m": {"production_scala_sql": R_PRODUCTION_M, "client_turf": R_TURF_M,
                        "harness_geosphere": R_HARNESS_M},
            "per_city": per_city, "curvature": curvature,
            "worst_ellipsoid_vs_production_at_max_answer_m": float(worst)}


# --------------------------------------------------------------- viewport frame contract

def canvas_to_centered_pov(pov_heading, pov_pitch, zoom, canvas_x, canvas_y, width, height):
    """util.pano.canvasCoordToCenteredPov with the frame passed explicitly (the #5085 contract)."""
    return pov_if_centered(canvas_x, canvas_y, pov_heading, pov_pitch, zoom, width, height)


def centered_pov_to_canvas(target_heading, target_pitch, pov_heading, pov_pitch, zoom, width, height):
    """Forward projection: where a direction lands on a frame (inverse of the above)."""
    fov = np.radians(get_3d_fov(zoom))
    f = 0.5 * width / np.tan(0.5 * fov)
    h0, p0 = np.radians(pov_heading), np.radians(pov_pitch)
    h, p = np.radians(np.asarray(target_heading, float)), np.radians(np.asarray(target_pitch, float))
    x = f * np.cos(p) * np.sin(h)
    y = f * np.cos(p) * np.cos(h)
    z = f * np.sin(p)
    # rotate into the viewport frame
    y_v = x * np.sin(h0) * np.cos(p0) + y * np.cos(h0) * np.cos(p0) + z * np.sin(p0)
    x_v = x * np.cos(h0) - y * np.sin(h0)
    z_v = -x * np.sin(h0) * np.sin(p0) - y * np.cos(h0) * np.sin(p0) + z * np.cos(p0)
    s = f / y_v
    return width / 2 + s * x_v, height / 2 - s * z_v


FRAMES = {"720x480 (today)": (720, 480), "1280x720": (1280, 720), "1920x1080": (1920, 1080),
          "2560x1080 (21:9)": (2560, 1080), "1440x1080 (4:3)": (1440, 1080)}


def viewport_frame_contract(shipped: dict, pano_lat: float = 47.6553, pano_lng: float = -122.3035) -> dict:
    """The estimator only sees angles. Project one set of label directions onto frames of
    several sizes and aspects, invert each click through ITS OWN frame, and the position is
    identical to the bit. Then invert through the wrong frame -- the constant 720x480 the
    codebase assumes today -- under the two conventions a naive port would use, and report the
    position error in meters. This is why SidewalkWebpage#5085 stores the frame per label."""
    pov_h, pov_p, zoom = 40.0, -8.0, 1.0
    rng = np.random.default_rng(666)
    n = 400
    t_h = pov_h + rng.uniform(-35, 35, n)
    t_p = rng.uniform(-30, -3, n)
    ok = None
    results = {}
    for name, (w, h) in FRAMES.items():
        cx, cy = centered_pov_to_canvas(t_h, t_p, pov_h, pov_p, zoom, w, h)
        inside = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        ok = inside if ok is None else (ok & inside)
        results[name] = (cx, cy)
    t_h, t_p = t_h[ok], t_p[ok]
    dist_true = estimate_distance_m(-t_p, shipped)
    lat_true, lng_true = destination(pano_lat, pano_lng, dist_true, t_h)
    out = {"n_directions": int(ok.sum()), "viewport": {"heading": pov_h, "pitch": pov_p, "zoom": zoom},
           "frames": []}
    for name, (w, h) in FRAMES.items():
        cx, cy = results[name][0][ok], results[name][1][ok]
        hh, pp = canvas_to_centered_pov(pov_h, pov_p, zoom, cx, cy, w, h)
        lat_o, lng_o = destination(pano_lat, pano_lng, estimate_distance_m(-pp, shipped), hh)
        own = haversine_m(lng_true, lat_true, lng_o, lat_o)
        # wrong frame A: coordinates scaled axis-by-axis into 720x480 (aspect distorted)
        hh, pp = canvas_to_centered_pov(pov_h, pov_p, zoom, cx * CANVAS_W / w, cy * CANVAS_H / h,
                                        CANVAS_W, CANVAS_H)
        lat_a, lng_a = destination(pano_lat, pano_lng, estimate_distance_m(-pp, shipped), hh)
        scaled = haversine_m(lng_true, lat_true, lng_a, lat_a)
        # wrong frame B: scaled by width only (aspect kept), then read with the 480 height
        k = CANVAS_W / w
        hh, pp = canvas_to_centered_pov(pov_h, pov_p, zoom, cx * k, cy * k, CANVAS_W, CANVAS_H)
        lat_b, lng_b = destination(pano_lat, pano_lng, estimate_distance_m(-pp, shipped), hh)
        width_scaled = haversine_m(lng_true, lat_true, lng_b, lat_b)
        out["frames"].append({
            "frame": name, "width": w, "height": h,
            "own_frame_max_error_m": float(np.max(own)),
            "axis_scaled_to_720x480": _stats(scaled),
            "width_scaled_read_as_720x480": _stats(width_scaled),
        })
    return out


# ------------------------------------------------------------- cross-implementation fixture

def parity_fixture(shipped: dict, n_random: int = 48, seed: int = 5084) -> dict:
    """Inputs and reference outputs for the Scala, JS and SQL implementations to reproduce.

    Reference values come from the Python port above on the production sphere. Edge cases
    are listed explicitly: the seam (pano_x = 0 and width - 1), a negative unwrapped heading,
    a click exactly at the blend angle, above-horizon clicks (the bounded tail), the nadir."""
    rng = np.random.default_rng(seed)
    sizes = [(16384, 8192), (13312, 6656), (5760, 2880), (8192, 4096)]
    cities = [(47.6553, -122.3035), (38.9072, -77.0369), (19.4326, -99.1332), (-23.5505, -46.6333),
              (52.3676, 4.9041), (40.8859, -74.0143), (-41.2865, 174.7762), (1.3521, 103.8198)]
    cases = []

    def add(name, lat, lng, x, y, w, h, ch):
        la, lo, d, hd, pt = production_to_latlng(lat, lng, x, y, w, h, ch, shipped)
        cases.append({"name": name, "pano_lat": float(lat), "pano_lng": float(lng), "pano_x": int(x),
                      "pano_y": int(y), "pano_width": int(w), "pano_height": int(h),
                      "camera_heading": float(ch),
                      "expected": {"heading_deg": float(np.mod(hd, 360.0)), "pitch_deg": float(pt),
                                   "distance_m": float(d), "lat": float(la), "lng": float(lo)}})

    add("center column, 22.5 deg down", 47.6553, -122.3035, 6656, 4160, 13312, 6656, 90.0)
    add("seam: pano_x = 0", 47.6553, -122.3035, 0, 5000, 16384, 8192, 10.0)
    add("seam: pano_x = width - 1", 47.6553, -122.3035, 16383, 5000, 16384, 8192, 10.0)
    add("negative unwrapped heading", 47.6553, -122.3035, 1664, 4160, 13312, 6656, 10.0)
    add("exactly the blend angle", 38.9072, -77.0369, 8192, 4608, 16384, 8192, 200.0)
    add("just below the blend angle", 38.9072, -77.0369, 8192, 4607, 16384, 8192, 200.0)
    add("at the horizon (tail max)", 19.4326, -99.1332, 4000, 4096, 16384, 8192, 300.0)
    add("above the horizon (clamped to the tail max)", 19.4326, -99.1332, 4000, 3000, 16384, 8192, 300.0)
    add("nadir", -23.5505, -46.6333, 123, 8191, 16384, 8192, 45.0)
    add("southern hemisphere, east of 180", -41.2865, 174.7762, 12000, 5200, 16384, 8192, 359.9)
    for i in range(n_random):
        lat, lng = cities[i % len(cities)]
        w, h = sizes[i % len(sizes)]
        add(f"random {i}", lat + rng.uniform(-0.05, 0.05), lng + rng.uniform(-0.05, 0.05),
            int(rng.integers(0, w)), int(rng.integers(int(0.45 * h), int(0.9 * h))), w, h,
            float(rng.uniform(0, 360)))
    return {"description": "Cross-implementation parity fixture for the approximation3 lat/lng "
                           "estimator (SidewalkWebpage#5084). Reference: python/signoff.py "
                           "production_to_latlng on the 6371 km production sphere.",
            "constants": {"camera_height_m": shipped["height_m"], "blend_deg": shipped["blend_deg"],
                          "max_distance_m": MAX_DISTANCE_M, "earth_radius_m": R_PRODUCTION_M},
            "tolerance": {"lat_lng_deg": 1e-9, "distance_m": 1e-9, "heading_deg": 1e-9},
            "cases": cases}


# ------------------------------------------------------------------------ examples

def pick_examples(human: pd.DataFrame, panos: pd.DataFrame) -> pd.DataFrame:
    """Four labels for the worked examples, chosen by rule rather than by hand: a close curb
    ramp, a mid-range obstacle-class label, a far near-horizon label, and a zoom-3 label --
    all representative-stratum human rows on panos captured 2022 or later (so the imagery
    is likely still served), where the regression misses by more than a metre."""
    ok = panos[panos["status"] == "ok"]["pano_id"]
    pool = human[human["pano_id"].isin(ok) & (human["capture_year"].astype(int) >= 2022)
                 & (human["stratum"] == "representative")].copy()
    pool["gain"] = pool["err_A"] - pool["err_approx3"]
    picks = []
    rules = [
        ("close curb ramp", (pool["label_type"] == "CurbRamp") & pool["depression_deg"].between(15, 30)),
        ("mid-range obstacle or surface problem",
         pool["label_type"].isin(["Obstacle", "SurfaceProblem"]) & pool["depression_deg"].between(8, 14)),
        ("far, near the horizon", pool["depression_deg"].between(3, 6)),
        ("zoom 3", pool["zoom_i"] == 3),
    ]
    for role, mask in rules:
        taken = [p["label_uid"] for p in picks]
        cand = pool[mask & (pool["err_A"] > 1.0) & (pool["err_approx3"] < 0.5) & ~pool["label_uid"].isin(taken)]
        if cand.empty:
            cand = pool[mask & ~pool["label_uid"].isin(taken)]
        best = cand.sort_values("gain", ascending=False).iloc[0]
        picks.append({"role": role, **best.to_dict()})
    return pd.DataFrame(picks)
