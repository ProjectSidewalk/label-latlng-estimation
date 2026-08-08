"""Issue #3 modern-truth validation: post-2021 labels scored against fresh GSV depth.

The Mapillary falsification (Stage 3, reports/2026-08-07-mapillary-falsification.md §8)
is scale-free by construction: self-consistency provably cannot see a shared scale error.
This module supplies the missing absolute check. Post-2021 labels' stored pano_x/pano_y
replay the front-end projection at 100.0000% (data/pov-inversion-summary.json), and both
pano_x and the depth raster are heading-centred (reports/2026-08-06-depth-coordinate-
conventions.md §1), so a stored pixel indexes a freshly fetched depth map directly:

    col = round(pano_x / pano_width * 512) % 512      # NO mirror, NO yaw rotation
    row = clamp(round(pano_y / pano_height * 256), 0, 255)

The no-mirror half is specific to gsv_depth's arrays, which keep the payload's own x
order; the conventions report's ``511 - round(...)`` recipe is for streetlevel's parsed
array, which mirrors x on output (§2). Truth is the ray's horizontal ground distance,
which is camera-relative and therefore immune to pano re-registration drift; no camera
height and no yaw enter it.

Population and provenance facts this module relies on (verified against production
2026-08-07, recorded in scripts/extraction/extract-modern-labels.sql):

- every post-2021 non-tutorial row is computation_method 'approximation2', i.e. stored
  lat/lng IS the deployed estimator's output — usable as the deployed prediction and as
  the circularity guard's check value, never as truth;
- AI-submitted labels are exactly the 'SidewalkAI' user's (all in vancouver); the
  ``is_ai`` flag lets human clicks and AI detections be scored separately;
- the era heading-bias constant is a 2017-2020 ground-truth property and does NOT apply
  to any row here (all post-evolution-179).

Scoring is in distance space, not lat/lng space: every candidate shares the exact POV
heading (#5), so distance is where the models differ, and it keeps the truth free of the
azimuth-frame question. No parameter is fitted anywhere in this module — blend D's
coefficients are read from the committed data/distance-refit-summary.json and scored
as-is (pure held-out validation).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import depth_validation as dv
import gsv_depth as gd
from label_latlng_estimation import MAX_LABELS_PER_PANO
from mapillary_falsification import CALIBRATION_HEIGHT, COT_CAMERA_HEIGHT
from distance_refit import DIST_CAP_M, predict_dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 666  # the repo's sampling seed everywhere

# Selection budgets (fetch successes, i.e. panos that resolve AND serve depth). Strata are
# walked in this order and each is oversampled 2x into a ranked candidate list, so ordinary
# attrition (gone panos, no-depth) is absorbed without starving the later strata.
REPRESENTATIVE_PANOS = 700   # uniform draw over human panos; the headline-metric population
NEAR_HORIZON_PANOS = 200     # panos containing a label at depression <= NEAR_HORIZON_DEG
TYPE_LABEL_QUOTA = 200       # per-type top-up target, counting labels earlier strata supply
AI_PANOS = 100               # SidewalkAI (vancouver) detections, scored separately
TARGET_PANOS = 1500          # overall fetch-success budget
OVERSAMPLE = 2.0
NEAR_HORIZON_DEG = 2.0
CITY_PANO_CAP = 150          # representative-stratum cap; no city dominates the headline

DEPTH_W, DEPTH_H = gd.DEPTH_W, gd.DEPTH_H

# Truth gates (recorded before filtering; thresholds locked by the contract tests).
NEIGHBOURHOOD_RATIO_BAND = (2.0 / 3.0, 1.5)  # ray must agree with its 3x3 neighbourhood
TRUTH_MAX_M = DIST_CAP_M                     # beyond the estimator cap, truth can't score it

# The circularity guard's echo criterion: a stored distance is an estimator echo when
# its era's formula reproduces it within this. Loose enough to absorb pano-origin drift
# (GSV re-registers panos ~0.8 m median; stored positions were computed from the
# at-insert origin, which is unobservable), tight enough that a genuinely moved or
# hand-adjusted label fails it. The full |diff| distribution is reported alongside.
GUARD_ECHO_M = 0.5

# v7.12.2 / evolution 179 (2023-03-29): the front end switched from fixed-frame
# sv_image_y to real pano_y pixels. Labels placed BEFORE it were computed in the
# coefficients' own 6656 frame (no #4765 apply-path defect); labels after it feed real
# pixels into fixed-frame coefficients (the defect population). The guard recomputes
# each row under its own era's formula — and the 2 m era discontinuity this leaves in
# stored positions is itself a finding.
EVOLUTION_179_UTC = pd.Timestamp("2023-03-29", tz="UTC")

MODEL_KEYS = ["A_deployed", "B_normalized", "C_anchor", "D_blend"]


# ---------------------------------------------------------------------------- extraction

def load_extraction(extract_dir: str) -> pd.DataFrame:
    """All modern-labels-*.csv.gz as one frame, with a city column and typed flags.

    ``label_id`` is a PER-SCHEMA serial: each city's ``label`` table numbers from 1, so
    concatenating 49 cities collides 76% of the rows (up to 33 ways on one id). Joining
    on it alone cross-joins labels between cities and pairs a label with another city's
    depth truth. ``label_uid`` is the real key and is what every downstream join uses.
    """
    paths = sorted(glob.glob(os.path.join(extract_dir, "modern-labels-*.csv.gz")))
    if not paths:
        raise FileNotFoundError(f"no modern-labels-*.csv.gz under {extract_dir}; "
                                "run scripts/extraction/extract-modern-labels.sh")
    frames = []
    for path in paths:
        city = os.path.basename(path)[len("modern-labels-"):-len(".csv.gz")]
        df = pd.read_csv(path, dtype={"pano_id": str})
        df["city"] = city
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    is_ai = df["is_ai"].map({"t": True, "f": False})
    if is_ai.isna().any():  # a silent .astype(bool) would read every NaN as an AI label
        raise ValueError(f"{int(is_ai.isna().sum())} rows carry an unparseable is_ai flag")
    df["is_ai"] = is_ai.astype(bool)
    df["label_uid"] = df["city"] + ":" + df["label_id"].astype(str)
    if not df["label_uid"].is_unique:
        raise ValueError("label_uid is not unique — (city, label_id) is no longer a key")
    df["time_created"] = pd.to_datetime(df["time_created"], utc=True, format="mixed")
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    return df


def add_depression(df: pd.DataFrame) -> pd.DataFrame:
    """The resolution-independent depression angle, positive below the horizon.

    This is the provisional-coefficients pixel->angle conversion verbatim
    ((pano_y - height/2) * 180 / height) — the same expression a production caller
    applies, so scoring consumes exactly the deployed conversion."""
    out = df.copy()
    out["depression_deg"] = ((out["pano_y"] - out["pano_height"] / 2.0)
                             * 180.0 / out["pano_height"])
    return out


def frame_census(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sampling-frame gates with a census of what each one removes.

    In-raster pixels and finite stored coordinates are prerequisites for scoring at all;
    the labels-per-pano cap is the 2021 cleaning's guard against pathological panos (the
    modern population contains one pano carrying 11,128 labels)."""
    in_y = (df["pano_y"] >= 0) & (df["pano_y"] < df["pano_height"])
    in_x = (df["pano_x"] >= 0) & (df["pano_x"] < df["pano_width"])
    finite = (df["lat"].between(-90, 90) & df["lng"].between(-180, 180)
              & df["lat"].notna() & df["lng"].notna())
    per_pano = df.groupby("pano_id")["label_id"].transform("size")
    overloaded = per_pano > MAX_LABELS_PER_PANO
    keep = in_y & in_x & finite & ~overloaded
    census = {
        "rows": int(len(df)),
        "panos": int(df["pano_id"].nunique()),
        "pano_y_out_of_raster": int((~in_y).sum()),
        "pano_x_out_of_raster": int((~in_x).sum()),
        "stored_latlng_absurd": int((~finite).sum()),
        "over_pano_cap": int(overloaded.sum()),
        "max_labels_per_pano": int(MAX_LABELS_PER_PANO),
        # why label_uid exists: label_id is a per-schema serial, so this many rows share
        # one with another city and any join keyed on it alone would cross-join them
        "rows_sharing_a_label_id": int(len(df) - df["label_id"].nunique()),
        "kept_rows": int(keep.sum()),
        "kept_panos": int(df.loc[keep, "pano_id"].nunique()),
        "ai_rows": int(df.loc[keep, "is_ai"].sum()),
    }
    return df[keep].copy(), census


# ---------------------------------------------------------------------------- selection

def pano_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pano: city, label counts (total, per type), AI flag, min depression."""
    g = df.groupby("pano_id")
    tab = pd.DataFrame({
        "city": g["city"].first(),
        "n_labels": g.size(),
        "any_ai": g["is_ai"].any(),
        "min_dep": g["depression_deg"].min(),
    })
    for t in sorted(df["label_type"].unique()):
        n = df[df["label_type"] == t].groupby("pano_id").size()
        tab["n_" + t] = n.reindex(tab.index, fill_value=0).astype(int)
    return tab.reset_index()


def select_panos(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """The ranked fetch plan: (pano_id, stratum, rank), deterministic under ``seed``.

    Strata in fetch order, each oversampled 2x against attrition:

    - representative: uniform over human panos with a per-city cap, so the headline
      numbers come from an approximately proportional draw rather than the quota strata;
    - near_horizon: panos holding a label at depression <= 2 deg (the regime where the
      blend's clamp and the deployed model's runaway actually differ);
    - type:<T>: rarest-first top-ups until each label type reaches TYPE_LABEL_QUOTA
      labels counting what earlier strata already supply (Crosswalk and Signal were
      never fitted — this is height_fallback_m's first contact with real data);
    - ai: SidewalkAI panos, scored separately from human clicks.
    """
    rng = np.random.default_rng(seed)
    tab = pano_table(df)
    tab["rand"] = rng.random(len(tab))
    type_cols = [c for c in tab.columns if c.startswith("n_") and c != "n_labels"]
    picked: dict[str, str] = {}  # pano_id -> stratum, first pick wins

    def take(cand: pd.DataFrame, n: int, stratum: str):
        cand = cand[~cand["pano_id"].isin(picked)]
        for pid in cand["pano_id"].head(n):
            picked[pid] = stratum

    human = tab[~tab["any_ai"]]

    # representative: uniform, but no city beyond CITY_PANO_CAP of the (oversampled) draw
    rep = human.sort_values("rand", kind="stable")
    rep = rep.groupby("city", group_keys=False).head(CITY_PANO_CAP)
    take(rep.sort_values("rand", kind="stable"),
         int(REPRESENTATIVE_PANOS * OVERSAMPLE), "representative")

    nh = human[human["min_dep"] <= NEAR_HORIZON_DEG].sort_values("rand", kind="stable")
    take(nh, int(NEAR_HORIZON_PANOS * OVERSAMPLE), "near_horizon")

    # rarest types first, so the scarce candidates aren't consumed as by-catch
    availability = {c: int(tab[c].sum()) for c in type_cols}
    for col in sorted(type_cols, key=lambda c: availability[c]):
        t = col[2:]
        covered = int(tab.loc[tab["pano_id"].isin(picked), col].sum())
        need_labels = int(TYPE_LABEL_QUOTA * OVERSAMPLE) - covered
        if need_labels <= 0:
            continue
        cand = (human[human[col] > 0]
                .sort_values([col, "rand"], ascending=[False, True], kind="stable"))
        cand = cand[~cand["pano_id"].isin(picked)]
        got, ids = 0, []
        for pid, n_t in zip(cand["pano_id"], cand[col]):
            if got >= need_labels:
                break
            ids.append(pid)
            got += int(n_t)
        for pid in ids:
            picked[pid] = f"type:{t}"

    ai = tab[tab["any_ai"]].sort_values("rand", kind="stable")
    take(ai, int(AI_PANOS * OVERSAMPLE), "ai")

    plan = pd.DataFrame({"pano_id": list(picked), "stratum": list(picked.values())})
    plan["rank"] = np.arange(len(plan))
    return plan


# ---------------------------------------------------------------------------- depth sampling

def modern_col_row(pano_x, pano_y, pano_width, pano_height):
    """Depth-raster cell for a stored post-evolution-179 pixel (module docstring math)."""
    col = int(round(pano_x / pano_width * DEPTH_W)) % DEPTH_W
    row = min(max(int(round(pano_y / pano_height * DEPTH_H)), 0), DEPTH_H - 1)
    return col, row


def control_col_row(col: int, row: int, control: str) -> tuple[int, int]:
    """(col, row) re-read under a deliberately wrong frame — the pilot's null hypotheses.

    Index transforms equivalent to dv.apply_frame_control on the raster: reading
    grid[row, col] of the transformed grid equals reading the original at these."""
    if control == "identity":
        return col, row
    if control == "x_mirror":
        return DEPTH_W - 1 - col, row
    if control == "rotate_180":
        return DEPTH_W - 1 - col, DEPTH_H - 1 - row
    if control == "row_flip":
        return col, DEPTH_H - 1 - row
    raise ValueError(f"unknown frame control {control!r}")


def classify_modern_label(payload: gd.DepthPayload, pano_x, pano_y, pano_width,
                          pano_height, camera_height: float,
                          geometry: dv.PayloadGeometry | None = None,
                          control: str = "identity") -> dv.LabelHit:
    """Depth hit for one modern label via the heading-centred lookup (no mirror, no yaw)."""
    col, row = modern_col_row(pano_x, pano_y, pano_width, pano_height)
    col, row = control_col_row(col, row, control)
    return dv.classify_depth_pixel(payload, col, row, camera_height, geometry)


# ---------------------------------------------------------------------------- models

# The deployed per-zoom distance coefficients, verbatim from SidewalkWebpage's
# PanoDataService.scala (LATLNG_ESTIMATION_PARAMS; selected there by round(zoom)).
# mapillary_falsification carries only the zoom-1 triple because the auto-labeler
# always submits at zoom 1; human labels use all three, and the circularity guard
# proves the selection empirically (a zoom-1-only recompute leaves 0.90 m / 2.79 m
# median residuals on the zoom-2/3 populations; per-zoom collapses them to 0.16/0.15).
DEPLOYED_DIST_COEF = {
    1: (18.6051843, 0.0138947, 0.0011023),
    2: (20.8794248, 0.0184087, 0.0022135),
    3: (25.2472682, 0.0264216, 0.0011071),
}


def _zoom_coef(zoom) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.clip(np.round(np.asarray(zoom, float)).astype(int), 1, 3)
    table = np.array([DEPLOYED_DIST_COEF[i] for i in (1, 2, 3)])
    coef = table[z - 1]
    return coef[..., 0], coef[..., 1], coef[..., 2]


def deployed_distance(pano_y, pano_height, canvas_y, zoom) -> np.ndarray:
    """The deployed apply path: real raster pixels into the fixed-frame per-zoom
    coefficients, floored at zero (PanoDataService.toLatLng, verbatim)."""
    intercept, pano_slope, canvas_slope = _zoom_coef(zoom)
    offset_px = np.asarray(pano_height, float) / 2.0 - np.asarray(pano_y, float)
    return np.maximum(0.0, intercept + pano_slope * offset_px
                      + canvas_slope * np.asarray(canvas_y, float))


def normalized_distance(dep_deg, canvas_y, zoom) -> np.ndarray:
    """SidewalkWebpage#4765's one-line fix in isolation: the same coefficients fed
    fixed-frame (6656-calibrated) pixels instead of real ones."""
    intercept, pano_slope, canvas_slope = _zoom_coef(zoom)
    offset_px = -np.asarray(dep_deg, float) / 180.0 * CALIBRATION_HEIGHT
    return np.maximum(0.0, intercept + pano_slope * offset_px
                      + canvas_slope * np.asarray(canvas_y, float))


def anchor_distance(dep_deg) -> np.ndarray:
    """The zero-parameter anchor: 2.6 m over tan(depression), capped like every rung."""
    dep = np.asarray(dep_deg, float)
    with np.errstate(divide="ignore", over="ignore"):
        cot = np.where(dep > 0, COT_CAMERA_HEIGHT / np.tan(np.radians(dep)), np.inf)
    return np.clip(cot, 0.0, DIST_CAP_M)


def load_blend_params(data_dir: str = os.path.join(ROOT, "data")) -> dict:
    """The ERA fit's committed coefficients — the candidate this module validates.

    Deliberately the era fit, not the calibrated production constants: D_blend is scored
    exactly as the refit shipped it, and ``final_coefficients`` is derived here FROM that
    comparison (scoring the calibrated form against the truth that calibrated it would be
    circular as a validation, so it appears only in the remedy/final blocks)."""
    with open(os.path.join(data_dir, "distance-refit-summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    key = ("era_fit_coefficients" if "era_fit_coefficients" in summary
           else "provisional_coefficients")
    return summary[key]["params"]


def model_predictions(df: pd.DataFrame, blend_params: dict) -> pd.DataFrame:
    """All four candidates on the scored rows; requires depression_deg."""
    out = df.copy()
    out["A_deployed"] = deployed_distance(df["pano_y"], df["pano_height"],
                                          df["canvas_y"], df["zoom"])
    out["B_normalized"] = normalized_distance(df["depression_deg"], df["canvas_y"],
                                              df["zoom"])
    out["C_anchor"] = anchor_distance(df["depression_deg"])
    out["D_blend"] = predict_dist(blend_params, df)
    return out


# ---------------------------------------------------------------------------- guard

def stored_distance(df: pd.DataFrame) -> np.ndarray:
    """Haversine pano origin -> stored label position; the distance production stored."""
    from label_latlng_estimation import haversine_m

    return haversine_m(df["pano_lng"].to_numpy(float), df["pano_lat"].to_numpy(float),
                       df["lng"].to_numpy(float), df["lat"].to_numpy(float))


def guard_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The circularity guard's per-row fields.

    Stored lat/lng post-2021 is estimator output in BOTH eras — the fixed-frame
    front-end formula before evolution 179, the real-pixel apply path after it — so the
    stored pano->label distance must reproduce from its own era's formula. Where it
    does (``is_echo``), any evaluation that treats stored positions as truth is
    self-grading — the check this validation exists to escape. The A_deployed
    prediction scored against truth is always the CURRENT apply path (what production
    computes today, and what a backfill recompute would write), independent of era."""
    out = df.copy()
    out["stored_dist_m"] = stored_distance(df)
    out["era"] = np.where(df["time_created"] < EVOLUTION_179_UTC,
                          "fixed_frame", "real_pixels")
    era_recompute = np.where(
        out["era"] == "fixed_frame",
        normalized_distance(df["depression_deg"], df["canvas_y"], df["zoom"]),
        deployed_distance(df["pano_y"], df["pano_height"], df["canvas_y"], df["zoom"]))
    out["guard_diff_m"] = out["stored_dist_m"] - era_recompute
    # the deliberately-wrong-era recompute, so the summary can show the separation
    cross = np.where(
        out["era"] == "real_pixels",
        normalized_distance(df["depression_deg"], df["canvas_y"], df["zoom"]),
        deployed_distance(df["pano_y"], df["pano_height"], df["canvas_y"], df["zoom"]))
    out["guard_cross_diff_m"] = out["stored_dist_m"] - cross
    out["is_echo"] = out["guard_diff_m"].abs() <= GUARD_ECHO_M
    return out


def guard_summary(labels: pd.DataFrame) -> dict:
    out = {"echo_threshold_m": GUARD_ECHO_M,
           "evolution_179_utc": str(EVOLUTION_179_UTC.date()),
           "frac_echo": float(labels["is_echo"].mean()) if len(labels) else None}
    for era, sub in labels.groupby("era"):
        d = sub["guard_diff_m"].astype(float)
        d = d[np.isfinite(d)]
        x = sub["guard_cross_diff_m"].astype(float)
        x = x[np.isfinite(x)]
        out[str(era)] = {
            "n": int(len(d)),
            "median_abs_diff_m": float(d.abs().median()) if len(d) else None,
            "p90_abs_diff_m": float(np.percentile(d.abs(), 90)) if len(d) else None,
            "frac_echo": float(sub["is_echo"].mean()),
            "wrong_era_median_abs_m": float(x.abs().median()) if len(x) else None,
        }
    return out


# ---------------------------------------------------------------------------- truth gates

def truth_gates(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-row gate flags plus the exclusion census. All rows keep their flags — nothing
    is dropped here — and ``gate_ok`` marks the scored population."""
    out = labels.copy()
    lo, hi = NEIGHBOURHOOD_RATIO_BAND
    hit_ok = out["hit_class"].isin(["ground", "terrain"])
    ratio = out["neighbourhood_range_ratio"].astype(float)
    ratio_ok = ratio.between(lo, hi) | ~np.isfinite(ratio)  # lone finite ray keeps its say
    truth = out["truth_m"].astype(float)
    cap_ok = np.isfinite(truth) & (truth < TRUTH_MAX_M)
    out["gate_hit"] = hit_ok
    out["gate_ratio"] = ratio_ok
    out["gate_cap"] = cap_ok
    out["gate_ok"] = hit_ok & ratio_ok & cap_ok
    census = {
        "rows": int(len(out)),
        "hit_class": {str(k): int(v) for k, v in
                      out["hit_class"].value_counts(dropna=False).items()},
        "failed_hit": int((~hit_ok).sum()),
        "failed_ratio": int((hit_ok & ~ratio_ok).sum()),
        "failed_cap": int((hit_ok & ratio_ok & ~cap_ok).sum()),
        "gate_ok": int(out["gate_ok"].sum()),
        "ratio_band": list(NEIGHBOURHOOD_RATIO_BAND),
        "truth_max_m": TRUTH_MAX_M,
    }
    return out, census


# ---------------------------------------------------------------------------- scoring

def range_slope(err: np.ndarray, truth: np.ndarray) -> float:
    """OLS slope of signed error on true distance — the compression statistic the
    Mapillary falsification ran scale-free, here against absolute truth."""
    t = truth - truth.mean()
    return float((t * (err - err.mean())).sum() / (t ** 2).sum())


def model_metrics(scored: pd.DataFrame) -> dict:
    truth = scored["truth_m"].to_numpy(float)
    out = {"n": int(len(scored))}
    for k in MODEL_KEYS:
        err = scored[k].to_numpy(float) - truth
        out[k] = {
            "median_abs_m": float(np.median(np.abs(err))),
            "signed_median_m": float(np.median(err)),
            "p90_abs_m": float(np.percentile(np.abs(err), 90)),
            "range_slope_m_per_m": range_slope(err, truth) if len(scored) >= 3 else None,
        }
    return out


def by_group_metrics(scored: pd.DataFrame, col: str, min_n: int = 25) -> dict:
    return {str(g): model_metrics(sub)
            for g, sub in scored.groupby(col) if len(sub) >= min_n}


NEAR_HORIZON_BIN_EDGES = (-90.0, 0.0, 2.0, 5.0, 11.25, 90.0)  # last interior edge = blend_deg


def near_horizon_metrics(scored: pd.DataFrame) -> dict:
    out = {}
    edges = NEAR_HORIZON_BIN_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = scored[(scored["depression_deg"] > lo) & (scored["depression_deg"] <= hi)]
        if len(sub):
            out[f"({lo:g}, {hi:g}]"] = model_metrics(sub)
    return out


def implied_heights(scored: pd.DataFrame, blend_params: dict,
                    min_dep_deg: float = 5.0, min_n: int = 25) -> dict:
    """median(truth * tan(dep)) per label type — the effective camera height the truth
    implies, directly comparable to the blend's fitted per-type heights. Restricted to
    dep >= min_dep_deg where the ground-plane geometry is stable (tan near 0 amplifies
    truth noise into nonsense heights)."""
    rows = scored[scored["depression_deg"] >= min_dep_deg]
    out = {}
    for t, sub in rows.groupby("label_type"):
        if len(sub) < min_n:
            continue
        implied = sub["truth_m"].to_numpy(float) * np.tan(
            np.radians(sub["depression_deg"].to_numpy(float)))
        fitted = blend_params["height_by_type_m"].get(t)
        out[str(t)] = {
            "n": int(len(sub)),
            "implied_height_m": float(np.median(implied)),
            "fitted_height_m": fitted,
            "uses_fallback": fitted is None,
        }
    return out


def curb_sensitivity(scored: pd.DataFrame, camera_height_m: float,
                     curb_m: float = 0.15) -> dict:
    """Blend bias on CurbRamp rows with and without the curb-height truth correction.

    The depth model represents the road surface, so a ray at a curb ramp's lip lands
    ~curb*d/h too far (depth_validation.curb_height_bias_m); correcting the truth down
    by that bias asks whether the fitted CurbRamp height already absorbs it.
    ``camera_height_m`` is THIS fetch's measured rig median, not a carried-over constant,
    so the correction is scaled by the same cameras the truth came from."""
    sub = scored[scored["label_type"] == "CurbRamp"]
    if not len(sub):
        return {"n": 0}
    truth = sub["truth_m"].to_numpy(float)
    err = sub["D_blend"].to_numpy(float) - truth
    corrected = truth - curb_m * truth / camera_height_m
    err_corr = sub["D_blend"].to_numpy(float) - corrected
    return {
        "n": int(len(sub)),
        "curb_m": curb_m,
        "camera_height_m": camera_height_m,
        "signed_median_m": float(np.median(err)),
        "signed_median_corrected_m": float(np.median(err_corr)),
    }


def remedy_check(labels: pd.DataFrame, blend_params: dict, seed: int = SEED,
                 min_dep_deg: float = 5.0) -> dict:
    """Candidate recalibrations of the blend's height scale, scored held-out.

    The validation finds a GLOBAL multiplicative gap between the blend's
    era-calibrated heights and modern measured-plane truth (every city, every capture
    year). Each remedy derives its parameter on a train half split BY PANO and is
    scored on the disjoint half (a fitted parameter demands a held-out split), human
    gated rows only:

    - ``rescale``: one factor k = median(implied_height / assigned_height) over train
      rows at dep >= min_dep_deg, applied to the whole height table;
    - ``flat``: the per-type table abandoned for a single train-median implied height
      (the per-type spread does not replicate on modern truth).

    This is a calibration measurement for the Stage 4 decision, not a refit: the
    functional form, blend angle and cap are untouched.
    """
    human = labels[labels["gate_ok"] & ~labels["is_ai"]]
    rng = np.random.default_rng(seed)
    panos = np.sort(human["pano_id"].unique())
    train_ids = set(rng.choice(panos, len(panos) // 2, replace=False))
    train = human[human["pano_id"].isin(train_ids)]
    test = human[~human["pano_id"].isin(train_ids)]

    tr = train[train["depression_deg"] >= min_dep_deg]
    implied = tr["truth_m"].to_numpy(float) * np.tan(
        np.radians(tr["depression_deg"].to_numpy(float)))
    assigned = (tr["label_type"].map(blend_params["height_by_type_m"])
                .fillna(blend_params["height_fallback_m"]).to_numpy(float))
    k = float(np.median(implied / assigned))
    flat_h = float(np.median(implied))

    rescaled = dict(blend_params,
                    height_by_type_m={t: k * h for t, h in
                                      blend_params["height_by_type_m"].items()},
                    height_fallback_m=k * blend_params["height_fallback_m"])
    flat = {"form": "blend", "blend_deg": blend_params["blend_deg"],
            "height_m": flat_h}

    def stats(err):
        return {"median_abs_m": float(np.median(np.abs(err))),
                "signed_median_m": float(np.median(err)),
                "p90_abs_m": float(np.percentile(np.abs(err), 90))}

    def score(params):
        return stats(predict_dist(params, test) - test["truth_m"].to_numpy(float))

    return {
        "split": {"n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
                  "by": "pano", "seed": seed, "min_dep_deg": min_dep_deg},
        "k_rescale": k,
        "flat_height_m": flat_h,
        "test_half": {
            # the deployed model on the SAME disjoint rows, so the remedy's headline
            # number has an apples-to-apples reference instead of a pooled-column one
            "A_deployed": stats(test["A_deployed"].to_numpy(float)
                                - test["truth_m"].to_numpy(float)),
            "D_blend_as_shipped": score(blend_params),
            "D_rescaled": score(rescaled),
            "D_flat": score(flat),
        },
    }


FINAL_MIN_DEP_DEG = 5.0  # implied-height stability floor, same as implied_heights


def final_coefficients(labels: pd.DataFrame, blend_params: dict) -> dict:
    """The Stage 4 production constants: the blend form with ONE flat camera height,
    calibrated to modern measured-plane truth.

    Decision recorded 2026-08-07 (the tradeoffs are articulated in
    reports/2026-08-07-modern-truth.md §9): the flat variant over the global rescale —
    same held-out accuracy, two physical parameters, and it removes label_type (and with
    it the unseen-type fallback rule) from the distance path entirely. The held-out
    remedy check (``remedies``) validates the calibration on a disjoint pano half; the
    shipped constant itself is then the full-sample median implied height."""
    from distance_refit import structural_max_m

    human = labels[labels["gate_ok"] & ~labels["is_ai"]]
    sub = human[human["depression_deg"] >= FINAL_MIN_DEP_DEG]
    implied = (sub["truth_m"].to_numpy(float)
               * np.tan(np.radians(sub["depression_deg"].to_numpy(float))))
    h = float(np.median(implied))
    params = {"form": "blend", "height_m": h, "blend_deg": blend_params["blend_deg"]}
    err = predict_dist(params, human) - human["truth_m"].to_numpy(float)
    return {
        "form": "blend",
        "params": {"height_m": h, "blend_deg": params["blend_deg"], "n_params": 2},
        "max_answer_m": structural_max_m(params),
        "derived_from": f"median(truth x tan(depression)) over the {len(sub)} gated human "
                        f"rows at depression >= {FINAL_MIN_DEP_DEG} deg; the disjoint-half "
                        "remedy check is the unbiased error estimate",
        "replaces": "the era fit's per-type height table (era_fit_coefficients in "
                    "distance-refit-summary.json): its 2.50-2.78 m scale is the era "
                    "truth's pinned-plane artifact, and its per-type spread does not "
                    "replicate on modern truth",
        "in_sample_human": {
            "median_abs_m": float(np.median(np.abs(err))),
            "signed_median_m": float(np.median(err)),
            "p90_abs_m": float(np.percentile(np.abs(err), 90)),
        },
        "no_label_type_input": "the flat height removes label_type from the distance "
                               "path entirely, and with it the unseen-type fallback rule",
        "geodesy": "spherical (turf destination), matching production toLatLng",
        "heading": "exact POV inversion (pov_if_centered), zero parameters; no era "
                   "constant on post-evolution-179 data",
        "caveats": [
            "the absolute reference is Google's measured ground planes - internally "
            "consistent (Crosswalk vs measured rig: 15 mm) but externally unanchored; "
            "bearing-only triangulation (#7) is the independent path",
            "residual signed median ~-0.17 m: the modern terrain model's remaining "
            "overshoot, inside the truth budget",
            "near-horizon clicks (<2 deg) stay undershot by every bounded model; the "
            "tail's structural max is max_answer_m",
            "era-truth metrics live in the inflated frame: this calibration scores ~4 cm "
            "worse there by construction - the two frames cannot both be satisfied",
        ],
    }


# ---------------------------------------------------------------------------- summary

def build_summary(frame_census: dict, panos: pd.DataFrame, labels: pd.DataFrame,
                  blend_params: dict, frame_controls: dict, gate_census: dict) -> dict:
    """Assemble data/modern-truth-summary.json from the built artifacts.

    ``labels`` carries gates, predictions and guard fields for every frame-gated label
    on a fetched pano; ``panos`` one row per attempted pano. Headline metrics come from
    the representative stratum's human rows (an approximately proportional draw); the
    quota strata exist for the per-type and near-horizon tables and would bias a pooled
    median, so pooled-human numbers are reported alongside, clearly labelled."""
    labels = labels.assign(label_year=labels["time_created"].str[:4],
                           capture_year=labels["capture_date"].str[:4])
    scored = labels[labels["gate_ok"]]
    human = scored[~scored["is_ai"]]
    head = human[human["stratum"] == "representative"]
    ai = scored[scored["is_ai"]]

    status = panos["status"].value_counts()
    heights = panos.loc[panos["status"] == "ok"]
    pinned = heights["ground_d_exactly_2p5"].fillna(True).astype(bool)
    measured = heights.loc[~pinned, "ground_height_m"].astype(float)
    measured_median = float(measured.median()) if len(measured) else None

    n_ok = int((panos["status"] == "ok").sum())
    n_resolved = int((panos["status"] != "gone").sum())
    city_n = human["city"].value_counts()

    return {
        "meta": {
            "seed": SEED,
            "budgets": {"representative": REPRESENTATIVE_PANOS,
                        "near_horizon": NEAR_HORIZON_PANOS,
                        "type_label_quota": TYPE_LABEL_QUOTA,
                        "ai": AI_PANOS, "target_panos": TARGET_PANOS,
                        "oversample": OVERSAMPLE,
                        "city_pano_cap": CITY_PANO_CAP,
                        "near_horizon_deg": NEAR_HORIZON_DEG},
            "n_panos_attempted": int(len(panos)),
            "n_panos_ok": int(status.get("ok", 0)),
            "n_labels_on_ok_panos": int(len(labels)),
            "n_scored": int(len(scored)),
            "n_scored_human": int(len(human)),
            "n_scored_ai": int(len(ai)),
            "blend_rung": "D_blend_type_l1 (committed distance-refit-summary.json)",
        },
        "frame_census": frame_census,
        "fetch": {
            "status": {str(k): int(v) for k, v in status.items()},
            "resolve_rate": float((panos["status"] != "gone").mean()),
            # among panos whose id still resolves, the share that yielded a usable
            # payload; the shortfall is metadata that streetlevel's parser rejects
            # (status 'parse_error'), which is why this is not a pure depth-service rate
            "usable_rate_among_resolved": n_ok / max(n_resolved, 1),
            "by_stratum_ok": {str(k): int(v) for k, v in
                              panos.loc[panos["status"] == "ok",
                                        "stratum"].value_counts().items()},
            # the type strata are LABEL-count budgets, not pano-count ones, so whether
            # they were met is a property of the delivered labels rather than of
            # by_stratum_ok. Recorded per type so a shortfall cannot pass unnoticed.
            "type_label_coverage": {
                str(t): {"n_labels_on_ok_panos": int(n),
                         "quota": TYPE_LABEL_QUOTA, "met": bool(n >= TYPE_LABEL_QUOTA)}
                for t, n in labels["label_type"].value_counts().sort_index().items()},
        },
        "gates": gate_census,
        "guard": guard_summary(labels),
        "matrix": {
            "headline_representative_human": model_metrics(head),
            "all_human": model_metrics(human),
            "ai": model_metrics(ai) if len(ai) else {"n": 0},
        },
        "near_horizon": near_horizon_metrics(human),
        "by_label_type": by_group_metrics(human, "label_type"),
        "implied_heights": implied_heights(human, blend_params),
        "by_city": by_group_metrics(human, "city", min_n=50),
        # by_city reports only cities clearing min_n; the rest are too thin to read a
        # median from, and saying so here keeps "every scoreable city" honest about
        # how many cities that phrase covers
        "by_city_coverage": {
            "min_n": 50,
            "n_cities_with_scored_human_rows": int(len(city_n)),
            "n_cities_scored": int((city_n >= 50).sum()),
            "rows_in_cities_below_min_n": int(city_n[city_n < 50].sum()),
        },
        "by_capture_year": by_group_metrics(human, "capture_year", min_n=50),
        "by_label_year": by_group_metrics(human, "label_year", min_n=50),
        "camera_heights": {
            "n_ok_panos": int(len(heights)),
            "pinned_2p5_frac": float(pinned.mean()) if len(heights) else None,
            "measured_median_m": measured_median,
            "measured_iqr_m": [float(measured.quantile(0.25)),
                               float(measured.quantile(0.75))] if len(measured) else None,
        },
        "frame_controls": frame_controls,
        "curb_sensitivity": curb_sensitivity(human, measured_median),
        "remedies": remedy_check(labels, blend_params),
        "final_coefficients": final_coefficients(labels, blend_params),
    }
