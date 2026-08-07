"""Exact click->POV inversion: the analysis behind issue #5.

The heading half of the 2021 estimator (est7) is three per-zoom linear fits,
``heading_diff ~ canvas_x``. But the front end computes the click's POV from
``(canvas_x, canvas_y, heading, pitch, zoom)`` with known projection geometry, so the linear
fit is a first-order approximation of math we can just run. This module replicates that math
exactly and swaps it into the heading half of est7 -- zero fitted parameters where est7 has
two per zoom.

What scoring against the 2017-2020 targets then revealed (details in
reports/2026-08-06-pov-inversion.md): the stored targets were themselves produced by this
projection, but run the era client's way -- POV heading/pitch truncated to whole degrees
(``parseInt`` in calculatePointPov) -- and pushed through a depth-grid lookup whose
``Math.ceil`` indexing biases every bearing about one grid column clockwise. est7's fitted
coefficients absorb both artifacts, which is the only sense in which the regression ever
"beat" exact geometry. The era-faithful variant below models the generative process instead
of fitting it; the plain exact inversion is the correct forward model for everything after
evolution 179.

Provenance of the math (both sources verified identical):

- ``calculatePovIfCentered`` / ``get3dFov`` in SidewalkWebpage's
  ``public/javascripts/common/UtilitiesPanomarker.js`` (tag v7.19.10) -- the PanoMarker-derived
  path the front end uses;
- the inlined transcription of exactly that call in evolution 179
  (``conf/evolutions/default/179.sql``), which recomputed every stored ``pano_x/pano_y`` as
  ``calculatePanoXYFromPov(calculatePovIfCentered(...), camera_heading, width, height)``.

That second source is also the fidelity check: the recovered CSVs carry evolution 179's output
in ``current_pano_x/current_pano_y`` (non-DC cities), so `replay_evolution_179` must reproduce
those stored integers from the raw click columns. If it does, this module runs the same math
production ran.

Geometry notes:

- The camera basis is right-handed with x east, y north, z up in the JS's naming; headings are
  degrees clockwise from north, so the click's absolute heading is ``atan2(x, y)``.
- ``pov_pitch`` is degrees above the horizon (negative = below). The elevation output exists
  for #3's cotangent distance candidates: the depression angle of a click is ``-pov_pitch``,
  replacing the per-zoom ``sv_image_y + canvas_y`` proxy (see the scope note on #5).
- ``sgn(cos(pitch))`` in the JS guards the pitch-beyond-vertical case; the cleaned data's pitch
  range is [-35, 0] degrees so the guard never fires here, but it is kept for fidelity.
- All functions are vectorized; degrees in, degrees out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CANVAS_W = 720.0  # every row in the recovered data; evolution 179 hardcoded the same
CANVAS_H = 480.0


def get_3d_fov(zoom) -> np.ndarray:
    """UtilitiesPanomarker's get3dFov: vertical--horizontal 3D FOV in degrees per zoom level.

    zoom 1 -> 89.75, zoom 2 -> 53, zoom 3 -> 27.68 (the "determined experimentally" branch).
    """
    z = np.asarray(zoom, float)
    return np.where(z <= 2, 126.5 - z * 36.75, 195.93 / np.power(1.92, z))


def pov_if_centered(canvas_x, canvas_y, heading, pitch, zoom,
                    canvas_width: float = CANVAS_W,
                    canvas_height: float = CANVAS_H) -> tuple[np.ndarray, np.ndarray]:
    """Exact replica of calculatePovIfCentered: the (heading, pitch) a click would have at
    canvas center. Returns (pov_heading, pov_pitch) in degrees, pov_heading in (-180, 180].
    """
    fov = np.radians(get_3d_fov(zoom))
    h0 = np.radians(np.asarray(heading, float))
    p0 = np.radians(np.asarray(pitch, float))
    f = 0.5 * canvas_width / np.tan(0.5 * fov)

    du = np.asarray(canvas_x, float) - canvas_width / 2
    dv = canvas_height / 2 - np.asarray(canvas_y, float)
    sg = np.where(np.cos(p0) >= 0, 1.0, -1.0)  # the JS's sgn(cos(p0))

    x = f * np.cos(p0) * np.sin(h0) + du * sg * np.cos(h0) - dv * np.sin(p0) * np.sin(h0)
    y = f * np.cos(p0) * np.cos(h0) - du * sg * np.sin(h0) - dv * np.sin(p0) * np.cos(h0)
    z = f * np.sin(p0) + dv * np.cos(p0)

    r = np.sqrt(x * x + y * y + z * z)
    return np.degrees(np.arctan2(x, y)), np.degrees(np.arcsin(z / r))


def wrap_deg(a) -> np.ndarray:
    """Wrap degrees to (-180, 180], matching the harness's heading_diff convention."""
    a = np.asarray(a, float)
    return np.where(a > 180, a - 360, np.where(a <= -180, a + 360, a))


def exact_heading_diff(df: pd.DataFrame) -> np.ndarray:
    """The exact-inversion replacement for est7's heading half: the click's heading offset from
    the POV heading, wrapped like the harness target. Zero fitted parameters."""
    pov_heading, _ = pov_if_centered(df["canvas_x"], df["canvas_y"],
                                     df["heading"], df["pitch"], df["zoom"])
    return wrap_deg(pov_heading - df["heading"].to_numpy(float))


def exact_depression_deg(df: pd.DataFrame) -> np.ndarray:
    """The click's depression angle below the horizon (positive down), for #3's cotangent
    candidates. Same inversion, vertical output."""
    _, pov_pitch = pov_if_centered(df["canvas_x"], df["canvas_y"],
                                   df["heading"], df["pitch"], df["zoom"])
    return -pov_pitch


# One column of the 512-wide depth grid the 2017-2020 client sampled for label lat/lng.
# The legacy targets carry a constant bearing bias of about this size (see era notes below).
DEPTH_GRID_COLUMN_DEG = 360.0 / 512


def era_heading_diff(df: pd.DataFrame) -> np.ndarray:
    """The 2017-2020 client's placement math, reproduced faithfully: same projection, but the
    era client ran ``parseInt`` on heading and pitch first (truncation toward zero) and its
    ``calculateImageCoordinateFromPointPov`` added half a degree-pixel to sv_image_x.

    This is the forward model of how the stored sv_image_x was actually produced -- verified
    against the recovered data at 99.8% within one pixel (see the report). It exists to
    separate two things the plain exact inversion conflates when scored against 2017-2020
    targets: projection geometry (identical here) and the era client's input quantization
    (modeled here, absent for any post-evolution-179 data).
    """
    h = df["heading"].to_numpy(float)
    pov_heading, _ = pov_if_centered(df["canvas_x"], df["canvas_y"],
                                     np.trunc(h), np.trunc(df["pitch"].to_numpy(float)),
                                     df["zoom"])
    return wrap_deg(pov_heading + 0.5 - h)


# ------------------------------------------------------------------- evolution 179 replay

def replay_evolution_179(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute pano_x/pano_y the way evolution 179 did, for comparison with the stored
    current_pano_x/current_pano_y.

    The SQL's per-pano camera_heading was the most recent label's photographer_heading; the
    recovered CSVs carry photographer_heading per row, which is constant per pano in practice
    -- the match rate reported by the caller measures any slack in that assumption along with
    everything else.

    Returns a DataFrame with replay_pano_x / replay_pano_y (ints, rounded like the SQL) for the
    rows where the evolution's WHERE clause held (pano dimensions present, camera pose not NaN).
    """
    ok = (df["pano_width"].notna() & df["pano_height"].notna()
          & df["photographer_heading"].notna() & df["photographer_pitch"].notna())
    d = df[ok]
    pov_heading, pov_pitch = pov_if_centered(d["canvas_x"], d["canvas_y"],
                                             d["heading"], d["pitch"], d["zoom"])
    width = d["pano_width"].to_numpy(float)
    height = d["pano_height"].to_numpy(float)
    camera_heading = d["photographer_heading"].to_numpy(float)

    # calculatePanoXYFromPov, with the SQL's round-then-wrap on x.
    heading_wrapped = (pov_heading + 360) % 360
    heading_pixel_zero = ((camera_heading + 180) % 360 + 360) % 360
    pano_x = (width + np.round(width * (heading_wrapped - heading_pixel_zero) / 360)) % width
    pano_y = height / 2 - np.round((height / 2) * (pov_pitch / 90))

    return pd.DataFrame({"label_id": d["label_id"].to_numpy(),
                         "replay_pano_x": pano_x.astype(int),
                         "replay_pano_y": pano_y.astype(int)}, index=d.index)


def fidelity_report(df: pd.DataFrame) -> dict:
    """Compare the evolution-179 replay against the stored current_pano_x/y for one city's raw
    rows. This is the proof the module runs the math production ran.

    Two known, measured caveats keep the pre-2021 x match rate below 100% without implicating
    the projection:

    - pano_y is a pure function of the click POV and must match exactly;
    - pano_x additionally needs the SQL's 2022-era per-pano camera_heading, which is
      unrecoverable when Google's pano metadata drifted before the 2026 recovery fetch. The
      drift signature (the mismatch is constant within a pano, varies across panos) is
      reported so the caveat stays a measurement rather than a hand-wave.
    """
    has = (df["current_pano_x"].notna() & df["current_pano_y"].notna()
           & df["pano_width"].notna() & df["pano_height"].notna())
    d = df[has]
    if d.empty:  # DC: evolution 179 never ran there
        return {"n_with_current_pano_xy": 0}
    rep = replay_evolution_179(d)
    d = d.loc[rep.index]
    w = d["pano_width"].to_numpy(float)

    dx = (rep["replay_pano_x"].to_numpy() - d["current_pano_x"].astype(int).to_numpy())
    dx = (dx + w / 2) % w - w / 2  # signed, wrapped at the seam
    dy = np.abs(rep["replay_pano_y"].to_numpy() - d["current_pano_y"].astype(int).to_numpy())

    # Within-pano constancy of the implied camera-heading delta on mismatching rows: the
    # signature that separates metadata drift (constant per pano) from projection error.
    mm = pd.DataFrame({"pano_id": d["pano_id" if "pano_id" in d else "gsv_panorama_id"].to_numpy(),
                       "ang": dx / w * 360})[np.abs(dx) > 1]
    per_pano_std = mm.groupby("pano_id")["ang"].std().dropna()

    # Era split: pre-cutoff rows got pano_x from the 2022 SQL recompute (exposed to
    # camera_heading drift); post-cutoff rows were written by the front end running this same
    # math live at placement, so they must match exactly.
    cutoff = pd.Timestamp("2021-01-01", tz="UTC")
    pre = (d["time_created"].isna() | (d["time_created"] < cutoff)).to_numpy()

    return {
        "n_with_current_pano_xy": int(len(d)),
        "pano_y_exact_match_rate": float((dy == 0).mean()),
        "pano_x_exact_match_rate": float((np.abs(dx) == 0).mean()),
        "pano_x_within_1px_rate": float((np.abs(dx) <= 1).mean()),
        "pano_x_exact_match_rate_pre_cutoff": float((np.abs(dx[pre]) == 0).mean()) if pre.any() else None,
        "pano_x_exact_match_rate_post_cutoff": (float((np.abs(dx[~pre]) == 0).mean())
                                                if (~pre).any() else None),
        "mismatch_within_pano_std_deg_median": (float(per_pano_std.median())
                                                if len(per_pano_std) else None),
        "mismatch_across_pano_std_deg": (float(mm.groupby("pano_id")["ang"].mean().std())
                                         if len(mm) else None),
    }


# ------------------------------------------------------------------- scoring vs est7

def score_heading_swap(models: dict, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per-label error columns for four heading models, each paired with est7's fitted
    distance so any difference is the heading half alone:

    - ``est7``: the 2021 per-zoom linear fits (6 fitted parameters);
    - ``exact``: the exact POV inversion (0 parameters) -- the physically-correct forward
      model, and the one #3 should hand off for post-evolution-179 data;
    - ``era``: the exact inversion run the way the 2017-2020 client ran it, truncated inputs
      and the half-degree-pixel offset (0 parameters);
    - ``era_cal``: ``era`` plus one global constant, the train-set mean residual (1
      parameter). The constant is the legacy depth-lookup bias -- Label.js indexed the
      512-column depth grid with Math.ceil, so the stored targets sit about one grid column
      (DEPTH_GRID_COLUMN_DEG = 0.703 deg) clockwise of the click's true bearing. It is a
      property of the 2017-2020 ground truth, not of any estimator, and must NOT be carried
      to data produced after evolution 179.

    Mirrors evaluate()'s column shapes so downstream summaries are comparable.
    """
    from label_latlng_estimation import latlng_error_m, predict_dist_heading

    dist7, head7 = predict_dist_heading(models, test, "est7")
    delta = float(np.mean(train["heading_diff"].to_numpy(float) - era_heading_diff(train)))
    heads = {
        "est7": head7,
        "exact": exact_heading_diff(test),
        "era": era_heading_diff(test),
        "era_cal": era_heading_diff(test) + delta,
    }

    out = pd.DataFrame({
        "label_id": test["label_id"].to_numpy(),
        "city": test["city"].to_numpy(),
        "zoom": test["zoom"].to_numpy(),
        "canvas_x": test["canvas_x"].to_numpy(),
        "canvas_y": test["canvas_y"].to_numpy(),
        "pitch": test["pitch"].to_numpy(float),
        "photographer_pitch": test["photographer_pitch"].to_numpy(float),
        "pano_dist": test["pano_dist"].to_numpy(float),
        "heading_diff": test["heading_diff"].to_numpy(float),
    })
    for name, h in heads.items():
        out[f"error_{name}"] = latlng_error_m(test, dist7, h, crude=False)
        out[f"heading_error_{name}"] = np.abs(test["heading_diff"].to_numpy(float) - h)
        out[f"heading_pred_{name}"] = h
    out.attrs["era_cal_delta_deg"] = delta
    return out


MODEL_NAMES = ("est7", "exact", "era", "era_cal")


def summarize_heading_swap(scored: pd.DataFrame) -> dict:
    """The findings dict: overall medians, the canvas-edge/|pitch| breakdown of where the
    linear approximation loses, and the photographer_pitch residual check."""
    def meds(prefix: str) -> dict:
        return {m: float(scored[f"{prefix}_{m}"].median()) for m in MODEL_NAMES}

    summary: dict = {
        "n_test": int(len(scored)),
        "era_cal_delta_deg": float(scored.attrs["era_cal_delta_deg"]),
        "depth_grid_column_deg": DEPTH_GRID_COLUMN_DEG,
        "heading_error_median_deg": meds("heading_error"),
        "latlng_error_median_m": meds("error"),
        "heading_error_median_deg_by_zoom": {
            int(z): {m: float(g[f"heading_error_{m}"].median()) for m in MODEL_NAMES}
            for z, g in scored.groupby("zoom")},
    }

    # Where the linear fit loses: the models differ deterministically (both are functions of
    # the click), so bin the disagreement and the realized errors by canvas-center offset and
    # by |pitch|.
    edge = pd.cut(np.abs(scored["canvas_x"] - CANVAS_W / 2),
                  [0, 90, 180, 270, 360], include_lowest=True)
    summary["by_canvas_x_offset"] = _binned(scored, edge)
    pitch_bin = pd.cut(np.abs(scored["pitch"]), [0, 5, 15, 25, 35], include_lowest=True)
    summary["by_abs_pitch"] = _binned(scored, pitch_bin)

    # photographer_pitch residual check (scope note on #5): if rig tilt mattered for the
    # heading half, the exact inversion's signed residual would track photographer_pitch.
    resid = scored["heading_diff"].to_numpy() - scored["heading_pred_exact"].to_numpy()
    pp = scored["photographer_pitch"].to_numpy()
    ok = np.isfinite(resid) & np.isfinite(pp)
    r = float(np.corrcoef(pp[ok], resid[ok])[0, 1])
    slope = float(np.polyfit(pp[ok], resid[ok], 1)[0])
    summary["photographer_pitch_residual_check"] = {
        "n": int(ok.sum()), "pearson_r": r, "slope_deg_per_deg": slope,
        "photographer_pitch_p5_p95": [float(np.percentile(pp[ok], 5)),
                                      float(np.percentile(pp[ok], 95))],
    }
    return summary


def _binned(scored: pd.DataFrame, bins: pd.Series) -> list[dict]:
    rows = []
    for interval, g in scored.groupby(bins, observed=True):
        rows.append({
            "bin": str(interval), "n": int(len(g)),
            "heading_error_median_deg": {m: float(g[f"heading_error_{m}"].median())
                                         for m in MODEL_NAMES},
            "mean_abs_disagreement_deg": float(
                np.abs(g["heading_pred_est7"] - g["heading_pred_exact"]).mean()),
        })
    return rows
