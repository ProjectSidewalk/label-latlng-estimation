"""The html4-era DC rows: labels clicked on Google's flat (non-perspective) Street View renderer.

Until Google retired Maps JS API 3.25 on 2017-02-15 the Project Sidewalk client asked for
``mode: 'html4'``, a flat tile renderer, and wrote ``sv_image_x/y`` with a matching LINEAR
formula (a fixed 13312x6656 frame, 4.6 px per canvas px in x and 4.65 in y, divided by the zoom
factor 1 / 2.1 / 4). The client kept that formula until v2.1 on 2017-03-21, so every DC row with
``label_id < 66003`` holds a linear ``sv_image_x/y``. The perspective inversion
(``pov_inversion.pov_if_centered``) is the wrong forward model for those rows: the two models
agree at the canvas centre and come close again at its edges (the linear slope was fitted to the
full canvas width), and differ most about 200 px out, by 3 deg at the median row and 5 deg at the
90th percentile. Only DC has such rows; it was the only city before 2017.

SidewalkWebpage#5920 repairs the stored heading/pitch of those labels in production (the legacy
``sidewalk_dc`` database this repo's extract came from is untouched). This module measures how
the rows look in this repo's committed data so the reports that pool them can be read correctly.
Everything is offline: committed CSVs in, one summary dict out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from label_latlng_estimation import bearing_deg, clean_data, load_city
from pov_inversion import CANVAS_H, CANVAS_W, pov_if_centered, wrap_deg

# The first DC label whose sv_image_x/y the v2.1 client wrote with perspective math.
FIRST_PERSPECTIVE_LABEL_ID = 66003
# The flat renderer's frame and the client's linear constants (SidewalkWebpage, pre-v2.1).
FRAME_W, FRAME_H = 13312.0, 6656.0
ALPHA_X, ALPHA_Y = 4.6, 4.65
ZOOM_FACTOR = {1: 1.0, 2: 2.1, 3: 4.0}
CANVAS_X_BINS = [0, 90, 180, 270, 360, 450, 540, 630, 721]
CANVAS_Y_BINS = [0, 80, 160, 240, 320, 400, 481]
PERCENTILES = (50, 90, 99, 100)
# gbm_transfer.SCALE_MIN_DEP_DEG, the implied-height stability floor; repeated here because
# importing gbm_transfer pulls in statsmodels and LightGBM for one number.
SCALE_MIN_DEP_DEG = 5.0


def is_html4(df: pd.DataFrame) -> pd.Series:
    """Rows whose stored sv_image_x/y came from the linear formula (DC, before v2.1), by id."""
    return (df["city"] == "dc") & (df["label_id"] < FIRST_PERSPECTIVE_LABEL_ID)


def linear_sv_image(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The pre-v2.1 client's sv_image_x/y from the stored POV record (unwrapped x)."""
    zoom = df["zoom"].to_numpy()
    if not set(np.unique(zoom)) <= set(ZOOM_FACTOR):
        unknown = sorted(set(zoom) - set(ZOOM_FACTOR))
        raise ValueError(f"zoom outside {sorted(ZOOM_FACTOR)}: {unknown}")
    zf = pd.Series(zoom).map(ZOOM_FACTOR).to_numpy(float)
    du = df["canvas_x"].to_numpy(float) - CANVAS_W / 2
    dv = CANVAS_H / 2 - df["canvas_y"].to_numpy(float)
    x = df["heading"].to_numpy(float) * FRAME_W / 360 + du * ALPHA_X / zf
    y = df["pitch"].to_numpy(float) * FRAME_H / 180 + dv * ALPHA_Y / zf
    return x, y


def perspective_pov(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(bearing, altitude) of the click under the perspective inversion, degrees."""
    h, p = pov_if_centered(df["canvas_x"], df["canvas_y"], df["heading"], df["pitch"], df["zoom"])
    return wrap_deg(h), p


def era_sv_image_x(df: pd.DataFrame) -> np.ndarray:
    """The v2.1-to-2020 client's sv_image_x: the perspective inversion with its parseInt-truncated
    inputs and half-pixel offset (the POV-inversion report's era replica, 2026-08-06 section 3)."""
    h, _ = pov_if_centered(df["canvas_x"], df["canvas_y"], np.trunc(df["heading"].to_numpy(float)),
                           np.trunc(df["pitch"].to_numpy(float)), df["zoom"])
    x = FRAME_W * (h / 360) + (FRAME_W / 360) / 2
    return np.where(x < 0, x + FRAME_W, x)


def _x_within_px(pred_x: np.ndarray, df: pd.DataFrame, tol: float = 1.0) -> np.ndarray:
    dx = (pred_x - df["sv_image_x"].to_numpy(float) + FRAME_W / 2) % FRAME_W - FRAME_W / 2
    return np.abs(dx) <= tol


def stored_formula(df: pd.DataFrame) -> pd.Series:
    """Which forward model reproduces each row's stored sv_image_x within 1 px: 'linear', 'era'
    (the perspective replica), 'both' (a click too close to the centre to tell) or 'neither'.
    Survives a re-extraction after SidewalkWebpage#5920 better than the id rule only in the sense
    that it says what it sees: that repair rewrites heading/pitch so that the perspective
    inversion lands on the click, so the html4 rows then classify as 'era' and ``is_html4``
    (by id) is the durable marker."""
    lin_x, _ = linear_sv_image(df)
    lin = _x_within_px(lin_x, df)
    era = _x_within_px(era_sv_image_x(df), df)
    kind = np.select([lin & era, lin, era], ["both", "linear", "era"], "neither")
    return pd.Series(kind, index=df.index)


def _percentiles(values: np.ndarray) -> dict:
    return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}


def _binned_median(values: np.ndarray, key: np.ndarray, bins: list[int]) -> list[dict]:
    cut = pd.cut(key, bins, right=False)
    out = []
    for interval, grp in pd.Series(values).groupby(cut, observed=True):
        out.append({"bin": f"[{int(interval.left)}, {int(min(interval.right, bins[-1] - 1))}]",
                    "n": int(len(grp)), "median_deg": float(grp.median())})
    return out


def _group_summary(d: pd.DataFrame) -> dict:
    lin_x, lin_y = linear_sv_image(d)
    b_lin = wrap_deg(lin_x * 360 / FRAME_W)
    a_lin = lin_y * 180 / FRAME_H
    b_per, a_per = perspective_pov(d)
    truth = bearing_deg(d["pano_lng"], d["pano_lat"], d["lng"], d["lat"])
    res_per = wrap_deg(truth - b_per)
    res_lin = wrap_deg(truth - b_lin)
    gap_bearing = np.abs(wrap_deg(b_lin - b_per))
    gap_altitude = np.abs(a_lin - a_per)
    deep = (-a_lin >= SCALE_MIN_DEP_DEG) & (-a_per >= SCALE_MIN_DEP_DEG)
    dist = d["pano_dist"].to_numpy(float)
    zoom1 = d["zoom"].to_numpy() == 1
    formula = stored_formula(d).value_counts()
    return {
        "rows": int(len(d)),
        "stored_formula_counts": {k: int(formula.get(k, 0))
                                  for k in ("linear", "era", "both", "neither")},
        "stored_sv_image_x_within_1px_of_linear": float(_x_within_px(lin_x, d).mean()),
        "stored_sv_image_x_within_1px_of_era_replica":
            float(_x_within_px(era_sv_image_x(d), d).mean()),
        "model_gap_deg": {"bearing": _percentiles(gap_bearing),
                          "altitude": _percentiles(gap_altitude)},
        "bearing_abs_residual_median_deg": {
            "perspective": float(np.median(np.abs(res_per))),
            "linear": float(np.median(np.abs(res_lin))),
        },
        "bearing_signed_residual_median_deg": {
            "perspective": float(np.median(res_per)),
            "linear": float(np.median(res_lin)),
        },
        "bearing_signed_residual_vs_perspective_by_canvas_x_zoom1":
            _binned_median(res_per[zoom1], d["canvas_x"].to_numpy(float)[zoom1], CANVAS_X_BINS),
        "altitude_linear_minus_perspective_by_canvas_y_zoom1":
            _binned_median((a_lin - a_per)[zoom1], d["canvas_y"].to_numpy(float)[zoom1],
                           CANVAS_Y_BINS),
        "implied_camera_height_m_median": {
            "perspective": float(np.median(dist[deep] * np.tan(np.radians(-a_per[deep])))),
            "linear": float(np.median(dist[deep] * np.tan(np.radians(-a_lin[deep])))),
        },
    }


def summarize(data_dir: str) -> dict:
    """Measure the html4-era rows against the later DC rows in the cleaned DC data."""
    raw = load_city(data_dir, "dc")
    cleaned, _ = clean_data(raw)
    html4 = is_html4(cleaned)
    return {
        "first_perspective_label_id": FIRST_PERSPECTIVE_LABEL_ID,
        "html4_rows_raw": int(is_html4(raw).sum()),
        "html4_rows_cleaned": int(html4.sum()),
        "later_rows_cleaned": int((~html4).sum()),
        "html4_share_of_cleaned_dc": float(html4.mean()),
        "html4": _group_summary(cleaned[html4]),
        "later": _group_summary(cleaned[~html4]),
    }
