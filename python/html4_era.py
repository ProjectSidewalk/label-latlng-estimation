"""The html4-era DC rows: labels clicked on Google's flat (non-perspective) Street View renderer.

Until Google retired Maps JS API 3.25 on 2017-02-15 the Project Sidewalk client asked for
``mode: 'html4'``, a flat tile renderer, and wrote ``sv_image_x/y`` with a matching LINEAR
formula (a fixed 13312x6656 frame, 4.6 px per canvas px in x and 4.65 in y, divided by the zoom
factor 1 / 2.1 / 4). The client kept that formula until v2.1 on 2017-03-21, so every DC row with
``label_id < 66003`` holds a linear ``sv_image_x/y``. The perspective inversion
(``pov_inversion.pov_if_centered``) is the wrong forward model for those rows: it is off by up
to ~5 deg, growing with distance from the canvas centre (top clicks too high, bottom too low,
side clicks too far out). Only DC has such rows; it was the only city before 2017.

SidewalkWebpage#5920 repairs the stored heading/pitch of those labels in production. This module
measures how the rows look in this repo's committed data so the reports that pool them can be
read correctly. Everything is offline: committed CSVs in, one summary dict out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from label_latlng_estimation import bearing_deg, clean_data, load_city
from pov_inversion import pov_if_centered, wrap_deg

# The first DC label whose sv_image_x/y the v2.1 client wrote with perspective math.
FIRST_PERSPECTIVE_LABEL_ID = 66003
# The flat renderer's frame and the client's linear constants (SidewalkWebpage, pre-v2.1).
FRAME_W, FRAME_H = 13312.0, 6656.0
ALPHA_X, ALPHA_Y = 4.6, 4.65
ZOOM_FACTOR = {1: 1.0, 2: 2.1, 3: 4.0}
CANVAS_W, CANVAS_H = 720.0, 480.0
CANVAS_X_BINS = [0, 90, 180, 270, 360, 450, 540, 630, 720]
CANVAS_Y_BINS = [0, 80, 160, 240, 320, 400, 480]


def is_html4(df: pd.DataFrame) -> pd.Series:
    """Rows whose stored sv_image_x/y came from the linear formula (DC, before v2.1)."""
    return (df["city"] == "dc") & (df["label_id"] < FIRST_PERSPECTIVE_LABEL_ID)


def linear_sv_image(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The pre-v2.1 client's sv_image_x/y from the stored POV record (unwrapped x)."""
    zf = df["zoom"].map(ZOOM_FACTOR).to_numpy(float)
    du = df["canvas_x"].to_numpy(float) - CANVAS_W / 2
    dv = CANVAS_H / 2 - df["canvas_y"].to_numpy(float)
    x = df["heading"].to_numpy(float) * FRAME_W / 360 + du * ALPHA_X / zf
    y = df["pitch"].to_numpy(float) * FRAME_H / 180 + dv * ALPHA_Y / zf
    return x, y


def perspective_sv_image(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The v2.1+ client's sv_image_x/y: the perspective inversion in the same frame."""
    h, p = pov_if_centered(df["canvas_x"], df["canvas_y"], df["heading"], df["pitch"], df["zoom"])
    return h * FRAME_W / 360, p * FRAME_H / 180


def _within_px(pred_x: np.ndarray, pred_y: np.ndarray, df: pd.DataFrame, tol: float = 1.0) -> float:
    dx = (pred_x - df["sv_image_x"].to_numpy(float) + FRAME_W / 2) % FRAME_W - FRAME_W / 2
    dy = pred_y - df["sv_image_y"].to_numpy(float)
    return float(((np.abs(dx) <= tol) & (np.abs(dy) <= tol)).mean())


def click_bearings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(perspective, linear) absolute click bearings in degrees, both replayed from the record."""
    h, _ = pov_if_centered(df["canvas_x"], df["canvas_y"], df["heading"], df["pitch"], df["zoom"])
    lin_x, _ = linear_sv_image(df)
    return wrap_deg(h), wrap_deg(lin_x * 360 / FRAME_W)


def click_altitudes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(perspective, linear) click altitudes in degrees above the horizon, replayed from the record."""
    _, p = pov_if_centered(df["canvas_x"], df["canvas_y"], df["heading"], df["pitch"], df["zoom"])
    _, lin_y = linear_sv_image(df)
    return p, lin_y * 180 / FRAME_H


def _binned_median(values: np.ndarray, key: np.ndarray, bins: list[int]) -> list[dict]:
    cut = pd.cut(key, bins, right=False)
    out = []
    for interval, grp in pd.Series(values).groupby(cut, observed=True):
        out.append({"bin": f"[{int(interval.left)}, {int(interval.right)})", "n": int(len(grp)),
                    "median_deg": float(grp.median())})
    return out


def summarize(data_dir: str) -> dict:
    """Measure the html4-era rows against the later DC rows in the cleaned DC data."""
    raw = load_city(data_dir, "dc")
    html4_raw = int(is_html4(raw).sum())
    cleaned, _ = clean_data(raw)
    cleaned = cleaned[cleaned["canvas_width"].isna() | (cleaned["canvas_width"] == CANVAS_W)]
    groups = {"html4": cleaned[is_html4(cleaned)], "later": cleaned[~is_html4(cleaned)]}

    summary: dict = {
        "first_perspective_label_id": FIRST_PERSPECTIVE_LABEL_ID,
        "html4_rows_raw": html4_raw,
        "html4_rows_cleaned": int(len(groups["html4"])),
        "later_rows_cleaned": int(len(groups["later"])),
        "html4_share_of_cleaned_dc": float(len(groups["html4"]) / len(cleaned)),
    }
    for name, d in groups.items():
        lin_x, lin_y = linear_sv_image(d)
        per_x, per_y = perspective_sv_image(d)
        b_per, b_lin = click_bearings(d)
        a_per, a_lin = click_altitudes(d)
        truth = bearing_deg(d["pano_lng"], d["pano_lat"], d["lng"], d["lat"])
        res_per = wrap_deg(truth - b_per)
        res_lin = wrap_deg(truth - b_lin)
        deep = (-a_lin >= 5) & (-a_per >= 5)
        dist = d["pano_dist"].to_numpy(float)
        zoom1 = d["zoom"].to_numpy() == 1
        summary[name] = {
            "rows": int(len(d)),
            "stored_sv_image_within_1px_of_linear": _within_px(lin_x, lin_y, d),
            "stored_sv_image_within_1px_of_perspective": _within_px(per_x, per_y, d),
            "bearing_abs_residual_median_deg": {
                "perspective": float(np.median(np.abs(res_per))),
                "linear": float(np.median(np.abs(res_lin))),
            },
            "bearing_signed_residual_vs_perspective_by_canvas_x_zoom1":
                _binned_median(res_per[zoom1], d["canvas_x"].to_numpy(float)[zoom1], CANVAS_X_BINS),
            "altitude_linear_minus_perspective_by_canvas_y_zoom1":
                _binned_median((a_lin - a_per)[zoom1], d["canvas_y"].to_numpy(float)[zoom1], CANVAS_Y_BINS),
            "implied_camera_height_m_median": {
                "perspective": float(np.median(dist[deep] * np.tan(np.radians(-a_per[deep])))),
                "linear": float(np.median(dist[deep] * np.tan(np.radians(-a_lin[deep])))),
            },
        }
    return summary
