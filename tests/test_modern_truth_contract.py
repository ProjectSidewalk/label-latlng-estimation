"""Invariants for the modern-truth harness (issue #3), independent of any committed run.

The trap-zone piece is the pixel lookup: stored post-evolution-179 pano_x/pano_y and the
depth raster are both heading-centred, and gsv_depth's arrays keep the payload's own x
order, so the lookup is round(pano_x/w*512) with NO mirror and NO yaw shift. These tests
pin that formula against constructed payloads where every wrong frame visibly fails, and
pin the shared classify_depth_pixel core to the legacy classify_label_hit bit for bit.
"""

import gzip
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

import depth_validation as dv  # noqa: E402
import gsv_depth as gd  # noqa: E402
import modern_truth as mt  # noqa: E402
from mapillary_falsification import CANVAS_CENTER_Y, model_distances  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "depth-pilot", "real-payload.json.gz")


def synthetic_payload(ground_height=2.5, half_ground=False):
    """A payload with sky above the horizon and one ground plane below it.

    With half_ground, only the left half (cols < 256) of the bottom is ground — an
    asymmetry every x-handedness mistake must trip over."""
    w, h = gd.DEPTH_W, gd.DEPTH_H
    indices = np.zeros(h * w, dtype=np.uint8)
    grid = indices.reshape(h, w)
    grid[h // 2:, : (w // 2 if half_ground else w)] = 1
    planes_n = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    planes_d = np.array([0.0, ground_height], dtype=np.float32)
    return gd.DepthPayload(header_size=8, n_planes=2, width=w, height=h, offset=8,
                           indices=indices, planes_n=planes_n, planes_d=planes_d,
                           was_compressed=False)


# ---------------------------------------------------------------------------- extraction

EXTRACT_COLS = ("label_id,label_type,lat,lng,canvas_x,canvas_y,heading,pitch,zoom,"
                "pano_x,pano_y,computation_method,pano_id,time_created,is_ai,pano_width,"
                "pano_height,pano_lat,pano_lng,camera_heading,camera_pitch,camera_roll,"
                "capture_date,pano_source")


def _write_city(tmp_path, city, rows):
    path = os.path.join(tmp_path, f"modern-labels-{city}.csv.gz")
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        f.write(EXTRACT_COLS + "\n")
        for label_id, is_ai in rows:
            f.write(f"{label_id},CurbRamp,47.6,-122.3,100,240,180,0,1,4096,5000,"
                    f"approximation2,PANO{label_id:018d},2024-01-01 00:00:00+00,{is_ai},"
                    "16384,8192,47.6,-122.3,180,0,0,2023-05-01,gsv\n")
    return path


def test_load_extraction_keys_on_city_not_the_per_schema_label_id(tmp_path):
    """label_id restarts at 1 in every city schema, so it is NOT a key across the
    concatenated frame. Joining on it would pair a label with another city's row."""
    _write_city(tmp_path, "seattle", [(1, "f"), (2, "f")])
    _write_city(tmp_path, "chicago", [(1, "f"), (2, "t")])
    df = mt.load_extraction(str(tmp_path))

    assert len(df) == 4
    assert not df["label_id"].is_unique          # the trap
    assert df["label_uid"].is_unique             # the key
    assert set(df["label_uid"]) == {"chicago:1", "chicago:2", "seattle:1", "seattle:2"}
    assert list(df.loc[df["is_ai"], "label_uid"]) == ["chicago:2"]

    # a self-join on label_id doubles the frame; on label_uid it does not
    assert len(df.merge(df, on="label_id")) == 8
    assert len(df.merge(df, on="label_uid")) == 4


def test_load_extraction_rejects_an_unparseable_is_ai_flag(tmp_path):
    """A silent .astype(bool) would read every unmapped value as an AI label, quietly
    moving human clicks out of the headline population."""
    _write_city(tmp_path, "seattle", [(1, "f"), (2, "TRUE")])
    with pytest.raises(ValueError, match="is_ai"):
        mt.load_extraction(str(tmp_path))


# ---------------------------------------------------------------------------- pixel math

def test_modern_col_row_formula():
    # 8192-px pano: pixel centre mapping, wrap on x, clamp on y
    assert mt.modern_col_row(0, 4096, 16384, 8192) == (0, 128)
    assert mt.modern_col_row(16384 - 1, 4096, 16384, 8192) == (0, 128)  # wraps, not 512
    assert mt.modern_col_row(8192, 8191, 16384, 8192) == (256, 255)
    assert mt.modern_col_row(32, 0, 16384, 8192) == (1, 0)
    col, row = mt.modern_col_row(4096, 6144, 16384, 8192)
    assert (col, row) == (128, 192)


def test_control_col_row_matches_apply_frame_control():
    grid = np.arange(gd.DEPTH_H * gd.DEPTH_W).reshape(gd.DEPTH_H, gd.DEPTH_W)
    rng = np.random.default_rng(0)
    cells = [(int(c), int(r)) for c, r in
             zip(rng.integers(0, gd.DEPTH_W, 50), rng.integers(0, gd.DEPTH_H, 50))]
    for control in dv.FRAME_CONTROLS:
        moved = dv.apply_frame_control(grid, control)
        for col, row in cells:
            c2, r2 = mt.control_col_row(col, row, control)
            assert moved[row, col] == grid[r2, c2], (control, col, row)


def test_synthetic_ground_plane_is_exact_cotangent():
    h_cam = 2.5
    payload = synthetic_payload(ground_height=h_cam)
    geom = dv.payload_geometry(payload)
    for row in (150, 180, 220, 250):
        hit = dv.classify_depth_pixel(payload, 100, row, h_cam, geom)
        assert hit.hit_class == "ground"
        dep = float(geom.depression[row])
        assert hit.horizontal_m == pytest.approx(h_cam / math.tan(dep), rel=1e-5)
        assert hit.flat_earth_m == pytest.approx(hit.horizontal_m, rel=1e-5)
        assert hit.height_above_ground_m == pytest.approx(0.0, abs=1e-5)


def test_x_mirror_reads_a_different_world():
    payload = synthetic_payload(half_ground=True)
    geom = dv.payload_geometry(payload)
    # pano_x in the left half -> ground under identity, sky under x_mirror
    pano_x, pano_y = 3000, 6000  # col 94, row 188 on a 16384x8192 pano
    hit = mt.classify_modern_label(payload, pano_x, pano_y, 16384, 8192, 2.5, geom)
    mirrored = mt.classify_modern_label(payload, pano_x, pano_y, 16384, 8192, 2.5, geom,
                                        control="x_mirror")
    assert hit.hit_class == "ground"
    assert mirrored.hit_class == "sky"


def test_classify_depth_pixel_reproduces_label_hit_on_real_payload():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        payload = gd.decode_depth_payload(json.load(f)["depth_b64"])
    geom = dv.payload_geometry(payload)
    rng = np.random.default_rng(SEED := 666)
    xs = rng.uniform(0, 13312, 200)
    ys = rng.uniform(-3300, 3300, 200)
    for x, y in zip(xs, ys):
        via_legacy = dv.classify_label_hit(payload, x, y, 2.5, geom)
        px = math.ceil(x * gd.SV_IMAGE_SCALE)
        py = math.ceil((gd.SV_IMAGE_Y_ORIGIN - y) * gd.SV_IMAGE_SCALE)
        via_core = dv.classify_depth_pixel(payload, px, py, 2.5, geom)
        for field in dv.LabelHit.__dataclass_fields__:
            a, b = getattr(via_legacy, field), getattr(via_core, field)
            same = (a == b) or (isinstance(a, float) and isinstance(b, float)
                                and math.isnan(a) and math.isnan(b))
            assert same, (field, a, b, x, y)


# ---------------------------------------------------------------------------- models

def test_deployed_distance_matches_falsification_model_a_at_zoom_1():
    dep = np.array([2.0, 5.0, 11.25, 30.0])
    pano_h = np.full(4, 8192.0)
    pano_y = pano_h / 2 + dep / 180.0 * pano_h
    ours = mt.deployed_distance(pano_y, pano_h, np.full(4, CANVAS_CENTER_Y), np.ones(4))
    theirs = model_distances(dep, pano_h)["A_status_quo"]
    assert np.allclose(ours, theirs, atol=1e-9)


def test_normalized_distance_matches_falsification_model_b_at_zoom_1():
    dep = np.array([2.0, 5.0, 11.25, 30.0])
    ours = mt.normalized_distance(dep, np.full(4, CANVAS_CENTER_Y), np.ones(4))
    theirs = model_distances(dep, np.full(4, 8192.0))["B_normalized"]
    assert np.allclose(ours, theirs, atol=1e-9)


def test_deployed_coefficients_are_per_zoom_and_pin_the_zoom_1_constants():
    import mapillary_falsification as mf

    assert mt.DEPLOYED_DIST_COEF[1] == (mf.EST_INTERCEPT, mf.EST_PANO_Y_SLOPE,
                                        mf.EST_CANVAS_Y_SLOPE)
    # round(zoom) selection, exactly as PanoDataService.toLatLng does it
    same_geometry = dict(pano_y=np.full(3, 5000.0), pano_height=np.full(3, 8192.0),
                         canvas_y=np.full(3, 240.0))
    d = mt.deployed_distance(zoom=np.array([1.4, 1.6, 3.0]), **same_geometry)
    assert d[0] != d[1] and d[1] != d[2]  # 1.4 -> z1, 1.6 -> z2, 3.0 -> z3


def test_blend_prediction_bounded_and_falls_back_for_unseen_types():
    params = mt.load_blend_params()
    df = pd.DataFrame({
        "depression_deg": np.linspace(-30, 80, 23),
        "label_type": ["Crosswalk"] * 23,  # never fitted: must use height_fallback_m
    })
    pred = mt.model_predictions(
        df.assign(pano_y=0, pano_height=8192, canvas_y=240, zoom=1.0), params)
    assert np.all(pred["D_blend"] >= 0) and np.all(pred["D_blend"] <= 50)
    h = params["height_fallback_m"]
    steep = df["depression_deg"] >= params["blend_deg"]
    expect = h / np.tan(np.radians(df.loc[steep, "depression_deg"]))
    assert np.allclose(pred.loc[steep, "D_blend"], expect, rtol=1e-9)


def test_depression_is_the_provisional_coefficients_conversion():
    df = pd.DataFrame({"pano_y": [4096.0, 6144.0, 0.0], "pano_height": [8192.0] * 3})
    out = mt.add_depression(df)
    assert list(out["depression_deg"]) == [0.0, 45.0, -90.0]


# ---------------------------------------------------------------------------- guard

def test_guard_flags_synthetic_echo_and_moved_label():
    lat0, lng0 = 47.6, -122.3
    df = pd.DataFrame({
        "pano_y": [5000.0, 5000.0, 5000.0], "pano_height": [8192.0] * 3,
        "canvas_y": [240.0] * 3, "zoom": [1.0, 2.0, 1.0],
        "pano_lat": [lat0] * 3, "pano_lng": [lng0] * 3,
        "time_created": pd.to_datetime(["2024-01-01", "2024-01-01", "2022-01-01"],
                                       utc=True),
    })
    df = mt.add_depression(df)
    a = mt.deployed_distance(df["pano_y"], df["pano_height"], df["canvas_y"], df["zoom"])
    b = mt.normalized_distance(df["depression_deg"], df["canvas_y"], df["zoom"])
    # rows: post-179 echo of A; post-179 moved by 1 m; pre-179 echo of its OWN era (B)
    dist = np.array([a[0], a[1] + 1.0, b[2]])
    meters_per_deg_lat = 6371008.8 * math.pi / 180.0
    df["lat"] = lat0 + dist / meters_per_deg_lat
    df["lng"] = lng0
    out = mt.guard_frame(df)
    assert list(out["era"]) == ["real_pixels", "real_pixels", "fixed_frame"]
    assert list(out["is_echo"]) == [True, False, True]
    assert abs(out["guard_diff_m"].iloc[1] - 1.0) < 0.01
    # the pre-179 row must NOT reproduce under the wrong era's (real-pixel) formula
    assert abs(out["guard_cross_diff_m"].iloc[2]) > mt.GUARD_ECHO_M


# ---------------------------------------------------------------------------- gates

def test_truth_gates_census_adds_up():
    labels = pd.DataFrame({
        "hit_class": ["ground", "terrain", "facade", "sky", "ground", "ground"],
        "neighbourhood_range_ratio": [1.0, 1.0, 1.0, np.nan, 3.0, np.nan],
        "truth_m": [10.0, 20.0, 30.0, np.nan, 10.0, 60.0],
    })
    out, census = mt.truth_gates(labels)
    assert census["gate_ok"] == 2  # rows 0, 1; row 4 fails ratio, row 5 fails cap
    assert census["failed_hit"] == 2
    assert census["failed_ratio"] == 1
    assert census["failed_cap"] == 1
    assert bool(out["gate_ok"][5]) is False  # lone finite ray keeps its say, cap still bites


# ---------------------------------------------------------------------------- selection

def synthetic_frame(n_panos=4000, seed=1):
    rng = np.random.default_rng(seed)
    pano_ids = np.array([f"P{i:06d}" for i in range(n_panos)])
    rows = []
    types = ["CurbRamp", "NoSidewalk", "Obstacle", "Crosswalk", "Signal",
             "Occlusion", "Other", "SurfaceProblem", "NoCurbRamp"]
    probs = [0.30, 0.24, 0.13, 0.10, 0.03, 0.006, 0.002, 0.145, 0.047]
    for i, pid in enumerate(pano_ids):
        city = f"city{i % 12}"
        ai = i % 40 == 0
        for _ in range(rng.integers(1, 5)):
            rows.append({
                "label_id": len(rows), "pano_id": pid, "city": city, "is_ai": ai,
                "label_type": rng.choice(types, p=np.array(probs) / sum(probs)),
                "depression_deg": rng.uniform(-1 if rng.random() < 0.02 else 3, 40),
            })
    return pd.DataFrame(rows)


def test_select_panos_is_deterministic_and_disjoint():
    frame = synthetic_frame()
    a = mt.select_panos(frame)
    b = mt.select_panos(frame)
    pd.testing.assert_frame_equal(a, b)
    assert a["pano_id"].is_unique
    strata = list(a["stratum"].drop_duplicates())
    assert strata[0] == "representative"
    assert "near_horizon" in strata and strata[-1] == "ai"
    assert (a.loc[a["stratum"] == "representative", "pano_id"]
             .isin(frame.loc[frame["is_ai"], "pano_id"]).sum()) == 0


def test_select_panos_type_strata_target_the_rare_types():
    frame = synthetic_frame()
    plan = mt.select_panos(frame)
    type_strata = {s for s in plan["stratum"] if s.startswith("type:")}
    assert "type:Other" in type_strata  # rarest type must need a top-up
    n_other = frame.merge(plan, on="pano_id")
    n_other = (n_other[n_other["label_type"] == "Other"]).shape[0]
    assert n_other >= min(mt.TYPE_LABEL_QUOTA,
                          (frame["label_type"] == "Other").sum())


# ---------------------------------------------------------------------------- metrics

def test_implied_heights_recovers_exact_geometry():
    params = {"height_by_type_m": {"CurbRamp": 2.783}, "height_fallback_m": 2.715}
    dep = np.linspace(6, 40, 60)
    df = pd.DataFrame({
        "label_type": ["CurbRamp"] * 60,
        "depression_deg": dep,
        "truth_m": 2.783 / np.tan(np.radians(dep)),
    })
    out = mt.implied_heights(df, params)
    assert out["CurbRamp"]["implied_height_m"] == pytest.approx(2.783, rel=1e-9)
    assert out["CurbRamp"]["uses_fallback"] is False


def test_range_slope_sees_pure_compression():
    truth = np.linspace(1, 30, 200)
    err = -0.5 * truth + 3.0  # compressive: overshoot near, undershoot far
    assert mt.range_slope(err, truth) == pytest.approx(-0.5, abs=1e-12)
