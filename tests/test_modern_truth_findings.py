"""The modern-truth findings, locked (issue #3; reports/2026-08-07-modern-truth.md).

Headline claims this file holds together with the committed artifacts:

- the sampler frame is right: on the 409 committed pilot payloads, the modern
  heading-centred lookup lands on the same raster cell as the yaw-corrected era lookup
  (row exactly, column within the ceil-vs-round half-pixel), and the x-mirrored lookup
  agrees nowhere; on the modern set itself, the identity frame beats every deliberately
  wrong frame on ground-hit share and on error;
- the circularity guard: stored post-2021 positions are estimator echoes in BOTH eras
  (fixed-frame before evolution 179, real pixels after), so no evaluation here may use
  stored positions as truth — and the era boundary leaves a measurable discontinuity;
- the absolute check: the shipped blend's error and bias against fresh-depth truth on
  post-2021 human labels, vs the deployed per-zoom linear path (see the locked numbers).

The final test re-derives the committed summary's headline block in-process from the
committed payloads + labels, so the JSON cannot drift from the code.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SUMMARY_PATH = os.path.join(DATA, "modern-truth-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH),
    reason="modern-truth artifacts not built (run_modern_truth.py build --write)")


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def labels():
    return pd.read_csv(os.path.join(DATA, "modern-truth-labels.csv.gz"),
                       dtype={"pano_id": str})


@pytest.fixture(scope="module")
def panos():
    return pd.read_csv(os.path.join(DATA, "modern-truth-panos.csv.gz"),
                       dtype={"pano_id": str})


@pytest.fixture(scope="module")
def payloads():
    out = {}
    with gzip.open(os.path.join(DATA, "modern-truth-payloads.jsonl.gz"), "rt",
                   encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["pano_id"]] = rec["b64"]
    return out


# ---------------------------------------------------------------------- artifact shape

def test_artifacts_are_consistent(summary, labels, panos, payloads):
    meta = summary["meta"]
    assert meta["n_panos_attempted"] == len(panos)
    assert meta["n_panos_ok"] == (panos["status"] == "ok").sum() == len(payloads)
    assert meta["n_labels_on_ok_panos"] == len(labels)
    assert meta["n_scored"] == labels["gate_ok"].sum()
    assert set(labels.loc[labels["gate_ok"], "hit_class"]) <= {"ground", "terrain"}
    assert labels["pano_id"].isin(set(payloads)).all()


def test_every_stratum_met_its_budget(summary):
    by = summary["fetch"]["by_stratum_ok"]
    b = summary["meta"]["budgets"]
    assert by["representative"] == b["representative"]
    assert by["near_horizon"] == b["near_horizon"]
    assert by["ai"] == b["ai"]


# ---------------------------------------------------------------------- the sampler frame

def test_modern_lookup_matches_era_lookup_on_the_409_pilot_payloads():
    """The offline cross-check: same physical pixel, two independent lookup paths."""
    import gsv_depth as gd
    import modern_truth as mt
    from label_latlng_estimation import load_data

    pilot_ids = set()
    with gzip.open(os.path.join(DATA, "depth-pilot-payloads.jsonl.gz"), "rt",
                   encoding="utf-8") as f:
        for line in f:
            pilot_ids.add(json.loads(line)["pano_id"])
    yaw = (pd.read_csv(os.path.join(DATA, "depth-validation-panometa.csv.gz"),
                       dtype={"pano_id": str})
           .set_index("pano_id")["yaw_deg"].to_dict())
    raw = load_data(DATA).rename(columns={"gsv_panorama_id": "pano_id"})
    sub = raw[raw["pano_id"].isin(pilot_ids)
              & raw["current_pano_x"].notna() & raw["pano_width"].notna()]

    d_col, d_row, d_mirror = [], [], []
    for r in sub.itertuples():
        y = yaw.get(r.pano_id)
        if y is None or not np.isfinite(y):
            continue
        col_m, row_m = mt.modern_col_row(r.current_pano_x, r.current_pano_y,
                                         r.pano_width, r.pano_height)
        bearing = r.sv_image_x / 13312.0 * 360.0
        fx = ((bearing - y + 180.0) % 360.0) / 360.0
        col_v6 = math.ceil(fx * 13312.0 * gd.SV_IMAGE_SCALE) % 512
        row_v6 = math.ceil((gd.SV_IMAGE_Y_ORIGIN - r.sv_image_y) * gd.SV_IMAGE_SCALE)
        d_col.append((col_m - col_v6 + 256) % 512 - 256)
        d_row.append(row_m - row_v6)
        d_mirror.append(((511 - col_m) - col_v6 + 256) % 512 - 256)

    d_col, d_row, d_mirror = map(np.asarray, (d_col, d_row, d_mirror))
    assert len(d_col) >= 500
    assert (np.abs(d_row) <= 1).mean() == 1.0
    assert (np.abs(d_col) <= 2).mean() >= 0.90  # tail = legacy camera-heading drift
    assert (np.abs(d_mirror) <= 2).mean() == 0.0  # the mirrored frame agrees NOWHERE


def test_identity_frame_beats_every_wrong_frame(summary):
    fc = summary["frame_controls"]
    ident = fc["identity"]
    for control in ("x_mirror", "rotate_180", "row_flip"):
        assert ident["ground_or_terrain_share"] > fc[control]["ground_or_terrain_share"]
        if fc[control]["D_blend_median_abs_m"] is not None:
            assert ident["D_blend_median_abs_m"] < fc[control]["D_blend_median_abs_m"]


# ---------------------------------------------------------------------- the guard

def test_stored_positions_are_estimator_echoes_in_both_eras(summary):
    g = summary["guard"]
    for era in ("fixed_frame", "real_pixels"):
        assert g[era]["median_abs_diff_m"] < 0.35
        # the wrong era's formula misses by many times the right one's
        assert g[era]["wrong_era_median_abs_m"] > 4 * g[era]["median_abs_diff_m"]


# ---------------------------------------------------------------------- locked numbers

def test_locked_headline_matrix(summary):
    head = summary["matrix"]["headline_representative_human"]
    assert head["n"] == 1502
    assert head["A_deployed"]["median_abs_m"] == pytest.approx(1.0981, abs=2e-4)
    assert head["A_deployed"]["signed_median_m"] == pytest.approx(-0.2816, abs=2e-4)
    assert head["B_normalized"]["signed_median_m"] == pytest.approx(1.7727, abs=2e-4)
    assert head["C_anchor"]["signed_median_m"] == pytest.approx(1.0191, abs=2e-4)
    assert head["D_blend"]["median_abs_m"] == pytest.approx(1.2044, abs=2e-4)
    assert head["D_blend"]["signed_median_m"] == pytest.approx(1.0853, abs=2e-4)
    # the blend keeps the best tail even as shipped
    assert head["D_blend"]["p90_abs_m"] < head["A_deployed"]["p90_abs_m"]


def test_the_blend_bias_is_a_uniform_scale_error(summary):
    """Every scoreable city and capture year floats the same direction."""
    for city, v in summary["by_city"].items():
        assert 0.3 < v["D_blend"]["signed_median_m"] < 2.1, city
    for year, v in summary["by_capture_year"].items():
        assert 0.7 < v["D_blend"]["signed_median_m"] < 1.5, year


def test_implied_heights_are_flat_at_the_measured_rig(summary):
    ch = summary["camera_heights"]
    assert ch["measured_median_m"] == pytest.approx(2.3544, abs=2e-3)
    assert ch["pinned_2p5_frac"] == pytest.approx(0.313, abs=5e-3)
    for t, v in summary["implied_heights"].items():
        assert 2.25 < v["implied_height_m"] < 2.40, t
        if not v["uses_fallback"]:
            # the fitted table sits well above what modern truth supports
            assert v["fitted_height_m"] - v["implied_height_m"] > 0.14, t
    # the road-paint consistency check: Crosswalk implies the measured rig itself
    assert (summary["implied_heights"]["Crosswalk"]["implied_height_m"]
            == pytest.approx(ch["measured_median_m"], abs=0.05))


def test_locked_guard_numbers(summary):
    g = summary["guard"]
    assert g["fixed_frame"]["median_abs_diff_m"] == pytest.approx(0.2919, abs=2e-3)
    assert g["real_pixels"]["median_abs_diff_m"] == pytest.approx(0.0979, abs=2e-3)
    assert g["real_pixels"]["frac_echo"] == pytest.approx(0.976, abs=5e-3)


def test_locked_remedies(summary):
    r = summary["remedies"]
    assert r["k_rescale"] == pytest.approx(0.8629, abs=2e-4)
    assert r["flat_height_m"] == pytest.approx(2.3314, abs=2e-3)
    t = r["test_half"]
    assert t["D_blend_as_shipped"]["median_abs_m"] == pytest.approx(1.2683, abs=2e-4)
    assert t["D_rescaled"]["median_abs_m"] == pytest.approx(0.4329, abs=2e-4)
    assert t["D_flat"]["median_abs_m"] == pytest.approx(0.4215, abs=2e-4)
    # the rescale removes the bias, not just the spread
    assert abs(t["D_rescaled"]["signed_median_m"]) < 0.3


def test_modern_truth_does_not_carry_the_era_curb_overshoot(summary):
    c = summary["curb_sensitivity"]
    # applying the classic curb correction to modern truth WORSENS the bias
    assert abs(c["signed_median_corrected_m"]) > abs(c["signed_median_m"])


def test_near_horizon_clamp_is_least_bad(summary):
    nh = summary["near_horizon"]["(0, 2]"]
    assert abs(nh["D_blend"]["signed_median_m"]) < abs(nh["A_deployed"]["signed_median_m"])
    assert abs(nh["D_blend"]["signed_median_m"]) < abs(nh["C_anchor"]["signed_median_m"])


# ---------------------------------------------------------------------- reproduction

def test_summary_reproduces_from_committed_artifacts(summary, labels, payloads):
    """Re-derive the headline matrix and a sample of truth values in-process."""
    import gsv_depth as gd
    import depth_validation as dv
    import modern_truth as mt

    human = labels[labels["gate_ok"] & ~labels["is_ai"]]
    head = human[human["stratum"] == "representative"]
    got = mt.model_metrics(head)
    want = summary["matrix"]["headline_representative_human"]
    assert got["n"] == want["n"]
    for key in mt.MODEL_KEYS:
        for stat in ("median_abs_m", "signed_median_m", "p90_abs_m"):
            assert got[key][stat] == pytest.approx(want[key][stat], abs=1e-9)

    # truth values in labels.csv.gz must come from the committed payload bytes
    rng = np.random.default_rng(666)
    sample = labels.sample(40, random_state=666)
    for r in sample.itertuples():
        payload = gd.decode_depth_payload(payloads[r.pano_id])
        qc = gd.camera_height_qc(payload)
        hit = mt.classify_modern_label(payload, r.pano_x, r.pano_y, r.pano_width,
                                       r.pano_height, qc.ground_height)
        assert hit.hit_class == r.hit_class
        if math.isfinite(hit.horizontal_m) or math.isfinite(r.truth_m):
            assert hit.horizontal_m == pytest.approx(r.truth_m, abs=1e-6)
