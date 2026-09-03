"""The SidewalkWebpage#5084 sign-off findings, locked (reports/2026-09-02-production-signoff.md).

Headline claims this file holds together with data/signoff-summary.json:

- modern truth: the shipped estimator's median error is under half the deployed regression's on
  the representative stratum, the held-out re-calibration lands in the same place on every one
  of 200 pano-half splits, and a height calibrated on every OTHER city beats the regression in
  every city;
- era truth (the regression's own held-out split): the shipped estimator still edges the
  regression overall (a cluster-bootstrap CI that excludes zero) while carrying the bias the
  era truth's inflated scale predicts, and with the same one-parameter budget in that frame it
  wins by roughly half a metre;
- geodesy: sphere vs WGS84 is centimetres at any answer the estimator can return, and the
  three sphere radii in play differ by far less than that;
- the frame contract: a click projected through its own frame reproduces the position to the
  bit on every frame; the two wrong-frame conventions do not;
- the parity fixture regenerates deterministically and its reference values are what the
  Python port of the production formula returns.

The last test re-derives the modern-frame block in-process so the JSON cannot drift.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SUMMARY_PATH = os.path.join(DATA, "signoff-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH),
    reason="sign-off artifacts not built (run_signoff.py build --write)")


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------ modern frame

def test_shipped_constants_are_the_final_coefficients(summary):
    with open(os.path.join(DATA, "modern-truth-summary.json"), encoding="utf-8") as f:
        fc = json.load(f)["final_coefficients"]["params"]
    assert summary["meta"]["shipped"]["height_m"] == fc["height_m"]
    assert summary["meta"]["shipped"]["blend_deg"] == fc["blend_deg"]


def test_modern_headline(summary):
    rep = summary["modern_frame"]["representative"]
    assert summary["modern_frame"]["n_representative"] == 1484
    assert rep["approx3"]["median_m"] < 0.45
    assert rep["A_deployed"]["median_m"] > 1.0
    assert rep["approx3"]["median_m"] < 0.5 * rep["A_deployed"]["median_m"]
    assert rep["approx3"]["p90_m"] < rep["A_deployed"]["p90_m"]
    assert rep["approx3"]["win_rate_vs_A"] > 0.65
    ci = summary["modern_frame"]["bootstrap_median_diff_vs_A"]["representative"]["ci95_m"]
    assert ci[1] < -0.5


def test_modern_holdout_is_honest_and_stable(summary):
    h = summary["modern_frame"]["repeated_holdout"]
    assert h["n_rep"] == 200
    assert 0.40 < h["approx3_median_m"]["mean"] < 0.50
    assert h["approx3_median_m"]["p95"] - h["approx3_median_m"]["p5"] < 0.1
    assert h["shipped_beats_deployed_in_every_split"]
    assert abs(h["fitted_height_m"]["mean"] - summary["meta"]["shipped"]["height_m"]) < 0.01


def test_one_height_transfers_across_cities(summary):
    rows = summary["modern_frame"]["leave_one_city_out"]
    assert len(rows) >= 12
    for r in rows:
        assert r["approx3_loco_median_m"] < r["A_deployed_median_m"], r["city"]
        assert abs(r["height_fitted_elsewhere_m"] - summary["meta"]["shipped"]["height_m"]) < 0.02, r["city"]


def test_modern_wins_every_zoom_type_and_resolution(summary):
    m = summary["modern_frame"]
    for key in ("by_zoom", "by_label_type", "by_pano_height", "by_capture_year"):
        for r in m[key]:
            assert r["approx3"]["median_m"] < r["A_deployed"]["median_m"], (key, r)


def test_approximation1_is_the_2020_stopgap_in_both_frames(summary):
    """approximation1 (evolution 93: 10 m along the viewport heading) scored in production
    vocabulary. On the era split it must land on the 2021 analysis's own 'estimate 1' (4.8439 m,
    tests/fixtures/r-baseline); on modern truth its distance half alone is several times the
    regression's error and nowhere near the shipped estimator's."""
    era = summary["era_frame"]["overall"]["approx1"]
    assert abs(era["median_m"] - 4.8439) < 5e-4
    assert era["dist"]["median_m"] < era["median_m"]  # the viewport heading costs the other metres
    assert era["win_rate_vs_est7"] < 0.25
    rep = summary["modern_frame"]["representative"]
    assert 3.0 < rep["approx1"]["median_m"] < 4.5
    assert rep["approx1"]["median_m"] > 2.5 * rep["A_deployed"]["median_m"]
    assert rep["approx1"]["win_rate_vs_A"] < 0.2


def test_ideal_floor_brackets_the_shipped_estimator(summary):
    """The single-click floor (0.3 deg click noise through d = h / tan dep) is what any one-click
    estimator can resolve. approximation3 sits within ~2x of it out to 15 m and the gap opens past
    that, where the bounded tail and the far-field truth take over."""
    import signoff as so
    ideal = summary["modern_frame"]["ideal"]
    assert ideal["click_noise_sigma_deg"] == 0.3 and ideal["truth_band_m"] == [0.12, 0.17]
    floor = {r["distance_m"]: r["click_floor_median_m"] for r in ideal["table"]}
    assert abs(floor[10.0] - 0.159) < 2e-3 and abs(floor[15.0] - 0.348) < 2e-3
    for r in ideal["table"]:
        assert r["click_floor_median_m"] == pytest.approx(
            float(so.single_click_floor_m(r["distance_m"], ideal["height_m"])), abs=1e-12)
        assert r["click_floor_conservative_median_m"] > r["click_floor_median_m"]
    by_dist = {r["dist_bin"]: r["approx3"]["median_m"] for r in summary["modern_frame"]["by_true_distance"]}
    assert by_dist["5-10"] < 2.5 * floor[10.0] + ideal["truth_band_m"][1]
    assert by_dist["10-15"] < 2.0 * floor[15.0] + ideal["truth_band_m"][1]
    assert by_dist["20-30"] > 2.0 * floor[30.0]


def test_rig_tilt_does_not_reach_the_estimator_beyond_a_few_percent(summary):
    """The RQ4 rider: a tilt-shaped term on the shipped estimator is bounded at a few percent of
    the variance and a fraction of the full-tilt sensitivity, and the implied height is flat to
    a couple of centimetres across the tilt bins that hold nearly every label."""
    t = summary["modern_frame"]["rig_tilt_rider"]
    assert t["n_labels"] > 2000 and t["n_panos"] > 800
    assert not t["db_camera_roll_available"]  # the DB never had roll; the fresh fetch supplies it
    ih, se = t["implied_height"], t["approx3_signed_error"]
    assert ih["r2"] < 0.08 and se["r2"] < 0.08
    exp_h = t["expected_slope_if_tilt_entered_m_per_deg"]
    exp_d = t["expected_signed_error_slope_if_tilt_entered_m_per_deg"]
    for k in ("slope_pitch_m_per_deg", "slope_roll_m_per_deg"):
        assert abs(ih[k]) < 0.5 * exp_h
        assert abs(se[k]) < 0.5 * exp_d
    # sign coherence: a steeper-read ray raises the implied height and lowers the signed error
    assert np.sign(ih["slope_roll_m_per_deg"]) == -np.sign(se["slope_roll_m_per_deg"])
    bins = t["by_abs_projected_tilt"]
    heavy = [b for b in bins if b["n"] >= 250]
    assert sum(b["n"] for b in heavy) > 0.95 * t["n_labels"]
    spread = max(b["implied_height_median_m"] for b in heavy) - min(b["implied_height_median_m"] for b in heavy)
    assert spread < 0.05
    assert 0.0 < t["pano_level"]["pearson_r_ground_tilt_vs_rig_tilt"] < 0.6


# --------------------------------------------------------------------------- era frame

def test_era_continuity_row(summary):
    e = summary["era_frame"]
    assert e["n_test"] == 79029
    assert abs(e["overall"]["est7"]["median_m"] - 1.4621) < 5e-4
    assert abs(e["overall"]["blend_type_era"]["median_m"] - 0.9335) < 5e-4


def test_era_shipped_edges_the_regression_with_the_predicted_bias(summary):
    e = summary["era_frame"]
    assert e["overall"]["approx3"]["median_m"] < e["overall"]["est7"]["median_m"]
    assert e["bootstrap_median_diff_vs_est7"]["approx3"]["ci95_m"][1] < 0
    # the era truth's scale is inflated (modern-truth report SS7): the shipped height reads as too near there
    assert e["overall"]["approx3"]["signed_median_m"] < -0.8
    assert e["implied_height_overall_m"] > 2.55
    by_ph = {r["pano_height_px"]: r["implied_height_m"] for r in e["implied_height_by_pano_height"]}
    assert by_ph["8192"] < 2.45 and by_ph["6656"] > 2.7 and by_ph["0"] > 2.7


def test_modern_wins_every_city(summary):
    for r in summary["modern_frame"]["by_city"]:
        assert r["approx3"]["median_m"] < r["A_deployed"]["median_m"], r["city"]


def test_era_slice_table_cells(summary):
    """The SS4.3 slice table's era-calibrated column, which review found mis-transcribed once."""
    by_ph = {r["pano_height_px"]: r for r in summary["era_frame"]["by_pano_height"]}
    assert by_ph["0"]["n"] == 46543 and abs(by_ph["0"]["approx3_eraflat"]["median_m"] - 0.979) < 1e-3
    assert abs(by_ph["6656"]["approx3_eraflat"]["median_m"] - 0.803) < 1e-3
    assert abs(by_ph["8192"]["approx3_eraflat"]["median_m"] - 1.016) < 1e-3
    assert abs(by_ph["6656"]["approx3"]["median_m"] - 1.629) < 1e-3
    assert abs(by_ph["8192"]["approx3"]["median_m"] - 0.521) < 1e-3


def test_era_equal_budget_wins_by_half_a_metre(summary):
    e = summary["era_frame"]
    assert 2.55 < e["era_flat_height_m"] < 2.75
    assert e["overall"]["approx3_eraflat"]["median_m"] < 1.0
    assert e["bootstrap_median_diff_vs_est7"]["approx3_eraflat"]["ci95_m"][1] < -0.4


def test_era_record_path_matches_the_harness(summary):
    r = summary["era_frame"]["record_path"]
    assert r["n_with_record"] > 30000
    assert r["record_vs_harness_m"]["median_m"] < 0.02
    assert r["record_vs_harness_m"]["p90_m"] < 0.05


# ----------------------------------------------------------------------------- geodesy

def test_geodesy_is_centimetres(summary):
    g = summary["geodesy"]
    assert g["radii_m"] == {"production_scala_sql": 6371000.0, "client_turf": 6371008.8,
                            "harness_geosphere": 6378137.0}
    assert g["worst_ellipsoid_vs_production_at_max_answer_m"] < 0.12
    for c in g["per_city"]:
        for r in c["rows"]:
            assert r["turf_vs_production_max_m"] < 1e-4
            assert r["harness_vs_production_max_m"] < 0.06
            assert r["ellipsoid_vs_production_max_m"] < 0.0046 * r["distance_m"] + 1e-6


# ------------------------------------------------------------------------ frame contract

def test_frame_contract(summary):
    frames = summary["viewport_frame_contract"]["frames"]
    assert len(frames) == 5
    for f in frames:
        assert f["own_frame_max_error_m"] < 1e-9, f["frame"]
        if f["width"] == 720:
            assert f["axis_scaled_to_720x480"]["p90_m"] < 1e-9
            assert f["width_scaled_read_as_720x480"]["p90_m"] < 1e-9
        else:
            assert f["axis_scaled_to_720x480"]["p90_m"] > 0.5
            assert f["width_scaled_read_as_720x480"]["p90_m"] > 4.0


# ------------------------------------------------------------------ fixture & re-derivation

def test_parity_fixture_is_deterministic_and_matches_the_port():
    import signoff as so
    shipped = so.load_shipped(DATA)
    a = so.parity_fixture(shipped)
    b = so.parity_fixture(shipped)
    assert a == b
    assert len(a["cases"]) == 58
    c = a["cases"][0]
    lat, lng, dist, heading, pitch = so.production_to_latlng(
        c["pano_lat"], c["pano_lng"], c["pano_x"], c["pano_y"], c["pano_width"], c["pano_height"],
        c["camera_heading"], shipped)
    assert float(lat) == c["expected"]["lat"] and float(lng) == c["expected"]["lng"]
    assert abs(float(dist) - 5.652204286630527) < 1e-9  # 22.5 deg down: h / tan(22.5)
    # the tail's structural maximum, reached at and above the horizon
    above = next(x for x in a["cases"] if x["name"].startswith("above the horizon"))
    assert abs(above["expected"]["distance_m"] - 23.848261259830384) < 1e-9


def test_modern_frame_rederives(summary):
    import signoff as so
    _, modern = so.modern_frame(so.load_shipped(DATA), DATA)
    for k in ("A_deployed", "approx3"):
        for stat in ("median_m", "p90_m", "signed_median_m"):
            assert modern["representative"][k][stat] == pytest.approx(
                summary["modern_frame"]["representative"][k][stat], abs=1e-9)
    assert modern["repeated_holdout"]["approx3_median_m"]["mean"] == pytest.approx(
        summary["modern_frame"]["repeated_holdout"]["approx3_median_m"]["mean"], abs=1e-9)
