"""The issue #5 findings, locked.

Convention (mirrors test_depth_validation_findings.py): the fast tests assert what the
2026-08-06 run measured, reading the committed data/pov-inversion-summary.json only. The
summary regenerates offline and deterministically with
`python python/run_pov_inversion.py --write`, and one session-scoped test below re-derives
the headline numbers in-process so the committed JSON cannot drift from the code.

Headline findings (reports/2026-08-06-pov-inversion.md):

- The exact POV inversion is verified against production three ways: pano_y replays
  exactly for every row, post-cutoff pano_x exactly in all six cities, and the pre-cutoff
  misses carry the per-pano-constant signature of camera_heading metadata drift.
- Scored on the R-fixture split, exact (0 params) beats est7 (6 params) pooled and at the
  canvas edges; est7's apparent zoom 2-3 win is two ground-truth artifacts: the era
  client's parseInt-truncated POV inputs, and one depth-grid column (360/512 deg) of
  bearing bias from Label.js's Math.ceil depth indexing.
- With both modeled (1 fitted constant), geometry beats the regression at every zoom.
- photographer_pitch carries no heading signal.
"""

import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "data", "pov-inversion-summary.json")
DEPTH_GRID_COLUMN_DEG = 360.0 / 512

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="pov-inversion summary not built yet"
)


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ fidelity replay

def test_pano_y_replays_exactly_everywhere(summary):
    """The vertical half has no free inputs; anything under 100% would be a math error."""
    for city, f in summary["fidelity"].items():
        if f["n_with_current_pano_xy"]:
            assert f["pano_y_exact_match_rate"] == 1.0, city


def test_post_cutoff_pano_x_replays_exactly(summary):
    """Post-evolution-179 labels were written by the front end running this same math."""
    for city, f in summary["fidelity"].items():
        if f["n_with_current_pano_xy"]:
            assert f["pano_x_exact_match_rate_post_cutoff"] == 1.0, city


def test_pre_cutoff_misses_are_metadata_drift_not_math(summary):
    """Constant within a pano (rounding-noise sigma), varying across panos."""
    f = summary["fidelity"]["seattle"]
    assert f["mismatch_within_pano_std_deg_median"] < 0.05
    assert f["mismatch_across_pano_std_deg"] > 0.2


def test_dc_has_no_replay_target(summary):
    """Evolution 179 never ran on the legacy DC database."""
    assert summary["fidelity"]["dc"]["n_with_current_pano_xy"] == 0


# ------------------------------------------------------------------ the comparison

def test_exact_beats_est7_pooled(summary):
    hm = summary["heading_error_median_deg"]
    assert hm["exact"] < hm["est7"]
    assert hm["est7"] == pytest.approx(1.3184, abs=0.02)
    assert hm["exact"] == pytest.approx(1.2500, abs=0.02)


def test_artifact_aware_model_beats_est7_at_every_zoom(summary):
    for z, v in summary["heading_error_median_deg_by_zoom"].items():
        assert v["era_cal"] < v["est7"], f"zoom {z}"


def test_est7_collapses_at_the_canvas_edge(summary):
    edge = summary["by_canvas_x_offset"][-1]
    assert edge["heading_error_median_deg"]["est7"] > 2.3
    assert edge["heading_error_median_deg"]["exact"] < 1.35


def test_latlng_error_improves(summary):
    lm = summary["latlng_error_median_m"]
    assert lm["era_cal"] < lm["exact"] < lm["est7"]


# ------------------------------------------------------------------ the two artifacts

def test_leftover_bias_is_one_depth_grid_column(summary):
    delta = summary["era_cal_delta_deg"]
    assert delta == pytest.approx(0.7198, abs=0.01)
    assert abs(delta - DEPTH_GRID_COLUMN_DEG) < 0.05


def test_era_model_beats_plain_exact(summary):
    """The truncation artifact is real: modeling it helps at every zoom."""
    for z, v in summary["heading_error_median_deg_by_zoom"].items():
        assert v["era"] < v["exact"], f"zoom {z}"


def test_photographer_pitch_carries_no_heading_signal(summary):
    ppc = summary["photographer_pitch_residual_check"]
    assert abs(ppc["pearson_r"]) < 0.02
    assert abs(ppc["slope_deg_per_deg"]) < 0.02


# ------------------------------------------------------------------ code <-> summary

def test_summary_reproduces_from_code(raw_data, summary):
    """Re-derive the headline numbers in-process so the committed JSON cannot drift from
    pov_inversion.py. Uses the session raw_data fixture; ~a minute of OLS and geodesy."""
    import sys

    sys.path.insert(0, os.path.join(ROOT, "python"))
    from label_latlng_estimation import (
        add_heading_diff, clean_data, fit_models, split_from_fixtures,
    )
    from pov_inversion import score_heading_swap, summarize_heading_swap

    cleaned, _ = clean_data(raw_data)
    cleaned = add_heading_diff(cleaned)
    train, test = split_from_fixtures(
        cleaned, os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    scored = score_heading_swap(fit_models(train, include_est6=False), train, test)
    fresh = summarize_heading_swap(scored)

    assert fresh["n_test"] == summary["n_test"]
    assert fresh["era_cal_delta_deg"] == pytest.approx(summary["era_cal_delta_deg"], abs=1e-9)
    for m, v in summary["heading_error_median_deg"].items():
        assert fresh["heading_error_median_deg"][m] == pytest.approx(v, rel=1e-6), m
    for m, v in summary["latlng_error_median_m"].items():
        assert fresh["latlng_error_median_m"][m] == pytest.approx(v, rel=1e-6), m


# ------------------------------------------------------------------ the era replica

def test_era_replica_reproduces_stored_sv_image_x(raw_data):
    """parseInt truncation + the half-degree offset reproduce the stored sv_image_x; the
    same math without truncation does not. This is the §3 evidence, re-derived."""
    import sys

    import pandas as pd

    sys.path.insert(0, os.path.join(ROOT, "python"))
    from pov_inversion import pov_if_centered

    d = raw_data[(raw_data["city"] == "seattle")
                 & (raw_data["computation_method"] == "depth")]
    cutoff = pd.Timestamp("2021-01-01", tz="UTC")
    d = d[d["time_created"].isna() | (d["time_created"] < cutoff)]
    stored = d["sv_image_x"].to_numpy(float)
    width = 13312.0

    def replica(truncate):
        h = d["heading"].to_numpy(float)
        p = d["pitch"].to_numpy(float)
        if truncate:
            h, p = np.trunc(h), np.trunc(p)
        pov_h, _ = pov_if_centered(d["canvas_x"], d["canvas_y"], h, p, d["zoom"])
        x = width * (pov_h / 360) + (width / 360) / 2
        x = np.where(x < 0, x + width, x)
        dx = (x - stored + width / 2) % width - width / 2
        return float((np.abs(dx) <= 1).mean())

    assert replica(truncate=True) > 0.99
    assert replica(truncate=False) < 0.2
