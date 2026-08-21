"""Layer 3 of the depth-pilot suite: the observed findings, locked.

Convention (mirrors test_findings_vs_published.py): these assert what the
2026-08-05 pilot actually measured against the committed artifacts, so any
regression in the decoder, the v6 replication, or the artifact build that
changes a conclusion turns the suite red. They are statements about THIS
committed fetch; a deliberate refetch updates them alongside the report.

Headline findings (reports/2026-08-05-depth-pilot.md):
- 209/606 sampled 2017-2020 pano ids still resolve; every one that resolves
  serves a depth payload.
- 45 of 195 classified panos are bit-stable: recomputed label positions agree
  with the stored ground truth at the float32 storage floor (median 0.34 m).
- The rest drifted slightly under Google reprocessing (label median 0.98 m
  overall -- still below the deployed estimator's own 1.46 m median error).
- Part B: all 200 modern locations have a current pano; all serve depth;
  96.5% serve 16384x8192 imagery.
- Camera height is per-pano: the structural 2-plane default never occurs, but
  a ground plane pinned at exactly 2.5 m is common (68% of the 2017-2020
  panos, 27% of modern); measured heights (excluding pinned) have median
  2.37 m -- below the auto-labeler's hardcoded 2.6 m.
"""

import json
import os

import pandas as pd
import pytest

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SUMMARY_PATH = os.path.join(DATA, "depth-pilot-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="depth-pilot artifacts not built yet"
)


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def panos():
    return pd.read_csv(os.path.join(DATA, "depth-pilot-panos.csv.gz"))


# ---------- Part A: transport and stability of the depth product

def test_attrition_and_availability(summary):
    a = summary["part_a"]
    assert a["attempted"] == 576
    assert a["resolve_rate"] == pytest.approx(0.3403, abs=1e-4)
    # THE availability result: every pano id that still resolves serves depth
    assert a["depth_rate_among_resolved"] == 1.0


def test_pano_classification(summary):
    counts = summary["part_a"]["pano_class_counts"]
    assert counts == {
        "changed": 120, "mostly_unchanged": 27,
        "no_comparable_labels": 2, "unchanged": 47,
    }


def test_bit_stable_panos_sit_at_storage_floor(summary):
    # On unchanged panos the median disagreement is ~1 float32 ulp (~0.4 m) --
    # the storage lattice, not a physical error.
    med = summary["part_a"]["label_median_disagreement_m_unchanged_panos"]
    assert med == pytest.approx(0.3577, abs=1e-3)


def test_overall_drift_below_estimator_error(summary):
    # Even pooling drifted panos, fresh depth agrees with the 2020 ground truth
    # to better than the deployed estimator's own 1.46 m median error.
    a = summary["part_a"]
    assert a["labels_compared"] == 723
    assert a["label_median_disagreement_m"] == pytest.approx(0.9881, abs=1e-3)
    assert a["label_median_disagreement_m"] < 1.46


def test_lattice_threshold_sensitivity(summary):
    a = summary["part_a"]
    assert a["consistent_frac_at_1p5_ulp"] == pytest.approx(0.397, abs=1e-3)
    assert a["consistent_frac_at_2_ulp"] == pytest.approx(0.5201, abs=1e-3)
    assert a["consistent_frac_at_3_ulp"] == pytest.approx(0.6833, abs=1e-3)


def test_pano_reregistration(summary):
    # Google has re-registered these panos by ~0.8 m (median) since labeling.
    assert summary["part_a"]["median_pano_shift_m"] == pytest.approx(0.767, abs=1e-2)


def test_edge_rows(summary):
    a = summary["part_a"]
    assert a["absurd_rows_checked"] == 8
    assert a["absurd_reproduced"] == 2
    assert a["lost_plane_labels"] == 1


# ---------- Part B: modern coverage

def test_modern_coverage_is_total(summary):
    b = summary["part_b"]
    assert b["locations"] == 200
    assert b["found_current_pano"] == 200
    assert b["depth_served"] == 200
    assert b["unique_panos"] == 200


def test_modern_resolution(summary):
    assert summary["part_b"]["max_resolution_16384_frac"] == pytest.approx(0.965, abs=1e-3)


# ---------- camera height

def test_structural_default_never_occurs(summary):
    # The issue-comment detection advice needs refining: "exactly 2.500" in the
    # wild is a pinned ground plane inside a full payload, never the bare
    # two-plane default payload.
    assert summary["camera_height"]["structural_default_frac"] == 0.0


def test_pinned_25_is_vintage_dependent(panos):
    ok = panos[panos["status"] == "ok"]
    pinned = ok["ground_d_exactly_2p5"].astype(bool)
    a_frac = pinned[ok["part"] == "a"].mean()
    b_frac = pinned[ok["part"] == "b"].mean()
    assert a_frac == pytest.approx(0.6794, abs=1e-3)
    assert b_frac == pytest.approx(0.265, abs=1e-3)


def test_measured_heights(summary):
    ch = summary["camera_height"]
    assert ch["n_heights_excl_2p5"] == 214
    assert ch["median_height_excl_2p5"] == pytest.approx(2.366, abs=1e-3)
    # below the auto-labeler's DEFAULT_CAMERA_HEIGHT_M = 2.6
    assert ch["median_height_excl_2p5"] < 2.6
