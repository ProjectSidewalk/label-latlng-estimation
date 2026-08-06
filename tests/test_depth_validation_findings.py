"""Layer 3 for the issue #9 depth validation: the observed findings, locked.

Convention (mirrors test_depth_pilot_findings.py): these assert what the 2026-08-06 run
actually measured against the committed artifacts, so a regression in the decoder, the
frame helpers, or the artifact build that changes a conclusion turns the suite red. They
are statements about THIS committed fetch; a deliberate refetch updates them alongside
the reports.

Headline findings (reports/2026-08-06-depth-validation.md):

- The depth payload registers against the panorama's own imagery, which arrives from a
  different Google host. The true frame beats every deliberately wrong one, and the
  pooled column-offset sweep bottoms out at exactly zero.
- The product is a constructed model, not a measurement: essentially nothing sits on a
  surface tilted between 15 and 75 degrees, and the ground is within a metre of naive
  h/tan(depression) almost everywhere.
- Under a Sidewalk label that leaves the payload close to plain trigonometry, so the
  2021 estimator was fitting a relationship the payload had largely already reduced to
  geometry.
- Two captures of one street agree on the ground but place building facades metres
  apart -- facades being the model's only genuinely independent geometry.
"""

import json
import os

import pytest

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SUMMARY_PATH = os.path.join(DATA, "depth-validation-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="depth-validation artifacts not built yet"
)


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- T1: registration

def test_only_panoramas_with_structure_are_scored(summary):
    """A bare suburban street is evidence of nothing, and is reported as such."""
    t1 = summary["t1_registration"]
    assert t1["panos_with_imagery"] >= 55
    assert t1["panos_with_structure"] <= t1["panos_with_imagery"]
    assert t1["panos_with_structure"] >= 45


def test_true_frame_beats_every_wrong_frame(summary):
    signs = summary["t1_registration"]["paired_sign_test"]
    # A mirror is the subtle one; the vertical flips should be annihilated.
    assert signs["x_mirror"]["identity_better"] > 3 * signs["x_mirror"]["rival_better"]
    for control in ("rotate_180", "row_flip"):
        assert signs[control]["identity_better"] > 20 * max(signs[control]["rival_better"], 1)


def test_payload_beats_a_permutation_null(summary):
    null = summary["t1_registration"]["permutation_null"]
    assert null["k_per_pano"] >= 5
    # The true pairing should sit low in its own null, not at the middle.
    assert null["median_null_percentile"] < 0.35


def test_pooled_column_sweep_bottoms_out_at_zero_offset(summary):
    """The sharpest test that the mapping carries no rotation offset."""
    sweep = summary["t1_registration"]["pooled_column_sweep"]
    assert sweep["argmin_cols"] == 0
    assert sweep["mean_violation_at_0"] < sweep["mean_violation_at_plus_64"]
    assert sweep["mean_violation_at_0"] < sweep["mean_violation_at_minus_64"]


# ---------------------------------------------------------------- T2: what it is

def test_the_plane_set_is_a_manhattan_world(summary):
    """The finding: no car roofs, no canopy, no pitched roofs, no driveway ramps."""
    t2 = summary["t2_what_it_is"]
    assert t2["payloads"] >= 400
    assert t2["tilt_pixel_share_oblique_15to75deg"] < 0.02
    assert t2["tilt_pixel_share_horizontal_le10deg"] > 0.75
    assert t2["tilt_pixel_share_vertical_ge80deg"] > 0.05


def test_the_ground_is_nearly_flat_earth(summary):
    t2 = summary["t2_what_it_is"]
    assert t2["flat_earth_frac_within_1m"] > 0.85
    assert abs(t2["flat_earth_median_residual_m"]) < 0.1


# ---------------------------------------------------------------- T3: labels

def test_labels_land_on_modelled_ground(summary):
    counts = summary["t3_label_hits"]["hit_class_counts"]
    ground_like = counts.get("ground", 0) + counts.get("terrain", 0)
    assert ground_like > 0.85 * sum(counts.values())


def test_depth_adds_little_over_trigonometry_for_a_label(summary):
    """The honest ceiling context for #3."""
    t3 = summary["t3_label_hits"]
    assert t3["frac_labels_within_1m_of_flat_earth"] > 0.75
    assert t3["median_curb_bias_m"] > 0.3  # a bias, not noise: ~1/3 of the 1.47 m error


def test_occlusion_adjudication_is_recorded(summary):
    adj = summary["t3_label_hits"]["occlusion_adjudication"]
    assert adj is not None
    assert adj["sample_size"] >= 30
    assert 0 <= adj["occluded"] <= adj["sample_size"]
    assert adj["panoramas_with_an_occlusion"] <= adj["panoramas_in_sample"]


# ---------------------------------------------------------------- T4: cross-vintage

def test_two_captures_agree_on_ground_but_not_on_walls(summary):
    """Ground agreement is nearly free -- a flat plane is shift-invariant. Walls are not."""
    t4 = summary["t4_cross_vintage"]
    assert t4["pairs"] >= 20
    assert t4["median_year_gap"] >= 5
    assert t4["ground_median_residual_m"] < 1.0
    assert t4["facade_median_offset_m"] > t4["ground_median_residual_m"]
