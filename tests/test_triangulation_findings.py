"""The issue #7 findings, locked: what bearing-only triangulation measured.

Convention (mirrors ``test_gbm_ceiling_findings.py``): these tests assert what the
2026-08-08 run measured, reading the committed ``data/triangulation-summary.json`` only.
The summary regenerates offline and deterministically with
``python python/run_triangulation.py build --write`` from the committed auto-labeler
inputs and the committed depth payloads. The estimator's *correctness* is tested
separately, on geometry with a known answer, in ``test_triangulation_contract.py``.

Headline findings (reports/2026-08-08-bearing-only-triangulation.md):

- **The estimator does not manufacture height.** Planting a known camera height on each
  run's own site geometry and regenerating every observation at that run's measured noise
  returns it to within 0.2% (bias factor 0.998-1.000, all six runs). So the spread between
  runs is the rigs and the detector, not the method.
- **The ecosystem's 2.6 m is too tall on every run**, GSV and Mapillary alike: the scale
  that makes multi-view ray geometry self-consistent is 0.898-0.984 of it. Measured with no
  depth data, no vertical model and no camera height assumed anywhere.
- **The two flattest GSV cities bracket the shipped 2.3412 m** (gainesville 2.33,
  paterson 2.38); bend and sao_paulo sit 5-9% above it. The anchor therefore confirms the
  shipped scale to within about 8% and decisively rejects 2.6 m — it does not confirm it
  to better than that.
- **Two independent measurements of the same pixels disagree by ~14%** (triangulated
  range over depth-derived range, pooled over four GSV cities). The disagreement is
  systematic: it survives every quality gate (bearing residual, conditioning, site size).
  Its cause is not resolved here and is recorded as open.
- **Pose quality is a property of the rig, not the imagery source**: richmond's four-rig
  Mapillary zoo has the worst panorama positions (sigma_pos 0.447 m) but clovis's single
  disciplined GoPro Fusion creator (0.138 m) beats paterson's GSV (0.185 m).
- **Camera tilt does not explain the depression trend.** The auto-labeler fused with
  ``apply_pose: false``, so uncorrected rig tilt was the leading hypothesis; applying the
  recorded pitch/roll under all four sign conventions leaves the trend *larger* than
  leaving it alone.
- **The bearings carry no systematic yaw error**: a fitted global rotation lands within
  0.15 deg of zero on every run and buys under 1% of the residual.
"""

from __future__ import annotations

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "data", "triangulation-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="triangulation summary not built yet"
)

GSV = ["paterson", "gainesville", "bend", "sao_paulo"]
MAPILLARY = ["richmond", "clovis"]
ALL = GSV + MAPILLARY


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ======================================================================================
# What the estimator is, and that it was not fed anything it claims not to use
# ======================================================================================

def test_declared_inputs_exclude_every_vertical_and_depth_quantity(summary):
    """The whole value of this issue is what the truth does NOT depend on."""
    meta = summary["meta"]
    for forbidden in ("vertical click angle", "camera height", "ground-plane",
                      "depth data", "resolution"):
        assert forbidden in meta["does_not_use"]
    assert meta["assumed_height_m"] == 2.6
    assert meta["shipped_height_m"] == pytest.approx(2.341219672825709)


def test_every_run_reported(summary):
    assert set(summary["scale_global"]) == set(ALL)
    for run in GSV:
        assert summary["imagery"][run] == "gsv"
    for run in MAPILLARY:
        assert summary["imagery"][run] == "mapillary"


# ======================================================================================
# Validation: the method returns a planted height
# ======================================================================================

def test_real_geometry_bootstrap_recovers_the_planted_height(summary):
    """The load-bearing validation: plant a known height on each run's own geometry,
    re-apply that run's measured noise, and the pipeline must return what it was given."""
    real = summary["validation"]["real_geometry"]
    assert set(real) == set(ALL)
    for run, v in real.items():
        assert v["bias_factor"] == pytest.approx(1.0, abs=0.01), run
        assert v["recovered_height_m"] == pytest.approx(v["planted_height_m"], abs=0.02), run


def test_synthetic_norm_convexity_correction_works(summary):
    """The raw leave-one-out range is inflated by norm convexity; the analytic correction
    must remove most of it, at every simulated noise level."""
    for key, v in summary["validation"]["synthetic"].items():
        assert v["bias_raw_m"] > 0, key
        assert abs(v["bias_corrected_m"]) < abs(v["bias_raw_m"]), key


# ======================================================================================
# The headline: absolute scale with no depth data
# ======================================================================================

def test_the_assumed_2p6_m_is_too_tall_on_every_run(summary):
    """The clearest actionable finding, and it needs no reference height at all: the
    multi-view-consistent scale is below 1.0 everywhere, GSV and Mapillary alike."""
    for run in ALL:
        k = summary["scale_global"][run]["k"]
        assert 0.85 < k < 1.0, f"{run}: k={k}"
        # and the fit is genuinely better than assuming 2.6 m
        g = summary["scale_global"][run]
        assert g["scatter_at_best_m"] < g["scatter_at_2p6_m"], run
        # an interior minimum: none of these k values is a clamp at the sweep boundary
        assert g["at_grid_edge"] is False, run


def test_implied_heights_bracket_the_shipped_constant(summary):
    """Two GSV cities land within a few percent of the shipped 2.3412 m and the other two
    sit above it. Recorded as a bracket, which is what the evidence supports."""
    h = {r: summary["scale_global"][r]["height_m"] for r in GSV}
    shipped = summary["meta"]["shipped_height_m"]
    assert h["gainesville"] == pytest.approx(shipped, rel=0.03)
    assert h["paterson"] == pytest.approx(shipped, rel=0.05)
    assert h["bend"] > shipped
    assert h["sao_paulo"] > shipped
    # every GSV run sits strictly between the shipped height and the assumed 2.6 m
    for run, v in h.items():
        assert shipped * 0.97 < v < 2.6, f"{run}: {v}"


def test_the_two_estimators_agree_within_a_decimetre(summary):
    """The per-member median and the global scale fit are different estimators of the same
    quantity, and they agree to within 0.10 m on every run.

    The median runs *higher* than the global fit everywhere (by 0.003-0.091 m), which is
    the expected direction: it divides by a noisy triangulated range, whereas the global
    fit estimates one scale from all members jointly and never takes that ratio. The global
    fit is therefore the one the report quotes.
    """
    for run in ALL:
        med = summary["scale"][run]["median_m"]
        glob = summary["scale_global"][run]["height_m"]
        assert abs(med - glob) < 0.10, f"{run}: {med} vs {glob}"
        assert med >= glob - 0.005, f"{run}: median below global fit, unexpected"


def test_only_the_global_fit_rejects_2p6_on_every_single_run(summary):
    """Stated precisely rather than conveniently: the global fit is below 2.6 m on all six
    runs, while the noisier per-member median lands essentially *at* 2.6 on sao_paulo
    (2.604). The rejection of 2.6 m is unanimous for the estimator the report quotes, and
    five-of-six for the other."""
    assert all(summary["scale_global"][r]["height_m"] < 2.6 for r in ALL)
    below = [r for r in ALL if summary["scale"][r]["median_m"] < 2.6]
    assert set(ALL) - set(below) == {"sao_paulo"}
    assert summary["scale"]["sao_paulo"]["median_m"] == pytest.approx(2.6, abs=0.02)


# ======================================================================================
# Robustness: the number is not an artefact of the gates or of the auto-labeler's fuse
# ======================================================================================

def test_conditioning_gate_does_not_move_the_answer(summary):
    """A real camera height is a property of the rig, so it must survive the gate sweep."""
    for run in ALL:
        vals = [v["median_m"] for v in
                summary["robustness"][run]["sensitivity"]["by_sigma_gate"].values()
                if v.get("median_m")]
        assert max(vals) - min(vals) < 0.05, f"{run}: {vals}"


def test_fuse_gate_selection_is_not_driving_the_scale(summary):
    """The auto-labeler fused at 2.6 m, so the population could in principle have been
    selected for consistency with it — but a wrong height only pushes members apart in
    proportion to how much their *ranges* differ, so the gate can only bite where the
    within-site range spread is large.

    The selection signature is therefore specific and directional: implied height should
    climb toward 2.6 m as the spread grows. It does not. On five of six runs the
    highest-spread stratum sits *below* the lowest-spread one, and the one exception
    (paterson) rises by 0.048 m — an order of magnitude too little to manufacture the
    0.2-0.3 m gaps this report is about.
    """
    for run in ALL:
        strata = summary["robustness"][run]["fuse_gate_selection"]
        vals = [v["implied_height_m"] for v in strata.values() if isinstance(v, dict)]
        assert len(vals) >= 3, run
        assert vals[-1] - vals[0] < 0.10, f"{run}: rises with range spread: {vals}"
        assert max(vals) - min(vals) < 0.20, f"{run}: {vals}"


def test_camera_tilt_hypothesis_is_rejected_on_the_gsv_runs(summary):
    """Uncorrected rig tilt was the leading explanation for the depression trend; the
    recorded pose makes it worse under all four sign conventions."""
    for run in GSV:
        t = summary["robustness"][run]["camera_tilt_hypothesis"]
        assert t["available"] is True, run
        assert "does NOT explain" in t["verdict"], run
        assert (t["best_corrected"]["depression_spread_m"]
                > t["uncorrected"]["depression_spread_m"]), run


def test_mapillary_runs_carry_no_pose_to_test(summary):
    for run in MAPILLARY:
        assert summary["robustness"][run]["camera_tilt_hypothesis"]["available"] is False


def test_no_systematic_yaw_error_in_the_bearings(summary):
    """A fitted global rotation is the check that the bearing half is clean; it must come
    back at essentially zero, otherwise every range here is suspect."""
    for run in ALL:
        off = summary["bearing_offset"][run]
        assert abs(off["best_offset_deg"]) <= 0.25, run
        gain = 1.0 - off["loss_at_best_m"] / off["loss_at_zero_m"]
        assert gain < 0.02, f"{run}: rotation bought {gain:.3f} of the residual"


# ======================================================================================
# The error budget
# ======================================================================================

def test_pose_quality_is_per_rig_not_per_imagery_source(summary):
    """Measured, and it contradicts the obvious guess: Mapillary poses are not uniformly
    worse than GSV's.

    richmond's four-rig zoo is the noisiest by a factor of ~2.4 (sigma_pos 0.447 m), but
    clovis — one creator, one GoPro Fusion, consistent sequences — comes in at 0.138 m,
    *better* than paterson's GSV (0.185 m). Pose quality tracks the rig and the capture
    discipline, not the imagery source.
    """
    n = {r: summary["noise"][r]["sigma_pos_m"] for r in ALL}
    assert n["richmond"] > 2 * max(n[r] for r in GSV)
    assert n["clovis"] < n["paterson"]


def test_bearing_noise_is_degree_scale_everywhere(summary):
    for run in ALL:
        assert 0.5 < summary["noise"][run]["sigma_bearing_deg"] < 4.0, run


def test_the_noise_decomposition_degenerates_on_two_runs(summary):
    """Recorded because it bounds where the error budget can be read.

    On gainesville and sao_paulo the fit puts essentially all of the perpendicular miss in
    the bearing term (sigma_pos -> 0, sigma_bearing 2.7-2.8 deg). The two components are
    told apart only by their range dependence, so when one is genuinely small the split is
    weakly identified — the contract tests demonstrate exactly this failure mode on
    synthetic data, and show the *height* is unaffected by it.
    """
    degenerate = [r for r in ALL if summary["noise"][r]["sigma_pos_m"] < 0.01]
    assert set(degenerate) == {"gainesville", "sao_paulo"}
    for run in degenerate:
        assert summary["noise"][run]["sigma_bearing_deg"] > 2.0
    # ...and the height still lands on both sides of the shipped constant, so the
    # degeneracy is not what produces the between-run spread
    assert summary["scale_global"]["gainesville"]["height_m"] < 2.4
    assert summary["scale_global"]["sao_paulo"]["height_m"] > 2.5


def test_split_half_precision_is_sub_metre(summary):
    """Model-free reproducibility: two disjoint halves of a site must land close."""
    for run in ALL:
        assert summary["split_half"][run]["median_half_separation_m"] < 1.5, run


# ======================================================================================
# Applicability — a subset estimator, and the report has to say how big the subset is
# ======================================================================================

def test_applicability_is_reported_and_honest(summary):
    for run in ALL:
        a = summary["applicability"][run]
        assert 0.0 < a["frac_sites_3plus_panos"] <= 1.0, run
        # error scales as 1/sin of the intersection angle, so the distribution matters
        assert a["intersection_angle_deg"]["median"] > 45, run
        assert a["intersection_angle_deg"]["frac_below_20deg"] < 0.10, run


# ======================================================================================
# The depth anchor: two independent systems on identical pixels
# ======================================================================================

def test_depth_anchor_present_and_sizeable(summary):
    a = summary["depth_anchor"]
    assert a["available"] is True
    assert a["n_panos"] >= 400
    assert a["n_detections"] >= 2000


def test_the_two_measurement_systems_disagree_by_about_fourteen_percent(summary):
    """The central open finding. Triangulated range runs ~14% longer than the depth-derived
    range at the very same detection pixels, consistently in all four GSV cities."""
    a = summary["depth_anchor"]
    assert a["pooled"]["median_ratio_tri_over_depth"] == pytest.approx(1.14, abs=0.04)
    for run, v in a["runs"].items():
        assert v["median_ratio_tri_over_depth"] > 1.05, run


def test_depth_implied_height_reproduces_prior_work(summary):
    """Sanity on the depth side: read at the auto-labeler's detection pixels, Google's
    depth implies a height a little below the 2.3412 m the modern-truth close-out measured
    on human clicks — the detector's click convention sits slightly above ground contact."""
    a = summary["depth_anchor"]
    assert 2.15 < a["implied_height_from_depth_m"] < a["implied_height_from_bearings_m"]
    assert a["implied_height_from_depth_m"] < summary["meta"]["shipped_height_m"]


def test_row_flipped_depth_lookup_finds_no_ground(summary):
    """A row-flipped raster reads sky, so it must lose essentially the whole population.
    The x-mirror control is deliberately NOT asserted to fail: the depth model is nearly
    flat-earth, so range is set by the row and barely by the column, and that control is
    weak by construction — the report says so rather than claiming a win."""
    c = summary["depth_anchor"]["frame_controls"]
    assert c["row_flip"]["n"] < 0.05 * c["identity"]["n"]
    assert c["rotate_180"]["n"] < 0.05 * c["identity"]["n"]


def test_panorama_positions_did_not_drift(summary):
    """The baselines triangulation stands on are the positions Google serves today: the
    auto-labeler's stored panorama positions match the freshly fetched photometa
    essentially exactly. Load-bearing — triangulated range scales with the baseline —
    and previously asserted only in prose; now computed and locked."""
    d = summary["depth_anchor"]["position_drift"]
    assert d["n"] == 480
    assert set(d["per_run"]) == set(GSV)
    assert d["median_m"] <= 0.001
    assert d["max_m"] < 0.15


def test_the_gap_is_a_scale_not_an_offset(summary):
    """The discriminating shape test between §8's two candidate causes.

    A detector centroid displaced toward the camera on a fixed-size object is an
    *additive* error: capped by the object's extent (a curb ramp is ~1-2 m), so the ratio
    would fall toward 1 with range — and the contract tests prove a radial displacement
    cannot bias the triangulated range at all. A depth model restating its own assumed
    ground plane is a *multiplicative* error: flat ratio, metre gap growing in proportion.
    The data is unambiguous: the ratio is flat from 4 m to 20 m while the metre gap grows
    to well past any ramp's extent, and the depth side's implied height is the same
    constant in every bin. What survives of the centroid candidate is only a
    proportional-in-range displacement, which a fixed-size object cannot produce."""
    prof = summary["depth_anchor"]["gap_range_profile"]
    rows = sorted(prof.values(), key=lambda v: v["median_r_tri_m"])
    assert len(rows) >= 5
    ratios = [v["median_ratio"] for v in rows]
    diffs = [v["median_diff_m"] for v in rows]
    assert all(1.05 < x < 1.20 for x in ratios), ratios
    assert max(ratios) - min(ratios) < 0.08, ratios
    assert diffs[0] < 0.6 and diffs[-1] > 2.0, diffs
    heights = [v["median_h_depth_m"] for v in rows]
    assert max(heights) - min(heights) < 0.05, heights


def test_the_gap_is_not_an_old_imagery_artifact(summary):
    """The bulk of the anchor sample is modern imagery and carries the full gap, so the
    13.8% is not inherited from the era-dependent plane scale the modern-truth close-out
    documented on old payloads (though the small pre-2016 stratum does run hotter, in the
    direction that history predicts)."""
    by = summary["depth_anchor"]["gap_by_capture_year"]
    assert by, "era stratification missing"
    bulk = max(by.values(), key=lambda v: v["n"])
    pooled = summary["depth_anchor"]["pooled"]["median_ratio_tri_over_depth"]
    assert bulk["n"] > 1000
    assert bulk["median_ratio"] == pytest.approx(pooled, abs=0.02)
    assert all(v["median_ratio"] > 1.05 for v in by.values())


# ======================================================================================
# Model scoring against a truth that shares none of the models' assumptions
# ======================================================================================

def test_deployed_linear_is_compressive_against_triangulated_truth(summary):
    """#4766's compression, reproduced once more against an absolute truth built from
    bearings — including on the two Mapillary cities, where no depth truth can exist."""
    for run in ALL:
        m = summary["scoring"][run]["models"]
        assert m["deployed_linear"]["range_slope"] < -0.2, run


def test_geometry_shaped_models_beat_the_deployed_linear_everywhere(summary):
    for run in ALL:
        m = summary["scoring"][run]["models"]
        assert (m["shipped_blend"]["median_abs_m"]
                < m["deployed_linear"]["median_abs_m"]), run


def test_shipped_blend_has_a_flatter_range_slope_than_the_deployed_model(summary):
    for run in ALL:
        m = summary["scoring"][run]["models"]
        assert abs(m["shipped_blend"]["range_slope"]) < abs(
            m["deployed_linear"]["range_slope"]), run


# ======================================================================================
# Cross-source: the one place the detector's click convention cancels
# ======================================================================================

def test_cross_source_reports_absolute_mapillary_rig_heights(summary):
    cs = summary["cross_source"]
    assert set(cs["absolute_rig_heights_m"]) == set(ALL)
    for run in MAPILLARY:
        v = cs["absolute_rig_heights_m"][run]["absolute_rig_m"]
        assert 1.8 < v < 3.2, f"{run}: {v}"
    assert "NOT identifiable from bearings alone" in cs["detector_click_offset_note"]
