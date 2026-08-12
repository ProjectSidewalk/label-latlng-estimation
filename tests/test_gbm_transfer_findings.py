"""The #6 transfer findings, locked: the ceiling does not survive a change of truth frame.

Convention (mirrors test_gbm_ceiling_findings.py): these tests assert what the 2026-08-10
run measured, reading the committed data/gbm-transfer-summary.json only. That summary
regenerates offline and deterministically with `python python/run_gbm_transfer.py --write`
(fixed seeds, deterministic LightGBM params, ~2 min); the runner asserts in-process that
its boosters reproduce data/gbm-ceiling-summary.json and that its closed-form rows
reproduce data/modern-truth-summary.json, and the cross-summary tests below re-check both
from the committed artifacts.

Headline findings (reports/2026-08-10-gbm-transfer.md):

- Scored against modern measured-plane truth, the era-trained booster keeps its whole
  advantage over the ERA blend (+108% in the era frame, +118% here) — so this is not a
  model that simply fails out of sample.
- But once each side carries one modern parameter fitted on a disjoint panorama half, the
  shipped 2-parameter closed form WINS: 0.410 m against 0.498 m for the booster plus a
  scale, and against every richer recalibration tried (affine, monotone quantile map) and
  against a booster trained on modern truth itself. Every paired cluster-bootstrap
  interval excludes zero.
- The mechanism: the era truth's implied camera height is 2.80 m where pano_height is
  absent (DC) and at 6656 px, but 2.35 m at 8192 px — within 2 cm of both the modern
  measurement and the shipped constant. A booster that can read pano_height learns which
  subpopulation answers on which scale. Those eight extra inputs are worth 0.44 m inside
  the era truth and -0.01 m outside it.
- What transfers cleanly is the tail: p90 3.55 m -> 2.80 m for the booster plus a scale
  and 1.99 m for the modern-trained one, where the era blend gets no such benefit. The
  far field beyond 15 m transfers only weakly — the booster beats the shipped form there,
  but so does the era blend by being biased 1.07 m long, and the booster leads outright
  only in the 30-50 m bin (54 rows).
"""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "data", "gbm-transfer-summary.json")
CEILING_PATH = os.path.join(ROOT, "data", "gbm-ceiling-summary.json")
MODERN_PATH = os.path.join(ROOT, "data", "modern-truth-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="gbm-transfer summary not built yet"
)

RECALIBRATED = ["gbm_l1_scaled", "gbm_dep_l1_scaled", "gbm_l2_scaled",
                "only_sv_image_y_scaled", "gbm_l1_affine", "gbm_l1_quantile",
                "gbm_dep_l1_affine", "gbm_dep_l1_quantile", "gbm_modern"]


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ceiling():
    with open(CEILING_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def modern():
    with open(MODERN_PATH, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- population

def test_the_population_is_the_stage4_scored_population(summary):
    pop = summary["population"]
    assert pop["n_gated_human"] == 2655
    assert pop["n_panos"] == 922
    assert pop["n_cities"] == 36
    assert pop["time_created_range"][0] >= "2021-01-01"  # disjoint from the era split


def test_the_modern_rows_sit_inside_the_era_training_support(summary):
    """A booster cannot extrapolate, so a transfer test on out-of-support rows would be
    measuring the edge leaves rather than the model. Every feature is interpolation."""
    for feature, block in summary["population"]["support_shift"].items():
        assert block["frac_outside_era_range"] < 0.02, feature


def test_two_label_types_are_new_and_the_cut_exists_for_them(summary):
    """Crosswalk and Signal postdate the era categorical, so 16% of rows reach the booster
    with a missing category. The seen/unseen cut is what stops that being an excuse."""
    lt = summary["population"]["label_types"]
    assert lt["unseen_types"] == ["Crosswalk", "Signal"]
    assert lt["n_unseen_rows"] == 433
    assert 0.16 < lt["frac_unseen_rows"] < 0.17
    seen = summary["by_label_type_seen"]["seen_types"]
    assert seen["n"] == 1123
    # on the rows whose type the booster DID see, the closed form still wins
    assert seen["D_flat"]["median_abs_m"] < seen["gbm_l1_scaled"]["median_abs_m"]


# ------------------------------------------------------------------------ frame mapping

def test_the_frame_mapping_is_exact_against_the_pixel_angle(summary):
    frame = summary["frame_mapping"]
    assert frame["n"] == 2655
    assert frame["max_abs_diff_vs_pixel_angle_px"] < 1e-9
    assert abs(frame["px_per_deg"] - 6656.0 / 180.0) < 1e-12


def test_the_stored_pixel_and_the_exact_projection_agree(summary):
    """The mapped column is a click angle, not a storage artifact: the #5 projection —
    computed from canvas and POV without touching pano_y — lands on it to a third of a
    pixel, which at 36.98 px/deg is under 0.01 degrees."""
    vs = summary["frame_mapping"]["vs_exact_projection_px"]
    assert abs(vs["median"]) < 0.01
    assert abs(vs["p10"]) < 0.5 and abs(vs["p90"]) < 0.5


def test_the_era_frames_own_pixels_discriminate_the_two_candidate_mappings(summary):
    """§3's third check, read off the artifact that now carries it.

    The mapping leaves the same ~15 px of pano re-registration drift in BOTH height
    groups. The unmapped alternative is identical at 6656 px — the same expression, since
    the calibration frame IS 6656 — and 140 px out at 8192, which is the 23% frame error
    #4765 was. That is what makes this a conversion rather than a fitted fudge.
    """
    era = summary["frame_mapping"]["era_frame_residuals_px"]
    assert era["n_rows"] == 162846
    at6656, at8192 = era["by_pano_height"]["6656"], era["by_pano_height"]["8192"]
    assert (at6656["n"], at8192["n"]) == (37494, 125352)
    assert abs(at6656["mapped_px"] - 15.000) < 5e-3
    assert abs(at8192["mapped_px"] - 14.562) < 5e-3
    assert abs(at6656["mapped_px"] - at8192["mapped_px"]) < 1.0   # the agreement
    assert abs(at6656["raw_px"] - at6656["mapped_px"]) < 1e-9     # degenerate at 6656
    assert at8192["raw_px"] > 100.0                               # and not subtle at 8192


# --------------------------------------------------------------------------- provenance

def test_the_summary_records_the_host_that_built_it(summary):
    """Issue #22: LightGBM does not reproduce bit-for-bit across platforms, so "which host
    built this" has to be a record. Asserted structurally, never by value — the whole point
    is that another machine can regenerate this file and write its own."""
    host = summary["meta"]["host"]
    assert {"platform", "machine", "python", "libraries"} <= set(host)
    assert host["libraries"]["lightgbm"] == summary["meta"]["lightgbm_version"]
    assert all(host[k] for k in ("platform", "machine", "python"))


# ------------------------------------------------------------- cross-summary comparability

def test_the_boosters_are_the_committed_ceiling_boosters(summary, ceiling):
    """Same rounds, same era-test numbers — otherwise 'the #6 model' means nothing here.

    `within_tolerance` is the assertion; `bit_identical` is a recorded fact about the host
    that ran it, deliberately NOT required (issue #22). A regeneration on another platform
    can land a fifth-decimal away on `gbm_dep_l1` and still be this artifact — what it
    cannot do is move any number this report quotes.
    """
    verdict = summary["meta"]["boosters_match_committed_ceiling"]
    assert verdict["within_tolerance"] is True
    assert verdict["best_iterations_match"] is True
    assert isinstance(verdict["bit_identical"], bool)
    assert verdict["max_abs_delta_m"] < max(verdict["tolerance_m"].values())
    assert verdict["exceeded"] == {}
    for key, ref in summary["era_reference"]["models"].items():
        for metric, value in ref.items():
            assert abs(value - ceiling["matrix"][key][metric]) < 1e-9, (key, metric)
    for key, rounds in summary["meta"]["best_iterations"].items():
        assert rounds == ceiling["meta"]["best_iterations"][key]


def test_the_closed_forms_are_the_committed_stage4_rows(summary, modern):
    """The held-out closed-form rows ARE modern_truth.remedy_check's, to float precision:
    same split, same parameters, so the booster rows are added to that table rather than
    compared against a re-derivation of it."""
    held = summary["held_out_half"]
    remedies = modern["remedies"]
    assert abs(held["k_rescale"] - remedies["k_rescale"]) < 1e-12
    assert abs(held["flat_height_m"] - remedies["flat_height_m"]) < 1e-12
    assert held["split"]["n_test_rows"] == remedies["split"]["n_test_rows"]
    for ours, theirs in (("A_deployed", "A_deployed"), ("D_blend", "D_blend_as_shipped"),
                         ("D_rescaled", "D_rescaled"), ("D_flat", "D_flat")):
        for metric in ("median_abs_m", "signed_median_m", "p90_abs_m"):
            assert abs(held["models"][ours][metric]
                       - remedies["test_half"][theirs][metric]) < 1e-9, (ours, metric)


def test_nothing_was_refitted_on_modern_data_except_one_number_per_model(summary):
    scales = summary["held_out_half"]["gbm_scales"]
    assert set(scales) == {"gbm_l1", "gbm_dep_l1", "gbm_l2", "only_sv_image_y"}
    # the era-trained boosters need almost no rescaling; the 1-D one needs the blend's
    assert all(0.99 < v < 1.06 for k, v in scales.items() if k != "only_sv_image_y")
    assert abs(scales["only_sv_image_y"] - 0.8865) < 0.001
    assert abs(scales["only_sv_image_y"] - summary["held_out_half"]["k_rescale"]) < 0.03


# ------------------------------------------------------------------------- the transfer

def test_the_raw_advantage_over_the_era_blend_survives(summary):
    """Held constant for the era scale handicap, the booster keeps its whole margin. This
    is the control that stops the headline being read as 'the model just broke'."""
    h = summary["headline"]
    assert h["era_frame_gap_pct_D_over_gbm_l1_dist"] > 100.0
    assert h["modern_raw_gap_pct_D_blend_over_gbm_l1"] > 100.0
    pooled = summary["pooled"]["models"]
    assert pooled["gbm_l1"]["median_abs_m"] < 0.62
    assert pooled["D_blend"]["median_abs_m"] > 1.25
    # and the booster arrives nearly unbiased where the era blend arrives a metre long
    assert abs(pooled["gbm_l1"]["signed_median_m"]) < 0.25
    assert pooled["D_blend"]["signed_median_m"] > 1.0


def test_the_calibrated_ceiling_inverts(summary):
    """The finding: one modern parameter each, and the 2-parameter closed form is ahead."""
    held = summary["held_out_half"]["models"]
    assert abs(held["D_flat"]["median_abs_m"] - 0.4100) < 0.001
    assert abs(held["gbm_l1_scaled"]["median_abs_m"] - 0.4977) < 0.001
    assert summary["headline"]["modern_calibrated_gap_pct_D_flat_over_gbm_l1_scaled"] < 0


def test_no_recalibration_rescues_the_booster(summary):
    """Richer recalibrations, and a booster trained on modern truth itself, all lose --
    so the result is not an artifact of granting the booster only one parameter."""
    held = summary["held_out_half"]["models"]
    d_flat = held["D_flat"]["median_abs_m"]
    for key in RECALIBRATED:
        assert held[key]["median_abs_m"] > d_flat, key
    assert summary["headline"]["modern_gap_pct_D_flat_over_best_recalibrated_gbm"] < 0


def test_every_paired_interval_excludes_zero(summary):
    """n = 1,362 clustered in 461 panoramas, so the separations are asserted with a paired
    cluster bootstrap rather than eyeballed."""
    boot = summary["held_out_half"]["bootstrap"]
    assert boot["cluster"] == "pano_id"
    assert boot["reference"] == "D_flat"
    for key, diff in boot["paired_diff_vs_reference"].items():
        if key == "D_rescaled":
            continue  # the other closed form: within noise of the shipped one, as Stage 4 found
        assert diff["delta_median_abs_m_lo"] > 0, key
        assert diff["frac_draws_better_than_reference"] < 0.05, key


def test_the_interaction_structure_is_worth_nothing_outside_the_era_truth(summary):
    """The load-bearing comparison, internal to the booster family so no closed form's
    calibration enters it: full booster minus single-signal booster, in each frame."""
    s = summary["headline"]["structure_worth"]
    assert s["era_frame"]["worth_m"] > 0.40
    assert abs(s["modern_calibrated"]["worth_m"]) < 0.05


# --------------------------------------------------------------------------- the mechanism

def test_the_era_truths_scale_splits_by_panorama_resolution(summary):
    """Why it happens: the era truth is not one scale, and resolution is the axis."""
    era = summary["truth_scale_by_resolution"]["era_truth"]["by_pano_height"]
    modern = summary["truth_scale_by_resolution"]["modern_truth"]["by_pano_height"]
    shipped = summary["truth_scale_by_resolution"]["shipped_flat_height_m"]
    assert abs(era["missing"]["implied_height_m"] - 2.80) < 0.02   # DC, 59% of era rows
    assert abs(era["1664"]["implied_height_m"] - 2.06) < 0.02      # 272 rows, the low end
    assert abs(era["6656"]["implied_height_m"] - 2.79) < 0.02
    assert abs(era["8192"]["implied_height_m"] - 2.35) < 0.02
    assert era["missing"]["implied_height_m"] - era["8192"]["implied_height_m"] > 0.4
    # the era truth's 8192 subpopulation already implied the shipped constant
    assert abs(era["8192"]["implied_height_m"] - shipped) < 0.02
    assert abs(modern["8192"]["implied_height_m"] - shipped) < 0.02
    # §7's table is every group the run kept, so its rows must account for the pooled n:
    # a resolution silently dropped from the report would argue the heterogeneity by omission
    pooled_n = summary["truth_scale_by_resolution"]["era_truth"]["pooled"]["n"]
    assert sum(g["n"] for g in era.values()) == pooled_n
    assert set(era) == {"missing", "1664", "6656", "8192"}


# ----------------------------------------------------------------- what does transfer

def test_the_crossover_against_the_shipped_form_sits_at_fifteen_metres(summary):
    """Below 15 m — 81% of rows — the shipped form leads; above it the booster does."""
    near, far = 0, 0
    for row in summary["by_distance"]:
        lo = float(row["bin_m"].strip("[)").split(",")[0])
        flat = row["per_model"]["D_flat"]["median_abs_m"]
        gbm = row["per_model"]["gbm_l1_scaled"]["median_abs_m"]
        if lo < 15.0:
            assert flat < gbm, row["bin_m"]
            near += row["n"]
        else:
            assert gbm < flat, row["bin_m"]
            far += row["n"]
    assert 0.15 < far / (near + far) < 0.25


def test_the_far_field_lead_is_mostly_positive_bias_not_structure(summary):
    """The caveat that removes most of the far-field story, locked so it cannot be dropped.

    The era blend is biased +1.07 m long and carries no conditional structure worth the
    name, yet it beats the booster in two of the three bins above 15 m — because where
    every bounded model undershoots, a positive bias reads as skill. The booster leads
    outright only at 30-50 m, on 54 rows.
    """
    outright = []
    for row in summary["by_distance"]:
        lo = float(row["bin_m"].strip("[)").split(",")[0])
        if lo < 15.0:
            continue
        pm = row["per_model"]
        assert pm["D_blend"]["median_abs_m"] < pm["D_flat"]["median_abs_m"], row["bin_m"]
        if pm["gbm_l1_scaled"]["median_abs_m"] < pm["D_blend"]["median_abs_m"]:
            outright.append((row["bin_m"], row["n"]))
    assert [b for b, _ in outright] == ["[30.0, 50.0)"]
    assert sum(n for _, n in outright) < 100
    assert summary["pooled"]["models"]["D_blend"]["signed_median_m"] > 1.0


def test_the_booster_owns_the_tail_everywhere(summary):
    """p90 is the metric that does not invert: the booster's tail is better on both the
    seen-type and unseen-type cuts, and pooled. Reported so the median headline cannot be
    read as 'the closed form is better full stop'."""
    held = summary["held_out_half"]["models"]
    assert held["gbm_l1_scaled"]["p90_abs_m"] < held["D_flat"]["p90_abs_m"]
    assert held["gbm_modern"]["p90_abs_m"] < 0.6 * held["D_flat"]["p90_abs_m"]
    for cut in summary["by_label_type_seen"].values():
        assert cut["gbm_l1_scaled"]["p90_abs_m"] < cut["D_flat"]["p90_abs_m"]
