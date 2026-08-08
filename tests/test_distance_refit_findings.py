"""The issue #3 (Stages 1+2) findings, locked.

Convention (mirrors test_pov_inversion_findings.py): the fast tests assert what the
2026-08-07 run measured, reading the committed data/distance-refit-summary.json only. The
summary regenerates offline and deterministically with
`python python/run_distance_refit.py --write`, and one session-scoped test below re-derives
the headline numbers in-process so the committed JSON cannot drift from the code.
The invariants that must hold for *any* refit — solver exactness, boundedness, monotonicity,
the fallback path — live next door in test_distance_refit_contract.py.

Headline findings (reports/2026-08-07-distance-refit.md):

- The zero-parameter anchor (2.6 m / tan of the exact depression angle) beats the
  12-parameter status quo outright: 0.99 m vs 1.46 m median lat/lng error.
- The chosen candidate — per-label-type camera heights on a C1 cotangent/linear blend, fit
  by L1 in disparity space, selected on TRAIN loss — reaches 0.93 m median (-36%) with 8
  parameters, all of them physical.
- sv_image_y is stored in a fixed 13312x6656 frame in every modern city, so #4765's
  resolution defect lives in the apply path, and the one-line normalization *as written*
  would make the dominant 8192-px GSV population ~1.7 m worse: production's raw-pixel
  apply path currently survives on two errors cancelling.
- Near the horizon every saturating form stays bounded by construction where the raw
  cotangent runs to the 50 m cap; at realistic click noise the geometry rungs degrade like
  the linear status quo (the 2016 objection, dissolved).
"""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "data", "distance-refit-summary.json")
POV_SUMMARY_PATH = os.path.join(ROOT, "data", "pov-inversion-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="distance-refit summary not built yet"
)

CHOSEN = "D_blend_type_l1"


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ continuity & harness

def test_est7_continuity_row(summary):
    """The unmodified 2021 pipeline still scores its published 1.46 m, and switching it to
    the production scoring convention (era_cal heading + spherical destination) is worth
    centimeters, not meters."""
    c = summary["continuity"]
    assert c["est7_legacy_median_m"] == pytest.approx(1.4621, abs=0.005)
    assert c["est7_spherical_median_m"] == pytest.approx(1.4438, abs=0.005)
    assert -0.05 < c["scoring_convention_delta_m"] < 0.0


def test_era_cal_delta_matches_pov_summary(summary):
    """The one heading constant is re-fit inside this run and must equal #5's to float
    precision — same recipe, same rows."""
    with open(POV_SUMMARY_PATH, encoding="utf-8") as f:
        pov = json.load(f)
    assert summary["meta"]["era_cal_delta_deg"] == pytest.approx(
        pov["era_cal_delta_deg"], abs=1e-9)


def test_every_rung_scored_on_the_full_split(summary):
    for key, row in summary["matrix"].items():
        if key == "anchor_served":
            assert row["n"] == 56  # its served-height subsample, reported honestly
        else:
            assert row["n"] == 79029, key


# ------------------------------------------------------------------ the frame (#4765)

def test_sv_image_y_is_fixed_frame_in_every_city(summary):
    """If sv_image_y scaled with the panorama raster, the 8192-px implied/exact ratio
    would sit 1.23x above the 6656-px one. Measured: equal to within 1.5% everywhere."""
    for city, row in summary["fixed_frame_check"]["cities"].items():
        for h in ("6656", "8192"):
            if h in row:
                assert 0.94 < row[h]["ratio_median"] < 1.01, (city, h)
        if "ratio_8192_over_6656" in row:
            assert 0.97 < row["ratio_8192_over_6656"] < 1.02, city
    pooled = summary["fixed_frame_check"]["pooled"]
    assert pooled["ratio_8192_over_6656"] == pytest.approx(0.9984, abs=0.01)
    assert pooled["if_real_pixel_frame"] == pytest.approx(8192 / 6656)


def test_apply_path_normalization_backfires(summary):
    """The #4765 one-liner as written: normalizing the deployed pixel input WITHOUT
    refitting the coefficients surfaces the fit's own +1.7 m too-far bias on 8192-px panos
    that the raw pixel overshoot currently cancels."""
    ap = summary["candidate_b"]["apply_path"]
    assert ap["raw"]["dist_median_m"] == pytest.approx(1.0693, abs=0.005)
    assert ap["normalized"]["dist_median_m"] == pytest.approx(1.8752, abs=0.005)
    assert ap["raw"]["h8192"]["signed_median_m"] == pytest.approx(-0.391, abs=0.005)
    assert ap["normalized"]["h8192"]["signed_median_m"] == pytest.approx(1.699, abs=0.005)
    # the 6656 group is the identity case: both variants must agree exactly
    assert ap["raw"]["h6656"]["signed_median_m"] == ap["normalized"]["h6656"]["signed_median_m"]


def test_in_frame_height_term_rejects_normalization(summary):
    """The sharp version: a normalized predictor requires the sv x (6656/height - 1)
    interaction to equal the sv slope itself. Measured, it has the OPPOSITE sign and sits
    20-70 standard errors below that value at every zoom (and B_norm scores worse than the
    plain fit on the same subset, under both losses)."""
    forms = summary["candidate_b"]["fixed_frame_forms"]
    for z in (1, 2, 3):
        iv = forms[f"zoom{z}"]["interact_vs_norm_prediction"]
        required = iv["sv_slope_it_would_have_to_match"]
        assert required > 0.01, z
        assert iv["interact_coef"] < 0, z  # normalization needs it strongly positive
        assert (required - iv["interact_coef"]) / iv["interact_se"] > 10, z
        for loss in ("ols", "l1"):
            a = forms[f"zoom{z}"]["A_sub"]["test_dist_median_m"][loss]
            b = forms[f"zoom{z}"]["B_norm"]["test_dist_median_m"][loss]
            assert b > a, (z, loss)


def test_candidate_b_is_measured_on_the_two_gsv_heights_only(summary):
    """The third rig (294 rows at 1664 px) carries a 4x normalization factor, i.e. 16x the
    leverage of an 8192-px row on the interaction term. It is excluded, like it is in the
    fixed-frame and apply-path checks, and the count it would have contributed is reported."""
    forms = summary["candidate_b"]["fixed_frame_forms"]
    assert forms["n_height_1664_excluded"] == 294
    assert forms["n_train"] + forms["n_test"] == 162846
    assert summary["candidate_b"]["apply_path"]["n"] == forms["n_test"]


# ------------------------------------------------------------------ the ladder

def test_zero_param_anchor_beats_the_fitted_status_quo(summary):
    """Pure geometry with the ecosystem's 2.6 m camera height and no fitted parameters
    beats all twelve fitted coefficients — the 2016 comparison, rerun on 79k test rows."""
    m = summary["matrix"]
    assert m["anchor"]["dist_median_m"] == pytest.approx(0.9394, abs=0.005)
    assert m["anchor"]["latlng_median_m"] == pytest.approx(0.9910, abs=0.005)
    assert m["anchor"]["dist_median_m"] < m["est7"]["dist_median_m"] - 0.4
    assert m["anchor"]["latlng_median_m"] < m["est7"]["latlng_median_m"] - 0.4


def test_fitted_heights_are_physical(summary):
    """One parameter, and it lands on the camera: C's fitted height sits between GSV's
    served heights (median 2.37 m) and the ecosystem constant (2.6 m); per-type heights
    order the way ground contact does (curb ramps at grade > surface problems)."""
    p = summary["params"]
    assert p["C_l1"]["height_m"] == pytest.approx(2.686, abs=0.01)
    assert p["C_ols"]["height_m"] == pytest.approx(2.565, abs=0.01)
    for key in ("C_type_l1", CHOSEN):
        hs = p[key]["height_by_type_m"]
        assert hs["CurbRamp"] > hs["SurfaceProblem"], key
        assert all(2.2 < v < 3.0 for v in hs.values()), key
    ch = summary["riders"]["camera_height"]
    assert ch["served_median_m_excl_pin"] == pytest.approx(2.366, abs=0.005)
    assert ch["n_served"] == 214


def test_every_per_type_fit_publishes_a_fallback_parameter(summary):
    """A modern caller will meet label types this 2017-2020 population never contained, so
    every per-type rung ships the pooled fit alongside the table. It must be a real pooled
    fit, which is to say inside the spread of the per-type values it stands in for."""
    typed = [k for k, p in summary["params"].items()
             if "height_by_type_m" in p or "c1_by_type" in p]
    assert len(typed) == 8  # C, D_floor, D_blend, D_soft x two losses
    for key in typed:
        p = summary["params"][key]
        table = p.get("height_by_type_m") or p["c1_by_type"]
        fallback = p.get("height_fallback_m", p.get("c1_fallback"))
        assert set(table) == {"CurbRamp", "NoCurbRamp", "NoSidewalk", "Obstacle",
                              "Occlusion", "Other", "SurfaceProblem"}, key
        assert min(table.values()) <= fallback <= max(table.values()), key


def test_chosen_candidate_and_headline(summary):
    """The recommendation was selected on train loss before test scoring (the honesty
    gate) and cuts the published median error by about a third."""
    chosen = summary["meta"]["chosen"]
    assert chosen["rung"] == CHOSEN
    assert chosen["chosen_on"].startswith("train")
    assert chosen["rung"] == min(chosen["train_median_abs_dist_error_m"],
                                 key=chosen["train_median_abs_dist_error_m"].get)
    m = summary["matrix"]
    assert m[CHOSEN]["latlng_median_m"] == pytest.approx(0.9335, abs=0.005)
    assert m[CHOSEN]["latlng_p90_m"] == pytest.approx(4.4755, abs=0.02)
    assert m[CHOSEN]["latlng_median_m"] < m["est7"]["latlng_median_m"] - 0.5
    assert m[CHOSEN]["latlng_p90_m"] < m["est7"]["latlng_p90_m"]
    assert m[CHOSEN]["n_params"] == 8


def test_parameter_counts_use_one_consistent_rule(summary):
    """Every coefficient counted, everywhere: est7 is 3 zooms x 3 distance + 3 x 2 heading =
    15, the status-quo distance half alone is 9, and the recommendation replaces them with
    7 camera heights plus a blend angle. (A '12' here would be distance slopes without their
    intercepts plus the full heading half — two rules in one column.)"""
    m = summary["matrix"]
    assert m["est7"]["n_params"] == m["est7_sph"]["n_params"] == 15
    assert m["A_ols"]["n_params"] == m["A_l1"]["n_params"] == 9
    assert m["anchor"]["n_params"] == 0
    assert m["C_l1"]["n_params"] == 1 and m["C_type_l1"]["n_params"] == 7
    assert m[CHOSEN]["n_params"] == 8 < m["A_ols"]["n_params"]


def test_l1_earns_its_place(summary):
    """Rider 1 (loss/metric alignment): on the median metric the L1 column beats OLS for
    the same functional form, ladder-wide."""
    m = summary["matrix"]
    for family in ("A", "C", "C_type", "D_floor", "D_blend", "D_blend_type"):
        assert (m[f"{family}_l1"]["latlng_median_m"]
                < m[f"{family}_ols"]["latlng_median_m"]), family


def test_isotonic_confirms_the_parametric_shape(summary):
    """Rung E exists to catch shape the closed forms miss. It doesn't find any: the free
    monotone fit lands within a few centimeters of the cotangent family."""
    m = summary["matrix"]
    for loss in ("ols", "l1"):
        assert abs(m[f"E_{loss}"]["dist_median_m"] - m[CHOSEN]["dist_median_m"]) < 0.08, loss


# ------------------------------------------------------------------ robustness (Stage 2)

def test_near_horizon_stays_bounded(summary):
    """The load-bearing D property: where the raw cotangent runs to the 50 m cap, the
    saturating forms answer in the 20s — and no rung's answer anywhere in either near-horizon
    bin may exceed the structural bound it publishes."""
    nh = {row["bin_deg"]: row for row in summary["near_horizon"]}
    assert [nh[b]["n"] for b in ("(-inf, 0.0]", "(0.0, 2.0]", "(2.0, 5.0]")] == [128, 300, 2299]
    bin02 = nh["(0.0, 2.0]"]["per_rung"]
    assert bin02["C_l1"]["dist_pred_max_m"] == 50.0
    assert bin02[CHOSEN]["dist_pred_max_m"] < 30.0
    assert bin02["E_l1"]["dist_pred_max_m"] < 25.0
    for row in summary["near_horizon"]:
        for key, v in row["per_rung"].items():
            bound = summary["bounds"].get(key) or summary["meta"]["dist_cap_m"]
            assert v["dist_pred_max_m"] <= bound + 1e-6, (row["bin_deg"], key)
    # and the saturating forms actually help there, vs the diverging cotangent
    assert bin02[CHOSEN]["latlng_median_m"] < 0.5 * bin02["C_l1"]["latlng_median_m"]


def test_the_recommendation_is_bounded_above_the_horizon_too(summary):
    """Regression test for the PR #12 review finding. The 128 test rows at or above the
    horizon are the degenerate case the D family exists for: before the tail was clamped at
    dep = 0, the chosen blend answered them with a linear runaway that reached the 50 m cap —
    exactly the behaviour it was recommended over. It now answers <= 28.4 m there, and that
    is a property of the form, not of these 128 rows (see the contract suite)."""
    above = {row["bin_deg"]: row for row in summary["near_horizon"]}["(-inf, 0.0]"]
    per_rung = above["per_rung"]
    assert above["n"] == 128
    assert per_rung[CHOSEN]["dist_pred_max_m"] == pytest.approx(
        summary["bounds"][CHOSEN], abs=1e-6)
    assert per_rung[CHOSEN]["dist_pred_max_m"] < 29.0
    # the unbounded rungs are still unbounded there, which is what makes this a real contrast
    for key in ("anchor", "C_l1", "D_soft_l1"):
        assert per_rung[key]["dist_pred_max_m"] == 50.0, key
    assert per_rung[CHOSEN]["latlng_median_m"] < 0.5 * per_rung["C_l1"]["latlng_median_m"]


def test_structural_bounds_are_published_for_every_rung(summary):
    """`bounds` is the largest answer each form can EVER return, swept over the whole
    depression domain — the number the report's '<= N m' claims mean. The per-bin
    dist_pred_max_m above is only what those rows happened to draw."""
    b = summary["bounds"]
    assert set(b) == set(summary["params"])
    assert b["A_ols"] is None and b["A_l1"] is None  # a pixel-domain form, not a dep one
    for key in ("anchor", "C_l1", "C_type_l1", "D_soft_l1", "D_soft_type_l1"):
        assert b[key] == pytest.approx(summary["meta"]["dist_cap_m"]), key
    assert b["D_floor_l1"] == pytest.approx(21.94, abs=0.05)
    assert b["D_floor_type_l1"] == pytest.approx(22.53, abs=0.05)
    assert b[CHOSEN] == pytest.approx(28.35, abs=0.05)
    assert b["E_l1"] == pytest.approx(24.62, abs=0.05)
    # the floor twin is the tighter bound; that is the trade it makes for its p90
    assert b["D_floor_type_l1"] < b[CHOSEN] < summary["meta"]["dist_cap_m"]


def test_the_soft_caps_bound_is_the_clip_not_saturation(summary):
    """D_soft's selling point was a bound at 1/c0 by construction. Measured: the c0 >= 1/cap
    constraint is active in all four variants, so 1/c0 IS the 50 m cap and the rung buys no
    saturation at all — one of the reasons floor/blend win."""
    for key in ("D_soft_ols", "D_soft_l1", "D_soft_type_ols", "D_soft_type_l1"):
        p = summary["params"][key]
        assert p["c0"] == pytest.approx(1.0 / summary["meta"]["dist_cap_m"], rel=1e-9), key
        assert summary["bounds"][key] == pytest.approx(summary["meta"]["dist_cap_m"]), key
        assert p["c1_floored"] is False, key  # no fitted slope ever needed the sign floor
    assert summary["params"]["D_soft_l1"]["projected"] is True
    assert summary["params"]["D_soft_ols"]["projected"] is False  # lsq_linear bounds instead


def test_click_noise_sensitivity_comparable_to_status_quo(summary):
    """The gsv-location-extraction-analysis objection, quantified: at 5 px of click noise
    every rung — geometric or fitted — loses under 5 cm of median accuracy, and at 10 px
    the chosen form stays within 1.5x of the status quo's degradation. No provenance
    gating needed."""
    ns = summary["noise_sweep"]["per_rung"]
    for key, row in ns.items():
        assert row["5.0"]["delta_median_m"] < 0.05, key
    assert ns[CHOSEN]["10.0"]["delta_median_m"] < 1.5 * ns["est7"]["10.0"]["delta_median_m"]


def test_zoom_collapses_into_the_projection(summary):
    """The exact depression angle consumed zoom; what survives per zoom is decimeters of
    behavioral residual, not the per-zoom refits' worth of structure."""
    for z, row in summary["zoom_residual_chosen"].items():
        assert abs(row["signed_median_m"]) < 0.25, z


def test_riders(summary):
    """photographer_pitch carries no distance signal, and the recorded two-component tilt
    explains essentially none of the depression residual on the 409 metadata panos."""
    r = summary["riders"]
    assert abs(r["photographer_pitch"]["pearson_r"]) < 0.05
    assert r["tilt_sinusoid"]["n"] == 791
    assert abs(r["tilt_sinusoid"]["pearson_r"]) < 0.1


def test_quantile_bands(summary):
    """The tau=0.1/0.9 disparity fits give a per-label interval nearly for free."""
    q = summary["quantiles"]
    assert q["interval_width_median_m"] == pytest.approx(1.70, abs=0.02)
    assert q["interval_width_p90_m"] == pytest.approx(3.48, abs=0.05)


def test_era_fit_coefficients_carry_the_conventions(summary):
    """Rider 2: the hand-off states its geodesy, keeps the era constant out of production,
    and — since the modern-truth check — points at the calibrated production constants
    rather than presenting its own scale as shippable."""
    pc = summary["era_fit_coefficients"]
    assert pc["rung"] == CHOSEN
    assert "spherical" in pc["geodesy"]
    assert "NOT" in pc["heading"]
    assert "modern-truth" in pc["status"] and "final_coefficients" in pc["status"]
    assert "pinned-plane" in pc["status"]  # names the scale artifact, not just a pointer
    assert len(pc["caveats"]) >= 5
    # the hand-off states what it can answer at worst and what to do with an unfamiliar type,
    # so neither has to be rediscovered by whoever ports this to JS
    assert pc["max_answer_m"] == pytest.approx(summary["bounds"][CHOSEN], abs=1e-9)
    assert "height_fallback_m" in pc["unseen_label_type"]
    assert "height_fallback_m" in pc["params"]


# ------------------------------------------------------------------ code <-> summary

def test_summary_reproduces_from_code(raw_data, summary):
    """Re-derive the headline subset in-process so the committed JSON cannot drift from
    distance_refit.py: the continuity row, the anchor row, the chosen rung's parameters
    and test metrics, and the heading constant. Uses the session raw_data fixture."""
    import sys

    sys.path.insert(0, os.path.join(ROOT, "python"))
    from label_latlng_estimation import (
        add_heading_diff, clean_data, fit_models, split_from_fixtures,
    )
    import distance_refit as dr

    cleaned, _ = clean_data(raw_data)
    cleaned = dr.add_depression(add_heading_diff(cleaned))
    train, test = split_from_fixtures(
        cleaned, os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    models = fit_models(train, include_est6=False)

    fits = {"anchor": dr.fit_anchor(), CHOSEN: dr.fit_blend(train, "l1", per_type=True)}
    for lt, h in summary["params"][CHOSEN]["height_by_type_m"].items():
        assert fits[CHOSEN]["height_by_type_m"][lt] == pytest.approx(h, rel=1e-9), lt
    assert fits[CHOSEN]["blend_deg"] == summary["params"][CHOSEN]["blend_deg"]
    assert fits[CHOSEN]["height_fallback_m"] == pytest.approx(
        summary["params"][CHOSEN]["height_fallback_m"], rel=1e-9)
    assert dr.structural_max_m(fits[CHOSEN]) == pytest.approx(
        summary["bounds"][CHOSEN], abs=1e-6)

    scored = dr.score_rungs(fits, models, train, test)
    assert scored.attrs["era_cal_delta_deg"] == pytest.approx(
        summary["meta"]["era_cal_delta_deg"], abs=1e-9)
    fresh = dr.matrix_table(scored, fits)
    for key in ("est7", "est7_sph", "anchor", CHOSEN):
        for metric in ("latlng_median_m", "latlng_p90_m", "dist_median_m", "dist_p90_m"):
            assert fresh[key][metric] == pytest.approx(
                summary["matrix"][key][metric], rel=1e-6), (key, metric)
    # and the committed near-horizon row for the chosen rung re-derives too, which is the
    # row the boundedness claim is read off
    fresh_nh = {r["bin_deg"]: r for r in dr.near_horizon_table(scored, keys=[CHOSEN])}
    for bin_deg, row in fresh_nh.items():
        want = {r["bin_deg"]: r for r in summary["near_horizon"]}[bin_deg]
        assert row["n"] == want["n"], bin_deg
        assert row["per_rung"][CHOSEN]["dist_pred_max_m"] == pytest.approx(
            want["per_rung"][CHOSEN]["dist_pred_max_m"], rel=1e-9), bin_deg
