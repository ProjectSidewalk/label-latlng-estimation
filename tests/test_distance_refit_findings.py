"""The issue #3 (Stages 1+2) findings, locked.

Convention (mirrors test_pov_inversion_findings.py): the fast tests assert what the
2026-08-07 run measured, reading the committed data/distance-refit-summary.json only. The
summary regenerates offline and deterministically with
`python python/run_distance_refit.py --write`, and one session-scoped test below re-derives
the headline numbers in-process so the committed JSON cannot drift from the code.

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
    interaction to equal the sv slope itself; the fitted coefficient is nowhere near it at
    any zoom (and B_norm scores worse than the plain fit on the same subset)."""
    forms = summary["candidate_b"]["fixed_frame_forms"]
    for z in (1, 2, 3):
        iv = forms[f"zoom{z}"]["interact_vs_norm_prediction"]
        assert iv["sv_slope_it_would_have_to_match"] > 0.01, z
        assert abs(iv["interact_coef"]) < 0.5 * iv["sv_slope_it_would_have_to_match"], z
        for loss in ("ols", "l1"):
            a = forms[f"zoom{z}"]["A_sub"]["test_dist_median_m"][loss]
            b = forms[f"zoom{z}"]["B_norm"]["test_dist_median_m"][loss]
            assert b > a, (z, loss)


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


def test_chosen_candidate_and_headline(summary):
    """The recommendation was selected on train loss before test scoring (the honesty
    gate) and cuts the published median error by about a third."""
    chosen = summary["meta"]["chosen"]
    assert chosen["rung"] == CHOSEN
    assert chosen["chosen_on"].startswith("train")
    m = summary["matrix"]
    assert m[CHOSEN]["latlng_median_m"] == pytest.approx(0.9336, abs=0.005)
    assert m[CHOSEN]["latlng_p90_m"] == pytest.approx(4.478, abs=0.02)
    assert m[CHOSEN]["latlng_median_m"] < m["est7"]["latlng_median_m"] - 0.5
    assert m[CHOSEN]["latlng_p90_m"] < m["est7"]["latlng_p90_m"]
    assert m[CHOSEN]["n_params"] == 8


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
    saturating forms answer in the 20s, and no rung can exceed the training-domain cap."""
    nh = {row["bin_deg"]: row for row in summary["near_horizon"]}
    assert [nh[b]["n"] for b in ("(-inf, 0.0]", "(0.0, 2.0]", "(2.0, 5.0]")] == [128, 300, 2299]
    bin02 = nh["(0.0, 2.0]"]["per_rung"]
    assert bin02["C_l1"]["dist_pred_max_m"] == 50.0
    assert bin02[CHOSEN]["dist_pred_max_m"] < 30.0
    assert bin02["E_l1"]["dist_pred_max_m"] < 25.0
    for row in summary["near_horizon"]:
        for key, v in row["per_rung"].items():
            assert v["dist_pred_max_m"] <= summary["meta"]["dist_cap_m"], (row["bin_deg"], key)
    # and the saturating forms actually help there, vs the diverging cotangent
    assert bin02[CHOSEN]["latlng_median_m"] < 0.5 * bin02["C_l1"]["latlng_median_m"]


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


def test_provisional_coefficients_carry_the_conventions(summary):
    """Rider 2: the hand-off states its geodesy, keeps the era constant out of production,
    and stays provisional until Stage 3 (Mapillary) runs."""
    pc = summary["provisional_coefficients"]
    assert pc["rung"] == CHOSEN
    assert "spherical" in pc["geodesy"]
    assert "NOT" in pc["heading"]
    assert "Stage 3" in pc["status"]
    assert len(pc["caveats"]) >= 5


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

    scored = dr.score_rungs(fits, models, train, test)
    assert scored.attrs["era_cal_delta_deg"] == pytest.approx(
        summary["meta"]["era_cal_delta_deg"], abs=1e-9)
    fresh = dr.matrix_table(scored, fits)
    for key in ("est7", "est7_sph", "anchor", CHOSEN):
        for metric in ("latlng_median_m", "latlng_p90_m", "dist_median_m", "dist_p90_m"):
            assert fresh[key][metric] == pytest.approx(
                summary["matrix"][key][metric], rel=1e-6), (key, metric)
