"""Layer 2 of the #3 distance-refit suite: contract tests on the code, not on one run.

`test_distance_refit_findings.py` asserts what the 2026-08-07 run *measured* — numbers that
move if anything is refit. This file asserts what has to be true of **any** run: the exactness
of the two closed-form solvers, the invariants every fitted rung's prediction must satisfy
(boundedness, monotonicity, no silent NaN), the recovery behaviour of each fit on data whose
answer is known by construction, and the harness properties the report's comparisons rest on
(one row per test label, the heading half held identical, selection on train alone).

Almost everything here is synthetic, so it stays true under a refit and runs in milliseconds.
The handful of tests that need the real split share one module-scoped pipeline fixture.

Why this file exists: the report's load-bearing claim about the D family is *structural* — the
largest answer a form can return anywhere, not the largest it happened to return on the thin
near-horizon slice of one test split. Value-locks against a committed JSON cannot see the
difference. `test_structural_max_*` and `test_blend_tail_is_clamped_above_the_horizon` can.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import distance_refit as dr  # noqa: E402
from label_latlng_estimation import (  # noqa: E402
    EARTH_RADIUS_M, add_heading_diff, clean_data, dest_point, error_stats, fit_models,
    haversine_m, spherical_dest, split_from_fixtures,
)

FIXTURES = os.path.join(ROOT, "tests", "fixtures", "r-baseline")
UNSEEN = "Crosswalk"  # a real modern Project Sidewalk type absent from the 2017-2020 data


# --------------------------------------------------------------------------- synthetic fits

def _typed(value):
    return {lt: value for lt in dr.LABEL_TYPES}


@pytest.fixture(scope="module")
def forms():
    """One hand-built params dict per prediction form, with values in the fitted range.

    Per-type variants deliberately give each type a *different* parameter, so a test that
    silently ignored the type column would fail rather than pass by symmetry.
    """
    spread = {lt: 2.4 + 0.05 * i for i, lt in enumerate(dr.LABEL_TYPES)}
    return {
        "cotangent": {"form": "cotangent", "loss": "l1", "height_m": 2.6, "n_params": 1},
        "cotangent_type": {"form": "cotangent", "loss": "l1", "height_by_type_m": spread,
                           "height_fallback_m": 2.6, "n_params": 7},
        "floor": {"form": "floor", "loss": "l1", "height_m": 2.6, "dep_min_deg": 7.0,
                  "n_params": 2},
        "floor_type": {"form": "floor", "loss": "l1", "height_by_type_m": spread,
                       "height_fallback_m": 2.6, "dep_min_deg": 7.0, "n_params": 8},
        "blend": {"form": "blend", "loss": "l1", "height_m": 2.6, "blend_deg": 11.25,
                  "n_params": 2},
        "blend_type": {"form": "blend", "loss": "l1", "height_by_type_m": spread,
                       "height_fallback_m": 2.6, "blend_deg": 11.25, "n_params": 8},
        "softcap": {"form": "softcap", "loss": "l1", "c0": 0.03, "c1": 1.1, "n_params": 2},
        "softcap_type": {"form": "softcap", "loss": "l1", "c0": 0.03,
                         "c1_by_type": {lt: 1.0 + 0.02 * i
                                        for i, lt in enumerate(dr.LABEL_TYPES)},
                         "c1_fallback": 1.1, "n_params": 8},
        "isotonic": {"form": "isotonic", "loss": "l1", "n_params": 4,
                     "knots_dep_deg": [0.5, 3.0, 12.0, 40.0],
                     "knots_dist_m": [24.0, 12.0, 4.0, 1.0]},
    }


def _sweep(params, dep=None, label_type="CurbRamp"):
    dep = np.linspace(-90.0, 90.0, 3601) if dep is None else np.asarray(dep, float)
    return dr.predict_dist(params, pd.DataFrame(
        {"depression_deg": dep, "label_type": label_type, "zoom": 1}))


# --------------------------------------------------------------------- the two exact solvers

@pytest.mark.parametrize("seed", range(5))
def test_weighted_median_minimizes_weighted_absolute_deviation(seed):
    """_weighted_median must return an exact minimizer of sum w*|v - m|, not an approximation
    (the one-parameter camera-height fits are built on it)."""
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 60)
    v, w = rng.normal(size=n), rng.uniform(0.01, 5, size=n)
    got = dr._weighted_median(v, w)
    loss = lambda m: float(np.sum(w * np.abs(v - m)))  # noqa: E731
    # the optimum of a weighted L1 problem is attained at one of the data points
    assert loss(got) <= min(loss(x) for x in v) + 1e-9


def test_weighted_median_matches_the_plain_median_for_equal_weights():
    for n in (1, 2, 3, 8, 9):
        v = np.arange(n, dtype=float)
        got = dr._weighted_median(v, np.ones(n))
        # lower median convention: for even n either central point is optimal
        assert abs(got - np.median(v)) <= 0.5 + 1e-12
        assert np.sum(np.abs(v - got)) <= np.sum(np.abs(v - np.median(v))) + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_lad_origin_slope_is_the_exact_l1_optimum(seed):
    """The optimum of sum|y - s*x| is attained at one of the y_i/x_i ratios, so the exact
    optimum is computable by enumeration — the closed form must match it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 4.0, 300)
    y = 2.5 * x + rng.standard_t(2, 300) * 0.5  # heavy tails: where L1 earns its place
    got = dr._lad_origin_slope(x, y)
    loss = lambda s: float(np.sum(np.abs(y - s * x)))  # noqa: E731
    assert loss(got) <= min(loss(c) for c in (y / x)) + 1e-9


def test_lad_origin_slope_ignores_rows_with_zero_x():
    """tan(max(dep, 0)) is exactly 0 for every above-horizon click. Those rows add a constant
    to the objective, so they must not move the answer."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 3, 200)
    y = 1.7 * x + rng.normal(0, 0.1, 200)
    base = dr._lad_origin_slope(x, y)
    padded = dr._lad_origin_slope(np.r_[x, np.zeros(50)], np.r_[y, rng.normal(0, 9, 50)])
    assert padded == pytest.approx(base, rel=1e-12)


def test_lad_origin_slope_weights_by_absolute_x():
    """Negative-x rows are weighted by |x|, not dropped. No call site produces them today;
    the solver is still required to be right rather than lucky."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 3, 150) * rng.choice([-1.0, 1.0], 150)
    y = -0.8 * x + rng.standard_t(3, 150) * 0.2
    got = dr._lad_origin_slope(x, y)
    loss = lambda s: float(np.sum(np.abs(y - s * x)))  # noqa: E731
    assert loss(got) <= min(loss(c) for c in (y / x)) + 1e-9
    # and the sign is recovered, which a `x > 0` filter would not guarantee
    assert got < 0


def test_disparity_floors_sub_meter_rows_and_counts_them():
    d = np.array([0.25, 0.5, 1.0, 2.0, 10.0])
    disp, n_capped = dr._disparity(d)
    assert n_capped == 2
    assert disp == pytest.approx([1.0, 1.0, 1.0, 0.5, 0.1])
    assert (disp <= 1.0).all()


def test_tan_dep_applies_its_floor_and_never_goes_negative():
    t = dr._tan_dep(np.array([-30.0, -1e-9, 0.0, 5.0, 45.0]), floor_deg=2.0)
    assert (t >= np.tan(np.radians(2.0)) - 1e-15).all()
    assert dr._tan_dep(np.array([-5.0, 10.0]))[0] == 0.0
    assert dr._tan_dep(np.array([45.0]))[0] == pytest.approx(1.0)


# ------------------------------------------------------------------- prediction invariants

def test_every_form_is_bounded_and_finite_over_the_whole_domain(forms):
    """No rung may return NaN, a negative distance, or more than the training-domain cap —
    anywhere in the depression domain, for any label type including one never fitted."""
    for name, params in forms.items():
        for lt in dr.LABEL_TYPES + [UNSEEN]:
            pred = _sweep(params, label_type=lt)
            assert np.isfinite(pred).all(), (name, lt)
            assert (pred >= 0.0).all(), (name, lt)
            assert (pred <= dr.DIST_CAP_M + 1e-9).all(), (name, lt)


def test_predictions_are_monotone_non_increasing_in_depression(forms):
    """Farther below the horizon must never mean farther away: the shape constraint the whole
    geometry argument rests on."""
    dep = np.linspace(-40.0, 60.0, 20001)
    for name, params in forms.items():
        pred = _sweep(params, dep=dep)
        assert (np.diff(pred) <= 1e-9).all(), name


def test_structural_max_matches_an_independent_dense_sweep(forms):
    for name, params in forms.items():
        swept = max(float(_sweep(params, label_type=lt).max())
                    for lt in dr.LABEL_TYPES + [UNSEEN])
        assert dr.structural_max_m(params) == pytest.approx(swept, abs=1e-6), name


def test_structural_max_matches_each_form_closed_form_bound(forms):
    """The bound is not an empirical maximum — every form has an analytic one, and this is
    where the report's '<= N m' numbers come from."""
    hs = forms["floor_type"]["height_by_type_m"]
    assert dr.structural_max_m(forms["floor"]) == pytest.approx(
        2.6 / np.tan(np.radians(7.0)), rel=1e-6)
    assert dr.structural_max_m(forms["floor_type"]) == pytest.approx(
        max(hs.values()) / np.tan(np.radians(7.0)), rel=1e-6)
    assert dr.structural_max_m(forms["softcap"]) == pytest.approx(1.0 / 0.03, rel=1e-6)
    assert dr.structural_max_m(forms["isotonic"]) == pytest.approx(24.0, rel=1e-9)
    # the blend's bound is its value at the horizon: h/tan(a) + |slope| * a
    a = np.radians(11.25)
    want = 2.6 / np.tan(a) + 2.6 * (np.pi / 180.0) / np.sin(a) ** 2 * 11.25
    assert dr.structural_max_m(forms["blend"]) == pytest.approx(want, rel=1e-6)
    # a raw cotangent has no bound below the cap, which is exactly why the D family exists
    assert dr.structural_max_m(forms["cotangent"]) == pytest.approx(dr.DIST_CAP_M)


def test_structural_max_is_none_for_the_pixel_domain_rung():
    """est7's form answers from pixels, not from an angle; a depression sweep cannot bound it,
    and structural_max_m must say so rather than invent a number."""
    a_fit = {"form": "per_zoom_linear", "loss": "ols", "n_params": 9,
             "coef": [{"(Intercept)": 18.6, "sv_image_y": 0.0139, "canvas_y": 0.0011}] * 3}
    assert dr.structural_max_m(a_fit) is None


def test_blend_tail_is_clamped_above_the_horizon(forms):
    """Regression test for the PR-#12 review finding: before the clamp, the blend's linear
    tail kept growing for dep < 0 and reached the 50 m cap by about -17 deg, so the 'answers
    22-28 m by construction' claim was false on exactly the unplaceable clicks it named."""
    for key in ("blend", "blend_type"):
        params = forms[key]
        worst = 0.0
        for lt in dr.LABEL_TYPES + [UNSEEN]:
            above = _sweep(params, dep=np.linspace(-90.0, 0.0, 5001), label_type=lt)
            assert above.max() - above.min() == pytest.approx(0.0, abs=1e-9), (key, lt)
            assert above[0] == pytest.approx(
                _sweep(params, dep=[0.0], label_type=lt)[0], abs=1e-12), (key, lt)
            worst = max(worst, float(above.max()))
        assert worst < 30.0, key
        assert dr.structural_max_m(params) == pytest.approx(worst, abs=1e-6), key


def test_blend_is_c1_at_the_blend_angle(forms):
    """The 'C1' in the report's name for this form: value and first derivative both match the
    cotangent at the blend angle, so there is no kink for click noise to amplify."""
    for key in ("blend", "blend_type"):
        a = forms[key]["blend_deg"]
        step = 1e-6
        left, mid, right = (_sweep(forms[key], dep=[a - step])[0],
                            _sweep(forms[key], dep=[a])[0],
                            _sweep(forms[key], dep=[a + step])[0])
        hh = forms[key].get("height_m") or forms[key]["height_by_type_m"]["CurbRamp"]
        want = -hh * (np.pi / 180.0) / np.sin(np.radians(a)) ** 2  # d/d(deg) of h*cot at a
        # C0: the two branches agree at a, so the gap over a step is first order in the step
        assert abs(left - mid) < abs(want) * step * 1.01, key
        assert abs(right - mid) < abs(want) * step * 1.01, key
        # C1: one-sided derivatives agree with each other and with the analytic slope
        d_left, d_right = (mid - left) / step, (right - mid) / step
        assert d_left == pytest.approx(d_right, rel=1e-4), key
        assert d_right == pytest.approx(want, rel=1e-4), key


def test_blend_equals_the_raw_cotangent_above_the_blend_angle(forms):
    dep = np.linspace(11.25, 60.0, 501)
    cot = 2.6 / np.tan(np.radians(dep))
    assert _sweep(forms["blend"], dep=dep) == pytest.approx(np.clip(cot, 0, dr.DIST_CAP_M))


def test_unseen_label_type_falls_back_to_the_pooled_parameter(forms):
    """A production caller WILL meet a label type the 2017-2020 population never contained.
    It must get the pooled answer, never NaN (a NaN here becomes a label placed nowhere)."""
    dep = np.linspace(-20.0, 45.0, 501)
    for key in ("cotangent_type", "floor_type", "blend_type", "softcap_type"):
        params = forms[key]
        unseen = _sweep(params, dep=dep, label_type=UNSEEN)
        assert np.isfinite(unseen).all(), key
        pooled = dict(params)
        if "height_by_type_m" in params:  # what the fallback height alone would predict
            pooled = {k: v for k, v in params.items() if k != "height_by_type_m"}
            pooled["height_m"] = params["height_fallback_m"]
        else:
            pooled = {k: v for k, v in params.items() if k != "c1_by_type"}
            pooled["c1"] = params["c1_fallback"]
        assert unseen == pytest.approx(_sweep(pooled, dep=dep), rel=1e-12), key


def test_per_type_parameters_actually_reach_the_prediction(forms):
    """Guard against a per-type rung silently collapsing to one height (a `.map` on the wrong
    column would still return finite numbers)."""
    for key in ("cotangent_type", "floor_type", "blend_type", "softcap_type"):
        preds = {lt: _sweep(forms[key], dep=[20.0], label_type=lt)[0] for lt in dr.LABEL_TYPES}
        assert len(set(np.round(list(preds.values()), 9))) == len(dr.LABEL_TYPES), key


def test_cotangent_served_is_nan_only_off_its_subsample():
    params = {"form": "cotangent_served", "loss": "none", "n_params": 0,
              "heights_by_pano": {"pano_a": 2.5, "pano_b": 2.2}, "n_panos": 2}
    df = pd.DataFrame({"depression_deg": [10.0, 10.0, 10.0],
                       "pano_id": ["pano_a", "pano_b", "pano_missing"]})
    pred = dr.predict_dist(params, df)
    assert np.isfinite(pred[:2]).all()
    assert np.isnan(pred[2])
    assert pred[0] == pytest.approx(2.5 / np.tan(np.radians(10.0)))


def test_unknown_form_raises_rather_than_guessing(forms):
    with pytest.raises(ValueError, match="unknown form"):
        dr.predict_dist({"form": "quadratic"}, pd.DataFrame({"depression_deg": [5.0]}))


# ------------------------------------------------------------------------ fits, on known data

def _synthetic(n=4000, height=2.55, seed=7, noise=0.0, heights_by_type=None):
    """Clicks whose true distance IS the cotangent, so every fit has a known right answer.

    The depression floor keeps every generated distance inside (1 m, 50 m): outside that band
    the harness's own disparity floor and distance cap would clip the truth, and a fit cannot
    be asked to recover a height from data the harness has deliberately censored.
    """
    rng = np.random.default_rng(seed)
    h_max = max(heights_by_type.values()) if heights_by_type else height
    dep = rng.uniform(np.degrees(np.arctan(h_max / 45.0)), 35.0, n)
    types = rng.choice(dr.LABEL_TYPES, n)
    h = (np.array([heights_by_type[t] for t in types]) if heights_by_type
         else np.full(n, height))
    dist = h / np.tan(np.radians(dep))
    if noise:
        dist = dist * (1.0 + rng.normal(0, noise, n))
    assert (dist > 1.0).all() and (dist < dr.DIST_CAP_M).all() or noise
    return pd.DataFrame({"depression_deg": dep, "pano_dist": np.clip(dist, 0.5, 50.0),
                         "label_type": types, "zoom": rng.choice([1, 2, 3], n)})


@pytest.mark.parametrize("loss", ["ols", "l1"])
def test_cotangent_recovers_the_generating_camera_height(loss):
    fit = dr.fit_cotangent(_synthetic(height=2.55), loss)
    assert fit["height_m"] == pytest.approx(2.55, rel=1e-9)
    assert fit["n_params"] == 1


@pytest.mark.parametrize("loss", ["ols", "l1"])
def test_per_type_cotangent_recovers_every_type_and_a_pooled_fallback(loss):
    hs = {lt: 2.3 + 0.07 * i for i, lt in enumerate(dr.LABEL_TYPES)}
    fit = dr.fit_cotangent(_synthetic(n=7000, heights_by_type=hs), loss, per_type=True)
    for lt, want in hs.items():
        assert fit["height_by_type_m"][lt] == pytest.approx(want, rel=1e-9), lt
    # the pooled fallback must be a real pooled fit, i.e. inside the per-type spread
    assert min(hs.values()) <= fit["height_fallback_m"] <= max(hs.values())


def test_l1_beats_ols_on_contaminated_heights():
    """Rider 1's premise, as a property rather than a measured number: with clustered
    outliers (occlusion labels, item G's rotated-column rows), the L1 height is the closer
    one. If this ever flips, the L1 column has stopped earning its place."""
    df = _synthetic(n=6000, height=2.55, noise=0.05)
    contaminated = df.copy()
    idx = df.index[:600]
    contaminated.loc[idx, "pano_dist"] = contaminated.loc[idx, "pano_dist"] * 4.0
    h_ols = dr.fit_cotangent(contaminated, "ols")["height_m"]
    h_l1 = dr.fit_cotangent(contaminated, "l1")["height_m"]
    assert abs(h_l1 - 2.55) < abs(h_ols - 2.55)


@pytest.mark.parametrize("loss", ["ols", "l1"])
def test_softcap_always_respects_its_two_constraints(loss):
    fit = dr.fit_softcap(_synthetic(noise=0.1), loss)
    assert fit["c0"] >= 1.0 / dr.DIST_CAP_M - 1e-12
    assert fit["c1"] >= 0.0
    assert dr.structural_max_m(fit) <= dr.DIST_CAP_M + 1e-9


@pytest.mark.parametrize("loss", ["ols", "l1"])
def test_softcap_floors_a_slope_that_would_invert_the_form(loss):
    """Feed it data whose disparity DECREASES with tan(dep) — the fit wants c1 < 0, which
    would answer farther clicks as nearer. Both losses must refuse."""
    rng = np.random.default_rng(3)
    dep = rng.uniform(2.0, 30.0, 3000)
    inverted = 3.0 + 0.6 * np.tan(np.radians(dep))  # distance rising with depression
    df = pd.DataFrame({"depression_deg": dep, "pano_dist": np.clip(inverted, 0.5, 50.0),
                       "label_type": rng.choice(dr.LABEL_TYPES, 3000)})
    fit = dr.fit_softcap(df, loss)
    assert fit["c1"] >= 0.0
    assert (_sweep(fit, dep=np.linspace(0, 40, 500)).max() <= dr.DIST_CAP_M + 1e-9)
    if loss == "l1":
        assert fit["c1_floored"] is True


@pytest.mark.parametrize("loss", ["ols", "l1"])
def test_floor_and_blend_stay_inside_their_profiled_grids(loss):
    """The profiled hyper-parameter must come back on its grid, and the resulting form must
    still respect the cap. (Whether the optimum is *interior* is a property of the real data,
    not of any data — synthetic pure-cotangent clicks have no degenerate region to saturate,
    so that claim is asserted on the real split in test_chosen_family_saturates_interior.)"""
    df = _synthetic(n=4000, noise=0.08)
    floor, blend = dr.fit_floor(df, loss), dr.fit_blend(df, loss)
    assert 0.5 <= floor["dep_min_deg"] <= 12.0
    assert 1.0 <= blend["blend_deg"] <= 12.0
    assert dr.structural_max_m(floor) <= dr.DIST_CAP_M + 1e-9
    assert dr.structural_max_m(blend) <= dr.DIST_CAP_M + 1e-9


@pytest.mark.parametrize("loss", ["ols", "l1"])
def test_per_type_floor_and_blend_carry_a_fallback_inside_their_spread(loss):
    hs = {lt: 2.3 + 0.07 * i for i, lt in enumerate(dr.LABEL_TYPES)}
    df = _synthetic(n=7000, heights_by_type=hs, noise=0.05)
    for fit in (dr.fit_floor(df, loss, per_type=True), dr.fit_blend(df, loss, per_type=True)):
        got = fit["height_by_type_m"]
        assert set(got) == set(dr.LABEL_TYPES)
        assert min(got.values()) <= fit["height_fallback_m"] <= max(got.values())
        assert fit["n_params"] == 1 + len(dr.LABEL_TYPES)


@pytest.mark.parametrize("loss", ["ols", "l1"])
def test_isotonic_knots_are_monotone_and_bounded(loss):
    fit = dr.fit_isotonic(_synthetic(n=5000, noise=0.15), loss, max_knots=24)
    kx, ky = np.array(fit["knots_dep_deg"]), np.array(fit["knots_dist_m"])
    assert len(kx) <= 24 and len(kx) == len(ky) == fit["n_params"]
    assert (np.diff(kx) > 0).all()
    assert (np.diff(ky) <= 1e-12).all()
    assert dr.structural_max_m(fit) == pytest.approx(ky[0], rel=1e-9)


def test_linear_rung_reproduces_the_ordinary_least_squares_solution():
    df = _synthetic(n=3000, noise=0.1)
    rng = np.random.default_rng(5)
    df["sv_image_y"] = -df["depression_deg"] * dr.SV_PX_PER_DEG
    df["canvas_y"] = rng.uniform(0, 480, len(df))
    fit = dr.fit_linear(df, "ols")
    assert fit["n_params"] == 9
    for z in (1, 2, 3):
        sub = df[df["zoom"] == z]
        X = np.column_stack([np.ones(len(sub)), sub["sv_image_y"], sub["canvas_y"]])
        want = np.linalg.lstsq(X, sub["pano_dist"].to_numpy(float), rcond=None)[0]
        got = fit["coef"][z - 1]
        assert [got["(Intercept)"], got["sv_image_y"], got["canvas_y"]] == pytest.approx(
            want, rel=1e-8)


def test_city_dummies_share_columns_when_a_city_is_missing_from_one_side():
    """The fit/predict designs must agree even if a subgroup is absent from one frame —
    derived per frame, the reference category would silently change meaning."""
    train = pd.DataFrame({"city": ["seattle"] * 3 + ["newberg"] * 3 + ["cdmx"] * 3})
    test = pd.DataFrame({"city": ["seattle", "cdmx"]})  # newberg missing entirely
    cities = [c for c in dr.MODERN_CITIES if (train["city"] == c).any()]
    fe_tr, names = dr._city_dummies(train, cities)
    fe_te, _ = dr._city_dummies(test, cities)
    assert fe_tr.shape[1] == fe_te.shape[1] == len(names) == 2
    assert names == ["city_newberg", "city_cdmx"]
    assert fe_te.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    # and the per-frame default would NOT have matched, which is why cities is passed
    assert dr._city_dummies(test)[0].shape[1] == 1


# --------------------------------------------------------- core geodesy moved by this PR

def test_spherical_dest_round_trips_through_haversine():
    rng = np.random.default_rng(11)
    lat, lng = rng.uniform(-60, 60, 200), rng.uniform(-180, 180, 200)
    brng, dist = rng.uniform(0, 360, 200), rng.uniform(0.5, 50, 200)
    lng2, lat2 = spherical_dest(lng, lat, brng, dist)
    assert haversine_m(lng, lat, lng2, lat2) == pytest.approx(dist, rel=1e-9, abs=1e-8)


def test_spherical_dest_matches_the_textbook_formula_on_cardinal_bearings():
    lng2, lat2 = spherical_dest([0.0, 0.0], [0.0, 0.0], [0.0, 90.0], [1000.0, 1000.0])
    step = np.degrees(1000.0 / EARTH_RADIUS_M)
    assert lat2[0] == pytest.approx(step, rel=1e-9)   # due north
    assert lng2[0] == pytest.approx(0.0, abs=1e-12)
    assert lng2[1] == pytest.approx(step, rel=1e-9)   # due east at the equator
    assert lat2[1] == pytest.approx(0.0, abs=1e-12)


def test_spherical_dest_differs_from_the_ellipsoidal_destination_by_centimeters():
    """The convention the refit picked deliberately (rider 2). Both are 'right'; production
    runs the spherical one, and the gap is the -1.8 cm the report quotes."""
    lng, lat = np.array([-122.31, -77.03]), np.array([47.61, 38.90])
    brng, dist = np.array([37.0, 210.0]), np.array([12.0, 30.0])
    s_lng, s_lat = spherical_dest(lng, lat, brng, dist)
    e_lng, e_lat = dest_point(lng, lat, brng, dist)
    gap = haversine_m(s_lng, s_lat, e_lng, e_lat)
    assert (gap < 0.25).all() and (gap > 0.0).all()


def test_error_stats_reports_the_p90_it_claims():
    base = np.arange(101, dtype=float)
    err = pd.DataFrame({"error_est7": base, "error_est1": base * 2,
                        "heading_error_est7": base, "heading_error_est1": base,
                        "dist_error_est7": base, "dist_error_est1": base})
    rows = {r["estimate"]: r for r in error_stats(err)["summary"]}
    assert rows["error_est7"]["p90"] == pytest.approx(np.quantile(np.arange(101.0), 0.9))
    assert rows["error_est1"]["p90"] == pytest.approx(2 * rows["error_est7"]["p90"])
    assert rows["error_est7"]["median"] == pytest.approx(50.0)


# --------------------------------------------------------------- harness, on the real split

@pytest.fixture(scope="module")
def pipeline(raw_data):
    cleaned, _ = clean_data(raw_data)
    cleaned = dr.add_depression(add_heading_diff(cleaned))
    train, test = split_from_fixtures(cleaned, FIXTURES)
    return cleaned, train, test, fit_models(train, include_est6=False)


def test_depression_is_the_exact_projection_and_covers_both_sides_of_the_horizon(pipeline):
    cleaned, train, test, _ = pipeline
    dep = cleaned["depression_deg"]
    assert len(cleaned) == 395147
    assert dep.notna().all()
    assert (dep < 0).any() and (dep > 0).any()  # the degenerate region really is populated
    assert len(train) + len(test) == len(cleaned)


def test_a_ols_is_est7s_distance_half_to_float_precision(pipeline):
    """The ladder's reference rung must BE the status quo, or every delta below it is a
    comparison against something the 2021 analysis never published."""
    _, train, _, models = pipeline
    fit = dr.fit_linear(train, "ols")
    for z in (1, 2, 3):
        for term, got in fit["coef"][z - 1].items():
            assert got == pytest.approx(models["est7"]["dist"][z - 1][term], abs=1e-9), (z, term)


def test_scoring_keeps_one_row_per_test_label_with_no_alignment_holes(pipeline):
    """score_rungs assembles a fresh RangeIndex frame from Series carrying the test index;
    an alignment slip would show up as NaN errors rather than as an exception."""
    _, train, test, models = pipeline
    fits = {"anchor": dr.fit_anchor(),
            "blend": dr.fit_blend(train, "l1", per_type=True)}
    scored = dr.score_rungs(fits, models, train, test)
    assert len(scored) == len(test)
    assert scored["label_id"].tolist() == test["label_id"].tolist()
    for key in ("est7", "est7_sph", "anchor", "blend"):
        assert scored[f"error_{key}"].notna().all(), key
        assert (scored[f"dist_pred_{key}"] <= dr.DIST_CAP_M + 1e-9).all(), key
    # est7's own row must equal the untouched legacy scorer, not a re-derivation of it
    from label_latlng_estimation import latlng_error_m, predict_dist_heading
    d7, h7 = predict_dist_heading(models, test, "est7")
    assert scored["error_est7"].to_numpy() == pytest.approx(
        latlng_error_m(test, d7, h7, crude=False))


def test_the_heading_half_is_identical_across_rungs(pipeline):
    """The ladder's central claim — that rung differences are the distance half alone —
    holds only if every non-est7 rung shares one heading prediction."""
    _, train, test, models = pipeline
    heading_pred, delta = dr.heading_for_scoring(train, test)
    assert len(heading_pred) == len(test)
    assert delta == pytest.approx(0.7198, abs=0.01)
    # two rungs whose distances differ by a constant factor must differ ONLY through distance
    fits = {"a": dr.fit_anchor(2.6), "b": dr.fit_anchor(5.2)}
    scored = dr.score_rungs(fits, models, train, test)
    near = (scored["dist_pred_b"] < dr.DIST_CAP_M - 1e-9).to_numpy()  # off the shared clip
    ratio = (scored["dist_pred_b"] / scored["dist_pred_a"]).to_numpy()[near]
    assert ratio == pytest.approx(2.0, rel=1e-9)
    assert near.sum() > 0.9 * len(scored)


def test_candidate_selection_reads_train_only(pipeline):
    """The honesty gate. Rebuild the choice from the recorded train table and confirm it is
    the argmin — and that scrambling the test split cannot move it."""
    _, train, test, _ = pipeline
    fits = {"D_a": dr.fit_floor(train, "l1"), "D_b": dr.fit_blend(train, "l1")}
    chosen = dr.choose_candidate(fits, train)
    table = chosen["train_median_abs_dist_error_m"]
    assert chosen["rung"] == min(table, key=table.get)
    assert set(table) == set(fits)
    assert "train" in chosen["chosen_on"]
    assert dr.choose_candidate(fits, train.iloc[::-1])["rung"] == chosen["rung"]


def test_noise_sweep_is_deterministic_and_degrades_with_sigma(pipeline):
    """Same seed, same numbers — the summary has to regenerate byte-identically — and heavy
    click noise must cost more than light click noise. (Light noise can *help* a hair by
    jittering a biased point estimate, so the sign at 2 px is not asserted.)"""
    _, train, test, models = pipeline
    small = test.iloc[:4000]
    fits = {"anchor": dr.fit_anchor(), "C_l1": dr.fit_cotangent(train, "l1")}
    kw = dict(keys=["anchor", "C_l1"], sigmas=(2.0, 20.0), n_draws=1)
    first = dr.noise_sweep(fits, models, train, small, **kw)
    second = dr.noise_sweep(fits, models, train, small, **kw)
    assert first == second
    assert dr.noise_sweep(fits, models, train, small, seed=999, **kw) != first
    for key in ("anchor", "C_l1"):
        rung = first["per_rung"][key]
        assert rung["2.0"]["delta_median_m"] < rung["20.0"]["delta_median_m"], key
        assert abs(rung["2.0"]["delta_median_m"]) < 0.2, key  # 2 px is a centimetre effect


def test_chosen_family_saturates_interior_on_the_real_split(pipeline):
    """On the real data — the claim the report actually makes — the D family's shape
    hyper-parameter is a genuine interior optimum, and the resulting form is bounded well
    below the 50 m cap. A grid-edge answer would mean the profile was truncated."""
    _, train, _, _ = pipeline
    floor = dr.fit_floor(train, "l1", per_type=True)
    blend = dr.fit_blend(train, "l1", per_type=True)
    assert 0.5 < floor["dep_min_deg"] < 12.0
    assert 1.0 < blend["blend_deg"] < 12.0
    assert dr.structural_max_m(floor) < 25.0
    assert dr.structural_max_m(blend) < 30.0
    # and the profiled minimum is real: both neighbours on the grid are worse on train loss
    dist = train["pano_dist"].to_numpy(float)
    dep = train["depression_deg"].to_numpy(float)
    types = train["label_type"].to_numpy(str)
    disp, _ = dr._disparity(dist)
    t = dr._tan_dep(dep)

    def train_mae(a):
        m = dep >= a
        h, _, fb = dr._heights_in_disparity(t[m], disp[m], types[m], "l1")
        params = {"height_by_type_m": h, "height_fallback_m": fb, "blend_deg": float(a)}
        return float(np.mean(np.abs(dr._predict_blend(params, dep, types) - dist)))

    a = blend["blend_deg"]
    assert train_mae(a) < train_mae(a - 0.25)
    assert train_mae(a) < train_mae(a + 0.25)
