"""Invariants of the bearing-only triangulation estimator (issue #7).

These tests do not look at the committed findings — they check that the estimator is
*correct*, on geometry whose answer is known in closed form, and that the corrections it
applies are the right size rather than merely the right sign. The findings themselves are
locked separately in ``test_triangulation_findings.py``.

The load-bearing ones:

- exact recovery on noise-free geometry, including the leave-one-out path used for truth
- the frame convention agrees with the auto-labeler's own stored positions, so a sign or
  axis swap cannot hide
- the Jensen correction removes the simulated bias it is supposed to remove
- ill-conditioned and behind-the-camera configurations are rejected, not silently returned
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mapillary_falsification as mf  # noqa: E402
import triangulation as tg  # noqa: E402


# ======================================================================================
# Geodesy and frame conventions
# ======================================================================================

def test_local_en_roundtrip():
    lat0, lng0 = 37.5480, -77.4479
    lat = np.array([37.5480, 37.5500, 37.5460])
    lng = np.array([-77.4479, -77.4450, -77.4500])
    e, n = tg.local_en(lat, lng, lat0, lng0)
    lat2, lng2 = tg.en_to_latlng(e, n, lat0, lng0)
    assert np.allclose(lat, lat2, atol=1e-12)
    assert np.allclose(lng, lng2, atol=1e-12)


def test_local_en_axes_are_east_and_north():
    """East must grow with longitude, north with latitude, at the right metric scale."""
    lat0, lng0 = 40.0, -74.0
    e, n = tg.local_en(np.array([40.0]), np.array([-74.0 + 1e-3]), lat0, lng0)
    assert e[0] > 0 and abs(n[0]) < 1e-9
    # 1e-3 deg of longitude at 40N is ~85.3 m
    assert 85.0 < e[0] < 85.6
    e, n = tg.local_en(np.array([40.0 + 1e-3]), np.array([-74.0]), lat0, lng0)
    assert n[0] > 0 and abs(e[0]) < 1e-9
    assert 111.0 < n[0] < 111.5


@pytest.mark.parametrize("run", ["richmond", "paterson"])
def test_bearing_frame_matches_stored_member_positions(run):
    """``pano position + range_m along bearing_deg`` must reproduce the member lat/lng.

    This pins the whole convention stack at once — bearing measured clockwise from north,
    ray direction ``(sin, cos)`` in (east, north) — against numbers the auto-labeler wrote
    independently. A sign flip or an axis swap fails here by tens of metres; the committed
    tolerance is set by the sites file's own coordinate rounding.
    """
    f = mf.member_frame(run)
    lat0, lng0 = float(f["pano_lat"].mean()), float(f["pano_lng"].mean())
    pe, pn = tg.local_en(f["pano_lat"], f["pano_lng"], lat0, lng0)
    th = np.radians(f["bearing_deg"].to_numpy())
    lat, lng = tg.en_to_latlng(pe + f["range_m"] * np.sin(th),
                               pn + f["range_m"] * np.cos(th), lat0, lng0)
    me, mn = tg.local_en(f["member_lat"], f["member_lng"], lat0, lng0)
    pe2, pn2 = tg.local_en(lat, lng, lat0, lng0)
    err = np.hypot(pe2 - me, pn2 - mn)
    assert float(np.max(err)) < 0.10, f"{run}: max frame error {np.max(err):.4f} m"


# ======================================================================================
# The estimator on known geometry
# ======================================================================================

def _bearings_to(pe, pn, te, tn):
    return np.degrees(np.arctan2(te - np.asarray(pe), tn - np.asarray(pn))) % 360.0


def test_triangulate_exact_on_noise_free_geometry():
    """Noise-free rays must intersect at the planted point to floating-point precision."""
    pe = np.array([-10.0, -5.0, 0.0, 7.0])
    pn = np.array([0.0, 0.0, 0.0, 0.0])
    target = np.array([3.0, 12.0])
    b = _bearings_to(pe, pn, *target)
    pt, cov = tg.triangulate(pe, pn, b)
    assert np.allclose(pt, target, atol=1e-9)
    assert cov.shape == (2, 2)


def test_triangulate_is_translation_and_rotation_equivariant():
    pe = np.array([-8.0, -2.0, 5.0])
    pn = np.array([0.0, 1.0, -1.0])
    target = np.array([1.0, 9.0])
    b = _bearings_to(pe, pn, *target)

    shift = np.array([250.0, -400.0])
    pt, _ = tg.triangulate(pe + shift[0], pn + shift[1], b)
    assert np.allclose(pt, target + shift, atol=1e-8)

    phi = np.radians(37.0)          # rotate the world clockwise -> bearings gain 37 deg
    rot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    p_rot = rot @ np.vstack([pe, pn])
    t_rot = rot @ target
    pt, _ = tg.triangulate(p_rot[0], p_rot[1], b + 37.0)
    assert np.allclose(pt, t_rot, atol=1e-8)


def test_triangulate_is_blind_to_radial_aim_error():
    """Pull every camera's aim point toward that camera along its own line of sight; no
    bearing changes, so the intersection must still land on the true object position.

    This is the geometric fact behind §8's candidate analysis: a detector centroid that
    sits on the *near face* of a fixed-size object (displaced radially toward each viewer)
    cannot bias the triangulated range at all — it can only enter the same-pixel
    comparison through the depth side, whose raster is read at the displaced point.
    """
    pe = np.array([-10.0, -4.0, 3.0, 9.0])
    pn = np.array([0.0, 1.0, -1.0, 0.5])
    target = np.array([1.0, 12.0])
    bearings = []
    for x, y in zip(pe, pn):
        v = target - np.array([x, y])
        aim = target - 0.6 * v / np.hypot(*v)     # 0.6 m toward this camera: a ramp face
        bearings.append(_bearings_to(np.array([x]), np.array([y]), *aim)[0])
    pt, _ = tg.triangulate(pe, pn, np.array(bearings))
    assert np.allclose(pt, target, atol=1e-9)


def test_triangulate_rejects_parallel_rays():
    """Perfectly collinear rays are singular and must raise, not return a silent answer."""
    with pytest.raises(np.linalg.LinAlgError):
        tg.triangulate(np.array([0.0, 5.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))


def test_covariance_grows_as_intersection_angle_shrinks():
    """Error scales as 1/sin of the intersection angle — the conditioning claim, checked."""
    prev = None
    for sep in (80.0, 40.0, 20.0, 10.0, 5.0):
        # two rays separated by `sep` degrees from a 10 m baseline
        pe = np.array([-5.0, 5.0])
        pn = np.array([0.0, 0.0])
        b = np.array([90.0 - sep / 2.0, 90.0 + sep / 2.0])
        _, cov = tg.triangulate(pe, pn, b)
        scale = float(np.trace(cov))
        if prev is not None:
            assert scale > prev, f"conditioning did not worsen at {sep} deg"
        prev = scale


# ======================================================================================
# Leave-one-out construction
# ======================================================================================

def _synthetic_frame(target=(2.0, 11.0), n=5, height_m=2.35):
    pe = np.linspace(-12.0, 12.0, n)
    pn = np.zeros(n)
    te, tn = target
    r = np.hypot(te - pe, tn - pn)
    return pd.DataFrame({
        "site_id": np.zeros(n, dtype=int),
        "pano_id": [f"p{i}" for i in range(n)],
        "pano_e": pe, "pano_n": pn,
        "bearing_deg": _bearings_to(pe, pn, te, tn),
        "range_m": r,
        "dep_deg": np.degrees(np.arctan2(height_m, r)),
    })


def test_loo_solve_recovers_the_point_for_every_member():
    """Each leave-one-out system must return the planted point — that is what makes the
    resulting range independent of the member it is truth for."""
    f = _synthetic_frame()
    sol = tg._loo_solve(f)
    assert np.allclose(sol["loo_e"], 2.0, atol=1e-8)
    assert np.allclose(sol["loo_n"], 11.0, atol=1e-8)


def test_loo_solve_matches_an_explicit_leave_one_out_loop():
    """The groupby-sums shortcut must equal the literal drop-one-and-refit computation."""
    f = _synthetic_frame(n=6)
    sol = tg._loo_solve(f)
    for i in range(len(f)):
        keep = np.ones(len(f), bool)
        keep[i] = False
        pt, _ = tg.triangulate(f["pano_e"][keep], f["pano_n"][keep],
                               f["bearing_deg"][keep])
        assert np.allclose([sol["loo_e"][i], sol["loo_n"][i]], pt, atol=1e-7)


def test_loo_is_blind_to_the_left_out_members_own_bearing():
    """Perturbing member i's bearing must not move member i's own truth at all.

    This is the property the whole report rests on: the leave-one-out range cannot be
    contaminated by the observation it is scoring.
    """
    f = _synthetic_frame(n=5)
    base = tg._loo_solve(f)
    g = f.copy()
    g.loc[2, "bearing_deg"] += 12.0
    moved = tg._loo_solve(g)
    assert np.allclose(base.loc[2, ["loo_e", "loo_n"]],
                       moved.loc[2, ["loo_e", "loo_n"]], atol=1e-10)
    # and it must move the *others*, otherwise the test proves nothing
    assert not np.allclose(base.loc[0, ["loo_e", "loo_n"]],
                           moved.loc[0, ["loo_e", "loo_n"]], atol=1e-6)


def test_implied_height_recovers_a_planted_camera_height_noise_free():
    """r_tri * tan(depression) must return the height the depressions were built from."""
    for h in (2.0, 2.3411, 2.6, 3.1):
        f = _synthetic_frame(height_m=h, n=5)
        sol = tg._loo_solve(f)
        de = sol["loo_e"] - f["pano_e"]
        dn = sol["loo_n"] - f["pano_n"]
        r = np.hypot(de, dn)
        implied = r * np.tan(np.radians(f["dep_deg"]))
        assert np.allclose(implied, h, atol=1e-8), h


# ======================================================================================
# Bias corrections
# ======================================================================================

def test_jensen_bias_formula_shape():
    r = np.array([5.0, 10.0, 20.0])
    s = np.full(3, 1.0)
    b = tg.jensen_bias_m(r, s)
    assert np.allclose(b, s ** 2 / (2 * r))
    assert np.all(np.diff(b) < 0), "bias must shrink with range"
    assert np.all(tg.jensen_bias_m(r, np.zeros(3)) == 0)


def test_jensen_correction_removes_most_of_the_simulated_bias():
    """The correction has to be right in magnitude, not just in sign."""
    res = tg.monte_carlo_bias_check(sigma_pos_m=1.0, sigma_bearing_deg=1.4,
                                    n_trials=8000, seed=7)
    assert res["bias_raw_m"] > 0, "norm convexity must inflate the raw range"
    assert abs(res["bias_corrected_m"]) < 0.4 * abs(res["bias_raw_m"])


def _synthetic_member_frame(sigma_bearing_deg, sigma_pos_m, height_m=2.35,
                            n_sites=2500, seed=3):
    """A member frame shaped like a real run, with known noise and a known camera height.

    Panoramas run along a street with jittered spacing, objects sit off to one side at
    4-24 m — the geometry the auto-labeler actually produces — so the recovered numbers
    are tested in the regime they are used in.
    """
    rng = np.random.default_rng(seed)
    lat0, lng0 = 40.0, -74.0
    rows = []
    for sid in range(n_sites):
        n = int(rng.integers(3, 7))
        pe = np.linspace(-14, 14, n) + rng.normal(0, 2, n)
        pn = np.zeros(n)
        te, tn = rng.uniform(-6, 6), rng.uniform(4, 24)
        r = np.hypot(te - pe, tn - pn)
        lat, lng = tg.en_to_latlng(pe + rng.normal(0, sigma_pos_m, n),
                                   pn + rng.normal(0, sigma_pos_m, n), lat0, lng0)
        dep = np.degrees(np.arctan2(height_m, r))
        rows.append(pd.DataFrame({
            "site_id": sid, "pano_id": [f"{sid}_{i}" for i in range(n)],
            "pano_lat": lat, "pano_lng": lng,
            "bearing_deg": _bearings_to(pe, pn, te, tn)
            + rng.normal(0, sigma_bearing_deg, n),
            # exactly what the real sites files carry: the range the auto-labeler's
            # assumed 2.6 m implies for this depression, NOT the true range — the global
            # scale fit reads this column, so the fixture must not hand it the answer
            "range_m": tg.COT_CAMERA_HEIGHT / np.tan(np.radians(dep)),
            "dep_deg": dep,
            "pano_height": 8192, "sequence_id": "s",
            "member_lat": lat, "member_lng": lng,
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("sigma_b,sigma_p", [(1.2, 0.6), (1.4, 0.45)])
def test_converged_noise_fit_recovers_planted_sigmas(sigma_b, sigma_p):
    """The damped fixed point must find the noise that was actually planted.

    The *first* pass deliberately over-states both sigmas — it has no leave-one-out
    covariance to subtract yet — which is why ``fit_noise`` iterates instead of taking it.

    Both parametrisations bracket what the real runs measure (sigma_bearing 1.2-1.4 deg,
    sigma_pos 0.24-0.56 m), which is the regime the decomposition is used in.
    """
    fit = tg.fit_noise("synthetic",
                       frame=_synthetic_member_frame(sigma_b, sigma_p))
    fit.pop("frame")
    assert fit["sigma_bearing_deg"] == pytest.approx(sigma_b, rel=0.20)
    assert fit["sigma_pos_m"] == pytest.approx(sigma_p, rel=0.20)


def test_noise_decomposition_degrades_when_position_noise_dominates():
    """The identifiability limit, recorded rather than hidden.

    The two variance components are told apart only by their range dependence. When
    panorama position error dominates, the bearing term is a small slope on a large
    intercept and ``sigma_bearing`` is over-stated — here by ~50%. This is documented
    because it bounds where the error budget can be trusted; the companion test below
    shows the *height* is unaffected, which is why the headline survives it.
    """
    fit = tg.fit_noise("synthetic", frame=_synthetic_member_frame(0.8, 1.0))
    fit.pop("frame")
    assert fit["sigma_bearing_deg"] > 0.8, "expected an over-statement, not an under-one"
    assert fit["sigma_bearing_deg"] < 2.0
    assert fit["sigma_pos_m"] == pytest.approx(1.0, rel=0.20)


@pytest.mark.parametrize("sigma_b,sigma_p", [(1.2, 0.6), (1.4, 0.45), (0.8, 1.0)])
def test_pipeline_recovers_a_planted_camera_height_under_realistic_noise(sigma_b, sigma_p):
    """The headline estimand, end to end, on geometry whose answer is known.

    This is the test the report's absolute claim stands on: at the noise levels actually
    measured on these runs, the full pipeline — leave-one-out triangulation, Jensen
    correction, conditioning gate, median — returns the camera height it was given. If
    the estimator manufactured height, it would show up here as a planted-vs-recovered
    gap of the same size as the gap the report reports against the shipped constant.
    """
    planted = 2.35
    fit = tg.fit_noise("synthetic",
                       frame=_synthetic_member_frame(sigma_b, sigma_p, height_m=planted))
    got = tg.implied_height(fit["frame"], n_boot=60)
    assert got["median_m"] == pytest.approx(planted, abs=0.03), got


def test_refine_argmin_recovers_an_off_grid_vertex_exactly():
    """The parabolic vertex refinement behind the bootstrap interval: exact on a true
    parabola, and falling back to the grid point at an edge or on a non-convex triple."""
    ks = np.arange(0.7, 1.3, 0.002)
    true_k = 0.94631
    losses = (ks - true_k) ** 2 + 0.8
    got = tg._refine_argmin(ks, losses, int(np.argmin(losses)))
    assert got == pytest.approx(true_k, abs=1e-12)
    assert tg._refine_argmin(ks, np.linspace(1.0, 2.0, len(ks)), 0) == pytest.approx(ks[0])
    assert tg._refine_argmin(ks, np.ones_like(ks), 5) == pytest.approx(ks[5])


def test_fit_model_scale_interval_contains_the_planted_height():
    """The interval must be nondegenerate and cover both the estimate and the truth —
    before the refinement, grid snapping could produce a width-zero band that excluded
    its own point estimate (bend, in the committed first build)."""
    fit = tg.fit_noise("synthetic",
                       frame=_synthetic_member_frame(1.2, 0.5, height_m=2.35))
    g = tg.fit_model_scale(fit["frame"], n_boot=40)
    lo, hi = g["ci95_m"]
    assert lo < hi
    assert lo <= g["height_m"] <= hi
    assert g["height_m"] == pytest.approx(2.35, abs=0.03)


def test_quality_gates_sweep_is_well_formed_on_synthetic_data():
    """The depth-anchor gate sweep: tightening a gate must shrink the population and, on
    data built at one constant ratio, must not move the measured ratio."""
    import triangulation_depth as td

    rng = np.random.default_rng(0)
    n = 600
    g = pd.DataFrame({
        "r_tri": rng.uniform(4, 20, n),
        "bearing_resid_deg": rng.normal(0, 1.2, n),
        "sigma_r_m": rng.uniform(0.1, 1.5, n),
        "n_panos": rng.integers(3, 8, n),
    })
    g["r_depth"] = g["r_tri"] / 1.138
    qg = td.quality_gates(g)
    for sweep in (qg["by_max_abs_bearing_resid_deg"], qg["by_sigma_r_m"]):
        ns = [sweep[k]["n"] for k in sorted(sweep, key=float)]
        assert ns == sorted(ns), ns                     # looser gate, larger population
    ns = [qg["by_min_panos"][k]["n"] for k in sorted(qg["by_min_panos"], key=int)]
    assert ns == sorted(ns, reverse=True), ns
    for sweep in qg.values():
        for v in sweep.values():
            assert v["median_ratio"] == pytest.approx(1.138, abs=0.001)


# ======================================================================================
# Gating
# ======================================================================================

def test_usable_rejects_behind_camera_and_out_of_range():
    f = pd.DataFrame({
        "forward_m": [10.0, -3.0, 10.0, 10.0, 10.0],
        "r_tri": [10.0, 10.0, 0.2, 400.0, 10.0],
        "dep_deg": [12.0, 12.0, 12.0, 12.0, -1.0],
        "sigma_r_m": [np.nan] * 5,
    })
    ok = tg.usable(f).to_numpy()
    assert ok.tolist() == [True, False, False, False, False]


def test_usable_applies_the_sigma_gate_when_present():
    f = pd.DataFrame({
        "forward_m": [10.0, 10.0],
        "r_tri": [10.0, 10.0],
        "dep_deg": [12.0, 12.0],
        "sigma_r_m": [0.5, 9.0],
    })
    assert tg.usable(f, sigma_gate_m=1.5).to_numpy().tolist() == [True, False]


def test_site_frame_drops_duplicate_panos_and_small_sites():
    """Two rays from one panorama share an origin and add no baseline, so only one may
    survive; and a site must have three distinct panoramas to support leave-one-out."""
    f = tg.site_frame("clovis")
    counts = f.groupby(["site_id", "pano_id"]).size()
    assert counts.max() == 1
    assert f.groupby("site_id")["pano_id"].size().min() >= tg.MIN_PANOS_FOR_LOO


def test_max_intersection_angle_bounds():
    assert tg._max_intersection_angle(np.array([0.0, 90.0])) == pytest.approx(90.0)
    assert tg._max_intersection_angle(np.array([0.0, 180.0])) == pytest.approx(0.0)
    assert tg._max_intersection_angle(np.array([10.0, 30.0])) == pytest.approx(20.0)


# ======================================================================================
# The shipped model, as this module evaluates it
# ======================================================================================

def test_blend_matches_the_shipped_parameters_and_is_continuous():
    p = tg.load_shipped_blend()["params"]
    assert p["height_m"] == pytest.approx(2.341219672825709)
    assert p["blend_deg"] == pytest.approx(11.25)
    a = p["blend_deg"]
    lo = tg._blend(np.array([a - 1e-7]), p["height_m"], a)[0]
    hi = tg._blend(np.array([a + 1e-7]), p["height_m"], a)[0]
    assert lo == pytest.approx(hi, abs=1e-4), "the blend must be continuous at the knot"
    # above the knot it is exactly the cotangent
    dep = np.array([15.0, 30.0, 45.0])
    assert np.allclose(tg._blend(dep, p["height_m"], a),
                       p["height_m"] / np.tan(np.radians(dep)))


def test_blend_matches_the_canonical_refit_implementation():
    """``tg._blend`` is a reimplementation of ``distance_refit._predict_blend`` (the third
    C1 blend in the repo); the two must agree everywhere, or the scoring here silently
    drifts from the refit the shipped parameters came out of."""
    import distance_refit as dr
    p = tg.load_shipped_blend()["params"]
    dep = np.linspace(-5.0, 60.0, 1301)
    ours = tg._blend(dep, p["height_m"], p["blend_deg"])
    theirs = dr._predict_blend({"height_m": p["height_m"], "blend_deg": p["blend_deg"]}, dep)
    assert np.allclose(ours, theirs, atol=1e-9)


def test_cotangent_reproduces_the_stored_auto_labeler_range():
    """The sites' own ``range_m`` is 2.6/tan(depression); if that ever stops being true,
    every ratio in the report silently changes meaning."""
    chk = mf.conventions_check("richmond")
    assert chk["max_abs_range_m_delta"] < 1e-3
