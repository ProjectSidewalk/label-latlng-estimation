"""Bearing-only triangulation: an estimator for multiply-observed labels, and a
resolution-independent, depth-free ground truth (issue #7).

Why this exists
---------------
Every distance estimate in this repository — the 2021 linear fit, the 2026 refit's
cotangent family, the shipped ``final_coefficients`` — converts a *vertical* click angle
into a ground distance. That conversion needs two things the horizontal axis does not:
a flat-ground assumption, and a camera height. The camera height was ultimately measured
against Google's own depth rasters, which is why ``final_coefficients`` carries the
caveat

    "the absolute reference is Google's measured ground planes - internally consistent
     ... but externally unanchored; bearing-only triangulation (#7) is the independent
     path"

This module is that path. Given two or more panoramas that see the same physical object,
the *bearings* alone fix its position: rays from known origins intersect. The range that
falls out uses

  - no vertical click angle,
  - no camera height,
  - no ground-plane assumption,
  - no depth data,
  - and no panorama resolution,

so it is an external metric anchor for everything the depth chain calibrated.

The leave-one-out construction
------------------------------
For a site (a fused cluster of detections of one object) seen from N >= 3 distinct
panoramas, member ``i``'s range truth is obtained by triangulating from the bearings of
members ``j != i`` and measuring from pano ``i`` to that point. Member ``i``'s own
observation contributes nothing to its own truth, and *no* member's vertical angle
contributes to any of it.

That last point is what makes this different from the Stage 3 diagnostics
(``mapillary_falsification``). Those score each model against a consensus built from the
model's own predicted ranges, so the residuals are demeaned within site by construction
and a shared scale error is provably invisible (that module's §"MODEL_KEYS" note). Fixing
the consensus by bearings alone breaks the demeaning, and absolute scale becomes
identifiable.

What has to be modelled before the number means anything
--------------------------------------------------------
A raw leave-one-out range is *not* an unbiased range truth, for three separate reasons,
and all three are handled here rather than assumed away:

1. **Norm convexity (Jensen).** ``r = |X_hat - p|`` is a convex function of the estimated
   point. Cross-ray error in ``X_hat`` therefore inflates ``E[r]`` by ``sigma_c^2 / (2r)``
   to second order — a *positive*, range-dependent bias that would masquerade as a taller
   camera. :func:`jensen_bias_m` computes it per member from the propagated covariance;
   :func:`monte_carlo_bias_check` verifies the formula against simulation.

2. **Two noise sources with different range dependence.** A bearing error contributes a
   perpendicular miss proportional to range; a panorama *position* error contributes one
   that is independent of range. They are separately identifiable by regressing squared
   perpendicular residuals on squared range (:func:`variance_components`), which is also
   the honest way to weight the fit and to state how good Mapillary's SfM positions are.

3. **Conditioning.** Error scales as ``1 / sin`` of the intersection angle, so nearly
   collinear rays triangulate badly. :func:`site_frame` reports the propagated per-member
   range sigma, which is the principled version of "report the conditioning distribution
   honestly" and is what the gates key on.

Frame conventions (shared with ``mapillary_falsification``, verified there)
--------------------------------------------------------------------------
``bearing_deg`` is an absolute bearing, clockwise from north, so a unit ray direction is
``(east, north) = (sin theta, cos theta)``. Local ENU metres come from an equirectangular
tangent plane about a per-run origin (:func:`local_en`); over a city-scale extent this is
well below the noise floor measured here, and :func:`test_triangulation_contract` pins it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mapillary_falsification as mf

ROOT = Path(__file__).resolve().parent.parent
#: Path, not str — ``mapillary_falsification`` indexes it with ``/``.
DATA_DIR = ROOT / "data"

EARTH_R = mf.EARTH_R
MAPILLARY_RUNS = list(mf.MAPILLARY_RUNS)
GSV_RUNS = list(mf.GSV_RUNS)
ALL_RUNS = GSV_RUNS + MAPILLARY_RUNS

#: The ecosystem's assumed camera height. The auto-labeler fused every run at this value,
#: so ``range_m`` in the sites files is exactly ``2.6 / tan(depression)`` — pinned to
#: sub-millimetre by ``mapillary_falsification.conventions_check``.
COT_CAMERA_HEIGHT = mf.COT_CAMERA_HEIGHT

#: Leave-one-out needs at least two *other* panoramas.
MIN_PANOS_FOR_LOO = 3

#: Physically admissible leave-one-out range. The auto-labeler's own fuse gate kept every
#: member inside 25 m at 2.6 m, so anything far outside this bracket is a triangulation
#: failure (near-parallel rays, a mis-clustered member), not a long sighting.
MIN_RANGE_M = 1.0
MAX_RANGE_M = 60.0

#: Default conditioning gate: the propagated 1-sigma uncertainty of the leave-one-out
#: range, in metres. Chosen in the report's error-budget section, not tuned on the answer;
#: :func:`scale_sensitivity` re-runs the headline across a sweep of this value.
SIGMA_R_GATE_M = 1.5

SEED = 20260808


# ======================================================================================
# Geodesy
# ======================================================================================

def local_en(lat, lng, lat0: float, lng0: float) -> tuple[np.ndarray, np.ndarray]:
    """Local east/north metres about (lat0, lng0), equirectangular tangent plane."""
    east = np.radians(np.asarray(lng, dtype=float) - lng0) \
        * np.cos(np.radians(lat0)) * EARTH_R
    north = np.radians(np.asarray(lat, dtype=float) - lat0) * EARTH_R
    return east, north


def en_to_latlng(east, north, lat0: float, lng0: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`local_en`."""
    lat = lat0 + np.degrees(np.asarray(north, dtype=float) / EARTH_R)
    lng = lng0 + np.degrees(np.asarray(east, dtype=float)
                            / (EARTH_R * np.cos(np.radians(lat0))))
    return lat, lng


# ======================================================================================
# The estimator
# ======================================================================================

def triangulate(east, north, bearing_deg, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares intersection of bearing rays.

    Minimises ``sum_i w_i (n_i . (x - p_i))^2`` where ``n_i`` is the unit normal to ray
    ``i`` — i.e. the weighted sum of squared *perpendicular* distances from the point to
    each ray. Returns ``(point_en, covariance)``; the covariance is the inverse normal
    matrix, which is the estimator covariance when ``w_i = 1 / Var(perpendicular miss)``.

    Raises ``np.linalg.LinAlgError`` on a singular (perfectly collinear) configuration.
    """
    e = np.asarray(east, dtype=float)
    n = np.asarray(north, dtype=float)
    theta = np.radians(np.asarray(bearing_deg, dtype=float))
    w = np.ones_like(e) if weights is None else np.asarray(weights, dtype=float)
    # unit normal to the ray direction (sin, cos)
    nx, ny = np.cos(theta), -np.sin(theta)
    c = nx * e + ny * n
    A = np.array([[np.sum(w * nx * nx), np.sum(w * nx * ny)],
                  [np.sum(w * nx * ny), np.sum(w * ny * ny)]])
    b = np.array([np.sum(w * nx * c), np.sum(w * ny * c)])
    cov = np.linalg.inv(A)
    return cov @ b, cov


def _loo_solve(frame: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    """Vectorised leave-one-out triangulation for every member of every site.

    The normal equations are additive over members, so member ``i``'s leave-one-out system
    is the site's full system minus member ``i``'s own rank-1 term. That turns what would
    be one 2x2 solve per (site, member) into groupby sums plus a closed-form 2x2 inverse
    over the whole table at once.
    """
    f = frame
    theta = np.radians(f["bearing_deg"].to_numpy())
    nx, ny = np.cos(theta), -np.sin(theta)
    w = np.ones(len(f)) if weight_col is None else f[weight_col].to_numpy()
    c = nx * f["pano_e"].to_numpy() + ny * f["pano_n"].to_numpy()

    parts = pd.DataFrame({
        "g": f["site_id"].to_numpy(),
        "xx": w * nx * nx, "xy": w * nx * ny, "yy": w * ny * ny,
        "bx": w * nx * c, "by": w * ny * c,
    })
    tot = parts.groupby("g")[["xx", "xy", "yy", "bx", "by"]].transform("sum")

    axx = tot["xx"].to_numpy() - w * nx * nx
    axy = tot["xy"].to_numpy() - w * nx * ny
    ayy = tot["yy"].to_numpy() - w * ny * ny
    bx = tot["bx"].to_numpy() - w * nx * c
    by = tot["by"].to_numpy() - w * ny * c

    det = axx * ayy - axy * axy
    with np.errstate(divide="ignore", invalid="ignore"):
        ex = np.where(det != 0, (ayy * bx - axy * by) / det, np.nan)
        en = np.where(det != 0, (-axy * bx + axx * by) / det, np.nan)
        # leave-one-out covariance = inverse of the leave-one-out normal matrix
        cxx = np.where(det != 0, ayy / det, np.nan)
        cxy = np.where(det != 0, -axy / det, np.nan)
        cyy = np.where(det != 0, axx / det, np.nan)
    return pd.DataFrame({"loo_e": ex, "loo_n": en,
                         "cov_xx": cxx, "cov_xy": cxy, "cov_yy": cyy,
                         "det": det}, index=f.index)


# ======================================================================================
# Site frame construction
# ======================================================================================

def site_frame(run: str, data_dir: Path = DATA_DIR,
               sigma_bearing_deg: float | None = None,
               sigma_pos_m: float | None = None,
               bearing_offset_deg: float = 0.0,
               frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per member of every site with >= :data:`MIN_PANOS_FOR_LOO` distinct panos,
    carrying the leave-one-out bearing-only range truth and its propagated uncertainty.

    When ``sigma_bearing_deg``/``sigma_pos_m`` are supplied the triangulation is weighted
    by the implied perpendicular variance ``r^2 sigma_theta^2 + sigma_p^2`` and the
    returned covariances are metric; with them ``None`` the fit is unweighted and the
    covariances are in units of "perpendicular-miss variance", used only for shape.

    ``bearing_offset_deg`` rotates every ray, used by :func:`fit_bearing_offset`.
    """
    f = mf.member_frame(run, data_dir) if frame is None else frame.copy()
    # One detection per (site, pano): two rays from a common origin add no baseline, and
    # the pair would otherwise inflate the apparent conditioning of that site.
    f = f.sort_values(["site_id", "pano_id"]).drop_duplicates(["site_id", "pano_id"])
    npano = f.groupby("site_id")["pano_id"].transform("size")
    f = f[npano >= MIN_PANOS_FOR_LOO].copy()
    if f.empty:
        return f

    lat0 = float(f["pano_lat"].mean())
    lng0 = float(f["pano_lng"].mean())
    f["pano_e"], f["pano_n"] = local_en(f["pano_lat"], f["pano_lng"], lat0, lng0)
    f["bearing_deg"] = f["bearing_deg"] + bearing_offset_deg

    weighted = sigma_bearing_deg is not None and sigma_pos_m is not None
    if weighted:
        # Weight by the variance the member's *own* (model-free) range implies. range_m is
        # used only as a scale for the weight, never as truth: a weight is second-order.
        sig_t = np.radians(sigma_bearing_deg)
        var = (f["range_m"].to_numpy() * sig_t) ** 2 + sigma_pos_m ** 2
        f["w"] = 1.0 / var
    sol = _loo_solve(f, "w" if weighted else None)
    f = pd.concat([f, sol], axis=1)

    de = f["loo_e"] - f["pano_e"]
    dn = f["loo_n"] - f["pano_n"]
    f["r_tri_raw"] = np.hypot(de, dn)
    theta = np.radians(f["bearing_deg"].to_numpy())
    ux, uy = np.sin(theta), np.cos(theta)
    # Forward distance along the member's own ray; negative means the leave-one-out point
    # sits *behind* the camera, which is a triangulation failure, not a sighting.
    f["forward_m"] = de * ux + dn * uy
    f["tri_bearing_deg"] = np.degrees(np.arctan2(de, dn)) % 360.0
    # Angle between the member's ray and the direction to the leave-one-out point: a pure
    # bearing-consistency residual, again free of any vertical model.
    f["bearing_resid_deg"] = ((f["tri_bearing_deg"] - f["bearing_deg"] + 180.0) % 360.0) - 180.0

    # Propagate the leave-one-out covariance onto the ray direction (along) and its
    # perpendicular (cross). "along" is the range uncertainty; "cross" drives Jensen.
    cxx, cxy, cyy = f["cov_xx"].to_numpy(), f["cov_xy"].to_numpy(), f["cov_yy"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = f["r_tri_raw"].to_numpy()
        uex, uen = np.where(rr > 0, de / rr, 0.0), np.where(rr > 0, dn / rr, 0.0)
    f["var_along"] = uex ** 2 * cxx + 2 * uex * uen * cxy + uen ** 2 * cyy
    f["var_cross"] = uen ** 2 * cxx - 2 * uex * uen * cxy + uex ** 2 * cyy

    if weighted:
        # Pano i's own position error is not in the leave-one-out covariance (it never
        # entered that fit) but does enter the measured range, in both components.
        f["sigma_r_m"] = np.sqrt(np.maximum(f["var_along"], 0) + sigma_pos_m ** 2)
        f["sigma_cross_m"] = np.sqrt(np.maximum(f["var_cross"], 0) + sigma_pos_m ** 2)
        f["jensen_m"] = jensen_bias_m(f["r_tri_raw"].to_numpy(),
                                      f["sigma_cross_m"].to_numpy())
        f["r_tri"] = f["r_tri_raw"] - f["jensen_m"]
    else:
        f["sigma_r_m"] = np.nan
        f["sigma_cross_m"] = np.nan
        f["jensen_m"] = 0.0
        f["r_tri"] = f["r_tri_raw"]

    f["dep_deg"] = f["dep_deg"].astype(float)
    f["implied_height_m"] = f["r_tri"] * np.tan(np.radians(f["dep_deg"]))
    f["n_panos"] = f.groupby("site_id")["pano_id"].transform("size")
    # Within-site spread of the *assumed-height* ranges. This is the lever the auto-
    # labeler's own fuse gate pulls on: a wrong camera height scales every member's range
    # by the same factor, so the members' projected positions disagree in proportion to
    # how much their ranges differ. Sites with little range spread are near-insensitive to
    # the height and so were retained regardless of it; sites with large spread are where
    # the gate could have selected for consistency with 2.6 m. See :func:`selection_probe`.
    f["site_range_spread_m"] = f.groupby("site_id")["range_m"].transform(
        lambda r: r.max() - r.min())
    return f.reset_index(drop=True)


def jensen_bias_m(r: np.ndarray, sigma_cross_m: np.ndarray) -> np.ndarray:
    """Second-order bias of ``|X_hat - p|`` from cross-ray error in ``X_hat``.

    ``r_measured = |(r + a, c)| ~= r + a + c^2 / (2r)`` with ``E[a] = 0``, so the range is
    inflated by ``sigma_c^2 / (2r)``. Positive and larger at short range, which is exactly
    the shape that would otherwise be misread as a taller camera in the near field.
    """
    r = np.asarray(r, dtype=float)
    s = np.asarray(sigma_cross_m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, s ** 2 / (2.0 * r), 0.0)


def usable(f: pd.DataFrame, sigma_gate_m: float = SIGMA_R_GATE_M) -> pd.Series:
    """The analysis population: a well-conditioned, physically admissible LOO range."""
    ok = (f["forward_m"] > 0) & f["r_tri"].notna()
    ok &= (f["r_tri"] > MIN_RANGE_M) & (f["r_tri"] < MAX_RANGE_M)
    ok &= f["dep_deg"] > 0
    if f["sigma_r_m"].notna().any():
        ok &= f["sigma_r_m"] <= sigma_gate_m
    return ok


# ======================================================================================
# Error budget
# ======================================================================================

def variance_components(f: pd.DataFrame) -> dict:
    """Split the perpendicular miss into a bearing term and a panorama-position term.

    A bearing error ``dtheta`` puts the ray ``r * dtheta`` off the true point; a panorama
    position error puts it off by a range-independent amount. So

        Var(perpendicular miss) = r^2 * sigma_theta^2 + sigma_pos^2

    and regressing the squared miss on squared range identifies both. The miss used here
    is the leave-one-out one — member ``i`` against a point it did not help fit — so the
    residual carries no fitted-degrees-of-freedom shrinkage; what it *does* carry is the
    leave-one-out point's own error, which is removed by subtracting the propagated
    ``var_cross`` before the regression.
    """
    ok = (f["forward_m"] > 0) & f["r_tri_raw"].notna()
    ok &= (f["r_tri_raw"] > MIN_RANGE_M) & (f["r_tri_raw"] < MAX_RANGE_M)
    d = f[ok]
    if len(d) < 50:
        return {"n": int(len(d)), "sigma_bearing_deg": None, "sigma_pos_m": None}
    # perpendicular miss of the member's ray from the leave-one-out point
    miss = d["r_tri_raw"].to_numpy() * np.sin(np.radians(d["bearing_resid_deg"].to_numpy()))
    r = d["r_tri_raw"].to_numpy()

    # Squared misses are heavy-tailed — a mis-clustered member misses by tens of metres —
    # so the scale is estimated per range bin with a normal-consistent MAD and the two
    # components are read off a bin-level regression. That keeps a handful of outliers
    # from setting the whole error budget, and it is what fig 26 plots.
    nbin = int(np.clip(len(d) // 400, 6, 14))
    edges = np.unique(np.quantile(r, np.linspace(0, 1, nbin + 1)))
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 25:
            continue
        s = 1.4826 * float(np.median(np.abs(miss[m] - np.median(miss[m]))))
        var_obs = s ** 2
        # remove the leave-one-out point's own contribution, which is already known
        if d["sigma_r_m"].notna().any():
            var_obs -= float(np.median(np.nan_to_num(d["var_cross"].to_numpy()[m])))
        rows.append((float(np.median(r[m]) ** 2), var_obs, int(m.sum())))
    if len(rows) < 3:
        return {"n": int(len(d)), "sigma_bearing_deg": None, "sigma_pos_m": None}
    r2 = np.array([x[0] for x in rows])
    v = np.array([x[1] for x in rows])
    w = np.array([x[2] for x in rows], dtype=float)
    X = np.column_stack([np.ones(len(r2)), r2])
    Xw = X * w[:, None]
    beta = np.linalg.solve(Xw.T @ X, Xw.T @ v)
    var_pos = max(float(beta[0]), 0.0)
    var_theta = max(float(beta[1]), 0.0)
    return {
        "n": int(len(d)),
        "n_bins": len(rows),
        "sigma_pos_m": round(float(np.sqrt(var_pos)), 4),
        "sigma_bearing_deg": round(float(np.degrees(np.sqrt(var_theta))), 4),
        "median_abs_bearing_resid_deg": round(
            float(np.median(np.abs(d["bearing_resid_deg"]))), 4),
        "bins": [{"r2": round(a, 2), "var_miss_m2": round(b, 4), "n": c}
                 for a, b, c in rows],
    }


def fit_noise(run: str, data_dir: Path = DATA_DIR, frame: pd.DataFrame | None = None,
              n_iter: int = 20, damping: float = 0.5, tol: float = 1e-3) -> dict:
    """Damped fixed point for (sigma_bearing, sigma_pos), and the frame fitted at them.

    The two are coupled: the weights depend on them, the leave-one-out covariance depends
    on the weights, and :func:`variance_components` subtracts that covariance. Iterating
    undamped oscillates — the first pass has no covariance to subtract and so over-states
    both sigmas, the second then over-subtracts and drives ``sigma_pos`` to zero. Averaging
    each step with the previous one converges in a handful of passes.

    Returns the converged sigmas, the iteration trace (so the report can show it settled
    rather than asserting it), and the fitted frame.
    """
    base = mf.member_frame(run, data_dir) if frame is None else frame
    f = site_frame(run, data_dir, frame=base)
    vc = variance_components(f)
    sb, sp = vc["sigma_bearing_deg"], vc["sigma_pos_m"]
    trace = [{"iter": 0, "sigma_bearing_deg": sb, "sigma_pos_m": sp}]
    if sb is None:
        return {"sigma_bearing_deg": None, "sigma_pos_m": None, "frame": f, "trace": trace}
    converged = False
    for i in range(1, n_iter + 1):
        f = site_frame(run, data_dir, sigma_bearing_deg=sb, sigma_pos_m=max(sp, 1e-3),
                       frame=base)
        vc = variance_components(f)
        if vc["sigma_bearing_deg"] is None:
            # the reweighted population fell below the decomposition's minimum (possible
            # on a tiny --runs subset); keep the last resolvable sigmas rather than crash
            break
        nb = damping * sb + (1 - damping) * vc["sigma_bearing_deg"]
        npos = damping * sp + (1 - damping) * vc["sigma_pos_m"]
        trace.append({"iter": i, "sigma_bearing_deg": round(nb, 4),
                      "sigma_pos_m": round(npos, 4)})
        converged = abs(nb - sb) < tol and abs(npos - sp) < tol
        sb, sp = nb, npos
        if converged:
            break
    f = site_frame(run, data_dir, sigma_bearing_deg=sb, sigma_pos_m=max(sp, 1e-3),
                   frame=base)
    final = variance_components(f)
    # the explicit flag, not `len(trace) - 1 < n_iter`: the degenerate early break above
    # (decomposition unresolvable on a tiny subset) must not report as convergence
    return {"sigma_bearing_deg": round(float(sb), 4), "sigma_pos_m": round(float(sp), 4),
            "n_iter": len(trace) - 1, "converged": bool(converged),
            "trace": trace, "bins_final": final.get("bins", []),
            "median_abs_bearing_resid_deg": final.get("median_abs_bearing_resid_deg"),
            "frame": f}


def fit_bearing_offset(run: str, data_dir: Path = DATA_DIR,
                       frame: pd.DataFrame | None = None,
                       grid_deg: float = 3.0, step_deg: float = 0.05) -> dict:
    """A global rotation applied to every ray, profiled on total squared perpendicular miss.

    A systematic yaw error in the panorama headings — plausible for Mapillary's SfM
    compass, less so for GSV — rotates every ray in a run together. It is identifiable
    because the true object positions are fixed: rotating all rays cannot be absorbed by
    moving the points. A non-zero offset would be a defect in the *bearing* half, and the
    scale estimate is re-run at the fitted offset to show it does not move the answer.
    """
    base = mf.member_frame(run, data_dir) if frame is None else frame

    def loss(off: float) -> float:
        f = site_frame(run, data_dir, bearing_offset_deg=float(off), frame=base)
        ok = (f["forward_m"] > 0) & f["r_tri_raw"].notna() & (f["r_tri_raw"] < MAX_RANGE_M)
        miss = f.loc[ok, "r_tri_raw"] * np.sin(np.radians(f.loc[ok, "bearing_resid_deg"]))
        return float(np.mean(np.abs(miss)))   # L1: robust to mis-clustered members

    # Coarse sweep then a local refinement: the loss is smooth and near-quadratic in the
    # offset, so a dense grid over the whole window buys nothing but wall-clock.
    coarse = np.arange(-grid_deg, grid_deg + 1e-9, 0.25)
    cl = np.array([loss(o) for o in coarse])
    c0 = float(coarse[int(np.argmin(cl))])
    fine = np.arange(c0 - 0.25, c0 + 0.25 + 1e-9, step_deg)
    fl = np.array([loss(o) for o in fine])
    i = int(np.argmin(fl))
    return {
        "best_offset_deg": round(float(fine[i]), 3),
        "loss_at_best_m": round(float(fl[i]), 4),
        "loss_at_zero_m": round(float(loss(0.0)), 4),
        "n_evals": len(coarse) + len(fine) + 1,
    }


def monte_carlo_bias_check(sigma_pos_m: float = 1.0, sigma_bearing_deg: float = 0.5,
                           n_panos: int = 4, r_true: float = 10.0,
                           n_trials: int = 20000, seed: int = SEED) -> dict:
    """Simulate the whole leave-one-out construction on synthetic geometry.

    Places one object at a known range from a line of panoramas, corrupts panorama
    positions and bearings at the given sigmas, runs the same estimator, and reports the
    residual bias before and after the analytic Jensen correction. This is the check that
    the correction is right in magnitude and not merely in sign.
    """
    rng = np.random.default_rng(seed)
    # Panoramas along a street (east), object offset to the north — the real geometry.
    spacing = 5.0
    pe0 = (np.arange(n_panos) - (n_panos - 1) / 2.0) * spacing
    pn0 = np.zeros(n_panos)
    obj = np.array([0.0, r_true])

    raw, corrected = [], []
    for _ in range(n_trials):
        pe = pe0 + rng.normal(0, sigma_pos_m, n_panos)
        pn = pn0 + rng.normal(0, sigma_pos_m, n_panos)
        true_bearing = np.degrees(np.arctan2(obj[0] - pe0, obj[1] - pn0)) % 360.0
        bearing = true_bearing + rng.normal(0, sigma_bearing_deg, n_panos)
        i = n_panos // 2
        m = np.ones(n_panos, bool)
        m[i] = False
        try:
            pt, cov = triangulate(pe[m], pn[m], bearing[m])
        except np.linalg.LinAlgError:
            continue
        de, dn = pt[0] - pe[i], pt[1] - pn[i]
        r = float(np.hypot(de, dn))
        if not np.isfinite(r) or r <= 0:
            continue
        ux, uy = de / r, dn / r
        var_cross = uy ** 2 * cov[0, 0] - 2 * ux * uy * cov[0, 1] + ux ** 2 * cov[1, 1]
        # the unweighted covariance is in "perpendicular-miss variance" units; scale it
        scale = (r_true * np.radians(sigma_bearing_deg)) ** 2 + sigma_pos_m ** 2
        sigma_cross = np.sqrt(max(var_cross * scale, 0.0) + sigma_pos_m ** 2)
        raw.append(r)
        corrected.append(r - float(jensen_bias_m(np.array([r]), np.array([sigma_cross]))[0]))
    r_true_i = float(np.hypot(obj[0] - pe0[n_panos // 2], obj[1] - pn0[n_panos // 2]))
    return {
        "n_trials": len(raw),
        "r_true_m": round(r_true_i, 4),
        "mean_raw_m": round(float(np.mean(raw)), 4),
        "mean_corrected_m": round(float(np.mean(corrected)), 4),
        "bias_raw_m": round(float(np.mean(raw)) - r_true_i, 4),
        "bias_corrected_m": round(float(np.mean(corrected)) - r_true_i, 4),
        "median_raw_m": round(float(np.median(raw)), 4),
        "median_corrected_m": round(float(np.median(corrected)), 4),
    }


def parametric_bootstrap_bias(f: pd.DataFrame, sigma_bearing_deg: float,
                              sigma_pos_m: float, height_m: float = 2.35,
                              n_rep: int = 3, seed: int = SEED,
                              sigma_gate_m: float = SIGMA_R_GATE_M) -> dict:
    """End-to-end bias check on the *real* site geometry, not a synthetic street.

    The synthetic check in :func:`monte_carlo_bias_check` uses one idealised configuration.
    This one takes every real site, treats its leave-one-out solution as the truth, plants
    an object there, and regenerates the whole observation set — bearings from the true
    geometry corrupted at the measured ``sigma_bearing_deg``, panorama positions corrupted
    at the measured ``sigma_pos_m``, depressions implied by a *known* camera ``height_m`` —
    then runs the identical pipeline and asks what height comes back.

    The ratio ``recovered / height_m`` is the multiplicative bias of the whole method under
    this run's own noise and geometry. It is the number that says whether a measured height
    above the shipped 2.34 m is the camera or the estimator.
    """
    rng = np.random.default_rng(seed)
    base = f[usable(f, sigma_gate_m)].copy()
    npano = base.groupby("site_id")["pano_id"].transform("size")
    base = base[npano >= MIN_PANOS_FOR_LOO]
    if base.empty:
        return {"n_sites": 0}

    # Truth: each site's object planted at that site's leave-one-out consensus, with the
    # panoramas exactly where they really are. Vectorised over every site at once — the
    # per-site loop this replaces cost minutes per run on the larger cities.
    te = base.groupby("site_id")["loo_e"].transform("median").to_numpy()
    tn = base.groupby("site_id")["loo_n"].transform("median").to_numpy()
    pe0, pn0 = base["pano_e"].to_numpy(), base["pano_n"].to_numpy()
    r_true = np.hypot(te - pe0, tn - pn0)
    good = np.isfinite(r_true) & (r_true >= MIN_RANGE_M)
    # a site is usable only if every one of its members is
    ok_site = pd.Series(good, index=base.index).groupby(base["site_id"]).transform("all")
    base, r_true = base[ok_site.to_numpy()], r_true[ok_site.to_numpy()]
    te, tn = te[ok_site.to_numpy()], tn[ok_site.to_numpy()]
    pe0, pn0 = base["pano_e"].to_numpy(), base["pano_n"].to_numpy()
    if base.empty:
        return {"n_sites": 0}
    true_bearing = np.degrees(np.arctan2(te - pe0, tn - pn0)) % 360.0
    n_sites = int(base["site_id"].nunique())
    m = len(base)

    recovered, recovered_raw = [], []
    for _ in range(n_rep):
        # the depression a camera at exactly height_m would see at the true range
        dep = np.degrees(np.arctan2(height_m, r_true))
        sim = pd.DataFrame({
            "site_id": base["site_id"].to_numpy(),
            "pano_id": base["pano_id"].to_numpy(),
            "pano_e": pe0 + rng.normal(0, sigma_pos_m, m),
            "pano_n": pn0 + rng.normal(0, sigma_pos_m, m),
            "bearing_deg": true_bearing + rng.normal(0, sigma_bearing_deg, m),
            "dep_deg": dep,
            # exactly what the real frame carries — the 2.6 m cotangent range, not r_true —
            # so the weights below are the same function of the observables as site_frame's
            "range_m": COT_CAMERA_HEIGHT / np.tan(np.radians(dep)),
        })
        sim["w"] = 1.0 / ((sim["range_m"] * np.radians(sigma_bearing_deg)) ** 2
                          + sigma_pos_m ** 2)
        sol = _loo_solve(sim, "w")
        sim = pd.concat([sim, sol], axis=1)
        de, dn = sim["loo_e"] - sim["pano_e"], sim["loo_n"] - sim["pano_n"]
        sim["r_tri_raw"] = np.hypot(de, dn)
        theta = np.radians(sim["bearing_deg"].to_numpy())
        sim["forward_m"] = de * np.sin(theta) + dn * np.cos(theta)
        rr = sim["r_tri_raw"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            uex, uen = np.where(rr > 0, de / rr, 0.0), np.where(rr > 0, dn / rr, 0.0)
        vc = (uen ** 2 * sim["cov_xx"] - 2 * uex * uen * sim["cov_xy"]
              + uex ** 2 * sim["cov_yy"])
        va = (uex ** 2 * sim["cov_xx"] + 2 * uex * uen * sim["cov_xy"]
              + uen ** 2 * sim["cov_yy"])
        sim["sigma_cross_m"] = np.sqrt(np.maximum(vc, 0) + sigma_pos_m ** 2)
        sim["sigma_r_m"] = np.sqrt(np.maximum(va, 0) + sigma_pos_m ** 2)
        sim["jensen_m"] = jensen_bias_m(rr, sim["sigma_cross_m"].to_numpy())
        sim["r_tri"] = sim["r_tri_raw"] - sim["jensen_m"]
        sim["implied_height_m"] = sim["r_tri"] * np.tan(np.radians(sim["dep_deg"]))
        keep = ((sim["forward_m"] > 0) & sim["r_tri"].notna()
                & (sim["r_tri"] > MIN_RANGE_M) & (sim["r_tri"] < MAX_RANGE_M)
                & (sim["sigma_r_m"] <= sigma_gate_m))
        if keep.sum() > 30:
            recovered.append(float(np.median(sim.loc[keep, "implied_height_m"])))
            raw_h = sim.loc[keep, "r_tri_raw"] * np.tan(np.radians(sim.loc[keep, "dep_deg"]))
            recovered_raw.append(float(np.median(raw_h)))
    if not recovered:
        return {"n_sites": n_sites}
    corr = np.array(recovered)
    raw = np.array(recovered_raw)
    return {
        "n_sites": n_sites,
        "n_rep": len(corr),
        "planted_height_m": height_m,
        "recovered_height_m": round(float(np.mean(corr)), 4),
        "recovered_height_uncorrected_m": round(float(np.mean(raw)), 4),
        "bias_factor": round(float(np.mean(corr)) / height_m, 4),
        "bias_factor_uncorrected": round(float(np.mean(raw)) / height_m, 4),
    }


# ======================================================================================
# Absolute scale
# ======================================================================================

def _cluster_bootstrap_ci(values: np.ndarray, groups: np.ndarray, stat=np.median,
                          n_boot: int = 400, seed: int = SEED) -> tuple[float, float]:
    """Percentile CI resampling *sites*, not members — members of a site share a point."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    starts = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="left")
    ends = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="right")
    idx_by_group = [order[s:e] for s, e in zip(starts, ends)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_group[p] for p in pick])
        out.append(stat(values[sel]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def implied_height(f: pd.DataFrame, sigma_gate_m: float = SIGMA_R_GATE_M,
                   n_boot: int = 400, seed: int = SEED) -> dict:
    """The headline: the camera height the bearings imply, with no depth data.

    ``h = r_tri * tan(depression)``. If the flat-ground cotangent is the right shape, this
    is a constant; its value is the *effective* height for the detector's click convention
    (rig height minus however far above ground contact the detection centroid sits), which
    is the same estimand ``modern_truth.implied_heights`` reports for human clicks.
    """
    d = f[usable(f, sigma_gate_m)]
    if len(d) < 30:
        return {"n": int(len(d)), "median_m": None}
    h = d["implied_height_m"].to_numpy()
    lo, hi = _cluster_bootstrap_ci(h, d["site_id"].to_numpy(), np.median, n_boot, seed)
    ratio = (d["r_tri"] / d["range_m"]).to_numpy()
    return {
        "n": int(len(d)),
        "n_sites": int(d["site_id"].nunique()),
        "median_m": round(float(np.median(h)), 4),
        "ci95_m": [round(lo, 4), round(hi, 4)],
        "p25_m": round(float(np.percentile(h, 25)), 4),
        "p75_m": round(float(np.percentile(h, 75)), 4),
        "median_ratio_to_2p6": round(float(np.median(ratio)), 4),
        "median_uncorrected_m": round(
            float(np.median(d["r_tri_raw"] * np.tan(np.radians(d["dep_deg"])))), 4),
        "median_sigma_r_m": round(float(np.median(d["sigma_r_m"])), 4),
    }


def _refine_argmin(ks: np.ndarray, losses: np.ndarray, i: int) -> float:
    """Vertex of the three-point parabola through a discrete argmin.

    The placement scatter is smooth and locally quadratic in ``k`` (every placement is
    linear in it), so the continuum minimum is recovered to far below the grid pitch.
    Without this, the estimate — and, worse, every bootstrap replicate — snaps to its
    grid, and a percentile interval of snapped values is quantised to the pitch (0.005 in
    ``k`` is 13 mm of height): bend's interval printed as a width-zero band that excluded
    its own point estimate. Falls back to the grid point at a sweep edge or on a locally
    non-convex triple, where a vertex is undefined.
    """
    if not 0 < i < len(ks) - 1:
        return float(ks[i])
    denom = float(losses[i - 1] - 2.0 * losses[i] + losses[i + 1])
    if denom <= 0:
        return float(ks[i])
    return float(ks[i] + 0.5 * (ks[i + 1] - ks[i])
                 * (losses[i - 1] - losses[i + 1]) / denom)


def fit_model_scale(f: pd.DataFrame, sigma_gate_m: float = SIGMA_R_GATE_M,
                    lo: float = 0.70, hi: float = 1.30, step: float = 0.002,
                    loss: str = "L1", n_boot: int = 0, seed: int = SEED) -> dict:
    """The camera height as a *global scale*, fitted on multi-view placement agreement.

    A second, more efficient route to the same quantity as :func:`implied_height`, and the
    one that makes the identifiability argument plainest. Each member places the object at
    ``pano + k * (2.6 / tan(depression)) * unit_bearing``; the members of a site agree only
    for the right ``k``, because scaling every range slides each placement along a
    *different* ray. So the within-site scatter is a quadratic in ``k`` with one minimum,
    and that minimum is an absolute camera height.

    This is exactly the information Stage 3 had and could not use: its three diagnostics
    (``rms_over_range``, ``range_slope``, ``height_slope``) are each deliberately
    normalised to be scale-invariant, so they discard the one degree of freedom measured
    here. Nothing about the consensus had to change — only the decision to read the raw
    scatter rather than a ratio of it.

    Preferred over the per-member median because it never divides by a noisy triangulated
    range: ``k`` is estimated once from all members jointly.
    """
    d = f[usable(f, sigma_gate_m)]
    if len(d) < 100:
        return {"n": int(len(d)), "height_m": None}
    th = np.radians(d["bearing_deg"].to_numpy())
    ux, uy = np.sin(th), np.cos(th)
    pe, pn = d["pano_e"].to_numpy(), d["pano_n"].to_numpy()
    r = d["range_m"].to_numpy()
    # integer-code the sites once: the scatter is evaluated a few hundred times per fit and
    # a few hundred more per bootstrap resample, so bincount beats a pandas groupby by
    # orders of magnitude here.
    _, g = np.unique(d["site_id"].to_numpy(), return_inverse=True)

    def scatter(k: float, idx=None) -> float:
        if idx is None:
            gi, pxi, pyi = g, pe + k * r * ux, pn + k * r * uy
        else:
            gi = idx[1]
            s = idx[0]
            pxi = pe[s] + k * r[s] * ux[s]
            pyi = pn[s] + k * r[s] * uy[s]
        n = np.bincount(gi)
        cx = np.bincount(gi, pxi) / n
        cy = np.bincount(gi, pyi) / n
        e = np.hypot(pxi - cx[gi], pyi - cy[gi])
        return float(np.mean(e)) if loss == "L1" else float(np.mean(e ** 2))

    ks = np.arange(lo, hi + 1e-9, step)
    losses = np.array([scatter(k) for k in ks])
    i_best = int(np.argmin(losses))
    k_best = _refine_argmin(ks, losses, i_best)
    out = {
        "n": int(len(d)), "n_sites": int(d["site_id"].nunique()), "loss": loss,
        "k": round(k_best, 4),
        # a minimum pinned to the sweep boundary is a clamp, not an estimate
        "at_grid_edge": bool(i_best in (0, len(ks) - 1)),
        "height_m": round(k_best * COT_CAMERA_HEIGHT, 4),
        "scatter_at_best_m": round(float(scatter(k_best)), 4),
        "scatter_at_2p6_m": round(float(scatter(1.0)), 4),
    }
    if n_boot:
        # Resample *sites*, not members: members of one site share a physical object, and
        # the resampled copies must stay distinct groups or the scatter collapses.
        rng = np.random.default_rng(seed)
        n_sites = int(g.max()) + 1
        order = np.argsort(g, kind="stable")
        starts = np.searchsorted(g[order], np.arange(n_sites), side="left")
        ends = np.searchsorted(g[order], np.arange(n_sites), side="right")
        idx_by = [order[a:b] for a, b in zip(starts, ends)]
        # a coarser grid for the interval is fine because the parabolic vertex refinement
        # below recovers the continuum minimum: the grid pitch never reaches the interval
        ks_b = np.arange(lo, hi + 1e-9, max(step, 0.005))
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, n_sites, n_sites)
            sel = np.concatenate([idx_by[p] for p in pick])
            newg = np.repeat(np.arange(n_sites), [len(idx_by[p]) for p in pick])
            bl = np.array([scatter(k, (sel, newg)) for k in ks_b])
            boots.append(_refine_argmin(ks_b, bl, int(np.argmin(bl)))
                         * COT_CAMERA_HEIGHT)
        out["ci95_m"] = [round(float(np.percentile(boots, 2.5)), 4),
                         round(float(np.percentile(boots, 97.5)), 4)]
    return out


def implied_height_by(f: pd.DataFrame, col: str, bins=None,
                      sigma_gate_m: float = SIGMA_R_GATE_M, min_n: int = 60) -> dict:
    """Implied height stratified by a column (or by binned numeric column).

    The two strata that matter scientifically:

    - **by range** — a wrong camera height is a pure scale error, constant in range; the
      fuse gate's 8 m match tolerance is a much larger fraction of a short range than a
      long one, so a selection artefact would show up here as a trend and a real height
      would not.
    - **by depression** — the flat-ground assumption's own test. Curvature here is the
      signature of the ground plane failing, not of the height being wrong.
    """
    d = f[usable(f, sigma_gate_m)].copy()
    if bins is not None:
        d["_k"] = pd.cut(d[col], bins=bins)
    else:
        d["_k"] = d[col]
    out = {}
    for key, g in d.groupby("_k", observed=True):
        if len(g) < min_n:
            continue
        out[str(key)] = {
            "n": int(len(g)),
            "median_m": round(float(np.median(g["implied_height_m"])), 4),
            "median_r_tri_m": round(float(np.median(g["r_tri"])), 3),
        }
    return out


def selection_probe(f: pd.DataFrame, sigma_gate_m: float = SIGMA_R_GATE_M) -> dict:
    """Was the site population selected for consistency with the assumed 2.6 m height?

    The auto-labeler fused every run at 2.6 m and dropped clusters whose members disagreed
    (``residual_per_dof_max`` 3.0, ``max_match_m`` 8 m). If the true height were lower,
    every range is inflated by the same factor, and the *disagreement* that gate sees grows
    with how much the members' ranges differ. So the gate can only have selected on the
    height where the within-site range spread is large.

    Stratifying by that spread therefore separates the two explanations: a real camera
    height is flat across the strata, while a number manufactured by the gate rises toward
    2.6 m as the spread grows. The low-spread stratum is the one to believe.
    """
    d = f[usable(f, sigma_gate_m)].copy()
    if len(d) < 200:
        return {"n": int(len(d))}
    qs = np.quantile(d["site_range_spread_m"], [0, 0.25, 0.5, 0.75, 1.0])
    d["_k"] = pd.cut(d["site_range_spread_m"], bins=np.unique(qs),
                     include_lowest=True, duplicates="drop")
    out = {}
    for key, g in d.groupby("_k", observed=True):
        if len(g) < 60:
            continue
        out[str(key)] = {
            "n": int(len(g)),
            "median_spread_m": round(float(np.median(g["site_range_spread_m"])), 3),
            "implied_height_m": round(float(np.median(g["implied_height_m"])), 4),
        }
    return out


def _wrap180(x):
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def tilt_probe(run: str, f: pd.DataFrame, data_dir: Path = DATA_DIR,
               dep_bins=(0, 6, 9, 12, 16, 22, 90), min_n: int = 60) -> dict:
    """Does the panoramas' uncorrected camera tilt explain the depression trend?

    The auto-labeler fused every run with ``apply_pose: false``, so the depression it
    stores is measured from the *image* mid-row rather than the true horizon. A tilted rig
    displaces the horizon by a sinusoid in relative azimuth,
    ``pitch cos(phi) + roll sin(phi)``, and an uncorrected angular offset of that kind bites
    hardest at small depression — which is the shape the implied height actually shows.
    It is therefore the first hypothesis to test, and cheap: ``camera_pitch``/``camera_roll``
    are committed for every GSV panorama (Mapillary carries none, itself worth recording).

    All four sign conventions are tried because GSV's is not documented here; the flatness
    of the implied height across depression bins picks the winner. If the correction were
    right, one convention would flatten the trend. Reported whichever way it comes out.
    """
    panos = mf.load_panos(run, data_dir)
    if "camera_pitch" not in panos or panos["camera_pitch"].notna().mean() < 0.5:
        return {"available": False, "reason": "no camera_pitch/camera_roll for this run"}
    p = panos[["pano_id", "camera_pitch", "camera_roll", "camera_heading"]].copy()
    p["camera_pitch"] = _wrap180(p["camera_pitch"])
    p["camera_roll"] = _wrap180(p["camera_roll"])
    d = f[usable(f)].merge(p, on="pano_id", how="left")
    d = d[d["camera_pitch"].notna() & d["camera_heading"].notna()]
    if len(d) < 200:
        return {"available": False, "reason": "too few panoramas with pose"}

    def flatness(dep, h):
        k = pd.cut(dep, bins=dep_bins)
        med = h.groupby(k, observed=True).median()
        n = h.groupby(k, observed=True).size()
        med = med[n >= min_n]
        return (round(float(med.max() - med.min()), 4) if len(med) > 1 else None,
                {str(a): round(float(b), 4) for a, b in med.items()})

    spread0, bins0 = flatness(d["dep_deg"], d["implied_height_m"])
    out = {"available": True, "n": int(len(d)),
           "uncorrected": {"median_m": round(float(d["implied_height_m"].median()), 4),
                           "depression_spread_m": spread0, "bins": bins0},
           "corrected": {}}
    phi = np.radians(_wrap180(d["bearing_deg"] - d["camera_heading"]))
    for sp_ in (1, -1):
        for sr in (1, -1):
            dep = d["dep_deg"] + (sp_ * d["camera_pitch"] * np.cos(phi)
                                  + sr * d["camera_roll"] * np.sin(phi))
            h = d["r_tri"] * np.tan(np.radians(dep))
            spread, bins = flatness(dep, h)
            out["corrected"][f"pitch{sp_:+d}_roll{sr:+d}"] = {
                "median_m": round(float(h.median()), 4),
                "depression_spread_m": spread, "bins": bins}
    best = min((v["depression_spread_m"], k) for k, v in out["corrected"].items()
               if v["depression_spread_m"] is not None)
    out["verdict"] = (
        "tilt explains it" if best[0] < spread0 else
        "tilt does NOT explain it: every sign convention leaves the depression trend "
        "larger than the uncorrected data, so the recorded pose is adding noise rather "
        "than removing a systematic")
    out["best_corrected"] = {"convention": best[1], "depression_spread_m": best[0]}
    return out


def scale_sensitivity(f: pd.DataFrame, gates=(0.75, 1.0, 1.5, 2.5, 4.0),
                      min_panos=(3, 4, 5)) -> dict:
    """Does the headline move when the conditioning gate or the site size moves?

    A real camera height is a property of the rig and must be flat across both sweeps. A
    number manufactured by ill-conditioned geometry or by the fuse gate's selection would
    drift with them.
    """
    out = {"by_sigma_gate": {}, "by_min_panos": {}}
    for gate in gates:
        r = implied_height(f, sigma_gate_m=gate, n_boot=120)
        out["by_sigma_gate"][str(gate)] = {"n": r["n"], "median_m": r.get("median_m")}
    for k in min_panos:
        sub = f[f["n_panos"] >= k]
        r = implied_height(sub, n_boot=120)
        out["by_min_panos"][str(k)] = {"n": r["n"], "median_m": r.get("median_m")}
    return out


# ======================================================================================
# Scoring the estimators against triangulated truth
# ======================================================================================

def load_shipped_blend(data_dir: Path = DATA_DIR) -> dict:
    """``final_coefficients`` from the modern-truth close-out — the shipped estimator."""
    with open(Path(data_dir) / "modern-truth-summary.json", encoding="utf-8") as fh:
        return json.load(fh)["final_coefficients"]


def _blend(dep_deg: np.ndarray, height_m: float, blend_deg: float,
           cap_m: float = 50.0) -> np.ndarray:
    """The shipped C1 blend: cotangent above ``blend_deg``, matched tangent line below."""
    dep = np.asarray(dep_deg, dtype=float)
    a = float(blend_deg)
    a_rad = np.radians(a)
    cot_at_a = height_m / np.tan(a_rad)
    tail_slope = height_m * (np.pi / 180.0) / np.sin(a_rad) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cot = np.where(dep > 0, height_m / np.tan(np.radians(np.maximum(dep, 1e-9))), np.inf)
    return np.where(dep >= a, cot,
                    np.clip(cot_at_a + tail_slope * (a - np.maximum(dep, 0.0)), 0.0, cap_m))


def model_predictions(f: pd.DataFrame, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Every candidate's predicted range for each member, keyed to the same rows."""
    dep = f["dep_deg"].to_numpy()
    heights = f["pano_height"].to_numpy()
    preds = mf.model_distances(dep, heights)
    shipped = load_shipped_blend(data_dir)["params"]
    out = pd.DataFrame({
        "deployed_linear": preds["A_status_quo"],
        "normalized_linear": preds["B_normalized"],
        "cotangent_2p6": preds["C_cotangent"],
        "era_blend": preds["D_blend"],
        "shipped_blend": _blend(dep, shipped["height_m"], shipped["blend_deg"]),
    }, index=f.index)
    return out


MODEL_ORDER = ["deployed_linear", "normalized_linear", "cotangent_2p6",
               "era_blend", "shipped_blend"]


def score_models(f: pd.DataFrame, data_dir: Path = DATA_DIR,
                 sigma_gate_m: float = SIGMA_R_GATE_M) -> dict:
    """Median/p90/signed error and range slope of each model against triangulated truth.

    This is the comparison Stage 3 could not make: an *absolute* score, on imagery no
    candidate was fit on, with a truth that shares none of the candidates' assumptions.
    """
    ok = usable(f, sigma_gate_m)
    d = f[ok]
    preds = model_predictions(d, data_dir)
    truth = d["r_tri"].to_numpy()
    out = {"n": int(len(d)), "n_sites": int(d["site_id"].nunique()), "models": {}}
    for key in MODEL_ORDER:
        err = preds[key].to_numpy() - truth
        finite = np.isfinite(err)
        e, t = err[finite], truth[finite]
        # slope of signed error on truth: negative = compression (far under-shot)
        slope = float(np.polyfit(t, e, 1)[0]) if len(t) > 10 else float("nan")
        out["models"][key] = {
            "median_abs_m": round(float(np.median(np.abs(e))), 4),
            "signed_median_m": round(float(np.median(e)), 4),
            "p90_abs_m": round(float(np.percentile(np.abs(e), 90)), 4),
            "range_slope": round(slope, 4),
            "n": int(len(e)),
        }
    return out


def score_by_range(f: pd.DataFrame, data_dir: Path = DATA_DIR,
                   bins=(0, 5, 10, 15, 20, 60),
                   sigma_gate_m: float = SIGMA_R_GATE_M) -> dict:
    """Signed error by true (triangulated) range — where compression is visible."""
    d = f[usable(f, sigma_gate_m)].copy()
    preds = model_predictions(d, data_dir)
    d = pd.concat([d, preds], axis=1)
    d["_bin"] = pd.cut(d["r_tri"], bins=bins)
    out = {}
    for key, g in d.groupby("_bin", observed=True):
        if len(g) < 40:
            continue
        out[str(key)] = {"n": int(len(g)), "models": {
            m: round(float(np.median(g[m] - g["r_tri"])), 4) for m in MODEL_ORDER}}
    return out


# ======================================================================================
# Applicability and precision of triangulation as an estimator
# ======================================================================================

def split_half_precision(f: pd.DataFrame, sigma_gate_m: float = SIGMA_R_GATE_M,
                         seed: int = SEED) -> dict:
    """Model-free precision: triangulate each site from two disjoint halves and compare.

    Nothing here is compared against a model, so the spread is the estimator's own
    reproducibility — the honest answer to "how good is a triangulated position?".
    Needs >= 4 panoramas so each half has >= 2.
    """
    rng = np.random.default_rng(seed)
    d = f[usable(f, sigma_gate_m)]
    d = d[d.groupby("site_id")["pano_id"].transform("size") >= 4]
    # A few thousand sites already pins the median to well under a centimetre; the larger
    # runs carry ten thousand and the loop is the slow part of the whole build.
    sids = d["site_id"].unique()
    if len(sids) > 4000:
        d = d[d["site_id"].isin(rng.choice(sids, 4000, replace=False))]
    seps = []
    for sid, g in d.groupby("site_id"):
        g = g.drop_duplicates("pano_id")
        if len(g) < 4:
            continue
        idx = rng.permutation(len(g))
        a, b = idx[: len(g) // 2], idx[len(g) // 2:]
        try:
            pa, _ = triangulate(g["pano_e"].to_numpy()[a], g["pano_n"].to_numpy()[a],
                                g["bearing_deg"].to_numpy()[a])
            pb, _ = triangulate(g["pano_e"].to_numpy()[b], g["pano_n"].to_numpy()[b],
                                g["bearing_deg"].to_numpy()[b])
        except np.linalg.LinAlgError:
            continue
        sep = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        if np.isfinite(sep) and sep < 100:
            seps.append(sep)
    if not seps:
        return {"n_sites": 0}
    seps = np.array(seps)
    return {
        "n_sites": int(len(seps)),
        "median_half_separation_m": round(float(np.median(seps)), 4),
        # two independent halves, so each half's own error is the separation / sqrt(2);
        # a full-site estimate uses twice the rays again, hence the further sqrt(2).
        "implied_position_sigma_m": round(float(np.median(seps)) / 2.0, 4),
        "p90_half_separation_m": round(float(np.percentile(seps, 90)), 4),
    }


def applicability(run: str, data_dir: Path = DATA_DIR,
                  frame: pd.DataFrame | None = None,
                  sigma_gate_m: float = SIGMA_R_GATE_M) -> dict:
    """What fraction of objects triangulation can serve, and how well-conditioned they are.

    Scope item 3 of #7: this is a subset estimator and the report has to say how big the
    subset is before anyone reads the accuracy.
    """
    base = mf.member_frame(run, data_dir) if frame is None else frame
    per_site_panos = base.drop_duplicates(["site_id", "pano_id"]).groupby("site_id").size()
    f = site_frame(run, data_dir, frame=base)
    n_all = int(len(per_site_panos))
    out = {
        "n_sites_multi_member": n_all,
        "n_sites_2plus_panos": int((per_site_panos >= 2).sum()),
        "n_sites_3plus_panos": int((per_site_panos >= 3).sum()),
        "frac_sites_3plus_panos": round(float((per_site_panos >= 3).mean()), 4),
        "median_panos_per_site": float(per_site_panos.median()),
        # the population the report's dataset table describes — sites that can support
        # leave-one-out — not the 2+ population `median_panos_per_site` summarises
        "median_panos_per_site_3plus": float(
            per_site_panos[per_site_panos >= MIN_PANOS_FOR_LOO].median()),
    }
    if f.empty:
        return out
    # intersection-angle conditioning: each site's best-separated ray *pair* (error scales
    # ~ 1/sin of this); every member of a site carries the same site-level value
    ang = f.groupby("site_id")["bearing_deg"].transform(
        lambda b: _max_intersection_angle(b.to_numpy()))
    out["intersection_angle_deg"] = {
        "p10": round(float(np.percentile(ang, 10)), 2),
        "median": round(float(np.percentile(ang, 50)), 2),
        "p90": round(float(np.percentile(ang, 90)), 2),
        "frac_below_20deg": round(float((ang < 20).mean()), 4),
    }
    return out


def _max_intersection_angle(bearings: np.ndarray) -> float:
    """Largest pairwise ray intersection angle in [0, 90]."""
    b = np.mod(np.asarray(bearings, dtype=float), 360.0)
    d = np.abs(b[:, None] - b[None, :])
    d = np.minimum(d, 360.0 - d)
    d = np.minimum(d, 180.0 - d)
    return float(d.max()) if len(b) > 1 else 0.0
