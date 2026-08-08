"""The distance half of the refit: geometry-shaped, horizon-saturating candidates (issue #3).

The 2021 estimator's distance half is six per-zoom linear coefficients on
``sv_image_y + canvas_y``. But after #5 the click's exact depression angle below the horizon is
available in closed form from ``(canvas_x, canvas_y, heading, pitch, zoom)``, and on flat ground
the geometry is a cotangent: ``dist = camera_height / tan(depression)``. This module fits the
candidate ladder from the issue — A (status quo), zero-parameter anchor, C (cotangent, fitted
camera height), D (horizon-saturating cotangent, three forms), E (monotone isotonic, exported as
piecewise-linear knots) — under both OLS and L1 loss, and scores every rung on the exact
published train/test split. Candidate B (the pano-height term, #4765) is handled separately in
``candidate_b_checks``/``apply_path_check`` because it only exists on the six modern cities.

Conventions, decided deliberately (issue #3 rider 2 and amendment 2):

- **Depression angle** comes from ``pov_inversion.exact_depression_deg`` — the same projection
  call that closed the heading half in #5. It needs no pano metadata, so every rung below runs
  on all seven cities including DC. Zoom is inside the projection; any zoom effect that survives
  is behavioral, not geometric (``zoom_residual_check`` measures it).
- **Fit space**: the geometry rungs (C, D_soft) are linear in *disparity* (1/distance), where
  pixel click noise is approximately Gaussian; they are fit there. D_floor and D_blend profile
  their one shape hyper-parameter on the *meters-space train loss* (squared for ols, absolute
  for l1) with the inner camera-height fit in disparity space — the hyper-parameter exists to
  fix meters-space behavior, so it is chosen on the metric we report.
- **Loss**: every fitted rung has an ``_ols`` and an ``_l1`` variant, so "geometry vs form" and
  "loss vs metric" stay unconfounded (the published headline metric is a median). Linear L1
  fits use statsmodels QuantReg at tau=0.5; the one-parameter disparity fits use the exact
  weighted-median LAD solution, which is deterministic and closed-form.
- **Boundedness**: the status quo's one virtue is that a linear form is bounded, and the D
  family is here to keep it. Every rung ends in ``clip(0, DIST_CAP_M)``, but that clip is the
  training-domain cap, not saturation — what matters is each form's *structural* bound, the
  largest distance it can return anywhere in the depression domain (``structural_max_m``
  measures it, ``bounds`` in the summary publishes it, and a findings test asserts every one):
  D_floor at ``h/tan(dep_min)``, D_blend at its horizon value (its linear tail is evaluated at
  ``max(dep, 0)``, so a click above the horizon gets the horizon's answer rather than a
  runaway extrapolation), E at its first knot. D_soft's ``1/c0`` bound is nominally structural
  but the ``c0 >= 1/DIST_CAP_M`` constraint is *active* in all four variants, so in practice it
  saturates at the cap — one of the reasons it loses to floor/blend.
- **Scoring geodesy**: lat/lng error for every rung uses turf-style ``spherical_dest`` — what
  production ``toLatLng`` actually runs — with the heading half held identical across rungs:
  the era-faithful exact inversion plus the one train-fitted constant from #5 (the stored
  2017-2020 targets carry that +0.72 deg lookup bias, so scoring without it would penalize
  every candidate for being right; the constant must NOT be applied to post-evolution-179
  data). The unmodified est7 pipeline is kept as the continuity row (1.4621 m).
- **Selection**: the recommended candidate is chosen among the D-family variants on the
  *train* median absolute distance error, before test scoring is looked at
  (``choose_candidate`` records the full train table). All variants' test numbers are
  published either way.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import isotonic_regression, lsq_linear
from statsmodels.regression.quantile_regression import QuantReg

from label_latlng_estimation import (
    MAX_DIST_FROM_PANO, _design, _named, _ols, haversine_m, latlng_error_m,
    predict_dist_heading, spherical_dest,
)
from pov_inversion import (
    CANVAS_H, CANVAS_W, era_heading_diff, exact_depression_deg, pov_if_centered,
)

DIST_CAP_M = float(MAX_DIST_FROM_PANO)  # the training-domain bound every rung respects
SV_PX_PER_DEG = 6656.0 / 180.0  # sv_image_y is stored in a fixed 13312x6656 frame (see report)
LABEL_TYPES = ["CurbRamp", "NoCurbRamp", "NoSidewalk", "Obstacle", "Occlusion", "Other",
               "SurfaceProblem"]
MODERN_CITIES = ["seattle", "newberg", "columbus", "spgg", "cdmx", "pittsburgh"]

# Every scored column, in display order. est7/est7_sph are handled by score_rungs itself.
KEYS = ["est7", "est7_sph", "A_ols", "A_l1", "anchor", "anchor_served",
        "C_ols", "C_l1", "C_type_ols", "C_type_l1",
        "D_floor_ols", "D_floor_l1", "D_floor_type_ols", "D_floor_type_l1",
        "D_blend_ols", "D_blend_l1", "D_blend_type_ols", "D_blend_type_l1",
        "D_soft_ols", "D_soft_l1", "D_soft_type_ols", "D_soft_type_l1",
        "E_ols", "E_l1"]
HEADLINE_KEYS = ["est7", "A_ols", "anchor", "C_l1", "D_soft_l1", "E_l1"]


def add_depression(df: pd.DataFrame) -> pd.DataFrame:
    """Depression angle of the click below the horizon (degrees, positive down), from the same
    exact projection that settled the heading half in #5. Split-independent click geometry, so
    call it once on the cleaned frame before splitting."""
    return df.assign(depression_deg=exact_depression_deg(df))


def heading_for_scoring(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, float]:
    """The heading half every distance rung is paired with: era-faithful exact inversion plus
    the one train-fitted constant (score_heading_swap's era_cal recipe, refit here). Every
    cleaned row is pre-cutoff by construction, so the legacy +0.72 deg target bias applies to
    the whole split; the constant is a property of the 2017-2020 ground truth and must never
    ship in production coefficients."""
    delta = float(np.mean(train["heading_diff"].to_numpy(float) - era_heading_diff(train)))
    return era_heading_diff(test) + delta, delta


# ------------------------------------------------------------------------------- fit helpers

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def _lad_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Exact L1 solution of y ~ slope * x (no intercept): the |x|-weighted median of y/x.

    sum|y - s*x| = sum |x| * |y/x - s| over the rows with x != 0, and the x == 0 rows add a
    constant independent of s, so they drop out. Every call site here passes a non-negative
    tan(depression), but the |x| weighting keeps the solver correct rather than merely lucky."""
    ok = x != 0
    return _weighted_median(y[ok] / x[ok], np.abs(x[ok]))


def _quantreg(X: np.ndarray, y: np.ndarray, q: float = 0.5) -> np.ndarray:
    return np.asarray(QuantReg(y, X).fit(q=q, max_iter=5000).params, float)


def _disparity(dist: np.ndarray, cap: float = 1.0) -> tuple[np.ndarray, int]:
    """1/distance with sub-1/cap rows capped (a handful of sub-meter rows would otherwise
    dominate an OLS fit in disparity space). Returns (disparity, n_capped)."""
    d = np.asarray(dist, float)
    return 1.0 / np.maximum(d, 1.0 / cap), int((d < 1.0 / cap).sum())


def _tan_dep(dep_deg: np.ndarray, floor_deg: float = 0.0) -> np.ndarray:
    return np.tan(np.radians(np.maximum(np.asarray(dep_deg, float), floor_deg)))


# ------------------------------------------------------------------------------------- rungs

def fit_linear(train: pd.DataFrame, loss: str) -> dict:
    """Rung A: the status quo distance half — per-zoom ``pano_dist ~ sv_image_y + canvas_y``.
    With loss='ols' this reproduces est7's distance coefficients exactly."""
    terms = ["sv_image_y", "canvas_y"]
    coef = []
    for z in (1, 2, 3):
        sub = train[train["zoom"] == z]
        X, y = _design(sub, terms), sub["pano_dist"].to_numpy(float)
        c = _ols(X, y) if loss == "ols" else _quantreg(X, y)
        coef.append(_named(c, terms))
    return {"form": "per_zoom_linear", "loss": loss, "n_params": 9, "coef": coef}


def fit_anchor(height_m: float = 2.6) -> dict:
    """The zero-fitted-parameter anchor: ``dist = height / tan(depression)``, capped. This is
    essentially Google's fromContainerPixelToLatLng with the ecosystem's 2.6 m camera height."""
    return {"form": "cotangent", "loss": "none", "n_params": 0, "height_m": float(height_m)}


def fit_anchor_served(data_dir: str) -> dict:
    """Anchor variant using the per-pano camera height GSV itself serves, where the #4 pilot
    measured one (the pinned-2.5 rows are the payload default, not a measurement — excluded).
    Scored only on its subsample; 2.6 m everywhere else would just duplicate ``anchor``."""
    panos = pd.read_csv(os.path.join(data_dir, "depth-pilot-panos.csv.gz"))
    pinned = panos["ground_d_exactly_2p5"].astype("boolean").fillna(True).astype(bool)
    ok = panos["ground_height_m"].notna() & ~pinned
    heights = dict(zip(panos.loc[ok, "pano_id"], panos.loc[ok, "ground_height_m"].astype(float)))
    return {"form": "cotangent_served", "loss": "none", "n_params": 0,
            "heights_by_pano": heights, "n_panos": len(heights)}


def fit_cotangent(train: pd.DataFrame, loss: str, per_type: bool = False) -> dict:
    """Rung C: ``dist = h / tan(depression)`` with h fitted in disparity space, optionally one
    h per label type (amendment 4: the ground-plane assumption is differently wrong per type).
    Clicks at or above the horizon cannot inform a ground-plane height and are excluded from
    the fit (counted); at predict time they saturate to the cap.

    The per-type variants also record ``height_fallback_m`` — the pooled height, fitted the
    same way on every usable row. It is what ``predict_dist`` uses for a label type this
    2017-2020 population never contained, so a production caller cannot silently get NaN out
    of a type the table has no row for (the seven fitted types cover every row scored here,
    so the fallback contributes nothing to any number in the summary)."""
    dep = train["depression_deg"].to_numpy(float)
    disp, n_capped = _disparity(train["pano_dist"].to_numpy(float))
    t = _tan_dep(dep)
    usable = dep > 0

    def _h(mask: np.ndarray) -> float:
        x, y = t[mask], disp[mask]
        s = float(x @ y / (x @ x)) if loss == "ols" else _lad_origin_slope(x, y)
        return 1.0 / s

    out = {"form": "cotangent", "loss": loss,
           "n_excluded_above_horizon": int((~usable).sum()), "n_disparity_capped": n_capped}
    if per_type:
        types = train["label_type"].to_numpy(str)
        out["height_by_type_m"] = {lt: _h(usable & (types == lt)) for lt in LABEL_TYPES}
        out["height_fallback_m"] = _h(usable)
        out["n_params"] = len(out["height_by_type_m"])
    else:
        out["height_m"] = _h(usable)
        out["n_params"] = 1
    return out


def _heights_in_disparity(t: np.ndarray, disp: np.ndarray, types: np.ndarray | None,
                          loss: str) -> tuple[float | dict, np.ndarray, float]:
    """Inner camera-height fit for the cotangent family: disparity ~ (1/h) * t, either one
    global h or one per label type. Returns (h params, per-row 1/h, pooled fallback h)."""
    def _h(x: np.ndarray, y: np.ndarray) -> float:
        return float(x @ y / (x @ x)) if loss == "ols" else _lad_origin_slope(x, y)

    pooled = _h(t, disp)
    if types is None:
        return 1.0 / pooled, np.full(len(t), pooled), 1.0 / pooled
    s_row = np.empty(len(t))
    hs = {}
    for lt in LABEL_TYPES:
        m = types == lt
        s = _h(t[m], disp[m])
        hs[lt] = 1.0 / s
        s_row[m] = s
    return hs, s_row, 1.0 / pooled


def fit_floor(train: pd.DataFrame, loss: str, per_type: bool = False) -> dict:
    """Rung D_floor: ``dist = h / tan(max(depression, dep_min))``. The floor is profiled on a
    0.5..12 deg grid against the full meters-space train loss (mean squared for ols, mean
    absolute for l1 — a hyper-parameter that only touches the rows below the floor cannot be
    profiled on a median), h refit in disparity space at each candidate. per_type combines
    this with amendment 4's per-label-type camera height (shared floor)."""
    dep = train["depression_deg"].to_numpy(float)
    dist = train["pano_dist"].to_numpy(float)
    disp, _ = _disparity(dist)
    types = train["label_type"].to_numpy(str) if per_type else None
    best = None
    for dep_min in np.arange(0.5, 12.01, 0.25):
        t = _tan_dep(dep, dep_min)
        h, s_row, fallback = _heights_in_disparity(t, disp, types, loss)
        pred = np.clip(1.0 / (s_row * t), 0.0, DIST_CAP_M)
        err = pred - dist
        train_loss = float(np.mean(err ** 2)) if loss == "ols" else float(np.mean(np.abs(err)))
        if best is None or train_loss < best[0]:
            best = (train_loss, h, float(dep_min), fallback)
    out = {"form": "floor", "loss": loss, "dep_min_deg": best[2]}
    if per_type:
        out.update(height_by_type_m=best[1], height_fallback_m=best[3],
                   n_params=1 + len(LABEL_TYPES))
    else:
        out.update(height_m=best[1], n_params=2)
    return out


def fit_blend(train: pd.DataFrame, loss: str, per_type: bool = False) -> dict:
    """Rung D_blend: cotangent above a blend angle, C1-continuous linear-in-depression below it
    (value and slope matched), capped. h is fit in disparity space on the cotangent region;
    the blend angle is profiled on the full meters-space train loss (mean squared / mean
    absolute — see fit_floor for why not a median). per_type gives each label type its own
    camera height (shared blend angle)."""
    dep = train["depression_deg"].to_numpy(float)
    dist = train["pano_dist"].to_numpy(float)
    disp, _ = _disparity(dist)
    types = train["label_type"].to_numpy(str) if per_type else None
    t = _tan_dep(dep)
    best = None
    for a in np.arange(1.0, 12.01, 0.25):
        m = dep >= a
        h, _, fallback = _heights_in_disparity(t[m], disp[m],
                                               types[m] if per_type else None, loss)
        params = ({"height_by_type_m": h, "height_fallback_m": fallback} if per_type
                  else {"height_m": h}) | {"blend_deg": float(a)}
        pred = _predict_blend(params, dep, types)
        err = pred - dist
        train_loss = float(np.mean(err ** 2)) if loss == "ols" else float(np.mean(np.abs(err)))
        if best is None or train_loss < best[0]:
            best = (train_loss, params)
    n_params = (1 + len(LABEL_TYPES)) if per_type else 2
    return {"form": "blend", "loss": loss, "n_params": n_params, **best[1]}


def fit_softcap(train: pd.DataFrame, loss: str, per_type: bool = False) -> dict:
    """Rung D_soft, the presumptive production form: linear in disparity space,
    ``1/dist = c0 + c1 * tan(max(depression, 0))`` with ``c0 >= 1/DIST_CAP_M``, so predicted
    distance is bounded at ``1/c0`` *by construction* — the saturation is the intercept, not a
    bolt-on. Optionally a per-label-type slope (shared cap). L1 uses QuantReg; if its
    unconstrained intercept falls below 1/cap it is projected there and the slopes refit
    (recorded in ``projected``). Measured outcome: that bound is active in all four variants,
    so this rung's "structural" bound is the 50 m cap itself — see ``bounds`` in the summary.

    Both losses hold the slopes at ``c1 >= 0`` (a negative slope would invert the form —
    farther clicks answered as nearer — and ``clip`` would hide it): OLS gets it from
    ``lsq_linear``'s bounds, L1 by flooring after the fit, recorded in ``c1_floored``."""
    dep = train["depression_deg"].to_numpy(float)
    disp, n_capped = _disparity(train["pano_dist"].to_numpy(float))
    t = _tan_dep(dep)
    c0_min = 1.0 / DIST_CAP_M
    types = train["label_type"].to_numpy(str)
    masks = [types == lt for lt in LABEL_TYPES] if per_type else None
    out = {"form": "softcap", "loss": loss, "n_disparity_capped": n_capped,
           "projected": False, "c1_floored": False}

    X = (np.column_stack([np.ones(len(t))] + [m * t for m in masks]) if per_type
         else np.column_stack([np.ones(len(t)), t]))
    if loss == "ols":
        lo = np.r_[c0_min, np.zeros(X.shape[1] - 1)]
        c = lsq_linear(X, disp, bounds=(lo, np.full(X.shape[1], np.inf))).x
    else:
        c = _quantreg(X, disp)
        if c[0] < c0_min:  # project onto the boundary, then refit the slopes there
            out["projected"] = True
            resid = disp - c0_min
            c = (np.r_[c0_min, [_lad_origin_slope(t[m], resid[m]) for m in masks]] if per_type
                 else np.array([c0_min, _lad_origin_slope(t, resid)]))
        if (c[1:] < 0).any():
            out["c1_floored"] = True
            c = np.r_[c[0], np.maximum(c[1:], 0.0)]

    c0 = float(max(c[0], c0_min))
    if per_type:
        # Fallback slope for a label type this population never contained: the pooled fit at
        # the same intercept, under the same loss (see fit_cotangent on why a fallback exists).
        resid = disp - c0
        pooled = (float(t @ resid / (t @ t)) if loss == "ols" else _lad_origin_slope(t, resid))
        out.update(c0=c0, c1_by_type={lt: float(v) for lt, v in zip(LABEL_TYPES, c[1:])},
                   c1_fallback=max(float(pooled), 0.0), n_params=1 + len(LABEL_TYPES))
    else:
        out.update(c0=c0, c1=float(c[1]), n_params=2)
    return out


def fit_isotonic(train: pd.DataFrame, loss: str, max_knots: int = 24) -> dict:
    """Rung E: shape-constrained nonparametric — monotone-nonincreasing distance vs depression,
    compressed to <= max_knots piecewise-linear knots (JS-viable; bounded by the first knot).
    loss='ols' is the exact L2 isotonic fit on all rows; loss='l1' runs the pool-adjacent-
    violators step on 200 quantile-bin medians (count-weighted) — a documented approximation,
    since exact L1 isotonic is not in scipy."""
    dep = train["depression_deg"].to_numpy(float)
    dist = train["pano_dist"].to_numpy(float)
    order = np.argsort(dep, kind="stable")
    if loss == "ols":
        x, fitted = dep[order], isotonic_regression(dist[order], increasing=False).x
    else:
        bins = pd.qcut(dep, 200, duplicates="drop")
        g = pd.DataFrame({"dep": dep, "dist": dist}).groupby(bins, observed=True)
        med, cnt = g["dist"].median(), g["dist"].size()
        x = g["dep"].median().to_numpy(float)
        fitted = isotonic_regression(med.to_numpy(float), weights=cnt.to_numpy(float),
                                     increasing=False).x
    qs = np.unique(np.r_[0.0, np.geomspace(0.001, 0.5, max_knots // 2 - 1),
                         1 - np.geomspace(0.001, 0.5, max_knots // 2 - 1)[::-1], 1.0])
    kx = np.unique(np.quantile(x, qs))
    ky = np.minimum.accumulate(np.interp(kx, x, fitted))
    return {"form": "isotonic", "loss": loss, "n_params": len(kx),
            "knots_dep_deg": [float(v) for v in kx], "knots_dist_m": [float(v) for v in ky]}


def fit_all_rungs(train: pd.DataFrame, models: dict, data_dir: str) -> dict:
    """The full fits dict keyed like KEYS (minus est7/est7_sph, which reuse ``models``).
    A_ols is asserted identical to est7's distance half — same design, same loss."""
    fits: dict = {}
    for loss in ("ols", "l1"):
        fits[f"A_{loss}"] = fit_linear(train, loss)
        fits[f"C_{loss}"] = fit_cotangent(train, loss)
        fits[f"C_type_{loss}"] = fit_cotangent(train, loss, per_type=True)
        fits[f"D_floor_{loss}"] = fit_floor(train, loss)
        fits[f"D_floor_type_{loss}"] = fit_floor(train, loss, per_type=True)
        fits[f"D_blend_{loss}"] = fit_blend(train, loss)
        fits[f"D_blend_type_{loss}"] = fit_blend(train, loss, per_type=True)
        fits[f"D_soft_{loss}"] = fit_softcap(train, loss)
        fits[f"D_soft_type_{loss}"] = fit_softcap(train, loss, per_type=True)
        fits[f"E_{loss}"] = fit_isotonic(train, loss)
    fits["anchor"] = fit_anchor()
    fits["anchor_served"] = fit_anchor_served(data_dir)
    for z in (1, 2, 3):
        for term, got in fits["A_ols"]["coef"][z - 1].items():
            want = models["est7"]["dist"][z - 1][term]
            assert abs(got - want) < 1e-9, f"A_ols must equal est7's distance half ({term}, z{z})"
    return fits


# -------------------------------------------------------------------------------- prediction

def _heights(params: dict, df: pd.DataFrame) -> np.ndarray:
    """Per-row camera height from a per-type table, pooled fallback for unseen types."""
    return (df["label_type"].map(params["height_by_type_m"])
            .fillna(params["height_fallback_m"]).to_numpy(float))


def _predict_blend(params: dict, dep: np.ndarray, types: np.ndarray | None = None) -> np.ndarray:
    """The C1 blend: cotangent above ``blend_deg``, its matched tangent line below.

    The tail is evaluated at ``max(dep, 0)``, so the form's largest possible answer is its
    value at the horizon (~28 m) rather than a linear runaway to the 50 m cap. Clicks above the
    horizon are unplaceable by definition — 0.16% of the test split — and clamping them to the
    horizon's answer is what makes this rung's boundedness structural rather than a claim about
    where the data happened to sit (see ``structural_max_m``)."""
    if "height_by_type_m" in params:
        h = (pd.Series(types).map(params["height_by_type_m"])
             .fillna(params["height_fallback_m"]).to_numpy(float))
    else:
        h = params["height_m"]
    a = params["blend_deg"]
    a_rad = np.radians(a)
    v = h / np.tan(a_rad)
    slope = -h * (np.pi / 180.0) / np.sin(a_rad) ** 2  # d/d(dep_deg) of h*cot(dep) at a
    cot = h / _tan_dep(dep, 1e-9)
    tail = v + slope * (np.maximum(dep, 0.0) - a)
    return np.clip(np.where(dep >= a, cot, tail), 0.0, DIST_CAP_M)


def predict_dist(params: dict, df: pd.DataFrame) -> np.ndarray:
    """Distance predictions for one fitted rung. Every form ends bounded in [0, DIST_CAP_M]
    (``structural_max_m`` reports the tighter bound each form actually holds); anchor_served
    returns NaN outside its served-height subsample, which is the one deliberate NaN here.

    Per-type variants fall back to the pooled parameter for a label type the fit never saw, so
    an unfamiliar type gets the population's answer rather than a silent NaN."""
    form = params["form"]
    dep = df["depression_deg"].to_numpy(float) if "depression_deg" in df else None
    typed = "height_by_type_m" in params

    if form == "per_zoom_linear":
        d = np.empty(len(df))
        zoom = df["zoom"].to_numpy()
        for z in (1, 2, 3):
            i = zoom == z
            c = params["coef"][z - 1]
            d[i] = (c["(Intercept)"] + c["sv_image_y"] * df["sv_image_y"].to_numpy(float)[i]
                    + c["canvas_y"] * df["canvas_y"].to_numpy(float)[i])
        return np.clip(d, 0.0, DIST_CAP_M)
    if form == "cotangent":
        h = _heights(params, df) if typed else params["height_m"]
        return np.clip(h / _tan_dep(dep, 1e-9), 0.0, DIST_CAP_M)
    if form == "cotangent_served":
        h = df["pano_id"].map(params["heights_by_pano"]).to_numpy(float)
        return np.clip(h / _tan_dep(dep, 1e-9), 0.0, DIST_CAP_M)
    if form == "floor":
        h = _heights(params, df) if typed else params["height_m"]
        return np.clip(h / _tan_dep(dep, params["dep_min_deg"]), 0.0, DIST_CAP_M)
    if form == "blend":
        return _predict_blend(params, dep, df["label_type"].to_numpy(str) if typed else None)
    if form == "softcap":
        if "c1_by_type" in params:
            c1 = (df["label_type"].map(params["c1_by_type"])
                  .fillna(params["c1_fallback"]).to_numpy(float))
        else:
            c1 = params["c1"]
        return np.clip(1.0 / (params["c0"] + c1 * _tan_dep(dep)), 0.0, DIST_CAP_M)
    if form == "isotonic":
        return np.clip(np.interp(dep, params["knots_dep_deg"], params["knots_dist_m"]),
                       0.0, DIST_CAP_M)
    raise ValueError(f"unknown form {form}")


def structural_max_m(params: dict, n: int = 72001) -> float | None:
    """The largest distance a fitted rung can EVER return, swept over the full depression
    domain and every label type — the bound the report's near-horizon claims mean.

    This is what ``near_horizon_table``'s per-bin ``dist_pred_max_m`` is *not*: that column is
    whatever the thin near-horizon test population happened to sample, so a form can look
    bounded there and still run to the cap on a click the split didn't contain. Returns None
    for the status-quo linear rung, whose answer is a function of pixels, not of depression."""
    if params["form"] == "per_zoom_linear":
        return None
    dep = np.linspace(-90.0, 90.0, n)
    frame = pd.DataFrame({"depression_deg": dep, "zoom": 1})
    if params["form"] == "cotangent_served":
        frame["pano_id"] = max(params["heights_by_pano"],
                               key=params["heights_by_pano"].get)  # the tallest served camera
        return float(np.nanmax(predict_dist(params, frame)))
    worst = 0.0
    for lt in LABEL_TYPES + ["__unseen__"]:  # the fallback path is part of the bound
        frame["label_type"] = lt
        worst = max(worst, float(np.nanmax(predict_dist(params, frame))))
    return worst


# ----------------------------------------------------------------------------------- scoring

def score_rungs(fits: dict, models: dict, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per-label error columns for every rung, the heading half held identical (era_cal) and the
    destination spherical, so differences between rungs are the distance half alone. est7 keeps
    its own fitted heading and the legacy ellipsoidal destination — the 1.4621 m continuity row
    — and est7_sph re-scores the same distances under the shared heading/geodesy, isolating
    what the scoring-convention switch itself is worth."""
    heading_pred, delta = heading_for_scoring(train, test)
    dist7, head7 = predict_dist_heading(models, test, "est7")

    out = pd.DataFrame({
        "label_id": test["label_id"].to_numpy(),
        "city": test["city"].to_numpy(),
        "zoom": test["zoom"].to_numpy(),
        "label_type": test["label_type"].to_numpy(str),
        "pano_id": test["pano_id"].to_numpy(str),
        "pano_dist": test["pano_dist"].to_numpy(float),
        "depression_deg": test["depression_deg"].to_numpy(float),
        "photographer_pitch": test["photographer_pitch"].to_numpy(float),
        "heading_diff": test["heading_diff"].to_numpy(float),
    })

    def latlng_err(d: np.ndarray) -> np.ndarray:
        lng_e, lat_e = spherical_dest(test["pano_lng"], test["pano_lat"],
                                      test["heading"].to_numpy(float) + heading_pred, d)
        return haversine_m(test["lng"], test["lat"], lng_e, lat_e)

    out["dist_pred_est7"] = dist7
    out["error_est7"] = latlng_error_m(test, dist7, head7, crude=False)
    out["dist_error_est7"] = np.abs(out["pano_dist"] - dist7)
    out["dist_pred_est7_sph"] = dist7
    out["error_est7_sph"] = latlng_err(dist7)
    out["dist_error_est7_sph"] = out["dist_error_est7"]

    for key, params in fits.items():
        d = predict_dist(params, test)
        out[f"dist_pred_{key}"] = d
        out[f"error_{key}"] = latlng_err(d)
        out[f"dist_error_{key}"] = np.abs(out["pano_dist"] - d)
    out.attrs["era_cal_delta_deg"] = delta
    return out


def _metrics(scored: pd.DataFrame, key: str) -> dict:
    err, derr = scored[f"error_{key}"], scored[f"dist_error_{key}"]
    ok = err.notna()
    return {"n": int(ok.sum()),
            "latlng_median_m": float(err[ok].median()),
            "latlng_p90_m": float(err[ok].quantile(0.9)),
            "dist_median_m": float(derr[ok].median()),
            "dist_p90_m": float(derr[ok].quantile(0.9))}


EST7_N_PARAMS = 15  # 3 zooms x (intercept + sv_image_y + canvas_y) = 9 distance, + 3 x
#                     (intercept + canvas_x) = 6 heading. The A_* rows report the distance
#                     half alone (9) on the same convention: every coefficient counted.


def matrix_table(scored: pd.DataFrame, fits: dict) -> dict:
    n_params = {"est7": EST7_N_PARAMS, "est7_sph": EST7_N_PARAMS,
                **{k: v["n_params"] for k, v in fits.items()}}
    return {k: {**_metrics(scored, k), "n_params": n_params[k]}
            for k in KEYS if f"error_{k}" in scored.columns}


def choose_candidate(fits: dict, train: pd.DataFrame) -> dict:
    """The honesty gate: the recommended saturating form is picked among the D-family variants
    on the TRAIN median absolute distance error (the loss-aligned version of the published
    metric), before any test number is consulted."""
    cands = [k for k in fits if k.startswith("D_")]
    dist = train["pano_dist"].to_numpy(float)
    meds = {k: float(np.median(np.abs(predict_dist(fits[k], train) - dist))) for k in cands}
    chosen = min(meds, key=meds.get)
    return {"rung": chosen, "chosen_on": "train median |dist error| (m)",
            "train_median_abs_dist_error_m": meds}


def near_horizon_table(scored: pd.DataFrame, keys: list[str] | None = None) -> list[dict]:
    """Error and worst-case behavior where the ground-plane geometry degenerates. The test
    population here is thin (GSV clicks rarely sit near the horizon), so the error medians are
    not the load-bearing column — but neither is ``dist_pred_max_m`` below, which is only the
    largest answer these particular rows drew. The structural bound each form holds everywhere
    is ``structural_max_m`` (summary key ``bounds``); read the two together."""
    keys = keys or HEADLINE_KEYS
    bins = pd.cut(scored["depression_deg"], [-np.inf, 0.0, 2.0, 5.0], right=True)
    rows = []
    for interval, g in scored.groupby(bins, observed=True):
        rows.append({"bin_deg": str(interval), "n": int(len(g)), "per_rung": {
            k: {"latlng_median_m": float(g[f"error_{k}"].median()),
                "latlng_p95_m": float(g[f"error_{k}"].quantile(0.95)),
                "latlng_max_m": float(g[f"error_{k}"].max()),
                "dist_pred_max_m": float(g[f"dist_pred_{k}"].max())} for k in keys}})
    return rows


def by_group_table(scored: pd.DataFrame, col: str, keys: list[str] | None = None) -> dict:
    keys = keys or HEADLINE_KEYS
    return {str(gv): {"n": int(len(g)), **{
        k: {"latlng_median_m": float(g[f"error_{k}"].median()),
            "dist_median_m": float(g[f"dist_error_{k}"].median())} for k in keys}}
        for gv, g in scored.groupby(col)}


def zoom_residual_check(scored: pd.DataFrame, key: str) -> dict:
    """Per-zoom median signed distance residual of one rung. The exact projection consumed
    zoom, so any structure left here is behavioral (what users choose to label per zoom), not
    geometric."""
    return {int(z): {"n": int(len(g)),
                     "signed_median_m": float((g[f"dist_pred_{key}"] - g["pano_dist"]).median())}
            for z, g in scored.groupby("zoom")}


def noise_sweep(fits: dict, models: dict, train: pd.DataFrame, test: pd.DataFrame,
                keys: list[str] | None = None, sigmas=(2.0, 5.0, 10.0), n_draws: int = 5,
                seed: int = 666, extra_predictors: dict | None = None) -> dict:
    """The gsv-location-extraction-analysis objection made quantitative: perturb the click by
    Gaussian pixel noise, re-derive every click-dependent input (canvas_y, depression via the
    exact projection, sv_image_y via the fixed-frame px/deg scale), and measure how each rung's
    error distribution degrades. The heading half stays at the unperturbed era_cal prediction:
    the sweep isolates the distance half's noise response (the heading half's was #5's §2).

    ``extra_predictors`` maps a key to a callable ``frame -> predicted distance (m)``, so a
    model that lives outside this module (#6's GBM benchmark) is scored on the *exact same*
    perturbed clicks instead of re-implementing the sweep and hoping the two copies stay in
    step. The rng is consumed identically whatever the key set is — one canvas_x draw and one
    canvas_y draw per repetition, sigma-major — so adding predictors cannot move the #3 rows.
    """
    extra = dict(extra_predictors or {})
    keys = list(keys or HEADLINE_KEYS) + [k for k in extra if k not in (keys or HEADLINE_KEYS)]
    heading_pred, _ = heading_for_scoring(train, test)
    rng = np.random.default_rng(seed)
    n = len(test)
    _, pitch0 = pov_if_centered(test["canvas_x"], test["canvas_y"],
                                test["heading"], test["pitch"], test["zoom"])

    def errors(frame: pd.DataFrame) -> dict:
        res = {}
        for k in keys:
            if k in extra:
                d = extra[k](frame)
            elif k == "est7":
                d, _h = predict_dist_heading(models, frame, "est7")
            else:
                d = predict_dist(fits[k], frame)
            lng_e, lat_e = spherical_dest(frame["pano_lng"], frame["pano_lat"],
                                          frame["heading"].to_numpy(float) + heading_pred, d)
            e = haversine_m(frame["lng"], frame["lat"], lng_e, lat_e)
            res[k] = (float(np.median(e)), float(np.quantile(e, 0.9)))
        return res

    base = errors(test)
    out = {"sigmas_px": list(sigmas), "n_draws": n_draws, "baseline_median_m":
           {k: base[k][0] for k in keys}, "per_rung": {k: {} for k in keys}}
    for sigma in sigmas:
        acc = {k: np.zeros(2) for k in keys}
        for _ in range(n_draws):
            frame = test.copy()
            frame["canvas_x"] = np.clip(test["canvas_x"].to_numpy(float)
                                        + rng.normal(0, sigma, n), 0, CANVAS_W)
            frame["canvas_y"] = np.clip(test["canvas_y"].to_numpy(float)
                                        + rng.normal(0, sigma, n), 0, CANVAS_H)
            _, pitch_p = pov_if_centered(frame["canvas_x"], frame["canvas_y"],
                                         frame["heading"], frame["pitch"], frame["zoom"])
            frame["depression_deg"] = -pitch_p
            frame["sv_image_y"] = (test["sv_image_y"].to_numpy(float)
                                   + (pitch_p - pitch0) * SV_PX_PER_DEG)
            e = errors(frame)
            for k in keys:
                acc[k] += np.array(e[k])
        for k in keys:
            med, p90 = acc[k] / n_draws
            out["per_rung"][k][str(sigma)] = {"delta_median_m": med - base[k][0],
                                              "delta_p90_m": p90 - base[k][1]}
    return out


# ----------------------------------------------------------------- candidate B and #4765

def fixed_frame_check(cleaned: pd.DataFrame, min_dep_deg: float = 2.0) -> dict:
    """Which frame is sv_image_y in? If it scaled with the real panorama raster, the
    px-implied depression (sv_image_y / SV_PX_PER_DEG) would overshoot the exact projection by
    height/6656 (1.23x for 8192-px panos). Measured: the implied/exact ratio is ~0.97-0.98 for
    BOTH height groups in every modern city — sv_image_y lives in a fixed 13312x6656 frame, so
    the 2021 fit has no pixel-scale defect and #4765's is an apply-path bug (see
    apply_path_check)."""
    sub = cleaned[cleaned["pano_height"].isin([6656, 8192])
                  & (cleaned["depression_deg"] > min_dep_deg)]
    implied = -sub["sv_image_y"].to_numpy(float) / SV_PX_PER_DEG
    ratio = pd.Series(implied / sub["depression_deg"].to_numpy(float), index=sub.index)
    out: dict = {"min_depression_deg": min_dep_deg, "cities": {}}
    for city, g in sub.groupby("city"):
        row = {int(h): {"n": int(len(gg)), "ratio_median": float(ratio.loc[gg.index].median())}
               for h, gg in g.groupby(g["pano_height"].astype(int))}
        if 6656 in row and 8192 in row:
            row["ratio_8192_over_6656"] = row[8192]["ratio_median"] / row[6656]["ratio_median"]
        out["cities"][city] = row
    pooled = {int(h): float(ratio.loc[g.index].median())
              for h, g in sub.groupby(sub["pano_height"].astype(int))}
    out["pooled"] = {"ratio_median_6656": pooled[6656], "ratio_median_8192": pooled[8192],
                     "ratio_8192_over_6656": pooled[8192] / pooled[6656],
                     "if_real_pixel_frame": 8192 / 6656}
    return out


def _city_dummies(df: pd.DataFrame,
                  cities: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """City fixed effects, first city as the reference. ``cities`` MUST be passed when building
    a test design: derived per frame, a city missing from one side would silently shift every
    dummy column's meaning between fit and predict (they match on this data — the point is that
    nothing enforced it)."""
    cities = cities if cities is not None else [c for c in MODERN_CITIES
                                                if (df["city"] == c).any()]
    return (np.column_stack([(df["city"] == c).to_numpy(float) for c in cities[1:]]),
            [f"city_{c}" for c in cities[1:]])


def candidate_b_checks(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """B(i): does a pano-height term improve the FIXED-FRAME predictor? Four forms per zoom on
    the modern-city subset (city fixed effects throughout, A refit on the same subset as the
    apples-to-apples reference):

    - A_sub:      sv_image_y + canvas_y                     (reference)
    - B_norm:     sv_image_y * 6656/pano_height + canvas_y  (#4765's one-line fix as written)
    - B_log:      A_sub + log(pano_height/6656)             (a level shift by height group)
    - B_interact: A_sub + sv_image_y*(6656/pano_height - 1) (the sharp test: under the
      normalization hypothesis this coefficient equals the sv slope; in a fixed frame it is 0)

    Reported per zoom: test-subset median |dist error| per form (both losses), and the
    B_interact coefficient with its OLS standard error against the sv slope it would have to
    match for #4765-as-written to hold on the 2021 predictor.

    Restricted to the two GSV panorama heights (6656/8192) that carry the population, matching
    fixed_frame_check and apply_path_check. The 294 cleaned rows at 1664 px are a third rig
    whose 4x normalization factor would be the tail wagging a 0.2% dog; the count they would
    have contributed is reported as ``n_height_1664_excluded``."""
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df[df["pano_height"].isin([6656, 8192])].copy()
        h = d["pano_height"].astype(float)
        d["sv_norm"] = d["sv_image_y"] * (6656.0 / h)
        d["log_h"] = np.log(h / 6656.0)
        d["sv_interact"] = d["sv_image_y"] * (6656.0 / h - 1.0)
        return d

    tr, te = prep(train), prep(test)
    forms = {"A_sub": ["sv_image_y", "canvas_y"],
             "B_norm": ["sv_norm", "canvas_y"],
             "B_log": ["sv_image_y", "canvas_y", "log_h"],
             "B_interact": ["sv_image_y", "canvas_y", "sv_interact"]}
    out: dict = {"n_train": len(tr), "n_test": len(te),
                 "n_height_1664_excluded": int((train["pano_height"] == 1664).sum()
                                               + (test["pano_height"] == 1664).sum())}

    for z in (1, 2, 3):
        trz, tez = tr[tr["zoom"] == z], te[te["zoom"] == z]
        cities = [c for c in MODERN_CITIES if (trz["city"] == c).any()]
        fe_tr, fe_names = _city_dummies(trz, cities)
        fe_te, _ = _city_dummies(tez, cities)  # same columns, same reference, by construction
        row: dict = {"n_train": len(trz), "n_test": len(tez)}
        for name, terms in forms.items():
            Xtr = np.column_stack([np.ones(len(trz))] +
                                  [trz[t].to_numpy(float) for t in terms] + [fe_tr])
            Xte = np.column_stack([np.ones(len(tez))] +
                                  [tez[t].to_numpy(float) for t in terms] + [fe_te])
            y = trz["pano_dist"].to_numpy(float)
            fit = sm.OLS(y, Xtr).fit()
            errs = {}
            for loss in ("ols", "l1"):
                c = np.asarray(fit.params) if loss == "ols" else _quantreg(Xtr, y)
                pred = np.clip(Xte @ c, 0.0, DIST_CAP_M)
                errs[loss] = float(np.median(np.abs(pred - tez["pano_dist"].to_numpy(float))))
            named = dict(zip(["(Intercept)"] + terms + fe_names, map(float, fit.params)))
            se = dict(zip(["(Intercept)"] + terms + fe_names, map(float, fit.bse)))
            row[name] = {"test_dist_median_m": errs,
                         "coef": {t: named[t] for t in terms},
                         "se": {t: se[t] for t in terms}}
        row["interact_vs_norm_prediction"] = {
            "interact_coef": row["B_interact"]["coef"]["sv_interact"],
            "interact_se": row["B_interact"]["se"]["sv_interact"],
            "sv_slope_it_would_have_to_match": row["B_interact"]["coef"]["sv_image_y"]}
        out[f"zoom{z}"] = row
    return out


def apply_path_check(models: dict, test: pd.DataFrame) -> dict:
    """B(ii): the defect #4765 actually describes, quantified on ground truth for the first
    time. Production toLatLng feeds the label's REAL-raster pixel offset from the horizon
    (pano_y) into coefficients that were fit on the fixed 13312x6656 frame. On the test rows
    that carry evolution-179's current_pano_y and a pano height, predict est7's distance both
    ways — raw pixel offset vs the offset normalized by 6656/pano_height — and report the
    error and the signed bias per height group.

    What it measures (see the report): the raw slot overshoots the fixed-frame sv_image_y by
    26% on 8192-px panos, but est7's coefficients are themselves biased ~+1.7 m too-far on
    that subgroup (fig6's separation), so on GSV the two errors largely CANCEL. Normalizing
    the pixels without refitting the coefficients removes the compensation and surfaces the
    +1.7 m bias — the one-line #4765 fix as written would make the dominant modern GSV
    population worse. The resolution dependence is real; the fix has to be a refit that has
    no pixel scale at all (the geometry rungs), not a rescaling in front of the old
    coefficients."""
    sub = test[test["current_pano_y"].notna() & test["pano_height"].isin([6656, 8192])].copy()
    h = sub["pano_height"].astype(float).to_numpy()
    raw_offset = h / 2.0 - sub["current_pano_y"].astype(float).to_numpy()
    slots = {"raw": raw_offset, "normalized": raw_offset * (6656.0 / h)}
    # (height/2 - pano_y) shares sv_image_y's sign convention: for 6656-px panos it equals the
    # stored sv_image_y exactly, so the "raw" variant reproduces est7 verbatim on that group
    # and the 8192 group isolates what feeding real-raster pixels into fixed-frame
    # coefficients does.

    out: dict = {"n": len(sub)}
    truth = sub["pano_dist"].to_numpy(float)
    zoom = sub["zoom"].to_numpy()
    for variant, slot in slots.items():
        d = np.empty(len(sub))
        for z in (1, 2, 3):
            i = zoom == z
            c = models["est7"]["dist"][z - 1]
            d[i] = (c["(Intercept)"] + c["sv_image_y"] * slot[i]
                    + c["canvas_y"] * sub["canvas_y"].to_numpy(float)[i])
        d = np.clip(d, 0.0, DIST_CAP_M)
        res = {"dist_median_m": float(np.median(np.abs(d - truth)))}
        for hh in (6656, 8192):
            i = sub["pano_height"].to_numpy() == hh
            res[f"h{hh}"] = {"n": int(i.sum()),
                             "dist_median_m": float(np.median(np.abs(d[i] - truth[i]))),
                             "signed_median_m": float(np.median(d[i] - truth[i]))}
        out[variant] = res
    return out


# ------------------------------------------------------------------------------------ riders

def rider_checks(scored: pd.DataFrame, cleaned: pd.DataFrame, fits: dict, chosen: str,
                 data_dir: str) -> dict:
    """Three small residual checks that cost no fitted parameters (issue #3 amendments):
    photographer_pitch (the DB's one tilt component), the full tilt sinusoid on the 409 panos
    whose yaw/pitch/roll the #9 fetch recorded, and the fitted-vs-served camera height."""
    out: dict = {}

    resid = scored[f"dist_pred_{chosen}"] - scored["pano_dist"]
    pp = scored["photographer_pitch"].to_numpy(float)
    ok = np.isfinite(pp) & np.isfinite(resid)
    out["photographer_pitch"] = {
        "n": int(ok.sum()),
        "pearson_r": float(np.corrcoef(pp[ok], resid[ok])[0, 1]),
        "slope_m_per_deg": float(np.polyfit(pp[ok], resid[ok], 1)[0])}

    # Tilt sinusoid: pano tilt enters depression as pitch*cos(view-yaw) + roll*sin(view-yaw)
    # (small-angle). Convert the chosen rung's residual to an implied depression residual and
    # regress it on that per-row prediction; a coefficient near 1 would mean the recorded
    # tilt explains the residual structure. Uses all cleaned rows on the 409 metadata panos.
    meta = pd.read_csv(os.path.join(data_dir, "depth-validation-panometa.csv.gz"))
    meta["roll_deg"] = np.where(meta["roll_deg"] > 180, meta["roll_deg"] - 360, meta["roll_deg"])
    j = cleaned.merge(meta, on="pano_id", how="inner")
    if len(j):
        h_fit = fits.get("C_l1", {}).get("height_m", 2.6)
        dep_from_truth = np.degrees(np.arctan2(h_fit, j["pano_dist"].to_numpy(float)))
        dep_resid = j["depression_deg"].to_numpy(float) - dep_from_truth
        h_rel = np.radians(j["heading"].to_numpy(float) - j["yaw_deg"].to_numpy(float))
        t = (j["pitch_deg"].to_numpy(float) * np.cos(h_rel)
             + j["roll_deg"].to_numpy(float) * np.sin(h_rel))
        ok = np.isfinite(t) & np.isfinite(dep_resid)
        coef = float(np.polyfit(t[ok], dep_resid[ok], 1)[0]) if ok.sum() > 2 else None
        r = float(np.corrcoef(t[ok], dep_resid[ok])[0, 1]) if ok.sum() > 2 else None
        out["tilt_sinusoid"] = {"n": int(ok.sum()), "n_panos": int(j["pano_id"].nunique()),
                                "coef_deg_per_deg": coef, "pearson_r": r}
    else:
        out["tilt_sinusoid"] = {"n": 0}

    panos = pd.read_csv(os.path.join(data_dir, "depth-pilot-panos.csv.gz"))
    pinned = panos["ground_d_exactly_2p5"].astype("boolean").fillna(True).astype(bool)
    served = panos.loc[panos["ground_height_m"].notna() & ~pinned, "ground_height_m"]
    out["camera_height"] = {"fitted_C_l1_m": fits["C_l1"]["height_m"],
                            "fitted_C_ols_m": fits["C_ols"]["height_m"],
                            "served_median_m_excl_pin": float(served.median()),
                            "n_served": int(len(served))}
    return out


def quantile_bands(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Stage-4 rider: the chosen family refit at tau=0.1/0.9 in disparity space gives each
    label a distance interval for free. Near the horizon the tau=0.1 band runs to the cap —
    'wide interval' is a better answer there than a silently confident point estimate."""
    dep_tr = train["depression_deg"].to_numpy(float)
    disp, _ = _disparity(train["pano_dist"].to_numpy(float))
    X = np.column_stack([np.ones(len(dep_tr)), _tan_dep(dep_tr)])
    t_te = _tan_dep(test["depression_deg"].to_numpy(float))
    out: dict = {}
    bands = {}
    for tau in (0.1, 0.9):
        c = _quantreg(X, disp, q=tau)
        c0 = max(float(c[0]), 1.0 / DIST_CAP_M)
        bands[tau] = np.clip(1.0 / (c0 + float(c[1]) * t_te), 0.0, DIST_CAP_M)
        out[f"tau_{tau}"] = {"c0": c0, "c1": float(c[1])}
    width = np.abs(bands[0.1] - bands[0.9])
    out["interval_width_median_m"] = float(np.median(width))
    out["interval_width_p90_m"] = float(np.quantile(width, 0.9))
    return out


# ----------------------------------------------------------------------------------- summary

def build_summary(scored: pd.DataFrame, fits: dict, chosen: dict, checks: dict,
                  keys: list[str] | None = None,
                  near_horizon_keys: list[str] | None = None) -> dict:
    """Assemble data/distance-refit-summary.json. ``checks`` carries the pieces the runner
    computed: fixed_frame, candidate_b, apply_path, noise, riders, quantiles, meta extras.
    ``keys``/``near_horizon_keys`` select the rungs the by-group and near-horizon tables carry
    (the runner swaps the chosen rung into both); everything else is ladder-wide."""
    params = {}
    for key, p in fits.items():
        q = {k: v for k, v in p.items() if k != "heights_by_pano"}  # 200+ pano ids stay out
        params[key] = q
    matrix = matrix_table(scored, fits)
    summary = {
        "meta": {"n_test": int(len(scored)),
                 "era_cal_delta_deg": float(scored.attrs["era_cal_delta_deg"]),
                 "dist_cap_m": DIST_CAP_M, "chosen": chosen, **checks.get("meta", {})},
        "continuity": {
            "est7_legacy_median_m": matrix["est7"]["latlng_median_m"],
            "est7_spherical_median_m": matrix["est7_sph"]["latlng_median_m"],
            "scoring_convention_delta_m": (matrix["est7_sph"]["latlng_median_m"]
                                           - matrix["est7"]["latlng_median_m"])},
        "matrix": matrix,
        "params": params,
        "bounds": {k: structural_max_m(p) for k, p in fits.items()},
        "by_zoom": by_group_table(scored, "zoom", keys=keys),
        "by_label_type": by_group_table(scored, "label_type", keys=keys),
        "near_horizon": near_horizon_table(scored, keys=near_horizon_keys),
        "zoom_residual_chosen": zoom_residual_check(scored, chosen["rung"]),
        "fixed_frame_check": checks["fixed_frame"],
        "candidate_b": {"fixed_frame_forms": checks["candidate_b"],
                        "apply_path": checks["apply_path"]},
        "noise_sweep": checks["noise"],
        "riders": checks["riders"],
        "quantiles": checks["quantiles"],
    }
    ch = fits[chosen["rung"]]
    summary["era_fit_coefficients"] = {
        "form": ch["form"], "rung": chosen["rung"], "params": params[chosen["rung"]],
        "max_answer_m": structural_max_m(ch),
        "unseen_label_type": "use height_fallback_m (the pooled fit); the seven fitted types "
                             "are every type the 2017-2020 population contains, and a modern "
                             "caller WILL meet others",
        "geodesy": "spherical (turf destination), matching production toLatLng",
        "heading": "exact POV inversion (pov_if_centered), zero parameters; the era_cal "
                   "constant is a property of the 2017-2020 ground truth and must NOT be "
                   "applied to post-evolution-179 data",
        "status": "era fit, final in form: Stage 3 (the Mapillary falsification) certified "
                  "the shape scale-free, and the modern-truth check calibrated the height "
                  "scale to modern measured planes. Production constants live in "
                  "modern-truth-summary.json final_coefficients (one flat height; these "
                  "per-type heights carry the era truth's pinned-plane scale and their "
                  "spread does not replicate on modern truth)",
        "caveats": [
            "truth is model-derived: GSV's terrain model, curb-height bias ~0.48 m systematic",
            "occlusion outliers are clustered, which is why the L1 column exists",
            "open item G: 6.6% of stored distances sampled a rotated depth column (p95 4.1 m)",
            "float32 storage grid: 0.21-0.42 m lat / 0.57-0.80 m lng resolution floor",
            "stored bearings carry +0.72 deg bias, handled at score time only",
            "per-type heights are fitted on GSV car geometry; a non-GSV rig needs a "
            "per-source base height (the falsification's per-sequence recipe)",
        ],
    }
    return summary
