"""Does the #6 ceiling transfer? Score the era-trained GBM on modern measured-plane truth.

The #6 benchmark (reports/2026-08-07-gbm-ceiling.md) measured a 0.40 m gap between the
closed form and a LightGBM given the same inputs, *inside the 2017-2020 truth frame that
also trained it*. Two things landed afterwards that make that gap re-examinable:

- Stage 4 (reports/2026-08-07-modern-truth.md) showed that era-fitted structure does not
  automatically survive a change of truth frame: the era fit's per-type height table --
  worth 4 cm on the era test split -- bought nothing at all on modern truth, and the
  shipped coefficients dropped it for a single flat height.
- The GBM's own report flagged (its section 5) that ``sv_image_y`` is the column the
  2017-2020 client fed into the depth lookup that PRODUCED its target, so some unknown
  share of the ceiling is truth-pipeline structure rather than scene geometry -- and
  called that share "unmeasurable from inside this dataset".

It is measurable from outside it. ``data/modern-truth-labels.csv.gz`` carries every
feature the GBM eats (canvas_x/y, zoom, label_type, pano_y, pano_height, heading, pitch)
on post-2021 rows whose truth is a freshly sampled modern GSV ground plane. Those rows are
disjoint from the era training split by the 2021-01-01 cutoff, so a booster fitted on the
era split is fully held out on them. Nothing here is refitted: the boosters are the #6
boosters, and the only parameter this module ever fits is a single global scale, on a
train half, scored on the disjoint half -- the exact treatment Stage 4 gave the blend.

Two questions, deliberately kept apart, because conflating them would rig the answer:

1. **Raw transfer.** Every era-trained model inherits the era truth's ~16% pinned-plane
   scale, so all of them predict long against modern truth. Comparing the GBM to the era
   blend D on the same modern rows holds that handicap constant and asks only whether the
   GBM's *era* advantage is still there.
2. **Scale-corrected transfer.** Give the GBM the same one-parameter modern recalibration
   the blend got (Stage 4's ``k_rescale`` analogue: one multiplicative factor, fitted on a
   train half of panos, scored on the disjoint half) and compare it against the shipped
   flat blend. This is the question that matters: with both sides modern-calibrated by one
   number, how much of the 0.40 m ceiling is left?

Frame mapping is the one place this could go quietly wrong, so it is derived rather than
guessed and pinned by ``tests/test_gbm_transfer_contract.py``. The era client stored
``sv_image_y`` as a fixed-frame offset from the horizon (negative downward, in the
13312x6656 frame the deployed coefficients were calibrated in), while the modern schema
stores ``pano_y``, an absolute row index in the panorama's real raster. Both encode the
same angle, so they convert exactly:

    sv_image_y = (pano_height/2 - pano_y) * 6656/pano_height = -depression_deg * 6656/180

Measured on the 162,846 era rows that carry a current real-pixel row: the mapped residual
is +15.0 px at 6656 and +14.6 px at 8192 -- the SAME small offset in both height groups
(it is pano re-registration drift between the era panorama and today's), against +140 px
for the unmapped raw offset at 8192. Two height groups agreeing is what makes this the
right conversion rather than a fitted fudge.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import distance_refit as dr
import modern_truth as mt
import run_gbm_ceiling as gc
from pov_inversion import exact_depression_deg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SEED = 666                 # the repo-wide seed; also modern_truth.remedy_check's
CALIBRATION_HEIGHT = 6656  # the fixed frame the era client and the deployed coefficients share
N_BOOT = 2000              # cluster bootstrap resamples (by panorama)
SCALE_MIN_DEP_DEG = 5.0    # remedy_check's implied-height stability floor, reused verbatim


# ------------------------------------------------------------------- population and frame

def load_modern_labels(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """The committed modern-truth rows, unchanged. ``pano_id`` must stay a string: it is
    the bootstrap's cluster key and the split key, and a silent numeric parse would break
    both without erroring."""
    return pd.read_csv(os.path.join(data_dir, "modern-truth-labels.csv.gz"),
                       dtype={"pano_id": str})


def gated_human(labels: pd.DataFrame) -> pd.DataFrame:
    """``modern_truth``'s scored population: gate-passing, human-placed rows.

    Same predicate as ``remedy_check`` and ``final_coefficients`` (``gate_ok & ~is_ai``),
    so this module's rows are the rows Stage 4's numbers were computed on."""
    return labels[labels["gate_ok"] & ~labels["is_ai"]].copy()


def to_era_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the era feature columns the #6 boosters were trained on.

    ``sv_image_y`` is the fixed-frame horizon offset derived in the module docstring, and
    it is the only feature this function has to get right: the booster's second vertical
    input, ``sv_norm``, is derived FROM it inside ``run_gbm_ceiling.build_features``
    (``sv_image_y * 6656/pano_height``, recomputed on every frame the booster ever sees),
    so precomputing a column of that name here would be dead weight that the fitting code
    ignores. That expression is arguably a second normalization of an already-normalized
    column -- the refit's ``fixed_frame_check`` is what established that -- but
    reproducing the feature the model was fitted on is the whole point, so it is left as
    the era code computes it, not corrected.

    ``depression_deg_exact`` is the #5 projection (canvas + POV), which is what the era
    ``gbm_dep_l1`` variant ate; the file's own ``depression_deg`` is the pixel-derived
    one and stays untouched, because every closed form here is scored through it.
    """
    out = df.copy()
    h = out["pano_height"].astype(float)
    out["sv_image_y"] = (h / 2.0 - out["pano_y"].astype(float)) * (CALIBRATION_HEIGHT / h)
    out["depression_deg_exact"] = exact_depression_deg(out)
    return out


def frame_mapping_evidence(df: pd.DataFrame) -> dict:
    """Two independent checks that the mapped column is the angle it claims to be.

    The first is algebraic (the mapping must equal the negated pixel depression scaled by
    the fixed frame's px/deg) and should be exact to floating point. The second is
    physical: the exact #5 projection, computed from canvas and POV without touching
    pano_y at all, must agree -- if the stored pixel and the projection disagreed here,
    the mapping would be reproducing a storage artifact rather than a click angle.
    """
    mapped = df["sv_image_y"].to_numpy(float)
    from_pixel_angle = -df["depression_deg"].to_numpy(float) * CALIBRATION_HEIGHT / 180.0
    from_exact_angle = -df["depression_deg_exact"].to_numpy(float) * CALIBRATION_HEIGHT / 180.0
    d_exact = mapped - from_exact_angle
    return {
        "n": int(len(df)),
        "px_per_deg": CALIBRATION_HEIGHT / 180.0,
        "max_abs_diff_vs_pixel_angle_px": float(np.max(np.abs(mapped - from_pixel_angle))),
        "vs_exact_projection_px": {
            "median": float(np.median(d_exact)),
            "p10": float(np.percentile(d_exact, 10)),
            "p90": float(np.percentile(d_exact, 90)),
        },
    }


ERA_RESIDUAL_HEIGHTS = (6656, 8192)


def era_pixel_residuals(cleaned: pd.DataFrame,
                        heights: tuple[int, ...] = ERA_RESIDUAL_HEIGHTS) -> dict:
    """The third frame-mapping check: the mapping against the era frame's OWN real pixels.

    The two checks above are computed on the modern rows. This one is computed on the era
    rows the boosters are fitted on, and it is the one that discriminates: those rows carry
    both the era ``sv_image_y`` the client stored and a ``current_pano_y`` read from
    today's raster, so the mapping can be checked against a pixel it did not derive from.

    ``mapped_px`` is the residual left by the conversion this module ships; ``raw_px`` is
    the residual left by the discriminating alternative, the unmapped
    ``pano_height/2 - pano_y``. The alternative is indistinguishable at 6656 px -- the two
    are the same expression there, since the frame IS 6656 -- and wrong by ~23% of the
    offset at 8192. Two height groups landing on the SAME small residual is what makes the
    mapping right rather than fitted: that residual is pano re-registration drift between
    the era panorama and today's, not a free parameter.

    Emitted into the summary rather than left in prose because §3 of the report publishes
    these exact figures, and under this repo's archival rule a published number needs an
    artifact behind it. Nearly free: the runner already holds the cleaned frame.
    """
    df = cleaned.dropna(subset=["sv_image_y", "current_pano_y", "pano_height"])
    out = {"n_rows": 0, "by_pano_height": {}}
    for h, g in df[df["pano_height"].isin(list(heights))].groupby("pano_height"):
        pano_y = g["current_pano_y"].astype(float)
        raw = float(h) / 2.0 - pano_y
        mapped = raw * (CALIBRATION_HEIGHT / float(h))
        out["by_pano_height"][str(int(h))] = {
            "n": int(len(g)),
            "mapped_px": float((g["sv_image_y"] - mapped).median()),
            "raw_px": float((g["sv_image_y"] - raw).median()),
        }
        out["n_rows"] += int(len(g))
    out["reading"] = ("the mapping leaves the same small drift in both height groups; the "
                      "unmapped alternative leaves it only at 6656 px, where the two "
                      "expressions coincide, and is off by ~23% of the offset at 8192")
    return out


def support_shift(era: pd.DataFrame, modern: pd.DataFrame,
                  cols=("sv_image_y", "canvas_x", "canvas_y", "zoom", "heading", "pitch",
                        "pano_height", "depression_deg")) -> dict:
    """Where the modern rows sit inside the era training support, feature by feature.

    A booster cannot extrapolate: outside the training range every split has already been
    taken, so the prediction flattens to the edge leaf. ``frac_outside_era_range`` is
    therefore the honest measure of how much of this transfer is interpolation. Reported
    for every feature rather than summarized, so a reader can see which one moved.
    """
    out = {}
    for c in cols:
        e = pd.to_numeric(era[c], errors="coerce").dropna()
        m = pd.to_numeric(modern[c], errors="coerce").dropna()
        lo, hi = float(e.min()), float(e.max())
        out[c] = {
            "era_p1": float(e.quantile(0.01)), "era_p99": float(e.quantile(0.99)),
            "era_min": lo, "era_max": hi,
            "modern_p1": float(m.quantile(0.01)), "modern_p99": float(m.quantile(0.99)),
            "modern_median": float(m.median()),
            "frac_outside_era_range": float(((m < lo) | (m > hi)).mean()),
        }
    return out


def label_type_census(modern: pd.DataFrame) -> dict:
    """Which modern label types the booster's categorical never saw.

    An unseen category is not an error -- it arrives as a missing categorical and
    LightGBM routes it down the default branch -- but it IS a handicap the shipped flat
    blend does not carry, because that model dropped label_type from the distance path
    entirely. Quantified here so the scoring tables can be split on it.
    """
    counts = modern["label_type"].value_counts()
    unseen = [t for t in counts.index if t not in dr.LABEL_TYPES]
    return {
        "era_categories": list(dr.LABEL_TYPES),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "unseen_types": unseen,
        "n_unseen_rows": int(counts[unseen].sum()) if unseen else 0,
        "frac_unseen_rows": float(counts[unseen].sum() / len(modern)) if unseen else 0.0,
    }


# ---------------------------------------------------------------------------- predictions

def flat_params(data_dir: str = DATA_DIR) -> dict:
    """The shipped estimator: ``final_coefficients`` from the modern-truth close-out."""
    with open(os.path.join(data_dir, "modern-truth-summary.json"), encoding="utf-8") as f:
        fc = json.load(f)["final_coefficients"]
    return {"form": "blend", "height_m": fc["params"]["height_m"],
            "blend_deg": fc["params"]["blend_deg"]}


def closed_form_predictions(df: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """A_deployed and D_blend come straight off the committed file (they are what
    ``run_modern_truth`` scored); ``D_flat_shipped`` is the shipped model.

    The name carries the ``_shipped`` suffix from birth because the runner needs a second
    flat model under the bare name ``D_flat``: the same functional form with its height
    refitted on the train half, which is the only one of the two that is held out on the
    rows it is scored against. Two models, two names, no reassignment in between.
    """
    out = pd.DataFrame(index=df.index)
    out["A_deployed"] = df["A_deployed"].to_numpy(float)
    out["D_blend"] = df["D_blend"].to_numpy(float)
    out["D_flat_shipped"] = dr.predict_dist(flat_params(data_dir), df)
    return out


def gbm_predictions(gbms: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Each frozen booster's distance on the modern rows.

    Driven through ``run_gbm_ceiling.predict_gbm`` itself, not a local reimplementation:
    the boosters must see the same feature-building and the same clipping that produced
    the #6 numbers, and the lesson from #6's own noise sweep is that a second copy of that
    code drifts. ``gbm_dep_l1`` additionally wants a ``depression_deg`` column holding the
    EXACT projection, which is the era column it was trained on, so it is swapped in for
    that one model only.
    """
    out = pd.DataFrame(index=df.index)
    swapped = df.assign(depression_deg=df["depression_deg_exact"])
    for key, model in gbms.items():
        frame = swapped if "depression_deg" in model["cols"] else df
        out[key] = gc.predict_gbm(model, frame)
    return out


# ------------------------------------------------------------------ the frozen-model guard

# What the phrase "these are the #6 boosters" is allowed to mean.
#
# Bit-identity is what a rerun gets on the host that built the committed summary, and the
# summary records whether it got it. It is NOT achievable across platforms, and asserting
# it at 1e-9 made this artifact regenerable on exactly one machine (issue #22): two
# platforms' libm disagree on arctan in the last bit, LightGBM chooses its splits at
# histogram bin boundaries, and one such bit is enough to move a split in the single
# variant that eats a trig-derived feature. Measured on Apple-silicon macOS in August 2026
# -- every best_iteration identical, three of four variants matching to ~1e-9, and
# gbm_dep_l1 off by 5.9e-5 m on the era-test median and 3.4e-3 m on its p90, identically
# across 8 thread counts, two Python versions and two NumPy/pandas stacks.
#
# So the guard asserts what the reports actually rest on. The bounds sit between the
# largest divergence anyone has measured and the smallest quantity either #6 report leans
# on -- the transfer report's tightest published number is a +0.014 m bootstrap bound, and
# its headline is a 0.40 m ceiling:
#
#   medians  1e-3 m   17x the measured divergence, 14x below that bootstrap bound
#   p90s     2e-2 m   an order statistic on a heavy tail moves by the gap between adjacent
#                     rows, so it is intrinsically the loosest of the four -- which is
#                     exactly the asymmetry macOS showed on one fit (5.9e-5 m median,
#                     3.4e-3 m p90). Still 6x that, and the p90 movements either report
#                     treats as a finding are 0.75 m and larger (section 8's tail)
#
# Past these the run stops exactly as it always did: a booster that far from the committed
# one is not the model the report describes, and scoring it would publish a different model
# under the same name.
CEILING_TOL_M = {"latlng_median_m": 1e-3, "latlng_p90_m": 2e-2,
                 "dist_median_m": 1e-3, "dist_p90_m": 2e-2}
BIT_IDENTICAL_TOL_M = 1e-9


def compare_to_committed_ceiling(era_matrix: dict, ceiling: dict,
                                 best_iterations: dict | None = None,
                                 tol: dict = CEILING_TOL_M) -> dict:
    """Verdict on whether the refitted boosters ARE the committed #6 boosters.

    Returns a verdict rather than raising, because the interesting case is neither pass nor
    fail: a rerun that is inside tolerance but not bit-identical is a legitimate
    regeneration of this artifact on another host, and the summary should record that it
    happened -- rather than the run dying, which is what shipped and what #22 caught, or
    the meta block claiming an identity this run does not have.
    """
    deltas = {f"{key}.{metric}": era_matrix[key][metric] - ceiling["matrix"][key][metric]
              for key in era_matrix for metric in tol}
    exceeded = {k: v for k, v in deltas.items() if abs(v) > tol[k.split(".", 1)[1]]}
    worst = max(deltas, key=lambda k: abs(deltas[k]))
    rounds = {k: [v, ceiling["meta"]["best_iterations"][k]]
              for k, v in (best_iterations or {}).items()}
    return {
        "within_tolerance": not exceeded,
        "bit_identical": abs(deltas[worst]) < BIT_IDENTICAL_TOL_M,
        "best_iterations_match": all(a == b for a, b in rounds.values()),
        "best_iterations_refit_vs_committed": rounds,
        "max_abs_delta_m": abs(deltas[worst]),
        "worst_metric": worst,
        "tolerance_m": dict(tol),
        "exceeded": exceeded,
        "meaning": "within_tolerance is the precondition this run enforces: the refitted "
                   "boosters answer the era test the same way the committed #6 matrix "
                   "does, to far inside anything either report claims. bit_identical is "
                   "recorded, not required -- LightGBM's splits are not reproducible "
                   "bit-for-bit across platforms (issue #22), so requiring it would make "
                   "this artifact regenerable on one machine rather than on any",
    }


# --------------------------------------------------------------------- split and rescaling

def pano_half_split(human: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """``modern_truth.remedy_check``'s split, reproduced exactly.

    Same seed, same ``np.sort`` on the unique pano ids, same ``rng.choice`` of half of
    them without replacement -- so the test half here is row-for-row the test half whose
    ``D_flat`` / ``D_rescaled`` numbers are already committed in
    ``modern-truth-summary.json``. The runner asserts that equality rather than trusting
    this comment.
    """
    rng = np.random.default_rng(seed)
    panos = np.sort(human["pano_id"].unique())
    train_ids = set(rng.choice(panos, len(panos) // 2, replace=False))
    in_train = human["pano_id"].isin(train_ids).to_numpy()
    return in_train, ~in_train


def scale_factor(pred: np.ndarray, truth: np.ndarray, dep_deg: np.ndarray,
                 min_dep_deg: float = SCALE_MIN_DEP_DEG) -> float:
    """One multiplicative recalibration: the median truth/prediction ratio on train rows.

    The blend's remedy fits ``k`` on implied camera height; in the cotangent regime a
    factor on height IS a factor on distance (d = h/tan(dep) is exactly linear in h), so
    the analogue for a model that has no height parameter is the median ratio of truth to
    prediction. Restricted to the same depression floor the remedy uses, where the
    geometry is stable, so the two sides' one parameter is fitted on the same rows.
    """
    ok = (dep_deg >= min_dep_deg) & np.isfinite(pred) & np.isfinite(truth) & (pred > 0)
    return float(np.median(truth[ok] / pred[ok]))


def affine_l1(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Two-parameter recalibration ``truth ~ a + b * pred``, fitted L1 on train rows.

    A scale alone cannot absorb an offset. This arm exists so that "the GBM only lost
    because it was given one parameter" is an answered objection rather than an open one:
    it is fitted under the same absolute-error loss the metric uses, seeded at the
    least-squares solution and polished by Nelder-Mead (deterministic, no seed).
    """
    from scipy.optimize import minimize

    b0, a0 = np.polyfit(pred, truth, 1)
    res = minimize(lambda p: np.abs(truth - (p[0] + p[1] * pred)).sum(),
                   x0=np.array([a0, b0]), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    return float(res.x[0]), float(res.x[1])


QUANTILE_GRID = np.linspace(0.0, 1.0, 101)


def quantile_map(pred_train: np.ndarray, truth_train: np.ndarray):
    """The most generous recalibration short of refitting: a monotone quantile map.

    Fitted on the train half, it replaces each predicted value by the truth value at the
    same rank -- so ANY monotone distortion of the model's answer scale, not just a linear
    one, is absorbed. What it cannot fix is the ordering of the predictions, which is
    exactly the conditional structure this test is asking about. If a model still loses
    after this, the loss is in which row it ranks where, not in the units it answers in.
    ``np.interp`` clamps outside the fitted range, which keeps the map bounded.
    """
    xp = np.quantile(pred_train, QUANTILE_GRID)
    fp = np.quantile(truth_train, QUANTILE_GRID)
    xp, keep = np.unique(xp, return_index=True)  # np.interp needs a strictly increasing xp
    fp = fp[keep]
    return lambda p: np.interp(np.asarray(p, float), xp, fp)


# ------------------------------------------------------------------------------- scoring

def _stats(err: np.ndarray, truth: np.ndarray) -> dict:
    """``modern_truth.model_metrics``' convention, so every number here is directly
    comparable to the committed Stage 4 tables."""
    return {
        "median_abs_m": float(np.median(np.abs(err))),
        "signed_median_m": float(np.median(err)),
        "p90_abs_m": float(np.percentile(np.abs(err), 90)),
        "range_slope_m_per_m": mt.range_slope(err, truth) if len(err) >= 3 else None,
        "n": int(len(err)),
    }


def score_frame(preds: pd.DataFrame, truth: np.ndarray, keys: list[str]) -> dict:
    return {k: _stats(preds[k].to_numpy(float) - truth, truth) for k in keys}


def bootstrap_medians(preds: pd.DataFrame, truth: np.ndarray, pano_id: np.ndarray,
                      keys: list[str], reference: str | None = None,
                      n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile intervals on each median absolute error, resampling PANORAMAS.

    Rows cluster hard by panorama (one pano carries up to a few dozen labels sharing a
    camera pose, a vintage and a ground plane), so a row bootstrap would understate the
    interval. When ``reference`` is given -- it must be one of ``keys`` -- the paired
    difference against it is taken WITHIN each resample, on identical rows, which is the
    only comparison that can separate two models this close together. That pairing is
    just a subtraction of two columns of ``draws``: both medians were already computed on
    the same ``idx``, so differencing them afterwards is exactly the within-draw
    difference, without medianing anything twice.
    """
    if reference is not None and reference not in keys:
        raise ValueError(f"reference {reference!r} must be one of keys {keys!r}")
    rng = np.random.default_rng(seed)
    panos, inv = np.unique(pano_id, return_inverse=True)
    rows_by_pano = [np.flatnonzero(inv == i) for i in range(len(panos))]
    errs = {k: preds[k].to_numpy(float) - truth for k in keys}

    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        pick = rng.integers(0, len(panos), len(panos))
        idx = np.concatenate([rows_by_pano[i] for i in pick])
        for k in keys:
            draws[k][b] = np.median(np.abs(errs[k][idx]))
    diffs = {k: draws[k] - draws[reference] for k in keys if reference and k != reference}

    out = {"n_boot": n_boot, "seed": seed, "cluster": "pano_id",
           "n_panos": int(len(panos)), "ci": {}}
    for k in keys:
        out["ci"][k] = {"median_abs_m_lo": float(np.percentile(draws[k], 2.5)),
                        "median_abs_m_hi": float(np.percentile(draws[k], 97.5))}
    if reference:
        out["reference"] = reference
        out["paired_diff_vs_reference"] = {
            k: {"delta_median_abs_m_lo": float(np.percentile(v, 2.5)),
                "delta_median_abs_m_hi": float(np.percentile(v, 97.5)),
                "frac_draws_better_than_reference": float((v < 0).mean())}
            for k, v in diffs.items()}
    return out


def implied_height_by_resolution(df: pd.DataFrame, truth_col: str, dep_col: str,
                                 min_dep_deg: float = SCALE_MIN_DEP_DEG,
                                 min_n: int = 25) -> dict:
    """median(truth x tan(depression)) per panorama height -- the camera height a truth
    set implies, cut by the axis the booster leans on hardest.

    This is the diagnostic that explains the transfer result rather than merely reporting
    it. If a truth set's implied scale is CONSTANT across resolutions, then a model that
    conditions on resolution has learned scene structure. If it is not, then part of what
    such a model learned is which subpopulation answers on which scale -- information that
    is worth a great deal inside that truth set and nothing at all on a population with a
    different resolution mix, where one global factor absorbs it.
    """
    sub = df[df[dep_col] >= min_dep_deg]
    implied = (sub[truth_col].to_numpy(float)
               * np.tan(np.radians(sub[dep_col].to_numpy(float))))
    out = {"min_dep_deg": min_dep_deg, "pooled": {"n": int(len(sub)),
                                                  "implied_height_m": float(np.median(implied))},
           "by_pano_height": {}}
    # to_numpy(float) first: pano_height is a nullable Int64 on the era frame, where an
    # elementwise == against pd.NA yields NA rather than False and poisons the mask
    h = sub["pano_height"].to_numpy(dtype=float, na_value=np.nan)
    for key, sel in [("missing", np.isnan(h))] + [
            (str(int(v)), h == v) for v in sorted(np.unique(h[~np.isnan(h)]))]:
        if sel.sum() >= min_n:
            out["by_pano_height"][key] = {"n": int(sel.sum()),
                                          "implied_height_m": float(np.median(implied[sel]))}
    return out


DIST_BIN_EDGES = gc.DIST_BIN_EDGES  # #6's bins, imported rather than copied


def error_vs_distance(preds: pd.DataFrame, truth: np.ndarray, keys: list[str],
                      min_n: int = 25) -> list[dict]:
    """Median absolute error by TRUE distance bin -- #6 section 7's cut, on modern truth.

    #6 found the GBM's advantage widening with distance and holding the 10-15 m band the
    blend's saturation trades away. Whether that shape survives the truth change is a
    sharper question than the pooled median, because a pure scale error would move every
    bin proportionally while genuine conditional structure would not.
    """
    bins = pd.cut(truth, DIST_BIN_EDGES, right=False)
    rows = []
    for interval in bins.categories:
        sel = bins == interval
        if sel.sum() < min_n:
            continue
        t = truth[sel]
        rows.append({"bin_m": str(interval), "n": int(sel.sum()), "per_model": {
            k: {"median_abs_m": float(np.median(np.abs(preds[k].to_numpy(float)[sel] - t)))}
            for k in keys}})
    return rows
