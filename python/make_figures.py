"""Generate the figures that document how the estimators work and how well they perform.

Outputs to figures/*.png (light surface, 200 dpi). Fits and errors come from the same library
code the tests verify, on the same R-fixture train/test split, so every number shown is the
tested pipeline's output.

Usage (repo root): python python/make_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    add_heading_diff, clean_data, evaluate, fit_models, load_data,
    predict_dist_heading, split_from_fixtures,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Palette (validated; see the dataviz notes in the PR/issue). Color follows the entity
# everywhere: est7 is always blue, the est5 tier orange, est4 aqua, naive baselines gray.
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_EST = {"est7": "#2a78d6", "est5": "#eb6834", "est6": "#eb6834", "est4": "#1baf7a",
         "est3": "#52514e", "est2": "#898781", "est1": "#b5b3ac"}
C_ZOOM = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}  # ordinal blue ramp
C_H8192, C_H6656 = "#2a78d6", "#eb6834"
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

PUBLISHED_DIST = [(18.6051843, 0.0138947, 0.0011023), (20.8794248, 0.0184087, 0.0022135),
                  (25.2472682, 0.0264216, 0.0011071)]
PUBLISHED_HEAD = [(-51.2401711, 0.1443374), (-27.5267447, 0.0784357), (-13.5675945, 0.0396061)]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8, "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK, "axes.titlesize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9,
})


def _title(fig, title, subtitle, wrap=112):
    import textwrap
    fig.suptitle(title, x=0.02, y=0.99, ha="left", va="top", fontsize=13, color=INK,
                 weight="bold")
    fig.text(0.02, 0.945, textwrap.fill(subtitle, wrap), ha="left", va="top", fontsize=9.5,
             color=SECONDARY, linespacing=1.35)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def _binned_median(x, y, bins):
    idx = np.digitize(x, bins)
    cx, cy = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 200:
            cx.append((bins[i - 1] + bins[i]) / 2)
            cy.append(np.median(y[m]))
    return np.array(cx), np.array(cy)


def fig_mechanism_dist(cleaned, models):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for z, ax in zip((1, 2, 3), axes):
        sub = cleaned[cleaned["zoom"] == z]
        x, y = sub["sv_image_y"].to_numpy(float), sub["pano_dist"].to_numpy(float)
        lo, hi = np.quantile(x, [0.005, 0.995])
        ax.hexbin(x, y, gridsize=55, extent=(lo, hi, 0, 50), cmap=DENSITY_CMAP,
                  norm=LogNorm(), linewidths=0)
        xs = np.linspace(lo, hi, 100)
        cd = models["est7"]["dist"][z - 1]
        med_cy = float(sub["canvas_y"].median())
        ax.plot(xs, cd["(Intercept)"] + cd["sv_image_y"] * xs + cd["canvas_y"] * med_cy,
                color=INK, lw=2, solid_capstyle="round")
        a, b, c = PUBLISHED_DIST[z - 1]
        ax.plot(xs, a + b * xs + c * med_cy, color="#eb6834", lw=2, ls=(0, (5, 3)))
        ax.set_title(f"zoom {z}   n={len(sub):,}", loc="left")
        ax.set_xlabel("sv_image_y  (px from horizon)")
        ax.set_ylim(0, 50)
    axes[0].set_ylabel("distance from pano (m)")
    axes[0].text(0.03, 0.90, "refit 2026", color=INK, transform=axes[0].transAxes, fontsize=9)
    axes[0].text(0.03, 0.82, "published 2021", color="#eb6834", transform=axes[0].transAxes,
                 fontsize=9)
    _title(fig, "The distance model: farther below the horizon = closer to the camera",
           "Depth-derived ground truth (395,147 labels, 7 cities) vs sv_image_y, by zoom. The 2026 refit "
           "(solid) lands on the published 2021 fit (dashed) — and the visible curvature is the linear "
           "model's compressive bias (SidewalkWebpage#4766). Lines drawn at each zoom's median canvas_y.")
    fig.subplots_adjust(top=0.74, wspace=0.08)
    _save(fig, "fig1-mechanism-distance.png")


def fig_mechanism_heading(cleaned, models):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for z, ax in zip((1, 2, 3), axes):
        sub = cleaned[cleaned["zoom"] == z]
        x, y = sub["canvas_x"].to_numpy(float), sub["heading_diff"].to_numpy(float)
        ax.hexbin(x, y, gridsize=55, extent=(0, 720, -75, 75), cmap=DENSITY_CMAP,
                  norm=LogNorm(), linewidths=0)
        xs = np.linspace(0, 720, 100)
        ch = models["est7"]["heading"][z - 1]
        ax.plot(xs, ch["(Intercept)"] + ch["canvas_x"] * xs, color=INK, lw=2)
        d, e = PUBLISHED_HEAD[z - 1]
        ax.plot(xs, d + e * xs, color="#eb6834", lw=2, ls=(0, (5, 3)))
        ax.set_title(f"zoom {z}", loc="left")
        ax.set_xlabel("canvas_x  (px)")
        ax.set_ylim(-75, 75)
    axes[0].set_ylabel("heading offset (°)")
    _title(fig, "The heading model: click position across the canvas maps linearly to bearing",
           "Heading offset (bearing to label minus camera heading) vs canvas_x, by zoom. Slope halves "
           "with each zoom step, tracking the narrower field of view. Refit solid, published 2021 dashed.")
    fig.subplots_adjust(top=0.74, wspace=0.08)
    _save(fig, "fig2-mechanism-heading.png")


def fig_error_ecdf(err):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    label_text = {"est1": "always 10 m", "est2": "median dist", "est3": "median by type",
                  "est4": "multivariate lm", "est5": "lm + zoom term", "est6": "mixed model",
                  "est7": "per-zoom lm (production)"}
    order = ["est1", "est2", "est3", "est4", "est6", "est5", "est7"]
    for est in order:
        e = np.sort(err[f"error_{est}"].to_numpy())
        yy = np.arange(1, len(e) + 1) / len(e)
        lw = 2.6 if est == "est7" else 1.8
        ls = (0, (4, 2)) if est == "est6" else "-"
        ax.plot(e, yy, color=C_EST[est], lw=lw, ls=ls,
                label=f"{est} — {label_text[est]} ({np.median(e):.2f} m)")
        med = float(np.median(e))
        ax.plot([med], [0.5], "o", ms=5, color=C_EST[est], zorder=5)
    ax.axhline(0.5, color=BASELINE, lw=0.8, zorder=0)
    ax.text(11.85, 0.51, "median", color=MUTED, fontsize=8.5, ha="right")
    ax.text(1.05, 0.62, "est7", color=C_EST["est7"], fontsize=10, weight="bold")
    ax.text(2.75, 0.66, "est5/6", color=C_EST["est5"], fontsize=10, weight="bold")
    ax.text(4.55, 0.60, "est4", color=C_EST["est4"], fontsize=10, weight="bold")
    ax.text(6.55, 0.55, "est1–3\nbaselines", color=C_EST["est3"], fontsize=9.5)
    ax.set_xlim(0, 12); ax.set_ylim(0, 1)
    ax.set_xlabel("position error (m)")
    ax.set_ylabel("fraction of test labels with error ≤ x")
    leg = ax.legend(loc="lower right", title="estimator — median error")
    leg.get_title().set_color(SECONDARY)
    _title(fig, "Seven candidate estimators: error distributions on the held-out 20%",
           "79,029 test labels. Every regression tier helps: naive baselines ~4.6–4.8 m median, one "
           "regression 3.4 m, zoom-aware 1.79 m, per-zoom fits (production, est7) 1.46 m. est6 (dashed) "
           "overlaps est5 almost exactly.", wrap=95)
    fig.subplots_adjust(top=0.82)
    _save(fig, "fig3-error-ecdf.png")


def fig_bias_by_distance(test, models):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    d_pred, _ = predict_dist_heading(models, test, "est7")
    resid = d_pred - test["pano_dist"].to_numpy(float)
    bins = np.arange(0, 31, 1.0)
    for z in (1, 2, 3):
        m = (test["zoom"] == z).to_numpy()
        cx, cy = _binned_median(test["pano_dist"].to_numpy(float)[m], resid[m], bins)
        ax.plot(cx, cy, color=C_ZOOM[z], lw=2, marker="o", ms=4)
        ax.annotate(f"zoom {z}", (cx[-1], cy[-1]), xytext=(6, 0), textcoords="offset points",
                    color=C_ZOOM[z], fontsize=9.5, va="center")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.text(0.5, 1.55, "↑ predicts too far", color=MUTED, fontsize=9)
    ax.text(0.5, -7.8, "↓ predicts too near", color=MUTED, fontsize=9)
    ax.set_xlim(0, 31); ax.set_xlabel("true distance from pano (m)")
    ax.set_ylabel("median signed error of est7 distance (m)")
    _title(fig, "The linear model's distance bias flips sign with range — a line chasing a cotangent",
           "Median (predicted − true) est7 distance in 1 m bins of true distance, test set, bins with "
           "≥200 labels. Too near for the closest labels (the fit dives to zero where true distance "
           "floors at ~3 m), too far at 5–13 m, then increasingly too near beyond ~15 m — the empirical "
           "case for the cotangent refit (SidewalkWebpage#4766).", wrap=95)
    fig.subplots_adjust(top=0.82)
    _save(fig, "fig4-bias-by-distance.png")


def fig_error_by_distance(test, err):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bins = np.arange(0, 31, 1.0)
    truth = test["pano_dist"].to_numpy(float)
    for est, name in (("est3", "median by type"), ("est5", "lm + zoom"), ("est7", "per-zoom lm")):
        cx, cy = _binned_median(truth, err[f"error_{est}"].to_numpy(), bins)
        ax.plot(cx, cy, color=C_EST[est], lw=2.2 if est == "est7" else 1.8, marker="o", ms=4)
        ax.annotate(f"{est} — {name}", (cx[-1], cy[-1]), xytext=(6, 0),
                    textcoords="offset points", color=C_EST[est], fontsize=9.5, va="center")
    ax.set_xlim(0, 34); ax.set_ylim(0, 22)
    ax.set_xlabel("true distance from pano (m)")
    ax.set_ylabel("median position error (m)")
    _title(fig, "Where the regressions win — and where every estimator degrades",
           "Median position error in 1 m bins of true distance, test set. Regressions help most in the "
           "8–25 m band; beyond ~25 m all estimators drift, because predicted distances rarely exceed "
           "the high-teens (the Rmd's original caveat, quantified).", wrap=95)
    fig.subplots_adjust(top=0.82)
    _save(fig, "fig5-error-by-distance.png")


def fig_height_groups(cleaned, models):
    both = cleaned[cleaned["pano_height"].isin([6656, 8192])].copy()
    d_pred, _ = predict_dist_heading(models, both, "est7")
    both["resid"] = d_pred - both["pano_dist"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    panels = [("all six modern cities", both), ("SPGG only (a natural A/B: 50/50 mix)",
                                                both[both["city"] == "spgg"])]
    for (name, sub), ax in zip(panels, axes):
        x = sub["sv_image_y"].to_numpy(float)
        lo, hi = np.quantile(x, [0.01, 0.99])
        bins = np.linspace(lo, hi, 13)
        for h, color in ((8192, C_H8192), (6656, C_H6656)):
            m = (sub["pano_height"] == h).to_numpy()
            cx, cy = _binned_median(x[m], sub["resid"].to_numpy()[m], bins)
            ax.plot(cx, cy, color=color, lw=2, marker="o", ms=4, label=f"{h} px panos")
        ax.axhline(0, color=BASELINE, lw=1)
        ax.set_title(f"{name}   n={len(sub):,}", loc="left")
        ax.set_xlabel("sv_image_y  (px from horizon)")
    axes[0].set_ylabel("median signed error of est7 distance (m)")
    axes[0].legend(loc="upper left")
    _title(fig, "Raw-pixel predictors mis-scale across pano resolutions",
           "Median (predicted − true) distance vs sv_image_y, split by panorama height. The same pixel "
           "offset means a different angle on a 6,656 px pano than on an 8,192 px one, so one raw-pixel "
           "model can't serve both (SidewalkWebpage#4765) — and with 37k+ labels on 6,656 px panos, the "
           "height term IS identifiable in this dataset.", wrap=120)
    fig.subplots_adjust(top=0.70, wspace=0.06)
    _save(fig, "fig6-height-resolution.png")


def main() -> None:
    data = load_data(os.path.join(ROOT, "data"))
    cleaned, _ = clean_data(data)
    cleaned = add_heading_diff(cleaned)
    train, test = split_from_fixtures(cleaned, os.path.join(ROOT, "tests", "fixtures", "r-baseline"))
    models = fit_models(train)
    err = evaluate(models, test)
    full_models = fit_models(cleaned)

    fig_mechanism_dist(cleaned, full_models)
    fig_mechanism_heading(cleaned, full_models)
    fig_error_ecdf(err)
    fig_bias_by_distance(test, models)
    fig_error_by_distance(test, err)
    fig_height_groups(cleaned, full_models)


if __name__ == "__main__":
    main()
