"""Figures 29-38 for reports/2026-09-02-production-signoff.md (SidewalkWebpage#5084).

Reads data/signoff-summary.json, the per-row cache run_signoff.py build leaves under
data/signoff-cache/, the committed depth payloads and the committed example tiles. Offline.

    python python/signoff_figures.py            # writes figures/fig29-*.png .. fig38-*.png
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsv_depth as gd  # noqa: E402
import modern_truth as mt  # noqa: E402
import signoff as so  # noqa: E402
from depth_validation import stitch_tiles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
CACHE = os.path.join(DATA, "signoff-cache")

# One fixed palette, one meaning per hue, never re-assigned between figures.
C_SHIP = "#2a78d6"    # approximation3 as shipped
C_REG = "#eb6834"     # the 2021 per-zoom regression (est7 / A_deployed)
C_ERA = "#1baf7a"     # the same form with the era-calibrated height (equal budget)
C_APX1 = "#a39f94"    # approximation1, the 2020 stopgap (10 m along the viewport heading)
C_TRUTH = "#0b0b0b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.labelsize": 9.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "legend.frameon": False, "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "savefig.facecolor": C_SURFACE, "savefig.dpi": 160, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def load():
    with open(os.path.join(DATA, "signoff-summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    era = pd.read_pickle(os.path.join(CACHE, "era_scored.pkl.gz"))
    modern = pd.read_pickle(os.path.join(CACHE, "modern_scored.pkl.gz"))
    return summary, era, modern


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def cdf(ax, err, color, label, xmax=6.0):
    e = np.sort(err[np.isfinite(err)])
    y = np.arange(1, len(e) + 1) / len(e)
    ax.plot(np.clip(e, 0, xmax), y, color=color, label=label)
    med = float(np.median(e))
    ax.plot([med], [0.5], "o", ms=7, color=color, mec=C_SURFACE, mew=1.5)
    return med


def med_by_bin(frame, bins, dist_col, err_cols):
    """Per-bin medians, plotted at each bin's own median distance rather than its centre: a
    continuous curve read against them (fig 29's single-click floor, which grows as d^2) is
    then compared at the distance the bin's mass actually sits at."""
    grouped = frame.groupby(pd.cut(frame[dist_col], bins, right=False), observed=True)
    return (grouped[dist_col].median().to_numpy(float), grouped[err_cols].median(),
            grouped.size())


# ------------------------------------------------------------------- fig 29 modern frame

def fig29(summary, modern):
    rep = modern[modern["stratum"] == "representative"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    cdf(ax, rep["err_approx1"].to_numpy(float), C_APX1, "approximation1 (2020 stopgap, distance half)")
    m_a = cdf(ax, rep["err_A"].to_numpy(float), C_REG, "approximation2: 2021 regression (deployed until 2026-08)")
    m_s = cdf(ax, rep["err_approx3"].to_numpy(float), C_SHIP, "approximation3 as shipped")
    ax.set_xlabel("absolute distance error (m), representative human stratum")
    ax.set_ylabel("share of labels")
    ax.set_title("Modern truth: error distribution")
    ax.annotate(f"median {m_s:.2f} m", (m_s, 0.5), xytext=(m_s + 0.35, 0.36), fontsize=8.5, color="#52514e")
    ax.annotate(f"median {m_a:.2f} m", (m_a, 0.5), xytext=(m_a + 0.35, 0.56), fontsize=8.5, color="#52514e")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    ax = axes[1]
    bins = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 30, 50]
    c, med, n = med_by_bin(modern, bins, "truth_m", ["err_A", "err_approx3"])
    ax.plot(c, med["err_A"], "-o", color=C_REG, ms=5, label="approximation2 (2021 regression)")
    ax.plot(c, med["err_approx3"], "-o", color=C_SHIP, ms=5, label="approximation3")
    # The two ideal lines: what one click can resolve, and what depth truth can measure.
    ideal = summary["modern_frame"]["ideal"]
    dd = np.geomspace(1.2, 45, 200)
    ax.plot(dd, so.single_click_floor_m(dd, ideal["height_m"], ideal["click_noise_sigma_deg"]),
            ":", color=C_TRUTH, lw=1.6,
            label=f"single-click floor ({ideal['click_noise_sigma_deg']:g}° click noise)")
    ax.axhspan(0.01, ideal["truth_band_m"][1], color=C_MUTED, alpha=0.15, lw=0,
               label=f"truth's own noise (≤{ideal['truth_band_m'][1]:.2f} m)")
    ax.set_xscale("log")
    ax.set_yscale("log")  # the floor spans 0.05-1.4 m; a linear axis hides the near field entirely
    ax.set_ylim(0.03, 25)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "5", "10", "20", "40"])
    ax.set_xlabel("true distance from the camera (m), log scale")
    ax.set_ylabel("median absolute error (m), log scale")
    ax.set_title("By distance (pooled human, n=%d)" % len(modern))
    ax.legend(loc="upper left")

    ax = axes[2]
    _, sgn, _ = med_by_bin(modern, bins, "truth_m", ["sderr_A", "sderr_approx3"])
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.plot(c, sgn["sderr_A"], "-o", color=C_REG, ms=5, label="2021 regression")
    ax.plot(c, sgn["sderr_approx3"], "-o", color=C_SHIP, ms=5, label="approximation3")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "5", "10", "20", "40"])
    ax.set_xlabel("true distance from the camera (m), log scale")
    ax.set_ylabel("median signed error (m), + = placed too far")
    ax.set_title("Compression: signed error by distance")
    ax.set_ylim(-18, 4)
    ax.legend(loc="lower left")
    fig.suptitle("Figure 29 - modern truth (fresh GSV depth at the stored click, post-2021 labels, 36 cities)",
                 x=0.01, ha="left", fontsize=10.5, color="#52514e")
    save(fig, "fig29-signoff-modern-frame.png")


# ---------------------------------------------------------------------- fig 30 era frame

def fig30(summary, era):
    e = summary["era_frame"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    cdf(ax, era["err_approx1"].to_numpy(float), C_APX1, "approximation1 (2020 stopgap)")
    m7 = cdf(ax, era["err_est7"].to_numpy(float), C_REG, "approximation2: 2021 regression (as published)")
    ms = cdf(ax, era["err_approx3"].to_numpy(float), C_SHIP, "approximation3 as shipped")
    me = cdf(ax, era["err_approx3_eraflat"].to_numpy(float), C_ERA, "same form, height fitted on era truth")
    ax.set_xlabel("lat/lng error vs the 2017-2020 depth positions (m)")
    ax.set_ylabel("share of labels")
    ax.set_title("Era truth: the regression's home turf (n=79,029)")
    ax.annotate(f"{me:.2f}", (me, 0.5), xytext=(me - 0.55, 0.58), fontsize=8.5, color="#52514e")
    ax.annotate(f"{ms:.2f}", (ms, 0.5), xytext=(ms + 0.15, 0.40), fontsize=8.5, color="#52514e")
    ax.annotate(f"{m7:.2f}", (m7, 0.5), xytext=(m7 + 0.3, 0.52), fontsize=8.5, color="#52514e")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    ax = axes[1]
    bins = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 30, 50]
    c, med, n = med_by_bin(era, bins, "pano_dist", ["err_est7", "err_approx3", "err_approx3_eraflat"])
    ax.plot(c, med["err_est7"], "-o", color=C_REG, ms=5, label="2021 regression")
    ax.plot(c, med["err_approx3"], "-o", color=C_SHIP, ms=5, label="approximation3 as shipped")
    ax.plot(c, med["err_approx3_eraflat"], "-o", color=C_ERA, ms=5, label="era-calibrated height")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "5", "10", "20", "40"])
    ax.set_xlabel("true distance from the camera (m), log scale")
    ax.set_ylabel("median lat/lng error (m)")
    ax.set_title("By distance")
    ax.legend(loc="upper left")

    ax = axes[2]
    rows = e["implied_height_by_pano_height"] + [{"pano_height_px": "all", "n": 0,
                                                  "implied_height_m": e["implied_height_overall_m"]}]
    labels = {"0": "no pano metadata (99% DC)", "6656": "6656-px panos", "8192": "8192-px panos", "all": "all era rows"}
    ys = np.arange(len(rows))
    vals = [r["implied_height_m"] for r in rows]
    ax.barh(ys, vals, height=0.5, color="#9ec5f4")
    for y, v, r in zip(ys, vals, rows):
        ax.text(v + 0.02, y, f"{v:.2f} m", va="center", fontsize=8.5, color="#52514e")
    ax.axvline(summary["meta"]["shipped"]["height_m"], color=C_SHIP, lw=2)
    ax.text(summary["meta"]["shipped"]["height_m"] - 0.03, -0.75, "shipped 2.34 m", ha="right", va="center",
            fontsize=8.5, color="#52514e")
    ax.axvline(2.354, color=C_MUTED, lw=1)
    ax.text(2.354 + 0.03, -0.75, "measured rig 2.35 m", fontsize=8, va="center", color=C_MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[r["pano_height_px"]] for r in rows])
    ax.set_ylim(-1.2, len(rows) - 0.4)
    ax.set_xlim(2.0, 3.0)
    ax.set_xlabel("camera height the era truth implies, median(truth x tan dep), m")
    ax.set_title("Why the era frame disagrees: its own scale")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 30 - the 2021 regression's own held-out split (2017-2020 depth positions as truth)",
                 x=0.01, ha="left", fontsize=10.5, color="#52514e")
    save(fig, "fig30-signoff-era-frame.png")


# ------------------------------------------------------------------------- fig 31 slices

def dumbbell(ax, rows, key, a_key, b_key, label_fn, title, xlabel, xmax=None):
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        a, b = r[a_key]["median_m"], r[b_key]["median_m"]
        ax.plot([a, b], [y, y], color="#c3c2b7", lw=1.5, zorder=1)
        ax.plot([a], [y], "o", color=C_REG, ms=7, mec=C_SURFACE, mew=1.5, zorder=2)
        ax.plot([b], [y], "o", color=C_SHIP, ms=7, mec=C_SURFACE, mew=1.5, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([label_fn(r) for r in rows])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xmax)


def fig31(summary):
    m, e = summary["modern_frame"], summary["era_frame"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 9, 3]})
    dumbbell(axes[0, 0], m["by_zoom"], "zoom_i", "A_deployed", "approx3",
             lambda r: f"zoom {r['zoom_i']}  (n={r['n']:,})", "Modern truth - by zoom", "median |error| (m)", 2.6)
    dumbbell(axes[0, 1], e["by_zoom"], "zoom", "est7", "approx3",
             lambda r: f"zoom {r['zoom']}  (n={r['n']:,})", "Era truth - by zoom", "median lat/lng error (m)", 2.6)
    dumbbell(axes[1, 0], m["by_label_type"], "label_type", "A_deployed", "approx3",
             lambda r: f"{r['label_type']}  (n={r['n']:,})", "By label type", "median |error| (m)", 4.0)
    dumbbell(axes[1, 1], e["by_label_type"], "label_type", "est7", "approx3",
             lambda r: f"{r['label_type']}  (n={r['n']:,})", "By label type", "median lat/lng error (m)", 4.0)
    ph = {"0": "no metadata (99% DC)", "6656": "6656 px", "8192": "8192 px"}
    dumbbell(axes[2, 0], m["by_pano_height"], "pano_height_px", "A_deployed", "approx3",
             lambda r: f"{ph.get(r['pano_height_px'], r['pano_height_px'])}  (n={r['n']:,})",
             "By panorama resolution", "median |error| (m)", 2.6)
    dumbbell(axes[2, 1], e["by_pano_height"], "pano_height_px", "est7", "approx3",
             lambda r: f"{ph.get(r['pano_height_px'], r['pano_height_px'])}  (n={r['n']:,})",
             "By panorama resolution", "median lat/lng error (m)", 2.6)
    h = [plt.Line2D([], [], marker="o", color=C_REG, ls="", ms=7, label="2021 regression"),
         plt.Line2D([], [], marker="o", color=C_SHIP, ls="", ms=7, label="approximation3 as shipped")]
    fig.legend(handles=h, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 31 - head-to-head by slice, both truth frames (lower is better)",
                 x=0.01, ha="left", fontsize=10.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig31-signoff-slices.png")


# ----------------------------------------------------------------- fig 32 generalization

def fig32(summary):
    m = summary["modern_frame"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    h = m["repeated_holdout"]
    ax.errorbar([0], [h["approx3_median_m"]["mean"]],
                yerr=[[h["approx3_median_m"]["mean"] - h["approx3_median_m"]["p5"]],
                      [h["approx3_median_m"]["p95"] - h["approx3_median_m"]["mean"]]],
                fmt="o", color=C_SHIP, ms=8, capsize=6, lw=2)
    ax.errorbar([1], [h["A_deployed_median_m"]["mean"]],
                yerr=[[h["A_deployed_median_m"]["mean"] - h["A_deployed_median_m"]["p5"]],
                      [h["A_deployed_median_m"]["p95"] - h["A_deployed_median_m"]["mean"]]],
                fmt="o", color=C_REG, ms=8, capsize=6, lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["approximation3\n(height re-fitted on the other half)", "2021 regression\n(same held-out half)"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("median |error| on the held-out pano half (m)")
    ax.set_title(f"{h['n_rep']} random pano-half splits: mean and 5-95% band")
    ax.text(0, h["approx3_median_m"]["p95"] + 0.06, f"{h['approx3_median_m']['mean']:.3f} m", ha="center",
            fontsize=9, color="#52514e")
    ax.text(1, h["A_deployed_median_m"]["p95"] + 0.06, f"{h['A_deployed_median_m']['mean']:.3f} m", ha="center",
            fontsize=9, color="#52514e")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    rows = sorted(m["leave_one_city_out"], key=lambda r: r["n"], reverse=True)
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        ax.plot([r["A_deployed_median_m"], r["approx3_loco_median_m"]], [y, y], color="#c3c2b7", lw=1.5, zorder=1)
        ax.plot([r["A_deployed_median_m"]], [y], "o", color=C_REG, ms=7, mec=C_SURFACE, mew=1.5, zorder=2)
        ax.plot([r["approx3_loco_median_m"]], [y], "o", color=C_SHIP, ms=7, mec=C_SURFACE, mew=1.5, zorder=3)
        ax.text(3.05, y, f"h = {r['height_fitted_elsewhere_m']:.3f} m", va="center", fontsize=8, color=C_MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['city']}  (n={r['n']})" for r in rows])
    ax.set_xlim(0, 3.6)
    ax.set_xlabel("median |error| (m); height calibrated on every OTHER city")
    ax.set_title("Leave-one-city-out: one height transfers across rigs")
    ax.grid(axis="y", visible=False)
    hh = [plt.Line2D([], [], marker="o", color=C_REG, ls="", ms=7, label="2021 regression"),
          plt.Line2D([], [], marker="o", color=C_SHIP, ls="", ms=7, label="approximation3, height fitted elsewhere")]
    ax.legend(handles=hh, loc="upper center", bbox_to_anchor=(0.42, -0.14), ncol=2)
    fig.suptitle("Figure 32 - does the one calibrated constant generalize? (modern truth, human gated rows)",
                 x=0.01, ha="left", fontsize=10.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig32-signoff-generalization.png")


# ------------------------------------------------------------------------ fig 33 geodesy

def fig33(summary):
    g = summary["geodesy"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    cities = g["per_city"]
    lats = [c["latitude"] for c in cities]
    order = np.argsort(np.abs(lats))
    for rank, i in enumerate(order):
        c = cities[i]
        d = [r["distance_m"] for r in c["rows"]]
        s = [r["ellipsoid_vs_production_max_m"] * 100 for r in c["rows"]]
        col = SEQ[min(6, 1 + rank * 6 // max(1, len(cities) - 1))]
        ax.plot(d, s, color=col, lw=1.6)
        if i in (order[0], order[-1]):
            ax.text(d[-1] + 0.6, s[-1], f"{c['city']} ({c['latitude']:.0f}°)", fontsize=8.5, va="center", color="#52514e")
    harness = [r["harness_vs_production_max_m"] * 100 for r in cities[0]["rows"]]
    turf = [r["turf_vs_production_max_m"] * 100 for r in cities[0]["rows"]]
    d = [r["distance_m"] for r in cities[0]["rows"]]
    ax.plot(d, harness, color=C_MUTED, lw=1.4)
    ax.text(d[-1] + 0.6, harness[-1], "research sphere\n(6378.137 km)", fontsize=8, va="center", color=C_MUTED)
    ax.plot(d, turf, color=C_MUTED, lw=1.4)
    ax.text(d[-1] + 0.6, turf[-1] + 1.2, "client turf sphere\n(6371.0088 km): 0.007 cm", fontsize=8, va="center", color=C_MUTED)
    for x, lab in ((g["median_label_distance_m"], "median\nlabel"), (so.MAX_ANSWER_M, "largest\nanswer")):
        ax.axvline(x, color="#c3c2b7", lw=0.8)
        ax.text(x + 0.4, 21.5, lab, fontsize=8, color=C_MUTED, va="top")
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 23)
    ax.set_xlabel("estimated distance from the camera (m)")
    ax.set_ylabel("worst-bearing displacement vs the production sphere (cm)")
    ax.set_title("WGS84 geodesic vs the 6371 km sphere, per city in the two datasets (blue = |latitude|)")

    ax = axes[1]
    cur = sorted(g["curvature"], key=lambda r: r["latitude"])
    ys = np.arange(len(cur))
    ns = [r["north_south_scale_error"] * 100 for r in cur]
    ew = [r["east_west_scale_error"] * 100 for r in cur]
    ax.barh(ys + 0.18, ns, height=0.34, color=SEQ[3], label="north-south (meridional radius)")
    ax.barh(ys - 0.18, ew, height=0.34, color=SEQ[1], label="east-west (prime-vertical radius)")
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['city']} ({r['latitude']:.0f}°)" for r in cur])
    ax.set_xlabel("sphere scale error, % (+ = the sphere places the label too near)")
    ax.set_title("Where it comes from: 6371 km vs the local radii of curvature")
    ax.legend(loc="upper right")  # the lower-right quadrant holds the two largest bars
    ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 33 - geodesy at label distances: centimeters, bounded, and shared by every implementation",
                 x=0.01, ha="left", fontsize=10.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig33-signoff-geodesy.png")


# ----------------------------------------------------------------- fig 34 frame contract

def fig34(summary):
    shipped = summary["meta"]["shipped"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    # Left: error map over a 1920x1080 frame when the click is scaled by width and read as 720x480.
    w, h = 1920, 1080
    pov_h, pov_p, zoom = 40.0, -8.0, 1.0
    xs = np.linspace(0, w, 97)
    ys = np.linspace(0, h, 55)
    gx, gy = np.meshgrid(xs, ys)
    hh, pp = so.canvas_to_centered_pov(pov_h, pov_p, zoom, gx.ravel(), gy.ravel(), w, h)
    dep_true = -pp
    k = 720.0 / w
    hw, pw = so.canvas_to_centered_pov(pov_h, pov_p, zoom, gx.ravel() * k, gy.ravel() * k, 720.0, 480.0)
    lat_t, lng_t = so.destination(47.6553, -122.3035, so.estimate_distance_m(dep_true, shipped), hh)
    lat_w, lng_w = so.destination(47.6553, -122.3035, so.estimate_distance_m(-pw, shipped), hw)
    err = so.haversine_m(lng_t, lat_t, lng_w, lat_w).reshape(gy.shape)
    err[dep_true.reshape(gy.shape) < 2.0] = np.nan  # above/near the horizon: nothing is placeable
    ax = axes[0]
    im = ax.imshow(err, extent=(0, w, h, 0), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ),
                   vmin=0, vmax=12, aspect="equal", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("position error (m)")
    cb.outline.set_visible(False)
    ax.set_title("A 1920x1080 click read through the 720x480 constant (width-scaled)")
    ax.set_xlabel("canvas x (px)")
    ax.set_ylabel("canvas y (px)")
    ax.grid(False)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    ax = axes[1]
    frames = summary["viewport_frame_contract"]["frames"]
    yy = np.arange(len(frames))[::-1]
    for y, f in zip(yy, frames):
        a = f["axis_scaled_to_720x480"]["p90_m"]
        b = f["width_scaled_read_as_720x480"]["p90_m"]
        ax.plot([0, b], [y, y], color="#c3c2b7", lw=1.5, zorder=1)
        ax.plot([f["own_frame_max_error_m"]], [y], "o", color=C_SHIP, ms=8, mec=C_SURFACE, mew=1.5, zorder=4)
        ax.plot([a], [y], "s", color=SEQ[2], ms=7, mec=C_SURFACE, mew=1.5, zorder=3)
        ax.plot([b], [y], "D", color=SEQ[5], ms=7, mec=C_SURFACE, mew=1.5, zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([f["frame"] for f in frames])
    ax.set_xlabel("p90 position error over the same 387 label directions (m)")
    ax.set_title("Per frame: own frame vs the two wrong conventions")
    ax.grid(axis="y", visible=False)
    hh = [plt.Line2D([], [], marker="o", color=C_SHIP, ls="", ms=8, label="click projected through its own frame (0 m)"),
          plt.Line2D([], [], marker="s", color=SEQ[2], ls="", ms=7, label="scaled axis-by-axis into 720x480"),
          plt.Line2D([], [], marker="D", color=SEQ[5], ls="", ms=7, label="scaled by width, read as 720x480")]
    ax.legend(handles=hh, loc="lower right")
    ax.set_xlim(-0.3, 14)
    fig.suptitle("Figure 34 - the frame contract: the estimator sees only angles, so the click must be projected "
                 "through the frame it was made in", x=0.01, ha="left", fontsize=10.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig34-signoff-frame-contract.png")


# ------------------------------------------------------------------ figs 35-38 examples

def load_payloads(pano_ids):
    out = {}
    with gzip.open(os.path.join(DATA, "modern-truth-payloads.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["pano_id"] in pano_ids:
                out[d["pano_id"]] = gd.decode_depth_payload(d["b64"])
    return out


def load_tiles():
    out = {}
    with gzip.open(os.path.join(DATA, "signoff-tiles.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tiles = [{"x": t["x"], "y": t["y"], "bytes": base64.b64decode(t["b64"])} for t in d["tiles"]]
            out[d["pano_id"]] = (stitch_tiles(tiles, d["width"], d["height"], d["tile_width"], d["tile_height"]),
                                 d["width"], d["height"])
    return out


def example_figure(i, ex, payload, image):
    img, iw, ih = image
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35], width_ratios=[1.6, 1.0, 1.0])
    # (a) the whole panorama strip with the click
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(img, extent=(0, ex["pano_width"], ex["pano_height"], 0), interpolation="bilinear")
    ax.plot([ex["pano_x"]], [ex["pano_y"]], "o", ms=12, mfc="none", mec="#eda100", mew=2.5)
    ax.axhline(ex["pano_height"] / 2, color="#ffffff", lw=0.6, alpha=0.7)
    ax.set_title(f"{ex['role']} - {ex['label_type']}, {ex['city']} label {ex['label_id']}, "
                 f"captured {ex['capture_date'][:7]}, zoom {ex['zoom']:.0f}: the stored click (ring) on the panorama")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    # (b) crop around the click
    ax = fig.add_subplot(gs[1, 0])
    half_w, half_h = ex["pano_width"] * 0.09, ex["pano_height"] * 0.12
    x0, x1 = ex["pano_x"] - half_w, ex["pano_x"] + half_w
    y0, y1 = ex["pano_y"] - half_h, ex["pano_y"] + half_h
    sx, sy = iw / ex["pano_width"], ih / ex["pano_height"]
    crop = img[int(max(0, y0 * sy)):int(min(ih, y1 * sy)), int(max(0, x0 * sx)):int(min(iw, x1 * sx))]
    ax.imshow(crop, extent=(max(0, x0), min(ex["pano_width"], x1), min(ex["pano_height"], y1), max(0, y0)),
              interpolation="bilinear")
    ax.plot([ex["pano_x"]], [ex["pano_y"]], "o", ms=16, mfc="none", mec="#eda100", mew=3)
    ax.set_title(f"Crop: depression {ex['depression_deg']:.1f}° below the horizon")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    # (c) the depth raster the truth was read from
    ax = fig.add_subplot(gs[1, 1])
    t = gd.compute_depth_t(payload)
    col, row = mt.modern_col_row(ex["pano_x"], ex["pano_y"], ex["pano_width"], ex["pano_height"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ[::-1])
    ax.imshow(np.clip(t, 0, 40), cmap=cmap, vmin=0, vmax=40, interpolation="nearest",
              extent=(0, payload.width, payload.height, 0))
    ax.plot([col + 0.5], [row + 0.5], "o", ms=12, mfc="none", mec="#eda100", mew=2.5)
    ax.set_title(f"GSV depth (ray range, m): truth {ex['truth_m']:.2f} m on '{ex['hit_class']}'")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    # (d) plan view
    ax = fig.add_subplot(gs[1, 2])
    b = np.radians(so.pov_from_pano_xy(ex["pano_x"], ex["pano_y"], ex["pano_width"], ex["pano_height"],
                                        ex["camera_heading"])[0])
    ux, uy = np.sin(b), np.cos(b)
    far = max(ex["truth_m"], ex["A_deployed"], ex["dist_approx3"]) * 1.15 + 1
    ax.plot([0, ux * far], [0, uy * far], color="#c3c2b7", lw=1)
    ax.plot([0], [0], "s", color=C_TRUTH, ms=8, label="camera")
    ax.plot([ux * ex["truth_m"]], [uy * ex["truth_m"]], "x", color=C_TRUTH, ms=11, mew=2.5,
            label=f"depth truth {ex['truth_m']:.2f} m")
    ax.plot([ux * ex["A_deployed"]], [uy * ex["A_deployed"]], "o", color=C_REG, ms=9, mec=C_SURFACE, mew=1.5,
            label=f"2021 regression {ex['A_deployed']:.2f} m (err {ex['err_A']:.2f})")
    ax.plot([ux * ex["dist_approx3"]], [uy * ex["dist_approx3"]], "o", color=C_SHIP, ms=9, mec=C_SURFACE, mew=1.5,
            label=f"approximation3 {ex['dist_approx3']:.2f} m (err {ex['err_approx3']:.2f})")
    lim = far
    ax.set_xlim(min(-2, ux * lim - 2) if ux < 0 else -2, max(2, ux * lim + 2) if ux > 0 else 2)
    ax.set_ylim(min(-2, uy * lim - 2) if uy < 0 else -2, max(2, uy * lim + 2) if uy > 0 else 2)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title("Plan view along the label's bearing")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, ncol=1)
    fig.suptitle(f"Figure {35 + i} - worked example {i + 1}: {ex['role']}", x=0.01, ha="left", fontsize=10.5,
                 color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"fig{35 + i}-signoff-example-{i + 1}.png")


def examples(summary):
    exs = summary["examples"]
    payloads = load_payloads({e["pano_id"] for e in exs})
    images = load_tiles()
    for i, ex in enumerate(exs):
        example_figure(i, ex, payloads[ex["pano_id"]], images[ex["pano_id"]])


def main():
    summary, era, modern = load()
    fig29(summary, modern)
    fig30(summary, era)
    fig31(summary)
    fig32(summary)
    fig33(summary)
    fig34(summary)
    examples(summary)


if __name__ == "__main__":
    main()
