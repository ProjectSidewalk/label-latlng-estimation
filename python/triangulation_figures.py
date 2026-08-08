"""Figures 24-27: bearing-only triangulation (issue #7).

Read from the committed ``data/triangulation-summary.json`` rather than recomputed — the
build is a multi-minute pass over six auto-labeler runs, and the findings tests already
hold the summary to the committed inputs, so the figures and the tests meet at the same
artifact (the pattern ``gbm_ceiling_figures.py`` established).

  fig24  the headline: implied camera height per run, bearings only, against the two
         reference heights — plus the planted-height recovery that says the method does
         not manufacture it
  fig25  the shape test: implied height vs depression angle. Flat = the flat-ground
         cotangent is right *absolutely*, which self-consistency could never show
  fig26  the error budget: perpendicular miss vs range, decomposed into the bearing and
         panorama-position terms
  fig27  every distance model scored against triangulated truth, by true range

Usage (repo root): python python/triangulation_figures.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (  # noqa: E402
    BASELINE, GRID, INK, MUTED, SECONDARY, _save, _title,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "data", "triangulation-summary.json")

C_GSV = "#2a78d6"        # GSV runs — the repo's status-quo blue
C_MAPILLARY = "#eb6834"  # Mapillary runs
C_SHIPPED = "#1baf7a"    # the shipped blend / measured modern rig
C_ASSUMED = "#898781"    # the ecosystem's assumed 2.6 m

MODEL_LABEL = {
    "deployed_linear": "deployed linear (2021)",
    "normalized_linear": "#4765 normalization alone",
    "cotangent_2p6": "cotangent @ 2.6 m",
    "era_blend": "era blend (#12)",
    "shipped_blend": "shipped blend (final_coefficients)",
}
MODEL_COLOR = {
    "deployed_linear": "#2a78d6", "normalized_linear": "#86b6ef",
    "cotangent_2p6": "#898781", "era_blend": "#eb6834",
    "shipped_blend": "#1baf7a",
}


def _load():
    with open(SUMMARY, encoding="utf-8") as f:
        return json.load(f)


def _runs(s):
    """Runs in a stable order: GSV first, then Mapillary, each alphabetical."""
    gsv = sorted(r for r, v in s["imagery"].items() if v == "gsv")
    mpl = sorted(r for r, v in s["imagery"].items() if v == "mapillary")
    return gsv + mpl


def _bin_mid(label):
    """Midpoint of a pandas Interval string like '(5, 10]'."""
    nums = re.findall(r"-?\d+\.?\d*", label)
    return (float(nums[0]) + float(nums[1])) / 2.0 if len(nums) >= 2 else np.nan


# ------------------------------------------------------------------ fig24: the headline
def fig24(s):
    runs = _runs(s)
    shipped = s["meta"]["shipped_height_m"]
    assumed = s["meta"]["assumed_height_m"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), width_ratios=[1.5, 1.0])

    ys = np.arange(len(runs))[::-1]
    ax.axvspan(shipped, assumed, color=GRID, alpha=0.45, zorder=1, lw=0)
    for y, run in zip(ys, runs):
        g = s["scale_global"][run]
        med = s["scale"][run]["median_m"]
        color = C_GSV if s["imagery"][run] == "gsv" else C_MAPILLARY
        lo, hi = g.get("ci95_m", [g["height_m"], g["height_m"]])
        ax.plot([lo, hi], [y, y], color=color, lw=3.0, solid_capstyle="round", zorder=4)
        ax.plot([g["height_m"]], [y], "o", color=color, ms=7.5, zorder=5)
        # the noisier per-member median, shown so the two estimators can be compared
        ax.plot([med], [y], "|", color=color, ms=11, mew=1.6, alpha=0.55, zorder=4)
        ax.text(g["height_m"], y + 0.26, f"{g['height_m']:.3f}", ha="center",
                fontsize=9, color=color, weight="bold")

    ax.axvline(shipped, color=C_SHIPPED, lw=1.8, zorder=3)
    ax.axvline(assumed, color=C_ASSUMED, lw=1.5, ls="--", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r}\n{s['imagery'][r]}, n={s['scale_global'][r]['n']:,}"
                        for r in runs], fontsize=8.5)
    ax.set_ylim(-1.15, len(runs) - 0.35)
    ax.set_xlim(shipped - 0.06, assumed + 0.055)
    ax.text(shipped, -1.08, f"shipped 2.341 m\n(depth, human clicks)", color=C_SHIPPED,
            fontsize=8.5, va="bottom", ha="center")
    ax.text(assumed, -1.08, f"assumed 2.6 m\n(auto-labeler)", color=C_ASSUMED,
            fontsize=8.5, va="bottom", ha="center")
    ax.text(0.5, -0.055, "every run lands inside this band", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.set_xlabel("camera height (m) implied by multi-view ray geometry")
    ax.set_title("What the bearings imply, with no depth data", loc="left")
    ax.grid(axis="y", visible=False)

    # right: planted-height recovery — the method does not manufacture height
    real = s.get("validation", {}).get("real_geometry", {})
    rr = [r for r in runs if r in real and real[r].get("recovered_height_m")]
    if rr:
        planted = real[rr[0]]["planted_height_m"]
        xs = np.arange(len(rr))
        got = [real[r]["recovered_height_m"] for r in rr]
        raw = [real[r]["recovered_height_uncorrected_m"] for r in rr]
        ax2.axhline(planted, color=C_SHIPPED, lw=1.6, zorder=2)
        ax2.plot(xs, raw, "o", color=BASELINE, ms=6, zorder=3)
        ax2.plot(xs, got, "o", color=INK, ms=6, zorder=4)
        for x, a, b in zip(xs, raw, got):
            ax2.plot([x, x], [a, b], color=BASELINE, lw=1.0, zorder=2)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(rr, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("recovered height (m)")
        span = max(0.02, max(abs(np.array(raw) - planted).max(),
                             abs(np.array(got) - planted).max()) * 1.8)
        ax2.set_ylim(planted - span, planted + span)
        ax2.text(0.02, 0.97, f"planted {planted:.3f} m", transform=ax2.transAxes,
                 color=C_SHIPPED, fontsize=9, va="top")
        ax2.text(0.02, 0.06,
                 "grey = before the norm-convexity correction\nblack = after",
                 transform=ax2.transAxes, color=MUTED, fontsize=8.5, va="bottom")
        ax2.set_title("Plant a known height, run the whole pipeline", loc="left")

    _title(fig, "Fig 24 - The camera height the bearings imply",
           "Each object is fixed by the intersection of several panoramas' bearings, and "
           "the camera height is whatever makes those views agree. No vertical model, no "
           "camera height, no depth and no panorama resolution enters anywhere. Dot = the "
           "global scale fit with its 95% site-bootstrap interval; tick = the noisier "
           "per-member median. Right: the same pipeline re-run on each site's real "
           "geometry with a KNOWN height planted and that run's measured noise re-applied "
           "- it returns what it was given, so the spread on the left is the rigs and the "
           "detector, not the estimator.")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, "fig24-triangulation-implied-height.png")


# --------------------------------------------------- fig25: the absolute shape test
def fig25(s):
    runs = _runs(s)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    ends = []
    for run in runs:
        by = s["robustness"][run]["by_depression_deg"]
        if not by:
            continue
        pts = sorted(((_bin_mid(k), v["median_m"]) for k, v in by.items()))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        color = C_GSV if s["imagery"][run] == "gsv" else C_MAPILLARY
        ax.plot(xs, ys, "-o", color=color, ms=4, lw=1.7, alpha=0.9, zorder=3)
        ends.append((ys[-1], xs[-1], run, color))

    # stack the end labels so they never overlap
    ends.sort()
    prev = -1e9
    for y, x, run, color in ends:
        y = max(y, prev + 0.028)
        ax.text(x + 1.5, y, run, fontsize=8.5, color=color, va="center")
        prev = y
    ax.axhline(s["meta"]["shipped_height_m"], color=C_SHIPPED, lw=1.5, zorder=2)
    ax.set_xlim(right=ax.get_xlim()[1] + 12)
    ax.text(ax.get_xlim()[1], s["meta"]["shipped_height_m"] - 0.006, "shipped 2.341 m ",
            color=C_SHIPPED, fontsize=8.5, va="top", ha="right")
    ax.set_xlabel("depression angle below horizon (deg)")
    ax.set_ylabel("implied camera height (m)")
    ax.set_title("It is not flat - and this is the open finding", loc="left")

    # right: the fuse-gate selection probe, which is the test that IS about selection
    for run in runs:
        strata = s["robustness"][run]["fuse_gate_selection"]
        pts = [(v["median_spread_m"], v["implied_height_m"])
               for v in strata.values() if isinstance(v, dict)]
        if len(pts) < 2:
            continue
        pts.sort()
        color = C_GSV if s["imagery"][run] == "gsv" else C_MAPILLARY
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color, ms=4,
                 lw=1.7, alpha=0.9, zorder=3)
    ax2.axhline(s["meta"]["shipped_height_m"], color=C_SHIPPED, lw=1.5, zorder=2)
    ax2.axhline(s["meta"]["assumed_height_m"], color=C_ASSUMED, lw=1.4, ls="--", zorder=2)
    ax2.text(ax2.get_xlim()[1], s["meta"]["assumed_height_m"], "assumed 2.6 m ",
             color=C_ASSUMED, fontsize=8.5, va="bottom", ha="right")
    ax2.set_xlabel("within-site spread of member ranges (m)")
    ax2.set_ylabel("implied camera height (m)")
    ax2.set_title("Flat here = the fuse gate did not select the scale", loc="left")

    _title(fig, "Fig 25 - The shape test the scale-free diagnostics could not run",
           "A camera height is a property of the rig, so if the flat-ground cotangent is "
           "the right form the height it implies must not depend on where in the image the "
           "click sits. It does (left): every run climbs with depression. Right is the "
           "selection control - the auto-labeler fused at 2.6 m, and a wrong height only "
           "pushes a site's members apart in proportion to how much their ranges differ, "
           "so a scale manufactured by that gate would climb toward the dashed line. It "
           "does not, on any run.")
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    _save(fig, "fig25-triangulation-shape.png")


# ------------------------------------------------------------- fig26: the error budget
def fig26(s):
    runs = _runs(s)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), width_ratios=[1.3, 1.0])

    for run in runs:
        noise = s["noise"][run]
        bins = noise.get("bins_final") or []
        color = C_GSV if s["imagery"][run] == "gsv" else C_MAPILLARY
        if bins:
            bins = sorted(bins, key=lambda b: b["r2"])
            r = np.sqrt([b["r2"] for b in bins])
            sd = np.sqrt(np.maximum([b["var_miss_m2"] for b in bins], 0))
            ax.plot(r, sd, "o", color=color, ms=4.5, alpha=0.85, zorder=3)
            grid = np.linspace(r.min(), r.max(), 60)
            fit = np.sqrt((grid * np.radians(noise["sigma_bearing_deg"])) ** 2
                          + noise["sigma_pos_m"] ** 2)
            ax.plot(grid, fit, "-", color=color, lw=1.6, alpha=0.9, zorder=2)
    ax.set_xlabel("triangulated range (m)")
    ax.set_ylabel("perpendicular miss, robust sigma (m)")
    ax.set_title("Two noise sources, told apart by their range dependence", loc="left")

    xs = np.arange(len(runs))
    sb = [s["noise"][r]["sigma_bearing_deg"] for r in runs]
    sp = [s["noise"][r]["sigma_pos_m"] for r in runs]
    colors = [C_GSV if s["imagery"][r] == "gsv" else C_MAPILLARY for r in runs]
    ax2.bar(xs - 0.19, sb, 0.36, color=colors, alpha=0.55, zorder=3)
    ax2b = ax2.twinx()
    ax2b.bar(xs + 0.19, sp, 0.36, color=colors, zorder=3)
    ax2b.grid(False)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(runs, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("sigma bearing (deg)   [pale]", fontsize=9)
    ax2b.set_ylabel("sigma panorama position (m)   [solid]", fontsize=9)
    ax2.set_title("GSV positions beat Mapillary's SfM", loc="left")

    _title(fig, "Fig 26 - The error budget, measured rather than assumed",
           "A bearing error misses the object by an amount proportional to range; a "
           "panorama position error misses by an amount that does not depend on range. "
           "Regressing the squared perpendicular miss on squared range separates them, "
           "which fixes the triangulation weights, the conditioning gate, and the size of "
           "the norm-convexity correction.")
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "fig26-triangulation-error-budget.png")


# ------------------------------------------- fig27: models against triangulated truth
def fig27(s):
    runs = _runs(s)
    n = len(runs)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.4, 3.5 * nrow), squeeze=False)

    for i, run in enumerate(runs):
        ax = axes[i // ncol][i % ncol]
        by = s["by_range"][run]
        order = sorted(by, key=_bin_mid)
        xs = [_bin_mid(k) for k in order]
        for key in MODEL_LABEL:
            ys = [by[k]["models"][key] for k in order]
            ax.plot(xs, ys, "-o", color=MODEL_COLOR[key], ms=4, lw=1.6, zorder=3)
        ax.axhline(0, color=BASELINE, lw=1.2, zorder=2)
        ax.set_title(f"{run} ({s['imagery'][run]})", loc="left", fontsize=10)
        ax.set_xlabel("triangulated range (m)")
        if i % ncol == 0:
            ax.set_ylabel("signed error (m)")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles = [plt.Line2D([], [], color=MODEL_COLOR[k], lw=2, marker="o", ms=4,
                          label=MODEL_LABEL[k]) for k in MODEL_LABEL]
    axes[0][ncol - 1].legend(handles=handles, loc="upper right", fontsize=8)

    _title(fig, "Fig 27 - Every distance model against triangulated truth",
           "Signed error (predicted minus triangulated) by true range. A model with the "
           "right shape is flat; one that is compressive slopes down. This is an absolute "
           "score on imagery none of these models was fit on - including the two Mapillary "
           "cities, where no depth ground truth exists or ever will.")
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.09 / nrow])
    _save(fig, "fig27-triangulation-model-scoring.png")


def main():
    s = _load()
    fig24(s)
    fig25(s)
    fig26(s)
    fig27(s)


if __name__ == "__main__":
    main()
