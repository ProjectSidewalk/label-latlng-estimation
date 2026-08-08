"""Figures 20-23: the modern-truth absolute validation (issue #3).

Everything is recomputed from the committed data/modern-truth-* artifacts — no numbers
are read from the committed summary (the findings tests hold the two together instead).

Usage (repo root): python python/modern_truth_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import BASELINE, GRID, INK, MUTED, SECONDARY, _save, _title  # noqa: E402
import modern_truth as mt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

C_EST7 = "#2a78d6"    # the status quo is always blue in this repo
C_GEOM = "#1baf7a"    # forward-looking geometry (the shipped blend)
C_NORM = "#8fb8e8"    # the #4765 one-liner (normalization only)
C_REF = "#52514e"     # zero-parameter reference (the repo's baseline gray, CVD-safe vs green)

MODELS = [
    ("A_deployed", "A deployed (raw px)", C_EST7),
    ("B_normalized", "B #4765-normalized", C_NORM),
    ("C_anchor", "C 2.6 m cotangent", C_REF),
    ("D_blend", "D shipped blend", C_GEOM),
]


def load():
    labels = pd.read_csv(os.path.join(DATA, "modern-truth-labels.csv.gz"),
                         dtype={"pano_id": str})
    panos = pd.read_csv(os.path.join(DATA, "modern-truth-panos.csv.gz"),
                        dtype={"pano_id": str})
    human = labels[labels["gate_ok"] & ~labels["is_ai"]].copy()
    return labels, panos, human


def binned_median(x, y, edges, min_n=30):
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            xs.append((lo + hi) / 2)
            ys.append(float(np.median(y[m])))
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fig20(human, labels):
    """A deployed | D as shipped | D rescaled (held-out half) — the full arc."""
    import modern_truth as mt

    params = mt.load_blend_params(DATA)
    remedy = mt.remedy_check(labels, params)
    flat_h = remedy["flat_height_m"]  # the train half's value; the disjoint half scores it
    rng = np.random.default_rng(mt.SEED)
    panos = np.sort(human["pano_id"].unique())
    train_ids = set(rng.choice(panos, len(panos) // 2, replace=False))
    test = human[~human["pano_id"].isin(train_ids)].copy()
    flat = {"form": "blend", "blend_deg": params["blend_deg"], "height_m": flat_h}
    test["D_flat"] = mt.predict_dist(flat, test)

    panels = [
        (human, MODELS[0][0], MODELS[0][1], MODELS[0][2]),
        (human, "D_blend", "D blend, era per-type heights", C_GEOM),
        (test, "D_flat", f"D flat {flat_h:.2f} m (held-out half)", "#0d6b4a"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharex=True, sharey=True)
    edges = np.arange(0, 51, 2)
    for ax, (rows, key, label, color) in zip(axes, panels):
        truth = rows["truth_m"].to_numpy(float)
        pred = rows[key].to_numpy(float)
        ax.hexbin(truth, pred, gridsize=44, extent=(0, 50, 0, 50), cmap="Greys",
                  mincnt=1, linewidths=0)
        ax.plot([0, 50], [0, 50], color=BASELINE, lw=1.2, zorder=3)
        bx, by, _ = binned_median(truth, pred, edges)
        ax.plot(bx, by, color=color, lw=2.2, zorder=4)
        err = pred - truth
        ax.text(0.03, 0.97, label, color=color, fontsize=10.5, weight="bold",
                transform=ax.transAxes, va="top")
        ax.text(0.03, 0.90,
                f"median |err| {np.median(np.abs(err)):.2f} m\n"
                f"signed median {np.median(err):+.2f} m",
                color=INK, fontsize=9, transform=ax.transAxes, va="top")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_xlabel("depth-derived true distance (m)")
    axes[0].set_ylabel("predicted distance (m)")
    _title(fig, "The absolute check self-consistency provably could not do",
           "Predicted vs depth-derived true ground distance on post-2021 human labels, all current GSV "
           "cities (gated rows; colored line = per-2 m binned median, gray diagonal = truth). Left: the "
           "deployed linear model bends across the diagonal — #4766's compression against absolute "
           "truth. Middle: the blend has the right shape but runs ~13% far — its heights carry the era "
           "truth's pinned-plane scale. Right: one flat modern-calibrated height, fitted on the other "
           "half of the panos, puts it on the diagonal at 0.41 m median error — the shipped fix.",
           wrap=128)
    fig.subplots_adjust(top=0.72, wspace=0.07)
    _save(fig, "fig20-modern-truth-pred-vs-truth.png")


def fig21(human):
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    dep = human["depression_deg"].to_numpy(float)
    truth = human["truth_m"].to_numpy(float)
    edges = np.concatenate([np.arange(0, 12, 1.0), np.arange(12, 30, 2.0), [34, 40, 50]])
    ax.axvspan(0, mt.NEAR_HORIZON_DEG, color=GRID, alpha=0.6, zorder=0)
    ax.axvline(11.25, color=BASELINE, lw=1.0, ls=(0, (4, 3)))
    ax.text(11.25, 6.4, "blend joint 11.25°", color=MUTED, fontsize=8.5, ha="center")
    ax.text(mt.NEAR_HORIZON_DEG / 2, 6.4, "≤2°", color=MUTED, fontsize=8.5, ha="center")
    ax.axhline(0, color=BASELINE, lw=1)
    label_y = {"A_deployed": -2.0, "B_normalized": -2.9, "C_anchor": -0.15,
               "D_blend": 0.95}  # staggered so the two converging pairs don't collide
    for key, label, color in MODELS:
        err = human[key].to_numpy(float) - truth
        bx, by, _ = binned_median(dep, err, edges)
        ls = (0, (5, 2)) if key == "C_anchor" else "-"
        ax.plot(bx, by, color=color, lw=2.0, ls=ls, zorder=3)
        ax.annotate(label, xy=(bx[-1], by[-1]), xytext=(46.5, label_y[key]),
                    color=color, fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                    shrinkA=2, shrinkB=2))
    ax.set_xlim(0, 50)
    ax.set_ylim(-4.5, 7)
    ax.set_xlabel("depression below the horizon (deg)")
    ax.set_ylabel("median signed distance error (m)")
    _title(fig, "Where each candidate is wrong, by viewing angle — on absolute truth",
           "Median (predicted − true) in depression bins, post-2021 human labels (bins with ≥30 rows). "
           "The deployed model and the raw #4765 normalization inherit the linear form's angle-dependent "
           "bias; the raw cotangent (dashed gray) diverges toward the horizon; the blend stays near zero "
           "and is clamped in the shaded ≤2° regime where a cotangent has no finite answer.")
    fig.subplots_adjust(top=0.80)
    _save(fig, "fig21-modern-truth-error-vs-depression.png")


def fig22(human, panos):
    params = mt.load_blend_params(DATA)
    heights = mt.implied_heights(human, params)
    order = sorted(heights, key=lambda t: heights[t]["implied_height_m"])
    ok = panos[panos["status"] == "ok"]
    measured = ok.loc[~ok["ground_d_exactly_2p5"].fillna(True).astype(bool),
                     "ground_height_m"].astype(float)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axvspan(measured.quantile(0.25), measured.quantile(0.75), color=GRID, alpha=0.6)
    ax.axvline(measured.median(), color=BASELINE, lw=1.2)
    ax.text(measured.median(), len(order) - 0.35,
            f"modern rig height\nmedian {measured.median():.2f} m (IQR shaded)",
            color=MUTED, fontsize=8, ha="center", va="top")
    rng = np.random.default_rng(mt.SEED)
    for i, t in enumerate(order):
        sub = human[(human["label_type"] == t)
                    & (human["depression_deg"] >= 5.0)]
        implied = (sub["truth_m"].to_numpy(float)
                   * np.tan(np.radians(sub["depression_deg"].to_numpy(float))))
        boot = [np.median(rng.choice(implied, len(implied))) for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        med = heights[t]["implied_height_m"]
        fallback = heights[t]["uses_fallback"]
        color = "#b0570f" if fallback else C_GEOM
        ax.plot([lo, hi], [i, i], color=color, lw=2.0, zorder=3)
        ax.plot(med, i, "o", ms=6, color=color, zorder=4)
        fitted = (params["height_fallback_m"] if fallback
                  else params["height_by_type_m"][t])
        ax.plot(fitted, i, marker="D", ms=6.5, mfc="none",
                mec=INK, mew=1.2, zorder=5)
        ax.text(2.955, i, f"n={heights[t]['n']:,}" + ("  (fallback)" if fallback else ""),
                color=MUTED, fontsize=8, va="center", ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(2.0, 2.97)
    ax.set_xlabel("effective camera height (m): truth × tan(depression), median")
    handles = [
        plt.Line2D([], [], marker="o", ls="-", color=C_GEOM, label="implied by modern truth (95% CI)"),
        plt.Line2D([], [], marker="o", ls="-", color="#b0570f",
                   label="never-fitted type (scored via fallback)"),
        plt.Line2D([], [], marker="D", ls="", mfc="none", mec=INK, label="blend's fitted height"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncols=3, fontsize=8.5, frameon=False, columnspacing=1.2)
    _title(fig, "The per-type height spread does not replicate — every type implies the measured rig",
           "Effective camera height each type's modern truth implies (dep ≥ 5°, human labels) against "
           "the blend's fitted heights (hollow diamonds; Crosswalk and Signal were never fitted and "
           "score through height_fallback_m). Every implied height lands in the measured rig band "
           "(2.35 m), flat across types: the fitted table's extra 0.20–0.45 m is the era truth's "
           "pinned-plane scale plus its terrain bias, not label-click geometry.", wrap=118)
    fig.subplots_adjust(top=0.80)
    _save(fig, "fig22-modern-truth-implied-heights.png")


def fig23(labels, summary_controls):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4), width_ratios=[1, 1])
    controls = list(summary_controls)
    share = [summary_controls[c]["ground_or_terrain_share"] for c in controls]
    err = [summary_controls[c]["D_blend_median_abs_m"] for c in controls]
    colors = [C_GEOM if c == "identity" else MUTED for c in controls]
    ax.barh(range(len(controls)), share, color=colors, height=0.62)
    for i, v in enumerate(share):
        ax.text(v + 0.015, i, f"{v:.0%}", color=INK, fontsize=9, va="center")
    ax.set_yticks(range(len(controls)))
    ax.set_yticklabels(controls, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("labels whose ray lands on ground/terrain")
    ax2.barh(range(len(controls)), err, color=colors, height=0.62)
    for i, v in enumerate(err):
        ax2.text(v + 0.25, i, f"{v:.2f} m", color=INK, fontsize=9, va="center")
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.set_xlabel("blend median |error| vs that frame's truth (m)")
    for a in (ax, ax2):
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
    _title(fig, "The frame is right — and which control has discriminating power",
           "The same labels re-read under the pilot's wrong-frame null hypotheses. The vertical "
           "conventions are annihilated (1% of rays still hit ground). The x-mirror is only weakly "
           "separated by ground DISTANCE — a road is nearly left-right symmetric in range — so the "
           "mirror is rejected by the pixel-level cross-check instead: on the 409 pilot payloads the "
           "mirrored column formula agrees with the era lookup exactly nowhere (findings tests).",
           wrap=118)
    fig.subplots_adjust(top=0.76, wspace=0.10)
    _save(fig, "fig23-modern-truth-frame-controls.png")


def main():
    import json

    labels, panos, human = load()
    fig20(human, labels)
    fig21(human)
    fig22(human, panos)
    # the control sweep needs the payloads; reuse the committed summary's block, which the
    # findings tests re-derive from the committed payloads in-process
    with open(os.path.join(DATA, "modern-truth-summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    fig23(labels, summary["frame_controls"])


if __name__ == "__main__":
    main()
