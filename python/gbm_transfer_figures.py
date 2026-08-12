"""Figure 28: does the #6 ceiling transfer to modern truth? (issue #6, second pass)

Read from the committed data/gbm-transfer-summary.json rather than recomputed — the
regeneration is four LightGBM retrains (python/run_gbm_transfer.py --write, 2-7 min) and the
findings tests already hold that summary to the committed data, so the figure and the
tests meet at the same artifact.

Usage (repo root): python python/gbm_transfer_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import BASELINE, INK, MUTED, SECONDARY, _save, _title  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "data", "gbm-transfer-summary.json")

C_SHIPPED = "#1baf7a"   # the shipped closed form — this repo's forward-looking geometry hue
C_CLOSED = "#86c9ac"    # the other closed forms
C_GBM = "#eb6834"       # the benchmark that must not ship
C_GBM_ALT = "#f3a683"   # its weaker arms
C_ERA = "#eb6834"       # era truth in the mechanism panel
C_MODERN = "#2a78d6"    # modern truth

# (summary key, label, colour) for the calibrated comparison — every row here carries a
# parameter fitted on the train half, so they are directly comparable.
BARS = [
    ("D_flat", "D flat (shipped)", C_SHIPPED),
    ("D_rescaled", "D rescaled", C_CLOSED),
    ("gbm_modern", "GBM trained on modern", C_GBM_ALT),
    ("gbm_l1_affine", "GBM + affine", C_GBM_ALT),
    ("only_sv_image_y_scaled", "GBM 1-D + scale", C_GBM_ALT),
    ("gbm_l1_scaled", "GBM + scale", C_GBM),
    ("gbm_dep_l1_scaled", "GBM+dep + scale", C_GBM_ALT),
    ("gbm_l1_quantile", "GBM + quantile map", C_GBM_ALT),
]
LINES = [("D_blend", "D era blend", "#b5b3ac"),
         ("D_flat", "D flat (shipped)", C_SHIPPED),
         ("gbm_l1_scaled", "GBM + scale", C_GBM)]


def main():
    with open(SUMMARY, encoding="utf-8") as f:
        s = json.load(f)
    held = s["held_out_half"]["models"]
    ci = s["held_out_half"]["bootstrap"]["ci"]

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16.6, 5.6),
                                       width_ratios=[1.25, 0.95, 1.05])

    # ---- left: the calibrated comparison, with cluster-bootstrap intervals
    rows = sorted(BARS, key=lambda r: held[r[0]]["median_abs_m"], reverse=True)
    ys = np.arange(len(rows))
    vals = [held[k]["median_abs_m"] for k, _, _ in rows]
    lo = [v - ci[k]["median_abs_m_lo"] for v, (k, _, _) in zip(vals, rows)]
    hi = [ci[k]["median_abs_m_hi"] - v for v, (k, _, _) in zip(vals, rows)]
    ax.barh(ys, vals, color=[c for _, _, c in rows], height=0.66, zorder=3)
    ax.errorbar(vals, ys, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1.1,
                capsize=3, zorder=4)
    for y, v, h_ in zip(ys, vals, hi):
        ax.text(v + h_ + 0.014, y, f"{v:.3f}", va="center", fontsize=8.6, color=SECONDARY)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=9)
    ax.set_xlabel("median absolute distance error (m), held-out panorama half")
    ax.set_xlim(0, max(v + h for v, h in zip(vals, hi)) * 1.24)
    ax.set_title(f"one modern parameter each  (n = {held['D_flat']['n']:,})", loc="left")
    ax.grid(axis="y", visible=False)
    best = min(vals)
    ax.axvline(best, color=C_SHIPPED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("the 2-parameter closed\nform wins every one",
                xy=(best, len(rows) - 0.60), xytext=(best + 0.195, len(rows) - 2.35),
                fontsize=8.5, color=SECONDARY,
                arrowprops=dict(arrowstyle="-|>", color=SECONDARY, lw=1.0,
                                connectionstyle="arc3,rad=0.2"))

    # ---- middle: the mechanism — implied camera height by panorama resolution
    diag = s["truth_scale_by_resolution"]
    # Every group the run kept, not a hand-picked three: the panel's whole claim is that
    # this axis is heterogeneous, so silently dropping a resolution would argue the point
    # by omission. "missing" (DC, which stores no pano_height) leads; the rest sort by
    # raster height.
    groups = ["missing"] + sorted(
        {g for block in ("era_truth", "modern_truth")
         for g in diag[block]["by_pano_height"] if g != "missing"}, key=int)
    labels = ["pano_height\nabsent (DC)" if g == "missing" else f"{g} px" for g in groups]
    xs = np.arange(len(groups))
    # the panel keeps one width whatever the run finds, so the type has to give way instead
    fs = 9.0 if len(groups) <= 3 else 8.0
    for off, (key, label, color) in ((-0.19, ("era_truth", "era truth (2017–2020)", C_ERA)),
                                     (+0.19, ("modern_truth", "modern truth (2021+)", C_MODERN))):
        block = diag[key]["by_pano_height"]
        vals = [block.get(g, {}).get("implied_height_m", np.nan) for g in groups]
        ns = [block.get(g, {}).get("n") for g in groups]
        ax2.bar(xs + off, vals, width=0.34, color=color, label=label, zorder=3)
        for x, v, n in zip(xs, vals, ns):
            if np.isfinite(v):
                ax2.text(x + off, v + 0.05, f"{v:.2f}", ha="center", fontsize=fs,
                         color=SECONDARY)
                compact = (f"{n/1000:.0f}k" if n >= 10000 else
                           f"{n/1000:.1f}k" if n >= 1000 else f"{n}")
                ax2.text(x + off, 0.12, compact, ha="center", fontsize=fs - 1.0,
                         color="#fcfcfb", zorder=4)
    shipped = diag["shipped_flat_height_m"]
    ax2.axhline(shipped, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=5)
    ax2.text(-0.44, shipped + 0.06, f"shipped {shipped:.3f} m", ha="left", fontsize=8.5,
             color=INK)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=fs)
    ax2.set_ylim(0, 3.5)
    ax2.set_xlim(-0.55, len(groups) - 0.45)
    ax2.set_ylabel("implied camera height, median(truth · tan dep) (m)")
    ax2.set_title("the era truth's scale is not one scale", loc="left")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="x", visible=False)

    # ---- right: where the booster's remaining structure lives
    bins = s["by_distance"]
    bx = np.arange(len(bins))
    for key, label, color in LINES:
        vals = [b["per_model"][key]["median_abs_m"] for b in bins]
        ax3.plot(bx, vals, "-o", color=color, ms=5, lw=2, zorder=3, label=label)
    ax3.set_yscale("log")
    ax3.set_yticks([0.2, 0.5, 1, 2, 5, 10])
    ax3.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax3.set_xticks(bx)
    ax3.set_xticklabels([f"{b['bin_m'].strip('[)').replace('.0', '').replace(', ', '–')}\n"
                         f"{b['n']}" for b in bins], fontsize=8.4)
    ax3.set_xlabel("true distance bin (m; second line = rows)")
    ax3.set_ylabel("median absolute distance error (m, log scale)")
    ax3.set_title("beyond 15 m the shipped form is worst", loc="left")
    ax3.set_ylim(0.13, 22)
    ax3.axvline(2.5, color=BASELINE, lw=1.0)
    ax3.text(2.38, 0.155, "shipped form ahead", ha="right", fontsize=8.5, color=MUTED)
    ax3.text(2.62, 0.155, "shipped form last", ha="left", fontsize=8.5, color=MUTED)
    ax3.legend(loc="upper left", fontsize=8.5)

    h = s["headline"]
    _title(fig,
           "The #6 ceiling does not survive a change of truth frame — it was a scale, "
           "not scene structure",
           "Left: on modern measured-plane truth, with one train-half parameter on every "
           "side, the shipped 2-parameter closed form beats every recalibrated booster; "
           "every paired cluster-bootstrap interval excludes zero. Middle: the reason — the "
           "era truth implies 2.80 m of camera height at DC and 6656-px panoramas but "
           "2.35 m at 8192 px, so a booster that can read pano_height learns which "
           "subpopulation answers on which scale. Those eight extra inputs are worth "
           f"{h['structure_worth']['era_frame']['worth_m']:.2f} m inside the era truth and "
           f"{h['structure_worth']['modern_calibrated']['worth_m']:+.2f} m outside it. "
           "Right: beyond 15 m the shipped form is the worst of the three — but the era "
           "blend, biased 1.07 m long with no conditional structure, beats the booster in "
           "two of those three bins, so most of that far-field deficit is the shipped "
           "saturation rather than structure a closed form is missing.", wrap=168)
    fig.subplots_adjust(top=0.70, bottom=0.12, wspace=0.32)
    _save(fig, "fig28-gbm-transfer.png")


if __name__ == "__main__":
    main()
