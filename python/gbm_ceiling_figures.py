"""Figure 19: the GBM ceiling benchmark (issue #6) — the gap by distance, and its price in noise.

Unlike figures 17-18, the numbers here are read from the committed
data/gbm-ceiling-summary.json rather than recomputed: regeneration is a ~6-minute LightGBM
retrain (python/run_gbm_ceiling.py --write), and the findings tests already hold the summary
to the committed data, so the figure and the tests meet at the same artifact.

Usage (repo root): python python/gbm_ceiling_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import BASELINE, INK, MUTED, SECONDARY, _save, _title  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "data", "gbm-ceiling-summary.json")

C_EST7 = "#2a78d6"   # the status quo is always blue in this repo
C_GEOM = "#1baf7a"   # the shipped blend (forward-looking geometry)
C_GBM = "#eb6834"    # the benchmark that must not ship — the repo's not-the-way-forward hue
MODELS = [("A_ols", "A status quo", C_EST7),
          ("D_blend_type_l1", "D shipped blend", C_GEOM),
          ("gbm_l1", "GBM L1 (ceiling)", C_GBM)]


def main():
    with open(SUMMARY, encoding="utf-8") as f:
        s = json.load(f)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.7), width_ratios=[1.25, 1.0])

    # left: median lat/lng error by true-distance bin (gbm_dep_l1 is indistinguishable
    # from gbm_l1 at this scale and is omitted; both are in the report table)
    bins = s["error_vs_distance"]
    xs = range(len(bins))
    for key, label, color in MODELS:
        ys = [b["per_model"][key]["latlng_median_m"] for b in bins]
        ax.plot(xs, ys, "-o", color=color, ms=5, lw=2, zorder=3)
        ax.text(len(bins) - 1 + 0.18, ys[-1], label, color=color, fontsize=9,
                va="center")
    d_1015 = bins[3]["per_model"]["D_blend_type_l1"]["latlng_median_m"]
    ax.annotate("the blend's one traded bin\n(behind even A; the GBM holds it)",
                xy=(3.05, d_1015 * 1.06), xytext=(2.3, 5.2), ha="center",
                fontsize=8.5, color=SECONDARY,
                arrowprops=dict(arrowstyle="-|>", color=SECONDARY, lw=1.0,
                                connectionstyle="arc3,rad=-0.15"))
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xticks(list(xs))
    labels = [b["bin_m"].replace("[", "").replace(")", "").replace(".0", "")
               .replace(", ", "–") for b in bins]
    ns = [f"{b['n']/1000:.1f}k" if b["n"] >= 1000 else str(b["n"]) for b in bins]
    ax.set_xticklabels([f"{l}\n{n}" for l, n in zip(labels, ns)], fontsize=8.4)
    ax.set_xlim(-0.4, len(bins) + 1.0)
    ax.set_xlabel("true distance bin (m; second line = test views)")
    ax.set_ylabel("median lat/lng error (m, log scale)")
    ax.set_title("test error by true distance  (n = 79,029)", loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # right: click-noise sweep — degradation vs sigma, same seeded draws as #3
    sweep = s["noise_sweep"]
    sigmas = [0.0] + [float(x) for x in sweep["sigmas_px"]]
    for key, label, color in MODELS:
        ys = [0.0] + [sweep["per_model"][key][f"{x:.1f}"]["delta_median_m"]
                      for x in sweep["sigmas_px"]]
        ax2.plot(sigmas, ys, "-o", color=color, ms=5, lw=2, zorder=3)
        ax2.text(sigmas[-1] + 0.25, ys[-1], label, color=color, fontsize=9, va="center")
    ax2.axhline(0, color=BASELINE, lw=1)
    g2 = sweep["per_model"]["gbm_l1"]["2.0"]["delta_median_m"]
    d2 = sweep["per_model"]["D_blend_type_l1"]["2.0"]["delta_median_m"]
    ax2.annotate(f"{g2 / d2:.1f}× the blend's\ndegradation at 2 px",
                 xy=(2, g2), xytext=(2.6, 0.115), fontsize=8.5, color=SECONDARY,
                 arrowprops=dict(arrowstyle="-|>", color=SECONDARY, lw=1.0))
    ax2.set_xticks(sigmas)
    ax2.set_xlim(-0.4, 13.2)
    ax2.set_xlabel("click-noise sigma (px on canvas x/y)")
    ax2.set_ylabel("increase in median lat/lng error (m)")
    ax2.set_title("degradation under seeded click noise", loc="left")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="-", color=c, label=l)
               for _, l, c in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncols=3, frameon=False, fontsize=9, columnspacing=1.6, handletextpad=0.5)
    _title(fig, "The GBM ceiling: a real gap that widens with distance, paid for in noise sensitivity",
           "Left: the GBM benchmark (explicitly not shippable) beats the shipped blend in every "
           "true-distance bin. Right: the #3 report's seeded click-noise sweep — the GBM "
           "degrades 4–5× faster at small sigma; the structure that buys the ceiling is "
           "what noise destroys first.")
    fig.subplots_adjust(top=0.76, bottom=0.24, wspace=0.30)
    _save(fig, "fig19-gbm-ceiling.png")


if __name__ == "__main__":
    main()
