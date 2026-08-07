"""Figures 17-18: the Stage 3 Mapillary falsification (issue #3).

Everything is recomputed from the same library code the findings tests verify, on the
committed data/falsification-* inputs — no numbers are read from the committed summary
(the findings tests hold the two together instead).

Usage (repo root): python python/falsification_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import BASELINE, INK, MUTED, SECONDARY, _save, _title  # noqa: E402
import mapillary_falsification as mf  # noqa: E402

C_EST7 = "#2a78d6"       # the status quo is always blue in this repo
C_GEOM = "#1baf7a"       # forward-looking geometry (the shipped blend)
C_LEGACY = "#eb6834"     # legacy behavior / the defect
C_NORM = "#8fb8e8"       # the #4765 one-liner (normalization only)

RUNS = ["richmond", "clovis", "paterson", "gainesville", "bend", "sao_paulo"]
MODELS = [
    ("A_status_quo", "A status quo (raw px)", C_EST7),
    ("B_normalized", "B #4765-normalized", C_NORM),
    ("C_cotangent", "C raw cotangent", MUTED),
    ("D_blend", "D shipped blend", C_GEOM),
]
# SidewalkWebpage#4766's published range slopes (its fuse snapshot, for validation)
PUBLISHED_4766 = {("richmond", "B_normalized"): -0.2901, ("richmond", "C_cotangent"): 0.1207,
                  ("paterson", "B_normalized"): -0.4496, ("paterson", "C_cotangent"): 0.0983}
XCLIP = -0.58


def fig17(diags):
    from matplotlib.transforms import blended_transform_factory

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.axvline(0, color=BASELINE, lw=1)
    ypos = {run: len(RUNS) - i for i, run in enumerate(RUNS)}
    offsets = np.linspace(0.27, -0.27, len(MODELS))
    for (key, label, color), dy in zip(MODELS, offsets):
        for run in RUNS:
            s = diags[run]["per_model"][key]["range_slope"]
            x, y = s["slope"], ypos[run] + dy
            clipped = x < XCLIP
            xd = max(x, XCLIP)
            ax.errorbar(xd, y, xerr=2 * s["se"], fmt="o", ms=5, color=color,
                        ecolor=color, elinewidth=1.2, capsize=0, zorder=3)
            if clipped:
                ax.annotate("", xy=(XCLIP - 0.035, y), xytext=(XCLIP + 0.02, y),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
                ax.text(XCLIP + 0.055, y + 0.22, f"{x:+.2f}", color=color, fontsize=8.5)
            pub = PUBLISHED_4766.get((run, key))
            if pub is not None:
                ax.plot(pub, y, marker="D", ms=6, mfc="none", mec=INK, mew=1.1, zorder=4)
    margin = blended_transform_factory(ax.transAxes, ax.transData)
    for run in RUNS:
        y = ypos[run]
        src = "Mapillary" if run in mf.MAPILLARY_RUNS else "GSV"
        n = diags[run]["n_members"]
        ax.text(-0.015, y + 0.10, run, ha="right", va="center", color=INK, fontsize=10,
                transform=margin)
        ax.text(-0.015, y - 0.22, f"{src}, {n:,} views", ha="right", va="center",
                color=MUTED, fontsize=8, transform=margin)
        if y > 1:
            ax.axhline(y - 0.5, color=BASELINE, lw=0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for _, l, c in MODELS]
    handles.append(plt.Line2D([], [], marker="D", ls="", mfc="none", mec=INK,
                              label="#4766 published (its earlier fuse)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncols=3, frameon=False, fontsize=9, columnspacing=1.4, handletextpad=0.4)
    ax.set_yticks([])
    ax.set_ylim(0.4, len(RUNS) + 0.6)
    ax.set_xlim(XCLIP - 0.09, 0.22)
    ax.set_xlabel("within-site range slope (m/m; 0 = no compression)  ±2 SE")
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    _title(fig, "The falsification's first axis: compression is gone under the shipped blend",
           "Along-ray residual vs predicted range, demeaned within fused multi-view sites "
           "(#4766's scale-free diagnostic, reimplemented). Negative = far views under-shoot "
           "their peers. The status quo compresses everywhere — catastrophically on clovis's "
           "2880 px panos — while the blend sits within ±0.09 of flat on every run; hollow "
           "diamonds: #4766's published values where rig mix cannot move them.")
    fig.subplots_adjust(top=0.80)
    _save(fig, "fig17-falsification-range-axis.png")


def fig18(diags, seq_scales, seq_table):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), width_ratios=[1.0, 1.2])

    # left: richmond's height axis — the height-blind band, the per-sequence collapse, and
    # the held-out transfer that says the collapse is not just the added parameters
    r = diags["richmond"]["per_model"]
    scaled = seq_scales["richmond"]["d_blend_per_sequence_scale"]
    hold = seq_scales["richmond"]["holdout"]
    bars = [("A\nraw px", r["A_status_quo"]["height_slope"], C_EST7),
            ("B\nnorm.", r["B_normalized"]["height_slope"], C_NORM),
            ("C\ncotan.", r["C_cotangent"]["height_slope"], MUTED),
            ("D\nblend", r["D_blend"]["height_slope"], C_GEOM),
            ("D + k\nin-sample", scaled["height_slope"], "#0d6b4a"),
            ("D + k\nheld out", hold["d_blend_transferred_scale"]["height_slope"], "#0d6b4a")]
    band = max(abs(r["B_normalized"]["height_slope"]["slope"]),
               abs(r["C_cotangent"]["height_slope"]["slope"]))
    ax.axhspan(-band, band, color=BASELINE, alpha=0.45, zorder=0)
    ax.axhline(0, color=BASELINE, lw=1)
    for i, (label, s, color) in enumerate(bars):
        ax.bar(i, s["slope"], width=0.62, color=color, zorder=2,
               hatch="///" if i == len(bars) - 1 else None, edgecolor="white" if i == len(bars) - 1 else None)
        ax.errorbar(i, s["slope"], yerr=2 * s["se"], fmt="none", ecolor=INK,
                    elinewidth=1.1, zorder=3)
    # the held-out half's own unscaled slope, so the last bar is read against its own baseline
    hu = hold["d_blend_unscaled"]["height_slope"]["slope"]
    ax.plot(len(bars) - 1, hu, marker="_", ms=18, mew=2.0, color=C_GEOM, zorder=4)
    ax.annotate("its own unscaled D", xy=(len(bars) - 1, hu), xytext=(3.4, -0.44),
                color=C_GEOM, fontsize=7.6, ha="center",
                arrowprops=dict(arrowstyle="-", color=C_GEOM, lw=0.9))
    ax.text(1.5, 0.20, "height-blind models (B, C)", color=SECONDARY, fontsize=8.5, ha="center")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([l for l, _, _ in bars], fontsize=8.0)
    ax.set_ylim(-0.78, 0.30)
    ax.set_ylabel("within-site height slope (per h/6656)")
    ax.set_title("richmond: residual height dependence  (whiskers ±2 SE)", loc="left",
                 fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # right: per-sequence relative camera heights by rig (the fit the caller already ran)
    fitted = seq_table[seq_table["n_fit_members"] >= mf.SEQ_MIN_MEMBERS]
    rig_order = (fitted.groupby("rig")["n_fit_members"].sum()
                 .sort_values(ascending=False).index)
    rng = np.random.default_rng(3)
    for i, rig in enumerate(rig_order):
        g = fitted[fitted["rig"] == rig]
        y = i + rng.uniform(-0.16, 0.16, len(g))
        ax2.scatter(g["k_rel"], y, s=8 + 1.1 * np.sqrt(g["n_fit_members"]), color=C_GEOM,
                    alpha=0.45, linewidths=0, zorder=2)
        med = float(g["k_rel"].median())
        ax2.plot([med, med], [i - 0.26, i + 0.26], color=INK, lw=2.2, zorder=3)
        ax2.text(med, i + 0.44, f"{med:.3f}", ha="center", color=INK, fontsize=8)
    ax2.axvline(1.0, color=BASELINE, lw=1)
    ax2.set_yticks(range(len(rig_order)))
    ax2.set_yticklabels(
        [f"{rig}\n({int(fitted[fitted['rig'] == rig]['n_fit_members'].sum()):,} members)"
         for rig in rig_order], fontsize=8.5)
    ax2.invert_yaxis()
    secax = ax2.secondary_xaxis("top", functions=(lambda k: k * mf.BLEND_H_M,
                                                  lambda h: h / mf.BLEND_H_M))
    secax.set_xlabel("implied camera height (m, CurbRamp h × k)", fontsize=8.5,
                     color=SECONDARY)
    secax.tick_params(labelsize=8, colors=MUTED)
    ax2.set_xlabel("per-sequence distance scale k (relative to run geometric mean)")
    ax2.set_title("richmond: fitted per-sequence scale by rig", loc="left", fontsize=10, pad=24)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    _title(fig, "Most of the height residual is the rigs — and the rigs transfer",
           "Left: only A reads pixels, so only A can carry a pixel-frame height defect — and it does, "
           "at 2.6× the band the height-blind models (B, C) show from rig confounding alone. D sits "
           "inside that band; one fitted height per sequence collapses it. The last bar is the honest "
           "test — scales fitted on a disjoint half remove 69% of the held-out half's slope (66–75% "
           "over five seeds): real transferable rig geometry, but not the whole residual. Right: the "
           "fitted scales order as mount geometry predicts.", wrap=128)
    fig.subplots_adjust(top=0.62, wspace=0.34)
    _save(fig, "fig18-falsification-height-axis.png")


def main():
    diags = {run: mf.diagnose_run(run) for run in RUNS}
    seq_scales = {"richmond": mf.sequence_scale_summary("richmond")}
    _, seq_table = mf.fit_sequence_scales("richmond")
    fig17(diags)
    fig18(diags, seq_scales, seq_table)
    print("wrote figures/fig17-falsification-range-axis.png and "
          "figures/fig18-falsification-height-axis.png")


if __name__ == "__main__":
    main()
