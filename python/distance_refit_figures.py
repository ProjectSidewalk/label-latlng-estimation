"""Figures 14-16: the distance-refit ladder (issue #3).

Everything is recomputed from the same library code the tests verify, on the same R-fixture
train/test split — no numbers are read from the committed summary (the findings tests hold the
two together instead).

Usage (repo root): python python/distance_refit_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    add_heading_diff, clean_data, fit_models, load_data, split_from_fixtures,
)
from make_figures import (  # noqa: E402
    BASELINE, DENSITY_CMAP, INK, MUTED, SECONDARY, _binned_median, _save, _title,
)
import distance_refit as dr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_EST7 = "#2a78d6"       # the status quo is always blue in this repo
C_GEOM = "#1baf7a"       # forward-looking geometry (as in fig13)
C_GEOM_DARK = "#0d6b4a"  # second geometry series
C_LEGACY = "#eb6834"     # legacy behavior / the defect
C_H6656, C_H8192 = "#eb6834", "#2a78d6"  # fig6's height-group pairing


def _pipeline():
    cleaned, _ = clean_data(load_data(os.path.join(ROOT, "data")))
    cleaned = dr.add_depression(add_heading_diff(cleaned))
    train, test = split_from_fixtures(cleaned, os.path.join(ROOT, "tests/fixtures/r-baseline"))
    models = fit_models(train, include_est6=False)
    fits = dr.fit_all_rungs(train, models, os.path.join(ROOT, "data"))
    chosen = dr.choose_candidate(fits, train)["rung"]
    scored = dr.score_rungs(fits, models, train, test)
    return cleaned, train, test, models, fits, chosen, scored


def _est7_z1_curve(models, dep, canvas_y):
    """est7's zoom-1 distance line expressed in depression space, using the fixed-frame
    px/deg scale (sv_image_y = -depression * SV_PX_PER_DEG) at a representative canvas_y."""
    c = models["est7"]["dist"][0]
    sv = -dep * dr.SV_PX_PER_DEG
    return np.maximum(0, c["(Intercept)"] + c["sv_image_y"] * sv + c["canvas_y"] * canvas_y)


def fig14(cleaned, models, fits, chosen, scored):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))

    ax = axes[0]
    dep = cleaned["depression_deg"].to_numpy(float)
    dist = cleaned["pano_dist"].to_numpy(float)
    ax.hexbin(dep, dist, gridsize=55, extent=(0, 35, 0, 50), cmap=DENSITY_CMAP,
              norm=LogNorm(), linewidths=0)
    xs = np.linspace(0.3, 35, 300)
    ax.plot(xs, _est7_z1_curve(models, xs, float(cleaned["canvas_y"].median())),
            color=C_EST7, lw=2, ls=(0, (5, 3)))
    ax.plot(xs, np.clip(2.6 / np.tan(np.radians(xs)), 0, 50), color=MUTED, lw=2)
    ax.plot(xs, dr.predict_dist(fits["D_blend_l1"],
                                __import__("pandas").DataFrame({"depression_deg": xs})),
            color=C_GEOM, lw=2.2, solid_capstyle="round")
    e = fits["E_l1"]
    ax.plot(e["knots_dep_deg"], e["knots_dist_m"], color=INK, lw=1.6, ls=(0, (2, 2)))
    ax.set_xlim(0, 35); ax.set_ylim(0, 50)
    ax.set_xlabel("exact depression angle (deg below horizon)")
    ax.set_ylabel("distance from pano (m)")
    ax.set_title("all cleaned labels (395,147)", loc="left")
    for label, color, y in (("est7 (z1, fixed-frame px)", C_EST7, 0.92),
                            ("anchor 2.6 m / tan (0 params)", MUTED, 0.84),
                            ("D blend, L1 (2 params)", C_GEOM, 0.76),
                            ("E isotonic knots", INK, 0.68)):
        ax.text(0.35, y, label, color=color, transform=ax.transAxes, fontsize=8.6)

    bins = np.arange(0, 31, 1.0)
    truth = scored["pano_dist"].to_numpy(float)
    ax = axes[1]
    for key, label, color in (("est7", "est7", C_EST7), ("anchor", "anchor", MUTED),
                              (chosen, "chosen D", C_GEOM), ("E_l1", "E", SECONDARY)):
        cx, cy = _binned_median(truth, scored[f"error_{key}"].to_numpy(float), bins)
        ax.plot(cx, cy, color=color, lw=2, solid_capstyle="round")
        ax.annotate(label, (cx[-1], cy[-1]), xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=8.6, va="center")
    ax.set_xlabel("true distance from pano (m)")
    ax.set_ylabel("median lat/lng error (m)")
    ax.set_title("error vs true distance (test)", loc="left")
    ax.set_xlim(0, 33)

    ax = axes[2]
    for key, label, color in (("est7", "est7", C_EST7), (chosen, "chosen D", C_GEOM)):
        cx, cy = _binned_median(truth, (scored[f"dist_pred_{key}"] - scored["pano_dist"])
                                .to_numpy(float), bins)
        ax.plot(cx, cy, color=color, lw=2, solid_capstyle="round")
        ax.annotate(label, (cx[-1], cy[-1]), xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=8.6, va="center")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xlabel("true distance from pano (m)")
    ax.set_ylabel("median signed distance error (m)")
    ax.set_title("compression bias, closed", loc="left")
    ax.set_xlim(0, 33)

    _title(fig, "The distance half is geometry too — fig 14",
           "Left: distance vs the exact depression angle from #5's projection — the cotangent family follows "
           "the data where the per-zoom linear fits (blue, at median canvas_y) are compressive. Middle: the "
           "0-parameter 2.6 m anchor already beats the 12-parameter status quo. Right: est7's compression "
           "bias (fig 4) is what the saturating cotangent removes.", wrap=150)
    fig.subplots_adjust(top=0.76, wspace=0.28)
    _save(fig, "fig14-distance-geometry.png")


def fig15(fits, chosen, scored, noise):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))

    ax = axes[0]
    import pandas as pd
    xs = np.linspace(-2, 10, 400)
    frame = pd.DataFrame({"depression_deg": xs,
                          "label_type": np.repeat("CurbRamp", len(xs)),
                          "zoom": np.ones(len(xs), int)})
    for key, color, ls in (("C_l1", C_LEGACY, (0, (5, 3))), ("D_floor_l1", C_GEOM, "-"),
                           ("D_blend_l1", C_GEOM_DARK, "-"), ("E_l1", INK, (0, (2, 2)))):
        ax.plot(xs, dr.predict_dist(fits[key], frame), color=color, lw=2, ls=ls)
    for label, color, x, y in (("C raw cotangent", C_LEGACY, 4.3, 41.0),
                               ("D floor", C_GEOM, -1.7, 19.0),
                               ("D blend", C_GEOM_DARK, -1.7, 32.5),
                               ("E isotonic", INK, 3.2, 26.3)):
        ax.text(x, y, label, color=color, fontsize=8.6)
    ax.axhline(dr.DIST_CAP_M, color=BASELINE, lw=1)
    ax.text(9.9, dr.DIST_CAP_M - 3, "50 m cap", color=MUTED, fontsize=8.2, ha="right")
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_xlabel("depression angle (deg)")
    ax.set_ylabel("predicted distance (m)")
    ax.set_ylim(0, 55)
    ax.set_title("predictions near the horizon", loc="left")

    ax = axes[1]
    rungs = ["est7", "anchor", "C_l1", "D_soft_l1", "D_floor_l1", chosen, "E_l1"]
    nh = dr.near_horizon_table(scored, keys=rungs)
    row = next(r for r in nh if r["bin_deg"] == "(0.0, 2.0]")
    ys = np.arange(len(rungs))[::-1]
    for y, k in zip(ys, rungs):
        v = row["per_rung"][k]
        color = {"est7": C_EST7, "anchor": MUTED, "C_l1": C_LEGACY,
                 "D_soft_l1": SECONDARY, "D_floor_l1": C_GEOM, chosen: C_GEOM_DARK,
                 "E_l1": INK}[k]
        ax.plot([v["latlng_median_m"], v["latlng_p95_m"]], [y, y], color=color, lw=2, alpha=0.45)
        ax.plot(v["latlng_median_m"], y, "o", color=color, ms=7)
        ax.text(56.5, y, f"<= {v['dist_pred_max_m']:.0f} m", color=MUTED, fontsize=8.2,
                va="center", ha="right")
    ax.set_yticks(ys, rungs)
    ax.set_xlabel("lat/lng error 0-2 deg (median; to p95)")
    ax.set_xlim(0, 57)
    ax.set_title(f"the near-horizon population (n={row['n']})", loc="left")

    ax = axes[2]
    sig = [float(s) for s in noise["sigmas_px"]]
    offsets = {"est7": -5, "anchor": 5, "C_l1": 5, chosen: 8, "E_l1": 0}
    for k, label, color in (("est7", "est7", C_EST7), ("anchor", "anchor", MUTED),
                            ("C_l1", "C", C_LEGACY), (chosen, "chosen D", C_GEOM),
                            ("E_l1", "E", INK)):
        ys = [noise["per_rung"][k][str(s)]["delta_median_m"] for s in noise["sigmas_px"]]
        ax.plot(sig, ys, "-o", color=color, lw=2, ms=5)
        ax.annotate(label, (sig[-1], ys[-1]), xytext=(6, offsets[k]),
                    textcoords="offset points", color=color, fontsize=8.6, va="center")
    ax.set_xlabel("click noise sigma (px, gaussian)")
    ax.set_ylabel("increase in median lat/lng error (m)")
    ax.set_xticks(sig)
    ax.set_xlim(1.5, 11.6)
    ax.set_title("click-noise sensitivity", loc="left")

    _title(fig, "At the horizon a raw cotangent diverges; the saturating forms stay bounded — fig 15",
           "Left: below ~6 deg the raw cotangent (C) runs to the 50 m cap; the floor, blend, and isotonic "
           "forms saturate in the low 20s like the data. Middle: in the thin 0-2 deg bin the saturating "
           "forms match est7 while C answers 28 m (right-edge text: each form's largest possible answer). "
           "Right: at realistic 2-5 px click noise every geometry rung degrades by centimeters — the 2016 "
           "noise objection, dissolved.", wrap=150)
    fig.subplots_adjust(top=0.76, wspace=0.34)
    _save(fig, "fig15-horizon-saturation.png")


def fig16(ffc, apply_path, cand_b):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))

    ax = axes[0]
    cities = list(ffc["cities"].keys())
    ys = np.arange(len(cities))[::-1]
    for y, city in zip(ys, cities):
        row = ffc["cities"][city]
        for h, color in ((6656, C_H6656), (8192, C_H8192)):
            if h in row:
                ax.plot(row[h]["ratio_median"], y, "o", color=color, ms=7)
    ax.axvline(1.0, color=BASELINE, lw=1)
    ax.axvline(8192 / 6656, color=C_LEGACY, lw=1.6, ls=(0, (5, 3)))
    ax.text(8192 / 6656 - 0.012, 1.6, "if sv_image_y scaled\nwith the raster",
            color=C_LEGACY, fontsize=8.2, ha="right", va="top")
    ax.set_yticks(ys, cities)
    ax.set_xlim(0.9, 1.3)
    ax.set_xlabel("median implied/exact depression ratio")
    ax.text(1.04, 0.5, "6656 px", color=C_H6656, fontsize=8.6)
    ax.text(1.04, 0.0, "8192 px", color=C_H8192, fontsize=8.6)
    ax.set_title("the frame is fixed, every city", loc="left")

    ax = axes[1]
    groups = [("raw", "deployed\n(raw pixels)"), ("normalized", "#4765 one-liner\n(normalized)")]
    width = 0.35
    for i, (variant, label) in enumerate(groups):
        for j, (h, color) in enumerate(((6656, C_H6656), (8192, C_H8192))):
            v = apply_path[variant][f"h{h}"]["signed_median_m"]
            ax.bar(i + (j - 0.5) * width, v, width * 0.92, color=color)
            ax.text(i + (j - 0.5) * width, v + (0.06 if v >= 0 else -0.13), f"{v:+.2f}",
                    ha="center", fontsize=8.6, color=INK)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylabel("median signed distance bias (m)")
    ax.set_ylim(-0.8, 2.1)
    ax.set_title("what each apply path does", loc="left")

    ax = axes[2]
    zs = (1, 2, 3)
    for i, z in enumerate(zs):
        iv = cand_b[f"zoom{z}"]["interact_vs_norm_prediction"]
        ax.errorbar(i, iv["interact_coef"], yerr=2 * iv["interact_se"], fmt="o", color=INK,
                    ms=6, capsize=3, lw=1.5)
        ax.plot(i, iv["sv_slope_it_would_have_to_match"], "o", ms=8, mfc="none",
                mec=C_LEGACY, mew=1.8)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(range(len(zs)), [f"zoom {z}" for z in zs])
    ax.set_ylabel("interaction coefficient (m/px)")
    ax.text(0.03, 0.92, "required if normalization\nwere right in-frame", color=C_LEGACY,
            transform=ax.transAxes, fontsize=8.6)
    ax.text(0.03, 0.20, "measured (+/-2 se)", color=INK, transform=ax.transAxes, fontsize=8.6)
    ax.set_title("in-frame height term ~zero", loc="left")

    _title(fig, "#4765's defect lives in the apply path, not the 2021 fit — fig 16",
           "Left: raw raster pixels would put 8192-px panos at 1.23; every city measures ~0.97 for both "
           "heights — the training frame is fixed. Middle: the raw-pixel apply path is nearly unbiased "
           "because two errors cancel (26% pixel overshoot vs the fit's own +1.7 m bias on 8192-px panos); "
           "normalizing without refitting surfaces the bias. Right: the height coefficient a normalized "
           "predictor would need is far outside what the data allows.", wrap=150)
    fig.subplots_adjust(top=0.76, wspace=0.34)
    _save(fig, "fig16-4765-apply-path.png")


def main() -> None:
    cleaned, train, test, models, fits, chosen, scored = _pipeline()
    print(f"pipeline ready (chosen: {chosen})")
    noise = dr.noise_sweep(fits, models, train, test,
                           keys=["est7", "anchor", "C_l1", chosen, "E_l1"])
    ffc = dr.fixed_frame_check(cleaned)
    apply_path = dr.apply_path_check(models, test)
    cand_b = dr.candidate_b_checks(train, test)
    fig14(cleaned, models, fits, chosen, scored)
    fig15(fits, chosen, scored, noise)
    fig16(ffc, apply_path, cand_b)


if __name__ == "__main__":
    main()
