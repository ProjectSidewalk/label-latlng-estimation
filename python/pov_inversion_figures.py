"""Figure 13: the #5 story — exact POV inversion vs est7's fitted heading, and why the
regression ever looked competitive.

Offline and deterministic like everything else: committed CSVs in, the R-exported split,
no network. Styling comes from make_figures (rcParams, palette, _title/_save) so this sits
beside figures 1-12 unchanged.

Usage (repo root): python python/pov_inversion_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    add_heading_diff, clean_data, fit_models, load_data, split_from_fixtures,
)
from make_figures import BASELINE, INK, MUTED, SECONDARY, _binned_median, _save, _title  # noqa: E402
from pov_inversion import (  # noqa: E402
    DEPTH_GRID_COLUMN_DEG, era_heading_diff, score_heading_swap,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color follows the entity: est7 keeps its blue from figures 1-8; the exact inversion (the
# forward-looking model) is green; the era-faithful replica is the secondary ink; the
# calibrated variant that absorbs the legacy depth-lookup bias is orange, like every other
# "legacy artifact" in the repo's figures. Set validated against the light surface.
C_MODEL = {"est7": "#2a78d6", "exact": "#1baf7a", "era": "#52514e", "era_cal": "#eb6834"}
LABELS = {"est7": "est7 (6 fitted params)", "exact": "exact inversion (0 params)",
          "era": "era-faithful exact (0 params)", "era_cal": "era-faithful + 1 constant"}


def fig13(scored, resid_train, delta):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))

    # (a) where the linear approximation loses: error vs canvas_x
    ax = axes[0]
    bins = np.arange(0, 721, 45)
    for m in ("est7", "exact", "era_cal"):
        cx, cy = _binned_median(scored["canvas_x"].to_numpy(float),
                                scored[f"heading_error_{m}"].to_numpy(float), bins)
        ax.plot(cx, cy, color=C_MODEL[m], lw=2, label=LABELS[m])
    ax.axvline(360, color=BASELINE, lw=0.8, zorder=0)
    ax.set_xlabel("canvas_x (px; canvas center at 360)")
    ax.set_ylabel("median |heading error| (deg)")
    ax.set_title("The fits fail at the canvas edges", loc="left")
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 4.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99))

    # (b) per-zoom medians, all four models, direct-labeled at zoom 1
    ax = axes[1]
    zooms = (1, 2, 3)
    offsets = {"est7": -0.24, "exact": -0.08, "era": 0.08, "era_cal": 0.24}
    short = {"est7": "est7", "exact": "exact", "era": "era", "era_cal": "era+c"}
    for m in ("est7", "exact", "era", "era_cal"):
        med = [float(scored.loc[scored["zoom"] == z, f"heading_error_{m}"].median())
               for z in zooms]
        x = np.array(zooms, float) + offsets[m]
        ax.scatter(x, med, s=42, color=C_MODEL[m], zorder=3)
        ax.vlines(x, 0, med, color=C_MODEL[m], lw=1.2, alpha=0.45)
        ax.annotate(short[m], xy=(x[0], med[0]), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=C_MODEL[m])
    ax.set_xticks(zooms)
    ax.set_xlabel("zoom level")
    ax.set_ylabel("median |heading error| (deg)")
    ax.set_title("Per zoom: the artifact-aware model wins", loc="left")
    ax.set_ylim(0, 1.85)

    # (c) the mechanism: the residual after the era-faithful model is one depth-grid column
    ax = axes[2]
    r = resid_train[np.abs(resid_train) < 5]
    ax.hist(r, bins=200, color=MUTED, alpha=0.75)
    ax.axvline(0, color=BASELINE, lw=1.0)
    ax.axvline(DEPTH_GRID_COLUMN_DEG, color=C_MODEL["era_cal"], lw=1.6, ls="--")
    ax.axvline(delta, color=INK, lw=1.0, ls=":")
    ymax = ax.get_ylim()[1]
    ax.annotate(f"one depth-grid column\n360/512 = {DEPTH_GRID_COLUMN_DEG:.3f} deg (dashed)",
                xy=(DEPTH_GRID_COLUMN_DEG, 0.93 * ymax), xytext=(1.9, 0.88 * ymax),
                color=C_MODEL["era_cal"], fontsize=9,
                arrowprops={"arrowstyle": "-", "color": C_MODEL["era_cal"], "lw": 0.8})
    ax.annotate(f"train-set mean\n{delta:+.3f} deg (dotted)",
                xy=(delta, 0.55 * ymax), xytext=(1.9, 0.50 * ymax),
                color=SECONDARY, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.8})
    ax.set_xlabel("target heading_diff − era-faithful prediction (deg)")
    ax.set_ylabel("training labels")
    ax.set_title("The leftover bias is the depth lookup's", loc="left")

    _title(fig,
           "The heading regression was approximating math we can just run — fig 13",
           "Exact click→POV inversion (calculatePointPov, verified to ≤1 px against the "
           "stored sv_image_x and evolution 179) replaces est7's six fitted heading "
           "coefficients. Against 2017–2020 targets it must also model how the targets were "
           "made: parseInt-truncated POV inputs, plus one depth-grid column of bearing bias "
           "from Label.js's Math.ceil indexing. So modeled, geometry beats the fit everywhere.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, "fig13-pov-inversion.png")


def main() -> None:
    data_dir = os.path.join(ROOT, "data")
    fixtures = os.path.join(ROOT, "tests", "fixtures", "r-baseline")
    cleaned, _ = clean_data(load_data(data_dir))
    cleaned = add_heading_diff(cleaned)
    train, test = split_from_fixtures(cleaned, fixtures)
    models = fit_models(train, include_est6=False)
    scored = score_heading_swap(models, train, test)
    resid_train = train["heading_diff"].to_numpy(float) - era_heading_diff(train)
    fig13(scored, resid_train, float(scored.attrs["era_cal_delta_deg"]))


if __name__ == "__main__":
    main()
