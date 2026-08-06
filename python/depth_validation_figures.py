"""Figures and the overlay gallery for the issue #9 depth validation.

Everything here runs offline from the committed artifacts -- the depth payloads from #4
and the verbatim imagery tiles from `depth-validation-tiles.jsonl.gz` -- so the figures
regenerate byte-for-byte on any machine with the repo and no network access.

Four figures, in the order the argument runs:

    fig9   the overlay itself: the model's skyline drawn on the panorama, identity
           against the mirrored frame. This is the evidence; the rest is bookkeeping.
    fig10  the same claim as statistics: paired frame controls, a permutation null over
           mismatched panoramas, and the pooled column-offset sweep.
    fig11  what the product actually is: the empty oblique band, and depth against a
           flat earth.
    fig12  what that costs a label: where label pixels land, and how far apart two
           captures of one street put the ground and the walls.

Styling comes from make_figures (rcParams, palette, _title/_save) so these sit beside
figures 1-8 unchanged.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import depth_validation as dv  # noqa: E402
import gsv_depth as gd  # noqa: E402

# Color follows the entity, as everywhere else in the repo: the true frame is the est7
# blue, wrong frames are the est5 orange, and the null/comparison grey is MUTED.
C_TRUE = "#2a78d6"
C_WRONG = "#eb6834"
C_THIRD = "#1baf7a"


# ---------------------------------------------------------------------------- loading

def load_artifacts(data_dir):
    panos = pd.read_csv(os.path.join(data_dir, "depth-validation-panos.csv.gz"))
    labels = pd.read_csv(os.path.join(data_dir, "depth-validation-labels.csv.gz"))
    cross = pd.read_csv(os.path.join(data_dir, "depth-validation-crossvintage.csv.gz"))
    with open(os.path.join(data_dir, "depth-validation-summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with gzip.open(os.path.join(data_dir, "depth-validation-sweeps.json.gz"), "rt") as f:
        sweeps = json.load(f)
    return panos, labels, cross, summary, sweeps


def payload_map(data_dir):
    out = {}
    with gzip.open(
        os.path.join(data_dir, "depth-pilot-payloads.jsonl.gz"), "rt", encoding="utf-8"
    ) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["pano_id"]] = rec["depth_b64"]
    return out


def skyline_rows(no_plane_grid):
    """Lowest no-plane row per column; NaN where the column has none."""
    height = no_plane_grid.shape[0]
    has = no_plane_grid.any(axis=0)
    last = height - np.argmax(no_plane_grid[::-1, :], axis=0) - 1
    out = last.astype(float)
    out[~has] = np.nan
    return out


def draw_overlay(ax, rgb, payload, control, show_facades=True):
    """One panorama with the depth model's skyline and facades drawn over it."""
    height, width = rgb.shape[:2]
    idx = payload.indices.reshape(payload.height, payload.width)
    grid = dv.resample_to_image(dv.apply_frame_control(idx, control), width, height)

    colour = C_TRUE if control == "identity" else C_WRONG
    ax.imshow(rgb)
    if show_facades:
        facade = dv.resample_to_image(
            dv.apply_frame_control(dv.facade_pixel_mask(payload), control), width, height
        )
        overlay = np.zeros((height, width, 4))
        overlay[facade] = (*_rgba(colour), 0.30)
        ax.imshow(overlay)
    ax.plot(np.arange(width), skyline_rows(grid == 0), color=colour, lw=1.9)
    ax.set_xticks([])
    ax.set_yticks([])


def _rgba(hex_colour):
    hex_colour = hex_colour.lstrip("#")
    return tuple(int(hex_colour[i:i + 2], 16) / 255 for i in (0, 2, 4))


# ---------------------------------------------------------------------------- figures

def cmd_figures(args):
    import make_figures as mf  # rcParams, palette, _title/_save conventions
    import matplotlib.pyplot as plt

    # Draw negative numbers with an ASCII hyphen. Matplotlib defaults to U+2212, which
    # the DejaVu fallback on some machines has no glyph for and renders as a box.
    plt.rcParams["axes.unicode_minus"] = False

    from run_depth_validation import load_tiles_artifact, rgb_from_record

    panos, labels, cross, summary, sweeps = load_artifacts(args.data_dir)
    tiles = load_tiles_artifact(args.data_dir)
    payloads = payload_map(args.data_dir)
    cohort = panos[panos["has_power"].fillna(False) & panos["has_imagery"].fillna(False)]

    # ---- fig 9: the overlay. Show the panoramas where the model has the most to say.
    picks = (
        cohort.sort_values("structure_fraction", ascending=False)["pano_id"]
        .head(3).tolist()
    )
    # Sized so the six 2:1 panels fill the axes region: three rows of two 2:1 images is
    # a 4:3 block, and the title band takes the rest. Any wider and the panels letterbox.
    fig, axes = plt.subplots(3, 2, figsize=(9.8, 9.2))
    for row, pid in enumerate(picks):
        record = tiles[pid].get("adjudication") or tiles[pid]["scoring"]
        rgb = rgb_from_record(record)
        payload = gd.decode_depth_payload(payloads[pid])
        for col, control in enumerate(("identity", "x_mirror")):
            draw_overlay(axes[row, col], rgb, payload, control)
            if row == 0:
                axes[row, col].set_title(
                    "as the client reads it" if control == "identity"
                    else "the same payload, mirrored",
                    loc="left", color=C_TRUE if col == 0 else C_WRONG,
                )
        axes[row, 0].set_ylabel(f"{pid[:10]}…", fontsize=8, color=mf.MUTED)
    mf._title(
        fig,
        "The depth model's skyline lands on the buildings — so the payload really is this panorama's scene",
        "Left: the no-plane boundary of today's depth payload (blue) and its vertical planes (shaded), drawn on the "
        "panorama's own imagery. Depth comes from Google's photometa endpoint and the imagery from the tile server — "
        "two independent hosts, so agreement cannot be a decoding artifact. Right: the identical payload mirrored in x, "
        "the error the #4 pilot could not have detected because it compared the pipeline against coordinates the same "
        "pipeline had written. Foliage reads as no-plane throughout: the model contains terrain and building footprints, "
        "nothing else.",
        wrap=92,
    )
    fig.subplots_adjust(top=0.79, bottom=0.01, left=0.04, right=0.99,
                        hspace=0.06, wspace=0.03)
    mf._save(fig, "fig9-depth-imagery-overlay.png")

    # ---- fig 10: the same claim, as statistics
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))

    ax = axes[0]
    rivals = [("x_mirror", "mirrored in x"), ("rotate_180", "rotated 180°"),
              ("row_flip", "rows flipped")]
    for i, (control, label) in enumerate(rivals):
        pair = cohort[["viol_identity", f"viol_{control}"]].dropna()
        better = int((pair["viol_identity"] < pair[f"viol_{control}"]).sum())
        worse = int((pair["viol_identity"] > pair[f"viol_{control}"]).sum())
        ax.barh(i + 0.16, better, height=0.32, color=C_TRUE)
        ax.barh(i - 0.16, worse, height=0.32, color=C_WRONG)
        ax.text(better + 0.7, i + 0.16, f"{better}", va="center", fontsize=8.5, color=C_TRUE)
        ax.text(worse + 0.7, i - 0.16, f"{worse}", va="center", fontsize=8.5, color=C_WRONG)
    ax.set_yticks(range(len(rivals)), [label for _, label in rivals])
    ax.invert_yaxis()
    ax.set_xlabel("panoramas (blue = the true frame fits better)")
    ax.set_title(f"paired against each wrong frame   n={len(cohort)}", loc="left")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    own = cohort["viol_identity"].dropna()
    null = cohort["viol_null_median"].dropna()
    bins = np.linspace(0, max(float(null.quantile(0.95)), 0.05), 26)
    ax.hist(own, bins=bins, color=C_TRUE, alpha=0.85, label="own panorama")
    ax.hist(null, bins=bins, color=mf.MUTED, alpha=0.55, label="mismatched (median of 10)")
    ax.axvline(float(own.median()), color=C_TRUE, lw=1.4)
    ax.axvline(float(null.median()), color=mf.MUTED, lw=1.4)
    ax.set_xlabel("sky violation (model covering open sky)")
    ax.set_ylabel("panoramas")
    ax.legend(loc="upper right")
    ax.set_title("against a permutation null", loc="left")

    ax = axes[2]
    ids = [p for p in cohort["pano_id"] if p in sweeps]
    shifts = np.array(sweeps[ids[0]][0], dtype=float)
    stack = np.array(
        [[np.nan if v is None else v for v in sweeps[p][1]] for p in ids], dtype=float
    )
    mean_curve = np.nanmean(stack, axis=0)
    ax.plot(shifts, mean_curve, color=C_TRUE, lw=2)
    ax.axvline(0, color=mf.BASELINE, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(
        "minimum at zero offset", (0, float(np.nanmin(mean_curve))),
        xytext=(14, 26), textcoords="offset points", fontsize=8.5, color=mf.INK,
        arrowprops=dict(arrowstyle="->", color=mf.MUTED, lw=0.9),
    )
    ax.set_xlabel("depth map shifted horizontally (payload columns)")
    ax.set_ylabel("mean sky violation")
    ax.set_title(f"the alignment optimum   n={len(ids)} panoramas", loc="left")

    mf._title(
        fig,
        "Registration holds up statistically: the true frame wins all three paired controls, and the offset optimum sits at zero",
        "Sky violation is the fraction of certainly-sky pixels the model covers with a surface — one-sided on purpose, "
        "because the model's no-plane region is a superset of the sky (it omits trees, poles and wires, so counting the "
        "reverse direction would penalise the correct frame for the model's blindness to foliage). Restricted to the "
        f"{len(cohort)} panoramas whose model puts something above the horizon; on a bare suburban street every frame "
        "convention reproduces 'ground below, nothing above' equally well and the test has no power.",
        wrap=124,
    )
    fig.subplots_adjust(top=0.66, wspace=0.42)
    mf._save(fig, "fig10-registration-statistics.png")

    # ---- fig 11: what the product actually is
    tilt = np.zeros(90)
    fe_rows = []
    pilot = pd.read_csv(os.path.join(args.data_dir, "depth-pilot-panos.csv.gz"))
    heights = (
        pilot.dropna(subset=["ground_height_m"])
        .groupby("pano_id")["ground_height_m"].first().to_dict()
    )
    with gzip.open(
        os.path.join(args.data_dir, "depth-pilot-payloads.jsonl.gz"), "rt", encoding="utf-8"
    ) as f:
        for line in f:
            rec = json.loads(line)
            payload = gd.decode_depth_payload(rec["depth_b64"])
            tilt += dv.tilt_histogram(payload)
            height = heights.get(rec["pano_id"], float("nan"))
            if np.isfinite(height):
                fe_rows.append(dv.flat_earth_comparison(payload, height).__dict__)
    tilt = tilt / tilt.sum()
    fe = pd.DataFrame(fe_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    ax.bar(np.arange(90) + 0.5, tilt, width=1.0, color=C_TRUE, edgecolor="none")
    ax.axvspan(15, 75, color=C_WRONG, alpha=0.10, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.6)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 15, 45, 75, 90], ["0°\nflat", "15°", "45°", "75°", "90°\nwall"])
    ax.set_yticks([1e0, 1e-2, 1e-4, 1e-6], ["100%", "1%", "0.01%", "0.0001%"])
    ax.text(
        45, 2e-3,
        f"everything a real reconstruction\nwould put here: {tilt[15:75].sum() * 100:.2f}%\n"
        "of pixels\n(cars, canopy, pitched roofs,\ndriveways, hillsides)",
        ha="center", va="center", fontsize=8.5, color=C_WRONG,
    )
    ax.set_xlabel("tilt of the surface a pixel sits on")
    ax.set_ylabel("share of pixels (log)")
    ax.set_title("409 payloads: a Manhattan world", loc="left")

    ax = axes[1]
    d = np.sort(fe["frac_within_1m"].to_numpy())
    ax.step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=C_TRUE, lw=2)
    ax.axvline(float(np.median(d)), color=mf.BASELINE, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(
        f"median panorama: {np.median(d) * 100:.0f}% of its ground\nwithin 1 m of flat earth",
        (float(np.median(d)), 0.5), xytext=(-14, 6), textcoords="offset points",
        fontsize=8.5, color=mf.INK, ha="right",
    )
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of ground-band pixels within 1 m of h/tan(depression)")
    ax.set_ylabel("fraction of panoramas at or below x")
    ax.set_title(f"depth vs. plain trigonometry   n={len(fe)} panoramas", loc="left")

    mf._title(
        fig,
        "GSV depth is a constructed model, not a measurement — and under a Sidewalk label it is very nearly flat earth",
        "Left: pixel-weighted distribution of the tilt of the plane each pixel belongs to, pooled over all 409 committed "
        "payloads. Surfaces are horizontal or vertical and almost nothing in between, which no photogrammetric "
        "reconstruction of a street produces — it corroborates streetlevel's note that the product is synthesised from "
        "elevation data and building footprints. Right: how much of each panorama's ground band the naive flat-earth "
        "formula already predicts to within a metre.",
        wrap=124,
    )
    fig.subplots_adjust(top=0.66, wspace=0.26)
    mf._save(fig, "fig11-what-the-product-is.png")

    # ---- fig 12: what it costs a label
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))

    ax = axes[0]
    lab = labels[labels["in_cleaned"]]
    order = ["ground", "terrain", "facade", "oblique", "sky"]
    counts = [int((lab["hit_class"] == k).sum()) for k in order]
    colours = [C_TRUE, "#86b6ef", C_THIRD, C_WRONG, mf.MUTED]
    ax.bar(range(len(order)), counts, color=colours, edgecolor="none", width=0.7)
    for i, c in enumerate(counts):
        if c:
            ax.text(i, c + 1.2, str(c), ha="center", fontsize=8.5, color=mf.SECONDARY)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("labels")
    ax.set_title(f"what a label's pixel lands on   n={len(lab)}", loc="left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    excess = lab["flat_earth_excess_m"].dropna()
    ax.hist(excess.clip(-6, 6), bins=np.linspace(-6, 6, 49), color=C_TRUE, edgecolor="none")
    ax.axvline(0, color=mf.BASELINE, lw=1.0)
    ax.set_xlabel("depth distance minus flat-earth distance (m)")
    ax.set_ylabel("labels")
    ax.set_title(
        f"{(excess.abs() < 1).mean() * 100:.0f}% of labels within 1 m of flat earth", loc="left"
    )

    ax = axes[2]
    ground = cross["median_residual_m"].dropna()
    facade = cross["median_facade_offset_m"].dropna()
    for i, (values, colour, label) in enumerate(
        ((ground, C_TRUE, "ground"), (facade, C_WRONG, "building facades"))
    ):
        jitter = np.linspace(-0.16, 0.16, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=16, color=colour,
                   alpha=0.75, linewidths=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(values)] * 2, color=mf.INK, lw=2)
        ax.text(i + 0.32, float(np.median(values)), f"{np.median(values):.2f} m",
                fontsize=8.5, color=mf.INK, va="center")
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 1.9)
    ax.set_xticks([0, 1], ["ground", "building facades"])
    # Explicit ticks: the log formatter writes exponents through mathtext, which
    # ignores axes.unicode_minus and reintroduces the missing U+2212 glyph.
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_ylabel("disagreement (m, log)")
    ax.set_title(f"same street, {int(cross['year_gap'].median())} years apart", loc="left")
    ax.grid(axis="x", visible=False)

    mf._title(
        fig,
        "For a label on the ground the model adds almost nothing to trigonometry — and its one independent claim disagrees with itself",
        "Left and centre: Sidewalk labels sit on modelled ground, and the distance the depth payload returns is within a "
        "metre of h/tan(depression) for most of them, so the 2021 estimator was fitting a relationship the payload had "
        "largely already reduced to geometry. Right: two captures of the same street, a median 11 years apart, agree on "
        "the ground to 0.12 m — but a flat ground plane is invariant under a horizontal camera shift, so that number is "
        "nearly free. Building facades, the model's only genuinely independent geometry, are placed metres apart.",
        wrap=124,
    )
    fig.subplots_adjust(top=0.64, wspace=0.46)
    mf._save(fig, "fig12-label-consequences.png")


# ---------------------------------------------------------------------------- gallery

# Tokens are the repo's existing figure palette (make_figures.SURFACE/INK/SECONDARY/
# MUTED/GRID and the est7 blue / est5 orange), so the gallery and figures 1-12 read as
# one system. The dark ground is a warm near-black rather than a neutral grey, keeping
# the same slight ochre bias the light surface has.
GALLERY_CSS = """
:root { color-scheme: light dark;
  --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --secondary:#52514e; --muted:#898781;
  --line:#e1e0d9; --true:#2a78d6; --wrong:#eb6834; --mark:#c8940a;
  --step:1.2; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#131311; --card:#1b1b18; --ink:#f4f3ef; --secondary:#c2c0b8; --muted:#918f87;
  --line:#302f2b; --true:#6ba6f0; --wrong:#f5865c; --mark:#e6b53d; } }
:root[data-theme="dark"] {
  --bg:#131311; --card:#1b1b18; --ink:#f4f3ef; --secondary:#c2c0b8; --muted:#918f87;
  --line:#302f2b; --true:#6ba6f0; --wrong:#f5865c; --mark:#e6b53d; }
:root[data-theme="light"] {
  --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --secondary:#52514e; --muted:#898781;
  --line:#e1e0d9; --true:#2a78d6; --wrong:#eb6834; --mark:#c8940a; }

body { background:var(--bg); color:var(--ink); margin:0; padding:2.5rem 1.25rem 5rem;
  font:16px/1.65 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:1180px; margin:0 auto; display:flex; flex-direction:column; gap:1.1rem; }
h1 { font-size:1.85rem; line-height:1.2; margin:0; letter-spacing:-.015em; text-wrap:balance;
     max-width:26ch; }
p.lede { color:var(--secondary); max-width:66ch; margin:0; }
p.lede em { color:var(--ink); font-style:normal; font-weight:600; }

.legend { display:flex; flex-wrap:wrap; gap:1.1rem; font-size:.83rem; color:var(--secondary);
  align-items:center; }
.legend span { display:flex; align-items:center; gap:.42rem; }
.swatch { width:1.1rem; height:.42rem; border-radius:2px; flex:none; }
.swatch.ring { width:.8rem; height:.8rem; border-radius:50%; background:none;
  border:2px solid #ffd400; box-shadow:0 0 0 1px #111; }

.controls { position:sticky; top:0; z-index:5; background:var(--bg); padding:.8rem 0;
  border-bottom:1px solid var(--line); display:flex; gap:.45rem; flex-wrap:wrap;
  align-items:center; }
button { font:inherit; font-size:.85rem; line-height:1; padding:.5rem .85rem;
  border:1px solid var(--line); border-radius:999px; background:var(--card);
  color:var(--secondary); cursor:pointer; }
button:hover { border-color:var(--muted); color:var(--ink); }
button:focus-visible { outline:2px solid var(--true); outline-offset:2px; }
button[aria-pressed="true"] { background:var(--true); border-color:var(--true); color:#fff; }
button[data-k="mirror"][aria-pressed="true"] { background:var(--wrong); border-color:var(--wrong); }
.note { color:var(--muted); font-size:.8rem; margin-left:auto; max-width:46ch; line-height:1.45; }

figure { margin:0; border:1px solid var(--line); border-radius:12px; overflow:hidden;
  background:var(--card); }
.stage { position:relative; line-height:0; overflow-x:auto; }
.stage img, .stage svg { width:100%; display:block; }
.stage svg { position:absolute; inset:0; height:100%; }
figcaption { padding:.7rem .95rem; font-size:.82rem; color:var(--muted);
  display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  border-top:1px solid var(--line); font-variant-numeric:tabular-nums; }
figcaption b { color:var(--ink); font-weight:600; }
figcaption code { font:inherit; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.95em; }
.hidden { display:none; }
@media (prefers-reduced-motion:no-preference) { .stage svg g { transition:opacity .12s ease; } }
"""

GALLERY_JS = """
const state = { skyline:true, facades:true, labels:true, compass:true, mirror:false };
function apply(){
  for (const k of ['skyline','facades','labels','compass']) {
    document.querySelectorAll('.'+k).forEach(e => e.classList.toggle('hidden', !state[k]));
  }
  document.querySelectorAll('.identity').forEach(e => e.classList.toggle('hidden', state.mirror));
  document.querySelectorAll('.mirrored').forEach(e => e.classList.toggle('hidden', !state.mirror));
  document.querySelectorAll('button[data-k]').forEach(b =>
    b.setAttribute('aria-pressed', String(state[b.dataset.k])));
}
document.addEventListener('click', e => {
  const b = e.target.closest('button[data-k]');
  if (!b) return;
  state[b.dataset.k] = !state[b.dataset.k];
  apply();
});
apply();
"""


def cmd_gallery(args):
    """Build the standalone overlay gallery: one page, every panel from committed bytes."""
    import base64
    import io

    from PIL import Image

    from run_depth_validation import load_tiles_artifact, rgb_from_record

    panos, labels, cross, summary, sweeps = load_artifacts(args.data_dir)
    tiles = load_tiles_artifact(args.data_dir)
    payloads = payload_map(args.data_dir)
    by_pano = dict(tuple(labels.groupby("pano_id"))) if not labels.empty else {}
    meta = panos.set_index("pano_id")
    # The raster is heading-centred, so placing a label needs the panorama's yaw;
    # see depth_validation.label_pixel_in_image.
    pilot = pd.read_csv(os.path.join(args.data_dir, "depth-pilot-panos.csv.gz"))
    yaw_by_pano = (
        pilot.dropna(subset=["fresh_heading_deg"])
        .groupby("pano_id")["fresh_heading_deg"].first().to_dict()
    )

    picked = [p for p in sorted(tiles) if "adjudication" in tiles[p] and p in payloads]
    figures = []
    for pid in picked:
        record = tiles[pid]["adjudication"]
        rgb = rgb_from_record(record)
        payload = gd.decode_depth_payload(payloads[pid])
        height, width = rgb.shape[:2]

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=74, optimize=True, progressive=True)
        src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        idx = payload.indices.reshape(payload.height, payload.width)
        facade_native = dv.facade_pixel_mask(payload)

        layers = []
        for control, cls, colour in (
            ("identity", "identity", "#2a78d6"), ("x_mirror", "mirrored hidden", "#eb6834")
        ):
            grid = dv.resample_to_image(dv.apply_frame_control(idx, control), width, height)
            line = skyline_rows(grid == 0)
            points = " ".join(
                f"{x},{y:.0f}" for x, y in enumerate(line)
                if np.isfinite(y) and (x % 2 == 0 or x == width - 1)
            )
            facade = dv.resample_to_image(
                dv.apply_frame_control(facade_native, control), width, height
            )
            # Facade extent as a rect per RUN of columns sharing a vertical extent.
            # One rect per column would emit tens of thousands of nodes per page.
            rects = []
            cols = facade.any(axis=0)
            extents = []
            for x in range(width):
                if not cols[x]:
                    extents.append(None)
                    continue
                ys = np.where(facade[:, x])[0]
                extents.append((int(ys.min()), int(ys.max())))
            start = 0
            while start < width:
                if extents[start] is None:
                    start += 1
                    continue
                end = start
                while end + 1 < width and extents[end + 1] == extents[start]:
                    end += 1
                top, bottom = extents[start]
                rects.append(f'<rect x="{start}" y="{top}" width="{end - start + 1}" '
                             f'height="{bottom - top + 1}"/>')
                start = end + 1
            layers.append(
                f'<g class="{cls}">'
                f'<g class="facades" fill="{colour}" fill-opacity="0.20">{"".join(rects)}</g>'
                f'<polyline class="skyline" fill="none" stroke="{colour}" '
                f'stroke-width="2.5" points="{points}"/>'
                f"</g>"
            )

        markers = []
        yaw = yaw_by_pano.get(pid)
        rows_here = by_pano.get(pid, pd.DataFrame()) if yaw is not None else pd.DataFrame()
        for _, row in rows_here.iterrows():
            x, y = dv.label_pixel_in_image(
                row["sv_image_x"], row["sv_image_y"], width, height, yaw
            )
            dist = row["horizontal_m"]
            markers.append(
                f'<g><circle cx="{x}" cy="{y}" r="9" fill="none" stroke="#111" '
                f'stroke-width="4"/><circle cx="{x}" cy="{y}" r="9" fill="none" '
                f'stroke="#ffd400" stroke-width="2"/>'
                f'<text x="{x + 13}" y="{y + 4}" font-size="13" fill="#ffd400" '
                f'stroke="#111" stroke-width="3.5" paint-order="stroke">'
                f'{row["label_type"]} · {dist:.1f} m</text></g>'
            )
        marker_layer = f'<g class="labels">{"".join(markers)}</g>' if markers else ""

        # Compass strip. The projection is the one thing a reader cannot check by eye,
        # so state it on the image: equirect column 0 is true bearing 0, and the ring
        # for a label sits at its own compass bearing. Verified against 395,147 labels
        # (depth-derived bearing tracks the recorded POV heading to a few degrees) and
        # against 17 panoramas whose heading is more than 90 deg off 180, where a
        # heading-centred convention would displace the depth by a third of the image.
        ticks = []
        for deg in range(0, 360, 45):
            tx = (((deg - (yaw_by_pano.get(pid) or 180.0) + 180.0) % 360.0) / 360.0) * width
            name = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][deg // 45]
            ticks.append(
                f'<line x1="{tx:.0f}" y1="0" x2="{tx:.0f}" y2="{height * 0.028:.0f}" '
                f'stroke="#fff" stroke-width="2" stroke-opacity="0.85"/>'
                f'<text x="{tx + 6:.0f}" y="{height * 0.032:.0f}" font-size="{height * 0.026:.0f}" '
                f'fill="#fff" stroke="#111" stroke-width="2.5" paint-order="stroke">{name}</text>'
            )
        compass = f'<g class="compass">{"".join(ticks)}</g>'

        info = meta.loc[pid] if pid in meta.index else None
        city = (info["city"] if info is not None and pd.notna(info.get("city")) else "—")
        year = (int(info["capture_year"])
                if info is not None and pd.notna(info.get("capture_year")) else "—")
        struct = (f"{info['structure_fraction'] * 100:.1f}%"
                  if info is not None and pd.notna(info.get("structure_fraction")) else "—")

        figures.append(
            f'<figure><div class="stage"><img src="{src}" alt="Street View panorama {pid}" '
            f'loading="lazy">'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
            f'{"".join(layers)}{marker_layer}{compass}</svg></div>'
            f"<figcaption><span><b>{city}</b> · captured {year} · "
            f"modelled structure above the horizon {struct}</span>"
            f"<span>pano <code>{pid}</code> · imagery {width}×{height} · Google</span>"
            f"</figcaption></figure>"
        )

    t1 = summary["t1_registration"]
    mirror_test = t1["paired_sign_test"]["x_mirror"]
    body = f"""<div class="wrap">
<h1>Does GSV depth describe the scene it claims to?</h1>
<p class="lede">Every panorama below is Google's own imagery, from the Street View tile server.
Drawn over it is the depth Google serves from a <em>different</em> endpoint: the blue line is where
the model runs out of surface, and the shading is where it places a wall. Nothing was fitted to the
imagery — where the two agree, they agree because the payload really is this panorama's scene.</p>
<p class="lede">Switch to the mirrored frame to see what a wrong convention looks like. The
earlier depth pilot could not have caught that error: it checked the pipeline against coordinates
the same pipeline had written.</p>
<p class="lede">Watch what the model leaves out. Trees, poles, wires and parked cars are simply
absent — it holds terrain and building footprints, nothing else. A label placed on something the
model does not contain gets the distance of the ground behind it instead. Rings mark stored
Sidewalk labels, with the distance the depth payload returns for each.</p>
<div class="legend">
  <span><i class="swatch" style="background:#2a78d6"></i>where the model ends (its skyline)</span>
  <span><i class="swatch" style="background:#2a78d6;opacity:.35"></i>modelled wall</span>
  <span><i class="swatch ring"></i>Sidewalk label, with depth distance</span>
  <span style="color:var(--muted)">column 0 is true north; a label sits at its own compass bearing</span>
</div>
<div class="controls">
  <button data-k="skyline">Skyline</button>
  <button data-k="facades">Walls</button>
  <button data-k="labels">Labels</button>
  <button data-k="compass">Compass</button>
  <button data-k="mirror">Mirror the depth map</button>
  <span class="note">{t1['panos_with_structure']} of {t1['panos_with_imagery']} panoramas model
  enough above the horizon to test. The true frame fits better on {mirror_test['identity_better']}
  of them, the mirror on {mirror_test['rival_better']}.</span>
</div>
{"".join(figures)}
</div>
<style>{GALLERY_CSS}</style>
<script>{GALLERY_JS}</script>"""

    out_path = os.path.join(args.fig_dir, "depth-overlay-gallery.html")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"<title>GSV depth overlay gallery</title>\n{body}\n")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"wrote {out_path} ({len(figures)} panoramas, {size_mb:.1f} MB)")
