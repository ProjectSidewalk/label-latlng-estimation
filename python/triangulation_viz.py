"""The issue #7 interactive conclusions page, built from committed artifacts.

Produces ``figures/triangulation-conclusions.html`` — a single self-contained page: an
orbitable 3D view of six real paterson sites (bearing rays, triangulated intersection,
same-pixel depth reads, an assumed-height slider), each camera's actual pixels juxtaposed
with Google's depth model over the identical window, the per-run implied-height charts,
and the §8 ratio-vs-offset adjudication. The page template lives next to this file
(``triangulation_viz_template.html``); everything it displays comes from the repository.

    python python/triangulation_viz.py fetch           # network: imagery tiles for the
                                                       #   six sites, the depth top-up and
                                                       #   the aerial ground patches
    python python/triangulation_viz.py fetch-basemaps  # network: aerial patches only,
                                                       #   leaving the GSV bundles alone
    python python/triangulation_viz.py build           # offline (~7 min): crops, depth
                                                       #   minis, ground, charts data
                                                       #   -> figures/triangulation-conclusions.html

Per this repo's archival rule (CLAUDE.md): the tiles are committed verbatim, so ``build``
and its page replay from a fresh checkout with no network. A re-``fetch`` observes a
different GSV state; regenerate by intent only. No number in the reports or tests depends
on this page — it is context for human eyes; its figures replay the committed summary.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

import gsv_depth as gd  # noqa: E402
import mapillary_falsification as mf  # noqa: E402
import run_depth_validation as rdv  # noqa: E402
import triangulation as tg  # noqa: E402
import triangulation_depth as td  # noqa: E402

DATA_DIR = tg.DATA_DIR
ROOT = tg.ROOT
TILES_BUNDLE = DATA_DIR / "triangulation-viz-tiles.jsonl.gz"
#: Depth payloads for showcase cameras that fall outside the §8 anchor's 480-pano sample,
#: so every camera view carries its depth-model panel. Kept separate from the anchor
#: bundle on purpose: the anchor population is locked by the findings tests and must not
#: grow, while these bytes are page context whose pixels carry no committed r_depth.
VIZ_PAYLOADS = DATA_DIR / "triangulation-viz-depth-payloads.jsonl.gz"
#: Aerial tiles under each showcase site's ground plane — orientation context for the
#: 3D scene, so a reader can see the corner the rays converge on. Like every other
#: fetched byte here it is committed verbatim and carries no number.
BASEMAPS = DATA_DIR / "triangulation-viz-basemaps.jsonl.gz"
CACHE = DATA_DIR / "triangulation-viz-cache"
TEMPLATE = Path(__file__).resolve().parent / "triangulation_viz_template.html"
OUT_HTML = ROOT / "figures" / "triangulation-conclusions.html"

#: The showcase run and the deterministic selection gates for its sites (see
#: :func:`pick_sites`): 4-6 panoramas, >=2 committed depth reads, median range 5-16 m,
#: max pairwise intersection angle >=60 deg — well-conditioned sites a reader can orbit.
VIZ_RUN = "paterson"
ZOOM, TILE, W, H = 3, 512, 4096, 2048
COL_HALF = 32.0 / 360.0          # crop: +/-32 deg of azimuth around the detection
ROW_LO, ROW_HI = 0.44, 0.80      # rows: elevation +10.8 deg .. -54 deg
OUT_W = 440                      # crop resize width, px

#: Aerial sources in preference order — (name, tile URL template, max zoom). The first
#: that serves real imagery for a site wins, and which one did is recorded per site and
#: named on the page. New Jersey's 2020 orthos are twice the ground resolution of Esri's
#: World Imagery here (0.11 vs 0.23 m/px at this latitude) but are, of course, New
#: Jersey only; World Imagery is the global fallback if the showcase run ever moves.
BASEMAP_SOURCES = (
    ("nj-orthos-2020",
     "https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2020_NJ_WM"
     "/MapServer/tile/{z}/{y}/{x}", 20),
    ("esri-world-imagery",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery"
     "/MapServer/tile/{z}/{y}/{x}", 19),
)
BASEMAP_HEADERS = {"User-Agent": "Mozilla/5.0 (label-latlng-estimation conclusions page)"}
BASEMAP_TILE = 256               # web-mercator tile size, px
BASEMAP_PX = 512                 # assembled patch, px per side
#: Esri answers 200 with a uniform "map data not yet available" placeholder past its real
#: zoom, so whether imagery exists has to be judged on the pixels, not the status code.
#: The placeholder holds ~120 distinct colours; genuine tiles here hold 9,000–24,000.
BASEMAP_MIN_COLOURS = 1000


# ======================================================================================
# Site selection and page data (all committed inputs)
# ======================================================================================

def pick_sites(f: "object", cmp_: "object") -> list:
    """The six showcase sites, by fixed gates then (depth coverage, size) — deterministic."""
    d = f[tg.usable(f)]
    cnt = cmp_.groupby("site_id").size()
    depth_by = cmp_.set_index(["site_id", "pano_id"])["r_depth"].to_dict()
    out = []
    for sid, gg in d[d["site_id"].isin(cnt[cnt >= 2].index)].groupby("site_id"):
        if not (4 <= len(gg) <= 6):
            continue
        if not (5.0 <= float(np.median(gg["r_tri"])) <= 16.0):
            continue
        ang = tg._max_intersection_angle(gg["bearing_deg"].to_numpy())
        if ang < 60:
            continue
        out.append((sid, gg, int(cnt.get(sid, 0)), round(float(ang), 1), depth_by))
    out.sort(key=lambda x: (-x[2], -len(x[1])))
    return out[:6]


def page_data(quick: bool = False) -> dict:
    """Everything the template's charts read, extracted from the committed summary and
    the (deterministically re-fitted) member frames."""
    s = json.load(open(DATA_DIR / "triangulation-summary.json", encoding="utf-8"))
    out = {"summary": {
        "imagery": s["imagery"],
        "scale_global": {r: {k: v[k] for k in ("k", "height_m", "ci95_m", "n", "n_sites")
                             if k in v} for r, v in s["scale_global"].items()},
        "scale_median": {r: s["scale"][r]["median_m"] for r in s["scale"]},
        "gap_range_profile": s["depth_anchor"]["gap_range_profile"],
        "gap_by_capture_year": s["depth_anchor"]["gap_by_capture_year"],
        "pooled_ratio": s["depth_anchor"]["pooled"]["median_ratio_tri_over_depth"],
        "pooled_gap_m": round(-s["depth_anchor"]["pooled"]["median_signed_diff_m"], 3),
        "n_anchor_panos": s["depth_anchor"]["n_panos"],
        "n_anchor_detections": s["depth_anchor"]["n_detections"],
        "position_drift": {k: s["depth_anchor"]["position_drift"][k]
                           for k in ("n", "median_m", "max_m")},
        "real_geometry": {r: v["bias_factor"]
                          for r, v in s["validation"]["real_geometry"].items()},
        "split_half": {r: s["split_half"][r]["implied_position_sigma_m"]
                       for r in s["split_half"]},
        "noise": {r: {"sb": s["noise"][r]["sigma_bearing_deg"],
                      "sp": s["noise"][r]["sigma_pos_m"]} for r in s["noise"]},
        "shipped": s["meta"]["shipped_height_m"],
        "assumed": s["meta"]["assumed_height_m"],
    }}

    runs = [VIZ_RUN] if quick else list(tg.ALL_RUNS)
    curves = {}
    frames = {}
    for run in runs:
        print(f"  [{run}] fitting ...", flush=True)
        f = tg.fit_noise(run)["frame"]
        frames[run] = f
        d = f[tg.usable(f)]
        # the same L1 scatter fit_model_scale minimises, swept over assumed height
        th = np.radians(d["bearing_deg"].to_numpy())
        ux, uy = np.sin(th), np.cos(th)
        pe, pn = d["pano_e"].to_numpy(), d["pano_n"].to_numpy()
        r = d["range_m"].to_numpy()
        _, g = np.unique(d["site_id"].to_numpy(), return_inverse=True)
        nsz = np.bincount(g)
        ks = np.arange(0.75, 1.1500001, 0.005)
        losses = []
        for k in ks:
            px, py = pe + k * r * ux, pn + k * r * uy
            cx, cy = np.bincount(g, px) / nsz, np.bincount(g, py) / nsz
            losses.append(float(np.mean(np.hypot(px - cx[g], py - cy[g]))))
        curves[run] = {"h": [round(float(k * 2.6), 4) for k in ks],
                       "scatter": [round(x, 4) for x in losses]}
    out["curves"] = curves
    out["frames"] = frames          # consumed (and stripped) by build()
    return out


# ======================================================================================
# Imagery: fetch tiles for the showcase sites; build crops from the committed bundle
# ======================================================================================

class _Throttle:
    def __init__(self, dt): self.dt, self.last = dt, 0.0
    def wait(self):
        d = self.dt - (time.time() - self.last)
        if d > 0:
            time.sleep(d)
        self.last = time.time()


def _crop_tiles(cf: float) -> tuple[list, int]:
    """(x, y) tile indices covering the crop window at column-fraction ``cf``."""
    x0 = (cf - COL_HALF) * W
    x1 = (cf + COL_HALF) * W
    c0 = int(np.floor(x0 / TILE))
    cols = range(c0, int(np.floor(x1 / TILE)) + 1)
    rows = range(int(ROW_LO * H) // TILE, (int(ROW_HI * H) - 1) // TILE + 1)
    return [(cx, cy) for cx in cols for cy in rows], c0


def _members(f, cmp_):
    """(site_id, member row, heading, column fraction) for every showcase member."""
    heading = mf.load_panos(VIZ_RUN).set_index("pano_id")["camera_heading"]
    for sid, gg, n_depth, ang, depth_by in pick_sites(f, cmp_):
        for m in gg.itertuples():
            hd = float(heading.loc[m.pano_id])
            cf = ((m.bearing_deg - hd + 180.0) % 360.0) / 360.0
            yield sid, m, n_depth, ang, depth_by, cf


def _fit_for_fetch() -> tuple:
    print(f"  [{VIZ_RUN}] fitting ...", flush=True)
    f = tg.fit_noise(VIZ_RUN)["frame"]
    cmp_ = td._comparable(td.depth_ranges(VIZ_RUN, f, td.load_payloads(), td.load_panos()))
    return f, cmp_


def _write_basemaps(bm: list) -> dict:
    with gzip.open(BASEMAPS, "wt", encoding="utf-8", newline="\n") as fh:
        for rec in bm:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return {"n_basemaps": len(bm),
            "n_basemap_tiles": sum(len(r["tiles"]) for r in bm),
            "basemap_sources": sorted({r["source"] for r in bm}),
            "basemaps_bundle": str(BASEMAPS)}


def fetch_basemaps_only() -> dict:
    """Just the aerial patches, leaving the committed GSV bundles untouched.

    Its own stage on purpose: those bundles are the evidence this page replays, and a
    whole re-``fetch`` observes a different remote state, so adding ground context must
    not be a reason to disturb them.
    """
    f, cmp_ = _fit_for_fetch()
    return _write_basemaps(fetch_basemaps(f, cmp_, requests.Session(), _Throttle(0.12)))


def fetch() -> dict:
    """Fetch the tiles every crop needs and commit them verbatim. The only network stage."""
    f, cmp_ = _fit_for_fetch()
    session = requests.Session()
    throttle = _Throttle(0.12)
    need = {}
    for _, m, _, _, _, cf in _members(f, cmp_):
        tiles, _ = _crop_tiles(cf)
        need.setdefault(m.pano_id, set()).update((cx % (W // TILE), cy) for cx, cy in tiles)
    n = 0
    with gzip.open(TILES_BUNDLE, "wt", encoding="utf-8", newline="\n") as fh:
        for pano_id in sorted(need):
            rec = {"pano_id": pano_id, "zoom": ZOOM, "tiles": []}
            for cx, cy in sorted(need[pano_id]):
                data, _ = rdv.fetch_tile_cached(pano_id, ZOOM, cx, cy, str(CACHE),
                                                throttle, session)
                rec["tiles"].append({"x": cx, "y": cy,
                                     "jpg_b64": base64.b64encode(data).decode()})
                n += 1
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # Depth top-up: payloads for showcase cameras outside the anchor's 480-pano sample.
    anchor_payloads = td.load_payloads()
    missing = sorted(p for p in need if p not in anchor_payloads)
    vp = []
    for pano_id in missing:
        throttle.wait()
        resp = gd.fetch_photometa_raw(pano_id)
        b64 = gd.extract_depth_b64(resp)
        if b64:
            vp.append({"pano_id": pano_id, "run": VIZ_RUN, "depth_b64": b64})
    with gzip.open(VIZ_PAYLOADS, "wt", encoding="utf-8", newline="\n") as fh:
        for p in vp:
            fh.write(json.dumps(p, sort_keys=True) + "\n")

    out = {"n_panos": len(need), "n_tiles": n, "bundle": str(TILES_BUNDLE),
           "n_viz_payloads": len(vp), "n_missing": len(missing),
           "payloads_bundle": str(VIZ_PAYLOADS)}
    out.update(_write_basemaps(fetch_basemaps(f, cmp_, session, throttle)))
    return out


def load_tiles() -> dict:
    """pano_id -> {(x, y): jpeg bytes} from the committed bundle."""
    out = {}
    with gzip.open(TILES_BUNDLE, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["pano_id"]] = {(t["x"], t["y"]): base64.b64decode(t["jpg_b64"])
                                   for t in rec["tiles"]}
    return out


def crop_image(tiles: dict, cf: float) -> Image.Image:
    coords, c0 = _crop_tiles(cf)
    rows0 = int(ROW_LO * H) // TILE
    ncols = max(cx for cx, _ in coords) - c0 + 1
    nrows = max(cy for _, cy in coords) - rows0 + 1
    canvas = Image.new("RGB", (ncols * TILE, nrows * TILE))
    for cx, cy in coords:
        img = Image.open(io.BytesIO(tiles[(cx % (W // TILE), cy)]))
        canvas.paste(img, ((cx - c0) * TILE, (cy - rows0) * TILE))
    x0 = (cf - COL_HALF) * W
    px0 = int(x0 - c0 * TILE)
    y0, y1 = int(ROW_LO * H), int(ROW_HI * H)
    img = canvas.crop((px0, y0 - rows0 * TILE,
                       px0 + int(2 * COL_HALF * W), y1 - rows0 * TILE))
    s = OUT_W / img.width
    return img.resize((OUT_W, int(img.height * s)), Image.LANCZOS)


_VIRIDIS = [(68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142), (33, 144, 141),
            (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37)]


def _colormap(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(np.clip(v, 0, 1)) * (len(_VIRIDIS) - 1)
    i = np.clip(v.astype(int), 0, len(_VIRIDIS) - 2)
    frac = (v - i)[..., None]
    a = np.array(_VIRIDIS, float)[i]
    b = np.array(_VIRIDIS, float)[i + 1]
    return (a + (b - a) * frac).astype(np.uint8)


def depth_crop(t_raster: np.ndarray, cf: float) -> Image.Image:
    """Google's modelled ray distance over the same heading-centred window (log 2-40 m)."""
    hd, wd = t_raster.shape
    ow = 220
    # Same aspect as the photo crop it sits beside: the window spans int(2*COL_HALF*W)
    # equirect columns by (ROW_HI-ROW_LO)*H rows — and H = W/2, so it is very nearly
    # square. Deriving it any other way (in azimuth/elevation fractions, say) drops or
    # doubles that factor of two and the panel renders at the wrong height.
    oh = int(ow * (int(ROW_HI * H) - int(ROW_LO * H)) / int(2 * COL_HALF * W))
    fx = (cf - COL_HALF + (np.arange(ow) + 0.5) / ow * 2 * COL_HALF) % 1.0
    fy = ROW_LO + (np.arange(oh) + 0.5) / oh * (ROW_HI - ROW_LO)
    cols = np.clip((fx * wd).astype(int), 0, wd - 1)
    rows = np.clip((fy * hd).astype(int), 0, hd - 1)
    t = t_raster[np.ix_(rows, cols)]
    v = (np.log(np.clip(t, 2.0, 40.0)) - np.log(2.0)) / (np.log(40.0) - np.log(2.0))
    rgb = _colormap(1.0 - v)                     # near = bright
    rgb[~np.isfinite(t)] = (24, 26, 30)          # no plane (sky)
    return Image.fromarray(rgb)


def _to_uri(img: Image.Image, fmt: str = "JPEG", q: int = 72) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, quality=q) if fmt == "JPEG" else img.save(buf, fmt, optimize=True)
    kind = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{kind};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ======================================================================================
# Aerial basemap: one georeferenced patch per site, so the scene sits on real ground
# ======================================================================================

def _run_origin(f: "object") -> tuple[float, float]:
    """The run's ENU origin (lat0, lng0), recovered by inverting :func:`tg.local_en`.

    Read back from a row rather than recomputed as the mean of the frame's latitudes:
    the mean is taken inside the fit, over whatever rows survived filtering there, so
    only the inverse is guaranteed to name the origin the ``pano_e``/``pano_n`` in hand
    were actually measured from.
    """
    r = f.iloc[0]
    lat0 = float(r["pano_lat"]) - float(np.degrees(float(r["pano_n"]) / tg.EARTH_R))
    lng0 = float(r["pano_lng"]) - float(np.degrees(
        float(r["pano_e"]) / (tg.EARTH_R * np.cos(np.radians(lat0)))))
    return lat0, lng0


def site_ext_m(gg: "object", ce: float, cn: float) -> int:
    """The scene's grid half-extent in metres. The page reads this back rather than
    recomputing it, so the aerial texture and the geometry drawn on it cannot disagree."""
    far = max(float(np.hypot(r.pano_e - ce, r.pano_n - cn)) for r in gg.itertuples())
    return int(np.ceil(max(far + 4.0, 12.0) / 2.0) * 2)


def _merc_px(lat: float, lng: float, zoom: int) -> tuple[float, float]:
    """Web-Mercator global pixel coordinates at ``zoom``."""
    n = BASEMAP_TILE * 2 ** zoom
    s = np.sin(np.radians(lat))
    return ((lng + 180.0) / 360.0 * n,
            (0.5 - np.log((1.0 + s) / (1.0 - s)) / (4.0 * np.pi)) * n)


def _patch_box(lat: float, lng: float, ext: float, zoom: int) -> tuple[float, float, float, float]:
    """Global-pixel box of the ground square the scene draws — the ENU +/-``ext`` metres
    the grid covers, carried through the very same tangent plane the geometry uses."""
    n_lat, _ = tg.en_to_latlng(0.0, ext, lat, lng)
    s_lat, _ = tg.en_to_latlng(0.0, -ext, lat, lng)
    _, e_lng = tg.en_to_latlng(ext, 0.0, lat, lng)
    _, w_lng = tg.en_to_latlng(-ext, 0.0, lat, lng)
    x0, y0 = _merc_px(float(n_lat), float(w_lng), zoom)
    x1, y1 = _merc_px(float(s_lat), float(e_lng), zoom)
    return x0, y0, x1, y1


def _is_imagery(jpg: bytes) -> bool:
    a = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).reshape(-1, 3)
    return len(np.unique(a, axis=0)) >= BASEMAP_MIN_COLOURS


def fetch_basemaps(f: "object", cmp_: "object", session, throttle) -> list[dict]:
    """Aerial tiles covering each showcase site's ground square, verbatim."""
    lat0, lng0 = _run_origin(f)
    out = []
    for sid, gg, _, _, _ in pick_sites(f, cmp_):
        ce, cn = float(gg["loo_e"].median()), float(gg["loo_n"].median())
        ext = site_ext_m(gg, ce, cn)
        lat, lng = (float(v) for v in tg.en_to_latlng(ce, cn, lat0, lng0))
        for name, url, zoom in BASEMAP_SOURCES:
            x0, y0, x1, y1 = _patch_box(lat, lng, ext, zoom)
            tiles, ok = [], True
            for tx in range(int(x0 // BASEMAP_TILE), int(x1 // BASEMAP_TILE) + 1):
                for ty in range(int(y0 // BASEMAP_TILE), int(y1 // BASEMAP_TILE) + 1):
                    throttle.wait()
                    r = session.get(url.format(z=zoom, x=tx, y=ty),
                                    headers=BASEMAP_HEADERS, timeout=30)
                    if r.status_code != 200 or "image" not in r.headers.get("content-type", ""):
                        ok = False
                        break
                    tiles.append({"x": tx, "y": ty,
                                  "jpg_b64": base64.b64encode(r.content).decode()})
                if not ok:
                    break
            if ok and all(_is_imagery(base64.b64decode(t["jpg_b64"])) for t in tiles):
                print(f"  [basemap] site {sid}: {name} z{zoom}, {len(tiles)} tiles, "
                      f"{2 * ext} m square", flush=True)
                out.append({"site": str(sid), "lat": round(lat, 7), "lng": round(lng, 7),
                            "ext": ext, "source": name, "zoom": zoom, "tiles": tiles})
                break
        else:
            print(f"  [basemap] no aerial source served site {sid} — "
                  f"the scene falls back to its bare grid", flush=True)
    return out


def load_basemaps() -> dict:
    """site_id -> basemap record from the committed bundle ({} if never fetched)."""
    if not BASEMAPS.exists():
        return {}
    out = {}
    with gzip.open(BASEMAPS, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["site"]] = rec
    return out


def basemap_patch(rec: dict) -> Image.Image:
    """The site's ground square, north-up, assembled offline from committed tiles."""
    tiles = {(t["x"], t["y"]): base64.b64decode(t["jpg_b64"]) for t in rec["tiles"]}
    x0, y0, x1, y1 = _patch_box(rec["lat"], rec["lng"], rec["ext"], rec["zoom"])
    tx0, ty0 = int(x0 // BASEMAP_TILE), int(y0 // BASEMAP_TILE)
    tx1, ty1 = int(x1 // BASEMAP_TILE), int(y1 // BASEMAP_TILE)
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * BASEMAP_TILE,
                               (ty1 - ty0 + 1) * BASEMAP_TILE))
    for (tx, ty), jpg in tiles.items():
        canvas.paste(Image.open(io.BytesIO(jpg)).convert("RGB"),
                     ((tx - tx0) * BASEMAP_TILE, (ty - ty0) * BASEMAP_TILE))
    # a float box, so the square lands on the tangent plane to well under a pixel
    box = (x0 - tx0 * BASEMAP_TILE, y0 - ty0 * BASEMAP_TILE,
           x1 - tx0 * BASEMAP_TILE, y1 - ty0 * BASEMAP_TILE)
    return canvas.resize((BASEMAP_PX, BASEMAP_PX), Image.LANCZOS, box=box)


# ======================================================================================
# Build
# ======================================================================================

def build(quick: bool = False) -> dict:
    data = page_data(quick=quick)
    frames = data.pop("frames")
    f = frames[VIZ_RUN]
    # r_depth values come from the ANCHOR payloads only — the committed §8 population.
    # The merged set adds the viz top-up so every camera can render its depth panel.
    anchor_payloads = td.load_payloads()
    payloads = {**anchor_payloads, **td.load_payloads(VIZ_PAYLOADS)}
    cmp_ = td._comparable(td.depth_ranges(VIZ_RUN, f, anchor_payloads, td.load_panos()))
    tiles = load_tiles()
    basemaps = load_basemaps()
    s = json.load(open(DATA_DIR / "triangulation-summary.json", encoding="utf-8"))

    rasters = {}
    sites = []
    for sid, gg, n_depth, ang, depth_by in pick_sites(f, cmp_):
        ce, cn = float(gg["loo_e"].median()), float(gg["loo_n"].median())
        heading = mf.load_panos(VIZ_RUN).set_index("pano_id")["camera_heading"]
        mem = []
        for m in gg.itertuples():
            hd = float(heading.loc[m.pano_id])
            cf = ((m.bearing_deg - hd + 180.0) % 360.0) / 360.0
            rd = depth_by.get((sid, m.pano_id))
            rec = {"e": round(m.pano_e - ce, 3), "n": round(m.pano_n - cn, 3),
                   "b": round(m.bearing_deg, 3), "dep": round(m.dep_deg, 3),
                   "rt": round(m.r_tri, 3), "r26": round(m.range_m, 3),
                   "rd": round(float(rd), 3) if rd is not None else None,
                   "img": _to_uri(crop_image(tiles[m.pano_id], cf)),
                   "cx": 0.5,
                   "cy": round((m.y_normalized - ROW_LO) / (ROW_HI - ROW_LO), 4),
                   "hz": round((0.5 - ROW_LO) / (ROW_HI - ROW_LO), 4),
                   "pano": m.pano_id[:10] + "…"}
            if m.pano_id in payloads:
                if m.pano_id not in rasters:
                    rasters[m.pano_id] = gd.compute_depth_t(
                        gd.decode_depth_payload(payloads[m.pano_id]))
                rec["dimg"] = _to_uri(depth_crop(rasters[m.pano_id], cf), "PNG")
            mem.append(rec)
        rec = {"run": VIZ_RUN, "site": str(sid),
               "height": s["scale_global"][VIZ_RUN]["height_m"],
               "n_depth": n_depth, "angle": ang,
               "ext": site_ext_m(gg, ce, cn), "members": mem}
        bm = basemaps.get(str(sid))
        if bm is not None:
            # The patch is cut for the extent the scene draws; if a re-fetch ever moved
            # one, the texture would silently sit at the wrong scale. Refuse instead.
            if bm["ext"] != rec["ext"]:
                raise SystemExit(f"site {sid}: basemap covers {bm['ext']} m but the scene "
                                 f"draws {rec['ext']} m — re-run `fetch`")
            rec["ground"] = {"img": _to_uri(basemap_patch(bm), "JPEG", 80),
                             "src": bm["source"], "zoom": bm["zoom"]}
        sites.append(rec)
    data["sites"] = sites

    tpl = TEMPLATE.read_text(encoding="utf-8")
    html = tpl.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
    OUT_HTML.write_text(html, encoding="utf-8", newline="\n")
    return {"sites": len(sites),
            "crops": sum(len(x["members"]) for x in sites),
            "depth_minis": sum(1 for x in sites for m in x["members"] if "dimg" in m),
            "basemaps": sum(1 for x in sites if "ground" in x),
            "html_kb": len(html) // 1024, "out": str(OUT_HTML)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fetch", "fetch-basemaps", "build"],
                    nargs="?", default="build")
    ap.add_argument("--quick", action="store_true",
                    help="build: skip the five non-showcase runs' scatter curves")
    args = ap.parse_args()
    if args.stage == "fetch":
        print(json.dumps(fetch(), indent=2))
    elif args.stage == "fetch-basemaps":
        print(json.dumps(fetch_basemaps_only(), indent=2))
    else:
        print(json.dumps(build(quick=args.quick), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
