"""Issue #4 pilot: fresh GSV depth vs the recovered depth-derived ground truth.

Three subcommands, run from the repo root:

    python python/run_depth_pilot.py fetch     # network: sample panos, fill the cache
    python python/run_depth_pilot.py build     # offline: cache -> committed artifacts
    python python/run_depth_pilot.py figures   # offline: artifacts -> figures/fig7,8

Part A samples panoramas from the recovered dataset (seed 666), fetches today's
depth payload for each surviving pano id, recomputes every raw label row with the
bit-level v6 replica in gsv_depth.py, and compares against the stored positions
on the float32 storage lattice (~84% of stored coordinates are exact float32;
one ulp is ~0.4-0.7 m here, so THAT is the agreement floor, not centimeters).

Part B asks a different question: at ~100 recovered-label locations each in
seattle and cdmx, what does the CURRENT panorama serve? (Depth availability,
resolution, per-pano camera height.) The by-location endpoint cannot return
depth, so each location costs two requests: search, then photometa by id.

Everything build/figures needs lives in data/depth-pilot-cache/ (gitignored);
`fetch` is idempotent over it, and `build` is deterministic from it (gzip
members are written with mtime=0 so rebuilt artifacts are byte-identical).

Classification of a Part A pano (payload state is a pano-level fact):
    gone       id no longer resolves (response code 2)
    no_depth   metadata resolves but no depth payload is served
    unchanged  every compared label within CONSISTENT_ULP per axis
    mostly_unchanged  >= 2/3 of labels within CONSISTENT_ULP (local plane edits)
    changed    the payload no longer produces the stored positions
Compared labels are cleaned-frame rows with finite stored lat/lng that hit a
plane (no-plane/out-of-bounds/absurd rows are scored separately).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gsv_depth as gd  # noqa: E402
from label_latlng_estimation import clean_data, haversine_m, load_data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 666  # the repo's sampling seed everywhere

# Part A allocation: fetch attempts per stratum. The calibration probe (n=10)
# saw ~80% of 2017-2020 ids fail to resolve, so the oversample is aggressive;
# every attempt is recorded and the per-stratum hit rate is itself a finding.
PART_A_CITY_ATTEMPTS = {
    "dc": 150, "seattle": 120, "spgg": 75, "columbus": 75,
    "newberg": 75, "cdmx": 45, "pittsburgh": 30,
}
EDGE_ATTEMPTS = {"absurd": 10, "seam_wrap": 10, "dc_overflow": 10}
PART_B_CITIES = ["seattle", "cdmx"]
PART_B_LOCATIONS_PER_CITY = 100
PART_B_MIN_SPACING_M = 100.0  # keep sampled locations from collapsing onto one pano

CONSISTENT_ULP = 2.0  # per-axis lattice tolerance; sensitivity reported at 1.5/2/3

REQUEST_TIMESTAMP = None  # set per fetch run, recorded in the fetch log


# ---------------------------------------------------------------------------- cache

def cache_dirs(cache_root):
    photometa = os.path.join(cache_root, "photometa")
    search = os.path.join(cache_root, "search")
    os.makedirs(photometa, exist_ok=True)
    os.makedirs(search, exist_ok=True)
    return photometa, search


def fetch_log_path(cache_root):
    return os.path.join(cache_root, "fetch-log.jsonl")


def log_fetch(cache_root, record):
    with open(fetch_log_path(cache_root), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_fetch_times(cache_root):
    """pano_id -> first fetch timestamp, from the append-only fetch log."""
    times = {}
    path = fetch_log_path(cache_root)
    if not os.path.exists(path):
        return times
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "photometa":
                times.setdefault(rec["key"], rec["ts"])
    return times


class Throttle:
    def __init__(self, rps):
        self.min_gap = 1.0 / rps
        self.last = 0.0

    def wait(self):
        gap = time.monotonic() - self.last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self.last = time.monotonic()


def fetch_photometa_cached(pano_id, cache_root, throttle, session):
    photometa, _ = cache_dirs(cache_root)
    path = os.path.join(photometa, pano_id + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), False
    last_err = None
    for attempt, backoff in enumerate([0, 2, 8]):
        if backoff:
            time.sleep(backoff)
        throttle.wait()
        try:
            resp = gd.fetch_photometa_raw(pano_id, session=session)
            break
        except Exception as e:  # requests errors, transient JSON hiccups
            last_err = e
    else:
        raise RuntimeError(f"photometa fetch failed for {pano_id}: {last_err}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resp, f)
    log_fetch(cache_root, {"ts": now_utc(), "kind": "photometa", "key": pano_id})
    return resp, True


def search_cached(lat, lng, cache_root, throttle, session):
    _, search = cache_dirs(cache_root)
    key = f"{lat:.6f}_{lng:.6f}"
    path = os.path.join(search, key + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), False
    from streetlevel import streetview

    last_err = None
    for attempt, backoff in enumerate([0, 2, 8]):
        if backoff:
            time.sleep(backoff)
        throttle.wait()
        try:
            pano = streetview.find_panorama(lat, lng, radius=50, session=session)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"search failed for {key}: {last_err}")
    result = {"pano_id": pano.id if pano else None}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    log_fetch(cache_root, {"ts": now_utc(), "kind": "search", "key": key})
    return result, True


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------- sampling

def load_frames(data_dir):
    """(raw with cleaned-style column names + flags, cleaned) for the whole repo."""
    raw = load_data(data_dir)
    cleaned, _ = clean_data(raw)
    raw = raw.rename(columns={
        "panorama_lat": "pano_lat", "panorama_lng": "pano_lng",
        "gsv_panorama_id": "pano_id",
    })
    raw["in_cleaned"] = raw["label_id"].isin(set(cleaned["label_id"]))
    raw["stored_absurd"] = ~(
        raw["lat"].between(-90, 90) & raw["lng"].between(-180, 180)
    ) | raw["lat"].isna() | raw["lng"].isna()
    # ceil(x * (1/26)) >= 512 walks into the next raster row -- compute it with
    # the exact lookup expression, not a hand-derived pixel threshold.
    px = raw["sv_image_x"].to_numpy(np.float64) * gd.SV_IMAGE_SCALE
    raw["seam_wrap"] = np.ceil(px) >= gd.DEPTH_W
    return raw, cleaned


def pano_frame(raw):
    g = raw.groupby("pano_id", sort=True)
    f = pd.DataFrame({
        "city": g["city"].first(),
        "n_labels_raw": g.size(),
        "n_labels_cleaned": g["in_cleaned"].sum(),
        "pano_height": g["pano_height"].first(),
        "has_absurd": g["stored_absurd"].any(),
        "has_seam_wrap": g["seam_wrap"].any(),
        "max_sv_image_x": g["sv_image_x"].max(),
    })
    return f.reset_index()


def sample_part_a(raw):
    """Seeded stratified sample of pano ids -> DataFrame(pano_id, city, stratum)."""
    rng = np.random.default_rng(SEED)
    panos = pano_frame(raw)
    picked = []

    def take(cands, k, stratum):
        cands = cands[~cands["pano_id"].isin({p for p, _, _ in picked})]
        k = min(k, len(cands))
        if k <= 0:
            return
        idx = rng.choice(len(cands), size=k, replace=False)
        for _, row in cands.iloc[idx].iterrows():
            picked.append((row["pano_id"], row["city"], stratum))

    # Headline strata: per city, split across resolution classes proportionally
    # (8192 / 6656 / other-or-null), preferring panos with >= 2 cleaned labels --
    # multi-label panos are what the coherence classification needs.
    for city, n_attempts in PART_A_CITY_ATTEMPTS.items():
        sub = panos[(panos["city"] == city) & (panos["n_labels_cleaned"] > 0)]
        multi = sub[sub["n_labels_cleaned"] >= 2]
        pool = multi if len(multi) >= n_attempts else sub
        classes = {
            "h8192": pool[pool["pano_height"] == 8192],
            "h6656": pool[pool["pano_height"] == 6656],
            "hnull": pool[~pool["pano_height"].isin([8192, 6656])],
        }
        total = sum(len(c) for c in classes.values())
        for name, cands in classes.items():
            if len(cands) == 0:
                continue
            k = max(1, round(n_attempts * len(cands) / total))
            take(cands, k, f"headline_{name}")

    # Edge strata: reported separately, never in headline metrics.
    take(panos[panos["has_absurd"]], EDGE_ATTEMPTS["absurd"], "edge_absurd")
    take(panos[panos["has_seam_wrap"]], EDGE_ATTEMPTS["seam_wrap"], "edge_seam_wrap")
    take(
        panos[panos["max_sv_image_x"] > 13312],
        EDGE_ATTEMPTS["dc_overflow"],
        "edge_dc_overflow",
    )

    return pd.DataFrame(picked, columns=["pano_id", "city", "stratum"])


def sample_part_b(cleaned):
    """Spaced label locations per Part B city -> DataFrame(city, lat, lng)."""
    rng = np.random.default_rng(SEED + 1)  # independent stream from Part A
    rows = []
    for city in PART_B_CITIES:
        locs = (
            cleaned[cleaned["city"] == city]
            .groupby("pano_id")[["pano_lat", "pano_lng"]]
            .first()
            .reset_index(drop=True)
        )
        order = rng.permutation(len(locs))
        kept = []
        for i in order:
            lat, lng = float(locs.iloc[i, 0]), float(locs.iloc[i, 1])
            if any(
                haversine_m(lng, lat, klng, klat) < PART_B_MIN_SPACING_M
                for klat, klng in kept
            ):
                continue
            kept.append((lat, lng))
            if len(kept) >= PART_B_LOCATIONS_PER_CITY:
                break
        rows.extend((city, lat, lng) for lat, lng in kept)
    return pd.DataFrame(rows, columns=["city", "lat", "lng"])


# ---------------------------------------------------------------------------- fetch

def cmd_fetch(args):
    raw, cleaned = load_frames(args.data_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    sample_a_path = os.path.join(args.cache_dir, "sample-a.json")
    sample_b_path = os.path.join(args.cache_dir, "sample-b.json")
    if os.path.exists(sample_a_path):
        sample_a = pd.read_json(sample_a_path)
        print(f"Part A sample: reusing {len(sample_a)} panos from cache")
    else:
        sample_a = sample_part_a(raw)
        sample_a.to_json(sample_a_path, orient="records", indent=2)
        print(f"Part A sample: {len(sample_a)} panos drawn (seed {SEED})")
    if os.path.exists(sample_b_path):
        sample_b = pd.read_json(sample_b_path)
        print(f"Part B sample: reusing {len(sample_b)} locations from cache")
    else:
        sample_b = sample_part_b(cleaned)
        sample_b.to_json(sample_b_path, orient="records", indent=2)
        print(f"Part B sample: {len(sample_b)} locations drawn (seed {SEED + 1})")

    import requests

    session = requests.Session()
    throttle = Throttle(args.rps)

    fetched = hit = 0
    for i, row in sample_a.iterrows():
        resp, was_network = fetch_photometa_cached(
            row["pano_id"], args.cache_dir, throttle, session
        )
        fetched += was_network
        hit += gd.extract_pano_meta(resp) is not None
        if (i + 1) % 50 == 0:
            print(f"  Part A {i + 1}/{len(sample_a)} ({hit} resolve so far)")
    print(f"Part A: {len(sample_a)} panos, {fetched} network fetches, {hit} resolve")

    fetched = found = 0
    for i, row in sample_b.iterrows():
        result, was_network = search_cached(
            row["lat"], row["lng"], args.cache_dir, throttle, session
        )
        fetched += was_network
        if result["pano_id"]:
            found += 1
            _, was_network2 = fetch_photometa_cached(
                result["pano_id"], args.cache_dir, throttle, session
            )
            fetched += was_network2
        if (i + 1) % 50 == 0:
            print(f"  Part B {i + 1}/{len(sample_b)}")
    print(f"Part B: {len(sample_b)} locations, {found} with a current pano, "
          f"{fetched} network fetches")


# ---------------------------------------------------------------------------- build

def classify_pano(compared):
    """Pano class from the per-label consistency flags of compared labels."""
    if len(compared) == 0:
        return "no_comparable_labels"
    frac = compared.mean()
    if frac == 1.0:
        return "unchanged"
    if frac >= 2 / 3:
        return "mostly_unchanged"
    return "changed"


def analyze_pano(pano_id, resp, labels):
    """One Part A pano: recompute every raw label row, classify, QC.

    Returns (pano_record, label_records). labels is the raw-frame slice.
    """
    meta = gd.extract_pano_meta(resp)
    rec = {
        "pano_id": pano_id,
        "n_labels_raw": len(labels),
        "n_labels_cleaned": int(labels["in_cleaned"].sum()),
    }
    if meta is None:
        rec["status"] = "gone"
        return rec, []
    b64 = gd.extract_depth_b64(resp)
    rec.update({
        "fresh_lat": meta["lat"], "fresh_lng": meta["lng"],
        "fresh_heading_deg": meta["heading_deg"],
        "fresh_pitch_deg": meta["pitch_deg"], "fresh_roll_deg": meta["roll_deg"],
        "fresh_elevation_m": meta["elevation_m"],
        "capture_year": meta["capture_year"], "capture_month": meta["capture_month"],
        "image_sizes": meta["image_sizes"],
    })
    if len(labels):
        splat = labels["pano_lat"].iloc[0]
        splng = labels["pano_lng"].iloc[0]
        rec["stored_pano_lat"] = splat
        rec["stored_pano_lng"] = splng
        rec["pano_shift_m"] = float(haversine_m(splng, splat, meta["lng"], meta["lat"]))
        ph = labels["photographer_heading"].iloc[0]
        if pd.notna(ph) and meta["heading_deg"] is not None:
            rec["heading_shift_deg"] = float(
                (meta["heading_deg"] - ph + 180) % 360 - 180
            )
    if b64 is None:
        rec["status"] = "no_depth"
        return rec, []

    payload = gd.decode_depth_payload(b64)
    qc = gd.camera_height_qc(payload)
    cloud = gd.compute_point_cloud(payload)
    rec.update({
        "status": "ok",
        "was_compressed": payload.was_compressed,
        "n_planes": qc.n_planes,
        "ground_d": qc.ground_d,
        "ground_height_m": qc.ground_height,
        "ground_tilt_deg": qc.ground_tilt_deg,
        "ground_pixel_share": qc.ground_pixel_share,
        "height_is_default": qc.is_default,
        "ground_d_exactly_2p5": qc.ground_d == gd.DEFAULT_CAMERA_HEIGHT,
        "band_height_median": qc.band_height_median,
        "band_height_mad": qc.band_height_mad,
    })

    label_rows = []
    consistency = []
    for _, r in labels.iterrows():
        out = gd.v6_to_latlng(
            int(r["sv_image_x"]), int(r["sv_image_y"]),
            float(r["pano_lat"]), float(r["pano_lng"]), cloud,
        )
        lr = {
            "label_id": int(r["label_id"]),
            "pano_id": pano_id,
            "city": r["city"],
            "label_type": r["label_type"],
            "zoom": int(r["zoom"]),
            "sv_image_x": int(r["sv_image_x"]),
            "sv_image_y": int(r["sv_image_y"]),
            "in_cleaned": bool(r["in_cleaned"]),
            "stored_lat": r["lat"], "stored_lng": r["lng"],
            "stored_absurd": bool(r["stored_absurd"]),
            "recomputed_lat": out.lat, "recomputed_lng": out.lng,
            "no_plane": out.no_plane,
            "seam_wrap": out.seam_wrap,
            "out_of_bounds": out.out_of_bounds,
        }
        comparable = (
            not (out.no_plane or out.out_of_bounds)
            and not lr["stored_absurd"]
            and math.isfinite(out.lat)
        )
        if comparable:
            lr["dlat_ulp"] = (out.lat - r["lat"]) / gd.ulp32(r["lat"])
            lr["dlng_ulp"] = (out.lng - r["lng"]) / gd.ulp32(r["lng"])
            lr["disagreement_m"] = float(
                haversine_m(out.lng, out.lat, r["lng"], r["lat"])
            )
            lr["on_grid_lat"] = gd.is_on_f32_grid(float(r["lat"]))
            lr["on_grid_lng"] = gd.is_on_f32_grid(float(r["lng"]))
            lr["consistent"] = (
                abs(lr["dlat_ulp"]) <= CONSISTENT_ULP
                and abs(lr["dlng_ulp"]) <= CONSISTENT_ULP
            )
            if r["in_cleaned"]:
                consistency.append(lr["consistent"])
        else:
            # The stored absurd rows should be no-plane hits today too; record
            # the reproduction instead of a meaningless coordinate delta.
            lr["absurd_reproduced"] = bool(lr["stored_absurd"] and out.no_plane)
        label_rows.append(lr)

    rec["n_labels_compared"] = len(consistency)
    rec["n_consistent"] = int(sum(consistency))
    rec["pano_class"] = classify_pano(pd.Series(consistency, dtype=float))
    return rec, label_rows


def write_gz_bytes(path, payload_bytes):
    """gzip with mtime=0 and no filename so rebuilds are byte-identical."""
    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as gz:
            gz.write(payload_bytes)


def write_csv_gz(df, path):
    write_gz_bytes(path, df.to_csv(index=False).encode("utf-8"))


def cmd_build(args):
    raw, cleaned = load_frames(args.data_dir)
    photometa, _ = cache_dirs(args.cache_dir)
    sample_a = pd.read_json(os.path.join(args.cache_dir, "sample-a.json"))
    sample_b = pd.read_json(os.path.join(args.cache_dir, "sample-b.json"))
    fetch_times = load_fetch_times(args.cache_dir)

    by_pano = dict(tuple(raw.groupby("pano_id")))

    # ---- Part A
    pano_records, label_records, payload_lines = [], [], []
    for _, srow in sample_a.sort_values("pano_id").iterrows():
        pid = srow["pano_id"]
        with open(os.path.join(photometa, pid + ".json"), encoding="utf-8") as f:
            resp = json.load(f)
        rec, labels = analyze_pano(pid, resp, by_pano[pid])
        rec.update({"part": "a", "city": srow["city"], "stratum": srow["stratum"]})
        pano_records.append(rec)
        label_records.extend(labels)
        b64 = gd.extract_depth_b64(resp)
        if b64:
            meta = gd.extract_pano_meta(resp)
            payload_lines.append({
                "pano_id": pid, "part": "a", "city": srow["city"],
                "fetched_utc": fetch_times.get(pid),
                "capture_year": meta["capture_year"],
                "capture_month": meta["capture_month"],
                "image_sizes": meta["image_sizes"],
                "depth_b64": b64,
            })

    # ---- Part B
    _, search_dir = cache_dirs(args.cache_dir)
    for _, srow in sample_b.iterrows():
        key = f"{srow['lat']:.6f}_{srow['lng']:.6f}"
        with open(os.path.join(search_dir, key + ".json"), encoding="utf-8") as f:
            result = json.load(f)
        rec = {
            "part": "b", "city": srow["city"], "stratum": "modern",
            "search_lat": srow["lat"], "search_lng": srow["lng"],
        }
        pid = result["pano_id"]
        if pid is None:
            rec.update({"pano_id": None, "status": "not_found"})
            pano_records.append(rec)
            continue
        with open(os.path.join(photometa, pid + ".json"), encoding="utf-8") as f:
            resp = json.load(f)
        meta = gd.extract_pano_meta(resp)
        b64 = gd.extract_depth_b64(resp)
        rec["pano_id"] = pid
        if meta is None:
            rec["status"] = "gone"
            pano_records.append(rec)
            continue
        rec.update({
            "fresh_lat": meta["lat"], "fresh_lng": meta["lng"],
            "fresh_heading_deg": meta["heading_deg"],
            "fresh_pitch_deg": meta["pitch_deg"], "fresh_roll_deg": meta["roll_deg"],
            "fresh_elevation_m": meta["elevation_m"],
            "capture_year": meta["capture_year"], "capture_month": meta["capture_month"],
            "image_sizes": meta["image_sizes"],
            "search_dist_m": float(
                haversine_m(srow["lng"], srow["lat"], meta["lng"], meta["lat"])
            ),
        })
        if b64 is None:
            rec["status"] = "no_depth"
        else:
            payload = gd.decode_depth_payload(b64)
            qc = gd.camera_height_qc(payload)
            rec.update({
                "status": "ok",
                "was_compressed": payload.was_compressed,
                "n_planes": qc.n_planes,
                "ground_d": qc.ground_d,
                "ground_height_m": qc.ground_height,
                "ground_tilt_deg": qc.ground_tilt_deg,
                "ground_pixel_share": qc.ground_pixel_share,
                "height_is_default": qc.is_default,
                "ground_d_exactly_2p5": qc.ground_d == gd.DEFAULT_CAMERA_HEIGHT,
                "band_height_median": qc.band_height_median,
                "band_height_mad": qc.band_height_mad,
            })
            meta2 = meta
            payload_lines.append({
                "pano_id": pid, "part": "b", "city": srow["city"],
                "fetched_utc": fetch_times.get(pid),
                "capture_year": meta2["capture_year"],
                "capture_month": meta2["capture_month"],
                "image_sizes": meta2["image_sizes"],
                "depth_b64": b64,
            })
        pano_records.append(rec)

    # ---- artifacts (deterministic order + gzip)
    pano_cols = [
        "part", "city", "stratum", "pano_id", "status", "pano_class",
        "n_labels_raw", "n_labels_cleaned", "n_labels_compared", "n_consistent",
        "stored_pano_lat", "stored_pano_lng", "fresh_lat", "fresh_lng",
        "pano_shift_m", "heading_shift_deg", "fresh_heading_deg",
        "fresh_pitch_deg", "fresh_roll_deg", "fresh_elevation_m",
        "capture_year", "capture_month", "image_sizes",
        "search_lat", "search_lng", "search_dist_m",
        "was_compressed", "n_planes", "ground_d", "ground_height_m",
        "ground_tilt_deg", "ground_pixel_share", "height_is_default",
        "ground_d_exactly_2p5", "band_height_median", "band_height_mad",
    ]
    panos = pd.DataFrame(pano_records).reindex(columns=pano_cols)
    panos = panos.sort_values(
        ["part", "city", "pano_id"], na_position="last"
    ).reset_index(drop=True)

    label_cols = [
        "city", "pano_id", "label_id", "label_type", "zoom",
        "sv_image_x", "sv_image_y", "in_cleaned", "stored_absurd",
        "stored_lat", "stored_lng", "recomputed_lat", "recomputed_lng",
        "dlat_ulp", "dlng_ulp", "disagreement_m", "consistent",
        "on_grid_lat", "on_grid_lng", "no_plane", "seam_wrap",
        "out_of_bounds", "absurd_reproduced",
    ]
    labels_df = pd.DataFrame(label_records).reindex(columns=label_cols)
    labels_df = labels_df.sort_values(["city", "pano_id", "label_id"]).reset_index(drop=True)

    out = args.out_dir
    write_csv_gz(panos, os.path.join(out, "depth-pilot-panos.csv.gz"))
    write_csv_gz(labels_df, os.path.join(out, "depth-pilot-labels.csv.gz"))

    payload_lines.sort(key=lambda r: (r["part"], r["pano_id"]))
    seen = set()
    unique_lines = []
    for line in payload_lines:
        if line["pano_id"] not in seen:  # a pano can appear in both parts
            seen.add(line["pano_id"])
            unique_lines.append(line)
    write_gz_bytes(
        os.path.join(out, "depth-pilot-payloads.jsonl.gz"),
        "".join(json.dumps(line) + "\n" for line in unique_lines).encode("utf-8"),
    )

    summary = summarize(panos, labels_df)
    with open(os.path.join(out, "depth-pilot-summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote depth-pilot-{{panos,labels,payloads,summary}} to {out}")


def summarize(panos, labels):
    """The headline numbers: everything the report claims, in one place."""
    a = panos[panos["part"] == "a"]
    head = a[a["stratum"].str.startswith("headline")]
    b = panos[panos["part"] == "b"]

    hit = head[head["status"].isin(["ok", "no_depth"])]
    ok = head[head["status"] == "ok"]
    cls = ok["pano_class"].value_counts().to_dict()

    lab = labels[
        labels["in_cleaned"]
        & labels["disagreement_m"].notna()
        & labels["pano_id"].isin(set(ok["pano_id"]))
    ]
    unchanged_ids = set(ok[ok["pano_class"] == "unchanged"]["pano_id"])
    lab_unchanged = lab[lab["pano_id"].isin(unchanged_ids)]

    def ulp_consistent_frac(threshold):
        c = (lab["dlat_ulp"].abs() <= threshold) & (lab["dlng_ulp"].abs() <= threshold)
        return round(float(c.mean()), 4) if len(lab) else None

    heights = panos[
        (panos["status"] == "ok")
        & ~panos["ground_d_exactly_2p5"].fillna(False).astype(bool)
    ]

    return {
        "part_a": {
            "attempted": int(len(head)),
            "resolve_rate": round(float(len(hit) / len(head)), 4) if len(head) else None,
            "resolve_rate_by_city": {
                c: round(float(g["status"].isin(["ok", "no_depth"]).mean()), 3)
                for c, g in head.groupby("city")
            },
            "depth_rate_among_resolved": round(float(len(ok) / len(hit)), 4) if len(hit) else None,
            "pano_class_counts": {k: int(v) for k, v in sorted(cls.items())},
            "labels_compared": int(len(lab)),
            "label_median_disagreement_m": round(float(lab["disagreement_m"].median()), 4)
            if len(lab) else None,
            "label_p90_disagreement_m": round(float(lab["disagreement_m"].quantile(0.9)), 4)
            if len(lab) else None,
            "label_median_disagreement_m_unchanged_panos": round(
                float(lab_unchanged["disagreement_m"].median()), 4
            ) if len(lab_unchanged) else None,
            "consistent_frac_at_1p5_ulp": ulp_consistent_frac(1.5),
            "consistent_frac_at_2_ulp": ulp_consistent_frac(2.0),
            "consistent_frac_at_3_ulp": ulp_consistent_frac(3.0),
            "stored_on_f32_grid_frac": round(float(
                (lab["on_grid_lat"] & lab["on_grid_lng"]).mean()
            ), 4) if len(lab) else None,
            "median_pano_shift_m": round(float(hit["pano_shift_m"].median()), 3)
            if hit["pano_shift_m"].notna().any() else None,
            "absurd_rows_checked": int(labels["stored_absurd"].sum()),
            "absurd_reproduced": int(labels["absurd_reproduced"].fillna(False).sum()),
            # a label whose stored position was sane but whose pixel hits no
            # plane TODAY is payload change the coordinate deltas can't see
            "lost_plane_labels": int(
                (
                    labels["in_cleaned"]
                    & labels["no_plane"].fillna(False)
                    & ~labels["stored_absurd"]
                ).sum()
            ),
        },
        "part_b": {
            "locations": int(len(b)),
            "found_current_pano": int((b["status"] != "not_found").sum()),
            "unique_panos": int(b["pano_id"].nunique()),
            "depth_served": int((b["status"] == "ok").sum()),
            "capture_years": {
                str(int(y)): int(n)
                for y, n in b["capture_year"].value_counts().items()
                if pd.notna(y)
            },
            "max_resolution_16384_frac": round(float(
                b["image_sizes"].dropna().str.contains("16384x8192").mean()
            ), 3) if b["image_sizes"].notna().any() else None,
        },
        "camera_height": {
            "panos_with_depth": int((panos["status"] == "ok").sum()),
            "ground_d_exactly_2p5_frac": round(float(
                panos[panos["status"] == "ok"]["ground_d_exactly_2p5"].mean()
            ), 3),
            "structural_default_frac": round(float(
                panos[panos["status"] == "ok"]["height_is_default"].mean()
            ), 3),
            "median_height_excl_2p5": round(float(heights["ground_height_m"].median()), 3)
            if len(heights) else None,
            "n_heights_excl_2p5": int(len(heights)),
        },
    }


# ---------------------------------------------------------------------------- figures

# One float32 ulp of latitude/longitude at DC (58% of labels), in meters -- the
# storage quantization of the recovered coordinates and hence the finest
# agreement any recompute can show. Across all seven cities the lat ulp is
# 0.21-0.42 m and the lng ulp 0.57-0.80 m; DC's pair is drawn as the guide band.
ULP_BAND_M = (0.42, 0.66)

CLASS_COLORS = {  # validated categorical trio; matches the repo's est colors
    "unchanged": "#2a78d6",
    "mostly_unchanged": "#1baf7a",
    "changed": "#eb6834",
}
CLASS_LABELS = {
    "unchanged": "unchanged",
    "mostly_unchanged": "mostly unchanged",
    "changed": "changed",
}


def cmd_figures(args):
    import make_figures as mf  # rcParams, palette, _title/_save conventions

    panos = pd.read_csv(os.path.join(args.data_dir, "depth-pilot-panos.csv.gz"))
    labels = pd.read_csv(os.path.join(args.data_dir, "depth-pilot-labels.csv.gz"))
    import matplotlib.pyplot as plt

    a = panos[
        (panos["part"] == "a")
        & (panos["status"] == "ok")
        & panos["stratum"].str.startswith("headline")
    ]
    lab = labels[
        labels["in_cleaned"].astype(bool)
        & labels["disagreement_m"].notna()
        & labels["pano_id"].isin(set(a["pano_id"]))
    ].merge(a[["pano_id", "pano_class", "capture_year"]], on="pano_id")

    # ---- fig 7: the cross-check
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    ax = axes[0]
    ax.axvspan(*ULP_BAND_M, color=mf.GRID, alpha=0.6, zorder=0)
    ax.text(0.52, 0.03, "float32\nstorage floor\n(1 ulp)", fontsize=8,
            color=mf.MUTED, ha="center", va="bottom")
    ax.axvline(1.46, color=mf.INK, lw=1.2, ls=(0, (5, 3)))
    ax.text(1.55, 0.05, "est7 median error (1.46 m)", fontsize=8, color=mf.SECONDARY,
            rotation=90, va="bottom")
    # direct labels at distinct heights so the three never collide
    label_at = {"unchanged": 0.92, "mostly_unchanged": 0.52, "changed": 0.60}
    label_dxy = {"unchanged": (-8, 4), "mostly_unchanged": (-10, -2), "changed": (12, -4)}
    label_ha = {"unchanged": "right", "mostly_unchanged": "right", "changed": "left"}
    for cls, color in CLASS_COLORS.items():
        d = np.sort(lab.loc[lab["pano_class"] == cls, "disagreement_m"].to_numpy())
        if len(d) == 0:
            continue
        ax.step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=color, lw=2)
        q = label_at[cls]
        ax.annotate(f"{CLASS_LABELS[cls]} panos", (np.quantile(d, q), q),
                    xytext=label_dxy[cls], textcoords="offset points",
                    fontsize=9, color=color, ha=label_ha[cls])
    ax.set_xscale("log")
    ax.set_xlim(0.03, 30)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recomputed vs stored label position (m, log)")
    ax.set_ylabel("fraction of labels ≤ x")
    ax.set_title(f"agreement by pano class   n={len(lab):,} labels", loc="left")

    ax = axes[1]
    per = (
        lab.groupby(["pano_id", "pano_class"], observed=True)
        .agg(med=("disagreement_m", "median"), year=("capture_year", "first"))
        .reset_index()
        .dropna(subset=["year"])
    )
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-0.28, 0.28, len(per))
    for cls, color in CLASS_COLORS.items():
        m = per["pano_class"] == cls
        ax.scatter(per.loc[m, "year"] + jitter[m.to_numpy()], per.loc[m, "med"],
                   s=14, color=color, alpha=0.75, linewidths=0)
    med_by_year = per.groupby("year")["med"].agg(["median", "size"])
    med_by_year = med_by_year[med_by_year["size"] >= 5]
    ax.plot(med_by_year.index, med_by_year["median"], color=mf.INK, lw=2,
            solid_capstyle="round", zorder=5)
    ax.annotate("yearly median", (med_by_year.index[0], med_by_year["median"].iloc[0]),
                xytext=(-6, 12), textcoords="offset points", fontsize=9, color=mf.INK)
    ax.axhspan(*ULP_BAND_M, color=mf.GRID, alpha=0.6, zorder=0)
    ax.text(2011.0, 0.52, "storage floor", fontsize=8, color=mf.MUTED, va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.08, 12)
    ax.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.set_xlabel("panorama capture year")
    ax.set_ylabel("per-pano median disagreement (m, log)")
    ax.set_title(f"drift grows with payload age   n={len(per)} panos", loc="left")

    mf._title(
        fig,
        "Fresh GSV depth reproduces the 2020 ground truth to ~1 m — and to the storage floor where payloads are unchanged",
        "Every label on 195 surviving 2017–2020 panos, recomputed from today's depth payload with the exact "
        "2020 client algorithm and compared against the stored depth-derived position. 23% of panos are "
        "bit-stable (agreement at the float32 write quantization); the rest drifted slightly under Google "
        "reprocessing — more for older captures — yet the overall label median (0.98 m) stays below the "
        "deployed estimator's own 1.46 m median error.",
        wrap=118,
    )
    fig.subplots_adjust(top=0.72, wspace=0.24)
    mf._save(fig, "fig7-depth-crosscheck.png")

    # ---- fig 8: camera height
    ok = panos[panos["status"] == "ok"].copy()
    ok["pinned"] = ok["ground_d_exactly_2p5"].fillna(False).astype(bool)
    ok = ok.dropna(subset=["capture_year"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    ax = axes[0]
    meas = ok[~ok["pinned"]]
    rng = np.random.default_rng(SEED)
    ax.scatter(meas["capture_year"] + rng.uniform(-0.28, 0.28, len(meas)),
               meas["ground_height_m"], s=14, color="#2a78d6", alpha=0.7, linewidths=0)
    med = meas.groupby("capture_year")["ground_height_m"].agg(["median", "size"])
    med = med[med["size"] >= 5]
    ax.plot(med.index, med["median"], color=mf.INK, lw=2, solid_capstyle="round", zorder=5)
    ax.axhline(2.5, color=mf.BASELINE, lw=1.2, ls=(0, (5, 3)))
    ax.text(2011.2, 2.52, "Google's pinned 2.5 m", fontsize=8, color=mf.MUTED)
    ax.axhline(2.6, color="#eb6834", lw=1.2, ls=(0, (5, 3)))
    ax.text(2011.2, 2.62, "auto-labeler constant 2.6 m", fontsize=8, color="#eb6834")
    ax.set_xlabel("panorama capture year")
    ax.set_ylabel("camera height from ground plane (m)")
    ax.set_ylim(1.5, 3.2)
    ax.set_title(f"measured heights   n={len(meas)} panos", loc="left")

    ax = axes[1]
    share = ok.groupby("capture_year")["pinned"].agg(["mean", "size"])
    share = share[share["size"] >= 5]
    ax.bar(share.index, share["mean"], width=0.72, color=mf.MUTED, edgecolor="none")
    for year, row in share.iterrows():
        ax.text(year, row["mean"] + 0.02, f"{int(row['size'])}", ha="center",
                fontsize=7.5, color=mf.MUTED)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("panorama capture year  (bar label = panos)")
    ax.set_ylabel("fraction of panos with pinned 2.5 m plane")
    ax.set_title("the 2.5 m pin is a vintage artifact", loc="left")

    mf._title(
        fig,
        "GSV camera height is per-panorama: measured plane heights cluster near 2.37 m, not 2.6 m",
        "Ground-plane camera height read directly from each depth payload's plane list (409 panos with "
        "depth: the 2017–2020 sample + the current pano at 200 modern locations). Panos whose ground plane sits at exactly "
        "2.500 m carry a pinned default, not a measurement — 68% of the 2017–2020 payloads but only 27% "
        "of modern ones. The structural two-plane default payload never occurs in this sample.",
        wrap=118,
    )
    fig.subplots_adjust(top=0.72, wspace=0.24)
    mf._save(fig, "fig8-camera-height.png")


# ---------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in [("fetch", cmd_fetch), ("build", cmd_build), ("figures", cmd_figures)]:
        sp = sub.add_parser(name)
        sp.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
        sp.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "depth-pilot-cache"))
        sp.set_defaults(fn=fn)
        if name == "fetch":
            sp.add_argument("--rps", type=float, default=3.0,
                            help="request rate cap against the unofficial endpoint")
        if name == "build":
            sp.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
        if name == "figures":
            sp.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
