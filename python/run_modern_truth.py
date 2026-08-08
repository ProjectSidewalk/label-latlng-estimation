"""Issue #3 modern-truth validation: post-2021 labels vs fresh GSV depth.

Subcommands, run from the repo root:

    python python/run_modern_truth.py fetch     # network: select panos, fill the cache
    python python/run_modern_truth.py build     # offline: cache -> committed artifacts
    python python/run_modern_truth.py figures   # offline: summary -> figures/fig20-23

``fetch`` consumes the (uncommitted) all-city extraction produced by
scripts/extraction/extract-modern-labels.sh, selects the stratified fetch plan
(modern_truth.select_panos, seed 666), and walks it stratum by stratum until each
stratum's success budget is met — a success being a pano that still resolves AND serves
a depth payload. Everything is cached under data/modern-truth-cache/ (gitignored) and
the walk is idempotent: re-running skips cached panos at full speed.

``build`` replays offline from the cache into the committed artifacts:

    data/modern-truth-payloads.jsonl.gz   verbatim base64 payloads, one line per pano
    data/modern-truth-panos.csv.gz        per-pano meta + camera-height QC + fetch status
    data/modern-truth-labels.csv.gz       per-label truth, gates, and model predictions
    data/modern-truth-summary.json        the findings the report and tests consume
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gsv_depth as gd  # noqa: E402
import modern_truth as mt  # noqa: E402
from run_depth_pilot import Throttle, fetch_photometa_cached  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(ROOT, "data", "modern-truth-cache")


def default_extract_dir() -> str:
    import glob

    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "data", "modern-extraction",
                                                    "modern-labels-extraction-*"))
                  if os.path.isdir(d))
    if not dirs:
        raise SystemExit("no extraction under data/modern-extraction/; run "
                         "scripts/extraction/extract-modern-labels.sh first")
    return dirs[-1]


# ---------------------------------------------------------------------------- fetch

def stratum_budget(stratum: str) -> int | None:
    """Success budget per stratum; type strata are label-count budgets handled inline."""
    if stratum == "representative":
        return mt.REPRESENTATIVE_PANOS
    if stratum == "near_horizon":
        return mt.NEAR_HORIZON_PANOS
    if stratum == "ai":
        return mt.AI_PANOS
    return None  # type:<T>


def cmd_fetch(args):
    import requests

    frame, census = mt.frame_census(mt.add_depression(mt.load_extraction(args.extract_dir)))
    plan = mt.select_panos(frame)
    tab = mt.pano_table(frame).set_index("pano_id")
    os.makedirs(args.cache_dir, exist_ok=True)
    plan.to_csv(os.path.join(args.cache_dir, "fetch-plan.csv"), index=False)
    print(f"frame: {census['kept_rows']} rows / {census['kept_panos']} panos; "
          f"plan: {len(plan)} candidates in {plan['stratum'].nunique()} strata")

    session = requests.Session()
    throttle = Throttle(args.rps)
    type_cover: dict[str, int] = {}
    ok_total = 0
    fetched = 0
    for stratum in plan["stratum"].drop_duplicates():
        cand = plan.loc[plan["stratum"] == stratum, "pano_id"]
        budget = stratum_budget(stratum)
        ok_here = 0
        for pano_id in cand:
            if ok_total >= mt.TARGET_PANOS:
                break
            if budget is not None and ok_here >= budget:
                break
            if budget is None:  # type:<T> — a label-count budget over ALL successes
                t = stratum[len("type:"):]
                if type_cover.get(t, 0) >= mt.TYPE_LABEL_QUOTA:
                    break
            try:
                resp, fresh = fetch_photometa_cached(pano_id, args.cache_dir,
                                                     throttle, session)
            except RuntimeError as e:
                print(f"  fetch error, skipping: {e}")
                continue
            fetched += fresh
            try:
                # streetlevel's parser dies on the occasional malformed payload;
                # that pano is simply not a success, and build classifies it precisely
                ok = (gd.extract_pano_meta(resp) is not None
                      and gd.extract_depth_b64(resp) is not None)
            except Exception:
                ok = False
            if ok:
                ok_total += 1
                ok_here += 1
                row = tab.loc[pano_id]
                for col in tab.columns:
                    if col.startswith("n_") and col != "n_labels" and row[col]:
                        type_cover[col[2:]] = type_cover.get(col[2:], 0) + int(row[col])
            if fetched and fetched % 50 == 0:
                print(f"  {ok_total} ok / {fetched} network fetches (in {stratum})")
        print(f"{stratum}: {ok_here} ok"
              + (f" / budget {budget}" if budget is not None else
                 f" (type cover {type_cover.get(stratum[len('type:'):], 0)}"
                 f"/{mt.TYPE_LABEL_QUOTA})"))
        if ok_total >= mt.TARGET_PANOS:
            break
    print(f"fetch done: {ok_total} panos with depth, {fetched} network fetches")


# ---------------------------------------------------------------------------- build

def analyze_modern_pano(pano_id, resp, labels, stratum):
    """One attempted pano -> (pano_record, label_rows, control_rows, b64 | None).

    ``labels`` is the frame-gated slice for this pano. Statuses: gone (id no longer
    resolves), parse_error (streetlevel's parser dies on a malformed payload),
    no_depth, decode_error, ok.
    """
    import depth_validation as dv

    rec = {"pano_id": pano_id, "stratum": stratum, "n_labels": int(len(labels))}
    try:
        meta = gd.extract_pano_meta(resp)
    except Exception:
        rec["status"] = "parse_error"
        return rec, [], [], None
    if meta is None:
        rec["status"] = "gone"
        return rec, [], [], None
    rec.update({
        "fresh_lat": meta["lat"], "fresh_lng": meta["lng"],
        "fresh_heading_deg": meta["heading_deg"], "fresh_pitch_deg": meta["pitch_deg"],
        "fresh_roll_deg": meta["roll_deg"], "capture_year": meta["capture_year"],
        "capture_month": meta["capture_month"], "image_sizes": meta["image_sizes"],
    })
    if len(labels):
        from label_latlng_estimation import haversine_m

        rec["pano_shift_m"] = float(haversine_m(
            labels["pano_lng"].iloc[0], labels["pano_lat"].iloc[0],
            meta["lng"], meta["lat"]))
    b64 = gd.extract_depth_b64(resp)
    if b64 is None:
        rec["status"] = "no_depth"
        return rec, [], [], None
    try:
        payload = gd.decode_depth_payload(b64)
    except Exception:
        rec["status"] = "decode_error"
        return rec, [], [], None
    qc = gd.camera_height_qc(payload)
    rec.update({
        "status": "ok",
        "n_planes": qc.n_planes,
        "ground_d": qc.ground_d,
        "ground_height_m": qc.ground_height,
        "ground_tilt_deg": qc.ground_tilt_deg,
        "ground_pixel_share": qc.ground_pixel_share,
        "ground_d_exactly_2p5": qc.ground_d == gd.DEFAULT_CAMERA_HEIGHT,
        "band_height_median": qc.band_height_median,
        "band_height_mad": qc.band_height_mad,
    })

    geom = dv.payload_geometry(payload)
    cam_h = qc.ground_height
    label_rows, control_rows = [], []
    for r in labels.itertuples():
        for control in dv.FRAME_CONTROLS:
            hit = mt.classify_modern_label(payload, r.pano_x, r.pano_y, r.pano_width,
                                           r.pano_height, cam_h, geom, control)
            if control == "identity":
                label_rows.append({
                    "label_uid": r.label_uid,  # (city, label_id); label_id alone collides
                    "hit_class": hit.hit_class,
                    "plane_idx": hit.plane_idx,
                    "truth_m": hit.horizontal_m,
                    "truth_range_m": hit.range_m,
                    "height_above_ground_m": hit.height_above_ground_m,
                    "flat_earth_m": hit.flat_earth_m,
                    "neighbourhood_range_ratio": hit.neighbourhood_range_ratio,
                })
            control_rows.append({
                "label_uid": r.label_uid,
                "control": control,
                "hit_class": hit.hit_class,
                "truth_m": hit.horizontal_m,
            })
    return rec, label_rows, control_rows, b64


def control_sweep(control_rows: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Per frame control: blend-D error against that control's truth.

    The prediction never changes — only which raster cell "truth" is read from — so a
    wrong frame must lose on error and on the share of rays that even land on ground.
    Gated here on hit class and the truth cap only: the neighbourhood-ratio gate is a
    property of the identity read and re-deriving it per control would change what the
    controls are being compared on."""
    merged = control_rows.merge(
        labels[["label_uid", "D_blend"]], on="label_uid", how="inner")
    out = {}
    for control, sub in merged.groupby("control"):
        gated = sub[sub["hit_class"].isin(["ground", "terrain"])
                    & np.isfinite(sub["truth_m"]) & (sub["truth_m"] < mt.TRUTH_MAX_M)]
        err = (gated["D_blend"] - gated["truth_m"]).abs()
        out[str(control)] = {
            "n_rows": int(len(sub)),
            "ground_or_terrain_share": float(
                sub["hit_class"].isin(["ground", "terrain"]).mean()),
            "n_gated": int(len(gated)),
            "D_blend_median_abs_m": float(err.median()) if len(gated) else None,
        }
    return out


def cmd_build(args):
    from run_depth_pilot import write_csv_gz, write_gz_bytes

    frame, census = mt.frame_census(mt.add_depression(mt.load_extraction(args.extract_dir)))
    plan = mt.select_panos(frame)
    photometa_dir = os.path.join(args.cache_dir, "photometa")
    blend_params = mt.load_blend_params(os.path.join(ROOT, "data"))

    pano_records, all_label_rows, all_control_rows, payload_lines = [], [], [], []
    planned = frame[frame["pano_id"].isin(set(plan["pano_id"]))]
    by_pano = dict(iter(planned.groupby("pano_id")))
    for r in plan.itertuples():
        path = os.path.join(photometa_dir, r.pano_id + ".json")
        if not os.path.exists(path):
            continue  # never attempted (fetch stopped before it); not an attempt record
        with open(path, encoding="utf-8") as f:
            resp = json.load(f)
        labels = by_pano.get(r.pano_id)
        rec, label_rows, control_rows, b64 = analyze_modern_pano(
            r.pano_id, resp, labels, r.stratum)
        pano_records.append(rec)
        all_label_rows.extend(label_rows)
        all_control_rows.extend(control_rows)
        if b64 is not None:
            payload_lines.append({"pano_id": r.pano_id, "b64": b64})

    panos = pd.DataFrame(pano_records)
    truth = pd.DataFrame(all_label_rows)
    controls_raw = pd.DataFrame(all_control_rows)

    if truth.empty:
        raise SystemExit("no pano in the cache built successfully; run fetch first")
    stratum_of = plan.set_index("pano_id")["stratum"]
    ok_ids = {p["pano_id"] for p in payload_lines}
    labels = frame[frame["pano_id"].isin(ok_ids)].copy()
    labels["stratum"] = labels["pano_id"].map(stratum_of)
    # label_uid, never label_id: the latter is a per-schema serial and joining on it
    # cross-joins same-numbered labels between cities, pairing each with the other's truth
    labels = labels.merge(truth, on="label_uid", how="inner", validate="one_to_one")
    labels = mt.model_predictions(labels, blend_params)
    labels = mt.guard_frame(labels)  # needs time_created as datetime (era split)
    labels, gate_census = mt.truth_gates(labels)
    labels["time_created"] = labels["time_created"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    labels["capture_date"] = labels["capture_date"].dt.strftime("%Y-%m-%d")

    summary = mt.build_summary(census, panos, labels, blend_params,
                               control_sweep(controls_raw, labels), gate_census)

    out_dir = os.path.join(ROOT, "data")
    payload_lines.sort(key=lambda d: d["pano_id"])
    write_gz_bytes(os.path.join(out_dir, "modern-truth-payloads.jsonl.gz"),
                   ("\n".join(json.dumps(d) for d in payload_lines) + "\n").encode())
    write_csv_gz(panos.sort_values("pano_id"),
                 os.path.join(out_dir, "modern-truth-panos.csv.gz"))
    write_csv_gz(labels.sort_values("label_uid"),  # label_id ties across cities
                 os.path.join(out_dir, "modern-truth-labels.csv.gz"))
    if args.write:
        out = os.path.join(out_dir, "modern-truth-summary.json")
        # newline="\n" or a Windows rerun writes CRLF and stops matching a fresh checkout
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"wrote {out}")
    else:
        print(json.dumps(summary["matrix"], indent=2))


def cmd_figures(args):
    import modern_truth_figures

    modern_truth_figures.main()


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="select panos and fill the cache (network)")
    p_fetch.add_argument("--extract-dir", default=None)
    p_fetch.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p_fetch.add_argument("--rps", type=float, default=3.0)

    p_build = sub.add_parser("build", help="cache -> committed artifacts (offline)")
    p_build.add_argument("--extract-dir", default=None)
    p_build.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p_build.add_argument("--write", action="store_true",
                         help="write data/modern-truth-summary.json")

    sub.add_parser("figures", help="summary -> figures/fig20-23 (offline)")

    args = ap.parse_args()
    if getattr(args, "extract_dir", None) is None and args.cmd in ("fetch", "build"):
        args.extract_dir = default_extract_dir()
    if args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "figures":
        cmd_figures(args)


if __name__ == "__main__":
    main()
