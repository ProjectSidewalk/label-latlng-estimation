"""Run the SidewalkWebpage#5084 production sign-off (reports/2026-09-02-production-signoff.md).

Usage (from the repo root):
    python python/run_signoff.py build --write     # both truth frames, geodesy, frame contract,
                                                   #   the parity fixture -> data/signoff-summary.json
    python python/run_signoff.py fetch             # imagery tiles for the four worked examples
                                                   #   -> data/signoff-tiles.jsonl.gz (network)
    python python/run_signoff.py fixture <path>    # write the cross-implementation fixture JSON
    python python/signoff_figures.py               # figures 29-34

``build`` is offline and deterministic: the era split reloads and refits the 2021 pipeline
(~2.5 minutes), everything else is seconds. ``fetch`` is the one network stage; the bytes it
commits are what the example figures replay.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signoff as so  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SUMMARY = os.path.join(DATA, "signoff-summary.json")
TILES = os.path.join(DATA, "signoff-tiles.jsonl.gz")
CACHE = os.path.join(DATA, "signoff-cache")
EXAMPLE_ZOOM = 3  # 4096x2048 on a 16384-wide pano: enough to read a curb ramp, 32 tiles


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def example_records(examples: pd.DataFrame) -> list[dict]:
    keep = ["role", "label_uid", "label_id", "city", "label_type", "zoom", "pano_id", "pano_width",
            "pano_height", "pano_x", "pano_y", "canvas_x", "canvas_y", "heading", "pitch", "pano_lat",
            "pano_lng", "camera_heading", "capture_date", "depression_deg", "truth_m", "hit_class",
            "A_deployed", "dist_approx3", "err_A", "err_approx3", "stored_dist_m"]
    return [_jsonable({k: r[k] for k in keep if k in r}) for r in examples.to_dict("records")]


def cmd_build(args):
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    shipped = so.load_shipped(DATA)
    step("modern frame (fresh-depth truth, post-2021 human labels)...")
    human, modern = so.modern_frame(shipped, DATA)
    step("era frame (the 2021 regression's own held-out split)...")
    era_scored, era = so.era_frame(shipped, DATA)
    # Per-row cache for the figures script (gitignored, like every other data/*-cache).
    os.makedirs(CACHE, exist_ok=True)
    era_scored.to_pickle(os.path.join(CACHE, "era_scored.pkl.gz"))
    human.to_pickle(os.path.join(CACHE, "modern_scored.pkl.gz"))
    step("geodesy...")
    lats = {r["city"]: r["latitude"] for r in modern["leave_one_city_out"]}
    for r in era["by_city"]:
        lats.setdefault(r["city"], None)
    # era cities' latitudes from their pano origins
    era_lat = {"dc": 38.9072, "seattle": 47.6062, "newberg": 45.3001, "columbus": 39.9612,
               "spgg": 25.6570, "cdmx": 19.4326, "pittsburgh": 40.4406}
    for k, v in list(lats.items()):
        if v is None:
            lats[k] = era_lat[k]
    geodesy = so.geodesy_displacements(lats)
    step("viewport frame contract...")
    frames = so.viewport_frame_contract(shipped)
    step("parity fixture...")
    fixture = so.parity_fixture(shipped)
    step("examples...")
    panos = pd.read_csv(os.path.join(DATA, "modern-truth-panos.csv.gz"), dtype={"pano_id": str})
    examples = so.pick_examples(human, panos)

    summary = _jsonable({
        "meta": {"issue": "SidewalkWebpage#5084", "generated": time.strftime("%Y-%m-%d"),
                 "shipped": shipped, "radii_m": geodesy["radii_m"]},
        "modern_frame": modern,
        "era_frame": era,
        "geodesy": geodesy,
        "viewport_frame_contract": frames,
        "parity_fixture": {"n_cases": len(fixture["cases"]), "tolerance": fixture["tolerance"]},
        "examples": example_records(examples),
    })
    m, e = summary["modern_frame"], summary["era_frame"]
    print()
    print("modern frame, representative human stratum (n=%d):" % m["n_representative"])
    for k in ("A_deployed", "approx3"):
        s = m["representative"][k]
        print(f"  {k:12s} median {s['median_m']:.3f} m  p90 {s['p90_m']:.2f}  signed {s['signed_median_m']:+.2f}")
    h = m["repeated_holdout"]
    print(f"  held-out (200 pano-halves): approx3 {h['approx3_median_m']['mean']:.3f} "
          f"[{h['approx3_median_m']['p5']:.3f}, {h['approx3_median_m']['p95']:.3f}] vs "
          f"deployed {h['A_deployed_median_m']['mean']:.3f}")
    print("era frame, published test split (n=%d):" % e["n_test"])
    for k in ("est7", "approx3", "approx3_eracal", "approx3_eraflat", "blend_type_era"):
        s = e["overall"][k]
        print(f"  {k:16s} median {s['median_m']:.4f} m  p90 {s['p90_m']:.3f}  signed {s['signed_median_m']:+.2f}")
    print(f"  era truth implies {e['implied_height_overall_m']:.3f} m; shipped height {shipped['height_m']:.3f}")
    g = summary["geodesy"]
    print(f"geodesy: sphere vs WGS84 at the {so.MAX_ANSWER_M:.1f} m maximum answer, worst city: "
          f"{g['worst_ellipsoid_vs_production_at_max_answer_m']*100:.1f} cm")
    for f in summary["viewport_frame_contract"]["frames"]:
        print(f"frame {f['frame']:18s} own {f['own_frame_max_error_m']:.1e} m | axis-scaled p90 "
              f"{f['axis_scaled_to_720x480']['p90_m']:.2f} m | width-scaled p90 "
              f"{f['width_scaled_read_as_720x480']['p90_m']:.2f} m")
    for ex in summary["examples"]:
        print(f"example {ex['role']:38s} {ex['label_uid']:20s} truth {ex['truth_m']:.2f} "
              f"reg {ex['A_deployed']:.2f} approx3 {ex['dist_approx3']:.2f}")

    if args.write:
        with open(SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        print(f"wrote {SUMMARY}")
    step("done")


def cmd_fetch(args):
    """Imagery tiles for the worked examples, committed verbatim (the repo's archival rule)."""
    import requests
    from run_depth_validation import TILE_HEADERS, TILE_URL

    with open(SUMMARY, encoding="utf-8") as f:
        examples = json.load(f)["examples"]
    session = requests.Session()
    lines = []
    for ex in examples:
        w = ex["pano_width"] // 2 ** (5 - EXAMPLE_ZOOM)
        h = ex["pano_height"] // 2 ** (5 - EXAMPLE_ZOOM)
        nx, ny = -(-w // 512), -(-h // 512)
        tiles = []
        for y in range(ny):
            for x in range(nx):
                url = TILE_URL.format(pano_id=ex["pano_id"], x=x, y=y, zoom=EXAMPLE_ZOOM)
                for backoff in (0, 2, 8):
                    if backoff:
                        time.sleep(backoff)
                    try:
                        resp = session.get(url, headers=TILE_HEADERS, timeout=30)
                        resp.raise_for_status()
                        break
                    except Exception as err:  # transient
                        last = err
                else:
                    raise RuntimeError(f"tile fetch failed {ex['pano_id']} {x},{y}: {last}")
                tiles.append({"x": x, "y": y, "b64": base64.b64encode(resp.content).decode("ascii")})
                time.sleep(0.15)
        lines.append({"pano_id": ex["pano_id"], "label_uid": ex["label_uid"], "zoom": EXAMPLE_ZOOM,
                      "width": w, "height": h, "tile_width": 512, "tile_height": 512, "tiles": tiles})
        print(f"{ex['pano_id']}: {len(tiles)} tiles at zoom {EXAMPLE_ZOOM} ({w}x{h})", flush=True)
    with gzip.open(TILES, "wt", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    print(f"wrote {TILES}")


def cmd_fixture(args):
    fixture = so.parity_fixture(so.load_shipped(DATA))
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {args.path} ({len(fixture['cases'])} cases)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--write", action="store_true")
    b.set_defaults(fn=cmd_build)
    fe = sub.add_parser("fetch")
    fe.set_defaults(fn=cmd_fetch)
    fx = sub.add_parser("fixture")
    fx.add_argument("path")
    fx.set_defaults(fn=cmd_fixture)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
