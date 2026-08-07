"""Run the #3 Stage 3 opening: the Mapillary projection/metadata census.

Usage (from the repo root):
    python python/run_mapillary_falsification.py            # print the census
    python python/run_mapillary_falsification.py --write    # also write data/falsification-summary.json

Offline and deterministic: the committed data/falsification-* inputs in, zero network.
Later Stage 3 stages (per-sequence camera heights, the two scale-free falsification
diagnostics) will extend this runner and the same summary file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapillary_falsification as mf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=mf.DATA_DIR)
    ap.add_argument("--write", action="store_true",
                    help="write data/falsification-summary.json for the findings tests")
    args = ap.parse_args()
    t0 = time.time()

    def step(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("census over the committed falsification inputs...")
    census = mf.build_census(args.data_dir)

    step("the two scale-free diagnostics, all six runs...")
    diagnostics = mf.build_diagnostics(args.data_dir)

    step("per-sequence camera heights (alternating least squares)...")
    seq_scales = mf.build_sequence_scales(args.data_dir)

    summary = {"census": census, "diagnostics": diagnostics,
               "sequence_scales": seq_scales}

    for run, c in census["mapillary"].items():
        print(f"\n{run}: {c['n_panos']} panos, {c['n_sequences']} sequences, "
              f"{c['n_creators']} creators, {c['capture_dates'][0]}..{c['capture_dates'][1]}")
        print(f"  camera_type {c['camera_type']}; true 2:1 equirect: {c['all_true_equirect']}")
        print(f"  pano heights {c['pano_heights']}")
        for rig, r in sorted(c["rigs"].items(), key=lambda kv: -kv[1]["n_panos"]):
            print(f"  rig {rig:<40} {r['n_panos']:>6} panos {r['n_sequences']:>4} seqs "
                  f"{r['n_site_members']:>6} site members  {r['pano_dims']}")
        d = c["raw_field_degeneracy"]
        print(f"  raw compass exact-zero {d['compass_angle_exact_zero']}/{c['n_panos']}; "
              f"camera_parameters on {d['camera_parameters_present']}")
        s = c["sfm_vs_raw"]
        print(f"  SfM vs raw: pos shift med {s['position_shift_m']['median']} m "
              f"(p90 {s['position_shift_m']['p90']}), |compass delta| med "
              f"{s['abs_compass_delta_deg']['median']} deg (p90 {s['abs_compass_delta_deg']['p90']}), "
              f"altitude delta med {s['altitude_delta_m']['median']} m")
        print("  capture modes: " + ", ".join(
            f"{m}: {v['n_sequences']} seqs / {v['n_panos']} panos / {v['n_site_members']} members"
            for m, v in c["capture_modes"].items()))

    print("\nGSV controls:")
    for run, c in census["gsv_control"].items():
        heights = {h: v["n_panos"] for h, v in c["pano_heights"].items()}
        members = {h: v["n_site_members"] for h, v in c["pano_heights"].items()}
        print(f"  {run:<12} {c['n_panos']:>6} panos {c['capture_dates'][0]}..{c['capture_dates'][1]} "
              f"heights {heights} site members {members}")

    def fmt_slope(s):
        return "      n/a      " if s["slope"] is None else f"{s['slope']:+.4f}±{s['se']:.4f}"

    print(f"\nConventions: cotangent(h=2.6) vs stored ray range, max |delta| "
          f"{diagnostics['conventions']['max_abs_range_m_delta']} m "
          f"(n={diagnostics['conventions']['n']})")
    print("\nScale-free diagnostics (within-site demeaned; height slope vs h/6656):")
    for run, d in diagnostics["runs"].items():
        print(f"  {run} ({d['n_sites']} sites, {d['n_members']} members)")
        for key in mf.MODEL_KEYS:
            v = d["per_model"][key]
            print(f"    {key:<14} rms/range {v['rms_over_range']:.4f}  "
                  f"range {fmt_slope(v['range_slope'])}  height {fmt_slope(v['height_slope'])}")

    print("\nPer-sequence camera heights (D_blend, relative to run gmean):")
    for run, s in seq_scales.items():
        print(f"  {run}: {s['n_sequences_fitted']} seqs fitted, "
              f"{s['n_multi_sequence_sites']}/{s['n_sites']} multi-sequence sites")
        for rig, r in sorted(s["per_rig"].items(), key=lambda kv: -kv[1]["n_members"]):
            print(f"    {rig:<40} {r['n_sequences']:>4} seqs {r['n_members']:>5} members  "
                  f"k_rel med {r['k_rel_median']:.3f} iqr {r['k_rel_iqr']}")
        for lbl, key in [("unscaled", "d_blend_unscaled"),
                         ("per-seq scale", "d_blend_per_sequence_scale")]:
            v = s[key]
            print(f"    D {lbl:<14} rms/range {v['rms_over_range']:.4f}  "
                  f"range {fmt_slope(v['range_slope'])}  height {fmt_slope(v['height_slope'])}")

    if args.write:
        out = args.data_dir / "falsification-summary.json"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nSummary written to {out}")
    step("done")


if __name__ == "__main__":
    main()
