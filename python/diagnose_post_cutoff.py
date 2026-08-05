"""Diagnostic: are post-2021 'depth'-stamped rows genuine depth data or estimator echoes?

Evolution 93 (applied to the city databases mid-2021) stamped computation_method='depth' on
every row that had lat/lng at that moment. Rows created after the 2021-01-01 extraction date
therefore carry the 'depth' stamp even though the depth API had already been shut down for new
panos — their lat/lng may instead have been produced by the then-deployed estimator (the very
coefficients this analysis published), which would make refitting on them circular.

The test: for each row, project the published est7 estimate from (sv_image_y, canvas_y,
canvas_x, heading) with turf-style spherical destination (what production ran) and measure the
distance to the stored lat/lng. Rows whose stored position IS that estimate sit at ~0 m;
genuine depth rows scatter at meters. Compares pre- vs post-cutoff distributions per city.

Usage: python python/diagnose_post_cutoff.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_latlng_estimation import (  # noqa: E402
    CITIES, CUTOFF_UTC, EARTH_RADIUS_M, haversine_m, load_city,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PUBLISHED_DIST = [(18.6051843, 0.0138947, 0.0011023),
                  (20.8794248, 0.0184087, 0.0022135),
                  (25.2472682, 0.0264216, 0.0011071)]
PUBLISHED_HEAD = [(-51.2401711, 0.1443374), (-27.5267447, 0.0784357), (-13.5675945, 0.0396061)]


def spherical_dest(lng, lat, brng_deg, dist_m):
    """turf.destination — spherical, like the production front end (NOT geosphere's destPoint)."""
    lat1, lng1, b = map(np.radians, (np.asarray(lat, float), np.asarray(lng, float),
                                     np.asarray(brng_deg, float)))
    d = np.asarray(dist_m, float) / EARTH_RADIUS_M
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(b))
    lng2 = lng1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lng2), np.degrees(lat2)


def residual_to_published_estimate(df) -> np.ndarray:
    zoom = df["zoom"].to_numpy()
    d = np.empty(len(df)); h = np.empty(len(df))
    for z in (1, 2, 3):
        i = zoom == z
        b0, b1, b2 = PUBLISHED_DIST[z - 1]
        c0, c1 = PUBLISHED_HEAD[z - 1]
        d[i] = b0 + b1 * df["sv_image_y"].to_numpy(float)[i] + b2 * df["canvas_y"].to_numpy(float)[i]
        h[i] = c0 + c1 * df["canvas_x"].to_numpy(float)[i]
    lng_e, lat_e = spherical_dest(df["panorama_lng"], df["panorama_lat"],
                                  df["heading"].to_numpy(float) + h, np.maximum(0, d))
    return haversine_m(df["lng"], df["lat"], lng_e, lat_e)


def main() -> None:
    print(f"{'city':<12}{'era':<22}{'rows':>8}{'median resid (m)':>18}{'% < 0.05 m':>12}{'NaN':>7}")
    for city in CITIES:
        df = load_city(os.path.join(ROOT, "data"), city)
        eras = {
            "pre-cutoff": df[df["time_created"].isna() | (df["time_created"] < CUTOFF_UTC)],
            "post-cutoff": df[df["time_created"].notna() & (df["time_created"] >= CUTOFF_UTC)],
        }
        for era, sub in eras.items():
            if not len(sub):
                continue
            r = residual_to_published_estimate(sub)
            nan = int(np.isnan(r).sum())  # rows with NaN heading etc.; dropped by the pipelines
            ok = r[~np.isnan(r)]
            print(f"{city:<12}{era:<22}{len(sub):>8}{float(np.median(ok)):>18.3f}"
                  f"{100 * float(np.mean(ok < 0.05)):>11.1f}%{nan:>7}")


if __name__ == "__main__":
    main()
