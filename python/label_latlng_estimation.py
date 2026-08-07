"""Python port of Project Sidewalk's 2021 label lat/lng estimation analysis.

Ported from scripts/label-latlng-estimation.Rmd (Mikey Saugstad, 2021-01-01), which stays in the
repo unmodified as the frozen historical record. This module is the authoritative implementation
going forward. See https://github.com/ProjectSidewalk/label-latlng-estimation/issues/2.

Fidelity notes:
- The geodesy helpers replicate exactly what the R `geosphere` functions the Rmd used compute,
  which is a MIX of models (verified empirically against geosphere 1.6.8):
    * distHaversine is SPHERICAL (radius 6378137 m) — replicated inline;
    * bearing and destPoint are WGS84 ELLIPSOIDAL geodesic operations — replicated via
      pyproj.Geod, which agrees with geosphere to ~1e-10.
  So the analysis measures distances on a sphere but derives headings and destination points on
  the ellipsoid; any future refit should pick one model deliberately. Note Project Sidewalk's
  front end applies the coefficients with turf's SPHERICAL `destination` — a centimeter-scale
  model mismatch between how the coefficients were fit and how they are used in production.
- All model fits are ordinary least squares on explicit design matrices (numpy lstsq); R's lm()
  solves the same problem, so coefficients agree to floating-point noise. The one exception is
  estimate 6 (lme4::lmer vs statsmodels MixedLM), where different REML optimizers agree only
  approximately.
- Cleaning filters, their order, and the time_created < 2021-01-01 UTC reconstruction cutoff
  mirror scripts/rerun-analysis.R (which documents why the cutoff exists).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6378137.0  # geosphere's default sphere radius
CITIES = ["dc", "seattle", "newberg", "columbus", "spgg", "cdmx", "pittsburgh"]
CUTOFF_UTC = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
MAX_LABELS_PER_PANO = 20
MAX_DIST_FROM_PANO = 50
TRAINING_FRAC = 0.8

EXPECTED_2021_COLUMNS = [
    "label_id", "label_type", "lat", "lng", "panorama_lat", "panorama_lng",
    "canvas_x", "canvas_y", "canvas_width", "canvas_height", "heading", "pitch",
    "zoom", "photographer_heading", "photographer_pitch", "sv_image_x", "sv_image_y",
    "gsv_panorama_id", "street_edge_id", "deleted", "tutorial", "computation_method",
]
EXPECTED_EXTRA_COLUMNS = [
    "pano_width", "pano_height", "time_created", "current_pano_x", "current_pano_y",
]

_BOOL_MAP = {"t": True, "f": False, "true": True, "false": False}


# ---------------------------------------------------------------------------- data loading

def load_city(data_dir: str, city: str) -> pd.DataFrame:
    """Read one city's labels-<city>-latlng.csv.gz, typed to match the R pipeline."""
    path = os.path.join(data_dir, f"labels-{city}-latlng.csv.gz")
    df = pd.read_csv(
        path,
        dtype={
            "label_id": "int64", "label_type": "string", "lat": "float64", "lng": "float64",
            "panorama_lat": "float64", "panorama_lng": "float64",
            "canvas_x": "int64", "canvas_y": "int64",
            "canvas_width": "Int64", "canvas_height": "Int64",
            "heading": "float64", "pitch": "float64", "zoom": "int64",
            "photographer_heading": "float64", "photographer_pitch": "float64",
            "sv_image_x": "int64", "sv_image_y": "int64",
            "gsv_panorama_id": "string", "street_edge_id": "Int64",
            "deleted": "string", "tutorial": "string", "computation_method": "string",
            "pano_width": "Int64", "pano_height": "Int64", "time_created": "string",
            "current_pano_x": "Int64", "current_pano_y": "Int64",
        },
    )
    for col in ("deleted", "tutorial"):
        df[col] = df[col].str.lower().map(_BOOL_MAP).astype("boolean")
    df["time_created"] = pd.to_datetime(df["time_created"], format="ISO8601", utc=True)
    df["city"] = city
    return df


def load_data(data_dir: str) -> pd.DataFrame:
    """All seven cities, concatenated in the Rmd's city order."""
    return pd.concat([load_city(data_dir, c) for c in CITIES], ignore_index=True)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The Rmd's filtering_data chunk plus the reconstruction cutoff.

    Returns (cleaned dataframe with pano_dist, attrition report). The report records the row
    count after each sequential step and, for the flag filters, how many rows each condition
    would drop on its own (useful for quantifying drift against the 2021 numbers).
    """
    report: dict = {"raw": len(df)}
    df = df.rename(columns={
        "panorama_lat": "pano_lat", "panorama_lng": "pano_lng", "gsv_panorama_id": "pano_id",
    })

    # NA time_created is kept: early DC-deployment rows predate the column and are all far
    # older than the cutoff (only 61 DC rows are genuinely post-cutoff).
    df = df[df["time_created"].isna() | (df["time_created"] < CUTOFF_UTC)]
    report["after_time_cutoff"] = len(df)

    report["marginal_drops"] = {
        "deleted": int(df["deleted"].sum()),
        "tutorial": int(df["tutorial"].sum()),
        "bad_canvas_or_latlng": int((~(
            df["lat"].between(-90, 90) & df["lng"].between(-180, 180)
            & (df["canvas_x"] > 0) & (df["canvas_y"] > 0)
        )).sum()),
        "not_depth": int((df["computation_method"] != "depth").sum()),
    }
    df = df[
        df["lat"].between(-90, 90) & df["lng"].between(-180, 180)
        & (df["canvas_x"] > 0) & (df["canvas_y"] > 0)
        & (df["computation_method"] == "depth")
        & ~df["tutorial"].fillna(False).astype(bool)
        & ~df["deleted"].fillna(False).astype(bool)
    ]
    report["after_validity_filters"] = len(df)

    df = df[df.groupby("pano_id")["label_id"].transform("size") < MAX_LABELS_PER_PANO]
    report["after_max_labels_per_pano"] = len(df)

    df = df.assign(pano_dist=haversine_m(df["pano_lng"], df["pano_lat"], df["lng"], df["lat"]))
    df = df[df["pano_dist"] < MAX_DIST_FROM_PANO]
    report["after_max_dist_from_pano"] = len(df)

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------- geodesy
# Exact replicas of the geosphere spherical formulas used by the Rmd. Inputs/outputs in degrees
# and meters; all functions are vectorized.

def haversine_m(lng1, lat1, lng2, lat2) -> np.ndarray:
    """geosphere::distHaversine (note the Rmd passes (lng, lat) point pairs)."""
    p1lat, p1lng, p2lat, p2lng = map(np.radians, (np.asarray(lat1, float), np.asarray(lng1, float),
                                                  np.asarray(lat2, float), np.asarray(lng2, float)))
    a = (np.sin((p2lat - p1lat) / 2) ** 2
         + np.cos(p1lat) * np.cos(p2lat) * np.sin((p2lng - p1lng) / 2) ** 2)
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * EARTH_RADIUS_M


def bearing_deg(lng1, lat1, lng2, lat2) -> np.ndarray:
    """geosphere::bearing — WGS84 geodesic forward azimuth, degrees in (-180, 180].

    NOT spherical: geosphere's bearing() defaults to the WGS84 ellipsoid (a=6378137,
    f=1/298.257223563) even though its distHaversine() and destPoint() are spherical. Verified
    against geosphere 1.6.8 to ~1e-10 degrees.
    """
    from pyproj import Geod
    fwd, _, _ = Geod(ellps="WGS84").inv(np.asarray(lng1, float), np.asarray(lat1, float),
                                        np.asarray(lng2, float), np.asarray(lat2, float))
    return np.asarray(fwd)


def dest_point(lng, lat, brng_deg, dist_m) -> tuple[np.ndarray, np.ndarray]:
    """geosphere::destPoint — WGS84 geodesic destination point. Returns (lng, lat) in degrees.

    NOT spherical: like bearing(), geosphere's destPoint() defaults to the WGS84 ellipsoid
    (a=6378137, f=1/298.257223563). Verified against geosphere 1.6.8. (turf's `destination`,
    which production uses to apply the coefficients, IS spherical — see module docstring.)
    """
    from pyproj import Geod
    n = np.broadcast(np.asarray(lng, float), np.asarray(brng_deg, float)).shape
    lng2, lat2, _ = Geod(ellps="WGS84").fwd(
        np.broadcast_to(np.asarray(lng, float), n).copy(),
        np.broadcast_to(np.asarray(lat, float), n).copy(),
        np.broadcast_to(np.asarray(brng_deg, float), n).copy(),
        np.broadcast_to(np.asarray(dist_m, float), n).copy())
    return np.asarray(lng2), np.asarray(lat2)


def add_heading_diff(df: pd.DataFrame) -> pd.DataFrame:
    """label_heading (bearing pano->label, [0, 360)) and heading_diff wrapped to (-180, 180]."""
    label_heading = bearing_deg(df["pano_lng"], df["pano_lat"], df["lng"], df["lat"]) % 360
    raw = label_heading - df["heading"].to_numpy()
    heading_diff = np.where(raw > 180, raw - 360, np.where(raw < -180, raw + 360, raw))
    return df.assign(label_heading=label_heading, heading_diff=heading_diff)


# ---------------------------------------------------------------------------- model fitting

def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _design(df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    """Design matrix with intercept. The pseudo-terms zoom2/zoom3 are R-style treatment dummies
    (baseline zoom 1), matching lm()'s coding of the Rmd's zoom factor."""
    cols = [np.ones(len(df))]
    for t in terms:
        if t in ("zoom2", "zoom3"):
            cols.append((df["zoom"] == int(t[-1])).to_numpy(float))
        else:
            cols.append(df[t].to_numpy(float))
    return np.column_stack(cols)


def _named(coefs: np.ndarray, terms: list[str]) -> dict:
    return {"(Intercept)": float(coefs[0]), **{t: float(c) for t, c in zip(terms, coefs[1:])}}


def fit_models(train: pd.DataFrame, include_est6: bool = True) -> dict:
    """Fit estimators 2-7 on a training set that already has heading_diff/pano_dist.

    include_est6=False skips the slow MixedLM fit (est6 becomes unavailable); callers that
    only need the closed-form estimators (e.g. the #5 runner) use this."""
    models: dict = {"est1": {"dist": 10.0, "heading_diff": 0.0}}
    models["est2"] = {"median_dist": float(train["pano_dist"].median())}
    models["est3"] = {"median_dist_by_label_type":
                      train.groupby("label_type")["pano_dist"].median().astype(float).to_dict()}

    t4 = ["canvas_y", "sv_image_y"]
    B = _ols(_design(train, t4), train[["heading_diff", "pano_dist"]].to_numpy(float))
    models["est4"] = {"coefficients": {"heading_diff": _named(B[:, 0], t4),
                                       "pano_dist": _named(B[:, 1], t4)}}

    t5d = ["sv_image_y", "canvas_y", "zoom2", "zoom3"]
    t5h = ["canvas_x", "zoom2", "zoom3"]
    models["est5"] = {
        "dist": _named(_ols(_design(train, t5d), train["pano_dist"].to_numpy(float)), t5d),
        "heading": _named(_ols(_design(train, t5h), train["heading_diff"].to_numpy(float)), t5h),
    }

    models["est6"] = _fit_est6(train) if include_est6 else {"available": False,
                                                            "error": "skipped by caller"}

    t7d, t7h = ["sv_image_y", "canvas_y"], ["canvas_x"]
    dist7, head7 = [], []
    for z in (1, 2, 3):
        sub = train[train["zoom"] == z]
        dist7.append(_named(_ols(_design(sub, t7d), sub["pano_dist"].to_numpy(float)), t7d))
        head7.append(_named(_ols(_design(sub, t7h), sub["heading_diff"].to_numpy(float)), t7h))
    models["est7"] = {"dist": dist7, "heading": head7}
    return models


def _fit_est6(train: pd.DataFrame) -> dict:
    """lme4::lmer analogue via statsmodels MixedLM (REML). Optimizers differ from lme4, so this
    is comparable to the R baseline only loosely; it lost the 2021 comparison anyway."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"available": False, "error": "statsmodels not installed"}
    try:
        m_h = smf.mixedlm("heading_diff ~ canvas_x", train, groups=train["zoom"]).fit(reml=True)
        m_d = smf.mixedlm("pano_dist ~ canvas_y + sv_image_y", train, groups=train["zoom"]).fit(reml=True)
        return {
            "available": True,
            "heading": {"fixef": {"(Intercept)": float(m_h.params["Intercept"]),
                                  "canvas_x": float(m_h.params["canvas_x"])},
                        "ranef_zoom": {str(k): float(v.iloc[0]) for k, v in m_h.random_effects.items()}},
            "dist": {"fixef": {"(Intercept)": float(m_d.params["Intercept"]),
                               "canvas_y": float(m_d.params["canvas_y"]),
                               "sv_image_y": float(m_d.params["sv_image_y"])},
                     "ranef_zoom": {str(k): float(v.iloc[0]) for k, v in m_d.random_effects.items()}},
        }
    except Exception as e:  # noqa: BLE001 - est6 is optional; record and move on
        return {"available": False, "error": str(e)}


# ---------------------------------------------------------------------------- prediction / eval

def predict_dist_heading(models: dict, df: pd.DataFrame, est: str) -> tuple[np.ndarray, np.ndarray]:
    """(distance, heading_diff) predictions for one estimator on df."""
    n = len(df)
    if est == "est1":
        return np.full(n, 10.0), np.zeros(n)
    if est == "est2":
        return np.full(n, models["est2"]["median_dist"]), np.zeros(n)
    if est == "est3":
        med = models["est3"]["median_dist_by_label_type"]
        return df["label_type"].map(med).to_numpy(float), np.zeros(n)
    if est == "est4":
        c = models["est4"]["coefficients"]
        X = _design(df, ["canvas_y", "sv_image_y"])
        d = X @ np.array([c["pano_dist"]["(Intercept)"], c["pano_dist"]["canvas_y"],
                          c["pano_dist"]["sv_image_y"]])
        h = X @ np.array([c["heading_diff"]["(Intercept)"], c["heading_diff"]["canvas_y"],
                          c["heading_diff"]["sv_image_y"]])
        return np.maximum(0, d), h
    if est == "est5":
        cd, ch = models["est5"]["dist"], models["est5"]["heading"]
        d = _design(df, ["sv_image_y", "canvas_y", "zoom2", "zoom3"]) @ np.array(
            [cd["(Intercept)"], cd["sv_image_y"], cd["canvas_y"], cd["zoom2"], cd["zoom3"]])
        h = _design(df, ["canvas_x", "zoom2", "zoom3"]) @ np.array(
            [ch["(Intercept)"], ch["canvas_x"], ch["zoom2"], ch["zoom3"]])
        return np.maximum(0, d), h
    if est == "est6":
        m = models["est6"]
        if not m.get("available"):
            raise ValueError("est6 model unavailable")
        fd, fh = m["dist"]["fixef"], m["heading"]["fixef"]
        rd = df["zoom"].astype(str).map(m["dist"]["ranef_zoom"]).to_numpy(float)
        rh = df["zoom"].astype(str).map(m["heading"]["ranef_zoom"]).to_numpy(float)
        d = (fd["(Intercept)"] + fd["canvas_y"] * df["canvas_y"].to_numpy(float)
             + fd["sv_image_y"] * df["sv_image_y"].to_numpy(float) + rd)
        h = fh["(Intercept)"] + fh["canvas_x"] * df["canvas_x"].to_numpy(float) + rh
        return np.maximum(0, d), h
    if est == "est7":
        d = np.empty(n); h = np.empty(n)
        zoom = df["zoom"].to_numpy()
        for z in (1, 2, 3):
            i = zoom == z
            cd = models["est7"]["dist"][z - 1]; ch = models["est7"]["heading"][z - 1]
            d[i] = (cd["(Intercept)"] + cd["sv_image_y"] * df["sv_image_y"].to_numpy(float)[i]
                    + cd["canvas_y"] * df["canvas_y"].to_numpy(float)[i])
            h[i] = ch["(Intercept)"] + ch["canvas_x"] * df["canvas_x"].to_numpy(float)[i]
        return np.maximum(0, d), h
    raise ValueError(f"unknown estimator {est}")


def latlng_error_m(df: pd.DataFrame, dist: np.ndarray, heading_diff: np.ndarray,
                   crude: bool) -> np.ndarray:
    """Meters between the true label position and the estimate.

    crude=True uses the Rmd's flat-earth formula (estimates 1-3); crude=False uses the spherical
    destination point (estimates 4-7), exactly as in the Rmd.
    """
    if crude:
        heading_rad = np.radians(df["heading"].to_numpy(float))
        lat_est = df["pano_lat"].to_numpy(float) + dist * np.cos(heading_rad) / 111111
        lng_est = (df["pano_lng"].to_numpy(float)
                   + dist * np.sin(heading_rad)
                   / (111111 * np.cos(np.radians(df["pano_lat"].to_numpy(float)))))
    else:
        lng_est, lat_est = dest_point(df["pano_lng"], df["pano_lat"],
                                      df["heading"].to_numpy(float) + heading_diff, dist)
    return haversine_m(df["lng"], df["lat"], lng_est, lat_est)


def evaluate(models: dict, test: pd.DataFrame) -> pd.DataFrame:
    """Per-label error columns for every available estimator, mirroring the Rmd's err columns."""
    out = pd.DataFrame({"label_id": test["label_id"].to_numpy(), "city": test["city"].to_numpy()})
    for est in ("est1", "est2", "est3", "est4", "est5", "est6", "est7"):
        if est == "est6" and not models["est6"].get("available"):
            continue
        d, h = predict_dist_heading(models, test, est)
        out[f"error_{est}"] = latlng_error_m(test, d, h, crude=est in ("est1", "est2", "est3"))
        out[f"dist_error_{est}"] = np.abs(test["pano_dist"].to_numpy(float) - d)
        out[f"heading_error_{est}"] = np.abs(test["heading_diff"].to_numpy(float) - h)
    return out


def error_stats(err: pd.DataFrame) -> dict:
    ests = [c for c in err.columns if c.startswith("error_est")]
    summary = [{
        "estimate": c, "mean": float(err[c].mean()), "median": float(err[c].median()),
        "min": float(err[c].min()), "max": float(err[c].max()), "sd": float(err[c].std(ddof=1)),
    } for c in ests]
    summary.sort(key=lambda r: r["median"])
    return {
        "summary": summary,
        "heading_error_medians": {c.replace("error", "heading_error"):
                                  float(err[c.replace("error", "heading_error")].median())
                                  for c in ests},
        "dist_error_medians": {c.replace("error", "dist_error"):
                               float(err[c.replace("error", "dist_error")].median())
                               for c in ests},
    }


# ---------------------------------------------------------------------------- orchestration

def split_from_fixtures(cleaned: pd.DataFrame, fixtures_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition using the (label_id, city) train/test lists exported by scripts/rerun-analysis.R,
    so R and Python fit on identical rows."""
    key = ["label_id", "city"]
    train_ids = pd.read_csv(os.path.join(fixtures_dir, "split_train.csv.gz"))
    test_ids = pd.read_csv(os.path.join(fixtures_dir, "split_test.csv.gz"))
    train = cleaned.merge(train_ids, on=key, how="inner")
    test = cleaned.merge(test_ids, on=key, how="inner")
    if len(train) != len(train_ids) or len(test) != len(test_ids):
        raise ValueError(
            f"split fixture mismatch: cleaned data matched {len(train)}/{len(train_ids)} train "
            f"and {len(test)}/{len(test_ids)} test rows — data and fixtures are out of sync")
    return train, test


def random_split(cleaned: pd.DataFrame, seed: int = 666) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split with numpy's RNG. NOT the same stream as R's set.seed(666) sample_frac — use
    split_from_fixtures for cross-language comparisons."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cleaned))
    n_train = round(len(cleaned) * TRAINING_FRAC)
    return cleaned.iloc[idx[:n_train]], cleaned.iloc[idx[n_train:]]


def run_analysis(data_dir: str, fixtures_dir: str | None = None,
                 data: pd.DataFrame | None = None) -> dict:
    """Full pipeline: load, clean, split, fit, evaluate. Returns a dict shaped like the R
    baseline fixture (tests/fixtures/r-baseline/baseline.json) for direct comparison.
    Pass `data` (a load_data() result) to skip re-reading the CSVs."""
    if data is None:
        data = load_data(data_dir)
    cleaned, report = clean_data(data)
    cleaned = add_heading_diff(cleaned)
    if fixtures_dir:
        train, test = split_from_fixtures(cleaned, fixtures_dir)
    else:
        train, test = random_split(cleaned)
    models = fit_models(train)
    err = evaluate(models, test)

    full_models = fit_models(cleaned)
    return {
        "meta": {
            "rows_raw": report["raw"],
            "attrition": report,
            "rows_after_cleaning": len(cleaned),
            "rows_train": len(train),
            "rows_test": len(test),
            "split": "r-fixture" if fixtures_dir else "numpy-seed-666",
        },
        **{k: models[k] for k in ("est1", "est2", "est3", "est4", "est5", "est6", "est7")},
        "est7_full": full_models["est7"],
        "error_stats": error_stats(err),
    }
