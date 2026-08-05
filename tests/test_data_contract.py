"""Contract tests for the committed reconstructed dataset (data/labels-*-latlng.csv.gz).

These pin the schema and invariants the analysis pipelines (Rmd, scripts/rerun-analysis.R,
python/) rely on, and keep data/MANIFEST.md honest about row counts. If the data is ever
re-extracted, failures here mean the manifest and R baseline fixtures must be regenerated too.
"""

import os
import re

import pytest

from label_latlng_estimation import (
    CITIES,
    EXPECTED_2021_COLUMNS,
    EXPECTED_EXTRA_COLUMNS,
    load_city,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module", params=CITIES)
def city_df(request):
    return request.param, load_city(DATA_DIR, request.param)


def test_all_seven_files_exist():
    for city in CITIES:
        assert os.path.exists(os.path.join(DATA_DIR, f"labels-{city}-latlng.csv.gz")), city


def test_column_names_and_order(city_df):
    city, df = city_df
    cols = [c for c in df.columns if c != "city"]
    assert cols[:22] == EXPECTED_2021_COLUMNS, f"{city}: 2021 column set/order changed"
    assert cols[22:] == EXPECTED_EXTRA_COLUMNS, f"{city}: extras column set/order changed"


def test_row_counts_match_r_baseline(city_df, baseline):
    city, df = city_df
    assert len(df) == baseline["meta"]["raw_rows_per_city"][city]


def test_row_counts_match_manifest(baseline):
    manifest = open(os.path.join(DATA_DIR, "MANIFEST.md"), encoding="utf-8").read()
    for city, n in baseline["meta"]["raw_rows_per_city"].items():
        m = re.search(rf"\|\s*{city}\s*\|[^|]*\|\s*([\d,]+)\s*\|", manifest)
        assert m, f"MANIFEST.md has no row-count table entry for {city}"
        assert int(m.group(1).replace(",", "")) == n, f"MANIFEST.md count stale for {city}"


def test_all_rows_are_depth(city_df):
    city, df = city_df
    assert (df["computation_method"] == "depth").all()


def test_zoom_values(city_df):
    city, df = city_df
    assert set(df["zoom"].unique()) <= {1, 2, 3}


def test_canvas_dimensions_are_constant(city_df):
    city, df = city_df
    assert (df["canvas_width"] == 720).all() and (df["canvas_height"] == 480).all()


def test_key_predictors_non_null(city_df):
    city, df = city_df
    for col in ("label_id", "panorama_lat", "panorama_lng", "canvas_x", "canvas_y",
                "heading", "zoom", "sv_image_x", "sv_image_y", "gsv_panorama_id"):
        assert df[col].notna().all(), f"{city}: NULLs in {col}"


def test_nan_latlng_is_rare_and_paired(city_df):
    """A small number of depth computations produced NaN lat/lng (NaN passes SQL IS NOT NULL).
    The cleaning bounds filter drops them, exactly as the 2021 analysis did. 1,219 rows total."""
    city, df = city_df
    assert (df["lat"].isna() == df["lng"].isna()).all()
    assert df["lat"].isna().mean() < 0.01, f"{city}: too many NaN positions"


def test_booleans_parsed(city_df):
    city, df = city_df
    assert df["deleted"].notna().all() and df["tutorial"].notna().all()


def test_label_ids_unique_per_city(city_df):
    city, df = city_df
    assert df["label_id"].is_unique


def test_coordinates_in_bounds(city_df):
    """Camera positions are always sane. Label lat/lng is raw depth output and may contain
    garbage (NaN, or absurd magnitudes in the legacy DC data) — that is exactly what the 2021
    analysis' invalid-lat/lng cleaning filter removes, so the raw contract only bounds its rate."""
    city, df = city_df
    assert df["panorama_lat"].between(-90, 90).all() and df["panorama_lng"].between(-180, 180).all()
    bad = 1 - (df["lat"].between(-90, 90) & df["lng"].between(-180, 180)).mean()
    assert bad < 0.02, f"{city}: {bad:.2%} invalid label positions"  # observed: DC 1.39%, rest <0.8%


def test_time_created_nulls_only_in_dc(city_df):
    """Early DC-deployment rows predate the time_created column; every other city is complete."""
    city, df = city_df
    if city != "dc":
        assert df["time_created"].notna().all()


def test_extras_null_pattern(city_df):
    """current_pano_x/y exist only where evolution 179 ran (never in the legacy DC database)."""
    city, df = city_df
    if city == "dc":
        assert df["current_pano_x"].isna().all() and df["current_pano_y"].isna().all()
    else:
        assert df["current_pano_x"].notna().all() and df["current_pano_y"].notna().all()
