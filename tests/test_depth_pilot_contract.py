"""Layer 2 of the depth-pilot suite: contract tests on the committed artifacts.

These validate structure and internal consistency of data/depth-pilot-*.csv.gz
and depth-pilot-payloads.jsonl.gz -- schema, value domains, cross-file joins,
and that every payload in the committed evidence file still decodes. The
observed *findings* (medians, class fractions) live in
test_depth_pilot_findings.py; this file must stay true under any refetch.
"""

import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

import gsv_depth as gd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

PANOS_PATH = os.path.join(DATA, "depth-pilot-panos.csv.gz")
LABELS_PATH = os.path.join(DATA, "depth-pilot-labels.csv.gz")
PAYLOADS_PATH = os.path.join(DATA, "depth-pilot-payloads.jsonl.gz")
SUMMARY_PATH = os.path.join(DATA, "depth-pilot-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PANOS_PATH), reason="depth-pilot artifacts not built yet"
)


@pytest.fixture(scope="module")
def panos():
    return pd.read_csv(PANOS_PATH)


@pytest.fixture(scope="module")
def labels():
    return pd.read_csv(LABELS_PATH)


@pytest.fixture(scope="module")
def payloads():
    rows = []
    with gzip.open(PAYLOADS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------- panos file

def test_pano_parts_and_statuses(panos):
    assert set(panos["part"].unique()) == {"a", "b"}
    assert set(panos["status"].dropna().unique()) <= {
        "ok", "no_depth", "gone", "not_found"
    }


def test_pano_classes_valid(panos):
    ok = panos[panos["status"] == "ok"]
    a_ok = ok[ok["part"] == "a"]
    assert set(a_ok["pano_class"].dropna().unique()) <= {
        "unchanged", "mostly_unchanged", "changed", "no_comparable_labels"
    }


def test_part_a_strata(panos):
    a = panos[panos["part"] == "a"]
    assert set(a["stratum"].unique()) <= {
        "headline_h8192", "headline_h6656", "headline_hnull",
        "edge_absurd", "edge_seam_wrap", "edge_dc_overflow",
    }
    assert a["pano_id"].is_unique


def test_part_a_cities_all_seven(panos):
    a = panos[panos["part"] == "a"]
    assert set(a["city"].unique()) == {
        "dc", "seattle", "spgg", "columbus", "newberg", "cdmx", "pittsburgh"
    }


def test_part_b_cities(panos):
    b = panos[panos["part"] == "b"]
    assert set(b["city"].unique()) == {"seattle", "cdmx"}
    assert (b["stratum"] == "modern").all()


def test_consistency_counts_within_bounds(panos):
    ok = panos[(panos["part"] == "a") & (panos["status"] == "ok")]
    assert (ok["n_consistent"] <= ok["n_labels_compared"]).all()
    assert (ok["n_labels_compared"] <= ok["n_labels_raw"]).all()


def test_camera_height_columns_only_where_depth(panos):
    no_depth = panos[panos["status"].isin(["gone", "not_found", "no_depth"])]
    assert no_depth["ground_d"].isna().all()
    ok = panos[panos["status"] == "ok"]
    assert ok["n_planes"].notna().all()
    assert (ok["n_planes"] >= 2).all()


def test_ground_heights_physical(panos):
    ok = panos[panos["status"] == "ok"]
    h = ok["ground_height_m"].dropna()
    assert (h > 0.5).all() and (h < 10).all()


# ---------- labels file

def test_labels_join_panos(labels, panos):
    a_ids = set(panos[panos["part"] == "a"]["pano_id"].dropna())
    assert set(labels["pano_id"].unique()) <= a_ids


def test_labels_only_for_depth_panos(labels, panos):
    with_depth = set(panos[panos["status"] == "ok"]["pano_id"])
    assert set(labels["pano_id"].unique()) <= with_depth


def test_label_ids_unique_within_city(labels):
    # label_id is a per-city database sequence, unique only with the city key --
    # same contract as the main dataset.
    assert not labels.duplicated(subset=["city", "label_id"]).any()


def test_in_cleaned_is_keyed_on_city_not_the_bare_label_id():
    """The flag asks whether THIS label survived cleaning, not whether some city's did.

    A row-uniqueness check cannot catch this: the bug was a correctly-shaped frame with a
    wrongly-computed boolean. seattle:1 is cleaned and chicago:1 is not, so a bare
    isin() on label_id marks both -- which is what shipped, on 9.65% of era rows.
    """
    from run_depth_pilot import in_cleaned_flag

    raw = pd.DataFrame({"city": ["seattle", "chicago", "chicago"], "label_id": [1, 1, 2]})
    cleaned = pd.DataFrame({"city": ["seattle", "chicago"], "label_id": [1, 2]})
    assert in_cleaned_flag(raw, cleaned).tolist() == [True, False, True]


def test_comparable_labels_have_full_comparison(labels):
    comp = labels[labels["disagreement_m"].notna()]
    assert comp["dlat_ulp"].notna().all()
    assert comp["dlng_ulp"].notna().all()
    assert comp["consistent"].notna().all()
    assert (comp["disagreement_m"] >= 0).all()


def test_flags_are_exclusive_of_comparison(labels):
    # no-plane / out-of-bounds / stored-absurd rows never carry a coordinate delta
    flagged = labels[
        labels["no_plane"].fillna(False)
        | labels["out_of_bounds"].fillna(False)
        | labels["stored_absurd"].fillna(False)
    ]
    assert flagged["disagreement_m"].isna().all()


def test_consistent_matches_ulp_threshold(labels):
    comp = labels[labels["disagreement_m"].notna()]
    expected = (comp["dlat_ulp"].abs() <= 2.0) & (comp["dlng_ulp"].abs() <= 2.0)
    assert (comp["consistent"].astype(bool) == expected).all()


# ---------- payloads file

def test_payloads_unique_and_joined(payloads, panos):
    ids = [p["pano_id"] for p in payloads]
    assert len(ids) == len(set(ids))
    ok_ids = set(panos[panos["status"] == "ok"]["pano_id"])
    assert set(ids) == ok_ids


def test_every_committed_payload_decodes(payloads):
    for rec in payloads:
        p = gd.decode_depth_payload(rec["depth_b64"])
        assert (p.width, p.height) == (gd.DEPTH_W, gd.DEPTH_H)
        assert p.n_planes >= 2


def test_payloads_carry_fetch_provenance(payloads):
    for rec in payloads:
        assert rec["fetched_utc"]
        assert rec["part"] in ("a", "b")


# ---------- summary file

def test_summary_reflects_artifacts(summary, panos, labels):
    a_head = panos[
        (panos["part"] == "a") & panos["stratum"].str.startswith("headline")
    ]
    assert summary["part_a"]["attempted"] == len(a_head)
    assert summary["part_b"]["locations"] == (panos["part"] == "b").sum()
    lab = labels[
        labels["in_cleaned"].astype(bool)
        & labels["disagreement_m"].notna()
        & labels["pano_id"].isin(
            set(a_head[a_head["status"] == "ok"]["pano_id"])
        )
    ]
    assert summary["part_a"]["labels_compared"] == len(lab)
