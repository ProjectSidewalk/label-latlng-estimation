"""The modern-truth findings, locked (issue #3; reports/2026-08-07-modern-truth.md).

Headline claims this file holds together with the committed artifacts:

- the sampler frame is right: on the 409 committed pilot payloads, the modern
  heading-centred lookup lands within one raster row of the yaw-corrected era lookup
  every time (exactly equal 68%; the rest is the era ceil vs modern round half-pixel)
  and within two columns 94%, while the x-mirrored lookup agrees nowhere; on the modern
  set itself, the identity frame beats every deliberately wrong frame on ground-hit
  share and on error;
- the circularity guard: stored post-2021 positions are estimator echoes in BOTH eras
  (fixed-frame before evolution 179, real pixels after), so no evaluation here may use
  stored positions as truth — and the era boundary leaves a measurable discontinuity;
- the absolute check: the shipped blend's error and bias against fresh-depth truth on
  post-2021 human labels, vs the deployed per-zoom linear path (see the locked numbers).

The final test re-derives the committed summary's headline block in-process from the
committed payloads + labels, so the JSON cannot drift from the code.
"""

import gzip
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SUMMARY_PATH = os.path.join(DATA, "modern-truth-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH),
    reason="modern-truth artifacts not built (run_modern_truth.py build --write)")


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def labels():
    return pd.read_csv(os.path.join(DATA, "modern-truth-labels.csv.gz"),
                       dtype={"pano_id": str})


@pytest.fixture(scope="module")
def panos():
    return pd.read_csv(os.path.join(DATA, "modern-truth-panos.csv.gz"),
                       dtype={"pano_id": str})


@pytest.fixture(scope="module")
def payloads():
    out = {}
    with gzip.open(os.path.join(DATA, "modern-truth-payloads.jsonl.gz"), "rt",
                   encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["pano_id"]] = rec["b64"]
    return out


# ---------------------------------------------------------------------- artifact shape

def test_artifacts_are_consistent(summary, labels, panos, payloads):
    meta = summary["meta"]
    assert meta["n_panos_attempted"] == len(panos)
    assert meta["n_panos_ok"] == (panos["status"] == "ok").sum() == len(payloads)
    assert meta["n_labels_on_ok_panos"] == len(labels)
    assert meta["n_scored"] == labels["gate_ok"].sum()
    assert set(labels.loc[labels["gate_ok"], "hit_class"]) <= {"ground", "terrain"}
    assert labels["pano_id"].isin(set(payloads)).all()


def test_the_join_key_is_city_scoped_and_unique(summary, labels):
    """label_id is a per-schema serial: joining on it alone pairs a label with another
    city's depth truth. label_uid is the key; label_id collides here by construction."""
    assert labels["label_uid"].is_unique
    assert (labels["label_uid"] == labels["city"] + ":" + labels["label_id"].astype(str)).all()
    # the hazard is real in this population, not hypothetical
    assert not labels["label_id"].is_unique
    assert summary["frame_census"]["rows_sharing_a_label_id"] > 900_000


def test_pano_strata_met_their_budgets_and_the_type_shortfall_is_recorded(summary):
    """The pano strata are pano-count budgets and were met. The type strata are
    LABEL-count budgets; one fell short, and the summary has to say which."""
    by = summary["fetch"]["by_stratum_ok"]
    b = summary["meta"]["budgets"]
    assert by["representative"] == b["representative"]
    assert by["near_horizon"] == b["near_horizon"]
    assert by["ai"] == b["ai"]

    cov = summary["fetch"]["type_label_coverage"]
    assert len(cov) == 9 and set(summary["implied_heights"]) <= set(cov)
    short = {t for t, v in cov.items() if not v["met"]}
    assert short == {"NoCurbRamp"}, short
    assert cov["NoCurbRamp"]["n_labels_on_ok_panos"] == 156  # of a 200-label quota


# ---------------------------------------------------------------------- the sampler frame

def test_modern_lookup_matches_era_lookup_on_the_409_pilot_payloads():
    """The offline cross-check: same physical pixel, two independent lookup paths."""
    import gsv_depth as gd
    import modern_truth as mt
    from label_latlng_estimation import load_data

    pilot_ids = set()
    with gzip.open(os.path.join(DATA, "depth-pilot-payloads.jsonl.gz"), "rt",
                   encoding="utf-8") as f:
        for line in f:
            pilot_ids.add(json.loads(line)["pano_id"])
    yaw = (pd.read_csv(os.path.join(DATA, "depth-validation-panometa.csv.gz"),
                       dtype={"pano_id": str})
           .set_index("pano_id")["yaw_deg"].to_dict())
    raw = load_data(DATA).rename(columns={"gsv_panorama_id": "pano_id"})
    sub = raw[raw["pano_id"].isin(pilot_ids)
              & raw["current_pano_x"].notna() & raw["pano_width"].notna()]

    d_col, d_row, d_mirror = [], [], []
    for r in sub.itertuples():
        y = yaw.get(r.pano_id)
        if y is None or not np.isfinite(y):
            continue
        col_m, row_m = mt.modern_col_row(r.current_pano_x, r.current_pano_y,
                                         r.pano_width, r.pano_height)
        bearing = r.sv_image_x / 13312.0 * 360.0
        fx = ((bearing - y + 180.0) % 360.0) / 360.0
        col_v6 = math.ceil(fx * 13312.0 * gd.SV_IMAGE_SCALE) % 512
        row_v6 = math.ceil((gd.SV_IMAGE_Y_ORIGIN - r.sv_image_y) * gd.SV_IMAGE_SCALE)
        d_col.append((col_m - col_v6 + 256) % 512 - 256)
        d_row.append(row_m - row_v6)
        d_mirror.append(((511 - col_m) - col_v6 + 256) % 512 - 256)

    d_col, d_row, d_mirror = map(np.asarray, (d_col, d_row, d_mirror))
    assert len(d_col) >= 500
    # rows agree within one everywhere; the +-1 tail is the era ceil vs modern round
    # half-pixel, so exact equality is a two-thirds majority, not the whole population
    assert (np.abs(d_row) <= 1).mean() == 1.0
    assert (d_row == 0).mean() == pytest.approx(0.676, abs=0.02)
    assert (np.abs(d_col) <= 2).mean() >= 0.90  # tail = legacy camera-heading drift
    # the mirrored frame agrees NOWHERE, and not marginally: col -> 511-col only lands
    # near the true column for labels near col 255, of which this population has none
    assert (np.abs(d_mirror) <= 2).mean() == 0.0
    assert np.abs(d_mirror).min() >= 4


def test_identity_frame_beats_every_wrong_frame(summary):
    fc = summary["frame_controls"]
    ident = fc["identity"]
    for control in ("x_mirror", "rotate_180", "row_flip"):
        assert ident["ground_or_terrain_share"] > fc[control]["ground_or_terrain_share"]
        if fc[control]["D_blend_median_abs_m"] is not None:
            assert ident["D_blend_median_abs_m"] < fc[control]["D_blend_median_abs_m"]


# ---------------------------------------------------------------------- the guard

def test_stored_positions_are_estimator_echoes_in_both_eras(summary):
    g = summary["guard"]
    for era in ("fixed_frame", "real_pixels"):
        assert g[era]["median_abs_diff_m"] < 0.35
        # the wrong era's formula misses by many times the right one's
        assert g[era]["wrong_era_median_abs_m"] > 4 * g[era]["median_abs_diff_m"]


# ---------------------------------------------------------------------- locked numbers

def test_locked_headline_matrix(summary):
    head = summary["matrix"]["headline_representative_human"]
    assert head["n"] == 1484
    assert head["A_deployed"]["median_abs_m"] == pytest.approx(1.0841, abs=2e-4)
    assert head["A_deployed"]["signed_median_m"] == pytest.approx(-0.2811, abs=2e-4)
    assert head["B_normalized"]["signed_median_m"] == pytest.approx(1.7838, abs=2e-4)
    assert head["C_anchor"]["signed_median_m"] == pytest.approx(1.0196, abs=2e-4)
    assert head["D_blend"]["median_abs_m"] == pytest.approx(1.1932, abs=2e-4)
    assert head["D_blend"]["signed_median_m"] == pytest.approx(1.0869, abs=2e-4)
    # the blend keeps the best tail even as shipped
    assert head["D_blend"]["p90_abs_m"] < head["A_deployed"]["p90_abs_m"]


def test_locked_pooled_human_matrix(summary):
    """The pooled column the report quotes for compression; separate from the headline
    so the two populations cannot be silently mixed."""
    pooled = summary["matrix"]["all_human"]
    assert pooled["n"] == 2655
    assert pooled["A_deployed"]["median_abs_m"] == pytest.approx(1.2283, abs=2e-4)
    assert pooled["A_deployed"]["p90_abs_m"] == pytest.approx(5.2401, abs=2e-4)
    assert pooled["A_deployed"]["range_slope_m_per_m"] == pytest.approx(-0.4375, abs=2e-4)
    assert pooled["D_blend"]["p90_abs_m"] == pytest.approx(3.9765, abs=2e-4)


def test_the_blend_bias_is_a_uniform_scale_error(summary):
    """Every scoreable city and capture year floats the same direction.

    "Scoreable" is a real restriction and the summary states its size: by_city reports
    the cities clearing 50 gated human rows, not all 36 that contribute any."""
    cov = summary["by_city_coverage"]
    assert cov["n_cities_scored"] == len(summary["by_city"]) == 13
    assert cov["n_cities_with_scored_human_rows"] == 36
    assert cov["rows_in_cities_below_min_n"] == 369  # 14% of the pooled human rows
    for city, v in summary["by_city"].items():
        assert 0.3 < v["D_blend"]["signed_median_m"] < 2.1, city
    for year, v in summary["by_capture_year"].items():
        assert 0.7 < v["D_blend"]["signed_median_m"] < 1.5, year


def test_implied_heights_are_flat_at_the_measured_rig(summary):
    ch = summary["camera_heights"]
    assert ch["measured_median_m"] == pytest.approx(2.3544, abs=2e-3)
    assert ch["pinned_2p5_frac"] == pytest.approx(0.313, abs=5e-3)
    for t, v in summary["implied_heights"].items():
        assert 2.25 < v["implied_height_m"] < 2.40, t
        if not v["uses_fallback"]:
            # the fitted table sits well above what modern truth supports
            assert v["fitted_height_m"] - v["implied_height_m"] > 0.14, t
    # the road-paint consistency check: Crosswalk implies the measured rig itself
    assert (summary["implied_heights"]["Crosswalk"]["implied_height_m"]
            == pytest.approx(ch["measured_median_m"], abs=0.05))

    # the spread claim, locked: what the era table asserts, what a rescale would keep,
    # and what modern truth actually shows. The last is the reason the flat height ships.
    implied = [v["implied_height_m"] for v in summary["implied_heights"].values()]
    fitted = [v["fitted_height_m"] for v in summary["implied_heights"].values()
              if not v["uses_fallback"]]
    k = summary["remedies"]["k_rescale"]
    assert max(fitted) - min(fitted) == pytest.approx(0.2841, abs=2e-3)
    assert k * (max(fitted) - min(fitted)) == pytest.approx(0.2457, abs=2e-3)
    assert max(implied) - min(implied) == pytest.approx(0.0978, abs=2e-3)


def test_locked_guard_numbers(summary):
    g = summary["guard"]
    assert g["fixed_frame"]["median_abs_diff_m"] == pytest.approx(0.2921, abs=2e-3)
    assert g["real_pixels"]["median_abs_diff_m"] == pytest.approx(0.0978, abs=2e-3)
    assert g["real_pixels"]["frac_echo"] == pytest.approx(0.976, abs=5e-3)
    assert g["fixed_frame"]["frac_echo"] == pytest.approx(0.756, abs=5e-3)


def test_locked_remedies(summary):
    r = summary["remedies"]
    assert r["k_rescale"] == pytest.approx(0.8648, abs=2e-4)
    assert r["flat_height_m"] == pytest.approx(2.3416, abs=2e-3)
    t = r["test_half"]
    assert t["D_blend_as_shipped"]["median_abs_m"] == pytest.approx(1.2786, abs=2e-4)
    assert t["D_rescaled"]["median_abs_m"] == pytest.approx(0.4425, abs=2e-4)
    assert t["D_flat"]["median_abs_m"] == pytest.approx(0.4100, abs=2e-4)
    # the rescale removes the bias, not just the spread
    assert abs(t["D_rescaled"]["signed_median_m"]) < 0.3
    # the deployed model on the SAME rows — the only fair reference for the 0.41 m
    assert t["A_deployed"]["median_abs_m"] == pytest.approx(1.1695, abs=2e-4)
    assert t["D_flat"]["median_abs_m"] < t["A_deployed"]["median_abs_m"]


def test_modern_truth_does_not_carry_the_era_curb_overshoot(summary):
    c = summary["curb_sensitivity"]
    # applying the classic curb correction to modern truth WORSENS the bias
    assert abs(c["signed_median_corrected_m"]) > abs(c["signed_median_m"])


def test_near_horizon_the_blend_beats_the_deployed_model_and_is_bounded(summary):
    """At <=2 deg truth outruns every bounded answer. The blend clearly beats the
    deployed linear model; against the raw cotangent it is a wash in magnitude (they
    miss by the same amount in opposite directions), so the blend's real advantage
    there is structural — a finite largest answer — not a median at n=20."""
    nh = summary["near_horizon"]["(0, 2]"]
    assert nh["n"] < 40  # thin by construction: read the direction, not the value
    assert abs(nh["D_blend"]["signed_median_m"]) < abs(nh["A_deployed"]["signed_median_m"])
    assert abs(abs(nh["D_blend"]["signed_median_m"])
               - abs(nh["C_anchor"]["signed_median_m"])) < 1.0
    # what actually separates them: the cotangent runs to the 50 m clip, the blend cannot
    with open(os.path.join(DATA, "distance-refit-summary.json"), encoding="utf-8") as f:
        assert json.load(f)["provisional_coefficients"]["max_answer_m"] < 30.0
    # and where the sample IS thick, the blend is the flattest of the four
    thick = summary["near_horizon"]["(11.25, 90]"]
    assert thick["n"] > 1000
    assert abs(thick["D_blend"]["signed_median_m"]) < 1.2


# ---------------------------------------------------------------------- reproduction

def _reproduce_truth(labels, payloads):
    """Re-read every given label's truth from its own pano's committed payload bytes.

    Yields the rows that disagree. Decoding and geometry are per-pano, so grouping is
    what makes a whole-population sweep affordable."""
    import gsv_depth as gd
    import depth_validation as dv
    import modern_truth as mt

    bad = []
    for pano_id, sub in labels.groupby("pano_id"):
        payload = gd.decode_depth_payload(payloads[pano_id])
        cam_h = gd.camera_height_qc(payload).ground_height
        geom = dv.payload_geometry(payload)
        for r in sub.itertuples():
            hit = mt.classify_modern_label(payload, r.pano_x, r.pano_y, r.pano_width,
                                           r.pano_height, cam_h, geom)
            same = hit.hit_class == r.hit_class and (
                (not math.isfinite(hit.horizontal_m) and not math.isfinite(r.truth_m))
                or abs(hit.horizontal_m - r.truth_m) < 1e-6)
            if not same:
                bad.append((r.label_uid, r.hit_class, hit.hit_class, r.truth_m,
                            hit.horizontal_m))
    return bad


def test_summary_reproduces_from_committed_artifacts(summary, labels, payloads):
    """Re-derive the headline matrix, and every truth value on a slice of the panos.

    The slice is by PANO, not by label: a mis-keyed join corrupts whole labels, and
    sampling panos keeps the payload decode amortised. 250 of 1,106 panos is ~23% of
    the rows, which detects a 1%-of-rows corruption essentially always — the earlier
    40-row label sample did not, and missed exactly such a bug."""
    import modern_truth as mt

    human = labels[labels["gate_ok"] & ~labels["is_ai"]]
    head = human[human["stratum"] == "representative"]
    got = mt.model_metrics(head)
    want = summary["matrix"]["headline_representative_human"]
    assert got["n"] == want["n"]
    for key in mt.MODEL_KEYS:
        for stat in ("median_abs_m", "signed_median_m", "p90_abs_m"):
            assert got[key][stat] == pytest.approx(want[key][stat], abs=1e-9)

    panos = pd.Series(sorted(labels["pano_id"].unique())).sample(250, random_state=666)
    sample = labels[labels["pano_id"].isin(set(panos))]
    assert len(sample) > 600
    assert _reproduce_truth(sample, payloads) == []


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="re-reads every committed label's truth from the payload bytes (~50 s); "
           "set RUN_SLOW=1")
def test_every_truth_value_reproduces_from_the_committed_payloads(labels, payloads):
    """The exhaustive form of the above: no row anywhere may disagree with the bytes."""
    assert _reproduce_truth(labels, payloads) == []
