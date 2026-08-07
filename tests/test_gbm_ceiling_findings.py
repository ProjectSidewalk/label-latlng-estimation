"""The issue #6 findings, locked: the GBM accuracy ceiling over the #3 closed forms.

Convention (mirrors test_distance_refit_findings.py): these tests assert what the
2026-08-07 run measured, reading the committed data/gbm-ceiling-summary.json only. The
summary regenerates offline and deterministically with
`python python/run_gbm_ceiling.py --write` (fixed seeds, deterministic LightGBM params);
the runner itself asserts in-process that its refit A/D baselines equal
data/distance-refit-summary.json, and the cross-summary tests below re-check that from
the committed artifacts.

Headline findings (reports/2026-08-07-gbm-ceiling.md):

- The GBM reaches 0.54 m median lat/lng error on the published test split vs the shipped
  blend D's 0.93 m — D sits ~74% above the ceiling, NOT within 10-15%. The closed form
  is not essentially free of modeling regret.
- But none of the gap is one-dimensional: a GBM given only sv_image_y lands on D
  (0.930 vs 0.934 m), and handing the GBM the exact depression angle changes nothing
  (~2 mm). The shipped geometry captures the 1-D vertical structure essentially
  perfectly; the headroom is interaction/context structure with no single carrier —
  drop-one ablations move the median <5 mm for every feature except sv_image_y.
- The GBM is the MORE noise-sensitive model: 4-5x D's median degradation at 2 px of
  click noise, ~1.8x at 10 px. Fine-grained structure is fragile structure.
- Benchmark only, explicitly not a production candidate (no JS runtime, no
  interpretable coefficients).
"""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "data", "gbm-ceiling-summary.json")
REFIT_SUMMARY_PATH = os.path.join(ROOT, "data", "distance-refit-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="gbm-ceiling summary not built yet"
)

D = "D_blend_type_l1"


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def refit():
    with open(REFIT_SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ harness comparability

def test_scored_on_the_published_split(summary):
    """Identical rows to every #3 candidate: the full published test split, every model."""
    assert summary["meta"]["n_train"] == 316118
    assert summary["meta"]["n_test"] == 79029
    for key, row in summary["matrix"].items():
        assert row["n"] == 79029, key


def test_baselines_lock_to_the_refit_summary(summary, refit):
    """The in-process refit of A_ols and blend D must reproduce the committed #3 numbers
    to float precision — same rows, same fits, same scoring geometry — or the GBM is not
    comparable to the #3 matrix at all."""
    for key in ("A_ols", D):
        for m in ("latlng_median_m", "latlng_p90_m", "dist_median_m", "dist_p90_m"):
            assert summary["matrix"][key][m] == pytest.approx(
                refit["matrix"][key][m], rel=1e-12), (key, m)
    assert summary["meta"]["era_cal_delta_deg"] == pytest.approx(
        refit["meta"]["era_cal_delta_deg"], abs=1e-12)


def test_noise_sweep_uses_the_same_draws_as_number_3(summary, refit):
    """The sweep mirrors distance_refit.noise_sweep's rng recipe (seed 666, sigma-major,
    two draws per repetition), so the recomputed A/D deltas must equal the committed #3
    sweep exactly — which locks the GBM rows to the identical perturbed clicks."""
    ns, ref = summary["noise_sweep"], refit["noise_sweep"]
    assert ns["sigmas_px"] == ref["sigmas_px"]
    assert ns["n_draws"] == ref["n_draws"]
    for key in ("A_ols", D):
        for s in ("2.0", "5.0", "10.0"):
            assert ns["per_model"][key][s]["delta_median_m"] == pytest.approx(
                ref["per_rung"][key][s]["delta_median_m"], rel=1e-12), (key, s)


def test_deterministic_benchmark_config(summary):
    """The reproducibility contract: fixed seed, deterministic LightGBM, no subsampling,
    and the explicit not-a-production-candidate marker."""
    p = summary["meta"]["lgb_params"]
    assert p["deterministic"] is True
    assert p["seed"] == 666
    assert p["feature_fraction"] == 1.0 and p["bagging_fraction"] == 1.0
    assert "benchmark only" in summary["meta"]["not_a_production_candidate"]


def test_dc_null_pano_height_is_carried_not_dropped(summary):
    """pano_height (and therefore sv_norm) is null on every DC row — 58-59% of both
    splits. The rows stay in as native NaN; nothing is dropped or imputed."""
    assert summary["meta"]["n_pano_height_missing_train"] == 185464
    assert summary["meta"]["n_pano_height_missing_test"] == 46543


# ------------------------------------------------------------------ the ceiling

def test_gbm_headline_numbers(summary):
    m = summary["matrix"]
    assert m["gbm_l1"]["latlng_median_m"] == pytest.approx(0.5378, abs=0.005)
    assert m["gbm_l1"]["latlng_p90_m"] == pytest.approx(3.291, abs=0.02)
    assert m["gbm_dep_l1"]["latlng_median_m"] == pytest.approx(0.5362, abs=0.005)
    assert m["gbm_l2"]["latlng_median_m"] == pytest.approx(0.5964, abs=0.005)
    # loss/metric alignment, same story as the ladder's l1 column
    assert m["gbm_l1"]["latlng_median_m"] < m["gbm_l2"]["latlng_median_m"]


def test_the_ceiling_question_answered_no(summary):
    """Issue #6's question: does blend D sit within ~10-15% of the GBM's test median?
    No — D is ~74% above it (A ~169%). There is real structure the geometry isn't using."""
    c = summary["ceiling"]
    assert c["blend_d_latlng_median_m"] == pytest.approx(0.9336, abs=0.005)
    assert c["gbm_best_latlng_median_m"] == pytest.approx(0.5362, abs=0.005)
    assert c["d_over_gbm_gap_pct"] == pytest.approx(74.1, abs=1.5)
    assert c["d_over_gbm_gap_pct"] > 15.0
    assert c["a_over_gbm_gap_pct"] == pytest.approx(169.2, abs=2.5)


def test_but_the_gap_is_not_one_dimensional(summary):
    """The structural finding that protects the #3 recommendation: a GBM restricted to the
    single vertical signal lands on the closed form (only_sv_image_y 0.930 vs D 0.934;
    only exact depression 0.985), so D has essentially no regret against the best 1-D
    model. The headroom is conditional/interaction structure."""
    m = summary["matrix"]
    assert abs(m["only_sv_image_y"]["latlng_median_m"] - m[D]["latlng_median_m"]) < 0.02
    assert m["only_depression"]["latlng_median_m"] == pytest.approx(0.985, abs=0.01)
    assert m["only_sv_image_y"]["latlng_median_m"] > m["gbm_l1"]["latlng_median_m"] + 0.3
    # canvas position alone knows almost nothing about distance
    assert m["only_canvas"]["latlng_median_m"] > 2.5


def test_exact_depression_adds_nothing_the_gbm_cannot_reconstruct(summary):
    """Handing the GBM the #5 exact projection as a feature moves the median ~2 mm: the
    raw inputs (canvas, heading, pitch, zoom, sv_image_y) already contain it."""
    m = summary["matrix"]
    assert abs(m["gbm_dep_l1"]["latlng_median_m"]
               - m["gbm_l1"]["latlng_median_m"]) < 0.005


# ------------------------------------------------------------------ the ablation

def test_ablation_sv_image_y_is_the_only_load_bearing_feature(summary):
    """Drop-one: removing sv_image_y costs +0.10 m dist median (and even that is mostly
    reconstructed from sv_norm and the click geometry); removing any other group moves
    the median by under 5 mm — several drops even improve it slightly. The headroom is a
    redundant pool, not a missing feature with a name."""
    drop = summary["ablation"]["drop"]
    assert drop["sv_image_y"]["delta_dist_median_m"] > 0.05
    for gname, row in drop.items():
        if gname == "sv_image_y":
            continue
        assert abs(row["delta_dist_median_m"]) < 0.01, gname
        assert abs(row["delta_latlng_median_m"]) < 0.01, gname


def test_gain_concentrates_on_the_vertical_signal(summary):
    imp = summary["feature_importance_gain_gbm_l1"]
    assert imp["sv_image_y"]["gain_share"] > 0.7
    # the rest of the pool is led by the resolution/era axis (#3's B_log confound)
    assert imp["sv_norm"]["gain_share"] + imp["pano_height"]["gain_share"] > 0.1


def test_gap_is_concentrated_beyond_10m_and_in_the_tail(summary):
    """Error vs true distance: the GBM's advantage over D grows with distance (where the
    depth truth is also weakest — read with §6 of the #3 report); the tail improves too
    (p90 3.29 vs 4.48)."""
    m = summary["matrix"]
    assert m["gbm_l1"]["latlng_p90_m"] < m[D]["latlng_p90_m"] - 1.0
    rows = {r["bin_m"]: r for r in summary["error_vs_distance"]}
    for b in ("[10.0, 15.0)", "[15.0, 20.0)", "[20.0, 30.0)", "[30.0, 50.0)"):
        gbm = rows[b]["per_model"]["gbm_l1"]["latlng_median_m"]
        d = rows[b]["per_model"][D]["latlng_median_m"]
        assert gbm < 0.85 * d, b
    assert rows["[30.0, 50.0)"]["per_model"]["gbm_l1"]["latlng_median_m"] == pytest.approx(
        8.36, abs=0.2)


# ------------------------------------------------------------------ noise

def test_gbm_is_the_more_noise_sensitive_model(summary):
    """Measured, not assumed (the issue left the direction open): at every sigma the GBM
    degrades more than both D and A — 4-5x D at 2 px, ~1.8x at 10 px. The interaction
    structure that buys the ceiling is exactly the part click noise destroys first."""
    ns = summary["noise_sweep"]["per_model"]
    for s in ("2.0", "5.0", "10.0"):
        for gbm_key in ("gbm_l1", "gbm_dep_l1"):
            assert (ns[gbm_key][s]["delta_median_m"]
                    > ns[D][s]["delta_median_m"]), (gbm_key, s)
            assert (ns[gbm_key][s]["delta_median_m"]
                    > ns["A_ols"][s]["delta_median_m"]), (gbm_key, s)
    assert ns["gbm_l1"]["10.0"]["delta_median_m"] == pytest.approx(0.259, abs=0.01)
    ratio = ns["gbm_l1"]["10.0"]["delta_median_m"] / ns[D]["10.0"]["delta_median_m"]
    assert 1.2 < ratio < 3.0
    # even fully degraded at 10 px, the GBM median stays below D's unperturbed median —
    # the ceiling conclusion is not an artifact of noise-free scoring
    assert (summary["noise_sweep"]["baseline_median_m"]["gbm_l1"]
            + ns["gbm_l1"]["10.0"]["delta_median_m"]
            < summary["noise_sweep"]["baseline_median_m"][D])
