"""Cross-language equivalence: the Python port must reproduce the R baseline.

The R baseline (tests/fixtures/r-baseline/, produced by scripts/rerun-analysis.R) and the Python
pipeline both run on the committed data with the SAME train/test rows (the R-exported split
fixture), so every closed-form quantity — OLS coefficients, medians, error statistics — must
agree to floating-point noise. This is the port-fidelity guarantee: if these pass, the Python
implementation is doing the same math the R analysis did.

Estimate 6 is the one exception: lme4's and statsmodels' REML optimizers converge to slightly
different points, so it gets a loose tolerance (it lost the 2021 comparison anyway).
"""

import pytest

TIGHT = 1e-8   # closed-form fits: identical input rows, deterministic OLS
STATS = 1e-6   # error statistics: same formulas, allow accumulation-order noise
LOOSE = 1e-2   # est6: different REML optimizers


def rel_ok(a, b, tol):
    # relative, with a 1e-6 absolute floor: min-error statistics can be sub-millimeter, where
    # nanometer-scale float noise otherwise trips a pure relative check
    return abs(a - b) <= max(tol * max(abs(a), abs(b)), 1e-6)


def assert_coef_dict(actual: dict, expected: dict, tol=TIGHT, label=""):
    assert set(actual) == set(expected), f"{label}: term names differ"
    for k in expected:
        assert rel_ok(actual[k], expected[k], tol), \
            f"{label}[{k}]: python {actual[k]!r} vs R {expected[k]!r}"


def test_split_sizes(analysis, baseline):
    assert analysis["meta"]["rows_after_cleaning"] == baseline["meta"]["rows_after_cleaning"]
    assert analysis["meta"]["rows_train"] == baseline["meta"]["rows_train"]
    assert analysis["meta"]["rows_test"] == baseline["meta"]["rows_test"]
    assert analysis["meta"]["split"] == "r-fixture"


def test_raw_counts(analysis, baseline):
    assert analysis["meta"]["attrition"]["raw"] == sum(
        baseline["meta"]["raw_rows_per_city"].values())


def test_est2_median_dist(analysis, baseline):
    assert rel_ok(analysis["est2"]["median_dist"], baseline["est2"]["median_dist"], TIGHT)


def test_est3_medians_by_label_type(analysis, baseline):
    assert_coef_dict(analysis["est3"]["median_dist_by_label_type"],
                     baseline["est3"]["median_dist_by_label_type"], TIGHT, "est3")


def test_est4_coefficients(analysis, baseline):
    for resp in ("heading_diff", "pano_dist"):
        assert_coef_dict(analysis["est4"]["coefficients"][resp],
                         baseline["est4"]["coefficients"][resp], TIGHT, f"est4.{resp}")


def test_est5_coefficients(analysis, baseline):
    assert_coef_dict(analysis["est5"]["dist"], baseline["est5"]["dist"], TIGHT, "est5.dist")
    assert_coef_dict(analysis["est5"]["heading"], baseline["est5"]["heading"], TIGHT, "est5.heading")


@pytest.mark.parametrize("which", ["est7", "est7_full"])
def test_est7_coefficients(analysis, baseline, which):
    for part in ("dist", "heading"):
        for z in range(3):
            assert_coef_dict(analysis[which][part][z], baseline[which][part][z],
                             TIGHT, f"{which}.{part}.zoom{z + 1}")


def test_est6_available_and_close(analysis, baseline):
    assert baseline["est6"]["available"] and analysis["est6"]["available"]
    for part in ("dist", "heading"):
        assert_coef_dict(analysis["est6"][part]["fixef"], baseline["est6"][part]["fixef"],
                         LOOSE, f"est6.{part}.fixef")
        for z in ("1", "2", "3"):
            assert abs(analysis["est6"][part]["ranef_zoom"][z]
                       - baseline["est6"][part]["ranef_zoom"][z]) < 0.05, f"est6.{part}.ranef[{z}]"


def test_error_statistics(analysis, baseline):
    r_rows = {r["estimate"]: r for r in baseline["error_stats"]["summary"]}
    p_rows = {r["estimate"]: r for r in analysis["error_stats"]["summary"]}
    assert set(r_rows) == set(p_rows)
    for est, r in r_rows.items():
        p = p_rows[est]
        if est == "error_est6":
            # optimizer differences shift individual predictions; aggregate stats still agree
            for stat in ("mean", "median", "sd"):
                assert rel_ok(p[stat], r[stat], LOOSE), f"{est}.{stat}"
        else:
            for stat in ("mean", "median", "min", "max", "sd"):
                assert rel_ok(p[stat], r[stat], STATS), f"{est}.{stat}: {p[stat]} vs {r[stat]}"


def test_error_ranking_identical(analysis, baseline):
    r_order = [r["estimate"] for r in baseline["error_stats"]["summary"]]
    p_order = [r["estimate"] for r in analysis["error_stats"]["summary"]]
    assert p_order == r_order


def test_component_error_medians(analysis, baseline):
    for key in ("heading_error_medians", "dist_error_medians"):
        for est, r_val in baseline["error_stats"][key].items():
            tol = LOOSE if est.endswith("est6") else STATS
            assert rel_ok(analysis["error_stats"][key][est], r_val, tol), f"{key}.{est}"
