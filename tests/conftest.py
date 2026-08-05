import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from label_latlng_estimation import load_data, run_analysis  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
FIXTURES_DIR = os.path.join(ROOT, "tests", "fixtures", "r-baseline")


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def baseline():
    """The R baseline produced by scripts/rerun-analysis.R on the committed data."""
    with open(os.path.join(FIXTURES_DIR, "baseline.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_data():
    """All seven committed csv.gz files, loaded once per test session."""
    return load_data(DATA_DIR)


@pytest.fixture(scope="session")
def analysis(raw_data):
    """The full Python pipeline run on the committed data with the R-exported train/test split,
    so every fit is on exactly the rows the R baseline used."""
    return run_analysis(DATA_DIR, FIXTURES_DIR, data=raw_data)
