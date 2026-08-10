"""The committed conclusions page (issue #7 viz) stays in lock-step with the summary.

The page is context, not evidence — no report number depends on it — but once committed
it must not silently drift from ``data/triangulation-summary.json``, and it must stay
self-contained (every pixel embedded, nothing fetched at view time).
"""

from __future__ import annotations

import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "figures", "triangulation-conclusions.html")
SUMMARY = os.path.join(ROOT, "data", "triangulation-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PAGE), reason="conclusions page not built yet"
)


@pytest.fixture(scope="module")
def page_data():
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert m, "page carries no embedded DATA blob"
    return json.loads(m.group(1))


def test_page_numbers_match_the_committed_summary(page_data):
    with open(SUMMARY, encoding="utf-8") as f:
        s = json.load(f)
    ps = page_data["summary"]
    assert ps["pooled_ratio"] == s["depth_anchor"]["pooled"]["median_ratio_tri_over_depth"]
    assert ps["shipped"] == s["meta"]["shipped_height_m"]
    for run, v in ps["scale_global"].items():
        assert v["height_m"] == s["scale_global"][run]["height_m"], run
    assert ps["gap_range_profile"] == s["depth_anchor"]["gap_range_profile"]


def test_page_is_self_contained_with_real_pixels(page_data):
    sites = page_data["sites"]
    assert len(sites) == 6
    for site in sites:
        assert len(site["members"]) >= 4, site["site"]
        for m in site["members"]:
            assert m["img"].startswith("data:image/jpeg;base64,"), site["site"]
        # every committed depth read is rendered as a depth panel, and vice versa
        assert (sum(1 for m in site["members"] if m.get("dimg"))
                == sum(1 for m in site["members"] if m["rd"] is not None)
                == site["n_depth"]), site["site"]
