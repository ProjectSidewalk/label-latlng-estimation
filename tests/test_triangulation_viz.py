"""The committed conclusions page (issue #7 viz) stays in lock-step with the summary.

The page is context, not evidence — no report number depends on it — but once committed
it must not silently drift from ``data/triangulation-summary.json``, and it must stay
self-contained (every pixel embedded, nothing fetched at view time).
"""

from __future__ import annotations

import base64
import io
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
            # the viz top-up bundle gives every camera a depth-model panel
            assert m.get("dimg", "").startswith("data:image/png;base64,"), site["site"]
        # ...but committed r_depth values remain exactly the §8 anchor population
        assert (sum(1 for m in site["members"] if m["rd"] is not None)
                == site["n_depth"]), site["site"]


def test_every_site_stands_on_its_own_georeferenced_ground(page_data):
    """The aerial patch is cut for exactly the +/-ext metres the scene draws.

    Nothing on screen would look wrong if the two disagreed — the imagery would simply
    sit at the wrong scale under geometry that is still drawn correctly, which is the
    kind of error a reader would read straight past. So the extent travels with the
    patch and is asserted here as well as refused at build time.
    """
    from PIL import Image

    for site in page_data["sites"]:
        g = site.get("ground")
        assert g is not None, f"site {site['site']} has no aerial ground"
        assert g["img"].startswith("data:image/jpeg;base64,"), site["site"]
        ext = site["ext"]
        assert isinstance(ext, int) and ext >= 12 and ext % 2 == 0, site["site"]
        # every camera has to stand on the ground that was fetched for it
        far = max(max(abs(m["e"]), abs(m["n"])) for m in site["members"])
        assert ext >= far, (f"site {site['site']}: ground reaches {ext} m but a camera "
                            f"stands at {far:.1f} m")
        patch = Image.open(io.BytesIO(base64.b64decode(g["img"].split(",", 1)[1])))
        assert patch.width == patch.height, f"site {site['site']}: patch is not square"


def test_depth_panel_covers_the_same_window_shape_as_the_photo(page_data):
    """The two panels claim to be the identical window, so they must share an aspect.

    They are rendered at different pixel widths on purpose — the depth raster is coarse
    and gains nothing from upsampling — but the page lays both out at one CSS width, so
    any aspect disagreement shows up directly as a stretched, taller-than-the-photo
    depth panel next to it.
    """
    from PIL import Image

    def shape(uri):
        return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).size

    for site in page_data["sites"]:
        for m in site["members"]:
            pw, ph = shape(m["img"])
            dw, dh = shape(m["dimg"])
            assert abs((dh / dw) / (ph / pw) - 1) < 0.02, (
                f"site {site['site']}: depth panel {dw}x{dh} does not match the "
                f"photo's {pw}x{ph} aspect")
