# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
from pathlib import Path

import mica.starcheck
import numpy as np
import pytest
from astropy.table import Table
from chandra_aca.aca_image import ACAImage, AcaPsfLibrary
from chandra_aca.transform import count_rate_to_mag, mag_to_count_rate
from Quaternion import Quat

from ..characteristics import CCD
from ..characteristics_guide import mag_spoiler
from ..core import StarsTable
from ..guide import (
    GUIDE,
    GuideTable,
    check_column_spoilers,
    check_mag_spoilers,
    check_spoil_contrib,
    get_ax_range,
    get_guide_catalog,
    get_pixmag_for_offset,
    run_cluster_checks,
)
from ..report_guide import make_report
from .test_common import DARK40, OBS_INFO, STD_INFO, mod_std_info

HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES["data_root"]).exists()


def test_select(proseco_agasc_1p7):
    """
    Regression test that 5 expected agasc ids are selected at an arbitrary ra/dec/roll .
    """
    selected = get_guide_catalog(
        att=(10, 20, 3), date="2018:001", dither=(8, 8), t_ccd=-13, n_guide=5
    )
    expected_star_ids = [156384720, 155980240, 156376184, 156381600, 156379416]
    assert selected["id"].tolist() == expected_star_ids


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason="Test requires starcheck archive")
def test_obsid_19461(proseco_agasc_1p7):
    """
    Regression tests that 5 expected agasc ids are selected in a poor star field
    corresponding to obsid 19461.
    """
    selected = get_guide_catalog(obsid=19461, n_guide=5)
    expected_star_ids = [450103048, 450101704, 394003312, 450109160, 450109016]
    assert selected["id"].tolist() == expected_star_ids


def test_common_column_obsid_19904():
    """
    Confirm in a specific configuration that a star with a column spoiler,
    1091705224, is not selected.  This test limits the star field
    star field to just 5 stars including the star and the column spoiler
    """
    att = (248.515786, -47.373203, 238.665124)
    agasc_ids = [1091709256, 1091698696, 1091705224, 1091702440, 1091704824]
    date = "2018:001"
    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    selected = get_guide_catalog(
        att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5, stars=stars
    )
    # Assert the column spoiled one isn't in the list
    assert 1091705224 not in selected["id"].tolist()
    assert selected["id"].tolist() == [1091702440, 1091698696, 1091704824]


# Cases for common column spoiler test
comm_cases = [
    # Star id 2 between id 1 and readout register and 5 mags brighter
    {"r1": -1, "c1": 0, "m1": 10.0, "r2": -5, "c2": 0, "m2": 5.0, "spoils": True},
    # Star id 2 between id 1 and readout register and 4 mags brighter (too faint to spoil)
    {"r1": -1, "c1": 0, "m1": 10.0, "r2": -5, "c2": 0, "m2": 6.0, "spoils": False},
    # Star id 2 between id 1 and readout register and 5 mags brighter
    {"r1": 10, "c1": 15, "m1": 9.0, "r2": 50, "c2": 6, "m2": 4.0, "spoils": True},
    # Star id 2 between id 1 and readout register and 5 mags brighter but 11 pix away in col
    {"r1": 10, "c1": 15, "m1": 9.0, "r2": 50, "c2": 4, "m2": 4.0, "spoils": False},
]


@pytest.mark.parametrize("case", comm_cases)
def test_common_column(case):
    """
    Test check_column_spoilers method using constructed two-star cases.
    The check_column_spoilers method uses n_sigma on MAG_ACA_ERR, but this test
    ignores MAG_ACA_ERR.
    """
    stars = StarsTable.empty()
    stars.add_fake_star(row=case["r1"], col=case["c1"], mag=case["m1"], id=1)
    stars.add_fake_star(row=case["r2"], col=case["c2"], mag=case["m2"], id=2)
    stars["offchip"] = False
    col_spoil, col_rej = check_column_spoilers(stars, [True, True], stars, n_sigma=3)
    assert col_spoil[0] == case["spoils"]


def test_box_mag_spoiler():
    """
    Test spoiled star rejection by manipulating a star position and magnitude
    to make it spoil another star in a specific attitude/config.
    """
    att = (0, 0, 0)
    agasc_ids = [688522000, 688523960, 611190016, 139192, 688522008]
    date = "2018:001"
    stars = StarsTable.from_agasc_ids(att, agasc_ids)

    stars.get_id(688522000)["mag"] = 16.0
    selected1 = get_guide_catalog(
        att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5, stars=stars
    )

    # Confirm the 688523960 star is selected as a nominal star in this config
    assert 688523960 in selected1["id"]

    # Set the spoiler to be 10th mag and closer to the second star
    stars.get_id(688522000)["mag"] = 9
    stars.get_id(688522000)["row"] = 50

    # Confirm the 688523960 star is not selected if the spoiler is brighter
    selected2 = get_guide_catalog(
        att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5, stars=stars
    )
    assert 688523960 not in selected2["id"]


def test_region_contrib(proseco_agasc_1p7):
    """Regression test of stars rejected by contributing starlight to readout region.

    Scenario test for too much light contribution by a spoiler star unto the
    region of a candidate star... by excluding and then including the spoiler
    star amongst the stars in the star field and confirming that the candidate
    is not selected when the spoiler star is included.

    """
    att = (8, 47, 0)
    date = "2018:001"
    agasc_ids = [
        425740488,
        425864552,
        426263240,
        425736928,
        426255616,
        426253528,
        426253768,
    ]
    stars = StarsTable.from_agasc_ids(att, agasc_ids)

    # Only pass the first 5 stars
    selected1 = get_guide_catalog(
        att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5, stars=stars[0:5]
    )
    assert 426255616 in selected1["id"]

    # The last two stars spoil the 5th star via too much light contrib in the region
    # so if we include all the stars, the 5th star should *not* be selected
    selected2 = get_guide_catalog(
        att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5, stars=stars
    )
    assert 426255616 not in selected2["id"]


def test_bad_star_list():
    """Test that a star with ID in the bad star list is not selected."""
    bad_id = 39980640
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=np.linspace(9, 10.3, 4), n_stars=4)
    # Bright star that would normally be selected
    stars.add_fake_star(yang=100, zang=100, mag=6.5, id=bad_id)
    kwargs = mod_std_info(stars=stars, dark=dark, n_guide=5)
    guides = get_guide_catalog(**kwargs)
    assert bad_id not in guides["id"]

    idx = guides.stars.get_id_idx(bad_id)
    assert guides.bad_stars_mask[idx]


def test_color1_0p7():
    """Test that a star with COLOR1=0.7 is not selected"""
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=np.linspace(9, 10.3, 4), n_stars=4)
    # Bright star that would normally be selected but with color=0.7
    bad_id = 11111111
    stars.add_fake_star(yang=100, zang=100, mag=6.5, id=bad_id, COLOR1=0.7)
    kwargs = mod_std_info(stars=stars, dark=dark, n_guide=5)
    guides = get_guide_catalog(**kwargs)
    assert bad_id not in guides["id"]

    idx = guides.stars.get_id_idx(bad_id)
    assert guides.bad_stars_mask[idx]


def test_avoid_trap():
    """
    Set up a scenario where a star is selected fine at one roll, and then
    confirm that it is not selected when roll places it on the trap.
    """
    agasc_ids = [156384720, 156376184, 156381600, 156379416, 156384304]
    date = "2018:001"
    ra = 9.769
    dec = 20.147
    roll1 = 295.078
    att = (ra, dec, roll1)

    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    selected1 = get_guide_catalog(
        att=att, date=date, t_ccd=-15, dither=(8, 8), n_guide=5, stars=stars
    )
    assert selected1["id"].tolist() == agasc_ids

    # Roll so that 156381600 is on the trap
    roll2 = 297.078
    att = (ra, dec, roll2)

    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    selected2 = get_guide_catalog(
        att=att, date=date, t_ccd=-15, dither=(8, 8), n_guide=5, stars=stars
    )
    assert 156381600 not in selected2["id"].tolist()


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason="Test requires starcheck archive")
def test_big_dither():
    """Regression test that the expected set of agasc ids selected for "big
    dither" obsid 20168 are selected.

    """
    selected = get_guide_catalog(obsid=20168, n_guide=5)
    expected = [977409032, 977930352, 977414712, 977416336, 977405808]
    assert selected["id"].tolist() == expected


def test_check_pixmag_offset():
    """Test the get_pixmag_for_offset guide function.

    get_pixmag_for_offset returns the magnitude required for an individual
    pixel to spoil the centroid of a candidate star by an specified offset.
    This test uses a range of pixel locations and intensities, and confirms
    that for any pixel that would cause an offset in the centroid position over
    the threshold given that the pixel value would be over the magnitude given
    by get_pixmag_for_offset.

    """
    APL = AcaPsfLibrary()

    # Then use one star and test with various pixels
    star = {"row": -0.5, "col": 0.5, "mag": 9.0, "id": 1}

    # Confirm that when the bad pixel is bright enough to shift the centroid
    # by N arcsecs that the mag/offset code agrees
    rs = range(-4, 4)
    cs = range(-4, 4)
    lims = [0.1, 0.2, 0.5, 1.0, 5.0]
    pixvals = range(250, 5000, 250)

    for r_dist, c_dist, lim, pixval in itertools.product(rs, cs, lims, pixvals):
        # Get a new star image every time because centroid_fm messes with it in-place
        star_img = APL.get_psf_image(
            star["row"], star["col"], norm=mag_to_count_rate(star["mag"])
        )
        pix = ACAImage(np.zeros((8, 8)), row0=-4, col0=-4)

        # Get the the non-spoiled centroid
        cr1, cc1, n = star_img.centroid_fm()
        pix.aca[r_dist, c_dist] = pixval
        star_img = star_img + pix.aca

        # Spoil with a pixel and get tne new centroid
        cr2, cc2, n = star_img.centroid_fm()

        # Get the offset in pixel space
        dr = np.sqrt(((cr1 - cr2) ** 2) + ((cc1 - cc2) ** 2))

        # Check that if it is spoiled, it would have been spoiled with the tool
        pmag = get_pixmag_for_offset(star["mag"], lim)
        if dr > lim:
            assert count_rate_to_mag(pixval) < pmag


def test_check_spoil_contrib():
    """
    Construct a case where a star spoils the edge of the 8x8 (edge and then background pixel).

    Note that for these mock stars, since we we are checking the status of
    the first star, ASPQ1 needs to be nonzero on that star or the
    check_spoil_contrib code will bail out before actually doing the check
    """
    stars = StarsTable.empty()
    stars.add_fake_star(row=0, col=0, mag=8.0, id=1, ASPQ1=1)
    stars.add_fake_star(row=0, col=-5, mag=6.0, id=2, ASPQ1=0)
    bg_spoil, reg_spoil, rej = check_spoil_contrib(
        stars, np.array([True, True]), stars, 0.05, 25
    )
    assert reg_spoil[0]

    # Construct a case where a star spoils just a background pixel
    stars = StarsTable.empty()
    stars.add_fake_star(row=0, col=0, mag=8.0, id=1, ASPQ1=1)
    stars.add_fake_star(row=-5.5, col=-5.5, mag=9.5, id=2, ASPQ1=0)

    bg_spoil, reg_spoil, rej = check_spoil_contrib(
        stars, np.array([True, True]), stars, 0.05, 25
    )
    assert bg_spoil[0]


def test_check_spoiler_cases():
    """
    Regression test guide star selection against a star and a spoiler

    This moves a spoiling star from center past and edge then moves a spoiling star diagonally.
    This should hit check_spoil_contrib, has_spoiler_in_box, and check_mag_spoilers tests in
    the guide star selection.

    """
    drcs = np.arange(0, 13, 2)
    mag0 = 8.0  # Baseline mag
    dmags = [0, 3, 7]  # Spoiler delta mag
    # Use a blank dark map to skip imposter checks
    dark = ACAImage(np.zeros((1024, 1024)), row0=-512, col0=-512)
    spoiled = []
    for dmag in dmags:
        for drc in drcs:
            r = 10
            c = 10
            stars = StarsTable.empty()
            stars.add_fake_star(row=r, col=c, mag=mag0, id=1, ASPQ1=1)
            # Add a "spoiling" star and move it from center past edge through
            # the drcs.  The spoiling star is set with CLASS=1 so it is also not a
            # selectable guide star.
            stars.add_fake_star(
                row=r + drc, col=c, mag=mag0 + dmag, id=2, ASPQ1=0, CLASS=1
            )
            selected = get_guide_catalog(**STD_INFO, stars=stars, dark=dark)
            # Is the id=1 star spoiled / not selected?
            spoiled.append(1 if (1 not in selected["id"]) else 0)
    spoiled = np.array(spoiled).reshape(-1, len(drcs)).tolist()
    #                    0  2  4  6  8 10 12 pixels
    expected_spoiled = [
        [1, 1, 1, 1, 1, 0, 0],  # dmag = 0
        [1, 1, 1, 1, 0, 0, 0],  # dmag = 3
        [1, 1, 1, 1, 0, 0, 0],
    ]  # dmag = 7
    assert spoiled == expected_spoiled

    spoiled = []
    dmags = [3, 5, 7]  # Spoiler delta mag
    for dmag in dmags:
        for drc in drcs:
            r = 10
            c = 10
            stars = StarsTable.empty()
            stars.add_fake_star(row=r, col=c, mag=mag0, id=1, ASPQ1=1)
            # Add a "spoiling" star 5 mags fainter and move it from center out through a corner
            # The spoiling star is set with CLASS=1 so it is also not a selectable guide star.
            stars.add_fake_star(
                row=r + drc, col=c + drc, mag=mag0 + dmag, id=2, ASPQ1=0, CLASS=1
            )
            selected = get_guide_catalog(**STD_INFO, stars=stars, dark=dark)
            spoiled.append(1 if (1 not in selected["id"]) else 0)
    spoiled = np.array(spoiled).reshape(-1, len(drcs)).tolist()
    #                    0  2  4  6  8 10 12 pixels
    expected_spoiled = [
        [1, 1, 1, 1, 0, 0, 0],  # dmag = 3
        [1, 1, 1, 1, 0, 0, 0],  # dmag = 5
        [1, 1, 1, 1, 0, 0, 0],
    ]  # dmag = 7
    assert spoiled == expected_spoiled


def test_overlap_spoiler():
    """
    Add a brighter "spoiler" that is a good guide star and confirm the test for
    overlapping selected stars works until just past (12 pixels).
    """

    # Use a blank dark map to skip imposter checks
    dark = ACAImage(np.zeros((1024, 1024)), row0=-512, col0=-512)

    spoiled = []
    drcs = np.arange(6, 17, 2)
    for drc in drcs:
        r = 10
        c = 10
        stars = StarsTable.empty()
        stars.add_fake_star(row=r, col=c, mag=9, id=1, ASPQ1=0)
        # Add a brighter spoiling star
        stars.add_fake_star(row=r, col=c + drc, mag=7, id=2, ASPQ1=0)
        selected = get_guide_catalog(**STD_INFO, stars=stars, dark=dark)
        spoiled.append(1 if (1 not in selected["id"]) else 0)
    #                   6  8 10 12 14 16  pixels
    expected_spoiled = [1, 1, 1, 1, 0, 0]
    assert spoiled == expected_spoiled


def test_overlap_spoiler_include():
    """
    Add test for overlap-star handling in cases where one or both
    stars is in includes_ids_guide.
    """
    stars = StarsTable.empty()

    # First, set this up with a constellation and 1 manual bright star
    stars.add_fake_constellation(n_stars=7, mag=9)
    stars.add_fake_star(id=1, mag=8, row=50, col=-50)
    aca1 = get_guide_catalog(
        **mod_std_info(n_fid=0, n_guide=8), obsid=40000, stars=stars, dark=DARK40
    )

    # The bright star should be included
    assert 1 in aca1["id"]

    # Add another bright star within 10 pixels of id 1
    stars.add_fake_star(id=2, mag=8.5, row=60, col=-50)
    aca2 = get_guide_catalog(
        **mod_std_info(n_fid=0, n_guide=8), obsid=40000, stars=stars, dark=DARK40
    )

    # The id 2 star is a (within 12 pixels) overlap spoiler and fainter
    # so should not be selected
    assert 2 not in aca2["id"]
    assert 1 in aca2["id"]

    # Force include the fainter star (id 2) and 1 should not be selected
    aca3 = get_guide_catalog(
        **mod_std_info(n_fid=0, n_guide=8),
        obsid=40000,
        stars=stars,
        dark=DARK40,
        include_ids_guide=[2],
    )
    assert 2 in aca3["id"]
    assert 1 not in aca3["id"]

    # Force include them both and they should still be selected.
    aca4 = get_guide_catalog(
        **mod_std_info(n_fid=0, n_guide=8),
        obsid=40000,
        stars=stars,
        dark=DARK40,
        include_ids_guide=[1, 2],
    )
    assert 2 in aca4["id"]
    assert 1 in aca4["id"]


pix_cases = [
    {"dither": (8, 8), "offset_row": 4, "offset_col": 4, "spoils": True},
    {"dither": (64, 8), "offset_row": 16, "offset_col": 0, "spoils": True},
    {"dither": (64, 8), "offset_row": 20, "offset_col": 0, "spoils": False},
    {"dither": (64, 8), "offset_row": 0, "offset_col": 16, "spoils": False},
    {"dither": (8, 64), "offset_row": 0, "offset_col": 16, "spoils": True},
    {"dither": (8, 64), "offset_row": 0, "offset_col": 20, "spoils": False},
]


@pytest.mark.parametrize("case", pix_cases)
def test_pix_spoiler(case):
    """
    Check that for various dither configurations, a hot pixel near a star will
    result in that star not being selected.
    """
    stars = StarsTable.empty()
    stars.add_fake_star(row=0, col=0, mag=7.0, id=1, ASPQ1=0)
    stars.add_fake_constellation(n_stars=4)
    dark = np.zeros((1024, 1024))
    pix_config = {
        "att": (0, 0, 0),
        "date": "2018:001",
        "t_ccd": -10,
        "n_guide": 5,
        "stars": stars,
    }
    # Use the "case" to try to spoil the first star with a bad pixel
    dark[
        case["offset_row"] + int(stars[0]["row"]) + 512,
        case["offset_col"] + int(stars[0]["col"]) + 512,
    ] = mag_to_count_rate(stars[0]["mag"])
    selected = get_guide_catalog(**pix_config, dither=case["dither"], dark=dark)
    assert (1 not in selected["id"]) == case["spoils"]


def test_check_mag_spoilers():
    """
    Check that spoiling stars that should spoil a candidated due to the
    mag/line test actually spoil the candidate star.

    The check_mag_spoilers function sets a star to spoil another star if it
    is closer than a required separation for the magnitude difference
    (a faint star can be closer to a candidate star without spoiling it).
    The line test is defined in the mag_spoiler parameters Intercept and Slope.
    """
    intercept = mag_spoiler["Intercept"]
    spoilslope = mag_spoiler["Slope"]
    star1 = {"row": 0, "col": 0, "mag": 9.0, "MAG_ACA_ERR": 0, "id": 1}

    # The mag spoiler check only works on stars that are within 10 pixels in row
    # or column, so don't bother simulating stars outside that distance
    r_dists = np.arange(-9.25, 10, 2)
    c_dists = np.arange(-9.5, 10, 2)
    magdiffs = np.arange(2, -5, -0.5)

    for r_dist, c_dist, magdiff in itertools.product(r_dists, c_dists, magdiffs):
        star2 = {
            "row": r_dist,
            "col": c_dist,
            "mag": star1["mag"] - magdiff,
            "MAG_ACA_ERR": 0,
            "id": 2,
        }
        dist = np.sqrt(r_dist**2 + c_dist**2)
        stars = Table([star1, star2])
        spoiled, rej = check_mag_spoilers(stars, np.array([True, True]), stars, 0)
        req_sep = intercept + magdiff * spoilslope
        assert (dist < req_sep) == spoiled[0]


def test_warnings():
    """
    Test that the ACACatalogTable warnings infrastructure works via a
    specific expected warning in get_guide_catalog (too few stars selected).
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=6)
    guides = get_guide_catalog(
        att=(0, 0, 0), date="2018:001", t_ccd=-10, dither=(8, 8), stars=stars, n_guide=8
    )
    assert guides.warnings == [
        "WARNING: Selected only 6 guide stars versus requested 8"
    ]


def make_fake_guides(rows, cols, mags, ids):
    stars = StarsTable.empty()
    for r, c, m, i in zip(rows, cols, mags, ids):
        stars.add_fake_star(row=r, col=c, mag=m, id=i)
    return stars


def test_select_catalog_prefers_best_cluster():
    # Make 5 stars, 3 close together, 2 far apart
    rows = [0, 0, 0, 300, -300]
    cols = [0, 20, -20, 0, 0]
    mags = [7, 7, 7, 7, 7]
    ids = [1, 2, 3, 4, 5]
    stars = make_fake_guides(rows, cols, mags, ids)
    guides = GuideTable()
    guides.n_guide = 3
    guides.cand_guides = stars
    guides.stars = stars
    guides.jupiter = None  # No Jupiter for this test

    # Should select the 3 most separated stars (not all clustered)
    selected = guides.select_catalog(stars)
    # The selected IDs should include the far-apart stars
    assert set(selected["id"]) == set([1, 4, 5]) or set(selected["id"]) == set(
        [3, 4, 5]
    )


def test_select_catalog_honors_include_ids():
    # Make 4 stars, force include one that would otherwise fail cluster check
    rows = [0, 0, 300, -300]
    cols = [0, 20, 0, 0]
    mags = [7, 7, 7, 7]
    ids = [1, 2, 3, 4]
    stars = make_fake_guides(rows, cols, mags, ids)
    guides = GuideTable()
    guides.n_guide = 3
    guides.cand_guides = stars
    guides.stars = stars
    guides.jupiter = None

    selected1 = guides.select_catalog(stars)
    # The id 2 star should not be present
    assert 2 not in selected1["id"]

    guides.include_ids = [2]
    selected2 = guides.select_catalog(stars)
    # The forced-included star should be present
    assert 2 in selected2["id"]


def test_select_catalog_jupiter_weighted():
    # Make 3 stars, Jupiter present, only one combo passes jupiter check
    # These stars are not spoiled directly by Jupiter.
    rows = [-20, 300, -300]
    cols = [100, -50, -100]
    mags = [7, 7, 7]
    ids = [1, 2, 3]
    stars = make_fake_guides(rows, cols, mags, ids)
    guides = GuideTable()
    guides.n_guide = 2
    guides.cand_guides = stars
    guides.stars = stars

    selected1 = guides.select_catalog(stars)
    assert set(selected1["id"]) == set([2, 3])

    # Simulate Jupiter data so only stars on opposite sides pass
    guides.jupiter = Table([{"row": [100], "col": [0]}])
    selected2 = guides.select_catalog(stars)
    # Should select the combo that passes the Jupiter check
    assert set(selected2["id"]) == set([1, 3])


def test_select_catalog_fallback():
    # No combination passes cluster or Jupiter check
    rows = [0, 0, 0]
    cols = [0, 1, -1]
    mags = [7, 7, 7]
    ids = [1, 2, 3]
    stars = make_fake_guides(rows, cols, mags, ids)
    guides = GuideTable()
    guides.n_guide = 2
    guides.cand_guides = stars
    guides.stars = stars
    guides.jupiter = None

    ## Patch cluster check to always fail
    # import proseco.guide
    # proseco.guide.run_cluster_checks = lambda cands: [False, False, False]

    selected = guides.select_catalog(stars)
    # Should return the first available set
    assert np.all(selected["id"] == [1, 2])

    # And there should be a 0 score log entry
    assert (
        guides.log_info["events"][-1]["data"] == "Selected stars with weighted score 0"
    )

    # And that this still works if there are no "combinations to check"
    guides.n_guide = 3
    selected3 = guides.select_catalog(stars)
    assert np.all(selected3["id"] == [1, 2, 3])
    assert (
        guides.log_info["events"][-1]["data"] == "Selected stars with weighted score 0"
    )
    assert guides.log_info["events"][-1]["tried_combinations"] == 1

    # If we have more force-included stars than n_guide, we should hit the
    # full fallback text as no combination will be evaluated.
    guides.n_guide = 2
    guides.include_ids = [1, 2, 3]
    selected3 = guides.select_catalog(stars)
    assert np.all(selected3["id"] == [1, 2])
    assert guides.log_info["events"][-1]["data"] == (
        "WARNING: No combination satisfied any checks, returning first available set"
    )
    assert guides.log_info["events"][-1]["tried_combinations"] == 0


def test_guides_include_exclude():
    """
    Test include and exclude stars for guide.  This uses a catalog with 11 stars:
    - 8 bright stars from 7.0 to 7.7 mag, where the 7.0 is EXCLUDED
    - 2 faint (but OK) stars 10.0, 10.1 where the 10.0 is INCLUDED
    - 1 very faint (bad) stars 12.0 mag is INCLUDED

    Both the 7.0 and 10.1 would normally get picked either initially
    or swapped in during optimization, and 12.0 would never get picked.
    """
    stars = StarsTable.empty()

    stars.add_fake_constellation(
        mag=[7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7],
        id=[1, 2, 3, 4, 5, 6, 7, 8],
        size=2000,
        n_stars=8,
    )
    stars.add_fake_constellation(
        mag=[10.0, 10.1, 12.0], id=[9, 10, 11], size=1500, n_stars=3
    )

    # Make sure baseline catalog is working like expected
    std_info = STD_INFO.copy()
    std_info.update(n_guide=8)
    guides = get_guide_catalog(**std_info, stars=stars)
    assert np.all(guides["id"] == np.arange(1, 9))

    # Define includes and excludes.
    include_ids = [9, 11]
    exclude_ids = [1]

    guides = get_guide_catalog(
        **std_info, stars=stars, include_ids=include_ids, exclude_ids=exclude_ids
    )

    assert guides.include_ids == include_ids
    assert guides.exclude_ids == exclude_ids

    assert all(id_ in guides.cand_guides["id"] for id_ in include_ids)

    assert all(id_ in guides["id"] for id_ in include_ids)
    assert all(id_ not in guides["id"] for id_ in exclude_ids)

    assert np.all(guides["id"] == [9, 11, 2, 3, 4, 5, 6, 7])
    assert np.allclose(guides["mag"], [10.0, 12.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6])


dither_cases = [(8, 8), (64, 8), (8, 64), (20, 20), (30, 20)]


def test_guides_include_bad():
    """
    Test include stars for guide where star is bad for some reason.

    - Including a class=1 star on the CCD is allowed.
    - Including a star (otherwise acceptable) just off the CCD is not allowed.

    """
    row_max = CCD["row_max"] - CCD["row_pad"] - CCD["window_pad"]
    col_max = CCD["col_max"] - CCD["col_pad"] - CCD["window_pad"]

    stars = StarsTable.empty()

    stars.add_fake_constellation(
        mag=[7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7],
        id=[1, 2, 3, 4, 5, 6, 7, 8],
        size=2000,
        n_stars=8,
    )

    # Bright star but class 1, not picked
    stars.add_fake_star(mag=6.5, row=row_max - 1, col=col_max - 1, id=10, CLASS=1)

    # Bright star just off the FOV, not picked
    stars.add_fake_star(mag=6.5, row=row_max + 1, col=col_max + 1, id=20)

    # Make sure baseline catalog is working like expected
    guides = get_guide_catalog(**STD_INFO, stars=stars)
    assert np.all(guides["id"] == [1, 2, 3, 4, 5])

    # Picking the class=1 star is fine
    guides = get_guide_catalog(**STD_INFO, stars=stars, include_ids=10)
    assert np.all(sorted(guides["id"]) == [1, 2, 3, 4, 10])

    # Picking the star off the CCD generates an exception
    with pytest.raises(ValueError) as err:
        get_guide_catalog(**STD_INFO, stars=stars, include_ids=20)
    assert "cannot include star id=20" in str(err)


def test_guides_include_close():
    """
    Test force include stars where they would not be selected due to
    clustering.
    """
    stars = StarsTable.empty()

    stars.add_fake_constellation(
        mag=[7.0, 7.0, 7.0, 7.0, 7.0], id=[25, 26, 27, 28, 29], size=2000, n_stars=5
    )

    stars.add_fake_star(mag=11.0, yang=100, zang=100, id=21)
    stars.add_fake_star(mag=11.0, yang=-100, zang=-100, id=22)
    stars.add_fake_star(mag=11.0, yang=100, zang=-100, id=23)
    stars.add_fake_star(mag=11.0, yang=-100, zang=100, id=24)

    cat1 = get_guide_catalog(**mod_std_info(n_guide=5), stars=stars)

    cluster_check_sum = np.sum(run_cluster_checks(cat1))
    assert cluster_check_sum == 3

    # Confirm that only bright stars are used
    assert np.count_nonzero(cat1["mag"] == 7.0) == 5

    # Force include the faint 4 stars that are also close together
    include_ids = [21, 22, 23, 24]
    cat2 = get_guide_catalog(
        **mod_std_info(n_guide=5),
        stars=stars,
        include_ids_guide=include_ids,
    )

    # Run the cluster checks and confirm all 3 fail
    cluster_check_sum = np.sum(run_cluster_checks(cat2))
    assert cluster_check_sum == 0
    assert np.all(np.in1d(include_ids, cat2["id"]))
    # And confirm that only one of the bright stars is used
    assert np.count_nonzero(cat2["mag"] == 7.0) == 1


@pytest.mark.parametrize("dither", dither_cases)
def test_edge_star(dither):
    """
    Add stars right at row and col max for various dithers.
    This test both confirms that the dark map extraction doesn't break and that
    the stars can still be selected.
    """
    stars = StarsTable.empty()

    stars.add_fake_constellation(
        mag=[7.0, 7.1, 7.2, 7.3], id=[1, 2, 3, 4], size=2000, n_stars=4
    )

    # Add stars exactly at 4 corners of allowed "in bounds" area for this dither
    row_dither = dither[0] / 5.0
    col_dither = dither[1] / 5.0
    row_max = CCD["row_max"] - (
        CCD["row_pad"] + CCD["window_pad"] + CCD["guide_extra_pad"] + row_dither
    )
    col_max = CCD["col_max"] - (
        CCD["col_pad"] + CCD["window_pad"] + CCD["guide_extra_pad"] + col_dither
    )
    stars.add_fake_star(row=row_max, col=col_max, mag=6.0)
    stars.add_fake_star(row=row_max * -1, col=col_max, mag=6.0)
    stars.add_fake_star(row=row_max * -1, col=col_max * -1, mag=6.0)
    stars.add_fake_star(row=row_max, col=col_max * -1, mag=6.0)
    info = mod_std_info(
        n_guide=8, dither_guide=(row_dither * 5, col_dither * 5), stars=stars
    )
    guides = get_guide_catalog(**info)
    # Confirm 4 generic stars plus four corner stars are selected
    assert len(guides) == 8

    # Do the same as above, but this time don't include the guide edge pad, so
    # the edge stars will be off the edge
    stars1 = StarsTable.empty()
    stars1.add_fake_constellation(
        mag=[7.0, 7.1, 7.2, 7.3], id=[1, 2, 3, 4], size=2000, n_stars=4
    )
    row_max = CCD["row_max"] - (CCD["row_pad"] + CCD["window_pad"] + row_dither)
    col_max = CCD["col_max"] - (CCD["col_pad"] + CCD["window_pad"] + col_dither)
    stars1.add_fake_star(row=row_max, col=col_max, mag=6.0)
    stars1.add_fake_star(row=row_max * -1, col=col_max, mag=6.0)
    stars1.add_fake_star(row=row_max * -1, col=col_max * -1, mag=6.0)
    stars1.add_fake_star(row=row_max, col=col_max * -1, mag=6.0)
    info = mod_std_info(
        n_guide=8, dither_guide=(row_dither * 5, col_dither * 5), stars=stars1
    )
    guides = get_guide_catalog(**info)
    # Confirm 4 generic stars are the only stars that can be used (edge stars are off the edges)
    assert len(guides) == 4


def test_get_ax_range():
    """
    Confirm that the ranges from get_ax_range are reasonable for a variety of
    center pixel locations and extents (extent = 4 + pix_dither)
    """
    ns = [0, 0.71, 495.3, -200.2]
    extents = [4.0, 5.6, 4.8, 9.0]
    for n, extent in itertools.product(ns, extents):
        minus, plus = get_ax_range(n, extent)
        # Confirm range divisable by 2
        assert (plus - minus) % 2 == 0
        # Confirm return order
        assert plus > minus
        # Confirm the range contains the full extent
        assert n + extent <= plus
        assert n - extent >= minus
        # Confirm the range does not contain more than 2 pix extra on either side
        assert n + extent + 2 > plus
        assert n - extent - 2 < minus


def test_make_report_guide(tmpdir):
    """
    Test making a guide report.  Use a big-box dither here
    to test handling of that in report (after passing through pickle).
    """
    obsid = 19387
    kwargs = OBS_INFO[obsid].copy()
    kwargs["dither"] = (8, 64)
    guides = get_guide_catalog(**OBS_INFO[obsid])

    tmpdir = Path(tmpdir)
    obsdir = tmpdir / f"obs{obsid:05}"
    outdir = obsdir / "guide"

    guides.to_pickle(rootdir=tmpdir)

    guides2 = make_report(obsid, rootdir=tmpdir)

    assert (outdir / "index.html").exists()
    assert len(list(outdir.glob("*.png"))) > 0

    assert repr(guides) == repr(guides2)
    assert repr(guides.cand_guides) == repr(guides2.cand_guides)
    for event, event2 in zip(guides.log_info, guides2.log_info):
        assert event == event2

    for attr in ["att", "date", "t_ccd", "man_angle", "dither"]:
        val = getattr(guides, attr)
        val2 = getattr(guides2, attr)
        if isinstance(val, float):
            assert np.isclose(val, val2)
        elif isinstance(val, Quat):
            assert np.allclose(val.q, val2.q)
        else:
            assert val == val2


def test_guide_faint_mag_limit():
    stars = StarsTable.empty()
    ids = np.arange(1, 6)

    # 4 bright stars + one star just slightly brighter than the nominal faint mag limit
    stars.add_fake_constellation(
        mag=[7.0] * 4 + [GUIDE.ref_faint_mag - 0.001], n_stars=5, id=ids
    )

    # Select stars at 0.1 degC colder than reference temperature, use previous default of
    # dyn_bgd_n_faint=0 for this test, expect 5 stars selected
    guides = get_guide_catalog(
        **mod_std_info(t_ccd=GUIDE.ref_faint_mag_t_ccd - 0.1, dyn_bgd_n_faint=0),
        stars=stars,
        dark=DARK40,
    )
    assert np.all(guides["id"] == ids)

    # Select stars at 0.1 degC warmer than reference temperature, expect 4 stars selected
    guides = get_guide_catalog(
        **mod_std_info(t_ccd=GUIDE.ref_faint_mag_t_ccd + 0.1, dyn_bgd_n_faint=0),
        stars=stars,
        dark=DARK40,
    )
    assert np.all(guides["id"] == [1, 2, 3, 4])
