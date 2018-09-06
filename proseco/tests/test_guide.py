# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pathlib import Path
import numpy as np
import itertools
from astropy.table import Table
import pytest

import mica.starcheck
from chandra_aca.aca_image import ACAImage, AcaPsfLibrary
from chandra_aca.transform import mag_to_count_rate, count_rate_to_mag
import agasc

from ..guide import (get_guide_catalog, check_spoil_contrib, get_pixmag_for_offset,
                     check_mag_spoilers)
from ..characteristics_guide import mag_spoiler


HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES['data_root']).exists()


def test_select():
    # "random" ra/dec/roll
    selected = get_guide_catalog(att=(10, 20, 3), date='2018:001', dither=(8, 8), t_ccd=-13, n_guide=5)
    expected_star_ids = [156384720, 155980240, 156376184, 156381600, 156379416]
    assert selected['id'].tolist() == expected_star_ids


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_obsid_19461():
    # overall poor star field
    selected = get_guide_catalog(obsid=19461, n_guide=5)
    expected_star_ids = [450103048, 450101704, 394003312, 450109160, 450109016]
    assert selected['id'].tolist() == expected_star_ids


def test_common_column_obsid_19904():
    # Should not select 1091705224 which has a column spoiler
    # Limit the star field to just a handful of stars including the star
    # and the column spoiler
    limited_stars = [1091709256, 1091698696, 1091705224, 1091702440, 1091704824]
    date = '2018:001'
    star_recs = [agasc.get_star(s, date=date) for s in limited_stars]
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    selected = get_guide_catalog(att=(248.515786,   -47.373203,   238.665124),
                                 date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                 stars=stars)
    # Assert the column spoiled one isn't in the list
    assert 1091705224 not in selected['id'].tolist()
    assert selected['id'].tolist() == [1091702440, 1091698696, 1091704824]


def test_box_mag_spoiler():
    # Manipulate a spoiler star in this test to first not be a spoiler
    limited_stars = [688522000, 688523960, 611190016, 139192, 688522008]
    date = '2018:001'
    star_recs = [agasc.get_star(s, date=date) for s in limited_stars]
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    stars['MAG_ACA'][stars['AGASC_ID'] == 688522000] = 16.0
    selected1 = get_guide_catalog(att=(0, 0, 0), date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars)
    # Confirm the 688523960 star is selected as a nominal star in this config
    assert 688523960 in selected1['id']
    # Set the spoiler to be 10th mag and closer to the second star
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    stars['MAG_ACA'][stars['AGASC_ID'] == 688522000] = 10.0
    # Confirm the 688523960 star is not selected if the spoiler is brighter
    selected2 = get_guide_catalog(att=(0, 0, 0), date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars)
    assert 688523960 not in selected2['id']


def test_region_contrib():
    date = '2018:001'
    limited_stars = [425740488, 425864552, 426263240, 425736928, 426255616, 426253528, 426253768]
    star_recs = [agasc.get_star(s, date=date) for s in limited_stars]
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    # Only pass the first 5 stars
    selected1 = get_guide_catalog(att=(8, 47, 0), date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars[0:5])
    assert 426255616 in selected1['id']
    # The last two stars spoil the 5th star via too much light contrib in the region
    # so if we include all the stars, the 5th star should *not* be selected
    selected2 = get_guide_catalog(att=(8, 47, 0), date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars)
    assert 426255616 not in selected2['id']


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_exclude_bad_star():
    # obsid 17896 attitude
    # Will need to find another bad star, as this one is now excluded via VAR
    selected = get_guide_catalog(obsid=6820)
    assert 614606480 not in selected['id']


def test_avoid_trap():
    # Set up a scenario where a star is selected fine at one roll, and then
    # confirm that it is not selected when roll places it on the trap
    limited_stars = [156384720, 156376184, 156381600, 156379416, 156384304]
    date = '2018:001'
    ra1 = 9.769
    dec1 = 20.147
    roll1 = 295.078
    star_recs = [agasc.get_star(s, date=date) for s in limited_stars]
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    selected1 = get_guide_catalog(att=(ra1, dec1, roll1), date=date, t_ccd=-15, dither=(8, 8),
                                  n_guide=5, stars=stars)
    assert selected1['id'].tolist() == limited_stars
    # Roll so that 156381600 is on the trap
    ra2 = 9.769
    dec2 = 20.147
    roll2 = 297.078
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    selected2 = get_guide_catalog(att=(ra2, dec2, roll2), date=date, t_ccd=-15, dither=(8, 8),
                                  n_guide=5, stars=stars)
    assert 156381600 not in selected2['id'].tolist()


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_big_dither():
    # Obsid 20168
    selected = get_guide_catalog(obsid=20168, n_guide=5)
    expected = [977409032,  977414712, 977416336, 977282416, 977405808]
    assert selected['id'].tolist() == expected


def test_check_pixmag_offset():
    APL = AcaPsfLibrary()
    # Then use one star and test with various pixels
    star = {'row': -.5, 'col': .5, 'MAG_ACA': 9.0, 'id': 1}
    # Confirm that when the bad pixel is bright enough to shift the centroid
    # by N arcsecs that the mag/offset code agrees
    rs = range(-4, 4)
    cs = range(-4, 4)
    lims = [0.1, 0.2, .5, 1.0, 5.0]
    pixvals = range(250, 5000, 250)
    for r_dist, c_dist, lim, pixval in itertools.product(rs, cs, lims, pixvals):
        # Get a new star image every time because centroid_fm messes with it in-place
        star_img = APL.get_psf_image(star['row'], star['col'],
                                     norm=mag_to_count_rate(star['MAG_ACA']))
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
        pmag = get_pixmag_for_offset(star['MAG_ACA'], lim)
        if dr > lim:
            assert count_rate_to_mag(pixval) < pmag


def test_check_spoil_contrib():
    # Construct a case where a star spoils the edge of the 8x8
    # Note that for these mock stars, since we we are checking the status of
    # the first star, ASPQ1 needs to be nonzero on that star or the
    # check_spoil_contrib code will bail out before actually doing the check
    star1 = {'row': 0, 'col': 0, 'MAG_ACA': 8.0, 'id': 1, 'ASPQ1': 1}
    star2 = {'row': 0, 'col': -5, 'MAG_ACA': 6.0, 'id': 2, 'ASPQ1': 0}
    stars = Table([star1, star2])
    bg_spoil, reg_spoil = check_spoil_contrib(stars, np.array([True, True]), stars, .05, 25)
    assert reg_spoil[0]
    # Construct a case where a star spoils just a background pixel
    star1 = {'row': 0, 'col': 0, 'MAG_ACA': 8.0, 'id': 1, 'ASPQ1': 1}
    star2 = {'row': -5.5, 'col': -5.5, 'MAG_ACA': 9.5, 'id': 2, 'ASPQ1': 0}
    stars = Table([star1, star2])
    bg_spoil, reg_spoil = check_spoil_contrib(stars, np.array([True, True]), stars, .05, 25)
    assert bg_spoil[0]


def test_check_mag_spoilers():
    # Check that stars that should fail the mag/line test actually fail
    intercept = mag_spoiler['Intercept']
    spoilslope = mag_spoiler['Slope']
    star1 = {'row': 0, 'col': 0, 'MAG_ACA': 9.0, 'mag_err': 0, 'id': 1}
    # The mag spoiler check only works on stars that are within 10 pixels in row
    # or column, so don't bother simulating stars outside that distance
    r_dists = np.arange(-9.25, 10, 1)
    c_dists = np.arange(-9.5, 10, 1)
    magdiffs = np.arange(2, -5, -.5)
    for r_dist, c_dist, magdiff in itertools.product(r_dists, c_dists, magdiffs):
        star2 = {'row': r_dist, 'col': c_dist, 'MAG_ACA': star1['MAG_ACA'] - magdiff,
                 'mag_err': 0, 'id': 2}
        dist = np.sqrt(r_dist ** 2 + c_dist ** 2)
        stars = Table([star1, star2])
        spoiled = check_mag_spoilers(stars, np.array([True, True]), stars, 0)
        req_sep = intercept + magdiff * spoilslope
        assert (dist < req_sep) == spoiled[0]
