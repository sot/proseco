# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pathlib import Path
import numpy as np
import itertools
from astropy.table import Table
import pytest

import mica.starcheck
from chandra_aca.aca_image import ACAImage, AcaPsfLibrary
from chandra_aca.transform import mag_to_count_rate, count_rate_to_mag

from ..guide import (get_guide_catalog, check_spoil_contrib, get_pixmag_for_offset,
                     check_mag_spoilers, get_ax_range)
from ..characteristics_guide import mag_spoiler, CCD
from ..core import StarsTable
from .test_common import STD_INFO, mod_std_info


HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES['data_root']).exists()


def test_select():
    """
    Arbitrary ra/dec/roll
    """
    selected = get_guide_catalog(att=(10, 20, 3), date='2018:001', dither=(8, 8),
                                 t_ccd=-13, n_guide=5)
    expected_star_ids = [156384720, 155980240, 156376184, 156381600, 156379416]
    assert selected['id'].tolist() == expected_star_ids


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_obsid_19461():
    """
    Overall poor star field
    """
    selected = get_guide_catalog(obsid=19461, n_guide=5)
    expected_star_ids = [450103048, 450101704, 394003312, 450109160, 450109016]
    assert selected['id'].tolist() == expected_star_ids


def test_common_column_obsid_19904():
    """
    Should not select 1091705224 which has a column spoiler
    Limit the star field to just a handful of stars including the star
    and the column spoiler
    """
    att = (248.515786, -47.373203, 238.665124)
    agasc_ids = [1091709256, 1091698696, 1091705224, 1091702440, 1091704824]
    date = '2018:001'
    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    selected = get_guide_catalog(att=att,
                                 date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                 stars=stars)
    # Assert the column spoiled one isn't in the list
    assert 1091705224 not in selected['id'].tolist()
    assert selected['id'].tolist() == [1091702440, 1091698696, 1091704824]


def test_box_mag_spoiler():
    # Manipulate a spoiler star in this test to first not be a spoiler
    att = (0, 0, 0)
    agasc_ids = [688522000, 688523960, 611190016, 139192, 688522008]
    date = '2018:001'
    stars = StarsTable.from_agasc_ids(att, agasc_ids)

    stars.get_id(688522000)['mag'] = 16.0
    selected1 = get_guide_catalog(att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars)

    # Confirm the 688523960 star is selected as a nominal star in this config
    assert 688523960 in selected1['id']

    # Set the spoiler to be 10th mag and closer to the second star
    stars.get_id(688522000)['mag'] = 9
    stars.get_id(688522000)['row'] = 50

    # Confirm the 688523960 star is not selected if the spoiler is brighter
    selected2 = get_guide_catalog(att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars)
    assert 688523960 not in selected2['id']


def test_region_contrib():
    att = (8, 47, 0)
    date = '2018:001'
    agasc_ids = [425740488, 425864552, 426263240, 425736928, 426255616, 426253528, 426253768]
    stars = StarsTable.from_agasc_ids(att, agasc_ids)

    # Only pass the first 5 stars
    selected1 = get_guide_catalog(att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars[0:5])
    assert 426255616 in selected1['id']

    # The last two stars spoil the 5th star via too much light contrib in the region
    # so if we include all the stars, the 5th star should *not* be selected
    selected2 = get_guide_catalog(att=att, date=date, t_ccd=-20, dither=(8, 8), n_guide=5,
                                  stars=stars)
    assert 426255616 not in selected2['id']


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_exclude_bad_star():
    """
    Obsid 17896 attitude
    Will need to find another bad star, as this one is now excluded via VAR
    """
    selected = get_guide_catalog(obsid=6820)
    assert 614606480 not in selected['id']


def test_avoid_trap():
    """
    Set up a scenario where a star is selected fine at one roll, and then
    confirm that it is not selected when roll places it on the trap
    """
    agasc_ids = [156384720, 156376184, 156381600, 156379416, 156384304]
    date = '2018:001'
    ra = 9.769
    dec = 20.147
    roll1 = 295.078
    att = (ra, dec, roll1)

    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    selected1 = get_guide_catalog(att=att, date=date, t_ccd=-15, dither=(8, 8),
                                  n_guide=5, stars=stars)
    assert selected1['id'].tolist() == agasc_ids

    # Roll so that 156381600 is on the trap
    roll2 = 297.078
    att = (ra, dec, roll2)

    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    selected2 = get_guide_catalog(att=att, date=date, t_ccd=-15, dither=(8, 8),
                                  n_guide=5, stars=stars)
    assert 156381600 not in selected2['id'].tolist()


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_big_dither():
    # Obsid 20168
    selected = get_guide_catalog(obsid=20168, n_guide=5)
    expected = [977409032, 977930352, 977414712, 977416336, 977405808]
    assert selected['id'].tolist() == expected


def test_check_pixmag_offset():
    APL = AcaPsfLibrary()

    # Then use one star and test with various pixels
    star = {'row': -.5, 'col': .5, 'mag': 9.0, 'id': 1}

    # Confirm that when the bad pixel is bright enough to shift the centroid
    # by N arcsecs that the mag/offset code agrees
    rs = range(-4, 4)
    cs = range(-4, 4)
    lims = [0.1, 0.2, .5, 1.0, 5.0]
    pixvals = range(250, 5000, 250)

    for r_dist, c_dist, lim, pixval in itertools.product(rs, cs, lims, pixvals):
        # Get a new star image every time because centroid_fm messes with it in-place
        star_img = APL.get_psf_image(star['row'], star['col'],
                                     norm=mag_to_count_rate(star['mag']))
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
        pmag = get_pixmag_for_offset(star['mag'], lim)
        if dr > lim:
            assert count_rate_to_mag(pixval) < pmag


def test_check_spoil_contrib():
    """
    Construct a case where a star spoils the edge of the 8x8
    Note that for these mock stars, since we we are checking the status of
    the first star, ASPQ1 needs to be nonzero on that star or the
    check_spoil_contrib code will bail out before actually doing the check
    """
    stars = StarsTable.empty()
    stars.add_fake_star(row=0, col=0, mag=8.0, id=1, ASPQ1=1)
    stars.add_fake_star(row=0, col=-5, mag=6.0, id=2, ASPQ1=0)
    bg_spoil, reg_spoil, rej = check_spoil_contrib(stars, np.array([True, True]), stars, .05, 25)
    assert reg_spoil[0]

    # Construct a case where a star spoils just a background pixel
    stars = StarsTable.empty()
    stars.add_fake_star(row=0, col=0, mag=8.0, id=1, ASPQ1=1)
    stars.add_fake_star(row=-5.5, col=-5.5, mag=9.5, id=2, ASPQ1=0)

    bg_spoil, reg_spoil, rej = check_spoil_contrib(stars, np.array([True, True]), stars, .05, 25)
    assert bg_spoil[0]


pix_cases = [{'dither': (8, 8), 'offset_row': 4, 'offset_col': 4, 'spoils': True},
             {'dither': (64, 8), 'offset_row': 16, 'offset_col': 0, 'spoils': True},
             {'dither': (64, 8), 'offset_row': 20, 'offset_col': 0, 'spoils': False},
             {'dither': (64, 8), 'offset_row': 0, 'offset_col': 16, 'spoils': False},
             {'dither': (8, 64), 'offset_row': 0, 'offset_col': 16, 'spoils': True},
             {'dither': (8, 64), 'offset_row': 0, 'offset_col': 20, 'spoils': False}]


@pytest.mark.parametrize('case', pix_cases)
def test_pix_spoiler(case):
    """
    Check that for various dither configurations, a hot pixel near a star will
    result in that star not being selected.
    """
    stars = StarsTable.empty()
    stars.add_fake_star(row=0, col=0, mag=7.0, id=1, ASPQ1=0)
    stars.add_fake_constellation(n_stars=4)
    dark = ACAImage(np.zeros((1024, 1024)), row0=-512, col0=-512)
    pix_config = {'att': (0, 0, 0),
                  'date': '2018:001',
                  't_ccd': -10,
                  'n_guide': 5,
                  'stars': stars}
    # Use the "case" to try to spoil the first star with a bad pixel
    dark.aca[case['offset_row'] + int(stars[0]['row']),
             case['offset_col'] + int(stars[0]['col'])] = mag_to_count_rate(stars[0]['mag'])
    selected = get_guide_catalog(**pix_config, dither=case['dither'], dark=dark)
    assert (1 not in selected['id']) == case['spoils']


def test_check_mag_spoilers():
    """
    Check that stars that should fail the mag/line test actually fail
    """
    intercept = mag_spoiler['Intercept']
    spoilslope = mag_spoiler['Slope']
    star1 = {'row': 0, 'col': 0, 'mag': 9.0, 'MAG_ACA_ERR': 0, 'id': 1}

    # The mag spoiler check only works on stars that are within 10 pixels in row
    # or column, so don't bother simulating stars outside that distance
    r_dists = np.arange(-9.25, 10, 2)
    c_dists = np.arange(-9.5, 10, 2)
    magdiffs = np.arange(2, -5, -.5)

    for r_dist, c_dist, magdiff in itertools.product(r_dists, c_dists, magdiffs):
        star2 = {'row': r_dist, 'col': c_dist, 'mag': star1['mag'] - magdiff,
                 'MAG_ACA_ERR': 0, 'id': 2}
        dist = np.sqrt(r_dist ** 2 + c_dist ** 2)
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
    guides = get_guide_catalog(att=(0, 0, 0), date='2018:001', t_ccd=-10, dither=(8, 8),
                               stars=stars, n_guide=8)
    assert guides.warnings == ['WARNING: Selected only 6 guide stars versus requested 8']


def test_guides_include_exclude():
    """
    Test include and exclude stars for guide.  This uses a catalog with 11 stars:
    - 8 bright stars from 7.0 to 7.7 mag, where the 7.0 is EXCLUDED
    - 2 faint (but OK) stars 10.0, 10.1 where the 10.0 is INCLUDED
    - 1 very faint (bad) stars 12.0 mag is INCLUDED

    Both the 7.0 and 10.1 would normally get picked either initially
    or swapped in during optimization, and 12.0 would never get picked.

    NOTE: right now this is a stub that just checks that the include_ids
    and exclude_ids get set accordingly.  Later the include/exclude
    functionality will be implemented and this test can be substantive.
    """
    stars = StarsTable.empty()

    stars.add_fake_constellation(mag=[7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7],
                                 id=[1, 2, 3, 4, 5, 6, 7, 8],
                                 size=2000, n_stars=8)
    stars.add_fake_constellation(mag=[10.0, 10.1, 12.0],
                                 id=[9, 10, 11],
                                 size=1500, n_stars=3)

    # Make sure baseline catalog is working like expected
    std_info = STD_INFO.copy()
    std_info.update(n_guide=8)
    guides = get_guide_catalog(**std_info, stars=stars)
    assert np.all(guides['id'] == np.arange(1, 9))

    # Define includes and excludes.
    include_ids = [9, 11]
    exclude_ids = [1]

    guides = get_guide_catalog(**std_info, stars=stars,
                               include_ids=include_ids,
                               exclude_ids=exclude_ids)

    assert guides.include_ids == include_ids
    assert guides.exclude_ids == exclude_ids

    assert all(id_ in guides.cand_guides['id'] for id_ in include_ids)

    assert all(id_ in guides['id'] for id_ in include_ids)
    assert all(id_ not in guides['id'] for id_ in exclude_ids)

    assert np.all(guides['id'] == [9, 11, 2, 3, 4, 5, 6, 7])
    assert np.allclose(guides['mag'], [10.0, 12.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6])


dither_cases = [(8, 8), (64, 8), (8, 64), (20, 20), (30, 20)]

def test_guides_include_bad():
    """
    Test include stars for guide where star is bad for some reason.

    - Including a class=1 star on the CCD is allowed.
    - Including a star (otherwise acceptable) just off the CCD is not allowed.

    """
    stars = StarsTable.empty()

    stars.add_fake_constellation(mag=[7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7],
                                 id=[1, 2, 3, 4, 5, 6, 7, 8],
                                 size=2000, n_stars=8)

    # Bright star but class 1, not picked
    stars.add_fake_star(mag=6.5, row=50, col=50, id=10, CLASS=1)

    # Bright star just off the FOV, not picked
    stars.add_fake_star(mag=6.5, row=512, col=0, id=20)

    # Make sure baseline catalog is working like expected
    guides = get_guide_catalog(**STD_INFO, stars=stars)
    assert np.all(guides['id'] == [1, 2, 3, 4, 5])

    # Picking the class=1 star is fine
    guides = get_guide_catalog(**STD_INFO, stars=stars, include_ids=10)
    assert np.all(sorted(guides['id']) == [1, 2, 3, 4, 10])

    # Picking the star off the CCD generates an exception
    with pytest.raises(ValueError) as err:
        get_guide_catalog(**STD_INFO, stars=stars, include_ids=20)
    assert 'cannot include star id=20' in str(err)

@pytest.mark.parametrize('dither', dither_cases)
def test_edge_star(dither):
    """
    Add stars right at row and col max for various dithers.
    This test both confirms that the dark map extraction doesn't break and that
    the stars can still be selected.
    """
    stars = StarsTable.empty()

    stars.add_fake_constellation(mag=[7.0, 7.1, 7.2, 7.3],
                                 id=[1, 2, 3, 4],
                                 size=2000, n_stars=4)

    # Add stars exactly at 4 corners of allowed "in bounds" area for this dither
    row_dither = dither[0] / 5.
    col_dither = dither[1] / 5.
    row_max = CCD['row_max'] - (CCD['row_pad'] + CCD['window_pad'] + row_dither)
    col_max = CCD['col_max'] - (CCD['col_pad'] + CCD['window_pad'] + col_dither)
    stars.add_fake_star(row=row_max, col=col_max, mag=6.0)
    stars.add_fake_star(row=row_max * -1, col=col_max, mag=6.0)
    stars.add_fake_star(row=row_max * -1, col=col_max * -1, mag=6.0)
    stars.add_fake_star(row=row_max, col=col_max * -1, mag=6.0)
    info = mod_std_info(n_guide=8, dither_guide=(row_dither * 5, col_dither * 5), stars=stars)
    guides = get_guide_catalog(**info)
    # Confirm 4 generic stars plus for corner stars are selected
    assert len(guides) == 8


def test_get_ax_range():
    """
    Confirm that the ranges from get_ax_range are reasonable for a variety of
    center pixel locations and extents (extent = 4 + pix_dither)
    """
    ns = [0, 0.71, 495.3, -200.2]
    extents = [4.0, 5.6, 4.8, 9.0]
    for (n, extent) in itertools.product(ns, extents):
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
