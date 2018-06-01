# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function  # For Py2 compatibility

import numpy as np

from chandra_aca.aca_image import AcaPsfLibrary
from chandra_aca.transform import mag_to_count_rate, yagzag_to_pixels

from ..acq import (get_p_man_err, P_NORMAL, P_BIG, P_ANOM, bin2x2,
                   get_imposter_stars, get_stars, get_acq_candidates,
                   get_image_props, calc_p_brightest, BOX_SIZES,
                   calc_p_in_box, MAX_CCD_ROW, MAX_CCD_COL,
                   get_acq_catalog,
                   )

TEST_DATE = '2018:144'  # Fixed date for doing tests
ATT = [10, 20, 3]  # Arbitrary test attitude
CACHE = {}  # Cache stuff for speed


def add_imposter(dark, acq, dyang, dzang, dmag):
    """
    For testing, add an imposter (single hot pixel) at the specified delta location
    and mag relative to an ``acq`` star.  Returns a new dark map.
    """
    dark = dark.copy()
    yang = acq['yang'] + dyang
    zang = acq['zang'] + dzang
    row, col = yagzag_to_pixels(yang, zang)
    row0 = int(row + 512)
    col0 = int(col + 512)
    dark[row0, col0] += mag_to_count_rate(acq['mag'] + dmag)

    return dark


def add_spoiler(stars, acq, dyang, dzang, dmag, mag_err=0.05):
    """
    For testing, add a spoiler stars at the specified delta location
    and mag relative to an ``acq`` star.  Returns a new stars table.
    """
    stars = stars.copy()
    ok = stars['AGASC_ID'] == acq['id']
    stars.add_row(stars[ok][0])
    star = stars[-1]
    star['AGASC_ID'] = -star['AGASC_ID']
    star['yang'] = acq['yang'] + dyang
    star['zang'] = acq['zang'] + dzang
    star['mag'] = acq['mag'] + dmag
    star['mag_err'] = mag_err
    star['MAG_ACA'] = star['mag']
    row, col = yagzag_to_pixels(star['yang'], star['zang'])
    star['row'] = row
    star['col'] = col
    star['CLASS'] = 0

    return stars


def test_get_p_man_err():
    assert get_p_man_err(30) == P_NORMAL
    assert get_p_man_err(60.0) == P_NORMAL
    assert np.isclose(get_p_man_err(60.001), P_BIG / 3)
    assert np.isclose(get_p_man_err(120), P_BIG / 3)
    assert np.isclose(get_p_man_err(160.0), P_ANOM / 2)
    try:
        get_p_man_err(170)
    except ValueError:
        pass
    else:
        assert False


def test_bin2x2():
    arr = bin2x2(np.arange(16).reshape(4, 4))
    assert arr.shape == (2, 2)
    assert np.all(arr == [[10, 18], [42, 50]])


def test_get_image_props():
    c_row = 10
    c_col = 20
    bgd = 40
    norm = mag_to_count_rate(9.0)
    APL = AcaPsfLibrary()
    psf_img = APL.get_psf_image(c_row, c_col, norm=norm, pix_zero_loc='edge')
    ccd_img = np.full((100, 100), fill_value=bgd, dtype=float)
    ccd_img[c_row - 4:c_row + 4, c_col - 4:c_col + 4] += psf_img
    img, img_sum, mag, row, col = get_image_props(ccd_img, c_row, c_col, bgd=bgd)
    print(img.astype(int))
    assert np.isclose(mag, 9.0, atol=0.05, rtol=0)
    assert np.isclose(row, c_row, atol=0.001, rtol=0)
    assert np.isclose(col, c_col, atol=0.001, rtol=0)


def setup_get_imposter_stars(val):
    bgd = 40
    dark = np.full((100, 100), fill_value=bgd, dtype=float)
    box_size = 6 * 5
    dark[30, 28] = val + bgd
    dark[32, 29] = val + bgd
    dark[30, 30] = val + bgd
    dark[26, 24] = val + bgd + 1000
    imposters = get_imposter_stars(dark, 30, 28, thresh=4000, box_size=box_size, test=True)
    return imposters


def test_get_imposters_3500():
    """
    Three nearby hits don't make the cut
    """
    imposters = setup_get_imposter_stars(3500)
    assert len(imposters) == 1
    imp = imposters[0]
    assert imp['row0'] == 23
    assert imp['col0'] == 21
    assert np.isclose(imp['mag'], 10.490, rtol=0, atol=0.001)
    assert imp['img_sum'] == 4500
    assert imp['img'][3, 3] == 4500


def test_get_imposters_5000():
    imposters = setup_get_imposter_stars(5000)
    assert len(imposters) == 2
    imp = imposters[1]
    assert imp['row0'] == 23
    assert imp['col0'] == 21
    assert np.isclose(imp['mag'], 10.178, rtol=0, atol=0.001)
    assert imp['img'][3, 3] == 6000.0
    assert imp['img_sum'] == 6000

    imp = imposters[0]
    assert imp['row0'] == 28
    assert imp['col0'] == 26
    assert imp['img_sum'] == 5000 * 3
    assert np.isclose(imp['mag'], 9.183, rtol=0, atol=0.001)


def get_test_stars():
    if 'stars' not in CACHE:
        CACHE['stars'] = get_stars(ATT)
    return CACHE['stars']


def get_test_cand_acqs():
    if 'cand_acqs' not in CACHE:
        stars = get_test_stars()
        CACHE['cand_acqs'], bads = get_acq_candidates(stars)
        # Don't care about bads for testing
    return CACHE['cand_acqs'].copy()


def test_calc_p_brightest_same_bright():
    """
    Test for an easy situation of three spoiler/imposters with exactly
    the same brightness as acq star so that p_brighter is always 0.5.
    As each one comes into the box you add another coin toss to the
    odds of the acq star being brightest.
    """
    bgd = 40
    dark = np.full((1024, 1024), dtype=float, fill_value=bgd)
    stars = get_test_stars()

    acq = get_test_cand_acqs()[0]
    # Imposter ends up with this mag_err (based on get_mag_std() for the
    # imposter mag), so set the acq and spoilers to have the same err.  This
    # makes the calculation be trivial to do analytically: prob that acq
    # is the brightest is 1 / N where N is the total number of objects
    # (acq + imposters + spoilers).
    mag_err = 0.032234
    acq['mag_err'] = mag_err

    dark_imp = add_imposter(dark, acq, dyang=-105, dzang=-105, dmag=0.0)
    stars_sp = add_spoiler(stars, acq, dyang=65, dzang=65, dmag=0.0, mag_err=mag_err)
    stars_sp = add_spoiler(stars_sp, acq, dyang=85, dzang=0, dmag=0.0, mag_err=mag_err)
    probs = [calc_p_brightest(acq, box_size, stars_sp, dark_imp, dither=0, bgd=bgd)
             for box_size in BOX_SIZES]

    #  Box size:                160   140   120    100   80   60  arcsec
    assert np.allclose(probs, [0.25, 0.25, 0.25, 0.3334, 0.5, 1.0], rtol=0, atol=0.01)


def test_calc_p_brightest_1mag_brighter():
    """
    Test for the situation of spoiler/imposter that is 1 mag brighter.
    """
    bgd = 40
    box_sizes = [100, 60]
    dark = np.full((1024, 1024), dtype=float, fill_value=bgd)
    stars = get_test_stars()

    # Bright spoiler at 65 arcsec
    acq = get_test_cand_acqs()[0]
    stars_sp = add_spoiler(stars, acq, dyang=65, dzang=65, dmag=-1.0)
    probs = [calc_p_brightest(acq, box_size, stars_sp, dark, dither=0, bgd=bgd)
             for box_size in box_sizes]
    assert np.allclose(probs, [0.0, 1.0], rtol=0, atol=0.001)

    # Comparable spoiler at 65 arcsec (within mag_err)
    acq = get_test_cand_acqs()[0]
    stars_sp = add_spoiler(stars, acq, dyang=65, dzang=65, dmag=-1.0, mag_err=1.0)
    probs = [calc_p_brightest(acq, box_size, stars_sp, dark, dither=0, bgd=bgd)
             for box_size in box_sizes]
    assert np.allclose(probs, [0.158974, 1.0], rtol=0, atol=0.01)

    # Bright imposter at 85 arcsec
    acq = get_test_cand_acqs()[0]
    dark_imp = add_imposter(dark, acq, dyang=-85, dzang=-85, dmag=-1.0)
    probs = [calc_p_brightest(acq, box_size, stars, dark_imp, dither=0, bgd=bgd)
             for box_size in box_sizes]
    assert np.allclose(probs, [0.0, 1.0], rtol=0, atol=0.001)


def test_calc_p_in_box():
    # Halfway off in both row and col, (1/4 of area remaining)
    p_in_boxes = calc_p_in_box(MAX_CCD_ROW, MAX_CCD_COL, [60, 120])
    assert np.allclose(p_in_boxes, [0.25, 0.25])

    # 3 of 8 pixels off in row (5/8 of area remaining)
    p_in_boxes = calc_p_in_box(MAX_CCD_ROW - 1, 0, [20])
    assert np.allclose(p_in_boxes, [0.625])

    # Same but for col
    p_in_boxes = calc_p_in_box(0, MAX_CCD_COL - 1, [20])
    assert np.allclose(p_in_boxes, [0.625])

    # Same but for a negative col number
    p_in_boxes = calc_p_in_box(0, -(MAX_CCD_COL - 1), [20])
    assert np.allclose(p_in_boxes, [0.625])


def test_get_acq_catalog():
    """Put it all together.  Mostly a regression test."""
    acqs = get_acq_catalog(21007)
    assert np.all(acqs['id'] == [189417400, 189410928, 189409160, 189417920,
                                 189406216, 189417752, 189015480, 189416328])
    assert np.all(acqs['halfw'] == [160, 160, 160, 120, 60, 100, 60, 60])
