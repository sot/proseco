# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function  # For Py2 compatibility

import numpy as np
import pytest
from pathlib import Path

from chandra_aca.aca_image import AcaPsfLibrary
from chandra_aca.transform import mag_to_count_rate, yagzag_to_pixels

from ..report import make_report
from ..acq import (get_p_man_err, bin2x2, CHAR,
                   get_imposter_stars, get_stars,
                   get_image_props, calc_p_brightest,
                   AcqTable, calc_p_on_ccd,
                   get_acq_catalog,
                   )

TEST_DATE = '2018:144'  # Fixed date for doing tests
ATT = [10, 20, 3]  # Arbitrary test attitude
CACHE = {}  # Cache stuff for speed
TEST_COLS = ('idx', 'slot', 'id', 'yang', 'zang', 'halfw', 'mag', 'p_acq')


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
    ok = stars['id'] == acq['id']
    stars.add_row(stars[ok][0])
    star = stars[-1]
    star['id'] = -star['id']
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
    # P_man_errs is a table that defines the probability of a maneuver error being
    # within a defined lo/hi bin [arcsec] given a maneuver angle [deg] within a
    # bin range.  The first two columns specify the maneuver error bins and the
    # subsequent column names give the maneuver angle bin in the format
    # <angle_lo>-<angle_hi>.  The source table does not include the row for
    # the 0-60 man err case: this is generated automatically from the other
    # values so the sum is 1.0.
    #
    # man_err_lo man_err_hi 0-5 5-20 20-40 40-60 60-80 80-100 100-120 120-180
    # ---------- ---------- --- ---- ----- ----- ----- ------ ------- -------
    #         60         80 0.1  0.2   0.5   0.6   1.6    4.0     8.0     8.0
    #         80        100 0.0  0.1   0.2   0.3   0.5    1.2     2.4     2.4
    #        100        120 0.0  0.0   0.1   0.2   0.3    0.8     0.8     0.8
    #        120        140 0.0  0.0  0.05  0.05   0.2    0.4     0.4     0.4
    #        140        160 0.0  0.0  0.05  0.05   0.2    0.2     0.2     0.4

    assert get_p_man_err(man_err=30, man_angle=0) == 0.999  # 1 - 0.1%
    assert get_p_man_err(60, 0) == 0.999  # Exactly 60 is in 0-60 bin
    assert get_p_man_err(60.00001, 0) == 0.001  # Just over 60 in 60-80 bin
    assert get_p_man_err(30, 5) == 1 - 0.001  # [0-5 deg bin]
    assert get_p_man_err(30, 5.0001) == 1 - (0.001 + 0.002)  # [5-20 deg bin]
    assert get_p_man_err(60.0001, 6) == 0.002  # + 0.001) [5-20 deg bin]
    assert get_p_man_err(60.0001, 0) == 0.001  # Just over 60 in 60-80 bin
    with pytest.raises(ValueError):
        get_p_man_err(170, 30)


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
        CACHE['stars'] = get_stars(ATT, date='2018:230')
    return CACHE['stars']


def get_test_cand_acqs():
    if 'cand_acqs' not in CACHE:
        acqs = AcqTable()
        stars = get_test_stars()
        CACHE['cand_acqs'], bads = acqs.get_acq_candidates(stars)
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
             for box_size in CHAR.box_sizes]

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


def test_calc_p_on_ccd():
    # These lines mimic the code in calc_p_on_ccd() which requires that
    # track readout box is fully within the usable part of CCD.
    max_ccd_row = CHAR.max_ccd_row - 5
    max_ccd_col = CHAR.max_ccd_col - 4

    # Halfway off in both row and col, (1/4 of area remaining)
    p_in_box = calc_p_on_ccd(max_ccd_row, max_ccd_col, 60)
    assert np.allclose(p_in_box, 0.25)

    p_in_box = calc_p_on_ccd(max_ccd_row, max_ccd_col, 120)
    assert np.allclose(p_in_box, 0.25)

    # 3 of 8 pixels off in row (5/8 of area remaining)
    p_in_box = calc_p_on_ccd(max_ccd_row - 1, 0, 20)
    assert np.allclose(p_in_box, 0.625)

    # Same but for col
    p_in_box = calc_p_on_ccd(0, max_ccd_col - 1, 20)
    assert np.allclose(p_in_box, 0.625)

    # Same but for a negative col number
    p_in_box = calc_p_on_ccd(0, -(max_ccd_col - 1), 20)
    assert np.allclose(p_in_box, 0.625)


def test_get_acq_catalog_19387():
    """Put it all together.  Regression test for selected stars.  This obsid
    actually changes out one of the initial catalog candidates.

    From ipython:
    >>> from proseco.acq import AcqTable
    >>> acqs = get_acq_catalog(19387)
    >>> TEST_COLS = ('idx', 'slot', 'id', 'yang', 'zang', 'halfw', 'mag', 'p_acq')
    >>> repr(acqs.meta['cand_acqs'][TEST_COLS]).splitlines()
    """
    obsid = 19387
    att = [188.617671, 2.211623, 231.249803]
    date = '2017:182:22:06:22.744'
    t_ccd = -14.1
    man_angle = 1.74
    dither = 4.0
    acqs = get_acq_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd,
                           man_angle=man_angle, dither=dither)
    # Expected
    exp = ['<AcqTable length=11>',
           ' idx  slot    id      yang     zang   halfw   mag    p_acq ',
           'int64 str3  int32   float64  float64  int64 float32 float64',
           '----- ---- -------- -------- -------- ----- ------- -------',
           '    0    0 38280776 -2254.09 -2172.43   160    8.77   0.985',
           '    1    1 37879960  -567.34  -632.27    80    9.20   0.984',
           '    2    2 37882072  2197.62  1608.89   160   10.16   0.857',
           '    3    3 37879992   318.47 -1565.92    60   10.41   0.933',
           '    4    4 37882416   481.80  2204.44   100   10.41   0.865',
           '    5    5 37880176   121.33 -1068.25    60   10.62   0.584',
           '    6    6 37881728  2046.89  1910.79   100   10.76   0.057',
           '    7    7 37880376 -1356.71  1071.32   100   10.80   0.084',
           '    8  ... 38276824 -1822.26 -1813.66   120   10.86   0.017',
           '    9  ... 37880152 -1542.43   970.39   120   10.88   0.008',
           '   10  ... 37882776  1485.00   127.97   120   10.93   0.007']

    assert repr(acqs.meta['cand_acqs'][TEST_COLS]).splitlines() == exp


def test_get_acq_catalog_21007():
    """Put it all together.  Regression test for selected stars.
    From ipython:
    >>> from proseco.acq import AcqTable
    >>> acqs = get_acq_catalog(21007)
    >>> TEST_COLS = ('idx', 'slot', 'id', 'yang', 'zang', 'halfw', 'mag', 'p_acq')
    >>> repr(acqs.meta['cand_acqs'][TEST_COLS]).splitlines()
    """
    obsid = 21007
    att = [184.371121, 17.670062, 223.997765]
    date = '2018:159:11:20:52.162'
    t_ccd = -11.3
    man_angle = 60.39
    dither = 8.0
    acqs = get_acq_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd,
                           man_angle=man_angle, dither=dither)

    exp = ['<AcqTable length=14>',
           ' idx  slot     id      yang     zang   halfw   mag    p_acq ',
           'int64 str3   int32   float64  float64  int64 float32 float64',
           '----- ---- --------- -------- -------- ----- ------- -------',
           '    0    0 189417400 -2271.86 -1634.77   160    7.71   0.985',
           '    1    1 189410928   -62.52  1763.04   160    8.84   0.982',
           '    2    2 189409160 -2223.75  1998.69   160    9.84   0.876',
           '    3    3 189417920  1482.94   243.72   160    9.94   0.807',
           '    4    4 189015480  2222.47  -580.99    60   10.01   0.538',
           '    5    5 189417752  1994.07   699.55   100   10.24   0.503',
           '    6    6 189406216 -2311.90  -240.18    80   10.26   0.514',
           '    7    7 189416328  1677.88   137.11    60   10.40   0.348',
           '    8  ... 189416496   333.11   -63.30   120   10.58   0.003',
           '    9  ... 189410280  -495.21  1712.02   120   10.62   0.005',
           '   10  ... 189416808  2283.31  2007.54   120   10.86   0.000',
           '   11  ... 189417392   163.37   165.65   120   10.95   0.000',
           '   12  ... 189017968  1612.35 -1117.76   120   10.98   0.000',
           '   13  ... 189011576   553.50 -2473.81   120   10.99   0.000']

    assert repr(acqs.meta['cand_acqs'][TEST_COLS]).splitlines() == exp


def test_box_strategy_20603():
    """Test for PR #32 that doesn't allow p_acq to be reduced below 0.1.
    The idx=8 (mag=10.50) star was previously selected with 160 arsec box.
    """
    obsid = 20603
    att = [201.561783, 7.748784, 205.998301]
    date = '2018:120:19:06:28.154'
    t_ccd = -11.2
    man_angle = 111.95
    dither = 8.0
    acqs = get_acq_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd,
                           man_angle=man_angle, dither=dither)

    exp = ['<AcqTable length=13>',
           ' idx  slot     id      yang     zang   halfw   mag    p_acq ',
           'int64 str3   int32   float64  float64  int64 float32 float64',
           '----- ---- --------- -------- -------- ----- ------- -------',
           '    0    0  40113544   102.74  1133.37   160    7.91   0.985',
           '    1    1 116791824   622.00  -953.60   160    9.01   0.958',
           '    2    2 116923496 -1337.79  1049.27   120    9.14   0.970',
           '    3    3  40114416   394.22  1204.43   160    9.78   0.885',
           '    4    4  40112304 -1644.35  2032.47    80    9.79   0.932',
           '    5    5 116923528 -2418.65  1088.40   160    9.84   0.593',
           '    6    6 116791744   985.38 -1210.19    60   10.29   0.501',
           '    7  ...  40108048     2.21  1619.17   120   10.46   0.023',
           '    8    7 116785920  -673.94 -1575.87    60   10.50   0.136',
           '    9  ... 116791664  2307.25 -1504.54   120   10.74   0.000',
           '   10  ... 116792320   941.59 -1784.10   120   10.83   0.000',
           '   11  ... 116923744  -853.18   937.73   120   10.84   0.000',
           '   12  ... 116918232 -2074.91 -1769.96   120   10.96   0.000']

    assert repr(acqs.meta['cand_acqs'][TEST_COLS]).splitlines() == exp


def test_make_report(tmpdir):
    obsid = 19387
    att = [188.617671, 2.211623, 231.249803]
    date = '2017:182:22:06:22.744'
    t_ccd = -14.1
    man_angle = 1.74
    dither = 4.0
    acqs = get_acq_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd,
                           man_angle=man_angle, dither=dither)

    tmpdir = Path(tmpdir)
    obsdir = tmpdir / f'obs{obsid:05}'

    acqs.to_pickle(rootdir=tmpdir)

    acqs2 = make_report(obsid, rootdir=tmpdir)

    assert (obsdir / 'index.html').exists()
    assert len(list(obsdir.glob('*.png'))) > 0

    assert repr(acqs) == repr(acqs2)
    assert repr(acqs.meta['cand_acqs']) == repr(acqs2.meta['cand_acqs'])
    for event, event2 in zip(acqs.log_info, acqs2.log_info):
        assert event == event2

    for attr in ['att', 'date', 't_ccd', 'man_angle', 'dither', 'p_safe']:
        val = acqs.meta[attr]
        val2 = acqs2.meta[attr]
        if isinstance(val, float):
            assert np.isclose(val, val2)
        else:
            assert val == val2
