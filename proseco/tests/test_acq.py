# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from pathlib import Path

from Quaternion import Quat
from chandra_aca.aca_image import AcaPsfLibrary
from chandra_aca.transform import mag_to_count_rate, yagzag_to_pixels

from ..report_acq import make_report
from ..acq import (get_p_man_err, bin2x2,
                   get_imposter_stars,
                   get_image_props,
                   AcqTable, calc_p_on_ccd,
                   get_acq_catalog,
                   )
from ..catalog import get_aca_catalog
from ..core import ACABox, StarsTable
from .test_common import OBS_INFO, STD_INFO, mod_std_info, DARK40

from .. import characteristics as ACA
from .. import characteristics_fid as FID
from .. import characteristics_acq as ACQ


TEST_DATE = '2018:144'  # Fixed date for doing tests
ATT = [10, 20, 3]  # Arbitrary test attitude
CACHE = {}  # Cache stuff for speed
TEST_COLS = ('idx', 'slot', 'id', 'yang', 'zang', 'halfw')


def calc_p_brightest(acq, box_size, stars, dark, man_err=0, dither=20, bgd=0):
    """
    Stub for original functional version of acq.calc_p_brightest, which was
    turned into an AcqTable method.
    """
    acqs = AcqTable()
    acqs.t_ccd = -10.0
    acqs.stars = stars
    acqs.dark = dark
    acqs.dither = dither
    return acqs.calc_p_brightest(acq, box_size, man_err, bgd)


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
    """Test the get_p_man_err function.

    P_man_errs is a table that defines the probability of a maneuver error being
    within a defined lo/hi bin [arcsec] given a maneuver angle [deg] within a
    bin range.  The first two columns specify the maneuver error bins and the
    subsequent column names give the maneuver angle bin in the format
    <angle_lo>-<angle_hi>.  The source table does not include the row for
    the 0-60 man err case: this is generated automatically from the other
    values so the sum is 1.0.
    ::

       man_err_lo man_err_hi 0-5 5-20 20-40 40-60 60-80 80-100 100-120 120-180
       ---------- ---------- --- ---- ----- ----- ----- ------ ------- -------
               60         80 0.0  0.2   0.5   0.6   1.6    4.0     8.0     8.0
               80        100 0.0  0.1   0.2   0.3   0.5    1.2     2.4     2.4
              100        120 0.0  0.0   0.1   0.2   0.3    0.8     0.8     0.8
              120        140 0.0  0.0  0.05  0.05   0.2    0.4     0.4     0.4
              140        160 0.0  0.0  0.05  0.05   0.2    0.2     0.2     0.4

    """
    assert get_p_man_err(man_err=30, man_angle=0) == 1.0
    assert get_p_man_err(60, 0) == 1.0  # Exactly 60 is in 0-60 bin
    assert get_p_man_err(60.00001, 0) == 0.0  # Just over 60 in 60-80 bin
    assert get_p_man_err(30, 5) == 1.0  # [0-5 deg bin]
    assert get_p_man_err(30, 5.0001) == 1 - (0.001 + 0.002)  # [5-20 deg bin]
    assert get_p_man_err(60.0001, 6) == 0.002  # + 0.001) [5-20 deg bin]
    assert get_p_man_err(60.0001, 0) == 0  # Just over 60 in 60-80" bin
    with pytest.raises(ValueError):
        get_p_man_err(170, 30)


def test_bin2x2():
    """Test the bin2x2 function"""
    arr = bin2x2(np.arange(16).reshape(4, 4))
    assert arr.shape == (2, 2)
    assert np.all(arr == [[10, 18], [42, 50]])


def test_get_image_props():
    """Test the get_image_props function"""
    c_row = 10
    c_col = 20
    bgd = 40
    norm = mag_to_count_rate(9.0)
    APL = AcaPsfLibrary()
    psf_img = APL.get_psf_image(c_row, c_col, norm=norm, pix_zero_loc='edge')
    ccd_img = np.full((100, 100), fill_value=bgd, dtype=float)
    ccd_img[c_row - 4:c_row + 4, c_col - 4:c_col + 4] += psf_img
    img, img_sum, mag, row, col = get_image_props(ccd_img, c_row, c_col, bgd=bgd)
    assert np.isclose(mag, 9.0, atol=0.05, rtol=0)
    assert np.isclose(row, c_row, atol=0.001, rtol=0)
    assert np.isclose(col, c_col, atol=0.001, rtol=0)


def setup_get_imposter_stars(val):
    """Setup for testing with the get_imposter_stars function."""
    bgd = 40
    dark = np.full((100, 100), fill_value=bgd, dtype=float)
    box_size = ACABox(6 * 5)
    dark[30, 28] = val + bgd
    dark[32, 29] = val + bgd
    dark[30, 30] = val + bgd
    dark[26, 24] = val + bgd + 1000
    imposters = get_imposter_stars(dark, 30, 28, thresh=4000, box_size=box_size, test=True)
    return imposters


def test_get_imposters_3500():
    """
    Test three nearby hits that don't make the cut.
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
    """
    Test get_imposters with an imposter of 5000.
    """
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
        CACHE['stars'] = StarsTable.from_agasc(ATT, date='2018:230')
    return CACHE['stars']


def get_test_cand_acqs():
    if 'cand_acqs' not in CACHE:
        acqs = AcqTable()
        acqs.p_man_errs = ACQ.p_man_errs['120-180']
        acqs.t_ccd = -10.0
        stars = get_test_stars()
        CACHE['cand_acqs'] = acqs.get_acq_candidates(stars)
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
    probs = [calc_p_brightest(acq, box_size, stars_sp, dark_imp, dither=ACABox(0), bgd=bgd)
             for box_size in ACQ.box_sizes]

    #  Box size:                160   140   120    100   80   60  arcsec
    assert np.allclose(probs, [0.25, 0.25, 0.25, 0.3334, 0.5, 1.0], rtol=0, atol=0.01)


def test_calc_p_brightest_same_bright_asymm_dither():
    """
    Test for an easy situation of three spoiler/imposters with exactly
    the same brightness as acq star so that p_brighter is always 0.5.
    As each one comes into the box you add another coin toss to the
    odds of the acq star being brightest.

    Use an asymmetric dither to test the handling of this.
    """
    bgd = 40
    dark = np.full((1024, 1024), dtype=float, fill_value=bgd)
    stars = get_test_stars()

    acq = get_test_cand_acqs()[0]
    # See test_calc_p_brightest_same_bright() for explanation of mag_err
    mag_err = 0.032234
    acq['mag_err'] = mag_err

    dither = ACABox((20, 60))

    # Add a spoiler star to stars and leave dark map the same.
    # Spoiler stars don't care about dither.
    stars_sp = add_spoiler(stars, acq, dyang=-105, dzang=105, dmag=0.0, mag_err=mag_err)
    probs = [calc_p_brightest(acq, box_size, stars=stars_sp, dark=dark, dither=dither,
                              bgd=bgd, man_err=0)
             for box_size in ACQ.box_sizes]

    #  Box size:               160  140  120  100   80   60  arcsec
    assert np.allclose(probs, [0.5, 0.5, 0.5, 1.0, 1.0, 1.0], rtol=0, atol=0.01)

    # Add an imposter to dark map and leave stars the same
    # 80" box + dither = (100, 140), so (105, 145) pixel is not included
    # 100" box + dither = (120, 160), so (105, 145) pixel is included
    acq['imposters'] = None
    acq['spoilers'] = None
    dark_imp = add_imposter(dark, acq, dyang=105, dzang=145, dmag=0.0)
    probs = [calc_p_brightest(acq, box_size, stars=stars, dark=dark_imp, dither=dither,
                              bgd=bgd, man_err=0)
             for box_size in ACQ.box_sizes]

    #  Box size:               160  140  120  100  80   60  arcsec
    assert np.allclose(probs, [0.5, 0.5, 0.5, 0.5, 1.0, 1.0], rtol=0, atol=0.01)

    # Both together
    acq['imposters'] = None
    acq['spoilers'] = None
    probs = [calc_p_brightest(acq, box_size, stars=stars_sp, dark=dark_imp, dither=dither,
                              bgd=bgd, man_err=0)
             for box_size in ACQ.box_sizes]

    #  Box size:                160    140    120  100   80   60  arcsec
    assert np.allclose(probs, [0.333, 0.333, 0.333, 0.5, 1.0, 1.0], rtol=0, atol=0.01)


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
    probs = [calc_p_brightest(acq, box_size, stars_sp, dark, dither=ACABox(0), bgd=bgd)
             for box_size in box_sizes]
    assert np.allclose(probs, [0.0, 1.0], rtol=0, atol=0.001)

    # Comparable spoiler at 65 arcsec (within mag_err)
    acq = get_test_cand_acqs()[0]
    stars_sp = add_spoiler(stars, acq, dyang=65, dzang=65, dmag=-1.0, mag_err=1.0)
    probs = [calc_p_brightest(acq, box_size, stars_sp, dark, dither=ACABox(0), bgd=bgd)
             for box_size in box_sizes]
    assert np.allclose(probs, [0.158974, 1.0], rtol=0, atol=0.01)

    # Bright imposter at 85 arcsec
    acq = get_test_cand_acqs()[0]
    dark_imp = add_imposter(dark, acq, dyang=-85, dzang=-85, dmag=-1.0)
    probs = [calc_p_brightest(acq, box_size, stars, dark_imp, dither=ACABox(0), bgd=bgd)
             for box_size in box_sizes]
    assert np.allclose(probs, [0.0, 1.0], rtol=0, atol=0.001)


def test_calc_p_on_ccd():
    """
    Test the calculation of probability of star being on usable area on the CCD.
    """
    # These lines mimic the code in calc_p_on_ccd() which requires that
    # track readout box is fully within the usable part of CCD.
    max_ccd_row = ACA.max_ccd_row - 5
    max_ccd_col = ACA.max_ccd_col - 4

    # Halfway off in both row and col, (1/4 of area remaining)
    p_in_box = calc_p_on_ccd(max_ccd_row, max_ccd_col, ACABox(60))
    assert np.allclose(p_in_box, 0.25)

    p_in_box = calc_p_on_ccd(max_ccd_row, max_ccd_col, ACABox(120))
    assert np.allclose(p_in_box, 0.25)

    # 3 of 8 pixels off in row (5/8 of area remaining)
    p_in_box = calc_p_on_ccd(max_ccd_row - 1, 0, ACABox(20))
    assert np.allclose(p_in_box, 0.625)

    # Same but for col
    p_in_box = calc_p_on_ccd(0, max_ccd_col - 1, ACABox(20))
    assert np.allclose(p_in_box, 0.625)

    # Same but for a negative col number
    p_in_box = calc_p_on_ccd(0, -(max_ccd_col - 1), ACABox(20))
    assert np.allclose(p_in_box, 0.625)


def test_calc_p_on_ccd_asymmetric_dither():
    """
    Test the calculation of probability of star being on usable area on the CCD
    for the case of asymmetric dither.
    """
    # These lines mimic the code in calc_p_on_ccd() which requires that
    # track readout box is fully within the usable part of CCD.
    max_ccd_row = ACA.max_ccd_row - 5
    max_ccd_col = ACA.max_ccd_col - 4

    # Halfway off in both row and col, (1/4 of area remaining).  These checks
    # don't change from symmetric case because of the placement of row, col.
    p_in_box = calc_p_on_ccd(max_ccd_row, max_ccd_col, ACABox((60, 120)))
    assert np.allclose(p_in_box, 0.25)

    p_in_box = calc_p_on_ccd(max_ccd_row, max_ccd_col, ACABox((120, 60)))
    assert np.allclose(p_in_box, 0.25)

    # 3 of 8 pixels off in row (5/8 of area remaining).  Dither_z (col) does not
    # matter here because only rows are off CCD.
    for dither_z in 5, 100:
        p_in_box = calc_p_on_ccd(max_ccd_row - 1, 0, ACABox((20, dither_z)))
        assert np.allclose(p_in_box, 0.625)

    # Same but for col
    for dither_y in 5, 100:
        p_in_box = calc_p_on_ccd(0, max_ccd_col - 1, ACABox((dither_y, 20)))
        assert np.allclose(p_in_box, 0.625)

        # Same but for a negative col number
        p_in_box = calc_p_on_ccd(0, -(max_ccd_col - 1), ACABox((dither_y, 20)))
        assert np.allclose(p_in_box, 0.625)

    # Show expected asymmetric behavior, starting right at the physical CCD edge.
    # In this case the only chance to get on the CCD is for dither to bring it
    # there.  (Note: the model assumes the dither spatial distribution is
    # flat, but it is not).

    # First, with dither_y <= 25 or dither_z <= 20, p_in_box is exactly zero
    for dither in ((25, 20), (60, 20), (25, 60)):
        p_in_box = calc_p_on_ccd(ACA.max_ccd_row, ACA.max_ccd_col, ACABox(dither))
        assert p_in_box == 0

    # Now some asymmetric cases.  p_in_ccd increases with larger dither.
    for dither, exp in [((30, 40), 0.02083333),
                        ((40, 30), 0.03125),
                        ((40, 200), 0.084375),
                        ((200, 40), 0.109375)]:
        p_in_box = calc_p_on_ccd(ACA.max_ccd_row, ACA.max_ccd_col, ACABox(dither))
        assert np.isclose(p_in_box, exp)


def test_get_acq_catalog_19387():
    """Put it all together.  Regression test for selected stars.  This obsid
    actually changes out one of the initial catalog candidates.

    From ipython:
    >>> from proseco.acq import AcqTable
    >>> acqs = get_acq_catalog(19387)
    >>> TEST_COLS = ('idx', 'slot', 'id', 'yang', 'zang', 'halfw')
    >>> repr(acqs.cand_acqs[TEST_COLS]).splitlines()
    """
    acqs = get_acq_catalog(**OBS_INFO[19387])
    # Expected
    exp = ['<AcqTable length=13>',
           ' idx   slot    id      yang     zang   halfw',
           'int64 int64  int32   float64  float64  int64',
           '----- ----- -------- -------- -------- -----',
           '    0     0 38280776 -2254.09 -2172.43    80',
           '    1     1 37879960  -567.34  -632.27    80',
           '    2     2 37882072  2197.62  1608.89    60',
           '    3     3 37879992   318.47 -1565.92    60',
           '    4   -99 37882416   481.80  2204.44    60',
           '    5     4 37880176   121.33 -1068.25    60',
           '    6     5 37881728  2046.89  1910.79    60',
           '    7     6 37880376 -1356.71  1071.32    60',
           '    8     7 38276824 -1822.26 -1813.66    60',
           '    9   -99 37882776  1485.00   127.97    60',
           '   10   -99 37880152 -1542.43   970.39    60',
           '   11   -99 37880584 -2005.80  2449.74    60',
           '   12   -99 38273720  -672.32 -2474.85    60']

    repr(acqs.cand_acqs)
    assert repr(acqs.cand_acqs[TEST_COLS]).splitlines() == exp


def test_get_acq_catalog_21007():
    """Put it all together.  Regression test for selected stars.

    From ipython::

      >>> from proseco.acq import AcqTable
      >>> acqs = get_acq_catalog(21007)
      >>> TEST_COLS = ('idx', 'slot', 'id', 'yang', 'zang', 'halfw')
      >>> repr(acqs.cand_acqs[TEST_COLS]).splitlines()

    """
    acqs = get_acq_catalog(**OBS_INFO[21007])

    exp = ['<AcqTable length=14>',
           ' idx   slot     id      yang     zang   halfw',
           'int64 int64   int32   float64  float64  int64',
           '----- ----- --------- -------- -------- -----',
           '    0     0 189417400 -2271.86 -1634.77   160',
           '    1     1 189410928   -62.52  1763.04   160',
           '    2     2 189409160 -2223.75  1998.69   160',
           '    3     3 189417920  1482.94   243.72   160',
           '    4     4 189015480  2222.47  -580.99   160',
           '    5     5 189417752  1994.07   699.55    60',
           '    6     6 189406216 -2311.90  -240.18   120',
           '    7     7 189416328  1677.88   137.11    80',
           '    8   -99 189416496   333.11   -63.30   120',
           '    9   -99 189410280  -495.21  1712.02   120',
           '   10   -99 189416808  2283.31  2007.54   120',
           '   11   -99 189017968  1612.35 -1117.76   120',
           '   12   -99 189417000    52.31  -769.11   120',
           '   13   -99 189011576   553.50 -2473.81   120']

    assert repr(acqs.cand_acqs[TEST_COLS]).splitlines() == exp

    exp = ['<AcqTable length=8>',
           ' idx   slot     id      yang     zang   halfw',
           'int64 int64   int32   float64  float64  int64',
           '----- ----- --------- -------- -------- -----',
           '    0     0 189417400 -2271.86 -1634.77   160',
           '    1     1 189410928   -62.52  1763.04   160',
           '    2     2 189409160 -2223.75  1998.69   160',
           '    3     3 189417920  1482.94   243.72   160',
           '    4     4 189015480  2222.47  -580.99    80',
           '    5     5 189417752  1994.07   699.55    60',
           '    6     6 189406216 -2311.90  -240.18   120',
           '    7     7 189416328  1677.88   137.11    80']

    assert repr(acqs[TEST_COLS]).splitlines() == exp


def test_box_strategy_20603():
    """Test for PR #32 that doesn't allow p_acq to be reduced below 0.1.

    The idx=8 (mag=10.50) star was previously selected with 160 arsec box.

    """
    acqs = get_acq_catalog(**OBS_INFO[20603])

    exp = ['<AcqTable length=13>',
           ' idx   slot     id      yang     zang   halfw',
           'int64 int64   int32   float64  float64  int64',
           '----- ----- --------- -------- -------- -----',
           '    0     0  40113544   102.74  1133.37   160',
           '    1     1 116923496 -1337.79  1049.27   120',
           '    2     2 116791824   622.00  -953.60   160',
           '    3     3  40114416   394.22  1204.43   160',
           '    4     4  40112304 -1644.35  2032.47    80',
           '    5     5 116923528 -2418.65  1088.40   160',
           '    6     6 116791744   985.38 -1210.19   100',
           '    7     7  40108048     2.21  1619.17   140',
           '    8   -99 116785920  -673.94 -1575.87   120',
           '    9   -99 116923744  -853.18   937.73   120',
           '   10   -99 116792320   941.59 -1784.10   120',
           '   11   -99 116918232 -2074.91 -1769.96   120',
           '   12   -99 116923672 -2307.80  1442.43   120']

    assert repr(acqs.cand_acqs[TEST_COLS]).splitlines() == exp

    exp = ['<AcqTable length=8>',
           ' idx   slot     id      yang     zang   halfw',
           'int64 int64   int32   float64  float64  int64',
           '----- ----- --------- -------- -------- -----',
           '    0     0  40113544   102.74  1133.37   160',
           '    1     1 116923496 -1337.79  1049.27   120',
           '    2     2 116791824   622.00  -953.60   160',
           '    3     3  40114416   394.22  1204.43   140',
           '    4     4  40112304 -1644.35  2032.47   160',
           '    5     5 116923528 -2418.65  1088.40   160',
           '    6     6 116791744   985.38 -1210.19   160',
           '    7     7  40108048     2.21  1619.17    60']

    assert repr(acqs[TEST_COLS]).splitlines() == exp


def test_make_report(tmpdir):
    """Test making an acquisition report.

    Use a big-box dither here to test handling of that in report (after passing
    through pickle).

    """
    obsid = 19387
    kwargs = OBS_INFO[obsid].copy()
    kwargs['dither'] = (8, 64)
    acqs = get_acq_catalog(**OBS_INFO[obsid])

    tmpdir = Path(tmpdir)
    obsdir = tmpdir / f'obs{obsid:05}'
    outdir = obsdir / 'acq'

    acqs.to_pickle(rootdir=tmpdir)

    acqs2 = make_report(obsid, rootdir=tmpdir)

    assert (outdir / 'index.html').exists()
    assert (outdir / 'acq_stars.png').exists()
    assert (outdir / 'candidate_stars.png').exists()
    assert len(list(outdir.glob('*.png'))) > 0

    assert repr(acqs) == repr(acqs2)
    assert repr(acqs.cand_acqs) == repr(acqs2.cand_acqs)
    for event, event2 in zip(acqs.log_info, acqs2.log_info):
        assert event == event2

    for attr in ['att', 'date', 't_ccd', 'man_angle', 'dither', 'p_safe']:
        val = getattr(acqs, attr)
        val2 = getattr(acqs2, attr)
        if isinstance(val, float):
            assert np.isclose(val, val2)
        elif isinstance(val, Quat):
            assert np.allclose(val.q, val2.q)
        else:
            assert val == val2


def test_cand_acqs_include_exclude():
    """Test include and exclude stars.

    This uses a catalog with 11 stars:

    - 8 bright stars from 7.0 to 7.7 mag, where the 7.0 is EXCLUDED
    - 2 faint (but OK) stars 10.0, 10.1 where the 10.0 is INCLUDED
    - 1 very faint (bad) stars 12.0 mag is INCLUDED

    Both the 7.0 and 10.1 would normally get picked either initially
    or swapped in during optimization, and 12.0 would never get picked.
    Check that the final catalog is [7.1 .. 7.7, 10.0, 12.0]

    Finally, starting from the catalog chosen with the include/exclude
    constraints applied, remove those constraints and re-optimize.
    This must come back to the original catalog of the 8 bright stars.

    """
    stars = StarsTable.empty()

    stars.add_fake_constellation(mag=[7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7],
                                 id=[1, 2, 3, 4, 5, 6, 7, 8],
                                 size=2000, n_stars=8)
    stars.add_fake_constellation(mag=[10.0, 10.1, 12.0],
                                 id=[9, 10, 11],
                                 size=1500, n_stars=3)

    # Put in a neighboring star that will keep star 9 out of the cand_acqs table
    star9 = stars.get_id(9)
    star9['ASPQ1'] = 20
    stars.add_fake_star(yang=star9['yang'] + 20, zang=star9['zang'] + 20,
                        mag=star9['mag'] + 2.5, id=90)

    # Make sure baseline catalog is working like expected
    acqs = get_acq_catalog(**STD_INFO, optimize=False, stars=stars)
    assert np.all(acqs['id'] == np.arange(1, 9))
    assert np.all(acqs['halfw'] == 160)
    assert np.all(acqs.cand_acqs['id'] == [1, 2, 3, 4, 5, 6, 7, 8, 10])

    # Define includes and excludes. id=9 is in nominal cand_acqs but not in acqs.
    include_ids = [9, 11]
    include_halfws = [45, 89]
    exp_include_halfws = [60, 80]
    exclude_ids = [1]

    for optimize in False, True:
        acqs = get_acq_catalog(**STD_INFO, optimize=optimize, stars=stars,
                               include_ids=include_ids, include_halfws=include_halfws,
                               exclude_ids=exclude_ids)

        assert acqs.include_ids == include_ids
        assert acqs.include_halfws == exp_include_halfws
        assert acqs.exclude_ids == exclude_ids
        assert all(id_ in acqs.cand_acqs['id'] for id_ in include_ids)

        assert all(id_ in acqs['id'] for id_ in include_ids)
        assert all(id_ not in acqs['id'] for id_ in exclude_ids)

        assert np.all(acqs['id'] == [2, 3, 4, 5, 6, 7, 9, 11])
        assert np.all(acqs['halfw'] == [160, 160, 160, 160, 160, 160, 60, 80])
        assert np.allclose(acqs['mag'], [7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 10.0, 12.0])

    # Re-optimize catalog now after removing include and exclude
    acqs.exclude_ids = []
    acqs.include_ids = []
    acqs.include_halfws = []

    # Now starting from the catalog chosen with the include/exclude
    # constraints applied, remove those constraints and re-optimize.
    # This must come back to the original catalog of the 8 bright stars.
    del acqs['slot']
    del acqs.cand_acqs['slot']
    acqs.optimize_catalog()
    acqs.sort('idx')
    assert np.all(acqs['id'] == np.arange(1, 9))
    assert np.all(acqs['halfw'] == 160)

    # Finally include all 8 stars
    include_ids = [1, 3, 4, 5, 6, 7, 9, 11]
    include_halfws = [45, 85, 101, 120, 140, 60, 80, 100]
    exp_include_halfws = [60, 80, 100, 120, 140, 60, 80, 100]

    acqs = get_acq_catalog(**STD_INFO, optimize=True, stars=stars,
                           include_ids=include_ids, include_halfws=include_halfws)

    assert acqs['id'].tolist() == include_ids
    assert acqs['halfw'].tolist() == exp_include_halfws


def test_dither_as_sequence():
    """
    Test that calling get_acq_catalog with a 2-element sequence (dither_y, dither_z)
    gives the expected response.  (Basically that it still returns a catalog).
    """

    stars = StarsTable.empty()
    stars.add_fake_constellation(size=1500, n_stars=8)
    kwargs = STD_INFO.copy()
    kwargs['dither'] = (8, 22)

    acqs = get_acq_catalog(**kwargs, stars=stars)
    assert len(acqs) == 8
    assert acqs.dither == (8, 22)


def test_n_acq():
    """
    Test that specifying n_acq with a value less than 8 gives the expected result.
    This test ensures that the optimization code that is testing against n_acq is
    also getting exercised by selecting a catalog where at least one candidate swap
    occurs.
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=np.linspace(10.0, 10.07, 8), size=2000)
    stars.add_fake_constellation(mag=np.linspace(10.005, 10.035, 4), size=1500, n_stars=4)
    acqs = get_acq_catalog(**STD_INFO, stars=stars, n_acq=6)
    exp = ['<AcqTable length=12>',
           ' idx   slot   id    yang     zang   halfw',
           'int64 int64 int32 float64  float64  int64',
           '----- ----- ----- -------- -------- -----',
           '    0     0   100  2000.00     0.00   140',
           '    1     1   108  1500.00     0.00   140',
           '    2     2   101     0.00  2000.00   140',
           '    3     3   109     0.00  1500.00   140',
           '    4     4   102 -2000.00     0.00   120',
           '    5     5   110 -1500.00     0.00   120',
           '    6   -99   103     0.00 -2000.00   120',
           '    7   -99   111     0.00 -1500.00   120',
           '    8   -99   104  1000.00  1000.00   120',
           '    9   -99   105  1000.00 -1000.00   120',
           '   10   -99   106 -1000.00  1000.00   120',
           '   11   -99   107 -1000.00 -1000.00   120']

    assert repr(acqs.cand_acqs[TEST_COLS]).splitlines() == exp


def test_warnings():
    """
    Test that the ACACatalogTable warnings infrastructure works via a
    specific expected warning in get_acq_catalog (too few stars selected).
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=6)
    acqs = get_acq_catalog(**STD_INFO, stars=stars, n_acq=8, optimize=False)
    assert acqs.warnings == ['WARNING: Selected only 6 acq stars versus requested 8']


def test_no_candidates():
    """
    Test that get_acq_catalog returns a well-formed but zero-length table for
    a star field with no acceptable candidates.
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=13.0, n_stars=2)
    acqs = get_acq_catalog(**STD_INFO, stars=stars)

    assert len(acqs) == 0
    assert 'id' in acqs.colnames
    assert 'halfw' in acqs.colnames


def get_dark_stars_simple(box_size_thresh, dither):
    """
    Set-up for tests of optimized acq and fid selection::

      id    mag  status
      100   9.5  ok
      101   9.6  ok
      102   9.7  ok
      103    10  ok
        2   8.2  spoiled by fid 2 for box > 90
        3  11.5  yellow spoiler for fid 3
        4  11.5  yellow spoiler for fid 4

    All fid sets have spoiler sum = 1 or 2.
    """
    offset = box_size_thresh + FID.spoiler_margin + dither

    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=[9.5, 9.6, 9.7, 10], n_stars=4)

    # Add stars near fid light positions.  For fids 3, 4 put in fid spoilers
    # so the initial fid set is empty.
    stars.add_fake_stars_from_fid(fid_id=[2, 3, 4],
                                  id=[2, 3, 4],
                                  mag=[8.2, 11.5, 11.5],
                                  offset_y=[offset, 10, 10], detector='HRC-S')

    return dark, stars


def test_acq_fid_catalog_probs_low_level():
    """
    Low-level tests of machinery to handle different fid light sets within
    acquisition probabilities.
    """
    # Put an acq star at an offset from fid light id=2 such that for a search
    # box size larger than box_size_thresh, that star will be spoiled.  This
    # uses the equation in FidTable.spoils().
    dither = 20
    box_size_thresh = 90

    dark, stars = get_dark_stars_simple(dither, box_size_thresh)

    # Get the catalogs (n_guide=0 so skip guide selection)
    kwargs = mod_std_info(stars=stars, dark=dark, dither=dither, raise_exc=True,
                          n_guide=0, n_acq=5, detector='HRC-S', optimize=False)
    aca = get_aca_catalog(**kwargs)
    acqs = aca.acqs

    assert np.all(acqs.dark[0:512, 0:512] == 40)

    # Initial fid set is empty () and we check baseline p_safe
    assert acqs.fid_set == ()
    assert np.allclose(np.log10(acqs.calc_p_safe()), -2.4,
                       rtol=0, atol=0.1)

    # This is the acq star spoiled by fid_id=2
    acq = acqs.get_id(2)
    p0 = acq['probs']

    assert p0.p_fid_id_spoiler(box_size_thresh - 1, fid_id=2, acqs=acqs) == 1.0  # OK
    assert p0.p_fid_id_spoiler(box_size_thresh + 1, fid_id=2, acqs=acqs) == 0.0  # spoiled

    # Caching of values
    assert p0._p_fid_id_spoiler == {(box_size_thresh - 1, 2): 1.0,
                                    (box_size_thresh + 1, 2): 0.0}

    # With fid_set = (), the probability multiplier for any fid in the fid set
    # spoiling this star is 1.0, i.e. no fids spoil this star (since there
    # are no fids).
    assert p0.p_fid_spoiler(box_size_thresh - 1, acqs) == 1.0
    assert p0.p_fid_spoiler(box_size_thresh + 1, acqs) == 1.0

    # Now change the fid set to include ones (in particular fid_id=2) that
    # spoils an acq star.  This makes the p_safe value much worse.
    acqs.fid_set = (4, 3, 2)
    assert acqs.fid_set == (2, 3, 4)  # gets sorted when set
    assert np.allclose(np.log10(acqs.calc_p_safe()), -1.3,
                       rtol=0, atol=0.1)

    # With fid_set = (1, 2, 4), the probability multiplier for catalog
    # ids 2 and 4 are spoiled.  This test checks for star id=2 (which is
    # near fid_id=2).
    assert p0.p_fid_spoiler(box_size_thresh - 1, acqs) == 1.0
    assert p0.p_fid_spoiler(box_size_thresh + 1, acqs) == 0.0

    # Reverting fid set also revert the p_safe value.  Note the (1, 3, 4)
    # set does not spoil an acq star.
    for fid_set in ((1, 3, 4), ()):
        acqs.fid_set = fid_set
        assert np.allclose(np.log10(acqs.calc_p_safe()), -2.4,
                           rtol=0, atol=0.1)

    # Check that p_acqs() method responds to fid_set in expected way
    for box_size in ACQ.box_sizes:
        for man_err in ACQ.man_errs:
            if man_err > box_size:
                continue  # p_acq always zero in this case, see AcqProbs.__init__()

            acqs.fid_set = ()
            p_acq0 = acq['probs'].p_acqs(box_size, man_err, acqs)

            acqs.fid_set = (2, 3, 4)
            p_acq1 = acq['probs'].p_acqs(box_size, man_err, acqs)

            if box_size > box_size_thresh:
                # Box includes spoiler so p_acq1 is 0
                assert p_acq0 > 0
                assert p_acq1 == 0
            else:
                # No spoiler, so no change in p_acq
                assert p_acq0 == p_acq1


@pytest.mark.parametrize('n_fid_exp_fid_ids', [(1, [1]),
                                               (2, [1, 4]),
                                               (3, [1, 3, 4])])
def test_acq_fid_catalog_n_fid(n_fid_exp_fid_ids):
    """
    Test optimizing acq and fid in a simple case (which does exercise acq-fid
    optimization) with n_fid=1, 2, 3::

      id    mag  status
      100   9.5  ok
      101   9.6  ok
      102   9.7  ok
      103    10  ok
        2   8.2  spoiled by fid 2 for box > 90
        3  11.5  yellow spoiler for fid 3
        4  11.5  yellow spoiler for fid 4

    For n_fid=1, this chooses fid_set=[1] because that is not spoiled and is
    not a spoiler.

    For n_fid=2, this chooses fid_set=[1, 4] to allow star id=2 to have a
    160" box (maintaining the no-fid opt_P2).  Set [1, 3] would be the same
    but the [1, 3] separation is less than the [1, 4] separation.

    For n_fid=3, this choose fid_set=[1, 3, 4], again so star id=2 is free to
    have a 160" box.  From the stage perspective the spoiler_score=1 fid sets
    have sufficiently degraded P2 that it moves on to the spoiler_score=2 set
    that includes fids 3 and 4.
    """
    # Put an acq star at an offset from fid light id=2 such that for a search
    # box size larger than box_size_thresh, that star will be spoiled.  This
    # uses the equation in FidTable.spoils().
    n_fid, exp_fid_ids = n_fid_exp_fid_ids  # this is a 2-tuple

    dither = 20
    box_size_thresh = 90

    dark, stars = get_dark_stars_simple(dither, box_size_thresh)

    # Get the catalogs (n_guide=0 so skip guide selection)
    kwargs = mod_std_info(stars=stars, dark=dark, dither=dither, raise_exc=True,
                          n_guide=0, n_fid=n_fid, n_acq=5, detector='HRC-S')
    aca = get_aca_catalog(**kwargs)
    acqs = aca.acqs

    assert acqs['id'].tolist() == [2, 100, 101, 102, 103]
    assert acqs['halfw'].tolist() == [160, 160, 160, 160, 80]
    assert acqs.fids['id'].tolist() == exp_fid_ids
    assert acqs.fid_set == tuple(exp_fid_ids)


def test_acq_fid_catalog_zero_cand_fid():
    """
    Test catalog selection with n_fid=3 requested but zero candidate fids.
    This should not happen in practice.
    """
    dither = 20
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5)
    dark = DARK40.copy()

    kwargs = mod_std_info(stars=stars, dark=dark, dither=dither, raise_exc=True,
                          n_guide=0, n_fid=3, n_acq=5,
                          detector='HRC-S', sim_offset=300000)
    aca = get_aca_catalog(**kwargs)

    assert not aca.fids.thumbs_up
    assert len(aca.fids) == 0
    assert len(aca.fids.cand_fids) == 0
    assert aca.acqs.fid_set == ()
    assert len(aca.acqs) == 5
    assert aca.acqs.thumbs_up


def test_acq_fid_catalog_one_cand_fid():
    """
    Test optimizing acq and fid for an ACIS-S observation with large sim_offset
    (-55000) such that only ACIS-6 is still on the ACA CCD.  Add a couple of stars:

    - id=1 is 8.2 mag star that should be selected but is spoiled by fid id=6
      for box size > 90.  Expect optimized size to be 80".
    - id=2 is 11.5 mag star that is a yellow spoiler for fid id=6.  Expect
      fid_set == (6,).

    This test is good because with fid_set=(6,) the final catalog does not meet
    stage requirements and thus excercise the path of never finding a good fid set.
    In that case the best available is selected with a warning.
    """
    box_size_thresh = 90
    dither = 8
    sim_offset = -55000
    offset = box_size_thresh + FID.spoiler_margin + dither

    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=[9.5, 9.6, 9.7, 10], n_stars=4)

    # Add stars near fid light positions.  For fids 3, 4 put in fid spoilers
    # so the initial fid set is empty.
    stars.add_fake_stars_from_fid(fid_id=[6, 6],
                                  id=[1, 2],  # assigned catalog ID
                                  mag=[8.2, 11.5],
                                  offset_y=[offset, 10],
                                  detector='ACIS-S', sim_offset=sim_offset)

    kwargs = mod_std_info(stars=stars, dark=dark, dither=dither, raise_exc=True,
                          n_guide=0, n_fid=3, n_acq=5,
                          detector='ACIS-S', sim_offset=sim_offset)
    aca = get_aca_catalog(**kwargs)

    assert aca.acqs['id'].tolist() == [1, 100, 101, 102, 103]
    assert aca.acqs['halfw'].tolist() == [80, 160, 160, 160, 120]

    assert aca.n_fid == 3
    assert aca.fids['id'].tolist() == [6]
    assert aca.acqs.fid_set == (6,)

    # Not enough fids
    assert aca.thumbs_up == 0
    assert aca.fids.thumbs_up == 0
    assert aca.acqs.thumbs_up == 1
    assert aca.warnings == ['WARNING: No acq-fid combination was '
                            'found that met stage requirements']


@pytest.mark.parametrize('n_fid', [2, 3])
def test_acq_fid_catalog_two_cand_fid(n_fid):
    """
    Test optimizing acq and fid for an HRC-I observation with large sim_offset
    (29829) such that only two fids id=(1, 2) are still on the ACA CCD.  This
    is an ACIS undercover observation (obsid 19793 from DEC0516B).

    Add a couple of stars:

    - id=1 is 8.2 mag star that should be selected but is spoiled by fid id=1
      for box size > 90.  Expect optimized size to be 80".
    - id=2 is 11.5 mag star that is a yellow spoiler for fid id=2.  Expect
      fid_set == (1, 2) with spoiler_score=1.

    """
    box_size_thresh = 90
    dither = 20
    sim_offset = 29829
    offset = box_size_thresh + FID.spoiler_margin + dither

    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=[9.5, 9.6, 9.7, 10], n_stars=4)

    # Add stars near fid light positions.
    stars.add_fake_stars_from_fid(fid_id=[1, 2],
                                  id=[1, 2],  # assigned catalog ID
                                  mag=[8.2, 11.5],
                                  offset_y=[offset, 10],
                                  detector='HRC-I', sim_offset=sim_offset)

    kwargs = mod_std_info(stars=stars, dark=dark, dither=dither, raise_exc=True,
                          n_guide=0, n_fid=n_fid, n_acq=5,
                          detector='HRC-I', sim_offset=sim_offset)
    aca = get_aca_catalog(**kwargs)

    assert aca.acqs['id'].tolist() == [1, 100, 101, 102, 103]
    assert aca.acqs['halfw'].tolist() == [80, 160, 160, 160, 120]

    assert aca.n_fid == n_fid
    assert aca.fids['id'].tolist() == [1, 2]
    assert aca.acqs.fid_set == (1, 2)

    # If n_fid=2 then getting only 2 fids is OK, but otherwise thumbs-down.
    thumbs_up = (1 if n_fid == 2 else 0)
    assert aca.thumbs_up == thumbs_up
    assert aca.fids.thumbs_up == thumbs_up
    assert aca.acqs.thumbs_up == 1


def test_0_5_degree_man_angle_bin():
    """
    It was decided at the 2018-09-26 SSAWG to set the probability of
    man_err > 60 to 0.0 for the 0-5 degree maneuver angle bin.  This tests
    that.

    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=[7.99, 8.01, 8.99, 9.01, 10.2], n_stars=5)
    kwargs = mod_std_info(stars=stars, dark=dark, t_ccd=-10.0,
                          n_guide=0, n_fid=0, n_acq=8, man_angle=4.8)
    acqs = get_acq_catalog(**kwargs)
    assert acqs['halfw'].tolist() == [100, 80, 80, 60, 60]
    assert acqs[0]['box_sizes'].tolist() == [100, 80, 60]
    assert acqs[1]['box_sizes'].tolist() == [80, 60]
    assert acqs[2]['box_sizes'].tolist() == [80, 60]
    assert acqs[3]['box_sizes'].tolist() == [60]
    assert acqs[4]['box_sizes'].tolist() == [60]


def test_0_5_degree_man_angle_bin_diff_t_ccd():
    """
    It was decided at the 2018-09-26 SSAWG to set the probability of
    man_err > 60 to 0.0 for the 0-5 degree maneuver angle bin.  This tests
    that.

    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=[7.99, 8.01, 8.99, 9.01, 10.2], n_stars=5)
    for t_ccd, halfws in [(-9, [80, 80, 60, 60, 60]),
                          (-11, [100, 100, 80, 80, 60])]:
        kwargs = mod_std_info(stars=stars, dark=dark, t_ccd=t_ccd,
                              n_guide=0, n_fid=0, n_acq=8, man_angle=4.8)
        acqs = get_acq_catalog(**kwargs)
        assert acqs['halfw'].tolist() == halfws


def test_bad_star_list():
    """
    Test that a star with ID in the bad star list is not selected.
    """
    bad_id = 39980640
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=np.linspace(9, 10.3, 5), n_stars=5)
    # Bright star that would normally be selected
    stars.add_fake_star(yang=100, zang=100, mag=6.5, id=bad_id)
    kwargs = mod_std_info(stars=stars, dark=dark,
                          n_guide=0, n_fid=0, n_acq=8, man_angle=4.8)
    acqs = get_acq_catalog(**kwargs)
    assert bad_id not in acqs['id']

    idx = acqs.stars.get_id_idx(bad_id)
    assert acqs.bad_stars_mask[idx]


def test_acq_include_optimize_halfw_ids():
    """
    Test that force-include acq stars with halfw=0 get halfw optimized.
    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=7.0, n_stars=8, size=2000)
    stars.add_fake_star(yang=200, zang=200, mag=7.5, id=200)
    stars.add_fake_star(yang=-200, zang=200, mag=10.5, id=201)
    stars.add_fake_star(yang=-200, zang=-200, mag=10.5, id=202)

    kwargs = mod_std_info(stars=stars, dark=dark,
                          n_guide=8, n_fid=0, n_acq=8, man_angle=90,
                          include_ids_acq=[200, 201, 202],
                          include_halfws_acq=[0, 0, 140])
    aca = get_aca_catalog(**kwargs)

    # Confirm that halfw values are good
    star = aca.get_id(200)
    assert star['halfw'] == 160  # Large box for a bright star
    star = aca.get_id(201)
    assert star['halfw'] == 80  # Small box for faint star
    star = aca.get_id(202)
    assert star['halfw'] == 140  # Specified box size (not optimized)

    assert aca.acqs.include_optimize_halfw_ids == [200, 201]
