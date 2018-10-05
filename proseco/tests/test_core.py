# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.io import ascii

import agasc
from ..core import (ACABox, get_kwargs_from_starcheck_text, calc_spoiler_impact,
                    StarsTable)


def test_agasc_1p7():
    """
    Ensure that AGASC 1.7 is being used.
    """
    # Updated with APASS info
    star = agasc.get_star(688522864)
    assert star['RSV3'] == 1
    assert star['RSV2'] == 11944

    # NOT updated with APASS info
    star = agasc.get_star(611193056)
    assert star['RSV3'] == 0
    assert star['RSV2'] == -9999


def test_box_init():
    """ACABox initialization functionality"""
    box = ACABox()
    assert box.y == 20
    assert box.z == 20

    box = ACABox(None)
    assert box.y == 20
    assert box.z == 20

    box = ACABox((15, 10))
    assert box.y == 15
    assert box.z == 10

    box = ACABox([15, 10])
    assert box.y == 15
    assert box.z == 10

    box2 = ACABox(box)
    assert box2.y == 15
    assert box2.z == 10

    with pytest.raises(ValueError) as err:
        ACABox([1, 2, 3])
        assert 'size arg must be' in str(err)


def test_box_row_col_max():
    """ACABox row, col properties and max() method"""
    box = ACABox((15, 10))
    assert box.row == 3
    assert box.col == 2
    assert box.max() == 15


def test_box_eq():
    """ACABox equality and inequality"""
    dither1 = (15, 10)
    dither3 = (16, 10)
    box1 = ACABox(dither1)
    box2 = ACABox(dither1)
    box3 = ACABox(dither3)

    # ACABox to ACABox test
    assert box1 == box2
    assert box1 != box3

    # Test against left and right sides that need conversion to ACABox first
    assert dither1 == box1
    assert dither1 != box3
    assert box1 == dither1
    assert box3 != dither1


def test_box_add():
    """ACABox left and right addition"""
    box1 = ACABox((10, 15))

    box2 = box1 + 5
    assert box2 == (15, 20)

    box2 = 5 + box1
    assert box2 == (15, 20)

    box2 = (5, 10) + box1
    assert box2 == (15, 25)

    box2 = box1 + (5, 10)
    assert box2 == (15, 25)


def test_box_greater():
    """ACABox 'greater than' comparison"""
    box1 = ACABox((10, 15))

    # True if either component of box1 is bigger than the test value
    assert box1 > (9.9, 14.9)
    assert box1 > (9.9, 15)
    assert box1 > (10, 14.9)

    assert not box1 > (10, 15)
    assert not box1 > box1
    assert not box1 > (11, 16)


def test_get_kwargs_from_starcheck_text():
    text = """
    OBSID: 21071  Kapteyn's Star         ACIS-S SIM Z offset:0     (0.00mm) Grating: NONE
    RA, Dec, Roll (deg):    77.976747   -45.066796    85.007351
    Dither: ON  Y_amp= 8.0  Z_amp= 8.0  Y_period=1000.0  Z_period= 707.1
    BACKSTOP GUIDE_SUMM OR MANVR DOT MAKE_STARS TLR

    MP_TARGQUAT at 2018:273:05:06:56.863 (VCDU count = 15011745)
      Q1,Q2,Q3,Q4: 0.30730924  0.61223190  0.22717786  0.69218737
      MANVR: Angle=  98.00 deg  Duration= 1967 sec  Slew err= 58.3 arcsec  End= 2018:273:05:39:39

    MP_STARCAT at 2018:273:05:06:58.506 (VCDU count = 15011751)
    ---------------------------------------------------------------------------------------------
     IDX SLOT        ID  TYPE   SZ   P_ACQ    MAG   MAXMAG   YANG   ZANG DIM RES HALFW PASS NOTES
    ---------------------------------------------------------------------------------------------
    [ 1]  0           1   FID  8x8     ---   7.000   8.000    922  -1737   1   1   25
    [ 2]  1           2   FID  8x8     ---   7.000   8.000   -773  -1741   1   1   25
    [ 3]  2           4   FID  8x8     ---   7.000   8.000   2140    166   1   1   25
    [ 4]  3   995373760   BOT  6x6   0.985   8.227   9.734    571   1383  32   1  180
    [ 5]  4  1058803776   BOT  6x6   0.985   8.591  10.094  -1871    288  28   1  160    a2
    [ 6]  5  1058806384   BOT  6x6   0.945   9.144  10.641  -1541   -971  28   1  160    a2
    [ 7]  6   995368032   GUI  6x6     ---   8.639  10.141   1711    912   1   1   25
    [ 8]  7         ---   MON  8x8     ---     ---  13.938     66     31   3   0   20    mX
    [ 9]  6   995373040   ACQ  6x6   0.944   9.470  10.969    642   1162  23   1  135    a2
    [10]  7   995371984   ACQ  6x6   0.934   9.684  11.188    427    974  23   1  135    a2
    [11]  0   995369464   ACQ  6x6   0.925   9.398  10.906   1420     21  28   1  160    a2
    [12]  1  1058803424   ACQ  6x6   0.930   9.521  11.031  -1409   2023  26   1  150    a2
    [13]  2  1059724992   ACQ  6x6   0.848   9.989  11.500  -1271  -2201  20   1  120    a3

    >> INFO   : CCD temperature exceeds -10.9 C
    >> INFO   : Monitor window special commanding meets requirements

    Probability of acquiring 2,3, and 4 or fewer stars (10^x):	-6.191	-4.565	-3.180
    Acquisition Stars Expected  : 7.49
    Predicted Max CCD temperature: -10.6 C       N100 Warm Pix Frac 0.270
    Dynamic Mag Limits: Yellow 10.08     Red 10.22
    """
    kwargs = get_kwargs_from_starcheck_text(text)
    exp = {'obsid': 21071,
           'att': [77.976747, -45.066796, 85.007351],
           'man_angle': 98.0,
           'dither': (8.0, 8.0),
           't_ccd': -10.6,
           'date': '2018:273:05:06:58.506',
           'n_guide': 4,
           'n_fid': 3,
           'detector': 'ACIS-S',
           'sim_offset': 0,
           'focus_offset': 0}
    assert kwargs == exp

cases = [dict(row=4, col=0, mag0=8, mag1=10.5, exp=(0.16, -0.01, 1.00)),
         dict(row=1, col=0, mag0=8, mag1=10.5, exp=(0.43, -0.00, 1.10)),
         dict(row=1, col=0, mag0=8, mag1=8, exp=(2.37, -0.01, 1.98)),
         dict(row=-3, col=3, mag0=8, mag1=8, exp=(99, 99, -99)),
         dict(row=-3, col=3, mag0=8, mag1=10, exp=(-0.54, 0.49, 0.35)),
         dict(row=-3, col=3, mag0=8, mag1=11, exp=(-0.10, 0.09, 0.74)),
         dict(row=3, col=-3, mag0=10, mag1=12, exp=(0.34, -0.31, 0.50)),
         dict(row=3, col=-3, mag0=10, mag1=13, exp=(0.08, -0.08, 0.81))]


@pytest.mark.parametrize('case', cases)
def test_calc_spoiler_impact(case):
    """
    Test that calc_spoiler_impact gives reasonable answer.  See also:
    http://nbviewer.jupyter.org/url/asc.harvard.edu/mta/ASPECT/proseco/spoiler-impact.ipynb
    """
    stars = StarsTable.empty()
    stars.add_fake_star(row=0, col=0, mag=case['mag0'])
    stars.add_fake_star(row=case['row'], col=case['col'], mag=case['mag1'])
    dy, dz, f = calc_spoiler_impact(stars[0], stars)
    # print(f'({dy:.2f}, {dz:.2f}, {f:.2f})')
    assert np.isclose(dy, case['exp'][0], rtol=0, atol=0.01)
    assert np.isclose(dz, case['exp'][1], rtol=0, atol=0.01)
    assert np.isclose(f, case['exp'][2], rtol=0, atol=0.01)


def test_calc_spoiler_impact_21068():
    """
    Confirm that for the Dragonfly-44 field that the impact of the spoiler
    is acceptable and the star can be used as a candidate.
    """
    # These values are taken from the stars table for 21068.
    stars = ascii.read(['row col mag id',
                        '-366.63 -92.75  8.73 1',
                        '-365.14 -92.93 10.52 2'])
    dy, dz, f = calc_spoiler_impact(stars[0], stars)
    assert np.abs(dy) < 1.5
    assert np.abs(dz) < 1.5
    assert np.abs(f) > 0.95
