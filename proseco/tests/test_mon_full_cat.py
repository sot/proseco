# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path

import agasc
import mica.starcheck
import numpy as np
import pytest
from astropy.table import Table
from chandra_aca.transform import radec_to_yagzag, yagzag_to_pixels

from .. import characteristics as ACA
from ..catalog import get_aca_catalog
from ..characteristics import MonCoord, MonFunc
from ..core import StarsTable
from .test_common import mod_std_info

HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES['data_root']).exists()
TEST_COLS = 'slot idx id type sz yang zang dim res halfw'.split()

# Do not use the AGASC supplement in testing by default since mags can change
os.environ[agasc.SUPPLEMENT_ENABLED_ENV] = 'False'


@pytest.fixture
def stars():
    out = StarsTable.empty()
    out.add_fake_constellation(
        mag=10.5 - np.arange(8) * 0.5, id=np.arange(100000, 100008)
    )
    return out


def test_monitor_input_processing_ra_dec(stars):
    ra, dec = 0.1, 0.2
    mag = 7.0
    monitors = [[ra, dec, MonCoord.RADEC, mag, MonFunc.MON_FIXED]]
    aca = get_aca_catalog(
        **mod_std_info(att=stars.att, n_fid=0, n_guide=5),
        stars=stars,
        monitors=monitors
    )
    monitors = aca.mons.monitors
    assert len(monitors) == 1
    assert isinstance(monitors, Table)
    yang, zang = radec_to_yagzag(ra, dec, aca.att)
    row, col = yagzag_to_pixels(yang, zang)
    assert np.allclose(monitors['yang'][0], yang)
    assert np.allclose(monitors['zang'][0], zang)
    assert np.allclose(monitors['ra'][0], ra)
    assert np.allclose(monitors['dec'][0], dec)
    assert np.allclose(monitors['row'][0], row)
    assert np.allclose(monitors['col'][0], col)

    ok = aca['type'] == 'MON'
    assert np.count_nonzero(ok) == 1
    mon = aca[ok][0]
    assert np.allclose(mon['yang'], yang)
    assert np.allclose(mon['zang'], zang)
    assert np.allclose(mon['mag'], mag)
    assert np.allclose(mon['maxmag'], ACA.monitor_maxmag)
    assert mon['id'] == 1000


def test_monitor_mon_fixed_auto():
    """In this case the MON_TRACK slot is not near a star"""
    monitors = [
        [-1700, 1900, MonCoord.YAGZAG, 7.5, MonFunc.MON_FIXED],
        [50, -50, MonCoord.ROWCOL, 7.5, MonFunc.MON_TRACK],
        [1053.38, -275.16, MonCoord.YAGZAG, 8.0, MonFunc.AUTO],
    ]
    aca = get_aca_catalog(
        monitors=monitors,
        **mod_std_info(n_guide=6, n_fid=2),
        include_ids_guide=[611192064]
    )

    exp = [
        'slot idx     id    type  sz   yang     zang   dim res halfw',
        '---- --- --------- ---- --- -------- -------- --- --- -----',
        '   0   1         4  FID 8x8  2140.23   166.63   1   1    25',
        '   1   2         5  FID 8x8 -1826.28   160.17   1   1    25',
        '   2   3 611190016  BOT 6x6   175.44 -1297.92  28   1   160',
        '   3   4    139192  BOT 6x6   587.27   802.49  28   1   160',
        '   7   5 611192384  BOT 8x8  1053.38  -275.16  28   1   160',
        '   4   6 611192064  GUI 6x6  2003.89 -1746.97   1   1    25',
        '   5   7      1001  MON 8x8  -219.72  -273.87   2   0    20',
        '   6   8      1000  MON 8x8 -1700.00  1900.00   6   0    20',
        '   0   9 688523960  ACQ 6x6  -202.71 -1008.91  28   1   160',
        '   1  10 688521312  ACQ 6x6  -739.68 -1799.85  28   1   160',
        '   4  11 611189488  ACQ 6x6  1536.83  -786.12  28   1   160',
        '   5  12 688522008  ACQ 6x6  -743.78 -2100.94  28   1   160',
        '   6  13    134680  ACQ 6x6  1314.00  1085.73  28   1   160',
    ]

    assert aca[TEST_COLS].pformat_all() == exp
    assert aca.n_guide == 4  # Two of 6 guide slots become MON slots

    mon = aca.get_id(1000, mon=True)
    assert mon['dim'] == mon['slot']  # Fixed MON
    assert mon['res'] == 0  # No convert to track

    mon = aca.get_id(1001, mon=True)
    assert mon['dim'] == 2  # Tracking brightest star
    assert mon['res'] == 0  # No convert to track

    # BOT from Monitor
    star = aca.get_id(611192384)
    assert star['slot'] == 7
    assert star['type'] == 'BOT'
    assert star['sz'] == '8x8'


def test_full_catalog():
    """Test a full catalog

    - 3 monitor requests:
      - 2 forced as GUIDE
      - 1 AUTO which is not an OK star and becomes MON
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(
        mag=10.5 - np.arange(8) * 0.5, id=np.arange(100000, 100008)
    )
    stars.add_fake_star(yang=400, zang=400, mag=10.5, id=100008)
    stars.add_fake_star(yang=100, zang=200, mag=6.5, id=50)
    stars.add_fake_star(yang=-1000, zang=-2000, mag=6.25, id=51)
    monitors = [
        [100, 200, MonCoord.YAGZAG, 6.123, MonFunc.GUIDE],
        [-1001, -2001, MonCoord.YAGZAG, 6.123, MonFunc.GUIDE],
        [0, 0, MonCoord.YAGZAG, 6.123, MonFunc.AUTO],
    ]

    aca = get_aca_catalog(
        **mod_std_info(att=stars.att, n_fid=1, n_guide=7),
        stars=stars,
        monitors=monitors,
        exclude_ids_acq=[100000, 100001, 51, 100003]
    )

    exp = [
        'slot idx   id   type  sz   yang     zang   dim res halfw  mag ',
        '---- --- ------ ---- --- -------- -------- --- --- ----- -----',
        '   0   1      1  FID 8x8   922.59 -1737.89   1   1    25  7.00',
        '   1   2 100007  BOT 6x6  -750.00  -750.00  28   1   160  7.00',
        '   2   3 100006  BOT 6x6  -750.00   750.00  28   1   160  7.50',
        '   3   4 100005  BOT 6x6   750.00  -750.00  28   1   160  8.00',
        '   6   5     50  BOT 8x8   100.00   200.00  28   1   160  6.50',
        '   4   6 100003  GUI 6x6     0.00 -1500.00   1   1    25  9.00',
        # Mag from star cat not monitors and sz=8x8. Position is also from
        # star cat since monitors yag/zag = -1001, -2001.
        '   7   7     51  GUI 8x8 -1000.00 -2000.00   1   1    25  6.25',
        # Mag from monitors and dim=7 (=> tracking brightest star in slot 7,
        # which happens to be a GUI from MON).
        '   5   8   1000  MON 8x8     0.00     0.00   7   0    20  6.12',
        '   0   9 100004  ACQ 6x6   750.00   750.00  28   1   160  8.50',
        '   4  10 100002  ACQ 6x6 -1500.00     0.00  28   1   160  9.50',
        '   5  11 100008  ACQ 6x6   400.00   400.00   8   1    60 10.50',
    ]

    assert aca[TEST_COLS + ['mag']].pformat_all() == exp
    assert aca.n_guide == 6  # One of 7 available guide slots becomes a MON


def test_mon_takes_guide():
    """Test that commanding a MON on top of a potential GUI star correctly
    precludes use of that star and takes one of the available n_guide slots.

    Also checks that when a MON is identified as a star in the catalog, the
    'id' and 'mag' correspond to the catalog. This overrides the mag in
    `monitors`.
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(
        mag=6.5 + np.arange(8) * 0.5, id=np.arange(100000, 100008)
    )
    monitors = [[1500, 0, MonCoord.YAGZAG, 8.123, MonFunc.MON_TRACK]]

    aca = get_aca_catalog(
        **mod_std_info(att=stars.att, n_fid=0, n_guide=5, n_acq=5),
        stars=stars,
        monitors=monitors
    )

    # This has 4 + 1 stars in the "guide" catalog, where the MON has taken one slot.
    exp = [
        'slot idx   id   type  sz   yang     zang   dim res halfw  mag ',
        '---- --- ------ ---- --- -------- -------- --- --- ----- -----',
        '   0   1 100001  BOT 8x8     0.00  1500.00  28   1   160  7.00',
        '   1   2 100002  BOT 8x8 -1500.00     0.00  28   1   160  7.50',
        '   2   3 100003  BOT 8x8     0.00 -1500.00  28   1   160  8.00',
        '   3   4 100004  BOT 8x8   750.00   750.00  28   1   160  8.50',
        # MON below is associated with id=100000 and has mag=6.50 not 8.12
        # The dim (DTS) value is 0, which is the slot of the brightest GUI.
        '   7   5 100000  MON 8x8  1500.00     0.00   0   0    20  6.50',
        '   4   6 100000  ACQ 8x8  1500.00     0.00  28   1   160  6.50',
    ]

    assert aca[TEST_COLS + ['mag']].pformat_all() == exp
    assert aca.n_guide == 4  # One of 5 available guide slots becomes a MON
