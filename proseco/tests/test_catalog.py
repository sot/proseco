# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from pathlib import Path

import mica.starcheck

from ..catalog import get_aca_catalog


HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES['data_root']).exists()


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_get_aca_catalog_19387():
    """Put it all together.  Regression test for selected stars.
    """
    # Force not using a bright star so there is a GUI-only (not BOT) star
    aca = get_aca_catalog(20603, exclude_ids=[40113544])
    # Expected
    exp = ['slot idx     id    type  sz p_acq  mag  maxmag   yang     zang   dim res halfw',
           '---- --- --------- ---- --- ----- ----- ------ -------- -------- --- --- -----',
           '   0   1         2  FID 8x8 0.000  7.00   8.00  -773.20 -1742.03   1   1    25',
           '   1   2         4  FID 8x8 0.000  7.00   8.00  2140.23   166.63   1   1    25',
           '   2   3         5  FID 8x8 0.000  7.00   8.00 -1826.28   160.17   1   1    25',
           '   3   4 116791824  BOT 6x6 0.958  9.01  10.51   622.00  -953.60  20   1   160',
           '   4   5  40114416  BOT 6x6 0.912  9.78  11.28   394.22  1204.43  20   1   140',
           '   5   6 116923528  BOT 6x6 0.593  9.84  11.34 -2418.65  1088.40  20   1   160',
           '   6   7  40112304  BOT 6x6 0.687  9.79  11.29 -1644.35  2032.47  20   1   160',
           '   7   8  40113544  GUI 6x6 0.000  7.91   9.41   102.74  1133.37   1   1    25',
           '   7   9 116923496  ACQ 6x6 0.970  9.14  10.64 -1337.79  1049.27  20   1   120',
           '   0  10 116791744  ACQ 6x6 0.347 10.29  11.79   985.38 -1210.19  20   1   140',
           '   1  11  40108048  ACQ 6x6 0.072 10.46  11.96     2.21  1619.17  20   1    60',
           '   2  12 116785920  ACQ 6x6 0.136 10.50  12.00  -673.94 -1575.87  20   1    60']
    repr(aca)  # Apply default formats
    assert aca.pformat(max_width=-1) == exp
