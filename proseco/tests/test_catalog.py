# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from pathlib import Path

import mica.starcheck

from ..core import StarsTable, ACACatalogTable
from ..catalog import get_aca_catalog


HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES['data_root']).exists()


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_get_aca_catalog_20603():
    """Put it all together.  Regression test for selected stars.
    """
    # Force not using a bright star so there is a GUI-only (not BOT) star
    aca = get_aca_catalog(20603, exclude_ids=[40113544], n_fid=2, n_guide=6, n_acq=7,
                          raise_exc=True)
    # Expected 2 fids, 6 guide, 7 acq
    exp = ['slot idx     id    type  sz p_acq  mag  maxmag   yang     zang   dim res halfw',
           '---- --- --------- ---- --- ----- ----- ------ -------- -------- --- --- -----',
           '   0   1         2  FID 8x8 0.000  7.00   8.00  -773.20 -1742.03   1   1    25',
           '   1   2         4  FID 8x8 0.000  7.00   8.00  2140.23   166.63   1   1    25',
           '   2   3 116791824  BOT 6x6 0.958  9.01  10.51   622.00  -953.60  20   1   160',
           '   3   4  40114416  BOT 6x6 0.912  9.78  11.28   394.22  1204.43  20   1   140',
           '   4   5 116923528  BOT 6x6 0.593  9.84  11.34 -2418.65  1088.40  20   1   160',
           '   5   6  40112304  BOT 6x6 0.687  9.79  11.29 -1644.35  2032.47  20   1   160',
           '   6   7 116791744  BOT 6x6 0.347 10.29  11.79   985.38 -1210.19  20   1   140',
           '   7   8  40113544  GUI 6x6 0.000  7.91   9.41   102.74  1133.37   1   1    25',
           '   7   9 116923496  ACQ 6x6 0.970  9.14  10.64 -1337.79  1049.27  20   1   120',
           '   0  10 116785920  ACQ 6x6 0.136 10.50  12.00  -673.94 -1575.87  20   1    60']
    repr(aca)  # Apply default formats
    assert aca.pformat(max_width=-1) == exp


def test_exception_handling():
    """
    Test top-level exception catching.
    """
    aca = get_aca_catalog(att=(0, 0, 0), man_angle=10, date='2018:001',
                          dither_acq=(8, 8), dither_guide=(8, 8),
                          t_ccd_acq=-10, t_ccd_guide=-10,
                          detector='ACIS-S', sim_offset=0, focus_offset=0,
                          n_guide=8, n_fid=3, n_acq=8,
                          include_ids=[1])  # Fail
    assert 'ValueError: cannot include star id=1' in aca.exception


def test_no_candidates():
    """
    Test that get_aca_catalog returns a well-formed but zero-length tables for
    a star field with no acceptable candidates.
    """
    test_info = dict(obsid=1,
                     n_guide=5, n_fid=3, n_acq=8,
                     att=(0, 0, 0),
                     detector='ACIS-S',
                     sim_offset=0,
                     focus_offset=0,
                     date='2018:001',
                     t_ccd_acq=-11, t_ccd_guide=-11,
                     man_angle=90,
                     dither_acq=(8.0, 8.0),
                     dither_guide=(8.0, 8.0))
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=13.0, n_stars=2)
    acas = get_aca_catalog(**test_info, stars=stars)

    assert 'id' in acas.acqs.colnames
    assert 'halfw' in acas.acqs.colnames
    assert 'id' in acas.guides.colnames
    assert 'id' in acas.fids.colnames


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_big_dither_from_mica_starcheck():
    """
    Test code that infers dither_acq and dither_guide for a big-dither
    observation like 20168.
    """
    aca = ACACatalogTable()
    aca.set_attrs_from_kwargs(obsid=20168)

    assert aca.detector == 'HRC-S'
    assert aca.dither_acq == (20, 20)
    assert aca.dither_guide == (64, 8)
