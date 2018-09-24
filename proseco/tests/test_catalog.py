# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pickle
import pytest
from pathlib import Path

import numpy as np
import mica.starcheck

from .test_common import STD_INFO, mod_std_info
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
           '   0   1         4  FID 8x8 0.000  7.00   8.00  2140.23   166.63   1   1    25',
           '   1   2         5  FID 8x8 0.000  7.00   8.00 -1826.28   160.17   1   1    25',
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


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_get_aca_catalog_20259():
    """
    Test obsid 20259 which has two spoiled fids: HRC-2 is yellow and HRC-4 is red.
    Expectation is to choose fids 1, 2, 3 (not 4).
    """
    # Force not using a bright star so there is a GUI-only (not BOT) star
    aca = get_aca_catalog(20259, raise_exc=True)
    exp = ['slot idx     id    type  sz p_acq mag  maxmag   yang     zang   dim res halfw',
           '---- --- --------- ---- --- ----- ---- ------ -------- -------- --- --- -----',
           '   0   1         1  FID 8x8 0.000 7.00   8.00 -1175.03  -468.23   1   1    25',
           '   1   2         2  FID 8x8 0.000 7.00   8.00  1224.70  -460.93   1   1    25',
           '   2   3         3  FID 8x8 0.000 7.00   8.00 -1177.69   561.30   1   1    25',
           '   3   4 896009152  BOT 6x6 0.985 7.25   8.75  1693.39   217.92  20   1   160',
           '   4   5 897712576  BOT 6x6 0.985 8.25   9.75 -1099.95  2140.23  20   1   160',
           '   5   6 897717296  BOT 6x6 0.985 8.20   9.70   932.58  1227.48  20   1   160',
           '   6   7 896013056  BOT 6x6 0.983 8.75  10.25  1547.25 -2455.12  20   1   160',
           '   7   8 897722680  GUI 6x6 0.000 8.85  10.35  1007.82  1676.78   1   1    25',
           '   7   9 896011576  ACQ 6x6 0.985 8.12   9.62   810.99   -69.21  20   1   160',
           '   0  10 897192352  ACQ 6x6 0.985 9.03  10.53 -2110.43  2005.21  20   1   100',
           '   1  11 896008728  ACQ 6x6 0.985 9.08  10.58  2223.64  -341.57  20   1    80',
           '   2  12 896010144  ACQ 6x6 0.985 9.11  10.61  1081.11   912.90  20   1    80']
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

    for obj in (aca, aca.acqs, aca.guides, aca.fids):
        assert len(obj) == 0
        assert 'id' in obj.colnames
        assert 'idx' in obj.colnames
        if obj.name == 'acqs':
            assert 'halfw' in obj.colnames


def test_unhandled_exception():
    with pytest.raises(TypeError):
        get_aca_catalog(obsid=None, raise_exc=True)


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


def test_pickle():
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=10.0, n_stars=5)
    aca = get_aca_catalog(stars=stars, raise_exc=True, **STD_INFO)

    assert aca.thumbs_up == 0
    assert aca.acqs.thumbs_up == 0
    assert aca.guides.thumbs_up == 1
    assert aca.fids.thumbs_up == 1

    aca2 = pickle.loads(pickle.dumps(aca))

    assert repr(aca) == repr(aca2)
    assert repr(aca.acqs.cand_acqs) == repr(aca2.acqs.cand_acqs)

    for cat in None, 'acqs', 'guides', 'fids':
        if cat:
            obj = getattr(aca, cat)
            obj2 = getattr(aca2, cat)
            for event, event2 in zip(obj.log_info['events'],
                                     obj2.log_info['events']):
                assert event == event2
        else:
            obj = aca
            obj2 = aca2

        for attr in ['att', 'date', 't_ccd', 'man_angle', 'dither', 'thumbs_up']:
            val = getattr(obj, attr)
            val2 = getattr(obj2, attr)
            if isinstance(val, float):
                assert np.isclose(val, val2)
            else:
                assert val == val2


def test_big_sim_offset():
    """
    Check getting a catalog for a large SIM offset that means there are
    no candidates.
    """
    aca = get_aca_catalog(**mod_std_info(sim_offset=200000, raise_exc=True))
    assert len(aca.acqs) == 8
    assert len(aca.guides) == 5
    assert len(aca.fids) == 0
    names = ['id', 'yang', 'zang', 'row', 'col', 'mag', 'spoiler_score', 'idx']
    assert all(name in aca.fids.colnames for name in names)
