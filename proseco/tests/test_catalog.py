# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pickle
import pytest
from pathlib import Path

import numpy as np
import mica.starcheck

from .test_common import STD_INFO, mod_std_info, DARK40
from ..core import StarsTable, ACACatalogTable
from ..catalog import get_aca_catalog
from ..fid import get_fid_catalog
from .. import characteristics as CHAR


HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES['data_root']).exists()
TEST_COLS = 'slot idx id type sz yang zang dim res halfw'.split()


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_get_aca_catalog_49531():
    """
    Test of getting an ER using the mica.starcheck archive for getting the
    obs parameters.  This tests a regression introduced in the acq-fid
    functionality.
    """
    aca = get_aca_catalog(49531, raise_exc=True)
    assert len(aca.acqs) == 8
    assert len(aca.guides) == 8
    assert len(aca.fids) == 0


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_get_aca_catalog_20603():
    """Put it all together.  Regression test for selected stars.
    """
    # Force not using a bright star so there is a GUI-only (not BOT) star
    aca = get_aca_catalog(20603, exclude_ids_acq=[40113544], n_fid=2, n_guide=6, n_acq=7,
                          raise_exc=True)
    # Expected 2 fids, 6 guide, 7 acq
    exp = ['slot idx     id    type  sz   yang     zang   dim res halfw',
           '---- --- --------- ---- --- -------- -------- --- --- -----',
           '   0   1         4  FID 8x8  2140.23   166.63   1   1    25',
           '   1   2         5  FID 8x8 -1826.28   160.17   1   1    25',
           '   2   3  40112304  BOT 6x6 -1644.35  2032.47  20   1   160',
           '   3   4  40114416  BOT 6x6   394.22  1204.43  20   1   140',
           '   4   5 116791824  BOT 6x6   622.00  -953.60  20   1   160',
           '   5   6 116923528  BOT 6x6 -2418.65  1088.40  20   1   160',
           '   6   7  40113544  GUI 6x6   102.74  1133.37   1   1    25',
           '   6   8 116923496  ACQ 6x6 -1337.79  1049.27  20   1   120',
           '   7   9 116791744  ACQ 6x6   985.38 -1210.19  20   1   160',
           '   0  10  40108048  ACQ 6x6     2.21  1619.17  20   1    60']

    repr(aca)  # Apply default formats
    assert aca[TEST_COLS].pformat(max_width=-1) == exp

    aca_pkl = pickle.dumps(aca)
    assert len(aca_pkl) < 180_000  # Nominally ~170k, warn if size grows


@pytest.mark.skipif('not HAS_SC_ARCHIVE', reason='Test requires starcheck archive')
def test_get_aca_catalog_20259():
    """
    Test obsid 20259 which has two spoiled fids: HRC-2 is yellow and HRC-4 is red.
    Expectation is to choose fids 1, 2, 3 (not 4).
    """
    aca = get_aca_catalog(20259, raise_exc=True)
    exp = ['slot idx     id    type  sz   yang     zang   dim res halfw',
           '---- --- --------- ---- --- -------- -------- --- --- -----',
           '   0   1         1  FID 8x8 -1175.03  -468.23   1   1    25',
           '   1   2         2  FID 8x8  1224.70  -460.93   1   1    25',
           '   2   3         3  FID 8x8 -1177.69   561.30   1   1    25',
           '   3   4 896009152  BOT 6x6  1693.39   217.92  20   1   100',
           '   4   5 896009240  BOT 6x6  -911.41   402.62  20   1    80',
           '   5   6 896013056  BOT 6x6  1547.25 -2455.12  20   1    80',
           '   6   7 897712576  BOT 6x6 -1099.95  2140.23  20   1    80',
           '   7   8 897717296  BOT 6x6   932.58  1227.48  20   1    80',
           '   0   9 896011576  ACQ 6x6   810.99   -69.21  20   1   100',
           '   1  10 897718208  ACQ 6x6   765.61  1530.27  20   1    80',
           '   2  11 897192352  ACQ 6x6 -2110.43  2005.21  20   1    80']

    repr(aca)  # Apply default formats
    assert aca[TEST_COLS].pformat(max_width=-1) == exp


def test_exception_handling():
    """
    Test top-level exception catching.
    """
    aca = get_aca_catalog(att=(0, 0, 0), man_angle=10, date='2018:001',
                          dither_acq=(8, 8), dither_guide=(8, 8),
                          t_ccd_acq=-10, t_ccd_guide=-10,
                          detector='ACIS-S', sim_offset=0, focus_offset=0,
                          n_guide=8, n_fid=3, n_acq=8,
                          include_ids_acq=[1])  # Fail
    assert 'include_ids and include_halfws must have same length' in aca.exception

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

    assert acas.thumbs_up == 0
    assert acas.acqs.thumbs_up == 0
    assert acas.guides.thumbs_up == 0
    assert acas.fids.thumbs_up == 0


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


def test_call_args_attr():
    aca = get_aca_catalog(**mod_std_info(optimize=False, n_guide=0, n_acq=0, n_fid=0))
    assert aca.call_args == {'att': (0, 0, 0),
                             'date': '2018:001',
                             'detector': 'ACIS-S',
                             'dither': 8.0,
                             'focus_offset': 0,
                             'man_angle': 90,
                             'n_acq': 0,
                             'n_fid': 0,
                             'n_guide': 0,
                             'obsid': 0,
                             'optimize': False,
                             'sim_offset': 0,
                             't_ccd': -11}


def test_bad_obsid():
    aca = get_aca_catalog(obsid='blah blah')  # Expects this to be starcheck catalog
    assert 'ValueError: text does not have OBSID' in aca.exception

    assert aca.thumbs_up == 0
    assert aca.acqs.thumbs_up == 0
    assert aca.guides.thumbs_up == 0
    assert aca.fids.thumbs_up == 0


def test_bad_pixel_dark_current():
    """
    Test avoidance of bad_pixels = [[-245, 0, 454, 454]]

    - Put a bright star near this bad column and confirm it is not picked at
      all.
    - Put a bright star at col = 454 - 20 / 5 - 105 / 5 (dither and search box
      size) and confirm it is picked with search box = 100 arcsec.

    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=np.linspace(8.0, 8.1, 4), n_stars=4)
    stars.add_fake_star(row=-205, col=450, mag=6.0, id=1)
    stars.add_fake_star(row=-205, col=454 - 20 / 5 - 105 / 5, mag=6.0, id=2)

    kwargs = mod_std_info(stars=stars, dark=dark, dither=20,
                          n_guide=8, n_fid=0, n_acq=8, man_angle=90)
    aca = get_aca_catalog(**kwargs)

    # Make sure bad pixels have expected value
    assert np.all(aca.acqs.dark.aca[-245:0, 454] == CHAR.bad_pixel_dark_current)

    exp_ids = [2, 100, 101, 102, 103]
    assert sorted(aca.guides['id']) == exp_ids
    assert aca.acqs['id'].tolist() == exp_ids
    assert aca.acqs['halfw'].tolist() == [100, 160, 160, 160, 160]


configs = [(8.5, 1, 1, 1),
           (10.1, 0, 0, 1),
           (10.25, 0, 0, 0)]


@pytest.mark.parametrize('config', configs)
def test_aca_acq_gui_thumbs_up(config):
    """
    Test the thumbs_up property of aca, acq, and guide selection.
    Fid thumbs up is tested separately.
    """
    mag, acat, guit, acqt = config
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=mag, n_stars=8)
    aca = get_aca_catalog(**mod_std_info(stars=stars, raise_exc=True,
                                         n_acq=8, n_guide=5))

    assert aca.thumbs_up == acat
    assert aca.acqs.thumbs_up == acqt
    assert aca.guides.thumbs_up == guit
    assert aca.fids.thumbs_up == 1


def test_fid_trap_effect():
    # Obsid 1576
    agasc_ids = [367148872, 367139768, 367144424, 367674552, 367657896]
    date = '2001:275'
    att = [10.659376, 40.980028, 181.012903]
    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    cat = get_aca_catalog(obsid=1576, stars=stars, raise_exc=True)
    assert 367674552 not in cat.guides['id']

    # Obsid 2365
    agasc_ids = [1184926344, 1184902704, 1184897704, 1184905208, 1185050656]
    date = '2001:294'
    att = [243.552030, -63.091108, 224.513314]
    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    cat = get_aca_catalog(obsid=2365, stars=stars, raise_exc=True)
    assert 1184897704 not in cat.guides['id']


def test_reject_column_spoilers():
    """
    Test that column spoiler handling is correct for guide, acq and fid selection.
    Also tests not selecting stars that are too bright.
    """
    stars = StarsTable.empty()
    fids = get_fid_catalog(**mod_std_info(stars=stars, detector='HRC-I')).cand_fids
    stars.add_fake_constellation(mag=8.0, n_stars=5)

    def offset(id, drow, dcol, dmag, rmult=1):
        if id < 10:
            star = fids.get_id(id)
        else:
            star = stars.get_id(id)

        return dict(row=(star['row'] + drow * np.sign(star['row'])) * rmult,
                    col=star['col'] + dcol,
                    mag=star['mag'] - dmag)

    # Spoil first star with downstream spoiler that is just in limits
    stars.add_fake_star(**offset(100, drow=70, dcol=9, dmag=4.6))

    # Just miss four others by mag, col, and row
    stars.add_fake_star(**offset(101, drow=70, dcol=11, dmag=4.6))  # Outside column
    stars.add_fake_star(**offset(102, drow=70, dcol=9, dmag=4.4))  # Not bright enough
    stars.add_fake_star(**offset(103, drow=70, dcol=9, dmag=4.6, rmult=-1))  # Wrong side
    stars.add_fake_star(**offset(104, drow=-70, dcol=9, dmag=4.6))  # Upstream

    # Fid spoilers: spoil fid_id=1, 4
    stars.add_fake_star(**offset(1, drow=30, dcol=9, dmag=4.6))
    stars.add_fake_star(**offset(4, drow=30, dcol=9, dmag=4.6))

    # Put in near-miss spoilers for fid_id=2, 3
    stars.add_fake_star(**offset(2, drow=30, dcol=9, dmag=4.4))  # Not bright enough
    stars.add_fake_star(**offset(2, drow=30, dcol=11, dmag=4.6))  # Outside column
    stars.add_fake_star(**offset(3, drow=30, dcol=9, dmag=4.6, rmult=-1))  # Wrong side
    stars.add_fake_star(**offset(3, drow=-30, dcol=9, dmag=4.6))  # Upstream

    kwargs = mod_std_info(stars=stars, n_guide=8, n_fid=3, n_acq=8,
                          dark=DARK40.copy(), detector='HRC-I')
    aca = get_aca_catalog(**kwargs)

    assert aca.fids['id'].tolist() == [2, 3]
    assert 100 not in aca.acqs.cand_acqs['id']
    assert aca.guides['id'].tolist() == [101, 102, 103, 104]
    assert aca.acqs['id'].tolist() == [101, 102, 103, 104]


def test_dense_star_field_regress():
    """
    Test getting stars at the most dense star field in the sky.  Taken from:

    https://github.com/sot/skanb/blob/master/star_selection/dense_sparse_cats.ipynb

    """
    att = (167.0672, -59.1235, 0)
    aca = get_aca_catalog(**mod_std_info(att=att, n_fid=3, n_guide=5, n_acq=8))
    exp = ['slot idx     id     type  sz   yang     zang   dim res halfw mag ',
           '---- --- ---------- ---- --- -------- -------- --- --- ----- ----',
           '   0   1          3  FID 8x8    40.01 -1871.10   1   1    25 7.00',
           '   1   2          4  FID 8x8  2140.23   166.63   1   1    25 7.00',
           '   2   3          5  FID 8x8 -1826.28   160.17   1   1    25 7.00',
           '   3   4 1130889232  BOT 6x6  -251.98 -1971.97  20   1   160 6.99',
           '   4   5 1130893664  BOT 6x6  1530.07 -2149.38  20   1   160 7.62',
           '   5   6 1130899056  BOT 6x6  2386.83 -1808.51  20   1   160 6.24',
           '   6   7 1130898232  GUI 6x6  1244.84  2399.68   1   1    25 7.38',
           '   7   8 1130773616  GUI 6x6 -1713.06  1312.10   1   1    25 7.50',
           '   6   9 1130770696  ACQ 6x6 -1900.42  2359.33  20   1   160 7.35',
           '   7  10 1130890288  ACQ 6x6  2030.55 -2011.89  20   1   160 7.67',
           '   0  11 1130890616  ACQ 6x6  1472.68  -376.72  20   1   160 7.77',
           '   1  12 1130893640  ACQ 6x6    64.32 -1040.81  20   1   160 7.77',
           '   2  13 1130894376  ACQ 6x6  -633.90  1186.80  20   1   160 7.78']

    repr(aca)  # Apply default formats
    assert aca[TEST_COLS + ['mag']].pformat(max_width=-1) == exp


def test_aca_acqs_include_exclude():
    """
    Test include and exclude stars.  This uses a catalog with 11 stars:
    - 8 bright stars from 7.0 to 7.7 mag, where the 7.0 is EXCLUDED
    - 2 faint (but OK) stars 10.0, 10.1 where the 10.0 is INCLUDED
    - 1 very faint (bad) stars 12.0 mag is INCLUDED

    Both the 7.0 and 10.1 would normally get picked either initially
    or swapped in during optimization, and 12.0 would never get picked.
    Check that the final catalog is [7.1 .. 7.7, 10.0, 12.0]

    This is a stripped down version of test_cand_acqs_include_exclude
    in test_acq.py along with test_guides_include_exclude in test_guide.py.
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

    # Define includes and excludes. id=9 is in nominal cand_acqs but not in acqs.
    include_ids = [9, 11]
    include_halfws = [45, 89]
    exp_include_halfws = [60, 80]
    exclude_ids = [1]

    aca = get_aca_catalog(**STD_INFO, stars=stars,
                          include_ids_acq=include_ids,
                          include_halfws_acq=include_halfws,
                          exclude_ids_acq=exclude_ids,
                          include_ids_guide=include_ids,
                          exclude_ids_guide=exclude_ids)
    acqs = aca.acqs
    assert acqs.include_ids == include_ids
    assert acqs.include_halfws == exp_include_halfws
    assert acqs.exclude_ids == exclude_ids
    assert all(id_ in acqs.cand_acqs['id'] for id_ in include_ids)

    assert all(id_ in acqs['id'] for id_ in include_ids)
    assert all(id_ not in acqs['id'] for id_ in exclude_ids)

    assert np.all(acqs['id'] == [2, 3, 4, 5, 6, 7, 9, 11])
    assert np.all(acqs['halfw'] == [160, 160, 160, 160, 160, 160, 60, 80])
    assert np.allclose(acqs['mag'], [7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 10.0, 12.0])

    guides = aca.guides
    assert guides.include_ids == include_ids
    assert guides.exclude_ids == exclude_ids
