# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import os

import matplotlib
from Quaternion import Quat

matplotlib.use('agg')  # noqa

import pickle
import pytest
from pathlib import Path

import numpy as np
import mica.starcheck
import agasc

from .test_common import STD_INFO, mod_std_info, DARK40, OBS_INFO
from ..core import StarsTable, includes_for_obsid, ACACatalogTable
from ..catalog import get_aca_catalog, ACATable
from ..fid import get_fid_catalog, FidTable
from .. import characteristics as ACA

HAS_SC_ARCHIVE = Path(mica.starcheck.starcheck.FILES['data_root']).exists()
TEST_COLS = 'slot idx id type sz yang zang dim res halfw'.split()

# Do not use the AGASC supplement in testing by default since mags can change
os.environ[agasc.SUPPLEMENT_ENABLED_ENV] = 'False'

HAS_MAG_SUPPLEMENT = len(agasc.get_supplement_table('mags')) > 0


def test_allowed_kwargs():
    """Test #332 where allowed_kwargs class attribute is unique for each subclass"""
    new_kwargs = ACATable.allowed_kwargs - ACACatalogTable.allowed_kwargs
    assert new_kwargs == {'call_args', 'version', 't_ccd_eff_acq', 't_ccd_eff_guide',
                          't_ccd_penalty_limit', 'duration', 'target_name'}

    new_kwargs = FidTable.allowed_kwargs - ACACatalogTable.allowed_kwargs
    assert new_kwargs == {'acqs', 'include_ids', 'exclude_ids'}


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason='Test requires starcheck archive')
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


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason='Test requires starcheck archive')
@pytest.mark.skipif(not HAS_MAG_SUPPLEMENT, reason='No estimated mags in AGASC supplement')
def test_get_aca_catalog_20603_with_supplement():
    """Test that results for 20603 are different if the AGASC supplement is used.
    """
    kwargs = dict(obsid=20603, exclude_ids_acq=[40113544],
                  n_fid=2, n_guide=6, n_acq=7, raise_exc=True)
    aca_no = get_aca_catalog(**kwargs)
    with agasc.set_supplement_enabled(True):
        aca = get_aca_catalog(**kwargs)

    assert (len(aca_no.guides) != len(aca.guides)
            or np.any(aca_no.guides['mag'] != aca.guides['mag']))
    assert (len(aca_no.acqs) != len(aca.acqs)
            or np.any(aca_no.acqs['mag'] != aca.acqs['mag']))


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason='Test requires starcheck archive')
def test_get_aca_catalog_20603():
    """Put it all together.  Regression test for selected stars.
    """
    # Force not using a bright star so there is a GUI-only (not BOT) star
    aca = get_aca_catalog(20603, exclude_ids_acq=[40113544], n_fid=2, n_guide=6, n_acq=7,
                          raise_exc=True)
    # Expected 2 fids, 4 guide, 7 acq
    exp = ['slot idx     id    type  sz   yang     zang   dim res halfw',
           '---- --- --------- ---- --- -------- -------- --- --- -----',
           '   0   1         4  FID 8x8  2140.23   166.63   1   1    25',
           '   1   2         5  FID 8x8 -1826.28   160.17   1   1    25',
           '   2   3 116791824  BOT 6x6   622.00  -953.60  28   1   160',
           '   3   4  40114416  BOT 6x6   394.22  1204.43  24   1   140',
           '   4   5  40112304  BOT 6x6 -1644.35  2032.47  12   1    80',
           '   5   6  40113544  GUI 6x6   102.74  1133.37   1   1    25',
           '   0   7 116923496  ACQ 6x6 -1337.79  1049.27  20   1   120',
           '   1   8 116923528  ACQ 6x6 -2418.65  1088.40  28   1   160',
           '   5   9 116791744  ACQ 6x6   985.38 -1210.19  28   1   160',
           '   6  10  40108048  ACQ 6x6     2.21  1619.17  24   1   140']

    assert aca[TEST_COLS].pformat(max_width=-1) == exp

    aca_pkl = pickle.dumps(aca)
    assert len(aca_pkl) < 180_000  # Nominally ~170k, warn if size grows

    # Test that plotting succeeds
    aca.plot()
    aca.acqs.plot()
    aca.guides.plot()
    aca.stars.plot()
    aca.fids.plot()

    assert aca.dark_date == '2018:100'


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason='Test requires starcheck archive')
def test_get_aca_catalog_20259():
    """
    Test obsid 20259 which has two spoiled fids: HRC-2 is yellow and HRC-4 is red.
    Expectation is to choose fids 1, 2, 3 (not 4).

    Also do a test that set_stars() processing is behaving as expected.

    """
    aca = get_aca_catalog(20259, raise_exc=True)
    exp = ['slot idx     id    type  sz   yang     zang   dim res halfw',
           '---- --- --------- ---- --- -------- -------- --- --- -----',
           '   0   1         1  FID 8x8 -1175.03  -468.23   1   1    25',
           '   1   2         2  FID 8x8  1224.70  -460.93   1   1    25',
           '   2   3         3  FID 8x8 -1177.69   561.30   1   1    25',
           '   3   4 896009152  BOT 6x6  1693.39   217.92  16   1   100',
           '   4   5 897712576  BOT 6x6 -1099.95  2140.23  12   1    80',
           '   5   6 897717296  BOT 6x6   932.58  1227.48  12   1    80',
           '   6   7 896013056  BOT 6x6  1547.25 -2455.12  12   1    80',
           '   7   8 896009240  BOT 6x6  -911.41   402.62  12   1    80',
           '   0   9 896011576  ACQ 6x6   810.99   -69.21  16   1   100',
           '   1  10 897718208  ACQ 6x6   765.61  1530.27  12   1    80',
           '   2  11 897192352  ACQ 6x6 -2110.43  2005.21  12   1    80']

    assert aca[TEST_COLS].pformat(max_width=-1) == exp

    # Check that acqs, guides, and fids are sharing the same stars table
    # but that it is different from the larger aca stars table.
    assert aca.stars is not aca.acqs.stars
    assert aca.fids.stars is aca.acqs.stars
    assert aca.guides.stars is aca.acqs.stars


def test_exception_handling():
    """
    Test top-level exception catching.
    """
    aca = get_aca_catalog(att=(0, 0, 0), man_angle=10, date='2018:001',
                          dither_acq=(8, 8), dither_guide=(8, 8),
                          t_ccd_acq=-10, t_ccd_guide=-10,
                          detector='ACIS-S', sim_offset=0, focus_offset=0,
                          n_guide=8, n_fid=3, n_acq=8,
                          include_ids_acq=[1], include_halfws_acq=[100, 120],
                          raise_exc=False)  # Fail
    assert 'include_ids and include_halfws must have same length' in aca.exception

    for obj in (aca, aca.acqs, aca.guides, aca.fids):
        assert len(obj) == 0
        assert 'id' in obj.colnames
        assert 'idx' in obj.colnames
        if obj.name == 'acqs':
            assert 'halfw' in obj.colnames


def test_unhandled_exception():
    with pytest.raises(ValueError, match=r'missing required parameters'):
        # TypeError in get_starcheck_catalog due to NoneType obsid
        get_aca_catalog(obsid=None, raise_exc=True)

    with pytest.raises(ValueError, match=r'missing required parameters'):
        # Obsid 0 implies all all pars must be provided
        get_aca_catalog(obsid=0, raise_exc=True)

    with pytest.raises(ValueError, match=r'missing required parameters'):
        # Obsid > 0 implies missing pars are to be found in starcheck archive,
        # but this one will certainly not be found.
        get_aca_catalog(obsid=99999, raise_exc=True)


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
    acas = get_aca_catalog(**test_info, stars=stars, raise_exc=False)

    assert 'id' in acas.acqs.colnames
    assert 'halfw' in acas.acqs.colnames
    assert 'id' in acas.guides.colnames
    assert 'id' in acas.fids.colnames


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason='Test requires starcheck archive')
def test_big_dither_from_mica_starcheck():
    """
    Test code that infers dither_acq and dither_guide for a big-dither
    observation like 20168.
    """
    aca = ACATable()
    aca.set_attrs_from_kwargs(obsid=20168)

    assert aca.detector == 'HRC-S'
    assert aca.dither_acq == (20, 20)
    assert aca.dither_guide == (64, 8)


def test_pickle():
    """Test that ACA, guide, acq, and fid catalogs round-trip through pickling.

    Known attributes that do NOT round-trip are below.  None of these are
    required for post-facto catalog evaluation and currently the reporting code
    handles ``stars`` and ``dark``.

    - stars
    - dark
    - aca.fids.acqs

    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=10.0, n_stars=5)
    aca = get_aca_catalog(stars=stars, dark=DARK40, raise_exc=True, **STD_INFO)

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

        for attr in ['att', 'date', 't_ccd', 'man_angle', 'dither']:
            val = getattr(obj, attr)
            val2 = getattr(obj2, attr)
            if isinstance(val, float):
                assert np.isclose(val, val2)
            elif isinstance(val, Quat):
                assert np.allclose(val.q, val2.q)
            else:
                assert val == val2

    # Test that calc_p_safe() gives the same answer, which implicitly tests
    # that the AcqTable.__setstate__ unpickling code has the right (weak)
    # reference to acqs within each AcqProbs object.  This also tests
    # that acqs.p_man_err and acqs.fid_set are the same.
    assert np.isclose(aca.acqs.calc_p_safe(), aca2.acqs.calc_p_safe(),
                      atol=0, rtol=1e-6)
    assert aca.acqs.fid_set == aca2.acqs.fid_set


def test_copy_deepcopy_pickle():
    """
    Test that copy, deepcopy and pickle all return the expected object which
    is independent of the original (where expected).

    :return:
    """
    aca = get_aca_catalog(**STD_INFO)

    def f1(x):
        return pickle.loads(pickle.dumps(x))

    f2 = copy.deepcopy
    f3 = copy.copy

    def f4(x):
        return x.__class__(x)

    for func in (f1, f2, f3, f4):
        aca2 = func(aca)

        # Functional test for #303, mostly just for pickle.
        assert aca2.dark_date == '2017:272'

        for attr in ('acqs', 'guides', 'fids'):
            val = getattr(aca, attr)
            val2 = getattr(aca2, attr)
            # New table appears the same but is not the same object
            assert repr(val) == repr(val2)
            assert val is not val2

            # Now do the copy func on the lower level table directly
            val2 = func(val)
            assert repr(val) == repr(val2)
            assert val is not val2


def test_clip_maxmag():
    """Test that clipping maxmag for guide and acq stars works
    """
    stars = StarsTable.empty()
    mag0 = ACA.max_maxmag - 1.5  # nominal star mag when clipping occurs (11.2 - 1.5 = 9.7)
    mags_acq = np.array([-1.5, -1, -0.5, -0.01, 0.01, 0.2, 0.3, 0.4]) + mag0
    mags_guide = np.array([-0.5, -0.01, 0.01, 0.2, 0.3]) + mag0
    stars.add_fake_constellation(mag=mags_acq, n_stars=8, size=2000)
    stars.add_fake_constellation(mag=mags_guide, n_stars=5, size=1000)
    aca = get_aca_catalog(stars=stars, dark=DARK40, raise_exc=True,
                          exclude_ids_guide=np.arange(100, 108),
                          exclude_ids_acq=np.arange(108, 113),
                          **STD_INFO)

    assert np.all(aca['maxmag'] <= ACA.max_maxmag)

    ok = aca['type'] == 'FID'
    assert np.allclose(aca['maxmag'][ok], 8.0)

    ok = aca['type'] == 'GUI'
    assert np.allclose(aca['maxmag'][ok], (mags_guide + 1.5).clip(None, ACA.max_maxmag))

    ok = aca['type'] == 'ACQ'
    assert np.allclose(aca['maxmag'][ok], (mags_acq + 1.5).clip(None, ACA.max_maxmag))


def test_big_sim_offset():
    """
    Check getting a catalog for a large SIM offset that means there are
    no candidates.

    Bonus: check that duration and target_name can be set.
    """
    aca = get_aca_catalog(**mod_std_info(sim_offset=200000, duration=10000,
                                         target_name='Target Name', raise_exc=True))
    assert len(aca.acqs) == 8
    assert len(aca.guides) == 5
    assert len(aca.fids) == 0
    names = ['id', 'yang', 'zang', 'row', 'col', 'mag', 'spoiler_score', 'idx']
    assert all(name in aca.fids.colnames for name in names)
    assert aca.duration == 10000
    assert aca.target_name == 'Target Name'


@pytest.mark.parametrize('call_t_ccd', [True, False])
def test_calling_with_t_ccd_acq_guide(call_t_ccd):
    """Test that calling get_aca_catalog with t_ccd or t_ccd_acq/guide args sets all
    CCD attributes correctly in the nominal case of a temperature
    below the penalty limit.

    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=8.0, n_stars=8)

    t_ccd = np.trunc(ACA.aca_t_ccd_penalty_limit - 1.0)

    if call_t_ccd:
        # Call with just t_ccd=t_ccd
        t_ccd_guide = t_ccd
        t_ccd_acq = t_ccd
        ccd_kwargs = {'t_ccd': t_ccd}
    else:
        # Call with separate values
        t_ccd_guide = t_ccd
        t_ccd_acq = t_ccd - 1
        ccd_kwargs = {'t_ccd_acq': t_ccd_acq, 't_ccd_guide': t_ccd_guide}

    kwargs = mod_std_info(stars=stars, dark=dark, **ccd_kwargs)
    aca = get_aca_catalog(**kwargs)

    assert aca.t_ccd == t_ccd_guide
    assert aca.t_ccd_acq == t_ccd_acq
    assert aca.t_ccd_guide == t_ccd_guide

    assert aca.t_ccd_eff_acq == t_ccd_acq
    assert aca.t_ccd_eff_guide == t_ccd_guide

    assert aca.acqs.t_ccd == t_ccd_acq
    assert aca.acqs.t_ccd_acq == t_ccd_acq
    assert aca.acqs.t_ccd_guide == t_ccd_guide

    assert aca.guides.t_ccd == t_ccd_guide
    assert aca.guides.t_ccd_guide == t_ccd_guide
    assert aca.guides.t_ccd_acq == t_ccd_acq

    assert aca.fids.t_ccd == t_ccd_guide
    assert aca.fids.t_ccd_guide == t_ccd_guide
    assert aca.fids.t_ccd_acq == t_ccd_acq


t_ccd_cases = [(-0.5, 0, 0),
               (0, 0, 0),
               (0.5, 1.5, 1.4)]


@pytest.mark.parametrize('t_ccd_case', t_ccd_cases)
def test_t_ccd_effective_acq_guide(t_ccd_case):
    """Test setting of effective T_ccd temperatures for cases above and
    below the penalty limit.

    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=8.0, n_stars=8)

    t_limit = ACA.aca_t_ccd_penalty_limit

    t_offset, t_penalty_acq, t_penalty_guide = t_ccd_case
    # Set acq and guide temperatures different
    t_ccd_acq = t_limit + t_offset
    t_ccd_guide = t_ccd_acq - 0.1

    kwargs = mod_std_info(stars=stars, t_ccd_acq=t_ccd_acq, t_ccd_guide=t_ccd_guide)
    aca = get_aca_catalog(**kwargs)
    assert aca.t_ccd_penalty_limit == t_limit

    assert np.isclose(aca.t_ccd_acq, t_ccd_acq)
    assert np.isclose(aca.t_ccd_guide, t_ccd_guide)

    # t_ccd + 1 + (t_ccd - t_limit) from proseco.catalog.get_effective_t_ccd()
    assert np.isclose(aca.t_ccd_eff_acq, t_ccd_acq + t_penalty_acq)
    assert np.isclose(aca.t_ccd_eff_guide, t_ccd_guide + t_penalty_guide)

    assert np.isclose(aca.t_ccd_eff_acq, aca.acqs.t_ccd)
    assert np.isclose(aca.t_ccd_eff_guide, aca.guides.t_ccd)


@pytest.mark.parametrize('t_ccd_case', t_ccd_cases)
def test_t_ccd_effective_acq_guide_via_kwarg(t_ccd_case):
    """Test setting of effective T_ccd temperatures for cases above and
    below a manually specified limit
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=8.0, n_stars=8)

    t_limit = -np.pi  # some number different from ACA.aca_t_ccd_penalty_limit

    t_offset, t_penalty_acq, t_penalty_guide = t_ccd_case
    # Set acq and guide temperatures different
    t_ccd_acq = t_limit + t_offset
    t_ccd_guide = t_ccd_acq - 0.1

    kwargs = mod_std_info(stars=stars, t_ccd_acq=t_ccd_acq, t_ccd_guide=t_ccd_guide)
    kwargs['t_ccd_penalty_limit'] = t_limit
    aca = get_aca_catalog(**kwargs)

    assert aca.t_ccd_penalty_limit == t_limit

    assert np.isclose(aca.t_ccd_acq, t_ccd_acq)
    assert np.isclose(aca.t_ccd_guide, t_ccd_guide)

    # t_ccd + 1 + (t_ccd - t_limit) from proseco.catalog.get_effective_t_ccd()
    assert np.isclose(aca.t_ccd_eff_acq, t_ccd_acq + t_penalty_acq)
    assert np.isclose(aca.t_ccd_eff_guide, t_ccd_guide + t_penalty_guide)

    assert np.isclose(aca.t_ccd_eff_acq, aca.acqs.t_ccd)
    assert np.isclose(aca.t_ccd_eff_guide, aca.guides.t_ccd)


def test_call_args_attr():
    aca = get_aca_catalog(**mod_std_info(optimize=False, n_guide=0, n_acq=0, n_fid=0),
                          raise_exc=False)
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
    # Expects this to be starcheck catalog
    aca = get_aca_catalog(obsid='blah blah', raise_exc=False)
    assert 'ValueError: text does not have OBSID' in aca.exception


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
    # For acq (105 arcsecs away)
    stars.add_fake_star(row=-205, col=454 - 20 / 5 - 105 / 5, mag=6.0, id=2)

    # For guide: 5 pixels + dither away (reject), 6 pixels + dither away (accept)
    stars.add_fake_star(row=-150, col=454 - 5 - 20 / 5, mag=6.2, id=3)  # reject
    stars.add_fake_star(row=-100, col=454 - 6 - 20 / 5, mag=6.2, id=4)  # accept

    kwargs = mod_std_info(stars=stars, dark=dark, dither=20,
                          n_guide=8, n_fid=0, n_acq=8, man_angle=90)
    aca = get_aca_catalog(**kwargs)

    # Make sure bad pixels have expected value
    assert np.all(aca.acqs.dark[-245 + 512:512, 454 + 512] == ACA.bad_pixel_dark_current)

    exp_ids = [2, 100, 101, 102, 103]
    assert sorted(aca.guides['id']) == sorted(exp_ids + [4])
    assert aca.acqs['id'].tolist() == exp_ids
    assert aca.acqs['halfw'].tolist() == [100, 160, 160, 160, 160]


def test_fid_trap_effect():
    """Test that guide stars impacted by fid trap effect are excluded.

    This uses two flight obsids that showed this issue.  See:
    http://cxc.cfa.harvard.edu/mta/ASPECT/aca_weird_pixels/

    """
    # Obsid 1576
    agasc_ids = [367148872, 367139768, 367144424, 367674552, 367657896]
    att = [10.659376, 40.980028, 181.012903]
    stars = StarsTable.from_agasc_ids(att, agasc_ids)
    cat = get_aca_catalog(obsid=1576, stars=stars, raise_exc=True)
    assert 367674552 not in cat.guides['id']

    # Obsid 2365
    # NOTE: the att below is in fact the actual 2365 attitude, which differs
    # from what mica.starcheck reports as [243.598372, -63.123245, 123.674233].
    # See: https://github.com/sot/mica/issues/184
    agasc_ids = [1184926344, 1184902704, 1184897704, 1184905208, 1185050656]
    att = [243.552030, -63.091108, 224.513314]
    stars = StarsTable.from_agasc_ids(att, agasc_ids)

    # Specify att kwarg explicitly to override what is found in mica.starcheck
    cat = get_aca_catalog(obsid=2365, stars=stars, raise_exc=True, att=att)
    assert 1184897704 not in cat.guides['id']


def test_monitors_and_target_offset_args():
    """
    Test #328 to add monitors and target_offset args to API.
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=8.0, n_stars=8)
    aca = get_aca_catalog(**mod_std_info(stars=stars, dark=DARK40))
    assert aca.monitors is None
    assert aca.target_offset == (0.0, 0.0)

    # Both of these are brighter than the brightest bona-fide star, stressing
    # the processing somewhat.
    monitors = [[-1700, 1900, ACA.MonCoord.YAGZAG, 7.5, ACA.MonFunc.MON_FIXED],
                [500, -500, ACA.MonCoord.YAGZAG, 7.5, ACA.MonFunc.MON_TRACK]]
    target_offset = (0.05, 0.1)
    aca = get_aca_catalog(**mod_std_info(monitors=monitors,
                                         n_guide=6, n_fid=0,
                                         target_offset=target_offset,
                                         stars=stars, dark=DARK40))
    exp = [' coord0 coord1 coord_type mag function',
           '------- ------ ---------- --- --------',
           '-1700.0 1900.0          2 7.5        3',
           '  500.0 -500.0          2 7.5        2']
    assert aca.monitors.pformat_all() == exp
    assert aca.target_offset is target_offset


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


def test_dark_property():
    """
    Test that in the case of a common t_ccd, all the dark current maps are
    actually the same object.

    :return: None
    """
    aca = get_aca_catalog(**STD_INFO)
    for attr in ('acqs', 'guides', 'fids'):
        assert aca.dark is getattr(aca, attr).dark

    kwargs = STD_INFO.copy()
    del kwargs['t_ccd']
    kwargs['t_ccd_acq'] = -12.5
    kwargs['t_ccd_guide'] = -11.5
    aca = get_aca_catalog(**kwargs)
    assert aca.dark is aca.guides.dark
    assert aca.dark is aca.fids.dark
    assert aca.dark is not aca.acqs.dark
    assert aca.dark.mean() > aca.acqs.dark.mean()


def test_dense_star_field_regress():
    """
    Test getting stars at the most dense star field in the sky.  Taken from:

    https://github.com/sot/skanb/blob/master/star_selection/dense_sparse_cats.ipynb

    """
    att = (167.0672, -59.1235, 0)
    aca = get_aca_catalog(**mod_std_info(att=att, n_fid=3, n_guide=5, n_acq=8))
    exp = ['slot idx     id     type  sz   yang     zang   dim res halfw  mag ',
           '---- --- ---------- ---- --- -------- -------- --- --- ----- -----',
           '   0   1          3  FID 8x8    40.01 -1871.10   1   1    25  7.00',
           '   1   2          4  FID 8x8  2140.23   166.63   1   1    25  7.00',
           '   2   3          5  FID 8x8 -1826.28   160.17   1   1    25  7.00',
           '   3   4 1130899056  BOT 6x6  2386.83 -1808.51  28   1   160  6.24',
           '   4   5 1130889232  BOT 6x6  -251.98 -1971.97  28   1   160  6.99',
           '   5   6 1130893664  BOT 6x6  1530.07 -2149.38  28   1   160  7.62',
           '   6   7 1130898232  GUI 6x6  1244.84  2399.68   1   1    25  7.38',
           '   7   8 1130773616  GUI 6x6 -1713.06  1312.10   1   1    25  7.50',
           '   0   9 1130770696  ACQ 6x6 -1900.42  2359.33  28   1   160  7.35',
           '   1  10 1130890288  ACQ 6x6  2030.55 -2011.89  28   1   160  7.67',
           '   2  11 1130890616  ACQ 6x6  1472.68  -376.72  28   1   160  7.77',
           '   6  12 1130893640  ACQ 6x6    64.32 -1040.81  28   1   160  7.77',
           '   7  13 1130894376  ACQ 6x6  -633.90  1186.80  28   1   160  7.78']
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


def test_report_from_objects(tmpdir):
    """
    Test making guide and acq reports without the intermediate pickle.

    This is just a sanity check that appropriate files are created.  More
    detailed testing (including pickle round trip) is tested in test_guide
    and test_acq.
    """
    rootdir = Path(tmpdir)

    obsid = 20603
    aca = get_aca_catalog(**OBS_INFO[obsid])
    aca.guides.make_report(rootdir=rootdir)
    aca.acqs.make_report(rootdir=rootdir)

    obsdir = rootdir / f'obs{obsid:05}'
    for subdir in 'acq', 'guide':
        outdir = obsdir / subdir
        assert (outdir / 'index.html').exists()
        assert len(list(outdir.glob('*.png'))) > 0


def test_force_catalog_from_starcheck():
    """
    Test forcing a catalog from starcheck output.
    """
    obs = """
    OBSID: 20551  1RXSJ235324.6-125657   ACIS-I SIM Z offset:-1165 (-2.93mm) Grating: NONE
    RA, Dec, Roll (deg):   358.341787   -12.949882   276.997597
    Dither: ON  Y_amp= 8.0  Z_amp= 8.0  Y_period=1000.0  Z_period= 707.1
    BACKSTOP GUIDE_SUMM OR MANVR DOT MAKE_STARS TLR

    MP_TARGQUAT at 2019:009:21:16:11.292 (VCDU count = 13351231)
      Q1,Q2,Q3,Q4: -0.65711794  0.09397558  0.06394855  0.74516789
      MANVR: Angle=  89.16 deg  Duration= 1849 sec  Slew err= 55.7 arcsec  End= 2019:009:21:46:55

    MP_STARCAT at 2019:009:21:16:12.935 (VCDU count = 13351237)
    ---------------------------------------------------------------------------------------------
     IDX SLOT        ID  TYPE   SZ   P_ACQ    MAG   MAXMAG   YANG   ZANG DIM RES HALFW PASS NOTES
    ---------------------------------------------------------------------------------------------
    [ 1]  0           1   FID  8x8     ---   7.000   8.000    919   -904   1   1   25
    [ 2]  1           4   FID  8x8     ---   7.000   8.000   2140    995   1   1   25
    [ 3]  2           5   FID  8x8     ---   7.000   8.000  -1828    993   1   1   25
    [ 4]  3   764677800   BOT  6x6   0.372  10.263  11.766  -2101   -899  20   1  120  a4g4
    [ 5]  4   765069008   BOT  6x6   0.988   9.030  10.531   2317   2263  28   1  160    a2
    [ 6]  5   765069552   BOT  6x6   0.669   9.972  11.469    120    736  28   1  160  a3g3
    [ 7]  6   765069664   BOT  6x6   0.664  10.060  11.562    209   1414  20   1  120  a3g3
    [ 8]  7   765069712   BOT  6x6   0.998   6.295   7.797    983   1993  28   1  160  a2g2
    [ 9]  0   765067472   ACQ  6x6   0.305  10.330  11.828   -753   -808  20   1  120    a4
    [10]  1   765068968   ACQ  6x6   0.185  10.471  11.969    594   -529  20   1  120    a4
    [11]  2   765070392   ACQ  6x6   0.261  10.378  11.875   -175    756  20   1  120    a4

    >> WARNING: Probability of 2 or fewer stars > 0.008
    >> WARNING: [ 4] Magnitude. Acq star 10.263
    >> WARNING: [ 9] Magnitude. Acq star 10.330
    >> WARNING: [10] Magnitude. Acq star 10.471
    >> WARNING: [11] Magnitude. Acq star 10.378
    >> WARNING: [ 7] Magnitude. Acq star 10.060

    Probability of acquiring 2,3, and 4 or fewer stars (10^x):	-1.499	-0.703	-0.275
    Acquisition Stars Expected  : 4.44
    Predicted Max CCD temperature: -9.9 C    N100 Warm Pix Frac 0.286
    Dynamic Mag Limits: Yellow 9.99          Red 10.17"""

    aca = get_aca_catalog(obs + '--force-catalog')
    assert aca['id'].tolist() == [1, 4, 5,
                                  765069712,
                                  765069008,
                                  765069552,
                                  765069664,
                                  764677800,
                                  765067472,
                                  765070392,
                                  765068968]
    assert aca['type'].tolist() == ['FID', 'FID', 'FID',
                                    'BOT', 'BOT', 'BOT', 'BOT', 'BOT',
                                    'ACQ', 'ACQ', 'ACQ']
    assert aca['halfw'].tolist() == [25, 25, 25,
                                     160, 160, 160, 120, 120,
                                     120, 120, 120]
    assert np.allclose(aca.att.equatorial, [358.341787, -12.949882, 276.997597], rtol=0, atol=1e-6)
    assert np.allclose(aca.acqs.man_angle, 89.16)


@pytest.mark.skipif(not HAS_SC_ARCHIVE, reason='Test requires starcheck archive')
def test_includes_for_obsid():
    """
    Test helper function to get the include_* kwargs for forcing a catalog.
    """
    exp = {'include_halfws_acq': [120, 120, 120, 120, 85, 120, 120, 120],
           'include_ids_acq': [31075128,
                               31076560,
                               31463496,
                               31983336,
                               32374896,
                               31075368,
                               31982136,
                               32375384],
           'include_ids_guide': [31075128, 31076560, 31463496, 31983336, 32374896],
           'include_ids_fid': [1, 5, 6]}

    out = includes_for_obsid(8008)
    assert out == exp


def test_dark_date_warning():
    aca = get_aca_catalog(**STD_INFO)
    acap = pickle.loads(pickle.dumps(aca))
    assert acap.dark_date == '2017:272'

    # Fudge date forward, after the 2018:002 dark cal
    acap.date = '2018:010'
    with pytest.warns(None) as warns:
        acap.dark  # Accessing the `dark` property triggers code to read it (and warn)

    assert len(warns) == 1
    assert 'Unexpected dark_date: dark_id nearest dark_date' in str(warns[0].message)


def test_img_size_guide():
    """
    Test img_size_guide for setting guide star readout image size
    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    stars.add_fake_constellation(mag=8.0, n_stars=8)

    # Confirm that for an inferred ER, boxes are 8x8
    aca = get_aca_catalog(**mod_std_info(stars=stars, dark=dark, n_fid=0))
    assert np.all(aca.guides['sz'] == '8x8')
    assert aca.guides.img_size is None

    # Confirm that for an inferred OR, boxes are 6x6
    aca = get_aca_catalog(**mod_std_info(stars=stars, dark=dark, n_fid=3))
    assert np.all(aca.guides['sz'] == '6x6')
    assert aca.guides.img_size is None

    # Confirm that for explicit img_size_guide of 8 boxes are '8x8'
    aca = get_aca_catalog(**mod_std_info(stars=stars, dark=dark, n_fid=3, img_size_guide=8))
    assert np.all(aca.guides['sz'] == '8x8')
    assert aca.guides.img_size == 8

    # Confirm that for explicit img_size_guide of 6 boxes are '6x6'
    aca = get_aca_catalog(**mod_std_info(stars=stars, dark=dark, n_fid=0, img_size_guide=6))
    assert np.all(aca.guides['sz'] == '6x6')
    assert aca.guides.img_size == 6

    # Confirm that for explicit img_size_guide of 6 boxes are '6x6'
    aca = get_aca_catalog(**mod_std_info(stars=stars, dark=dark, n_fid=0, img_size_guide=4))
    assert np.all(aca.guides['sz'] == '4x4')
    assert aca.guides.img_size == 4

    with pytest.raises(ValueError, match='img_size must be 4, 6, 8, or None'):
        get_aca_catalog(**mod_std_info(stars=stars, dark=dark, img_size_guide=3))


def test_dyn_bgd_star_bonus():
    stars = StarsTable.empty()

    stars.add_fake_constellation(mag=[9.5] * 3,
                                 size=2000, n_stars=3)
    stars.add_fake_constellation(mag=[10.3, 10.4, 10.5, 10.6, 10.7, 12.0],
                                 size=1500, n_stars=6)

    aca_leg = get_aca_catalog(**STD_INFO, dark=DARK40, stars=stars, dyn_bgd_n_faint=0)
    aca_dyn = get_aca_catalog(**STD_INFO, dark=DARK40, stars=stars, dyn_bgd_n_faint=2,
                              dyn_bgd_dt_ccd=-4.0)
    assert len(aca_leg.guides) == 3
    assert len(aca_dyn.guides) == 5
    assert np.allclose(aca_leg.guides['mag'], [9.5, 9.5, 9.5])
    assert np.allclose(aca_dyn.guides['mag'], [9.5, 9.5, 9.5, 10.3, 10.4])


def test_man_angle_away():
    aca1 = get_aca_catalog(**STD_INFO)
    aca2 = get_aca_catalog(**STD_INFO, man_angle_next=5)
    assert np.all(aca1['id'] == aca2['id'])
    assert np.all(aca1.acqs['halfw'] == aca2.acqs['halfw'])

    # Confirm that the man_angle_next attribute has the default value
    # for aca1 (180) and the specified kw value for aca2
    assert aca1.man_angle_next == 180
    assert aca2.man_angle_next == 5
