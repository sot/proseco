import pytest
import numpy as np

from ..fid import get_fid_positions, get_fid_catalog
from ..acq import get_acq_catalog
from ..core import StarsTable
from .test_common import OBS_INFO, STD_INFO, mod_std_info, DARK40
from .. import characteristics_fid as FID


# Reference fid positions for spoiling tests
FIDS = get_fid_catalog(stars=StarsTable.empty(), **STD_INFO)


def test_get_fid_position():
    """
    Compare computed fid positions to flight values from starcheck reports.
    """
    # Obsid 20975
    yang, zang = get_fid_positions('ACIS-I', focus_offset=0.0, sim_offset=-583)
    fidset = [0, 4, 5]
    assert np.allclose(yang[fidset], [919, -1828, 385], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-874, 1023, 1667], rtol=0, atol=1.1)

    # Obsid 20201
    yang, zang = get_fid_positions('ACIS-I', focus_offset=0.0, sim_offset=0.0)
    fidset = [0, 4, 5]
    assert np.allclose(yang[fidset], [919, -1828, 385], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-844, 1053, 1697], rtol=0, atol=1.1)

    # Obsid 20334
    yang, zang = get_fid_positions('HRC-I', focus_offset=0.0, sim_offset=0.0)
    fidset = [0, 1, 2]
    assert np.allclose(yang[fidset], [-776, 836, -1204], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-1306, -1308, 998], rtol=0, atol=1.1)

    # Obsid 20168
    # Note the focus_offset offset of -70 is not a strong test, since this also
    # passes for focus_offset=-700.  But since code is just copied from Matlab
    # take that as heritage.
    yang, zang = get_fid_positions('HRC-S', focus_offset=-70.0, sim_offset=0.0)
    fidset = [0, 1, 2]
    assert np.allclose(yang[fidset], [-1174, 1224, -1177], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-468, -460, 561], rtol=0, atol=1.1)


def test_get_initial_catalog():
    """Test basic catalog with no stars in field using standard 2-4-5 config."""
    exp = ['<FidTable length=6>',
           '  id    yang     zang     row     col     mag   spoiler_score  idx   slot',
           'int64 float64  float64  float64 float64 float64     int64     int64 int64',
           '----- -------- -------- ------- ------- ------- ------------- ----- -----',
           '    1   922.59 -1737.89 -180.05 -344.10    7.00             0     0   -99',
           '    2  -773.20 -1742.03  160.79 -345.35    7.00             0     1     0',
           '    3    40.01 -1871.10   -2.67 -371.00    7.00             0     2   -99',
           '    4  2140.23   166.63 -424.51   39.13    7.00             0     3     1',
           '    5 -1826.28   160.17  372.97   36.47    7.00             0     4     2',
           '    6   388.59   803.75  -71.49  166.10    7.00             0     5   -99']
    assert repr(FIDS.cand_fids).splitlines() == exp
    assert np.all(FIDS['id'] == [2, 4, 5])

    # Make catalogs with some fake stars (at exactly fid positions) that spoil
    # the fids.
    stars = StarsTable.empty()
    for fid in FIDS.cand_fids:
        stars.add_fake_star(mag=fid['mag'], mag_err=0.1,
                            yang=fid['yang'], zang=fid['zang'])

    # Spoil fids 1, 2
    fids2 = get_fid_catalog(stars=stars[:2], **STD_INFO)
    exp = ['<FidTable length=6>',
           '  id    yang     zang     row     col     mag   spoiler_score  idx   slot',
           'int64 float64  float64  float64 float64 float64     int64     int64 int64',
           '----- -------- -------- ------- ------- ------- ------------- ----- -----',
           '    1   922.59 -1737.89 -180.05 -344.10    7.00             4     0   -99',
           '    2  -773.20 -1742.03  160.79 -345.35    7.00             4     1   -99',
           '    3    40.01 -1871.10   -2.67 -371.00    7.00             0     2     0',
           '    4  2140.23   166.63 -424.51   39.13    7.00             0     3     1',
           '    5 -1826.28   160.17  372.97   36.47    7.00             0     4     2',
           '    6   388.59   803.75  -71.49  166.10    7.00             0     5   -99']
    assert repr(fids2.cand_fids).splitlines() == exp
    assert np.all(fids2['id'] == [3, 4, 5])

    # Spoil fids 1, 2, 3
    fids3 = get_fid_catalog(stars=stars[:3], **STD_INFO)
    assert np.all(fids3['id'] == [4, 5, 6])
    assert fids3.thumbs_up

    # Spoil fids 1, 2, 3, 4 => no initial catalog gets found
    fids4 = get_fid_catalog(stars=stars[:4], **STD_INFO)
    assert len(fids4) == 0
    assert not fids4.thumbs_up
    assert all(name in fids4.colnames for name in ['id', 'yang', 'zang', 'row', 'col'])


def test_n_fid():
    """Test specifying number of fids.
    """
    # Get only 2 fids
    fids = get_fid_catalog(n_fid=2, **STD_INFO)
    assert len(fids) == 2
    assert fids.thumbs_up


@pytest.mark.parametrize('dither_z', [8, 64])
def test_fid_spoiling_acq(dither_z):
    """Test fid spoiling acq.

    Check fid spoiling acq:

    - 20" (4 pix) positional err on fid light
    - 4 pixel readout halfw for fid light
    - 2 pixel PSF of fid light that could creep into search box
    - Acq search box half-width
    - Dither amplitude (since OBC adjusts search box for dither)

    For this case (100" halfw and dither) the threshold for spoiling
    is 20 + 20 + 10 + 100 + dither = 150" + dither.  So this test puts stars at the
    positions of ACIS-S 2, 4, 5 but offset by 82, 149 and 151 arcsec + dither.
    Only ACIS-S-5 is allowed, so we end up with the first fid set using
    1, 3, 5, 6, which is 1, 5, 6.

    """
    dither_y = 8
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5)

    for fid, offset in zip(FIDS[:3], [82, 149, 151]):
        stars.add_fake_star(yang=fid['yang'] + offset + dither_y,
                            zang=fid['zang'] + offset + dither_z,
                            mag=7.0)

    std_info = STD_INFO.copy()
    std_info['dither'] = (dither_y, dither_z)
    acqs = get_acq_catalog(stars=stars, **std_info)
    acqs['halfw'] = 100
    fids5 = get_fid_catalog(acqs=acqs, **std_info)
    exp = ['<FidTable length=6>',
           '  id    yang     zang     row     col     mag   spoiler_score  idx   slot',
           'int64 float64  float64  float64 float64 float64     int64     int64 int64',
           '----- -------- -------- ------- ------- ------- ------------- ----- -----',
           '    1   922.59 -1737.89 -180.05 -344.10    7.00             0     0     0',
           '    2  -773.20 -1742.03  160.79 -345.35    7.00             0     1   -99',
           '    3    40.01 -1871.10   -2.67 -371.00    7.00             0     2   -99',
           '    4  2140.23   166.63 -424.51   39.13    7.00             0     3   -99',
           '    5 -1826.28   160.17  372.97   36.47    7.00             0     4     1',
           '    6   388.59   803.75  -71.49  166.10    7.00             0     5     2']

    assert repr(fids5.cand_fids).splitlines() == exp


def test_fid_mult_spoilers():
    """
    Test of fix for bug #54.  19605 and 20144 were previous crashing.
    """
    acqs = get_acq_catalog(**OBS_INFO[19605])
    fids = get_fid_catalog(acqs=acqs, **OBS_INFO[19605])
    cand_fids = fids.cand_fids
    assert np.all(cand_fids['spoiler_score'] == [0, 0, 1, 4, 0, 0])
    assert len(cand_fids['spoilers'][2]) == 1
    assert cand_fids['spoilers'][2]['warn'][0] == 'yellow'


def test_dither_as_sequence():
    """
    Test that calling get_acq_catalog with a 2-element sequence (dither_y, dither_z)
    gives the expected response.  (Basically that it still returns a catalog).
    """
    std_info = STD_INFO.copy()
    std_info['dither'] = (8, 22)
    fids = get_fid_catalog(**std_info)
    assert len(fids) == 3
    assert fids.dither_acq == (8, 22)
    assert fids.dither_guide == (8, 22)


def test_fid_spoiler_score():
    """Test computing the fid spoiler score."""
    dither_y = 8
    dither_z = 64
    stars = StarsTable.empty()
    for fid, offset in zip(FIDS[:2], [-1, 1]):
        stars.add_fake_star(yang=fid['yang'] + FID.spoiler_margin + dither_y + offset,
                            zang=fid['zang'] + FID.spoiler_margin + dither_z + offset,
                            mag=7.0)

    dither = (dither_y, dither_z)

    std_info = STD_INFO.copy()
    std_info['dither'] = dither
    fids = get_fid_catalog(stars=stars, **std_info)
    assert np.all(fids.cand_fids['spoiler_score'] == [0, 4, 0, 0, 0, 0])


def test_big_sim_offset():
    """Test of an observation with a big SIM offset"""
    fids = get_fid_catalog(**mod_std_info(stars=StarsTable.empty(), sim_offset=300000))
    names = ['id', 'yang', 'zang', 'row', 'col', 'mag', 'spoiler_score', 'idx']
    assert all(name in fids.colnames for name in names)


def test_fid_hot_pixel_reject():
    """Test hot pixel rejecting a fid"""
    lim = FID.hot_pixel_spoiler_limit
    dark = DARK40.copy()
    for fid_id, off, dc in [(1, 8.0, lim * 1.05),  # spoiler,
                            (2, 12.0, lim * 1.05),  # not spoiler (spatially)
                            (3, 8.0, lim * 0.95),  # not spoiler (dark current too low)
                            (4, -8.0, lim * 1.05),  # spoiler
                            (5, 0.0, lim * 1.05)]:  # spoiler
        fid = FIDS.cand_fids.get_id(fid_id)
        r = int(round(fid['row'] + off))
        c = int(round(fid['col'] + off))
        dark.aca[r, c] = dc

    fids = get_fid_catalog(stars=StarsTable.empty(), dark=dark, **STD_INFO)
    assert fids['id'].tolist() == [2, 3, 6]
