import numpy as np

from chandra_aca.transform import yagzag_to_pixels

from ..fid import get_fid_positions, get_fid_catalog
from ..acq import get_acq_catalog
from ..core import StarsTable
from .test_common import OBS_INFO, STD_INFO


# Reference fid positions for spoiling tests
FIDS = get_fid_catalog(detector='ACIS-S')


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
    # Basic catalog with no stars in field.  Standard 2-4-5 config.
    fids = get_fid_catalog(detector='ACIS-S')
    exp = ['<FidTable length=6>',
           '  id    yang     zang     row     col     mag   spoiler_score slot',
           'int64 float64  float64  float64 float64 float64     int64     str3',
           '----- -------- -------- ------- ------- ------- ------------- ----',
           '    1   922.59 -1737.89 -180.05 -344.10    7.00             0  ...',
           '    2  -773.20 -1742.03  160.79 -345.35    7.00             0    0',
           '    3    40.01 -1871.10   -2.67 -371.00    7.00             0  ...',
           '    4  2140.23   166.63 -424.51   39.13    7.00             0    1',
           '    5 -1826.28   160.17  372.97   36.47    7.00             0    2',
           '    6   388.59   803.75  -71.49  166.10    7.00             0  ...']
    assert repr(fids.meta['cand_fids']).splitlines() == exp
    assert np.all(fids['id'] == [2, 4, 5])

    # Make catalogs with some fake stars (at exactly fid positions) that spoil
    # the fids.
    stars = fids.meta['cand_fids'].copy()
    stars['mag_err'] = 0.1

    # Spoil fids 1, 2
    fids2 = get_fid_catalog(detector='ACIS-S', stars=stars[:2], dither=8)
    exp = ['<FidTable length=6>',
           '  id    yang     zang     row     col     mag   spoiler_score slot',
           'int64 float64  float64  float64 float64 float64     int64     str3',
           '----- -------- -------- ------- ------- ------- ------------- ----',
           '    1   922.59 -1737.89 -180.05 -344.10    7.00             4  ...',
           '    2  -773.20 -1742.03  160.79 -345.35    7.00             4  ...',
           '    3    40.01 -1871.10   -2.67 -371.00    7.00             0    0',
           '    4  2140.23   166.63 -424.51   39.13    7.00             0    1',
           '    5 -1826.28   160.17  372.97   36.47    7.00             0    2',
           '    6   388.59   803.75  -71.49  166.10    7.00             0  ...']
    assert repr(fids2.meta['cand_fids']).splitlines() == exp
    assert np.all(fids2['id'] == [3, 4, 5])

    # Spoil fids 1, 2, 3
    fids3 = get_fid_catalog(detector='ACIS-S', stars=stars[:3], dither=8)
    assert np.all(fids3['id'] == [4, 5, 6])

    # Spoil fids 1, 2, 3, 4 => no initial catalog gets found
    fids4 = get_fid_catalog(detector='ACIS-S', stars=stars[:4], dither=8)
    assert len(fids4) == 0


def test_fid_spoiling_acq():
    """
    Check fid spoiling acq:
    - 20" (4 pix) positional err on fid light
    - 4 pixel readout halfw for fid light
    - 2 pixel PSF of fid light that could creep into search box
    - Acq search box half-width
    - Dither amplitude (since OBC adjusts search box for dither)

    For this case (100" halfw and 8" dither) the threshold for spoiling
    is 20 + 20 + 10 + 100 + 8 = 158".  So this test puts stars at the
    positions of ACIS-S 2, 4, 5 but offset by 90, 157 and 159 arcsec.
    Only ACIS-S-5 is allowed, so we end up with the first fid set using
    1, 3, 5, 6, which is 1, 5, 6.
    """
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5)

    for fid, offset in zip(FIDS[:3], [90, 157, 159]):
        stars.add_fake_star(yang=fid['yang'] + offset,
                            zang=fid['zang'] + offset, mag=7.0)

    acqs = get_acq_catalog(stars=stars, **STD_INFO)
    acqs['halfw'] = 100
    fids5 = get_fid_catalog(acqs=acqs)
    exp = ['<FidTable length=6>',
           '  id    yang     zang     row     col     mag   spoiler_score slot',
           'int64 float64  float64  float64 float64 float64     int64     str3',
           '----- -------- -------- ------- ------- ------- ------------- ----',
           '    1   922.59 -1737.89 -180.05 -344.10    7.00             0    0',
           '    2  -773.20 -1742.03  160.79 -345.35    7.00             0  ...',
           '    3    40.01 -1871.10   -2.67 -371.00    7.00             0  ...',
           '    4  2140.23   166.63 -424.51   39.13    7.00             0  ...',
           '    5 -1826.28   160.17  372.97   36.47    7.00             0    1',
           '    6   388.59   803.75  -71.49  166.10    7.00             0    2']

    assert repr(fids5.meta['cand_fids']).splitlines() == exp


def test_fid_mult_spoilers():
    """
    Test of fix for bug #54.  19605 and 20144 were previous crashing.
    """
    acqs = get_acq_catalog(**OBS_INFO[19605])
    fids = get_fid_catalog(acqs=acqs)
    cand_fids = fids.meta['cand_fids']
    assert np.all(cand_fids['spoiler_score'] == [0, 0, 1, 4, 0, 0])
    assert len(cand_fids['spoilers'][2]) == 1
    assert cand_fids['spoilers'][2]['warn'][0] == 'yellow'


def test_dither_as_sequence():
    """
    Test that calling get_acq_catalog with a 2-element sequence (dither_y, dither_z)
    gives the expected response.  (Basically that it still returns a catalog).
    """
    fids = get_fid_catalog(detector='ACIS-S', dither=(8, 22))
    assert len(fids) == 3
    assert fids.meta['dither'] == 22  # Will be (8, 22) for release 4.0
