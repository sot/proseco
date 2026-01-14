import numpy as np
import pytest

from .. import characteristics_fid as FID
from ..acq import get_acq_catalog
from ..core import StarsTable
from ..fid import get_fid_catalog, get_fid_positions
from .test_common import DARK40, OBS_INFO, STD_INFO, mod_std_info


# Reference fid positions for tests.
@pytest.fixture(scope="module")
def FIDS():
    return get_fid_catalog(stars=StarsTable.empty(), **STD_INFO)


def test_get_fid_position():
    """
    Compare computed fid positions to flight values from starcheck reports.
    """
    # Obsid 20975
    yang, zang = get_fid_positions("ACIS-I", focus_offset=0.0, sim_offset=-583)
    fidset = [0, 4, 5]
    assert np.allclose(yang[fidset], [919, -1828, 385], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-874, 1023, 1667], rtol=0, atol=1.1)

    # Obsid 20201
    yang, zang = get_fid_positions("ACIS-I", focus_offset=0.0, sim_offset=0.0)
    fidset = [0, 4, 5]
    assert np.allclose(yang[fidset], [919, -1828, 385], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-844, 1053, 1697], rtol=0, atol=1.1)

    # Obsid 20334
    yang, zang = get_fid_positions("HRC-I", focus_offset=0.0, sim_offset=0.0)
    fidset = [0, 1, 2]
    assert np.allclose(yang[fidset], [-776, 836, -1204], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-1306, -1308, 998], rtol=0, atol=1.1)

    # Obsid 20168
    # Note the focus_offset offset of -70 is not a strong test, since this also
    # passes for focus_offset=-700.  But since code is just copied from Matlab
    # take that as heritage.
    yang, zang = get_fid_positions("HRC-S", focus_offset=-70.0, sim_offset=0.0)
    fidset = [0, 1, 2]
    assert np.allclose(yang[fidset], [-1174, 1224, -1177], rtol=0, atol=1.1)
    assert np.allclose(zang[fidset], [-468, -460, 561], rtol=0, atol=1.1)


def test_get_fid_pos_with_offsets(monkeypatch):
    # Confirm that if env var is set to 'False' then no offset is applied.
    monkeypatch.setenv("PROSECO_ENABLE_FID_OFFSET", "False")
    yang1, zang1 = get_fid_positions("ACIS-S", focus_offset=0.0, sim_offset=0.0)

    # Confirm that if env var is set to 'True' and t_ccd and date are specified, then
    # offset is applied.
    monkeypatch.setenv("PROSECO_ENABLE_FID_OFFSET", "True")
    yang2, zang2 = get_fid_positions(
        "ACIS-S", focus_offset=0.0, sim_offset=0.0, date="2023:235", t_ccd=-13.65
    )
    assert np.allclose(yang1 - yang2, -32.47, rtol=0, atol=0.1)
    assert np.allclose(zang1 - zang2, -9.63, rtol=0, atol=0.1)

    # Confirm that if env var is not set and t_ccd and date are specified, then
    # offset is applied.
    monkeypatch.delenv("PROSECO_ENABLE_FID_OFFSET", raising=False)
    yang3, zang3 = get_fid_positions(
        "ACIS-S", focus_offset=0.0, sim_offset=0.0, date="2023:235", t_ccd=-13.65
    )
    assert np.all(yang2 == yang3)
    assert np.all(zang2 == zang3)

    # Confirm that if env var is not set and t_ccd and date not specified, then
    # no offset is applied.
    monkeypatch.delenv("PROSECO_ENABLE_FID_OFFSET", raising=False)
    yang4, zang4 = get_fid_positions(
        "ACIS-S",
        focus_offset=0.0,
        sim_offset=0.0,
    )
    assert np.all(yang1 == yang4)
    assert np.all(zang1 == zang4)


def test_get_fid_pos_errs(monkeypatch):
    # Confirm if env var is set to 'True' and t_ccd and date not specified, then
    # there's an error.
    monkeypatch.setenv("PROSECO_ENABLE_FID_OFFSET", "True")
    with pytest.raises(ValueError, match="t_ccd_acq and date must be provided"):
        get_fid_positions("ACIS-S", focus_offset=0.0, sim_offset=0.0)

    # Confirm an error if env var set to not-allowed value
    monkeypatch.setenv("PROSECO_ENABLE_FID_OFFSET", "foo")
    with pytest.raises(
        ValueError, match='env var must be either "True", "False", or not set,'
    ):
        get_fid_positions(
            "ACIS-S", focus_offset=0.0, sim_offset=0.0, t_ccd=-13.65, date="2023:235"
        )


def test_fid_catalog_t_ccd():
    """
    Test that t_ccd vs t_ccd_acq/guide is applied to get_fid_catalog
    """
    aca_args1 = STD_INFO.copy()
    for key in ["t_ccd_acq", "t_ccd_guide", "t_ccd"]:
        if key in aca_args1:
            del aca_args1[key]
    aca_args1["t_ccd"] = -14
    fids1 = get_fid_catalog(**aca_args1)

    aca_args2 = STD_INFO.copy()
    for key in ["t_ccd_acq", "t_ccd_guide", "t_ccd"]:
        if key in aca_args2:
            del aca_args2[key]
    aca_args2["t_ccd_acq"] = -14
    aca_args2["t_ccd_guide"] = 5
    fids2 = get_fid_catalog(**aca_args2)

    assert fids1.t_ccd_acq == fids2.t_ccd_acq
    assert fids1.t_ccd_guide != fids2.t_ccd_guide
    assert fids1.t_ccd != fids2.t_ccd
    assert np.all(fids1["yang"] == fids2["yang"])
    assert np.all(fids1["zang"] == fids2["zang"])


def test_get_initial_catalog(FIDS):
    """Test basic catalog with no stars in field using standard 2-4-5 config."""
    exp = [
        "  id    yang     zang     row     col     mag   spoiler_score fid_trap_spoiler  idx   slot",
        "int64 float64  float64  float64 float64 float64     int64           bool       int64 int64",
        "----- -------- -------- ------- ------- ------- ------------- ---------------- ----- -----",
        "    1   918.09 -1741.51 -179.15 -344.83    7.00             0            False     0   -99",
        "    2  -777.70 -1745.65  161.69 -346.09    7.00             0            False     1     0",
        "    3    35.52 -1874.72   -1.77 -371.73    7.00             0            False     2   -99",
        "    4  2135.73   163.01 -423.60   38.40    7.00             0            False     3     1",
        "    5 -1830.77   156.55  373.88   35.74    7.00             0            False     4     2",
        "    6   384.10   800.13  -70.59  165.37    7.00             0            False     5   -99",
    ]
    cols = [
        "id",
        "yang",
        "zang",
        "row",
        "col",
        "mag",
        "spoiler_score",
        "fid_trap_spoiler",
        "idx",
        "slot",
    ]
    assert (
        FIDS.cand_fids[cols].pformat(show_dtype=True, max_width=-1, max_lines=-1) == exp
    )
    assert np.all(FIDS["id"] == [2, 4, 5])

    # Make catalogs with some fake stars (at exactly fid positions) that spoil
    # the fids.
    stars = StarsTable.empty()
    for fid in FIDS.cand_fids:
        stars.add_fake_star(
            mag=fid["mag"], mag_err=0.1, yang=fid["yang"], zang=fid["zang"]
        )

    # Spoil fids 1, 2
    fids2 = get_fid_catalog(stars=stars[:2], **STD_INFO)
    exp = [
        "  id    yang     zang     row     col     mag   spoiler_score fid_trap_spoiler  idx   slot",
        "int64 float64  float64  float64 float64 float64     int64           bool       int64 int64",
        "----- -------- -------- ------- ------- ------- ------------- ---------------- ----- -----",
        "    1   918.09 -1741.51 -179.15 -344.83    7.00             4            False     0   -99",
        "    2  -777.70 -1745.65  161.69 -346.09    7.00             4            False     1   -99",
        "    3    35.52 -1874.72   -1.77 -371.73    7.00             0            False     2     0",
        "    4  2135.73   163.01 -423.60   38.40    7.00             0            False     3     1",
        "    5 -1830.77   156.55  373.88   35.74    7.00             0            False     4     2",
        "    6   384.10   800.13  -70.59  165.37    7.00             0            False     5   -99",
    ]
    cols = [
        "id",
        "yang",
        "zang",
        "row",
        "col",
        "mag",
        "spoiler_score",
        "fid_trap_spoiler",
        "idx",
        "slot",
    ]
    assert (
        fids2.cand_fids[cols].pformat(show_dtype=True, max_width=-1, max_lines=-1)
        == exp
    )
    assert np.all(fids2["id"] == [3, 4, 5])

    # Spoil fids 1, 2, 3
    fids3 = get_fid_catalog(stars=stars[:3], **STD_INFO)
    assert np.all(fids3["id"] == [4, 5, 6])

    # Spoil fids 1, 2, 3, 4 => no initial catalog gets found
    fids4 = get_fid_catalog(stars=stars[:4], **STD_INFO)
    assert len(fids4) == 0
    assert all(name in fids4.colnames for name in ["id", "yang", "zang", "row", "col"])


def test_n_fid():
    """Test specifying number of fids."""
    # Get only 2 fids
    fids = get_fid_catalog(**mod_std_info(n_fid=2))
    assert len(fids) == 2


@pytest.mark.parametrize("dither_z", [8, 64])
def test_fid_spoiling_acq(dither_z, FIDS):
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
        stars.add_fake_star(
            yang=fid["yang"] + offset + dither_y,
            zang=fid["zang"] + offset + dither_z,
            mag=7.0,
        )

    std_info = STD_INFO.copy()
    std_info["dither"] = (dither_y, dither_z)
    acqs = get_acq_catalog(stars=stars, **std_info)
    acqs["halfw"] = 100
    fids5 = get_fid_catalog(acqs=acqs, **std_info)
    exp = [
        "  id    yang     zang     row     col     mag   spoiler_score fid_trap_spoiler  idx   slot",
        "int64 float64  float64  float64 float64 float64     int64           bool       int64 int64",
        "----- -------- -------- ------- ------- ------- ------------- ---------------- ----- -----",
        "    1   918.09 -1741.51 -179.15 -344.83    7.00             0            False     0     0",
        "    2  -777.70 -1745.65  161.69 -346.09    7.00             0            False     1   -99",
        "    3    35.52 -1874.72   -1.77 -371.73    7.00             0            False     2   -99",
        "    4  2135.73   163.01 -423.60   38.40    7.00             0            False     3   -99",
        "    5 -1830.77   156.55  373.88   35.74    7.00             0            False     4     1",
        "    6   384.10   800.13  -70.59  165.37    7.00             0            False     5     2",
    ]
    cols = [
        "id",
        "yang",
        "zang",
        "row",
        "col",
        "mag",
        "spoiler_score",
        "fid_trap_spoiler",
        "idx",
        "slot",
    ]

    assert (
        fids5.cand_fids[cols].pformat(show_dtype=True, max_width=-1, max_lines=-1)
        == exp
    )


def test_fid_mult_spoilers(disable_fid_offsets, proseco_agasc_1p7):
    """
    Test of fix for bug #54.  19605 and 20144 were previous crashing.
    """
    acqs = get_acq_catalog(**OBS_INFO[19605])
    fids = get_fid_catalog(acqs=acqs, **OBS_INFO[19605])
    cand_fids = fids.cand_fids
    assert np.all(cand_fids["spoiler_score"] == [0, 0, 1, 4, 0, 0])
    assert len(cand_fids["spoilers"][2]) == 1
    assert cand_fids["spoilers"][2]["warn"][0] == "yellow"


def test_dither_as_sequence():
    """
    Test that calling get_acq_catalog with a 2-element sequence (dither_y, dither_z)
    gives the expected response.  (Basically that it still returns a catalog).
    """
    std_info = STD_INFO.copy()
    std_info["dither"] = (8, 22)
    fids = get_fid_catalog(**std_info)
    assert len(fids) == 3
    assert fids.dither_acq == (8, 22)
    assert fids.dither_guide == (8, 22)


def test_fid_spoiler_score(FIDS):
    """Test computing the fid spoiler score."""
    dither_y = 8
    dither_z = 64
    stars = StarsTable.empty()
    for fid, offset in zip(FIDS[:2], [-1, 1]):
        stars.add_fake_star(
            yang=fid["yang"] + FID.spoiler_margin + dither_y + offset,
            zang=fid["zang"] + FID.spoiler_margin + dither_z + offset,
            mag=7.0,
        )

    dither = (dither_y, dither_z)

    std_info = STD_INFO.copy()
    std_info["dither"] = dither
    fids = get_fid_catalog(stars=stars, **std_info)
    assert np.all(fids.cand_fids["spoiler_score"] == [0, 4, 0, 0, 0, 0])


def test_big_sim_offset():
    """Test of an observation with a big SIM offset"""
    fids = get_fid_catalog(**mod_std_info(stars=StarsTable.empty(), sim_offset=300000))
    names = ["id", "yang", "zang", "row", "col", "mag", "spoiler_score", "idx"]
    assert all(name in fids.colnames for name in names)


def test_fid_hot_pixel_reject(FIDS):
    """Test hot pixel rejecting a fid"""
    lim = FID.hot_pixel_spoiler_limit
    dark = DARK40.copy()
    for fid_id, off, dc in [
        (1, 8.0, lim * 1.05),  # spoiler,
        (2, 12.0, lim * 1.05),  # not spoiler (spatially)
        (3, 8.0, lim * 0.95),  # not spoiler (dark current too low)
        (4, -8.0, lim * 1.05),  # spoiler
        (5, 0.0, lim * 1.05),
    ]:  # spoiler
        fid = FIDS.cand_fids.get_id(fid_id)
        r = int(round(fid["row"] + off))
        c = int(round(fid["col"] + off))
        dark[r + 512, c + 512] = dc

    fids = get_fid_catalog(stars=StarsTable.empty(), dark=dark, **STD_INFO)
    assert fids["id"].tolist() == [2, 3, 6]


def test_fids_include_exclude():
    """
    Test include and exclude fids.
    """
    fids = get_fid_catalog(stars=StarsTable.empty(), dark=DARK40, **STD_INFO)
    assert np.all(fids["id"] == [2, 4, 5])

    # Define includes and excludes.
    include_ids = [1, 4]
    exclude_ids = [5]

    fids = get_fid_catalog(
        stars=StarsTable.empty(),
        dark=DARK40,
        **STD_INFO,
        include_ids=include_ids,
        exclude_ids=exclude_ids,
    )

    assert fids.include_ids == include_ids
    assert fids.exclude_ids == exclude_ids

    for cand_set in fids.cand_fid_sets:
        assert all(id_ in cand_set for id_ in include_ids)
        assert all(id_ not in cand_set for id_ in exclude_ids)

    assert all(id_ in fids["id"] for id_ in include_ids)
    assert all(id_ not in fids["id"] for id_ in exclude_ids)

    assert np.all(fids["id"] == [1, 2, 4])


def test_fids_include_bad():
    """
    Test include bad id fid and fid not on CCD
    """
    fids = get_fid_catalog(stars=StarsTable.empty(), dark=DARK40, **STD_INFO)
    assert np.all(fids["id"] == [2, 4, 5])

    # Define includes and excludes.
    include_ids = [10]
    exclude_ids = []

    # If you force-include a non-existent fid, you raise ValueError
    with pytest.raises(ValueError):
        get_fid_catalog(
            stars=StarsTable.empty(),
            dark=DARK40,
            **STD_INFO,
            include_ids=include_ids,
            exclude_ids=exclude_ids,
        )

    # Set up a scenario with large offset so only two are on the CCD
    include_ids = [4]
    exclude_ids = []
    fids = get_fid_catalog(**mod_std_info(stars=StarsTable.empty(), sim_offset=80000))
    assert np.all(fids.cand_fids["id"] == [1, 2])

    # Force include one that isn't on the CCD
    fids = get_fid_catalog(
        **mod_std_info(stars=StarsTable.empty(), sim_offset=80000),
        include_ids=include_ids,
        exclude_ids=exclude_ids,
    )

    assert np.all(fids["id"] == [1, 2, 4])

    # If you force-include a fid that is off the CCD, it is still off the CCD
    assert fids.off_ccd(fids.get_id(4))
