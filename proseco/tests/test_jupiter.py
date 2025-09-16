import numpy as np
import pytest
from astropy.table import Table
from chandra_aca import planets, transform
from cheta import fetch
from cxotime import CxoTime
from Quaternion import Quat

from proseco import get_aca_catalog, jupiter
from proseco.characteristics_jupiter import JupiterPositionTable
from proseco.core import StarsTable
from proseco.tests.test_common import DARK40, mod_std_info

HAS_CHETA_EPHEM = False
try:
    dat = fetch.Msid("orbitephem0_x", start="2025:001", stop="2025:002")
    assert len(dat.vals) > 0
    HAS_CHETA_EPHEM = True
except Exception:
    HAS_CHETA_EPHEM = False


@pytest.mark.skipif(not HAS_CHETA_EPHEM, reason="Requires cheta ephemeris access")
def test_jupiter_position():
    """
    Test jupiter.get_jupiter_position

    Test jupiter.get_jupiter_position against chandra_aca.planets.get_planet_chandra for
    a known date and attitude. This is from obsid 23375.

    The proseco code is using the stk ephemeris instead of the cheta predictive
    ephemeris used by chandra_aca.planets.get_planet_chandra.
    """
    att = Quat(q=[-0.51186291, 0.27607314, -0.17243277, 0.79501379])
    date = "2021:290:11:33:16.000"
    duration = 36000
    jupiter_proseco_data = jupiter.get_jupiter_position(date, duration, att)
    eci = planets.get_planet_chandra("jupiter", jupiter_proseco_data["time"])
    ra, dec = transform.eci_to_radec(eci)
    yag, zag = transform.radec_to_yagzag(ra, dec, att)
    row, col = transform.yagzag_to_pixels(yag, zag, allow_bad=True)

    jupiter_aca_data = JupiterPositionTable(
        {"time": jupiter_proseco_data["time"], "row": row, "col": col}
    )
    # Compare the two tables
    assert len(jupiter_proseco_data) == len(jupiter_aca_data)
    assert np.allclose(
        jupiter_proseco_data["row"], jupiter_aca_data["row"], atol=0.1, rtol=0
    )
    assert np.allclose(
        jupiter_proseco_data["col"], jupiter_aca_data["col"], atol=0.1, rtol=0
    )


def test_jupiter_exclude_dates():
    # Dates within the exclude range should return True
    assert jupiter.date_is_excluded("2026:150")
    assert jupiter.date_is_excluded("2026-05-10")
    assert jupiter.date_is_excluded("2026:300")
    # Dates outside the exclude range should return False
    assert not jupiter.date_is_excluded("2026:100")
    assert not jupiter.date_is_excluded("2026:310")
    assert not jupiter.date_is_excluded("2025-09-01")
    assert not jupiter.date_is_excluded("2027:150")


def test_jupiter_offset_left():
    """
    Test a case where Jupiter is on the left side of the CCD.
    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    att = Quat(q=[-0.49963289, 0.25613709, -0.16664083, 0.81055018])
    stars.att = att

    # Add a fake star right on jupiter
    stars.add_fake_star(id=200, mag=6.5, row=-92, col=198)

    # Add a fake star same row as Jupiter but away in column
    stars.add_fake_star(id=201, mag=7.5, row=-92, col=300)

    # And another star on the left side that is bright
    stars.add_fake_star(id=202, mag=6, row=-300, col=300)

    # and two more stars on the right, where the second one is close enough in
    # column to the first that it should have a reduced acq box.
    stars.add_fake_star(id=203, mag=8, row=100, col=100)
    stars.add_fake_star(id=204, mag=8, row=100, col=250)

    aca = get_aca_catalog(
        **mod_std_info(
            stars=stars,
            dark=dark,
            date="2021:249:12:00:00.000",
            att=att,
            duration=20000,
            target_name="Jupiter 1",
            detector="HRC-I",
            n_guide=2,
            raise_exc=True,
        )
    )
    # Confirm the fake star right on jupiter is not in the catalog
    assert 200 not in aca["id"]

    # Confirm this one is an acq
    assert 201 in aca.acqs["id"]

    # Confirm 204 is an acq with reduced box
    # But that as an acq star it has a reduced box size
    ok = aca["id"] == 204
    assert aca["halfw"][ok][0] == 80

    # Confirm that there are only two fid lights selected. Jupiter column-spoils the
    # other two HRC-I fids.
    assert len(aca.fids) == 2

    # Confirm other counts look good
    assert len(aca.acqs) == 4
    assert len(aca.guides) == 2

    # Confirm Jupiter is all on the left side of the CCD
    assert np.all(aca.jupiter["row"] < 0)

    # Confirm two guide stars on the opposite side of the CCD
    # This checks optimization because there is a brighter star
    # on the left side.
    assert np.sum(aca.guides["row"] > 0) >= 2

    # Confirm no guide stars within 15 columns of Jupiter
    for jcol in aca.jupiter["col"]:
        dcol = np.abs(aca.guides["col"] - jcol)
        assert np.all(dcol > 15)


def test_jupiter_offset_right():
    """
    Test a case where Jupiter is on the right side of the CCD.
    """
    dark = DARK40.copy()
    att = Quat(q=[-0.50066275, 0.25809145, -0.16348241, 0.80993772])
    stars = StarsTable.empty()
    stars.att = att

    # Add stars on the same side as Jupiter
    stars.add_fake_star(id=201, mag=8, row=-100, col=300)
    stars.add_fake_star(id=202, mag=6, row=100, col=100)
    stars.add_fake_star(id=203, mag=8, row=100, col=250)

    # Add one right on Jupiter
    stars.add_fake_star(id=204, mag=8, row=177, col=5)

    # Add one below Jupiter
    stars.add_fake_star(id=205, mag=8, row=177, col=-40)

    # And star on the left side
    stars.add_fake_star(id=206, mag=8, row=-300, col=-40)

    aca = get_aca_catalog(
        **mod_std_info(
            stars=stars,
            dark=dark,
            date="2021:251:22:00:00.000",
            att=att,
            duration=20000,
            target_name="Jupiter 1",
            detector="HRC-I",
            raise_exc=True,
            n_guide=2,
        )
    )
    # >>> aca.jupiter
    # <JupiterPositionTable length=22>
    #        time              row                 col
    #      float64           float64             float64
    # ----------------- ------------------ -------------------
    #     747525669.184 177.24311544187495    5.48947125483997
    # 747526621.5649524  177.3229077138035   4.546971313655077
    #               ...                ...                 ...
    # 747543764.4220952   178.986298841097 -12.390976350022335
    # 747544716.8030477 179.09250845640196 -13.329891937053558
    #     747545669.184 179.18608156041896 -14.258603110515383

    # <ACATable length=8>
    #  slot  idx    id  type  sz   p_acq    mag    maxmag   yang     zang   halfw
    # int64 int64 int64 str3 str3 float64 float64 float64 float64  float64  int64
    # ----- ----- ----- ---- ---- ------- ------- ------- -------- -------- -----
    #     0     1     1  FID  8x8   0.000    7.00    8.00  -764.49 -1303.66    25
    #     1     2     2  FID  8x8   0.000    7.00    8.00   848.16 -1305.26    25
    #     2     3     3  FID  8x8   0.000    7.00    8.00 -1192.95  1000.94    25
    #     3     4   201  BOT  8x8   0.978    8.00    9.50   531.22  1469.38   160
    #     4     5   206  BOT  8x8   0.915    8.00    9.50  1524.19  -226.31    80
    #     0     6   202  ACQ  8x8   0.978    6.00    7.50  -467.90   475.47   160
    #     1     7   203  ACQ  8x8   0.978    8.00    9.50  -465.94  1222.88   160
    #     2     8   205  ACQ  8x8   0.915    8.00    9.50  -853.34  -222.81    80

    # Confirm the fake star right on jupiter is not in the catalog
    assert 204 not in aca["id"]

    # Confirm the fake star below jupiter is an ACQ with small box
    assert 205 in aca.acqs["id"]
    ok = aca["id"] == 205
    assert aca["halfw"][ok][0] == 80

    # Confirm that there are 3 fid lights selected
    assert len(aca.fids) == 3

    # Confirm other counts look good
    assert len(aca.acqs) == 5
    assert len(aca.guides) == 2

    # Confirm Jupiter is all on the right side of the CCD
    assert np.all(aca.jupiter["row"] > 0)

    # Confirm two guide stars on the opposite side of the CCD
    # This checks optimization because there is a brighter star
    # on the left side.
    assert np.sum(aca.guides["row"] < 0) >= 2

    # Confirm no guide stars within 15 columns of Jupiter
    for jcol in aca.jupiter["col"]:
        dcol = np.abs(aca.guides["col"] - jcol)
        assert np.all(dcol > 15)


def test_jupiter_midline():
    """
    Test a case where Jupiter crosses the midline of the CCD.
    """
    dark = DARK40.copy()
    stars = StarsTable.empty()
    att = Quat(q=[-0.313343, -0.4476205, 0.42355607, 0.72253187])
    stars.att = att

    stars.add_fake_star(id=200, mag=6.5, row=-150, col=150)
    stars.add_fake_star(id=201, mag=6.5, row=150, col=150)
    stars.add_fake_star(id=202, mag=6.5, row=150, col=-150)
    stars.add_fake_star(id=203, mag=6.5, row=-150, col=-150)
    stars.add_fake_star(id=204, mag=9, row=-300, col=300)
    stars.add_fake_star(id=205, mag=9, row=300, col=300)

    aca = get_aca_catalog(
        **mod_std_info(
            stars=stars,
            dark=dark,
            date="2025:093:12:26:04.000",
            att=att,
            duration=30000,
            target_name="1Jupiter1",
            detector="HRC-I",
            raise_exc=True,
            n_guide=4,
        )
    )
    # Confirm Jupiter is on both sides of the CCD
    assert np.max(aca.jupiter["row"]) > 0
    assert np.min(aca.jupiter["row"]) < 0

    # Confirm two guide stars on each side of the CCD
    assert np.sum(aca.guides["row"] > 0) >= 2
    assert np.sum(aca.guides["row"] < 0) >= 2

    # Confirm that a fainter star is selected which is a weak confirmation
    # that the cluster checks are still working
    assert 204 in aca.guides["id"]

    # Confirm no guide stars within 15 columns of Jupiter
    for jcol in aca.jupiter["col"]:
        dcol = np.abs(aca.guides["col"] - jcol)
        assert np.all(dcol > 15)

    return aca


def test_jupiter_acquisition():
    """
    Test how jupiter is handled during acquisition.

    This test loops over a range of column offsets for a star
    near Jupiter to confirm that the star is excluded from acquisition
    when it could be within a search box + maneuver error (though maneuver
    error is not explicitly used in this test).
    """
    for col_dist_arcsec in np.arange(-300, 305, 20):
        dark = DARK40.copy()
        stars = StarsTable.empty()
        att = Quat(q=[-0.49963289, 0.25613709, -0.16664083, 0.81055018])
        stars.att = att
        date = "2021:249:12:00:00.000"
        jupiter_pos = jupiter.get_jupiter_position(date, 30000, att)
        jupiter_acq_pos = jupiter.get_jupiter_acq_pos(date, jupiter=jupiter_pos)

        col_dist = int(col_dist_arcsec / 5)
        stars.add_fake_star(id=200, mag=6.5, row=-300, col=400)
        stars.add_fake_star(id=201, mag=6.5, row=150, col=150)
        stars.add_fake_star(id=202, mag=6.5, row=150, col=-150)

        # Move one star through the acquisition exclusion region in column for jupiter
        stars.add_fake_star(
            id=203, mag=6.5, row=-150, col=jupiter_acq_pos.col + col_dist
        )

        aca = get_aca_catalog(
            **mod_std_info(
                stars=stars,
                dark=dark,
                date=date,
                att=att,
                man_angle=90,  # This isn't a change from mod_std_info but just to be explicit
                duration=30000,
                target_name="Jupiter 1",
                detector="HRC-I",
                raise_exc=True,
                n_guide=4,
            )
        )

        # With maneuver error, we just don't get id=203 if within 190 arcsec
        if np.abs(col_dist_arcsec) <= 190:
            # Confirm that there are 3 acqs
            assert len(aca.acqs) == 3
            assert 203 not in aca.acqs["id"]
        else:
            # Confirm that there are 4 acqs
            assert len(aca.acqs) == 4
            assert 203 in aca.acqs["id"]


def test_jupiter_2():
    """
    Test the jupiter code with a real star field and Jupiter present.
    """
    aca = get_aca_catalog(
        **mod_std_info(
            date="2025:093:12:26:04.000",
            att=Quat(q=[-0.43419701, -0.51408310, 0.33920339, 0.65736792]),
            duration=25000,
            detector="ACIS-S",
            target_name="Jupiter 2",
            raise_exc=True,
        )
    )
    assert len(aca.acqs) == 8
    assert len(aca.guides) == 5
    assert len(aca.fids) == 3
    names = ["id", "yang", "zang", "row", "col", "mag", "spoiler_score", "idx"]
    assert all(name in aca.fids.colnames for name in names)
    assert aca.target_name == "Jupiter 2"


def test_jupiter_3():
    """
    Test the jupiter code with a real star field and Jupiter present.
    This time use HRC-I
    """
    aca = get_aca_catalog(
        **mod_std_info(
            date="2025:093:12:26:04.000",
            att=Quat(q=[-0.43419701, -0.51408310, 0.33920339, 0.65736792]),
            duration=25000,
            detector="HRC-I",
            target_name="Jupiter 3",
            raise_exc=True,
        )
    )
    assert len(aca.acqs) == 8
    assert len(aca.guides) == 5
    assert len(aca.fids) == 3
    names = ["id", "yang", "zang", "row", "col", "mag", "spoiler_score", "idx"]
    assert all(name in aca.fids.colnames for name in names)
    assert aca.target_name == "Jupiter 3"


def test_get_jupiter_position_returns_table():
    # Use a known date, duration, and a simple attitude (RA, Dec, Roll)
    date = "2025:093:12:26:04.000"
    duration = 1000  # seconds
    att = Quat(q=[-0.43419701, -0.51408310, 0.33920339, 0.65736792])
    out = jupiter.get_jupiter_position(date, duration, att)

    # Should return a Table or None
    assert isinstance(out, Table)
    assert all(col in out.colnames for col in ["time", "row", "col"])
    assert len(out) > 0

    # Confirm the times in the table are reasonable for the date and duration
    start_sec = CxoTime(date).secs
    assert np.all((out["time"] >= start_sec) & (out["time"] <= start_sec + duration))


def test_jupiter_distribution_check_1():
    # Simulate jupiter_data crosses 0
    jupiter_data = JupiterPositionTable({"row": [-10, 20]})
    cand_guide_set = Table({"row": [-400, -300, 400, 300]})
    # Should pass: at least two on each side
    assert jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)
    # Should fail: only with all on one side
    cand_guide_set = Table({"row": [-400, -300, -200]})
    assert not jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)


def test_jupiter_distribution_check_2():
    # Simulate jupiter_data all positive
    jupiter_data = JupiterPositionTable({"row": [10, 20]})
    cand_guide_set = Table({"row": [-400, -300, 400, 300]})
    # Should pass: at least two on each side
    assert jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)
    # Should fail: only with all on one side
    cand_guide_set = Table({"row": [400, 300, 200]})
    assert not jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)


def test_check_spoiled_by_jupiter():
    # Simulate candidate stars and jupiter data
    cand_stars = Table({"id": [1, 2, 3], "col": [10, 50, 100], "row": [0, 0, 0]})
    jupiter_data = JupiterPositionTable({"col": [40, 60], "row": [0, 0]})
    mask, rej = jupiter.check_spoiled_by_jupiter(cand_stars, jupiter_data)
    # Only star with col=50 should be spoiled (within 15 pixels of Jupiter)
    assert np.array_equal(mask, [False, True, False])
    assert len(rej) == 1
    assert rej[0]["id"] == 2


def test_add_jupiter_as_spoilers():
    date = "2025:220:12:00:00"
    stars = StarsTable.empty()
    jupiter_data = Table({"time": [CxoTime(date).secs], "row": [100], "col": [100]})
    out = jupiter.add_jupiter_as_lots_of_acq_spoilers(date, stars, jupiter_data)
    # Should add a new star with id=20 at Jupiter's position
    assert len(out) > 1000
    assert 1000 in out["id"]


def test_add_jupiter_as_spoilers_no_jupiter():
    # Should return stars unchanged if jupiter has zero length
    stars = Table({"row": [0], "col": [0], "mag": [8.0], "id": [1], "CLASS": [0]})
    out = jupiter.add_jupiter_as_lots_of_acq_spoilers(
        "2025:220:12:00:00", stars, JupiterPositionTable.empty()
    )
    assert np.array_equal(stars["id"], out["id"])
