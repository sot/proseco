import numpy as np
from astropy.table import Table
import pytest

from Quaternion import Quat
from proseco import get_aca_catalog, jupiter
from proseco.core import StarsTable
from proseco.tests.test_common import DARK40, mod_std_info
from chandra_aca import planets, transform
from cheta import fetch


HAS_CHETA_EPHEM = False
try:
    dat = fetch.Msid("orbitephem0_x", start="2025:001", stop="2025:002")
    assert len(dat.vals) > 0
    HAS_CHETA_EPHEM = True
except Exception:
    HAS_CHETA_EPHEM = False


@pytest.mark.skipif(not HAS_CHETA_EPHEM, reason="Requires cheta ephemeris access")
def test_jupiter_position():
    # For an historical date and attitude, compare jupiter.get_jupiter_position
    # to chandra_aca.planets.get_planet_chandra.  The proseco code is using the
    # stk ephemeris instead of the cheta predictive ephemeris.
    # This is from obsid 23375
    att = Quat(q=[-0.51186291,  0.27607314,  -0.17243277,  0.79501379])
    date = "2021:290:11:33:16.000"
    duration = 36000
    jupiter_proseco_data = jupiter.get_jupiter_position(date, duration, att)
    eci = planets.get_planet_chandra("jupiter", jupiter_proseco_data["time"])
    ra, dec = transform.eci_to_radec(eci)
    yag, zag = transform.radec_to_yagzag(ra, dec, att)
    row, col = transform.yagzag_to_pixels(yag, zag, allow_bad=True)
    lim0 = -512
    lim1 = 511
    # Only plot planet within the image limits
    ok = (row >= lim0) & (row <= lim1) & (col >= lim0) & (col <= lim1)
    jupiter_aca_data = Table({"time": jupiter_proseco_data["time"][ok], "row": row[ok], "col": col[ok]})
    # Compare the two tables
    assert len(jupiter_proseco_data) == len(jupiter_aca_data)
    assert np.allclose(jupiter_proseco_data["row"], jupiter_aca_data["row"], atol=0.1, rtol=0)
    assert np.allclose(jupiter_proseco_data["col"], jupiter_aca_data["col"], atol=0.1, rtol=0)


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
    dark = DARK40.copy()
    stars = StarsTable.empty()
    att = Quat(q=[-0.49963289, 0.25613709, -0.16664083, 0.81055018])
    stars.att = att
    stars.add_fake_constellation(mag=8.0, n_stars=5)

    # Add a fake star right on jupiter
    stars.add_fake_star(id=200, mag=6.5, row=-92, col=198)

    # Add a fake star near jupiter but outside the spoiler margin
    stars.add_fake_star(id=201, mag=7.5, row=-92, col=240)

    aca = get_aca_catalog(
        **mod_std_info(
            stars=stars,
            dark=dark,
            date="2021:249:12:00:00.000",
            att=att,
            duration=20000,
            target_name="Jupiter 1",
            detector="HRC-I",
            n_guide=4,
            raise_exc=True,
        )
    )
    # Confirm the fake star right on jupiter is not in the catalog
    assert 200 not in aca["id"]

    # Confirm the other fake star is a BOT
    assert 201 in aca.guides["id"]
    assert 201 in aca.acqs["id"]
    # But that as an acq star it has a reduced box size
    ok = aca["id"] == 201
    assert aca["halfw"][ok][0] == 80

    # Confirm that one of the HRC fid lights is spoiled by jupiter but still selected
    assert aca.fids["spoiler_score"][aca.fids["id"] == 3][0] == 5

    # Confirm counts look good
    assert len(aca.acqs) == 6
    assert len(aca.guides) == 4
    assert len(aca.fids) == 3

    # Confirm Jupiter is all on the left side of the image
    assert np.all(aca.jupiter["row"] < 0)

    # Confirm two guide stars on the opposite side of the image
    assert np.sum(aca.guides["row"] > 0) >= 2

    # Confirm no guide stars within 15 columns of Jupiter
    for jcol in aca.jupiter["col"]:
        dcol = np.abs(aca.guides["col"] - jcol)
        assert np.all(dcol > 15)

    raise ValueError


def test_jupiter_offset_right():
    dark = DARK40.copy()
    stars = StarsTable.empty()
    att = Quat(q=[-0.50066275,  0.25809145, -0.16348241,  0.80993772])
    stars.att = att
    stars.add_fake_constellation(mag=8.0, n_stars=8)
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
            n_guide=3,
        )
    )

def test_jupiter_midline():
    dark = DARK40.copy()
    stars = StarsTable.empty()
    att = Quat(q=[-0.43419701, -0.51408310, 0.33920339, 0.65736792])
    stars.att = att
    # stars.add_fake_constellation(mag=8.0, n_stars=8)

    # t_ccd = np.trunc(ACA.aca_t_ccd_penalty_limit - 1.0)

    aca = get_aca_catalog(
        **mod_std_info(
            stars=stars,
            dark=dark,
            # date="2025:093:13:26:04.000",
            date="2025:093:12:26:04.000",
            att=att,
            duration=25000,
            target_name="Jupiter 1",
            raise_exc=True,
        )
    )
    # assert len(aca.acqs) == 8
    # assert len(aca.guides) == 5
    # assert len(aca.fids) == 3
    # names = ["id", "yang", "zang", "row", "col", "mag", "spoiler_score", "idx"]
    # assert all(name in aca.fids.colnames for name in names)
    # assert aca.duration == 20000
    # assert aca.target_name == "Jupiter 1"


def test_jupiter_2():
    aca = get_aca_catalog(
        **mod_std_info(
            # date="2025:093:13:26:04.000",
            date="2025:093:12:26:04.000",
            att=Quat(q=[-0.43419701, -0.51408310, 0.33920339, 0.65736792]),
            duration=25000,
            target_name="Jupiter 2",
            raise_exc=True,
        )
    )
    assert len(aca.acqs) == 8
    assert len(aca.guides) == 5
    assert len(aca.fids) == 3
    names = ["id", "yang", "zang", "row", "col", "mag", "spoiler_score", "idx"]
    assert all(name in aca.fids.colnames for name in names)
    # assert aca.duration == 20000
    assert aca.target_name == "Jupiter 2"


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
    from cxotime import CxoTime

    start_sec = CxoTime(date).secs
    assert np.all((out["time"] >= start_sec) & (out["time"] <= start_sec + duration))


def test_jupiter_distribution_check_1():
    # Simulate jupiter_data crosses 0
    jupiter_data = Table({"row": [-10, 20]})
    cand_guide_set = Table({"row": [-400, -300, 400, 300]})
    # Should pass: at least two on each side
    assert jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)
    # Should fail: only with all on one side
    cand_guide_set = Table({"row": [-400, -300, -200]})
    assert not jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)


def test_jupiter_distribution_check_2():
    # Simulate jupiter_data all positive
    jupiter_data = Table({"row": [10, 20]})
    cand_guide_set = Table({"row": [-400, -300, 400, 300]})
    # Should pass: at least two on each side
    assert jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)
    # Should fail: only with all on one side
    cand_guide_set = Table({"row": [400, 300, 200]})
    assert not jupiter.jupiter_distribution_check(cand_guide_set, jupiter_data)


def test_check_spoiled_by_jupiter():
    # Simulate candidate stars and jupiter data
    cand_stars = Table({"id": [1, 2, 3], "col": [10, 50, 100], "row": [0, 0, 0]})
    jupiter_data = Table({"col": [40, 60], "row": [0, 0]})
    mask, rej = jupiter.check_spoiled_by_jupiter(cand_stars, jupiter_data)
    # Only star with col=50 should be spoiled (within 15 pixels of Jupiter)
    assert np.array_equal(mask, [False, True, False])
    assert len(rej) == 1
    assert rej[0]["id"] == 2


def test_add_jupiter_as_spoiler():
    # Simulate stars and jupiter data
    from cxotime import CxoTime

    from proseco.core import StarsTable

    date = "2025:220:12:00:00"
    att = (0, 0, 0)
    stars = StarsTable.empty()
    jupiter_data = Table({"time": [CxoTime(date).secs], "row": [100], "col": [100]})
    out = jupiter.add_jupiter_as_spoiler(date, stars, jupiter_data)
    # Should add a new star with id=20 at Jupiter's position
    assert len(out) == 1
    assert 20 in out["id"]


def test_add_jupiter_as_spoiler_no_jupiter():
    # Should return stars unchanged if jupiter is None
    stars = Table({"row": [0], "col": [0], "mag": [8.0], "id": [1], "CLASS": [0]})
    out = jupiter.add_jupiter_as_spoiler("2025:220:12:00:00", stars, None)
    assert np.array_equal(stars["id"], out["id"])
