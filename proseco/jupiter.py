import astropy.units as u
import numpy as np
from astropy.table import Table
from chandra_aca import planets
from chandra_aca.transform import eci_to_radec, radec_to_yagzag, yagzag_to_pixels
from cheta.comps import ephem_stk
from cxotime import CxoTime


def date_is_excluded(date):
    date = CxoTime(date)
    from proseco.characteristics_jupiter import exclude_dates

    for ex in exclude_dates:
        if CxoTime(ex["start"]) <= date <= CxoTime(ex["stop"]):
            return True
    return False


def get_jupiter_position(date, duration, att):
    date0 = CxoTime(date)
    n_times = int(duration / 1000) + 1
    dates = date0 + np.linspace(0, duration, n_times) * u.s
    times = np.atleast_1d(dates.secs)

    chandra_ephem = ephem_stk.get_ephemeris_stk(start=dates[0], stop=dates[-1])

    # Get Jupiter positions using chandra_aca.planets
    ephem = {
        key: np.interp(times, chandra_ephem["time"], chandra_ephem[key])
        for key in ["x", "y", "z"]
    }
    pos_earth = planets.get_planet_barycentric("earth", dates)

    chandra_eci = np.zeros_like(pos_earth)
    chandra_eci[..., 0] = ephem["x"].reshape(times.shape) / 1000
    chandra_eci[..., 1] = ephem["y"].reshape(times.shape) / 1000
    chandra_eci[..., 2] = ephem["z"].reshape(times.shape) / 1000
    eci = planets.get_planet_eci("jupiter", dates, pos_observer=pos_earth + chandra_eci)

    # Convert to RA, Dec
    ra, dec = eci_to_radec(eci)
    yag, zag = radec_to_yagzag(ra, dec, att)
    row, col = yagzag_to_pixels(yag, zag, allow_bad=True)

    lim0 = -512
    lim1 = 511
    # Only care about planet within the CCD
    ok = (row >= lim0) & (row <= lim1) & (col >= lim0) & (col <= lim1)
    if np.any(ok):
        out = Table(
            {
                "times": times[ok],
                "rows": row[ok],
                "cols": col[ok],
            }
        )
    else:
        out = None
    return out


def jupiter_distribution_check(cand_guide_set, jupiter_data):
    """
    Check that there are at least two candidate guide stars on the side of the CCD
    that does not have Jupiter.

    :param cand_guide_set: Table of candidate guide stars
    :returns: bool (True if check passes)
    """
    # It looks like jupiter ang diam goes from 30 to 45 arcsec
    # so use 25 arcsec radius as reasonable value which is 5 pixels
    jupiter_size = 5  # pixels
    sign_max = np.sign(np.max(jupiter_data["rows"] + jupiter_size))
    sign_min = np.sign(np.min(jupiter_data["rows"] - jupiter_size))
    return (np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_max) >= 2) and (
        np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_min) >= 2
    )


def check_spoiled_by_jupiter(cands, jupiter):
    """

    :param cand_stars: candidate star Table
    :param fids: fid Table
    :param dither: dither ACABox
    :returns: mask on cand_stars of fid trap spoiled stars, list of rejection info dicts
    """
    if jupiter is None:
        return np.zeros(len(cands), dtype=bool), []

    # Check that the candidates aren't within 15 columns of Jupiter
    colmax_jup = np.max(jupiter["cols"])
    colmin_jup = np.min(jupiter["cols"])
    tol = 15  # pixels
    ok = (cands["col"] < (colmin_jup - tol)) | (cands["col"] > (colmax_jup + tol))

    # The OK stars are OK the not OK ones are spoiled
    if np.all(ok):
        return np.zeros(len(cands), dtype=bool), []

    # Create rejection info dicts
    rej_info = [
        {
            "id": cands["id"][idx],
            "row": cands["row"][idx],
            "col": cands["col"][idx],
            "reason": "spoiled by Jupiter",
            "stage": 0,
        }
        for idx in np.where(~ok)[0]
    ]

    # return the not-ok mask and the rej_info
    return ~ok, rej_info


def update_fid_spoiler_score(cand_fids, jupiter):
    mask, _ = check_spoiled_by_jupiter(cand_fids, jupiter)
    if not np.any(mask):
        return
    cand_fids["spoiler_score"][mask] = 5


def add_jupiter_as_spoiler(date, stars, jupiter=None):
    """
    Add Jupiter as a bright object to the supplied stars table.

    :param date: Observation date
    :param stars: Table of stars
    :param jupiter: Optional Jupiter data (if None, it will be computed)
    :returns: Updated stars table with Jupiter added
    """
    if jupiter is None:
        return stars

    # Use 5 minutes as the nominal acquisition time
    acq_start = CxoTime(date)
    acq_duration = 5 * 60 * u.s
    ok = (jupiter["times"] >= acq_start.secs) & (
        jupiter["times"] <= (acq_start + acq_duration).secs
    )
    if not np.any(ok):
        return stars

    # Pick the median time for the jupiter position during the acquisition
    spoil_time = np.median(jupiter["times"][ok])
    spoil_idx = np.argmin(np.abs(jupiter["times"] - spoil_time))

    out = stars.copy()
    out.add_fake_star(
        row=jupiter["rows"][spoil_idx],
        col=jupiter["cols"][spoil_idx],
        mag=-3,  # V mag of Jupiter
        id=20,
        CLASS=100,
    )
    return out
