import astropy.units as u
import numpy as np
from astropy.table import Table
from chandra_aca import planets
from chandra_aca.transform import eci_to_radec, radec_to_yagzag, yagzag_to_pixels
from cheta.comps import ephem_stk
from cxotime import CxoTime


def date_is_excluded(date):
    """
    Check if the given date is in the list of excluded dates for Jupiter observations.

    If the date is within any of the excluded date ranges, Jupiter checks do not apply,
    and this function returns True.

    Parameters
    ----------
    date : str or CxoTime
        The date to check.

    Returns
    -------
    bool
        True if the date is in an excluded range, False otherwise.
    """
    date = CxoTime(date)
    from proseco.characteristics_jupiter import exclude_dates

    for ex in exclude_dates:
        if CxoTime(ex["start"]) <= date <= CxoTime(ex["stop"]):
            return True
    return False


def get_jupiter_position(date, duration, att):
    """
    Get the position of Jupiter on the ACA CCD.

    Parameters
    ----------
    date : str or CxoTime
        The start date of the observation (acquisition time)
    duration : float
        The duration of the observation in seconds.
    att : Quaternion or Quat-compatible
        The attitude Quaternion.

    Returns
    -------
    Table or None
        A table with columns 'time', 'row', 'col' for the times when Jupiter
        is on the CCD. If Jupiter is never on the CCD, returns None.
    """
    date0 = CxoTime(date)

    # First 5 minutes: sample every 60 seconds
    first_phase = np.arange(0, min(300, duration) + 1, 60)
    # After 5 minutes: sample every 1000 seconds, not duplicating the 300s point
    if duration > 300:
        second_phase = np.arange(300 + 1000, duration, 1000)
    else:
        second_phase = np.array([])
    # Always include the end point
    all_times = np.unique(np.concatenate([first_phase, second_phase, [duration]]))
    dates = date0 + all_times * u.s
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
                "time": times[ok],
                "row": row[ok],
                "col": col[ok],
            }
        )
    else:
        out = None
    return out


def jupiter_distribution_check(cand_guide_set, jupiter_data):
    """
    Check for guide star CCD distribution in presence of Jupiter.

    Check that there are at least two candidate guide stars on the side of the CCD
    that does not have Jupiter.

    Parameters
    ----------
    cand_guide_set : Table
        Table of candidate guide stars with 'row' column.
    jupiter_data : Table
        Table with Jupiter positions with 'row' column.

    Returns
    -------
    bool
        True if the candidate guide stars are correctly distributed with respect to Jupiter,
        False otherwise.
    """
    # It looks like jupiter ang diam goes from 30 to 45 arcsec
    # so use 25 arcsec radius as reasonable value which is 5 pixels
    jupiter_size = 4.5  # pixels
    dither = 4 # pixels
    sign_max = np.sign(np.max(jupiter_data["row"] + jupiter_size + dither))
    sign_min = np.sign(np.min(jupiter_data["row"] - jupiter_size - dither))
    return (np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_max) >= 2) and (
        np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_min) >= 2
    )


def is_spoiled_by_jupiter(cand, jupiter):
    """
    Check if a single candidate object is spoiled by Jupiter.

    This is intended to be used for checking a single fid light, though
    could also be used for stars.

    Parameters
    ----------
    cand : Table Row
        A single astropy Table Row representing the candidate object and
        containing 'row' and 'col' columns.
    jupiter : Table or None
        Table with Jupiter positions with 'row' and 'col' columns, or None
        if Jupiter is not present.

    Returns
    -------
    bool

    """
    # convert the cand Table Row into a Table of one row
    single_row_table = Table(cand)
    return check_spoiled_by_jupiter(single_row_table, jupiter)[0][0]


def check_spoiled_by_jupiter(cands, jupiter, tolerance=15):
    """
    Check which candidate are spoiled by Jupiter.

    A candidate is considered spoiled if it is within `tolerance` pixels
    of Jupiter in column.

    This method also returns a list of rejection info dicts for the spoiled candidates.

    Parameters
    ----------
    cands: Table
        Table of candidate objects with 'col' columns.
    jupiter: Table or None
        Table with Jupiter positions with 'col' columns, or None if Jupiter is not present.
    tolerance: int
        The tolerance in pixels for considering a candidate spoiled by Jupiter.
        Default is 15 pixels.

    Returns
    -------
    mask: np.ndarray
        A boolean mask on `cands` where True indicates the candidate is spoiled by Jupiter.
    rej_info: list of dict
        A list of rejection info dicts for the spoiled candidates.
    """
    if jupiter is None:
        return np.zeros(len(cands), dtype=bool), []

    # Check that the candidates aren't within 15 columns of Jupiter
    colmax_jup = np.max(jupiter["col"])
    colmin_jup = np.min(jupiter["col"])
    tol = tolerance  # pixels
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


def get_jupiter_acq_pos(date, jupiter):
    """
    Get the position of Jupiter during acquisition.

    This uses `date` as the acquisition time and assumes a nominal
    acquisition duration of 5 minutes to find the median position of
    Jupiter during that time.

    Parameters
    ----------
    date : str or CxoTime
        The acquisition date.
    jupiter : Table
        Table with Jupiter positions with 'time', 'row', and 'col' columns.

    Returns
    -------
    (row, col) : tuple of float or None, None
        The (row, col) position of Jupiter during acquisition, or None, None
        if Jupiter is not present during acquisition.
    """
    # Use 5 minutes as the nominal acquisition time
    acq_start = CxoTime(date)
    acq_duration = 5 * 60 * u.s
    ok = (jupiter["time"] >= acq_start.secs) & (
        jupiter["time"] <= (acq_start + acq_duration).secs
    )
    if not np.any(ok):
        return None, None

    # Pick the median time for the jupiter position during the acquisition
    spoil_time = np.median(jupiter["time"][ok])
    spoil_idx = np.argmin(np.abs(jupiter["time"] - spoil_time))
    return (jupiter["row"][spoil_idx], jupiter["col"][spoil_idx])


def add_jupiter_as_lots_of_acq_spoilers(date, stars, jupiter=None):
    """
    Add Jupiter as a bunch of bright objects to the supplied stars table.

    This is specific to acquisition as it uses the acquisition time as a the
    reference time for the position of Jupiter.

    Parameters
    ----------
    date : CxoTime or CxoTime-compatible
        The observation date.
    stars : StarsTable
        The stars table to which to add the fake stars representing Jupiter.
    jupiter : Table or None
        Table with Jupiter positions with 'time', 'row', and 'col' columns,
        or None if Jupiter is not present.

    Returns
    -------
    StarsTable
        A copy of the input `stars` table with the fake stars added.
    """
    if jupiter is None:
        return stars

    # Jupiter acq position
    row, col = get_jupiter_acq_pos(date, jupiter=jupiter)

    out = stars.copy()
    idincr = 0
    for irow in np.arange(-505, 510, 5):
        for icol in np.arange(col - 15, col + 16, 5):
            out.add_fake_star(
                row=irow,
                col=icol,
                mag=-3,  # V mag of Jupiter
                id=20 + idincr,
                CLASS=100,
            )
            idincr += 1
    return out
