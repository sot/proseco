from typing import TYPE_CHECKING, NamedTuple

import astropy.units as u
import numpy as np
from astropy.table import Table
from chandra_aca import planets
from chandra_aca.transform import eci_to_radec, radec_to_yagzag, yagzag_to_pixels
from cheta.comps import ephem_stk
from cxotime import CxoTime, CxoTimeLike
from Quaternion import QuatLike

from proseco import characteristics_jupiter
from proseco.characteristics_jupiter import JupiterPositionTable

if TYPE_CHECKING:
    from proseco.core import StarsTable


def date_is_excluded(date: CxoTimeLike) -> bool:
    """
    Check if the given date is in the list of excluded dates for Jupiter observations.

    If the date is within any of the excluded date ranges, Jupiter checks do not apply,
    and this function returns True.

    Parameters
    ----------
    date : CxoTimeLike
        The date to check.

    Returns
    -------
    bool
        True if the date is in an excluded range, False otherwise.
    """
    date_str = CxoTime(date).date
    # Exclude dates are in CXO date format and we compare as strings. This prevents a
    # warning from pyerfa about dubius dates in the future where leap seconds may be
    # uncertain.
    for exclude_date in characteristics_jupiter.exclude_dates:
        if exclude_date["start"] <= date_str <= exclude_date["stop"]:
            return True
    return False


def get_jupiter_position(
    date: CxoTimeLike,
    duration: float,
    att: QuatLike,
) -> JupiterPositionTable:
    """
    Get the position of Jupiter on the ACA CCD.

    Parameters
    ----------
    date : CxoTimeLike
        The start date of the observation (acquisition time)
    duration : float
        The duration of the observation in seconds.
    att : Quaternion or Quat-compatible
        The attitude Quaternion.

    Returns
    -------
    JupiterPositionTable
        A table with columns 'time', 'row', 'col' for the times when Jupiter
        is on the CCD. If Jupiter is never on the CCD this is a table with zero rows.
    """
    date0 = CxoTime(date)
    # dates is a 1-d array with a minimum of 2 points (the end points)
    dates = CxoTime.linspace(date0, date0 + duration * u.s, step_max=1000 * u.s)
    times = dates.secs

    chandra_ephem = ephem_stk.get_ephemeris_stk(start=dates[0], stop=dates[-1])

    # Get Jupiter positions using chandra_aca.planets
    ephem = {
        key: np.interp(times, chandra_ephem["time"], chandra_ephem[key])
        for key in ["x", "y", "z"]
    }
    # shape (len(dates), 3), units km
    pos_earth = planets.get_planet_barycentric("earth", dates)

    chandra_eci = np.array(
        [
            ephem["x"] / 1000,  # convert m from get_ephemeris_stk to km,
            ephem["y"] / 1000,
            ephem["z"] / 1000,
        ]
    ).transpose()
    eci = planets.get_planet_eci("jupiter", dates, pos_observer=pos_earth + chandra_eci)

    # Convert ECI position to RA, Dec => yag, zag => row, col
    ra, dec = eci_to_radec(eci)
    yag, zag = radec_to_yagzag(ra, dec, att)
    row, col = yagzag_to_pixels(yag, zag, allow_bad=True)

    lim0 = -512
    lim1 = 511
    # Only care about planet within the CCD
    ok = (row >= lim0) & (row <= lim1) & (col >= lim0) & (col <= lim1)
    out = JupiterPositionTable(
        {
            "time": times[ok],
            "row": row[ok],
            "col": col[ok],
        }
    )
    return out


def jupiter_distribution_check(
    cand_guide_set: Table,
    jupiter_data: JupiterPositionTable,
) -> bool:
    """
    Check for guide star CCD distribution in presence of Jupiter.

    Check that there are at least two candidate guide stars on the side of the CCD that
    does not have Jupiter.

    Parameters
    ----------
    cand_guide_set : Table
        Table of candidate guide stars with 'row' column.
    jupiter_data : JupiterPositionTable
        Table with Jupiter positions with 'row' column.

    Returns
    -------
    bool
        True if the candidate guide stars are correctly distributed with respect to
        Jupiter, False otherwise.
    """
    # If there is no Jupiter on CCD, then the check passes.
    if len(jupiter_data) == 0:
        return True

    # It looks like jupiter ang diam goes from 30 to 45 arcsec
    # so use 45 / 2 = 22.5 arcsec radius -> 4.5 pixels
    # and add a 4 pixel dither pad corresponding to the 20 arcsec HRC pattern
    jupiter_size = 4.5  # pixels
    dither = 4  # pixels
    sign_max = np.sign(np.max(jupiter_data["row"] + jupiter_size + dither))
    sign_min = np.sign(np.min(jupiter_data["row"] - jupiter_size - dither))
    return (np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_max) >= 2) and (
        np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_min) >= 2
    )


def is_spoiled_by_jupiter(cand: Table, jupiter: JupiterPositionTable) -> bool:
    """
    Check if a single candidate object is spoiled by Jupiter.

    This is intended to be used for checking a single fid light, though
    could also be used for stars.

    Parameters
    ----------
    cand : Table Row
        A single astropy Table Row representing the candidate object and
        containing 'row' and 'col' columns.
    jupiter : JupiterPositionTable
        Table with Jupiter positions with 'row' and 'col' columns, or None
        if Jupiter is not present.

    Returns
    -------
    bool

    """
    # convert the cand Table Row into a Table of one row
    single_row_table = Table(cand)
    return check_spoiled_by_jupiter(single_row_table, jupiter)[0][0]


def check_spoiled_by_jupiter(
    cands: Table, jupiter: JupiterPositionTable, tolerance: int = 15
) -> tuple[np.ndarray, list[dict]]:
    """
    Check which candidate are spoiled by Jupiter.

    A candidate is considered spoiled if it is within `tolerance` pixels of Jupiter in
    column.

    This method also returns a list of rejection info dicts for the spoiled candidates.

    Parameters
    ----------
    cands: Table
        Table of candidate objects with 'col' columns.
    jupiter: JupiterPositionTable
        Table with Jupiter positions with 'col' columns.
    tolerance: int
        The tolerance in pixels for considering a candidate spoiled by Jupiter. Default
        is 15 pixels.

    Returns
    -------
    mask: np.ndarray
        A boolean mask on `cands` where True indicates the candidate is spoiled by
        Jupiter.
    rej_info: list of dict
        A list of rejection info dicts for the spoiled candidates.
    """
    if len(jupiter) == 0:
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


JupiterAcqPos = NamedTuple(
    "JupiterAcqPos",
    [
        ("row", float | None),
        ("col", float | None),
    ],
)


def get_jupiter_acq_pos(
    date: CxoTimeLike, jupiter: JupiterPositionTable
) -> JupiterAcqPos:
    """
    Get the position of Jupiter during acquisition.

    This uses `date` as the acquisition time uses the Jupiter position at that time.
    If Jupiter is not on the CCD within 2 ks of acquisition start, returns (None, None).

    Parameters
    ----------
    date : CxoTimeLike
        The acquisition date.
    jupiter : Table
        Table with Jupiter positions with 'time', 'row', and 'col' columns.

    Returns
    -------
    acquisition_position : JupiterAcqPos
        The (row, col) position of Jupiter during acquisition, or None, None
        if Jupiter is not present during acquisition.
    """
    # Use 5 minutes as the nominal acquisition time
    acq_start = CxoTime(date)

    # If the first time in the jupiter table is not within 2000 seconds then return
    # None, None. This reflects a rare but possible situation where Jupiter drifts onto
    # the CCD well after the acquisition time.
    if len(jupiter) == 0 or np.abs(jupiter["time"][0] - acq_start.secs) > 2000:
        return JupiterAcqPos(None, None)

    # Otherwise use the first row and col in the jupiter table
    return JupiterAcqPos(jupiter["row"][0], jupiter["col"][0])


def add_jupiter_as_lots_of_acq_spoilers(
    date: "CxoTime | CxoTimeLike", stars: "StarsTable", jupiter: JupiterPositionTable
) -> "StarsTable":
    """Enforce 15-column keepout zone around Jupiter using many fake bright stars.

    This adds a bunch of bright objects to the supplied stars table. This is specific to
    acquisition as it uses the acquisition time as a the reference time for the position
    of Jupiter.

    Parameters
    ----------
    date : CxoTimeLike
        The observation date.
    stars : StarsTable
        The stars table to which to add the fake stars representing Jupiter.
    jupiter : JupiterPositionTable
        Table with Jupiter positions with 'time', 'row', and 'col' columns.

    Returns
    -------
    StarsTable
        A copy of the input `stars` table with the fake stars added.
    """
    if len(jupiter) == 0:
        return stars

    # Jupiter acq position
    acq_pos = get_jupiter_acq_pos(date, jupiter=jupiter)

    out = stars.copy()
    idincr = 0
    for irow in np.arange(-505, 510, 5):
        for icol in np.arange(acq_pos.col - 15, acq_pos.col + 16, 5):
            out.add_fake_star(
                row=irow,
                col=icol,
                mag=-3,  # V mag of Jupiter
                id=20 + idincr,
                CLASS=100,
            )
            idincr += 1
    return out
