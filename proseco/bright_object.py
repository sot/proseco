from typing import TYPE_CHECKING, NamedTuple

import astropy.units as u
import numpy as np
from astropy.table import Table
from chandra_aca import planets
from cxotime import CxoTime, CxoTimeLike
from Quaternion import QuatLike

if TYPE_CHECKING:
    from chandra_aca.planets import PlanetPositionTable

    from proseco.core import StarsTable


def check_for_close_planets(
    date: CxoTimeLike,
    duration: float,
    att: QuatLike,
    tol=2.0,
) -> dict[str, "PlanetPositionTable"]:
    from chandra_aca.planets import (
        convert_time_format_spk,
        get_planet_angular_sep,
        get_planet_chandra_ccd_position,
    )
    from cxotime import CxoTime

    date0 = CxoTime(date)
    if duration is None:
        time_secs = convert_time_format_spk(date0, "secs")
    else:
        times = date0 + ([0, 0.5, 1] * u.s) * duration
        time_secs = convert_time_format_spk(times, "secs")

    planets_dict = {}
    for planet in planets.BRIGHT_PLANETS:
        sep = get_planet_angular_sep(
            planet,
            ra=att.ra,
            dec=att.dec,
            time=time_secs,
            observer_position="earth",
        )
        if np.all(sep > tol + 0.25):
            continue

        sep = get_planet_angular_sep(
            planet,
            ra=att.ra,
            dec=att.dec,
            time=time_secs,
            observer_position="chandra",
        )
        if np.all(sep > tol):
            continue

        planets_dict[planet] = get_planet_chandra_ccd_position(
            planet=planet,
            date=date0,
            duration=duration if duration is not None else 0.0,
            att=att,
            ccd_pad=0.0,
            ephem_source="stk",
        )

    return planets_dict


def bright_object_distribution_check(
    cand_guide_set: Table,
    bright_object_data: Table,
    dither: float = 4.0,
) -> tuple[bool, bool]:
    """
    Check for guide star CCD distribution in presence of a bright object.

    Check that there are at least two candidate guide stars on the side of the CCD that
    does not have the bright object.

    Parameters
    ----------
    cand_guide_set : Table
        Table of candidate guide stars with 'row' column.
    bright_object_data : Table
        Table with bright object positions with 'row' column.
    Returns
    -------
    tuple[bool, bool]
        Returns ``(distribution_ok, crosses_midline)`` where ``distribution_ok``
        indicates whether the candidate guide stars are correctly distributed with
        respect to the bright object and ``crosses_midline`` indicates whether the
        bright object padded row range crosses CCD row=0.
    """
    # If there is no bright object on CCD, then the check passes.
    if len(bright_object_data) == 0:
        return True, False

    # It looks like jupiter ang diam goes from 30 to 45 arcsec
    # so use 45 / 2 = 22.5 arcsec radius -> 4.5 pixels
    # and add a 4 pixel dither pad (default) corresponding to the 20 arcsec HRC pattern
    bright_object_size = 4.5  # pixels
    sign_max = np.sign(np.max(bright_object_data["row"] + bright_object_size + dither))
    sign_min = np.sign(np.min(bright_object_data["row"] - bright_object_size - dither))
    distribution_ok = (
        np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_max) >= 2
    ) and (np.count_nonzero(np.sign(cand_guide_set["row"]) != sign_min) >= 2)
    # Midline crossing is defined at CCD column 0.
    sign_max_col = np.sign(
        np.max(bright_object_data["col"] + bright_object_size + dither)
    )
    sign_min_col = np.sign(
        np.min(bright_object_data["col"] - bright_object_size - dither)
    )
    crosses_midline = sign_max_col != sign_min_col

    return distribution_ok, crosses_midline


def is_spoiled_by_bright_object(cand: Table, bright_object: Table) -> bool:
    """
    Check if a single candidate object is spoiled by a bright object.

    This is intended to be used for checking a single fid light, though
    could also be used for stars.

    Parameters
    ----------
    cand : Table Row
        A single astropy Table Row representing the candidate object and
        containing 'row' and 'col' columns.
    bright_object : Table
        Table with bright object positions with 'row' and 'col' columns, or None
        if the bright object is not present.

    Returns
    -------
    bool

    """
    # convert the cand Table Row into a Table of one row
    single_row_table = Table(cand)
    return check_spoiled_by_bright_object(single_row_table, bright_object)[0][0]


def check_spoiled_by_bright_object(
    cands: Table, bright_object: Table, tolerance: int = 15
) -> tuple[np.ndarray, list[dict]]:
    """
    Check which candidates are spoiled by a bright object.

    A candidate is considered spoiled if it is within `tolerance` pixels of the bright
    object in column.

    This method also returns a list of rejection info dicts for the spoiled candidates.

    Parameters
    ----------
    cands : Table
        Table of candidate objects with 'col' columns.
    bright_object : Table
        Table with bright object positions with 'col' columns.
    tolerance : int
        The tolerance in pixels for considering a candidate spoiled by the bright object.
        Default is 15 pixels.

    Returns
    -------
    mask : np.ndarray
        A boolean mask on `cands` where True indicates the candidate is spoiled by
        the bright object.
    rej_info : list of dict
        A list of rejection info dicts for the spoiled candidates.
    """
    if bright_object is None or len(bright_object) == 0:
        return np.zeros(len(cands), dtype=bool), []

    # Check that the candidates aren't within tolerance columns of the bright object
    colmax_obj = np.max(bright_object["col"])
    colmin_obj = np.min(bright_object["col"])
    tol = tolerance  # pixels
    ok = (cands["col"] < (colmin_obj - tol)) | (cands["col"] > (colmax_obj + tol))

    # The OK stars are OK the not OK ones are spoiled
    if np.all(ok):
        return np.zeros(len(cands), dtype=bool), []

    # Create rejection info dicts
    rej_info = [
        {
            "id": cands["id"][idx],
            "row": cands["row"][idx],
            "col": cands["col"][idx],
            "reason": "spoiled by bright object",
            "stage": 0,
        }
        for idx in np.where(~ok)[0]
    ]

    # return the not-ok mask and the rej_info
    return ~ok, rej_info


BrightObjectAcqPos = NamedTuple(
    "BrightObjectAcqPos",
    [
        ("row", float | None),
        ("col", float | None),
    ],
)


def get_bright_object_acq_pos(
    date: CxoTimeLike, bright_object: Table
) -> BrightObjectAcqPos:
    """
    Get the position of a bright object during acquisition.

    This uses `date` as the acquisition time uses the bright object position at that time.
    If the bright object is not on the CCD within 2 ks of acquisition start, returns (None, None).

    Parameters
    ----------
    date : CxoTimeLike
        The acquisition date.
    bright_object : Table
        Table with bright object positions with 'time', 'row', and 'col' columns.

    Returns
    -------
    acquisition_position : BrightObjectAcqPos
        The (row, col) position of the bright object during acquisition, or None, None
        if the bright object is not present during acquisition.
    """
    # Use 5 minutes as the nominal acquisition time
    acq_start = CxoTime(date)

    # If the first time in the bright_object table is not within 2000 seconds then return
    # None, None. This reflects a rare but possible situation where the object drifts onto
    # the CCD well after the acquisition time.
    if (
        len(bright_object) == 0
        or np.abs(bright_object["time"][0] - acq_start.secs) > 2000
    ):
        return BrightObjectAcqPos(None, None)

    # Otherwise use the first row and col in the bright_object table
    return BrightObjectAcqPos(bright_object["row"][0], bright_object["col"][0])


def add_bright_object_as_acq_spoilers(
    date: "CxoTime | CxoTimeLike",
    stars: "StarsTable",
    bright_object: Table,
    mag: float = -3.0,
    tolerance: int = 15,
) -> "StarsTable":
    """Enforce column keepout zone around bright object using many fake bright stars.

    This adds a bunch of bright objects to the supplied stars table. This is specific to
    acquisition as it uses the acquisition time as a the reference time for the position
    of the bright object.

    Parameters
    ----------
    date : CxoTimeLike
        The observation date.
    stars : StarsTable
        The stars table to which to add the fake stars representing the bright object.
    bright_object : Table
        Table with bright object positions with 'time', 'row', and 'col' columns.
    mag : float, optional
        Magnitude of the bright object (default=-3.0 for Jupiter).
    tolerance : int, optional
        Column tolerance for the keepout zone (default=15 pixels).

    Returns
    -------
    StarsTable
        A copy of the input `stars` table with the fake stars added.
    """
    if len(bright_object) == 0:
        return stars

    # Bright object acq position
    acq_pos = get_bright_object_acq_pos(date, bright_object=bright_object)

    out = stars.copy()
    idincr = 0
    for irow in np.arange(-505, 510, 5):
        for icol in np.arange(acq_pos.col - tolerance, acq_pos.col + tolerance + 1, 5):
            out.add_fake_star(
                row=irow,
                col=icol,
                mag=mag,
                id=20 + idincr,
                CLASS=100,
            )
            idincr += 1
    return out
