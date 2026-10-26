# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Check in flight telemetry whether a fid light or a nearby bright star was acquired.

For ``obsid`` and ``fid_id`` in a load ``*_proseco.pkl.gz`` this prints:

- Commanded fid yang/zang and slot from the flight catalog.
- Telemetry for that slot (AOACFID, AOACFCT, AOACYAN, AOACZAN, AOACMAG) from the
  start of the observation through the fid search and initial track.
- AGASC stars brighter than ``--mag-limit`` within ``--radius`` of the fid, with
  position at the catalog attitude and at the estimated attitude (AOATTQT) at the
  time of first fid track.

A tracked position that matches the star (and drifts as the attitude error is
corrected) means the star was acquired instead of the fid. A fid is fixed in the ACA
frame. Requires cheta telemetry access.

Example::

  python validate/pr-416/check_fid_acq.py \\
      $SKA/data/mpcrit1/mplogs/2026/OCT1226/oflsa/output/OCT1226A_proseco.pkl.gz 33497 6
"""

import argparse
import gzip
import pickle

import agasc
import astropy.units as u
import numpy as np
from astropy.table import Table
from chandra_aca.transform import radec_to_yagzag
from cheta import fetch
from cxotime import CxoTime
from kadi.commands.observations import get_observations
from Quaternion import Quat, normalize


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("pickle", help="Load *_proseco.pkl.gz file")
    parser.add_argument("obsid", type=int, help="Obsid")
    parser.add_argument("fid_id", type=int, help="Fid id")
    parser.add_argument(
        "--dt",
        type=float,
        default=300,
        help="Telemetry duration after obs start (sec, default=300)",
    )
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=7.25,
        help="AGASC star mag limit (default=7.25)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=200,
        help="AGASC search radius from fid (arcsec, default=200)",
    )
    return parser


def main(args=None):
    opt = get_parser().parse_args(args)

    with gzip.open(opt.pickle, "rb") as fh:
        aca = pickle.load(fh)[float(opt.obsid)]
    ok = (aca["type"] == "FID") & (aca["id"] == opt.fid_id)
    if not np.any(ok):
        raise ValueError(f"fid {opt.fid_id} not in catalog for obsid {opt.obsid}")
    fid = aca[ok][0]
    slot = fid["slot"]
    print(
        f"Obsid {opt.obsid} fid {opt.fid_id} slot {slot}: commanded "
        f"yang={fid['yang']:.1f} zang={fid['zang']:.1f}"
    )

    obs = get_observations(obsid=opt.obsid)[0]
    start = CxoTime(obs["obs_start"])
    print(f"obs_start={start.date}")

    msids = [f"AOAC{name}{slot}" for name in ("FID", "FCT", "YAN", "ZAN", "MAG")]
    dat = fetch.MSIDset(msids, start, start + opt.dt * u.s)
    dat.interpolate(times=dat[msids[0]].times, filter_bad=False)
    telem = Table({msid: dat[msid].vals for msid in msids})
    telem.add_column(CxoTime(dat.times).date, index=0, name="date")
    for msid in msids[2:]:
        telem[msid].format = ".2f"

    # Fid search and track rows: print status changes and every 30th row (~30 s)
    is_fid = np.char.strip(telem[msids[0]].astype(str)) == "FID"
    fct = telem[msids[1]]
    changed = np.concatenate([[True], fct[1:] != fct[:-1]])
    show = is_fid & (changed | (np.arange(len(telem)) % 30 == 0))
    telem[show].pprint_all()

    trak = is_fid & (np.char.strip(fct.astype(str)) == "TRAK")
    if not np.any(trak):
        print("No fid track in telemetry interval")
        return
    t_trak = CxoTime(telem["date"][np.flatnonzero(trak)[0]])

    # Estimated attitude at first fid track
    q_msids = [f"AOATTQT{ii}" for ii in range(1, 5)]
    qdat = fetch.MSIDset(q_msids, t_trak - 10 * u.s, t_trak + 10 * u.s)
    idx = np.argmin(np.abs(qdat[q_msids[0]].times - t_trak.secs))
    q_est = Quat(q=normalize([qdat[msid].vals[idx] for msid in q_msids]))

    # Bright AGASC stars near the fid
    q_cat = aca.att if isinstance(aca.att, Quat) else Quat(aca.att)
    ra, dec = q_cat.ra, q_cat.dec
    stars = agasc.get_agasc_cone(ra, dec, radius=1.5, date=aca.date)
    stars = stars[stars["MAG_ACA"] < opt.mag_limit]
    rows = []
    for star in stars:
        y_cat, z_cat = radec_to_yagzag(star["RA_PMCORR"], star["DEC_PMCORR"], q_cat)
        if np.hypot(y_cat - fid["yang"], z_cat - fid["zang"]) > opt.radius:
            continue
        y_est, z_est = radec_to_yagzag(star["RA_PMCORR"], star["DEC_PMCORR"], q_est)
        rows.append(
            (
                star["AGASC_ID"],
                star["MAG_ACA"],
                y_cat - fid["yang"],
                z_cat - fid["zang"],
                y_est,
                z_est,
            )
        )

    print(f"\nFirst fid TRAK at {t_trak.date}")
    if not rows:
        print(
            f"No AGASC stars with MAG_ACA < {opt.mag_limit} within "
            f"{opt.radius} arcsec of fid"
        )
        return
    out = Table(
        rows=rows,
        names=["agasc_id", "mag_aca", "dy_cat", "dz_cat", "yang_est", "zang_est"],
    )
    for name in out.colnames[1:]:
        out[name].format = ".2f"
    print(
        f"AGASC stars with MAG_ACA < {opt.mag_limit} within {opt.radius} arcsec "
        "of fid:\n  dy_cat, dz_cat: offset from fid at catalog attitude\n"
        "  yang_est, zang_est: position at estimated attitude at first fid TRAK"
    )
    out.pprint_all()


if __name__ == "__main__":
    main()
