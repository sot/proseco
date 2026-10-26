# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Compare reselected catalogs from two ``rerun_loads.py`` output directories.

Catalogs are compared on ``id``, ``type``, ``sz`` and ``halfw``. For each load this
also counts catalogs in the reference run that differ from the flight catalog (input
pickle), which is expected when the flight catalog was made with a different proseco
version.

For each changed catalog this prints the fid sets, acq P2 and any fid search
spoilers (star mag < fid mag + FID.search_spoiler_mag_margin) in the new run.

Example from the proseco repo root::

  python validate/pr-416/diff_loads.py runs/master runs/pr416
"""

import argparse
import gzip
import pickle
from pathlib import Path

NAMES = ["id", "type", "sz", "halfw"]
SEARCH_SPOILER_MAG_MARGIN = 0.25  # Same as FID.search_spoiler_mag_margin in PR #416


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("ref_dir", help="Reference run directory (e.g. master)")
    parser.add_argument("new_dir", help="New run directory (e.g. PR branch)")
    return parser


def cat_lines(aca):
    """Sorted table lines of key catalog columns for comparison."""
    out = aca[NAMES].copy()
    out.sort(["type", "id"])
    return out.pformat(max_width=-1)


def get_p2(aca):
    return -aca.acqs.get_log_p_2_or_fewer()


def get_search_spoilers(aca):
    """List of fid search spoiler descriptions for candidate fids in ``aca``."""
    out = []
    cand_fids = aca.fids.cand_fids
    if cand_fids is None:
        return out
    for fid in cand_fids:
        if len(fid["spoilers"]) == 0:
            continue
        out.extend(
            f"fid {fid['id']}: {star['id']} mag={star['mag']:.2f} "
            f"dy={star['yang'] - fid['yang']:+.0f} "
            f"dz={star['zang'] - fid['zang']:+.0f}"
            for star in fid["spoilers"]
            if star["mag"] < fid["mag"] + SEARCH_SPOILER_MAG_MARGIN
        )
    return out


def main(args=None):
    opt = get_parser().parse_args(args)
    ref_dir = Path(opt.ref_dir)
    new_dir = Path(opt.new_dir)

    n_cat = 0
    n_cat_fids = 0
    n_ref_flight_diff = {}
    changes = []

    ref_pkls = sorted(ref_dir.glob("*.pkl"))
    for ref_pkl in ref_pkls:
        with open(ref_pkl, "rb") as fh:
            ref = pickle.load(fh)
        with open(new_dir / ref_pkl.name, "rb") as fh:
            new = pickle.load(fh)
        with gzip.open(ref["input_pickle"], "rb") as fh:
            flight_acas = pickle.load(fh)
        load_name = ref_pkl.stem

        for obsid, aca_flight in flight_acas.items():
            aca_ref = ref["acas"][obsid]
            aca_new = new["acas"][obsid]
            n_cat += 1
            n_cat_fids += aca_flight.n_fid > 0
            if cat_lines(aca_ref) != cat_lines(aca_flight):
                n_ref_flight_diff[load_name] = n_ref_flight_diff.get(load_name, 0) + 1

            lines_ref = cat_lines(aca_ref)
            lines_new = cat_lines(aca_new)
            if lines_ref == lines_new:
                continue

            other = set(lines_ref) ^ set(lines_new)
            changes.append(
                {
                    "load": load_name,
                    "obsid": int(obsid),
                    "date": aca_ref.date[:8],
                    "detector": aca_ref.detector,
                    "target": aca_ref.target_name or "",
                    "fids_ref": sorted(int(x) for x in aca_ref.fids["id"]),
                    "fids_new": sorted(int(x) for x in aca_new.fids["id"]),
                    "p2_ref": get_p2(aca_ref),
                    "p2_new": get_p2(aca_new),
                    "n_other": len([x for x in other if " FID " not in x]) // 2,
                    "spoilers": get_search_spoilers(aca_new),
                }
            )

    for label, run_dir in (("Reference", ref_dir), ("New", new_dir)):
        with open(run_dir / ref_pkls[0].name, "rb") as fh:
            run = pickle.load(fh)
        print(
            f"{label} proseco {run['proseco_version']} "
            f"(git {run.get('proseco_git_commit')}) from {run['proseco_file']}"
        )
    print(f"Loads: {len(ref_pkls)}  catalogs: {n_cat}  with fids: {n_cat_fids}")
    print(
        f"Reference != flight: {sum(n_ref_flight_diff.values())} catalogs "
        f"in {len(n_ref_flight_diff)} loads"
    )
    print(f"New != reference: {len(changes)} catalogs\n")
    for chg in changes:
        print(
            f"{chg['load']} {chg['obsid']:>6} {chg['date']} {chg['detector']:6} "
            f"{chg['target'][:22]:22} fids {chg['fids_ref']} -> {chg['fids_new']}  "
            f"P2 {chg['p2_ref']:.2f} -> {chg['p2_new']:.2f}  "
            f"n_other_changed={chg['n_other']}"
        )
        for spoiler in chg["spoilers"]:
            print(f"    {spoiler}")


if __name__ == "__main__":
    main()
