# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Reselect every catalog in load proseco pickle files with the importable proseco.

Each catalog is reselected with the flight ``call_args`` and the flight dark cal
(``dark_date``), so with the same proseco version the result reproduces the flight
catalog. For each input ``<LOAD>_proseco.pkl.gz`` this writes ``<out-dir>/<LOAD>.pkl``
containing a dict with the proseco version and file used, the input pickle path, and
``acas``, a dict of ``{obsid: ACATable}``.

Use ``PYTHONPATH`` to choose the proseco version, e.g. from the proseco repo root::

  git fetch origin
  git worktree add --detach /tmp/proseco-master origin/master
  env PYTHONPATH=/tmp/proseco-master python validate/pr-416/rerun_loads.py \\
      --out-dir runs/master $SKA/data/mpcrit1/mplogs/2026/*/oflsa/output/*_proseco.pkl.gz
  env PYTHONPATH=$PWD python validate/pr-416/rerun_loads.py \\
      --out-dir runs/pr416 $SKA/data/mpcrit1/mplogs/2026/*/oflsa/output/*_proseco.pkl.gz
"""

import argparse
import gzip
import pickle
import subprocess
import warnings
from multiprocessing import Pool
from pathlib import Path

import proseco
from proseco import get_aca_catalog


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("pickles", nargs="+", help="Load *_proseco.pkl.gz files")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--n-proc", type=int, default=8, help="Number of processes (default=8)"
    )
    return parser


def get_proseco_git_commit():
    """Git commit of the imported proseco source, or None if not in a git repo.

    ``proseco.__version__`` can come from installed package metadata, so it does not
    reliably identify the source when proseco is imported via PYTHONPATH.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(proseco.__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def rerun_load(in_pkl, out_dir):
    """Reselect all catalogs in ``in_pkl`` and write ``<out_dir>/<LOAD>.pkl``."""
    warnings.filterwarnings("ignore")
    in_pkl = Path(in_pkl)
    load_name = in_pkl.name.split("_")[0]
    out_pkl = Path(out_dir) / f"{load_name}.pkl"

    with gzip.open(in_pkl, "rb") as fh:
        acas_flight = pickle.load(fh)

    acas = {
        obsid: get_aca_catalog(dark_date=aca.dark_date, **aca.call_args)
        for obsid, aca in acas_flight.items()
    }
    out = {
        "proseco_version": proseco.__version__,
        "proseco_file": proseco.__file__,
        "proseco_git_commit": get_proseco_git_commit(),
        "input_pickle": str(in_pkl.absolute()),
        "acas": acas,
    }
    with open(out_pkl, "wb") as fh:
        pickle.dump(out, fh)

    return f"{load_name}: {len(acas)} catalogs -> {out_pkl}"


def main(args=None):
    opt = get_parser().parse_args(args)
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Using proseco {proseco.__version__} (git {get_proseco_git_commit()}) "
        f"from {proseco.__file__}"
    )

    with Pool(opt.n_proc) as pool:
        results = [
            pool.apply_async(rerun_load, (in_pkl, out_dir)) for in_pkl in opt.pickles
        ]
        for result in results:
            print(result.get())


if __name__ == "__main__":
    main()
