"""
Do three steps to run starcheck on proseco-generated catalogs for one or more flight weekly loads:

- Copy the flight load products into a local working dir
- Generate proseco catalogs and update products in place to reflect the new cats
- Run starcheck

It is assumed the flight load products are in a structure like::

  <load_root>/2018/JUL0218/oflsa  <==  For <load_name> = JUL0218A

The outputs go into a directory like::

  <out_root>/JUL0218A/

Typically this is going to be run from the proseco git repo, using::

  $ ska3
  $ cd ~/git/proseco
  $ git checkout <branch/tag>  # If doing run-for-record, be sure to use the release tag
  $ cd validate
  $ env PYTHONPATH=$PWD/.. ipython
  >>> from starcheck_proseco import *
  >>> run_all()
  >>> exit()

Publish the results with (where <version> is the proseco version):

  $ rsync -zarv --prune-empty-dirs --include "*/"  --include="starcheck*" --include="starcheck/*" \
    --exclude="*" loads/proseco/ /proj/sot/ska/www/ASPECT/proseco/loads-<version>/

This whole process is set up to not repeat existing steps (since it is generally slow).
To re-do things, either completely delete the proseco load directory (probably your best bet),
or remove:

- 000-proseco   : to re-do updating products
- starcheck.txt : to re-do starcheck

Clean some known non-errors:

  perl -pi -e 's/.*not found within 10 arcsec of.*//' */starcheck.html
  perl -pi -e 's/.*is in star catalog but.*//' */starcheck.html
  perl -pi -e 's/.*Fewer than 3 ACQ stars.*//' */starcheck.html

"""

import pickle
from pathlib import Path
import subprocess
import shutil

import parse_cm
from proseco import get_aca_catalog
from proseco.tests.test_common import get_starcheck_obs_kwargs
import Ska.File


def run_starcheck_proseco(
    load_name='JUL0218A',
    out_root=Path('loads', 'proseco'),
    load_root=Path('/data', 'mpcrit1', 'mplogs'),
):
    """
    Do three steps to run starcheck on proseco-generated catalogs for a flight weekly load:

    - Copy the flight load products into a local working dir
    - Generate proseco catalogs and update products in place to reflect the new cats
    - Run starcheck

    It is assumed the flight load products are in a structure like::

      <load_root>/2018/JUL0218/oflsa  <==  For JUL0218A

    :param load_name: Load name like JUL0218A
    :param out_root: Output root, where results go into <out_door>/<load_name>
    :param load_root: Source root for load products (default=/data/mpcrit1/mplogs)
    """
    copy_load_products(load_name, out_root, load_root)
    update_products(load_name, out_root)
    run_starcheck(load_name, out_root)


def run_all(out_root=Path('loads', 'proseco')):
    """
    Convenience function to run starcheck_proseco on a standard set of 10 weekly loads
    for run-for-record validation testing.
    """
    load_names = [
        'DEC0318B',
        'JUL0918A',  # dark cals
        'JUL0218A',
        'JUN2318A',
        'JUN1118A',
        'JUN0418A',
        'MAY2118A',
        'MAY1418A',
        'MAY0718A',
        'APR3018A',
    ]
    for load_name in load_names:
        run_starcheck_proseco(load_name)


def copy_load_products(load_name, out_root, load_root):
    out_dir = Path(out_root, load_name)
    if out_dir.exists():
        print(f'Skipping copy_load_products {out_dir} already exists')
        return

    load_root = Path(load_root)
    load_version = load_name[-1:].lower()
    load_year = '20' + load_name[-3:-1]

    load_dir = load_root / load_year / load_name[:-1] / ('ofls' + load_version)
    print(f'Copying load products from {load_dir} to {out_dir}')

    globs = (
        'CR*.tlr',
        'CR*.backstop',
        'starcheck.txt',
        'mps/md*.dot',
        'mps/or/*.or',
        'mps/ode/characteristics/CHARACTERIS_*',
        'mps/m*.sum',
        'output/*_ManErr.txt',
        'output/*_dynamical_offsets.txt',
        'output/TEST_mechcheck.txt',
        'History/DITHER.txt',
        'History/FIDSEL.txt',
        'History/RADMON.txt',
        'History/SIMFOCUS.txt',
        'History/SIMTRANS.txt',
    )

    for glob in globs:
        for src in load_dir.glob(glob):
            dest = out_dir / src.relative_to(load_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def update_products(load_name, out_root, orv_pickle=None):
    load_dir = Path(out_root) / load_name
    touch_file = load_dir / '000-proseco'
    if touch_file.exists():
        print('Skipping update products, already done')
        return

    gspath = list(load_dir.glob('mps/mg*.sum'))[0]
    bspath = list(load_dir.glob('CR*.backstop'))[0]
    dotpath = list(load_dir.glob('mps/md*.dot'))[0]

    gs = parse_cm.read_guide_summary(gspath)
    bs = parse_cm.read_backstop(bspath)
    dt = parse_cm.read_dot_as_list(dotpath)

    if orv_pickle is None:
        obs_kwargs = get_starcheck_obs_kwargs(load_dir / 'starcheck.txt')
        cats = {}
        for obsid, kwargs in obs_kwargs.items():
            print(f'  Generating proseco catalog for {obsid}')
            cats[obsid] = get_aca_catalog(raise_exc=True, **kwargs)
    else:
        orvcats = pickle.load(open(orv_pickle, 'rb'))
        # The ORV proseco pickle uses strings as the keys, so int them
        cats = {int(k): v for k, v in orvcats.items()}
        # It looks like the proseco pickle has catalogs for no-catalog
        # observations (dark cal), so trim those
        extras = set(cats.keys()) - set([summ['obsid'] for summ in gs['summs']])
        for obs in extras:
            del cats[obs]

    print('Updating backstop, DOT and guide summary products')
    gs = parse_cm.replace_starcat_guide_summary(gs, cats)
    bs = parse_cm.replace_starcat_backstop(bs, cats)
    dt = parse_cm.replace_starcat_dot(dt, cats)

    print('Writing ...')
    parse_cm.write_guide_summary(gs, gspath)
    parse_cm.write_backstop(bs, bspath)
    parse_cm.write_dot(dt, dotpath)

    touch_file.touch()


def run_starcheck(load_name, out_root):
    load_dir = Path(out_root, load_name)
    starcheck_html = load_dir / 'starcheck.html'
    if starcheck_html.exists():
        print(f'Skipping because {starcheck_html} already exists')
        return
    print(f'Running starcheck in {load_dir}')
    with Ska.File.chdir(str(load_dir)):
        subprocess.run(['bash', '/proj/sot/ska/bin/starcheck'])


def get_starcheck_proseco_outputs(load_name, out_root='.'):
    """Get the complete set of output proseco pickles for each obsid and the corresponding
    set of observations parameters and backstop catalogs by parsing starcheck.txt.

    :param load_name: load name e.g. 'AUG2018T'
    :param out_root: directory containing the <load_name> directory (default='.')

    :returns: proseco_cats, starcheck_obss: dicts by obsid
    """
    load_dir = Path(out_root, load_name)
    starcheck_txt = load_dir / 'starcheck.txt'
    proseco_pickle = load_dir / f'{load_name}_proseco.pkl'

    # First get all the expected calling arguments for all obsids in starcheck output.
    # This returns a list of dict with get_aca_catalog calling args.  This also returns
    # the catalog as a Table in the 'cat' key.
    starcheck_obss = get_starcheck_obs_kwargs(starcheck_txt)

    # Now load the load products pickle with all catalogs
    proseco_cats = pickle.load(open(proseco_pickle, 'rb'))

    # Change obsid key from str to int (that's the way MATLAB stores them).
    for str_obsid in list(proseco_cats):
        proseco_cat = proseco_cats[str_obsid]
        proseco_cats[int(str_obsid)] = proseco_cat
        del proseco_cats[str_obsid]

    return proseco_cats, starcheck_obss
