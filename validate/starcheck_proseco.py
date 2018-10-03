from pathlib import Path
import subprocess
import shutil

import parse_cm
from proseco import get_aca_catalog
from proseco.core import get_kwargs_from_starcheck_text
import Ska.File


def copy_load_products(load_name, out_root, load_root=None):
    # Default location of loads on HEAD network
    if load_root is None:
        load_root = '/data/mpcrit1/mplogs'

    out_dir = Path(out_root, load_name)
    if out_dir.exists():
        print(f'Skipping copy_load_products {out_dir} already exists')
        return

    load_root = Path(load_root)
    load_version = load_name[-1:].lower()
    load_year = '20' + load_name[-3:-1]

    load_dir = load_root / load_year / load_name[:-1] / ('ofls' + load_version)
    print(f'Copying load products from {load_dir} to {out_dir}')

    globs = ('CR*.tlr',
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
             'History/SIMTRANS.txt')

    for glob in globs:
        for src in load_dir.glob(glob):
            dest = out_dir / src.relative_to(load_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def get_starcheck_obs_kwargs(filename):
    """
    Parse the starcheck.txt file to get keyword arg dicts for get_aca_catalog()

    :param filename: file name of starcheck.txt in load products
    :returns: dict (by obsid) of kwargs for get_aca_catalog()

    """
    delim = "==================================================================================== "
    with open(filename, 'r') as fh:
        text = fh.read()
    chunks = text.split(delim)
    outs = {}
    for chunk in chunks:
        if "No star catalog for obsid" in chunk:
            continue
        try:
            out = get_kwargs_from_starcheck_text(chunk)
        except ValueError:
            continue
        else:
            outs[out['obsid']] = out

    return outs


def update_products(load_name, out_root):
    load_dir = Path(out_root) / load_name
    touch_file = load_dir / '000-proseco'
    if touch_file.exists():
        print('Skipping update products, already done')
        return

    gspath = list(load_dir.glob('mps/mg*.sum'))[0]
    bspath = list(load_dir.glob('CR*.backstop'))[0]
    dotpath = list(load_dir.glob('mps/md*.dot'))[0]

    obs_kwargs = get_starcheck_obs_kwargs(load_dir / 'starcheck.txt')

    gs = parse_cm.read_guide_summary(gspath)
    bs = parse_cm.read_backstop(bspath)
    dt = parse_cm.read_dot_as_list(dotpath)

    cats = {}
    for obsid, kwargs in obs_kwargs.items():
        print(f'  Generating proseco catalog for {obsid}')
        cats[obsid] = get_aca_catalog(raise_exc=True, **kwargs)

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
    starcheck_txt = load_dir / 'starcheck.txt'
    if starcheck_txt.exists():
        print(f'Skipping because {starcheck_txt} already exists')
        return
    print(f'Running starcheck in {load_dir}')
    with Ska.File.chdir(str(load_dir)):
        subprocess.run(['bash', '/proj/sot/ska/bin/starcheck'])


def run_starcheck_proseco(load_name='JUL0218A',
                          out_root=Path('loads', 'proseco'),
                          load_root=None):
    copy_load_products(load_name, out_root, load_root)
    update_products(load_name, out_root)
    run_starcheck(load_name, out_root)


def run_all(out_root='loads'):
    load_names = ['JUL0918A',  # dark cals
                  'JUL0218A',
                  'JUN2318A',
                  'JUN1118A',
                  'JUN0418A',
                  'MAY2118A',
                  'MAY1418A',
                  'MAY0718A',
                  'APR3018A',
                  'APR1618A',
                  'APR2318C']
    for load_name in load_names:
        run_starcheck_proseco(load_name)
