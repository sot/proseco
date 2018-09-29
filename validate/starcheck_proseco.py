import subprocess
from pathlib import Path
import shutil

import numpy as np
import parse_cm
from mica.starcheck import get_starcat
from proseco.acq import get_acq_catalog
import Ska.File


def copy_load_products(load_name, out_root,
                       load_root='/data/mpcrit1/mplogs'):
    out_dir = Path(out_root) / load_name
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


def merge_cats(cat, pcat):
    """
    Merge the proseco acq star catalog ``pcat`` into the mica starcat
    catalog ``cat``.
    """
    cat = cat.copy()

    for star in cat:
        if star['type'] in ('BOT', 'GUI'):
            size = star['sz']
            break

    # First remove existing acq stars in `cat`
    for idx, star in enumerate(cat):
        if star['type'] == 'BOT':
            star['type'] = 'GUI'
            star['dim'] = 1
            star['halfw'] = 25

    cat = cat[cat['type'] != 'ACQ']

    cat['idx'] = np.arange(len(cat))

    # Find existing GUI stars for possible upgrade to BOT
    gui_idxs = {star['id']: idx for idx, star in enumerate(cat)
                if star['type'] == 'GUI'}

    # Keep track of available slots for an acq star
    acq_slots = list(range(8))

    for acq in pcat:
        if acq['id'] in gui_idxs:
            # In-place update of existing GUI row
            star = cat[gui_idxs[acq['id']]]
            star['type'] = 'BOT'
            star['dim'] = (acq['halfw'] - 20) / 5
            star['halfw'] = acq['halfw']
            star['pass'] = 'P'
            # Mark acq_slot as being used
            acq_slots.remove(star['slot'])

    for acq in pcat:
        if acq['id'] not in gui_idxs:
            # Add a row for ACQ star
            cat0 = cat[0]
            star = {name: cat0[name] for name in cat.colnames}
            star['id'] = acq['id']
            star['idx'] = len(cat)
            star['yang'] = acq['yang']
            star['zang'] = acq['zang']
            star['type'] = 'ACQ'
            star['dim'] = (acq['halfw'] - 20) / 5
            star['halfw'] = acq['halfw']
            star['pass'] = 'P'
            star['slot'] = acq_slots.pop(0)
            star['mag'] = acq['mag']
            star['minmag'] = max(acq['mag'] - 1.5, 5.8)
            star['maxmag'] = acq['mag'] + 1.5
            star['sz'] = size
            cat.add_row(star)

    # Finally sort
    cat['typ'] = 0
    for idx, typ in enumerate(('FID', 'BOT', 'GUI', 'ACQ', 'MON')):
        ok = cat['type'] == typ
        cat['typ'][ok] = idx
    cat.sort('typ')
    del cat['typ']
    cat['idx'] = np.arange(len(cat))

    return cat


def update_products(load_name, outroot):
    load_dir = Path(outroot) / load_name
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

    obsids = [summ['obsid'] for summ in gs['summs']]

    mcats = {}
    for obsid in obsids:
        print('Getting starcat for obsid {}'.format(obsid))
        cat = get_starcat(obsid)
        print('  Generating proseco catalog and merging')
        pcat = get_acq_catalog(obsid)
        mcats[obsid] = merge_cats(cat, pcat)

    print('Updating backstop, DOT and guide summary products')
    gs = parse_cm.replace_starcat_guide_summary(gs, mcats)
    bs = parse_cm.replace_starcat_backstop(bs, mcats)
    dt = parse_cm.replace_starcat_dot(dt, mcats)

    print('Writing ...')
    parse_cm.write_guide_summary(gs, gspath)
    parse_cm.write_backstop(bs, bspath)
    parse_cm.write_dot(dt, dotpath)

    touch_file.touch()


def run_starcheck(load_name, outroot):
    load_dir = Path(outroot) / load_name
    starcheck_txt = load_dir / 'starcheck.txt'
    if starcheck_txt.exists():
        print(f'Skipping because {starcheck_txt} already exists')
        return
    print(f'Running starcheck in {load_dir}')
    with Ska.File.chdir(str(load_dir)):
        subprocess.run(['bash', '/proj/sot/ska/bin/starcheck'])


def run_starcheck_proseco(load_name='JUL0218A', outroot='test_loads'):
    copy_load_products(load_name, outroot)
    update_products(load_name, outroot)
    run_starcheck(load_name, outroot)


def run_all(outroot='test_loads'):
    load_names = ['JUL0918A',  # dark cals
                  'JUL0218A',
                  'JUN2318A',
                  # 'JUN1818A',  ??
                  'JUN1118A',
                  'JUN0418A',
                  # 'MAY2818A',  no pred_temp for 20293
                  'MAY2118A',
                  'MAY1418A',
                  'MAY0718A',
                  'APR3018A',
                  'APR1618A',
                  'APR2318C']
    for load_name in load_names:
        run_starcheck_proseco(load_name)
