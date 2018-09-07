import numpy as np

from .core import ACACatalogTable
from .guide import get_guide_catalog
from .acq import get_acq_catalog
from .fid import get_fid_catalog


def get_aca_catalog(obsid=0, att=None, man_angle=None, date=None,
                    dither_acq=None, dither_guide=None,
                    t_ccd_acq=None, t_ccd_guide=None,
                    detector=None, sim_offset=None, focus_offset=None,
                    n_guide=None, n_fid=None, n_acq=None, monitor_windows=None,
                    include_ids=None, include_halfws=None,
                    print_log=False):
    catalog = ACACatalogTable(print_log=print_log)

    # Put at least the obsid in top level meta for now
    catalog.meta['obsid'] = obsid

    catalog.acqs = get_acq_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd_acq,
                                   dither=np.max(dither_acq), man_angle=man_angle,
                                   print_log=print_log)
    catalog.fids = get_fid_catalog(detector=detector, sim_offset=sim_offset,
                                   focus_offset=focus_offset, acqs=catalog.acqs,
                                   stars=catalog.acqs.meta['stars'], print_log=print_log)
    catalog.guides = get_guide_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd_guide,
                                       n_guide=n_guide, dither=dither_guide,
                                       stars=catalog.acqs.meta['stars'], print_log=print_log)

    merge_cat = merge_cats(fids=catalog.fids, guides=catalog.guides, acqs=catalog.acqs,
                           mons=monitor_windows)
    for col in merge_cat.colnames:
        catalog[col] = merge_cat[col]

    return catalog


def merge_cats(fids=[], guides=[], acqs=[], mons=[]):

    # Start with just a list of entries and ids
    entries = []
    for fid in fids:
        entries.append({'type': 'FID', 'id': fid['id']})
    for bot_id in set(acqs['id']).intersection(guides['id']):
        entries.append({'type': 'BOT', 'id': bot_id})
    for gui_id in set(guides['id']) - set(acqs['id']):
        entries.append({'type': 'GUI', 'id': gui_id})
    for mon in mons:
        entries.append({'type': 'MON', 'id': ''})
    for acq_id in set(acqs['id']) - set(guides['id']):
        entries.append({'type': 'ACQ', 'id': acq_id})

    # Assign index
    table = ACACatalogTable(entries)
    table['idx'] = np.arange(1, len(entries) + 1)

    # Assign slots
    n_fid = len(fids)
    n_bot = np.count_nonzero(table['type'] == 'BOT')
    n_gui = np.count_nonzero(table['type'] == 'GUI')
    n_acq = np.count_nonzero(table['type'] == 'ACQ')
    n_mon = np.count_nonzero(table['type'] == 'MON')

    fidnums = np.arange(0, n_fid)
    botnums = np.arange(0, n_bot) + n_fid
    acqnums = np.arange(0, n_acq) + n_fid + n_bot
    guinums = np.arange(0, n_gui) + n_fid + n_bot
    monnums = np.arange(0, n_mon) + n_fid + n_bot + n_gui
    nums = np.concatenate([fidnums, botnums, guinums, monnums, acqnums])

    table['slot'] = nums % 8

    return table
