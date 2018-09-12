import traceback
import numpy as np

from .core import ACACatalogTable
from .guide import get_guide_catalog, GuideTable
from .acq import get_acq_catalog, AcqTable
from .fid import get_fid_catalog, FidTable


def get_aca_catalog(**kwargs):
    try:
        aca = _get_aca_catalog(**kwargs)
    except Exception:
        #aca = ACACatalogTable.empty()  # TBD Makes zero-length table with correct columns
        aca = ACACatalogTable()
        aca.meta['exception'] = traceback.format_exc()

        if aca.acqs is None:
            #aca.acqs = AcqTable.empty()
            aca.acqs = []
        if aca.fids is None:
            #aca.fids = FidTable.empty()
            aca.fids = []
        if aca.guides is None:
            #aca.guides = GuideTable.empty()
            aca.guides = []
    return aca


def _get_aca_catalog(obsid=0, att=None, man_angle=None, date=None,
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

    # I don't want an issue just with making the pretty catalog to be a possible problem
    # for operational work, so catching any errors on this separately
    try:
        merge_cat = merge_cats(fids=catalog.fids, guides=catalog.guides, acqs=catalog.acqs,
                               mons=monitor_windows)
        for col in merge_cat.colnames:
            catalog[col] = merge_cat[col]
    except:
        catalog['dummy_col'] = [0]

    return catalog


def merge_cats(fids=None, guides=None, acqs=None, mons=None):

    fids = [] if fids is None else fids
    guides = [] if guides is None else guides
    acqs = [] if acqs is None else acqs
    mons = [] if mons is None else mons

    wantcols = ['id', 'type', 'sz', 'p_acq', 'mag', 'maxmag', 'yang', 'zang', 'dim', 'res', 'halfw']
    if len(fids) > 0:
        fids['type'] = 'FID'
        fids['mag'] = 7.0
        fids['maxmag'] = 8.0
        fids['dim'] = 1
        fids['res'] = 1
        fids['halfw'] = 25
        fids['p_acq'] = 0
        fids['sz'] = '8x8'
    guide_size = '8x8' if len(fids) == 0 else '6x6'
    if len(guides) > 0:
        guides['type'] = 'GUI'
        guides['maxmag'] = guides['mag'] + 1.5
        guides['p_acq'] = 0
        guides['dim'] = 1
        guides['res'] = 1
        guides['halfw'] = 25
        guides['sz'] = guide_size
    if len(acqs) > 0:
        acqs['type'] = 'ACQ'
        acqs['maxmag'] = acqs['mag'] + 1.5
        acqs['dim'] = 20
        acqs['sz'] = guide_size
        acqs['res'] = 1

    # Start with just a list of dicts because all fancy table manip is breaking for me
    entries = []
    for fid in fids:
        fid = fid[wantcols]
        entries.append(dict(zip(fid.colnames, fid.as_void())))
    for bot_id in set(acqs['id']) & set(guides['id']):
        bot = acqs[wantcols][acqs['id'] == bot_id][0]
        bot['type'] = 'BOT'
        entries.append(dict(zip(bot.colnames, bot.as_void())))
    for gui_id in set(guides['id']) - set(acqs['id']):
        guide = guides[wantcols][guides['id'] == gui_id][0]
        entries.append(dict(zip(guide.colnames, guide.as_void())))
    for mon in mons:
        # Haven't really decided on MON input syntax (table or list of dicts or tuples)
        # if we even want mons in here. So putting these at 0, 0 to start
        entries.append({'type': 'MON', 'id': '', 'sz': '8x8', 'mag': 0, 'maxmag': 0,
                        'p_acq': 0, 'yang': 0, 'zang': 0, 'dim': -1,
                        'res': 0, 'halfw': 20})
    for acq_id in (set(acqs['id']) - set(guides['id'])):
        acq = acqs[wantcols][acqs['id'] == acq_id][0]
        entries.append(dict(zip(acq.colnames, acq.as_void())))

    # Initialize table and assign idx
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
    return_cols = ['idx', 'slot'] + wantcols
    return table[return_cols]
