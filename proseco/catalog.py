import traceback
import numpy as np

from .core import ACACatalogTable
from .guide import get_guide_catalog, GuideTable
from .acq import get_acq_catalog, AcqTable
from .fid import get_fid_catalog, FidTable


def get_aca_catalog(obsid=0, **kwargs):
    try:
        aca = _get_aca_catalog(obsid, **kwargs)

    except Exception:
        aca = ACACatalogTable.empty()  # Makes zero-length table with correct columns
        aca.meta['exception'] = traceback.format_exc()

        if aca.acqs is None:
            aca.acqs = AcqTable.empty()
        if aca.fids is None:
            aca.fids = FidTable.empty()
        if aca.guides is None:
            aca.guides = GuideTable.empty()

    return aca


def _get_aca_catalog(obsid=0, att=None, man_angle=None, date=None,
                     dither_acq=None, dither_guide=None,
                     t_ccd_acq=None, t_ccd_guide=None,
                     detector=None, sim_offset=None, focus_offset=None,
                     n_guide=None, n_fid=None, n_acq=None,
                     include_ids=None, include_halfws=None,
                     print_log=False):

    aca = ACACatalogTable()

    # Put at least the obsid in top level meta for now
    aca.meta['obsid'] = obsid

    aca.acqs = get_acq_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd_acq,
                               dither=np.max(dither_acq), man_angle=man_angle,
                               print_log=print_log)

    aca.fids = get_fid_catalog(detector=detector, sim_offset=sim_offset,
                               focus_offset=focus_offset, acqs=aca.acqs,
                               stars=aca.acqs.meta['stars'], print_log=print_log)

    aca.guides = get_guide_catalog(obsid=obsid, att=att, date=date, t_ccd=t_ccd_guide,
                                   n_guide=n_guide, dither=dither_guide,
                                   stars=aca.acqs.meta['stars'], print_log=print_log)

    # Make a merged starcheck-like catalog.  Catch any errors at this point to avoid
    # impacting operational work (call from Matlab).
    try:
        merge_cat = merge_cats(fids=aca.fids, guides=aca.guides, acqs=aca.acqs)
        for name in merge_cat.colnames:
            aca[name] = merge_cat[name]
    except Exception:
        aca['id'] = []  # Equivalent to ACACatalogTable.empty()
        aca.meta['exception'] = traceback.format_exc()

    return aca


def merge_cats(fids=None, guides=None, acqs=None):

    fids = [] if fids is None else fids
    guides = [] if guides is None else guides
    acqs = [] if acqs is None else acqs

    colnames = ['id', 'type', 'sz', 'p_acq', 'mag', 'maxmag', 'yang', 'zang', 'dim', 'res', 'halfw']

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
        fid = fid[colnames]
        entries.append(dict(zip(fid.colnames, fid.as_void())))

    for bot_id in set(acqs['id']) & set(guides['id']):
        bot = acqs[colnames][acqs['id'] == bot_id][0]
        bot['type'] = 'BOT'
        entries.append(dict(zip(bot.colnames, bot.as_void())))

    for gui_id in set(guides['id']) - set(acqs['id']):
        guide = guides[colnames][guides['id'] == gui_id][0]
        entries.append(dict(zip(guide.colnames, guide.as_void())))

    for acq_id in set(acqs['id']) - set(guides['id']):
        acq = acqs[colnames][acqs['id'] == acq_id][0]
        entries.append(dict(zip(acq.colnames, acq.as_void())))

    # Initialize table and assign idx
    table = ACACatalogTable(entries)
    table['idx'] = np.arange(1, len(entries) + 1)

    # Assign slots
    n_fid = len(fids)
    n_bot = np.count_nonzero(table['type'] == 'BOT')
    n_gui = np.count_nonzero(table['type'] == 'GUI')
    n_acq = np.count_nonzero(table['type'] == 'ACQ')

    fidnums = np.arange(0, n_fid)
    botnums = np.arange(0, n_bot) + n_fid
    acqnums = np.arange(0, n_acq) + n_fid + n_bot
    guinums = np.arange(0, n_gui) + n_fid + n_bot
    nums = np.concatenate([fidnums, botnums, guinums, acqnums])

    table['slot'] = nums % 8

    return table[['idx', 'slot'] + colnames]
