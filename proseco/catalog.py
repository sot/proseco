import traceback
import numpy as np
from itertools import count

from astropy.table import Column

from .core import ACACatalogTable
from .guide import get_guide_catalog, GuideTable
from .acq import get_acq_catalog, AcqTable
from .fid import get_fid_catalog, FidTable


def get_aca_catalog(obsid=0, **kwargs):
    """
    Get a catalog of guide stars, acquisition stars and fid lights.

    :param obsid: obsid (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param n_acq: desired number of acquisition stars (default=8)
    :param n_fid: desired number of fid lights (req'd unless obsid spec'd)
    :param n_guide: desired number of guide stars (req'd unless obsid spec'd)
    :param man_angle: maneuver angle (deg)
    :param t_ccd_acq: ACA CCD temperature for acquisition (degC)
    :param t_ccd_guide: ACA CCD temperature for guide (degC)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither_acq: acq dither size (2-element sequence (y, z), arcsec)
    :param dither_guide: guide dither size (2-element sequence (y, z), arcsec)
    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param sim_offset: SIM translation offset from nominal [steps] (default=0)
    :param focus_offset: SIM focus offset [steps] (default=0)
    :param stars: table of AGASC stars (will be fetched from agasc if None)
    :param include_ids: list of AGASC IDs of stars to include in selected catalog
    :param include_halfws: list of acq halfwidths corresponding to ``include_ids``
    :param exclude_ids: list of AGASC IDs of stars to exclude from selected catalog
    :param optimize: optimize star catalog after initial selection (default=True)
    :param verbose: provide extra logging info (mostly calc_p_safe) (default=False)
    :param print_log: print the run log to stdout (default=False)

    :returns: AcaCatalogTable of stars and fids

    """
    try:
        aca = _get_aca_catalog(obsid=obsid, **kwargs)

    except Exception:
        if kwargs.get('raise_exc'):
            # This is for debugging
            raise

        aca = ACACatalogTable.empty()  # Makes zero-length table with correct columns
        aca.exception = traceback.format_exc()

        if aca.acqs is None:
            aca.acqs = AcqTable.empty()
        if aca.fids is None:
            aca.fids = FidTable.empty()
        if aca.guides is None:
            aca.guides = GuideTable.empty()

    return aca


def get_kwargs(args, kwargs):
    out = {}
    for key in args:
        if isinstance(key, tuple):
            # key is (out_key, kwargs_key), translate input kwargs e.g. from
            # 'dither_acq' to 'dither' (which is needed by get_acq_catalog).
            if key[1] in kwargs:
                out[key[0]] = kwargs[key[1]]
        elif key in kwargs:
            out[key] = kwargs[key]
    return out


def _get_aca_catalog(**kwargs):
    raise_exc = kwargs.get('raise_exc')

    # Pluck off the kwargs that are relevant for get_acq_catalog
    args = ('obsid', 'att', 'date', 'man_angle',
            'include_ids', 'include_halfws', 'exclude_ids',
            'detector', 'sim_offset', 'sim_focus', 'stars',
            ('dither', 'dither_acq'), ('t_ccd', 't_ccd_acq'),
            'print_log', 'n_acq')
    acq_kwargs = get_kwargs(args, kwargs)

    # Pluck off the kwargs that are relevant for get_guide_catalog
    args = ('obsid', 'att', 'date', 'print_log', 'n_guide',
            ('dither', 'dither_guide'), ('t_ccd', 't_ccd_guide'))
    guide_kwargs = get_kwargs(args, kwargs)

    # Pluck off the kwargs that are relevant for get_fid_catalog
    args = ('n_fid',)
    fid_kwargs = get_kwargs(args, kwargs)

    aca = ACACatalogTable()

    # Put at least the obsid in top level meta for now
    aca.meta['obsid'] = kwargs.get('obsid')

    aca.acqs = get_acq_catalog(**acq_kwargs)
    aca.fids = get_fid_catalog(acqs=aca.acqs, **fid_kwargs)
    aca.guides = get_guide_catalog(stars=aca.acqs.meta['stars'], **guide_kwargs)

    # Get overall catalog thumbs_up
    aca.thumbs_up = aca.acqs.thumbs_up & aca.fids.thumbs_up & aca.guides.thumbs_up

    # Make a merged starcheck-like catalog.  Catch any errors at this point to avoid
    # impacting operational work (call from Matlab).
    try:
        merge_cat = merge_cats(fids=aca.fids, guides=aca.guides, acqs=aca.acqs)
        for name in merge_cat.colnames:
            aca[name] = merge_cat[name]
    except Exception:
        if raise_exc:
            raise
        aca['id'] = []  # Equivalent to ACACatalogTable.empty()
        aca.exception = traceback.format_exc()

    return aca


def merge_cats(fids=None, guides=None, acqs=None):

    fids = [] if fids is None else fids
    guides = [] if guides is None else guides
    acqs = [] if acqs is None else acqs

    colnames = ['slot', 'id', 'type', 'sz', 'p_acq', 'mag', 'maxmag',
                'yang', 'zang', 'dim', 'res', 'halfw']

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
        guides['slot'] = 0  # Filled in later
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

    # Accumulate a list of table Row objects to be assembled into the final table.
    # This has the desired side effect of back-populating 'slot' and 'type' columns
    # in the original acqs, guides, fids tables.
    rows = []

    # Generate a list of AGASC IDs for the BOT, GUI, and ACQ stars
    # in the merged catalog.  Use filt() function below to maintain
    # the original order (instead of undefined order of set).
    def filt(tbl, id_set):
        return [row['id'] for row in tbl if row['id'] in id_set]

    bot_ids = filt(guides, set(guides['id']) & set(acqs['id']))
    gui_ids = filt(guides, set(guides['id']) - set(acqs['id']))
    acq_ids = filt(acqs, set(acqs['id']) - set(guides['id']))

    n_fid = len(fids)
    n_bot = len(bot_ids)

    # FIDs.  Slot starts at 0.
    for slot, fid in zip(count(0), fids):
        fid['slot'] = slot
        rows.append(fid[colnames])

    # BOT stars.  Slot starts after fid slots.
    for slot, bot_id in zip(count(n_fid), bot_ids):
        acq = acqs.get_id(bot_id)
        guide = guides.get_id(bot_id)
        guide['slot'] = acq['slot'] = slot
        guide['type'] = acq['type'] = 'BOT'
        rows.append(acq[colnames])

    # GUI-only stars. Slot stars after fid and BOT slots.
    for slot, gui_id in zip(count(n_fid + n_bot), gui_ids):
        guide = guides.get_id(gui_id)
        guide['slot'] = slot
        rows.append(guide[colnames])

    # ACQ-only stars. Slot stars after fid and BOT slots.
    for slot, acq_id in zip(count(n_fid + n_bot), acq_ids):
        acq = acqs.get_id(acq_id)
        acq['slot'] = slot % 8
        rows.append(acq[colnames])

    # Create final table and assign idx
    aca = ACACatalogTable(rows=rows, names=colnames)
    idx = Column(np.arange(1, len(aca) + 1), name='idx')
    aca.add_column(idx, index=1)

    return aca
