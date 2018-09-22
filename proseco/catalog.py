import traceback
import numpy as np
from itertools import count

from astropy.table import Column, Table

from .core import ACACatalogTable, get_kwargs_from_starcheck_text, MetaAttribute
from .guide import get_guide_catalog, GuideTable
from .acq import get_acq_catalog, AcqTable
from .fid import get_fid_catalog, FidTable
from . import characteristics_fid as FID
from . import characteristics as ACQ


def get_aca_catalog(obsid=0, **kwargs):
    """
    Get a catalog of guide stars, acquisition stars and fid lights.

    If ``obsid`` is supplied and is a string, then it is taken to be starcheck text
    with required info.  User-supplied kwargs take precedence, however (e.g.
    one can override the dither from starcheck).

    :param obsid: obsid (int) or starcheck text (str) (default=0)
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
        # If obsid is supplied as a string then it is taken to be starcheck text
        # with required info.  User-supplied kwargs take precedence, however.
        if isinstance(obsid, str):
            kw = get_kwargs_from_starcheck_text(obsid)
            obsid = kw.pop('obsid')
            for key, val in kw.items():
                if key not in kwargs:
                    kwargs[key] = val

        raise_exc = kwargs.pop('raise_exc', None)

        aca = _get_aca_catalog(obsid=obsid, raise_exc=raise_exc, **kwargs)

    except Exception:
        if raise_exc:
            # This is for debugging
            raise

        aca = ACATable.empty()  # Makes zero-length table with correct columns
        aca.exception = traceback.format_exc()

    return aca


def _get_aca_catalog(**kwargs):
    raise_exc = kwargs.pop('raise_exc')

    aca = ACATable()
    aca.set_attrs_from_kwargs(**kwargs)

    aca.acqs = get_acq_catalog(**kwargs)
    aca.fids = get_fid_catalog(acqs=aca.acqs, **kwargs)

    aca.acqs.fids = aca.fids

    if len(aca.fids) == 0 and aca.optimize:
        optimize_acqs_fids(aca.acqs, aca.fids)

    stars = kwargs.pop('stars', aca.acqs.stars)
    aca.guides = get_guide_catalog(stars=stars, **kwargs)

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

        empty = ACACatalogTable.empty()
        for name in empty.colnames:
            aca[name] = empty[name]

        aca.exception = traceback.format_exc()

    return aca


class ACATable(ACACatalogTable):
    optimize = MetaAttribute(default=True)

    @classmethod
    def empty(cls):
        out = super().empty()
        out.acqs = AcqTable.empty()
        out.fids = FidTable.empty()
        out.guides = GuideTable.empty()
        return out


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


def optimize_acqs_fids(acqs, fids):
    """
    Concurrently optimize acqs and fids in the case where there is not
    already a good (no spoilers) fid set available.

    This updates the tables in place.

    :param acqs: AcqTable object
    :param fids: FidTable object
    """
    opt_P2 = -acqs.get_log_p_2_or_fewer()
    orig_ids = acqs['id'].tolist()
    orig_halfws = acqs['halfw'].tolist()

    cand_fids = fids.cand_fids
    if len(cand_fids) < 2:
        acqs.log('Fewer than 2 candidate fid lights, skipping optimization',
                 warning=True)
        return

    cand_fids_ids = set(cand_fids['id'])

    # Get list of fid_sets that are consistent with candidate fids. These
    # fid sets are the combinations of 3 (or 2) fid lights in preferred
    # order.
    rows = []
    for fid_set in FID.fid_sets[fids.detector]:
        if fid_set <= cand_fids_ids:
            spoiler_score = sum(cand_fids.get_id(fid_id)['spoiler_score']
                                for fid_id in fid_set)
            rows.append((fid_set, spoiler_score))

    fid_sets = Table(rows=rows, names=('fid_ids', 'spoiler_score'))
    fid_sets['P2'] = -1.0  # Marker for unfilled values
    fid_sets['halfws'] = None
    fid_sets['ids'] = None
    fid_sets = fid_sets.group_by('spoiler_score')

    for fid_set_group in fid_sets.groups:
        spoiler_score = fid_set_group['spoiler_score'][0]

        for fid_set in fid_set_group:
            acqs.fid_set = fid_set['fid_ids']
            acqs.optimize_catalog()
            acqs.update_p_acq_column()
            fid_set['P2'] = -acqs.get_log_p_2_or_fewer()
            fid_set['halfws'] = acqs['halfw'].tolist()
            fid_set['ids'] = acqs['id'].tolist()
            acqs.update_ids_halfws(orig_ids, orig_halfws)

        idx = np.argmax(fid_sets['P2'])
        best_P2 = fid_sets['P2'][idx]
        stage = ACQ.fid_acq_stages.loc[spoiler_score]
        stage_min_P2 = stage['min_P2'](opt_P2)

        if best_P2 >= stage_min_P2:
            best_ids = fid_sets['ids'][idx]
            best_halfws = fid_sets['halfws'][idx]
            acqs.update_ids_halfws(best_ids, best_halfws)
            acqs.fid_set = fid_sets['fid_ids'][idx]
            # Update fids also!
            return

    # This should never happen and indicates a flaw in program logic
    raise RuntimeError('optimize_acqs_fids failed to find a fid set')
