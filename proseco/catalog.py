# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from proseco.characteristics import MonFunc
import traceback
import numpy as np
import copy

from astropy.table import Table

from .core import (ACACatalogTable, get_kwargs_from_starcheck_text, MetaAttribute,
                   get_dim_res, get_img_size)
from .guide import BadMonitorError, get_guide_catalog, GuideTable
from .acq import get_acq_catalog, AcqTable
from .fid import get_fid_catalog, FidTable
from . import characteristics_acq as ACQ
from . import characteristics as ACA
from . import __version__ as VERSION


def get_aca_catalog(obsid=0, **kwargs):
    """Get a catalog of guide stars, acquisition stars and fid lights.

    If ``obsid`` is supplied and is a string, then it is taken to be starcheck
    text with required info.  User-supplied kwargs take precedence, however
    (e.g.  one can override the dither from starcheck).

    In this situation if the ``obsid`` text includes the string
    ``--force-catalog`` anywhere then the final proseco guide and acq catalogs
    will be forced to match the input starcheck catalog.  This can be done by
    appending this string, e.g. with ``obs_text + '--force-catalog'`` in the
    call to ``get_aca_catalog``.

    NOTE on API:

    Keywords that have ``_acq`` and/or ``_guide`` suffixes are handled with
    the AliasAttribute in core.py.  If one calls get_aca_catalog() with e.g.
    ``t_ccd=-10`` then that will set the CCD temperature for both acq and
    guide selection.  This is not part of the public API but is a private
    feature of the implementation that works for now.

    :param obsid: obsid (int) or starcheck text (str) (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param n_acq: desired number of acquisition stars (default=8)
    :param n_fid: desired number of fid lights (req'd unless obsid spec'd)
    :param n_guide: desired number of guide stars (req'd unless obsid spec'd)
    :param monitors: N x 5 float array specifying monitor windows
    :param man_angle: maneuver angle (deg)
    :param t_ccd_acq: ACA CCD temperature for acquisition (degC)
    :param t_ccd_guide: ACA CCD temperature for guide (degC)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither_acq: acq dither size (2-element sequence (y, z), arcsec)
    :param dither_guide: guide dither size (2-element sequence (y, z), arcsec)
    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param sim_offset: SIM translation offset from nominal [steps] (default=0)
    :param focus_offset: SIM focus offset [steps] (default=0)
    :param target_offset: (y, z) target offset including dynamical offset
                          (2-element sequence (y, z), deg)
    :param stars: table of AGASC stars (will be fetched from agasc if None)
    :param include_ids_acq: list of AGASC IDs of stars to include in acq catalog
    :param include_halfws_acq: list of acq halfwidths corresponding to ``include_ids``.
                               For values of ``0`` proseco chooses the best halfwidth(s).
    :param exclude_ids_acq: list of AGASC IDs of stars to exclude from acq catalog
    :param include_ids_fid: list of fiducial lights to include by index.  If no possible
                            sets of fids include the id, no fids will be selected.
    :param exclude_ids_fid: list of fiducial lights to exclude by index
    :param include_ids_guide: list of AGASC IDs of stars to include in guide catalog
    :param exclude_ids_guide: list of AGASC IDs of stars to exclude from guide catalog
    :param img_size_guide: readout window size for guide stars (6, 8, or ``None``).
                           For default value ``None`` use 8 for no fids, 6 for fids.
    :param optimize: optimize star catalog after initial selection (default=True)
    :param verbose: provide extra logging info (mostly calc_p_safe) (default=False)
    :param print_log: print the run log to stdout (default=False)
    :param raise_exc: raise exception if it occurs in processing (default=True)

    :returns: ACATable of stars and fids

    """
    raise_exc = kwargs.pop('raise_exc', True)  # This cannot credibly fail

    try:
        # If obsid is supplied as a string then it is taken to be starcheck text
        # with required info.  User-supplied kwargs take precedence, however.
        if isinstance(obsid, str):
            force_catalog = '--force-catalog' in obsid
            kw = get_kwargs_from_starcheck_text(obsid, force_catalog=force_catalog)
            obsid = kw.pop('obsid')
            for key, val in kw.items():
                if key not in kwargs:
                    kwargs[key] = val

        if 'monitors' in kwargs:
            aca = _get_aca_catalog_monitors(obsid=obsid, raise_exc=raise_exc, **kwargs)
        else:
            aca = _get_aca_catalog(obsid=obsid, raise_exc=raise_exc, **kwargs)

    except Exception:
        if raise_exc:
            # This is for debugging
            raise

        aca = ACATable.empty()  # Makes zero-length table with correct columns
        aca.exception = traceback.format_exc()

    return aca


def _get_aca_catalog(**kwargs):
    """
    Do the actual work of getting the acq, fid and guide catalogs and
    assembling the merged aca catalog.
    """
    raise_exc = kwargs.pop('raise_exc')
    img_size_guide = kwargs.pop('img_size_guide', None)

    aca = ACATable()
    aca.call_args = kwargs.copy()
    aca.set_attrs_from_kwargs(**kwargs)

    # Override t_ccd related inputs with effective temperatures for downstream
    # action by AcqTable, GuideTable, FidTable.  See set_attrs_from_kwargs()
    # method for more details.
    if 't_ccd' in kwargs:
        del kwargs['t_ccd']
    kwargs['t_ccd_acq'] = aca.t_ccd_eff_acq
    kwargs['t_ccd_guide'] = aca.t_ccd_eff_guide

    # Get stars (typically from AGASC) and do not filter for stars near
    # the ACA FOV.  This leaves the full radial selection available for
    # later roll optimization.  Use aca.stars or aca.acqs.stars from here.
    aca.set_stars(filter_near_fov=False)
    kwargs.pop('stars', None)

    aca.log('Starting get_acq_catalog')
    aca.acqs = get_acq_catalog(stars=aca.stars, **kwargs)

    # Store the date of the dark cal if available.
    if hasattr(aca.acqs, 'dark_date'):
        aca.dark_date = aca.acqs.dark_date

    # Note that aca.acqs.stars is a filtered version of aca.stars and includes
    # only stars that are in or near ACA FOV.  Use this for fids and guides stars.
    aca.log('Starting get_fid_catalog')
    aca.fids = get_fid_catalog(stars=aca.acqs.stars, acqs=aca.acqs, **kwargs)
    aca.acqs.fids = aca.fids

    if aca.optimize:
        aca.log('Starting optimize_acqs_fids')
        aca.optimize_acqs_fids()

    aca.acqs.fid_set = aca.fids['id']

    aca.log('Starting get_guide_catalog')
    aca.guides = get_guide_catalog(stars=aca.acqs.stars, fids=aca.fids,
                                   img_size=img_size_guide, **kwargs)

    # Make a merged starcheck-like catalog.  Catch any errors at this point to avoid
    # impacting operational work (call from Matlab).
    try:
        aca.log('Starting merge_cats')
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

    aca.log('Finished aca_get_catalog')
    return aca


def _get_aca_catalog_monitors(**kwargs):
    """Get the ACA catalog for a case with possible monitor windows / stars.

    If there are only pure MON windows or Guide-From-Mon windows (no possible
    auto-conversion to GUI) then just run the standard _get_aca_catalog, which
    handles these cases.

    Otherwise, first get the catalog forcing tracking MON window commanding.
    This will be short a guide star for each MON window. Second get the catalog
    forcing MON conversion to GUI. This requires that an acceptable guide star
    is within 2 arcsec of the MON position. "Acceptable" means that ``guides``
    has a candidate (``cand_guides``) guide star within 2 arcsec. If the catalog
    with the converted GUI star has no new critical warnings then it is
    accepted.
    """
    kwargs_orig = kwargs.copy()
    kwargs.pop('raise_exc')

    # Make a stub aca to get monitors attribute
    aca = ACATable()
    aca.set_attrs_from_kwargs(**kwargs)

    # If no auto-convert mon stars then do the normal star selection.
    monitors = aca.monitors
    if np.all(monitors['function'] != MonFunc.AUTO):
        return _get_aca_catalog(**kwargs_orig)

    # Find the entries with auto-convert
    is_auto = monitors['function'] == MonFunc.AUTO  # Auto-convert to guide

    # First get the catalog with automon entries scheduled as tracking MON windows.
    kwargs = kwargs_orig.copy()
    monitors['function'][is_auto] = MonFunc.MON_TRACK  # Tracking MON window
    kwargs['monitors'] = monitors

    # Get the catalog and do a sparkles review
    aca_mon = _get_aca_catalog(**kwargs)
    acar_mon = aca_mon.get_review_table()
    acar_mon.run_aca_review()
    crits_mon = set(msg['text'] for msg in (acar_mon.messages >= 'critical'))

    # Now get the catalog with automon entries scheduled as guide stars
    monitors['function'][is_auto] = MonFunc.GUIDE
    kwargs['raise_exc'] = True
    try:
        aca_gui = _get_aca_catalog(**kwargs)
    except BadMonitorError as exc:
        aca_mon.log(f'unable to convert monitor to guide: {exc}')
        return aca_mon

    # Get the catalog and do a sparkles review
    acar_gui = aca_gui.get_review_table()
    acar_gui.run_aca_review()
    crits_gui = set(msg['text'] for msg in (acar_gui.messages >= 'critical'))

    # If there are no new critical messages then schedule as guide star(s).
    # This checks that every critical in crit_gui is also in crits_mon.
    out = aca_gui if crits_gui <= crits_mon else aca_mon
    return out


def get_effective_t_ccd(t_ccd):
    """Return the effective T_ccd used for selection and catalog evaluation.

    For details see Dynamic ACA limits in baby steps section in:
    https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/StarWorkingGroupMeeting2019x02x13

    :param t_ccd:
    :return:
    """
    t_limit = ACA.aca_t_ccd_penalty_limit
    if t_ccd > t_limit:
        return t_ccd + 1 + (t_ccd - t_limit)
    else:
        return t_ccd


class ACATable(ACACatalogTable):
    """Container ACACatalogTable class that has merged guide / acq / fid catalogs
    as attributes and other methods relevant to the merged catalog.

    """
    # Define base set of allowed keyword args to __init__. Subsequent MetaAttribute
    # or AliasAttribute properties will add to this.
    allowed_kwargs = ACACatalogTable.allowed_kwargs.copy()

    optimize = MetaAttribute(default=True)
    call_args = MetaAttribute(default={})
    version = MetaAttribute()

    # For validation with get_aca_catalog(obsid), store the starcheck
    # catalog in the ACATable meta.
    starcheck_catalog = MetaAttribute(is_kwarg=False)

    # Effective T_ccd used for dynamic ACA limits (see updates_for_t_ccd_effective()
    # method below).
    t_ccd_eff_acq = MetaAttribute(is_kwarg=False)
    t_ccd_eff_guide = MetaAttribute(is_kwarg=False)

    def __copy__(self):
        # Astropy Table now does a light key-only copy of the `meta` dict, so
        # copy.copy(aca) does not copy the underlying table attributes.  Force
        # a deepcopy instead.
        return copy.deepcopy(self)

    @property
    def t_ccd(self):
        # For top-level ACATable object use the guide temperature, which is always
        # greater than or equal to the acq temperature.
        return self.t_ccd_guide

    @t_ccd.setter
    def t_ccd(self, value):
        self.t_ccd_guide = value
        self.t_ccd_acq = value

    @classmethod
    def empty(cls):
        out = super().empty()
        out.acqs = AcqTable.empty()
        out.fids = FidTable.empty()
        out.guides = GuideTable.empty()
        return out

    def set_attrs_from_kwargs(self, **kwargs):
        """Set object attributes from kwargs.

        After calling the base class method which does all the real work, then
        compute the effective T_ccd temperatures.

        In this ACATable object:

        - t_ccd_eff_{acq,guide} are the effective T_ccd values which are adjusted
          if the actual t_ccd{acq,guide} values are above ACA.aca_t_ccd_penalty_limit.
        - t_ccd_{acq,guide} are the actual (or predicted) values from the call

        The downstream AcqTable, GuideTable, and FidTable are initialized with the
        *effective* values as t_ccd.  Those classes do not have the concept of effective
        temperature.

        :param kwargs: dict of input kwargs
        :return: dict
        """
        super().set_attrs_from_kwargs(**kwargs)

        self.t_ccd_eff_acq = get_effective_t_ccd(self.t_ccd_acq)
        self.t_ccd_eff_guide = get_effective_t_ccd(self.t_ccd_guide)
        self.version = VERSION

    def get_review_table(self):
        """Get ACAReviewTable object based on self.

        :returns: ACAReviewTable object
        """
        from sparkles.core import ACAReviewTable

        return ACAReviewTable(self)

    def get_candidates_mask(self, stars):
        """Return a boolean mask indicating which ``stars`` are acceptable candidates
        for the parent class.

        For ACATable this is the logical OR of the guide and acq values,
        i.e. for ACA a candidate is shown as OK if it is OK for either guide or
        acq.  This is just used for plotting.

        :param stars: StarsTable of stars
        :returns: bool mask

        """
        ok = (self.acqs.get_candidates_mask(stars) |
              self.guides.get_candidates_mask(stars))
        return ok

    def make_report(self, rootdir='.'):
        """
        Make summary HTML report for acq and guide selection process and outputs.

        Outputs are in ``<rootdir>/obs<obsid>/{acq,guide}/index.html`` plus related
        images in that directory.

        :param rootdir: root directory for outputs

        """
        self.acqs.make_report(rootdir=rootdir)
        self.guides.make_report(rootdir=rootdir)

    def optimize_acqs_fids(self):
        """
        Concurrently optimize acqs and fids in the case where there is not
        already a good (no spoilers) fid set available.

        This updates the acqs and fids tables in place.

        :param acqs: AcqTable object
        :param fids: FidTable object
        """
        acqs = self.acqs
        fids = self.fids

        # IF get_fid_catalog returned a good catalog,
        #    OR no fids were requested,
        #    OR no candidate fids are available,
        #    OR no candidate fid sets are available
        # THEN no optimization action required here.
        if (len(self.fids) > 0 or self.n_fid == 0 or len(self.fids.cand_fids) == 0 or
                len(self.fids.cand_fid_sets) == 0):
            return

        # Start with the no-fids optimum catalog and save required info to restore
        opt_P2 = -acqs.get_log_p_2_or_fewer()
        orig_acq_idxs = acqs['idx'].tolist()
        orig_acq_halfws = acqs['halfw'].tolist()

        self.log(f'Starting opt_P2={opt_P2:.2f}: ids={orig_acq_idxs} halfws={orig_acq_halfws}')

        # If not at least 2 fids then punt on optimization.
        cand_fids = fids.cand_fids

        # Get list of fid_sets that are consistent with candidate fids. These
        # fid sets are the combinations of 3 (or 2) fid lights in preferred
        # order.
        rows = []
        for fid_set in fids.cand_fid_sets:
            spoiler_score = sum(cand_fids.get_id(fid_id)['spoiler_score']
                                for fid_id in fid_set)
            rows.append((fid_set, spoiler_score))

        # Make a table to keep track of candidate fid_sets along with the
        # ranking metric P2 and the acq catalog info halfws and star ids.
        fid_sets = Table(rows=rows, names=('fid_ids', 'spoiler_score'))
        fid_sets['P2'] = -99.0  # Marker for unfilled values
        fid_sets['acq_halfws'] = None
        fid_sets['acq_idxs'] = None

        # Group the table into groups by spoiler score.  This preserves the
        # original fid set ordering within a group.
        fid_sets = fid_sets.group_by('spoiler_score')

        # Iterate through each spoiler_score group and then within that iterate
        # over each fid set.
        for fid_set_group in fid_sets.groups:
            spoiler_score = fid_set_group['spoiler_score'][0]
            self.log(f'Checking fid sets with spoiler_score={spoiler_score}', level=1)

            for fid_set in fid_set_group:
                # Set the internal acqs fid set.  This does validation of the set
                # and also calls update_p_acq_column().
                acqs.fid_set = fid_set['fid_ids']

                # If P2 is effectively unchanged after updating the fid set,
                # that means there are no fids spoiling an acq star in the
                # optimum (no-fid) acq catalog, and there is no need to
                # re-optimize.  Furthermore this fid set is guaranteed to meet
                # the stage_min_P2 acceptance so no need to check any other fid
                # sets.  Only optimize if P2 decreased by at least 0.001,
                # remembering P2 is the -log10(prob).
                fid_set_P2 = -acqs.get_log_p_2_or_fewer()
                found_good_set = fid_set_P2 - opt_P2 > -0.001
                if found_good_set:
                    self.log(f'No change in P2 for fid set {acqs.fid_set}, '
                             f'skipping optimization')
                else:
                    # Re-optimize the catalog with the fid set selected and get new probs.
                    acqs.optimize_catalog()
                    acqs.update_p_acq_column(acqs)  # Needed for get_log_p_2_or_fewer

                # Store optimization results
                fid_set['P2'] = -acqs.get_log_p_2_or_fewer()
                fid_set['acq_idxs'] = acqs['idx'].tolist()
                fid_set['acq_halfws'] = acqs['halfw'].tolist()

                self.log(f"Fid set {fid_set['fid_ids']}: P2={fid_set['P2']:.2f} "
                         f"acq_idxs={fid_set['acq_idxs']} halfws={fid_set['acq_halfws']}",
                         level=2)

                if found_good_set:
                    break
                else:
                    # Put the catalog back to the original no-fid optimum and continue
                    # trying remaining fid sets.
                    acqs.update_idxs_halfws(orig_acq_idxs, orig_acq_halfws)

            # Get the best fid set / acq catalog configuration so far.  Fid sets not
            # yet considered have P2 = -99.
            best_idx = np.argmax(fid_sets['P2'])
            best_P2 = fid_sets['P2'][best_idx]

            # Get the row of the fid / acq stages table to determine the required minimum
            # P2 given the fid spoiler score.
            stage = ACQ.fid_acq_stages.loc[spoiler_score]
            stage_min_P2 = stage['min_P2'](opt_P2)

            self.log(f'Best P2={best_P2:.2f} at idx={best_idx} vs. '
                     'stage_min_P2={stage_min_P2:.2f}', level=1)

            # If we have a winner then use that.
            if best_P2 >= stage_min_P2:
                break

        # Set the acqs table to the best catalog
        best_acq_idxs = fid_sets['acq_idxs'][best_idx]
        best_acq_halfws = fid_sets['acq_halfws'][best_idx]
        acqs.update_idxs_halfws(best_acq_idxs, best_acq_halfws)
        acqs.fid_set = fid_sets['fid_ids'][best_idx]

        # Finally set the fids table to the desired fid set
        fids.set_fid_set(acqs.fid_set)

        self.log(f"Best acq-fid set: P2={best_P2:.2f} "
                 f"acq_idxs={best_acq_idxs} halfws={best_acq_halfws} fid_ids={acqs.fid_set}")

        if best_P2 < stage_min_P2:
            self.log(f'No acq-fid combination was found that met stage requirements',
                     warning=True)


# Slot-based catalogs corresponding to OBC 8-element acq and guide catalogs
class ObcCat(list):
    def __init__(self, *args, **kwargs):
        super().__init__([None] * 8)

    def __setitem__(self, item, value):
        if self[item] is not None:
            raise IndexError(f'slot {item} is already set => program logic error')
        value['slot'] = item
        super().__setitem__(item, value)

    def add(self, value, descending=False):
        # First check if item is already there, if so then return that slot
        for slot in range(8):
            if (row := self[slot]) is not None and row['id'] == value['id']:
                return slot

        # Else fill in the next available slot
        slots = range(7, -1, -1) if descending else range(8)

        for slot in slots:
            if self[slot] is None:
                self[slot] = value
                return slot
        else:
            raise IndexError('catalog is full')


def merge_cats(fids=None, guides=None, acqs=None):
    # Make an empty (zero-length) catalog with the right columns
    if acqs is not None:
        empty = acqs[0:0]
    elif guides is not None:
        empty = guides[0:0]
    elif fids is not None:
        empty = fids[0:0]
    else:
        raise ValueError('cannot call merge_cats with no catalog inputs')

    fids = empty if fids is None else fids
    guides = empty if guides is None else guides
    acqs = empty if acqs is None else acqs
    gfms = empty  # Guide converted from monitor (must be 8x8)
    mons = empty  # Monitor stars

    colnames = ['slot', 'id', 'type', 'sz', 'p_acq', 'mag', 'maxmag',
                'yang', 'zang', 'dim', 'res', 'halfw']

    if len(fids) > 0:
        fids['type'] = 'FID'
        fids['mag'] = 7.0
        fids['maxmag'] = 8.0
        fids['halfw'] = 25
        fids['dim'], fids['res'] = get_dim_res(fids['halfw'])
        fids['p_acq'] = 0
        fids['sz'] = '8x8'

    if len(guides) > 0:
        guides['slot'] = 0  # Filled in later
        guides['p_acq'] = 0

        if not np.all(ok := (guides['type'] == 'GUI')):
            mons = guides[guides['type'] == 'MON']  # Monitor window
            gfms = guides[guides['type'] == 'GFM']  # Guide From Mon
            guides = guides[ok]  # "Normal" guide stars

    if len(acqs) > 0:
        img_size = get_img_size(len(fids))
        acqs['type'] = 'ACQ'
        acqs['maxmag'] = (acqs['mag'] + 1.5).clip(None, ACA.max_maxmag)
        acqs['dim'], acqs['res'] = get_dim_res(acqs['halfw'])
        acqs['sz'] = f'{img_size}x{img_size}'

    cat_guides = ObcCat()
    cat_acqs = ObcCat()

    # Guide from Monitors (descending from slot 7)
    for gfm in gfms:
        gfm['type'] = 'GUI'
        slot = cat_guides.add(gfm, descending=True)
        # If the GFM is also an acq then add to same slot
        if len(acqs) > 0 and gfm['id'] in acqs['id']:
            acq = acqs.get_id(gfm['id'])
            acq['sz'] = gfm['sz']  # Set size to 8x8
            cat_acqs[slot] = acq

    # Monitors (descending from slot 7)
    for mon in mons:
        cat_guides.add(mon, descending=True)

    # Now do fids (ascending from slot 0)
    for fid in fids:
        cat_guides.add(fid[colnames])

    # BOT stars, ascending in slot
    for acq_id in acqs['id']:
        if acq_id in guides['id']:
            acq = acqs.get_id(acq_id)
            guide = guides.get_id(acq_id)
            acq['sz'] = guide['sz']
            slot = cat_guides.add(guide)
            cat_acqs[slot] = acq

    # Fill in the rest of the guides (ascending from slot 0). Any pre-existing
    # ones are ignored.
    for guide in guides:
        cat_guides.add(guide)

    # Fill in the rest of the acqs (ascending from slot 0). Any pre-existing
    # ones are ignored.
    for acq in acqs:
        cat_acqs.add(acq)

    # Accumulate a list of table Row objects to be assembled into the final table.
    rows = []

    # Fids
    for fid in cat_guides:
        if fid['type'] == 'FID':
            rows.append(fid[colnames])

    # Add BOT stars
    for guide, acq in zip(cat_guides, cat_acqs):
        if guide['type'] == 'GUI' and guide['id'] == acq['id']:
            guide['type'] = 'BOT'
            acq['type'] = 'BOT'
            rows.append(acq[colnames])

    # Guide only
    for guide, acq in zip(cat_guides, cat_acqs):
        if guide['type'] == 'GUI' and guide['id'] != acq['id']:
            rows.append(guide[colnames])

    # Monitor stars
    for guide in cat_guides:
        if guide['type'] == 'MON':
            rows.append(guide[colnames])

    # Acq only
    for guide, acq in zip(cat_guides, cat_acqs):
        if guide['id'] != acq['id']:
            rows.append(acq[colnames])

    # Create final table and assign idx
    aca = ACACatalogTable(rows=rows, names=colnames)
    aca.add_column(np.arange(1, len(aca) + 1), name='idx', index=1)

    # For monitor windows, at this point the `dim` column corresponds to the
    # id of the tracking slot. Here we convert that to the correct slot.
    if len(mons) > 0:
        for row in aca:
            if row['type'] == 'MON':
                row['dim'] = aca.get_id(row['dim'])['slot']

    return aca
