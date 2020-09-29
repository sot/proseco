import traceback
import numpy as np
from itertools import count
import copy

from astropy.table import Column, Table

from .core import ACACatalogTable, get_kwargs_from_starcheck_text, MetaAttribute
from .guide import get_guide_catalog, GuideTable
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

    # Set guide star image readout size, either from explicit user input or
    # using the default rule defined in ``get_img_size()``.
    img_size = guides.get_img_size(len(fids))

    if len(guides) > 0:
        guides['slot'] = 0  # Filled in later
        guides['type'] = 'GUI'
        guides['maxmag'] = (guides['mag'] + 1.5).clip(None, ACA.max_maxmag)
        guides['p_acq'] = 0
        guides['dim'] = 1
        guides['res'] = 1
        guides['halfw'] = 25
        guides['sz'] = f'{img_size}x{img_size}'

    if len(acqs) > 0:
        acqs['type'] = 'ACQ'
        acqs['maxmag'] = (acqs['mag'] + 1.5).clip(None, ACA.max_maxmag)
        acqs['dim'] = 20
        acqs['sz'] = f'{img_size}x{img_size}'
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

    # Sort the BOTs in agasc id order to match what will happen within the MATLAB code
    bot_ids = sorted(filt(guides, set(guides['id']) & set(acqs['id'])))

    # Do not sort the GUI and ACQ stars (leave in selected order)
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
