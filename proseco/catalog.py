# Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import os
import traceback

import numpy as np
from astropy.table import Table

from proseco.characteristics import MonFunc

from . import __version__ as VERSION
from . import characteristics as ACA
from . import characteristics_acq as ACQ
from . import get_aca_catalog as _get_aca_catalog_package
from .acq import AcqTable, get_acq_catalog, get_maxmag
from .core import (
    ACACatalogTable,
    MetaAttribute,
    get_dim_res,
    get_img_size,
    get_kwargs_from_starcheck_text,
)
from .fid import FidTable, get_fid_catalog
from .guide import GuideTable, get_guide_candidates, get_guide_catalog, get_t_ccds_bonus
from .monitor import BadMonitorError, get_mon_catalog

# Colnames and types for final ACA catalog
ACA_CATALOG_DTYPES = {
    "slot": np.int64,
    "idx": np.int64,
    "id": np.int64,
    "type": "U3",
    "sz": "U3",
    "p_acq": np.float64,
    "mag": np.float64,
    "maxmag": np.float64,
    "yang": np.float64,
    "zang": np.float64,
    "dim": np.int64,
    "res": np.int64,
    "halfw": np.int64,
}


def get_aca_catalog(obsid=0, **kwargs):
    # Docstring is in __init__.py and defined below.

    raise_exc = kwargs.pop("raise_exc", True)  # This cannot credibly fail

    try:
        # If obsid is supplied as a string then it is taken to be starcheck text
        # with required info.  User-supplied kwargs take precedence, however.
        if isinstance(obsid, str):
            force_catalog = "--force-catalog" in obsid
            kw = get_kwargs_from_starcheck_text(obsid, force_catalog=force_catalog)
            obsid = kw.pop("obsid")
            for key, val in kw.items():
                if key not in kwargs:
                    kwargs[key] = val

        if "monitors" in kwargs:
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


get_aca_catalog.__doc__ = _get_aca_catalog_package.__doc__


def _get_aca_catalog(**kwargs):
    """
    Do the actual work of getting the acq, fid and guide catalogs and
    assembling the merged aca catalog.
    """
    raise_exc = kwargs.pop("raise_exc")
    img_size_guide = kwargs.pop("img_size_guide", None)

    aca = ACATable()
    aca.set_attrs_from_kwargs(**kwargs)
    aca.call_args = kwargs.copy()
    aca.version = VERSION

    # Override t_ccd related inputs with effective temperatures for downstream
    # action by AcqTable, GuideTable, FidTable.  See set_attrs_from_kwargs()
    # - t_ccd_eff_{acq,guide} are the effective T_ccd values which are adjusted
    #   if the actual t_ccd{acq,guide} values are above ACA.aca_t_ccd_penalty_limit.
    # - t_ccd_{acq,guide} are the actual (or predicted) values from the call
    # The downstream AcqTable, GuideTable, and FidTable are initialized with the
    # *effective* values as t_ccd.  Those classes do not have the concept of effective
    # temperature.
    if aca.t_ccd_eff_acq is None:
        aca.t_ccd_eff_acq = get_effective_t_ccd(aca.t_ccd_acq)
    if aca.t_ccd_eff_guide is None:
        aca.t_ccd_eff_guide = get_effective_t_ccd(aca.t_ccd_guide)

    kwargs["t_ccd_acq"] = aca.t_ccd_eff_acq
    kwargs["t_ccd_guide"] = aca.t_ccd_eff_guide

    # These are allowed inputs to get_aca_catalog but should not be passed to
    # get_{acq,guide,fid}_catalog. Pop them from kwargs.
    for kwarg in (
        "t_ccd",
        "t_ccd_eff_acq",
        "t_ccd_eff_guide",
        "stars",
        "t_ccd_penalty_limit",
    ):
        kwargs.pop(kwarg, None)

    # Get stars (typically from AGASC) and do not filter for stars near
    # the ACA FOV.  This leaves the full radial selection available for
    # later roll optimization.  Use aca.stars or aca.acqs.stars from here.
    # Set the agasc_file MetaAttribute if it is available in the stars table meta.
    aca.set_stars(filter_near_fov=False)
    if "agasc_file" in aca.stars.meta:
        aca.agasc_file = aca.stars.meta["agasc_file"]

    aca.log("Starting get_acq_catalog")
    aca.acqs = get_acq_catalog(stars=aca.stars, **kwargs)

    # Store the date of the dark cal if available.
    if hasattr(aca.acqs, "dark_date"):
        aca.dark_date = aca.acqs.dark_date

    # Get initial guide candidates
    guides = get_guide_candidates(stars=aca.acqs.stars, **kwargs)
    initial_guide_cands = guides.cand_guides

    # Note that aca.acqs.stars is a filtered version of aca.stars and includes
    # only stars that are in or near ACA FOV.  Use this for fids and guides stars.
    aca.log("Starting get_fid_catalog")
    aca.fids = get_fid_catalog(
        stars=aca.acqs.stars, acqs=aca.acqs, guide_cands=initial_guide_cands, **kwargs
    )
    aca.acqs.fids = aca.fids

    if aca.optimize:
        aca.log("Starting optimize_acqs_fids")
        aca.optimize_acqs_fids(**kwargs)

    aca.acqs.fid_set = aca.fids["id"]

    aca.log("Starting get_mon_catalog")
    aca.mons = get_mon_catalog(stars=aca.acqs.stars, **kwargs)

    aca.log("Starting get_guide_catalog")
    aca.guides = get_guide_catalog(
        stars=aca.acqs.stars,
        fids=aca.fids,
        mons=aca.mons,
        guides=guides,
        img_size=img_size_guide,
        **kwargs,
    )

    # Set output catalog aca.n_guide to the number of requested guide stars as
    # determined in guide star selection processing. This differs from the input
    # arg value of n_guide which is (confusingly) the number of available slots
    # for guide + monitor stars / windows. Thus if the input n_guide is set to
    # 5 and there is a monitor window then aca.n_guide will be 4.
    aca.n_guide = aca.guides.n_guide

    # Make a merged starcheck-like catalog.  Catch any errors at this point to avoid
    # impacting operational work (call from Matlab).
    try:
        aca.log("Starting merge_cats")
        merge_cat = merge_cats(
            fids=aca.fids, guides=aca.guides, acqs=aca.acqs, mons=aca.mons
        )
        for name in merge_cat.colnames:
            aca[name] = merge_cat[name]
    except Exception:
        if raise_exc:
            raise

        empty = ACACatalogTable.empty()
        for name in empty.colnames:
            aca[name] = empty[name]

        aca.exception = traceback.format_exc()

    aca.log("Finished aca_get_catalog")
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
    kwargs.pop("raise_exc")

    # Make a stub aca to get monitors attribute
    aca = ACATable()
    aca.set_attrs_from_kwargs(**kwargs)

    # If no auto-convert mon stars then do the normal star selection.
    monitors = aca.monitors
    if np.all(monitors["function"] != MonFunc.AUTO):
        return _get_aca_catalog(**kwargs_orig)

    # Find the entries with auto-convert
    is_auto = monitors["function"] == MonFunc.AUTO  # Auto-convert to guide

    # First get the catalog with automon entries scheduled as tracking MON windows.
    kwargs = kwargs_orig.copy()
    monitors["function"][is_auto] = MonFunc.MON_TRACK  # Tracking MON window
    kwargs["monitors"] = monitors

    # Get the catalog and do a sparkles review
    aca_mon = _get_aca_catalog(**kwargs)
    aca_mon.call_args = kwargs_orig.copy()  # Needed for roll optimization, see #364
    acar_mon = aca_mon.get_review_table()
    acar_mon.run_aca_review()
    crits_mon = set(msg["text"] for msg in (acar_mon.messages >= "critical"))

    # Now get the catalog with automon entries scheduled as guide stars
    monitors["function"][is_auto] = MonFunc.GUIDE
    kwargs["raise_exc"] = True
    try:
        aca_gui = _get_aca_catalog(**kwargs)
        aca_gui.call_args = kwargs_orig.copy()  # Needed for roll optimization, see #364
    except BadMonitorError as exc:
        aca_mon.log(f"unable to convert monitor to guide: {exc}")
        return aca_mon

    # Get the catalog and do a sparkles review
    acar_gui = aca_gui.get_review_table()
    acar_gui.run_aca_review()
    crits_gui = set(msg["text"] for msg in (acar_gui.messages >= "critical"))

    # If there are no new critical messages then schedule as guide star(s).
    # This checks that every critical in crit_gui is also in crits_mon.
    out = aca_gui if crits_gui <= crits_mon else aca_mon

    return out


def get_effective_t_ccd(t_ccd, t_ccd_penalty_limit=None):
    """Return the effective T_ccd used for selection and catalog evaluation.

    For details see Dynamic ACA limits in baby steps section in:
    https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/StarWorkingGroupMeeting2019x02x13

    :param t_ccd: float
        Actual (predicted) CCD temperature
    :param t_ccd_penalty_limit: float, None
        ACA penalty limit to use (degC). Default = aca_t_ccd_penalty_limit from
        proseco.characteristics.
    :returns: t_ccd_eff
        Effective CCD temperature (degC) for use in star selection
    """
    t_limit = (
        ACA.aca_t_ccd_penalty_limit
        if t_ccd_penalty_limit is None
        else t_ccd_penalty_limit
    )
    if t_limit is not None and t_ccd > t_limit:
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

    required_attrs = (
        "att",
        "n_fid",
        "n_guide",
        "man_angle",
        "t_ccd_acq",
        "t_ccd_guide",
        "dither_acq",
        "dither_guide",
        "date",
        "detector",
    )

    optimize = MetaAttribute(default=True)
    call_args = MetaAttribute(default={})
    version = MetaAttribute()
    agasc_file = MetaAttribute(is_kwarg=False)

    # For validation with get_aca_catalog(obsid), store the starcheck
    # catalog in the ACATable meta.
    starcheck_catalog = MetaAttribute(is_kwarg=False)

    # Effective T_ccd used for dynamic ACA limits (see updates_for_t_ccd_effective()
    # method below).
    t_ccd_eff_acq = MetaAttribute()
    t_ccd_eff_guide = MetaAttribute()
    t_ccd_penalty_limit = MetaAttribute()

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

        if self.t_ccd_penalty_limit is None:
            self.t_ccd_penalty_limit = ACA.aca_t_ccd_penalty_limit
        self.t_ccd_eff_acq = get_effective_t_ccd(
            self.t_ccd_acq, self.t_ccd_penalty_limit
        )
        self.t_ccd_eff_guide = get_effective_t_ccd(
            self.t_ccd_guide, self.t_ccd_penalty_limit
        )
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
        ok = self.acqs.get_candidates_mask(stars) | self.guides.get_candidates_mask(
            stars
        )
        return ok

    def make_report(self, rootdir="."):
        """
        Make summary HTML report for acq and guide selection process and outputs.

        Outputs are in ``<rootdir>/obs<obsid>/{acq,guide}/index.html`` plus related
        images in that directory.

        :param rootdir: root directory for outputs

        """
        self.acqs.make_report(rootdir=rootdir)
        self.guides.make_report(rootdir=rootdir)

    def optimize_acqs_fids(self, guides=None, **kwargs):
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
        if (
            len(self.fids) > 0
            or self.n_fid == 0
            or len(self.fids.cand_fids) == 0
            or len(self.fids.cand_fid_sets) == 0
        ):
            self.log("No acq-fid optimization required")
            return

        from chandra_aca.star_probs import guide_count

        no_fid_guides = get_guide_catalog(
            stars=acqs.stars,
            **kwargs,
        )

        # Start with the no-fids optimum catalog and save required info to restore
        opt_P2 = -acqs.get_log_p_2_or_fewer()
        t_ccd_applied = get_t_ccds_bonus(
            no_fid_guides["mag"],
            no_fid_guides.t_ccd,
            no_fid_guides.dyn_bgd_n_faint,
            no_fid_guides.dyn_bgd_dt_ccd,
        )
        opt_guide_count = guide_count(no_fid_guides["mag"], t_ccd_applied)
        orig_acq_idxs = acqs["idx"].tolist()
        orig_acq_halfws = acqs["halfw"].tolist()

        self.log(
            f"Starting opt_P2={opt_P2:.2f}: ids={orig_acq_idxs} halfws={orig_acq_halfws}"
        )

        # If not at least 2 fids then punt on optimization.
        cand_fids = fids.cand_fids

        # Get list of fid_sets that are consistent with candidate fids. These
        # fid sets are the combinations of 3 (or 2) fid lights in preferred
        # order.
        rows = []
        for fid_set in fids.cand_fid_sets:
            spoiler_score = sum(
                cand_fids.get_id(fid_id)["spoiler_score"] for fid_id in fid_set
            )
            rows.append((fid_set, spoiler_score))

        # Make a table to keep track of candidate fid_sets along with the
        # ranking metric P2 and the acq catalog info halfws and star ids.
        fid_sets = Table(rows=rows, names=("fid_ids", "spoiler_score"))
        fid_sets["P2"] = -99.0  # Marker for unfilled values
        fid_sets["guide_count"] = -99.0
        fid_sets["acq_halfws"] = None
        fid_sets["acq_idxs"] = None

        # Group the table into groups by spoiler score.  This preserves the
        # original fid set ordering within a group.
        fid_sets = fid_sets.group_by("spoiler_score")

        # Iterate through each spoiler_score group and then within that iterate
        # over each fid set.
        for fid_set_group in fid_sets.groups:
            spoiler_score = fid_set_group["spoiler_score"][0]
            self.log(f"Checking fid sets with spoiler_score={spoiler_score}", level=1)

            for fid_set in fid_set_group:
                # get the rows of candidate fids that correspond to the fid_ids
                # in the current fid_set.
                fid_mask = [fid_id in fid_set["fid_ids"] for fid_id in cand_fids["id"]]
                fids_for_set = cand_fids[fid_mask]
                local_guides = get_guide_catalog(
                    stars=acqs.stars,
                    fids=fids_for_set,
                    **kwargs,
                )
                local_t_ccd_applied = get_t_ccds_bonus(
                    local_guides["mag"],
                    local_guides.t_ccd,
                    local_guides.dyn_bgd_n_faint,
                    local_guides.dyn_bgd_dt_ccd,
                )
                local_guide_count = guide_count(
                    local_guides["mag"], local_t_ccd_applied
                )
                # Set the internal acqs fid set.  This does validation of the set
                # and also calls update_p_acq_column().
                acqs.fid_set = fid_set["fid_ids"]

                # If P2 is effectively unchanged after updating the fid set,
                # that means there are no fids spoiling an acq star in the
                # optimum (no-fid) acq catalog, and there is no need to
                # re-optimize.  Furthermore this fid set is guaranteed to meet
                # the stage_min_P2 acceptance so no need to check any other fid
                # sets.  Only optimize if P2 decreased by at least 0.001,
                # remembering P2 is the -log10(prob).
                fid_set_P2 = -acqs.get_log_p_2_or_fewer()
                found_good_set = (fid_set_P2 - opt_P2 > -0.001) & (
                    local_guide_count >= (opt_guide_count - 0.05)
                )
                if found_good_set:
                    self.log(
                        f"No change in P2 or guide_count for fid set {acqs.fid_set}, "
                        f"skipping optimization"
                    )
                else:
                    # Re-optimize the catalog with the fid set selected and get new probs.
                    acqs.optimize_catalog()
                    acqs.update_p_acq_column(acqs)  # Needed for get_log_p_2_or_fewer

                # Store optimization results
                fid_set["P2"] = -acqs.get_log_p_2_or_fewer()
                fid_set["guide_count"] = local_guide_count
                fid_set["acq_idxs"] = acqs["idx"].tolist()
                fid_set["acq_halfws"] = acqs["halfw"].tolist()

                self.log(
                    f"Fid set {fid_set['fid_ids']}: P2={fid_set['P2']:.2f} "
                    f" guide_count={fid_set['guide_count']:.2f} "
                    f"acq_idxs={fid_set['acq_idxs']} halfws={fid_set['acq_halfws']}",
                    level=2,
                )

                if found_good_set:
                    break
                else:
                    # Put the catalog back to the original no-fid optimum and continue
                    # trying remaining fid sets.
                    acqs.update_idxs_halfws(orig_acq_idxs, orig_acq_halfws)

            # Get the best fid set / acq catalog configuration so far.  Fid sets not
            # yet considered have P2 = -99.

            # Are there sets with P2 >= 2 and guide_count >= 4
            passable = (fid_sets["P2"] >= 2) & (fid_sets["guide_count"] >= 4)
            if np.any(passable):
                fid_sets_passable = fid_sets[passable]
                best_idx = np.argmax(fid_sets_passable["P2"])
                best_P2 = fid_sets_passable["P2"][best_idx]
                best_idx = np.where(passable)[0][best_idx]

            # If none passable then just get the best P2
            else:
                best_idx = np.argmax(fid_sets["P2"])
                best_P2 = fid_sets["P2"][best_idx]

            # Get the row of the fid / acq stages table to determine the required minimum
            # P2 given the fid spoiler score.
            stage = ACQ.fid_acq_stages.loc[spoiler_score]
            stage_min_P2 = stage["min_P2"](opt_P2)
            self.log(
                f"Best P2={best_P2:.2f} at idx={best_idx} vs. "
                f"stage_min_P2={stage_min_P2:.2f}",
                level=1,
            )

            # If we have a winner then use that.
            if (best_P2 >= stage_min_P2) and np.any(passable):
                break

        # Set the acqs table to the best catalog
        best_acq_idxs = fid_sets["acq_idxs"][best_idx]
        best_acq_halfws = fid_sets["acq_halfws"][best_idx]
        acqs.update_idxs_halfws(best_acq_idxs, best_acq_halfws)
        acqs.fid_set = fid_sets["fid_ids"][best_idx]

        # Finally set the fids table to the desired fid set
        fids.set_fid_set(acqs.fid_set)

        self.log(
            f"Best acq-fid set: P2={best_P2:.2f} "
            f"acq_idxs={best_acq_idxs} halfws={best_acq_halfws} fid_ids={acqs.fid_set}"
        )

        if best_P2 < stage_min_P2:
            self.log(
                "No acq-fid combination was found that met stage requirements",
                warning=True,
            )


class ObcCat(list):
    """Slot-based catalog corresponding to OBC 8-element acq and guide catalogs.

    Each row is a catalog object (acq, guide, fid, mon) with a dict-like
    interface (dict or Table Row). This class is a list subclass with special
    properties:

    - Initialized as a length 8 list of dict {'id': None, 'type': None}
    - Each of the 8 items correspond to that ACA slot.
    - The default value corresponds to a "empty" slot.
    - Setting an entry in a slot that is occupied raises an exception.
    - Allows adding an entry to the first/last available slot, doing nothing if
      the ``id`` is already in the catalog.
    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "")
        self.debug = kwargs.pop("debug", None)
        super().__init__()
        for _ in range(8):
            self.append({"id": None, "type": None})

    def __setitem__(self, item, value):
        """Set list ``item`` to ``value``

        Raises an exception if row ``item`` already has a non-default value.

        :param item: int
            List item to set
        :param value: dict, Row
            Value to set
        """
        if self[item]["type"] is not None:
            raise IndexError(f"slot {item} is already set => program logic error")
        value["slot"] = item
        super().__setitem__(item, value)
        if self.debug:
            print(self)

    def add(self, value, descending=False):
        """Add ``value`` to catalog in first/last empty slot, returning the slot.

        If ``value['id']`` is already in the catalog then return that slot.

        If the catalog already has 8 entries then an IndexError is raised.

        :param value: dict, Row
            Catalog entry to add
        :param descending: bool
            Direction in which to find the first empty slot (starting from
            beginning or end of catalog)
        :returns: int
            Slot in which ``value`` was placed
        """
        # First check if item is already there, if so then return that slot
        for slot in range(8):
            if self[slot]["id"] == value["id"]:
                return slot

        # Else fill in the next available slot
        slots = range(7, -1, -1) if descending else range(8)

        for slot in slots:
            if self[slot]["type"] is None:
                self[slot] = value
                return slot

        raise IndexError("catalog is full")

    def as_table(self):
        colnames = list(ACA_CATALOG_DTYPES)
        colnames.remove("idx")
        rows = []
        for row in self:
            out = {}
            for name in ACA_CATALOG_DTYPES:
                try:
                    out[name] = row[name]
                except KeyError:
                    pass
            rows.append(out)
        return ACACatalogTable(rows)[colnames]

    def __repr__(self):
        out = "\n".join([self.name, str(self.as_table()), ""])
        return out


def _merge_cats_debug(cat_debug, tbl, message):
    if cat_debug and len(tbl) > 0:
        print("*" * 80)
        print(f"Adding {message}")
        print("*" * 80)


def merge_cats(fids=None, guides=None, acqs=None, mons=None):
    """Merge ``fids``, ``guides``, and ``acqs`` catalogs into one catalog.

    The output of this function is a catalog which corresponds to the final
    "flight" catalog that can be translated directly to appropriate OBC
    spacecraft commanding, e.g. a DOT ACQ command. The ordering of entries
    is intended to match the legacy behavior of MATLAB tools.

    :param fids: FidTable, None (optional)
        Table of fids
    :param guides: GuideTable, None (optional)
        Table of guide stars
    :param acqs: AcqTable, None (optional)
        Table of acquisition stars
    :returns: ACACatalogTable
        Merged catalog
    """
    # Make an empty (zero-length) catalog with the right columns. This is used
    # just below to replace any missing catalogs.
    for cat in (acqs, guides, fids, mons):
        if cat is not None:
            empty = cat[0:0]
            break
    else:
        raise ValueError("cannot call merge_cats with no catalog inputs")

    fids = empty if fids is None else fids
    guides = empty if guides is None else guides
    acqs = empty if acqs is None else acqs
    mons = empty if mons is None else mons
    gfms = empty  # Guide converted from monitor (must be 8x8)

    # Columns in the final merged catalog, except we leave out idx since that
    # is included just at the end.
    colnames = [key for key in ACA_CATALOG_DTYPES if key != "idx"]

    # First add or modify columns for the fids, guides, and acqs tables so that
    # they are all consistent and ready for merging into a single final catalog.

    if len(fids) > 0:
        fids["type"] = "FID"
        fids["mag"] = 7.0
        fids["maxmag"] = 8.0
        fids["halfw"] = 25
        fids["dim"], fids["res"] = get_dim_res(fids["halfw"])
        fids["p_acq"] = 0.0
        fids["sz"] = "8x8"

    if len(guides) > 0:
        guides["slot"] = 0  # Filled in later
        guides["p_acq"] = 0.0

        # Guides table can include three sub-types: GUI (plain guide star), MON
        # (monitor window), and GFM (guide from monitor, required to be sz=8x8
        # and fill from slot 7 down).
        if not np.all(ok := (guides["type"] == "GUI")):
            # Monitor windows MFX = MON fixed, MTR = Mon tracked
            gfms = guides[guides["type"] == "GFM"]  # Guide From Mon
            guides = guides[ok]  # "Normal" guide stars

    if len(acqs) > 0:
        # TODO: move these into acq.py where possible
        img_size = guides.img_size or get_img_size(len(fids))
        acqs["type"] = "ACQ"

        # Set maxmag for acqs. Start w/ legacy version corresponding to behavior
        # prior to using the search_hits < 50 constraint.
        maxmags_legacy = np.clip(
            acqs["mag"] + ACA.max_delta_maxmag, a_min=None, a_max=11.2
        )
        if "PROSECO_IGNORE_MAXMAGS_CONSTRAINTS" in os.environ:
            # Use the legacy version
            acqs["maxmag"] = maxmags_legacy
        else:
            # Use the min of the legacy and search hits < 50 limits
            maxmags_search = [get_maxmag(acq["halfw"], acqs.t_ccd) for acq in acqs]
            acqs["maxmag"] = np.minimum(maxmags_legacy, maxmags_search)

        acqs["dim"], acqs["res"] = get_dim_res(acqs["halfw"])
        acqs["sz"] = f"{img_size}x{img_size}"

    if len(acqs) > 8:
        raise ValueError(f"catalog has too many acq entries: n_acq={len(acqs)}")

    if len(guides) + len(mons) + len(gfms) + len(fids) > 8:
        raise ValueError(
            "catalog has too many guide entries: "
            f"n_guide={len(guides)} n_fid={len(fids)} "
            f"n_mon={len(mons)} n_gfm={len(gfms)}"
        )

    # Create two 8-slot tables where all slots are initially empty. These
    # correspond to the OBC acquisition and guide tables. The guide table
    # includes fids and guides. This includes a special debug flag to print
    # the catalog each time an entry is added.
    cat_debug = "PROSECO_PRINT_OBC_CAT" in os.environ
    cat_acqs = ObcCat(name="Acquisition catalog", debug=cat_debug)
    cat_guides = ObcCat(name="Fid/guide/mon catalog", debug=cat_debug)

    # Fill in the acq and guide tables in a specific order:
    # - GUI from MON (GFM) in descending slot order (starting from 7)
    # - MON in descending slot order
    # - FID in ascending slot order (starting from 0)
    # - BOT in ascending slot order
    # - GUI-only in ascending slot order
    # - ACQ-only in ascending slot order

    # Guide from Monitors (descending from slot 7)
    _merge_cats_debug(cat_debug, gfms, "Guide from Monitors (GFM)")
    for gfm in gfms:
        gfm["type"] = "GUI"
        slot = cat_guides.add(gfm, descending=True)
        # If the GFM is also an acq then add to same slot
        # TODO: probably don't need the len(acqs) > 0 check.
        if len(acqs) > 0 and gfm["id"] in acqs["id"]:
            acq = acqs.get_id(gfm["id"])
            acq["sz"] = gfm["sz"]  # Set acq size to 8x8
            cat_acqs[slot] = acq

    # Monitors (descending from slot 7)
    _merge_cats_debug(cat_debug, mons, "Monitors (MON)")
    for mon in mons:
        cat_guides.add(mon, descending=True)

    # Now do fids (ascending from slot 0)
    _merge_cats_debug(cat_debug, fids, "Fids (FID)")
    for fid in fids:
        # TODO: why is this fid[colnames], unlike guides and acqs?
        cat_guides.add(fid[colnames])

    # BOT stars, ascending in slot
    _merge_cats_debug(cat_debug, gfms, "Both stars (BOT)")
    for acq_id in acqs["id"]:
        if acq_id in guides["id"]:
            acq = acqs.get_id(acq_id)
            guide = guides.get_id(acq_id)
            acq["sz"] = guide["sz"]
            slot = cat_guides.add(guide)
            cat_acqs[slot] = acq

    # Fill in the rest of the guides (ascending from slot 0). Any pre-existing
    # ones are ignored.
    _merge_cats_debug(cat_debug, gfms, "Guide-only stars (GUI)")
    for guide in guides:
        cat_guides.add(guide)

    # Fill in the rest of the acqs (ascending from slot 0). Any pre-existing
    # ones are ignored.
    _merge_cats_debug(cat_debug, gfms, "Acq-only stars (ACQ)")
    for acq in acqs:
        cat_acqs.add(acq)

    # Accumulate a list of star/fid entries in the specific order FID, BOT, GUI,
    # MON, ACQ, to be assembled into the final table.
    rows = []

    # Fids
    for guide in cat_guides:
        if guide["type"] == "FID":
            rows.append(guide[colnames])  # noqa: PERF401

    # Add BOT stars
    for guide, acq in zip(cat_guides, cat_acqs):
        if guide["type"] == "GUI" and guide["id"] == acq["id"]:
            guide["type"] = "BOT"
            acq["type"] = "BOT"
            rows.append(acq[colnames])

    # Guide only
    for guide, acq in zip(cat_guides, cat_acqs):
        if guide["type"] == "GUI" and guide["id"] != acq["id"]:
            rows.append(guide[colnames])

    # Monitor stars
    for guide in cat_guides:
        if guide["type"] in ("MTR", "MFX"):
            rows.append(guide[colnames])  # noqa: PERF401

    # Acq only
    for guide, acq in zip(cat_guides, cat_acqs):
        if acq["type"] is not None and guide["id"] != acq["id"]:
            rows.append(acq[colnames])

    # Create final table and assign idx
    aca = ACACatalogTable(
        rows=rows, names=colnames, dtype=[ACA_CATALOG_DTYPES[name] for name in colnames]
    )
    aca.add_column(
        np.arange(1, len(aca) + 1, dtype=ACA_CATALOG_DTYPES["idx"]), name="idx", index=1
    )

    # Finally, fix up the monitor window designated track slots (DIM/DTS)
    for row in aca:
        if row["type"] not in ("MTR", "MFX"):
            continue

        if row["type"] == "MTR":
            # Find the slot of the brightest guide star
            guides = aca[np.isin(aca["type"], ["GUI", "BOT"])]
            idx = np.argmin(guides["mag"])
            aca_id = guides["id"][idx]
            row["dim"] = aca.get_id(aca_id)["slot"]  # Mon window tracks this slot
        else:
            row["dim"] = row["slot"]  # Fixed (desig track slot is self slot)
        # Change type to standard MON
        row["type"] = "MON"

    if cat_debug:
        print("*" * 80)
        print("Final catalogs")
        print("*" * 80)
        print(cat_acqs)
        print(cat_guides)

    return aca
