# Licensed under a 3-clause BSD style license - see LICENSE.rst

from itertools import combinations
from typing import TYPE_CHECKING

import chandra_aca.aca_image
import numpy as np
from chandra_aca.aca_image import ACAImage, AcaPsfLibrary
from chandra_aca.transform import (
    count_rate_to_mag,
    mag_to_count_rate,
    snr_mag_for_t_ccd,
)

from proseco.characteristics import MonFunc

from . import characteristics as ACA
from . import characteristics_guide as GUIDE
from .core import (
    ACACatalogTable,
    AliasAttribute,
    MetaAttribute,
    bin2x2,
    get_dim_res,
    get_img_size,
)

if TYPE_CHECKING:
    from astropy.table import Table


CCD = ACA.CCD
APL = AcaPsfLibrary()

STAR_PAIR_DIST_CACHE = {}


def get_guide_catalog(obsid=0, **kwargs):
    """
    Get a catalog of guide stars

    If ``obsid`` corresponds to an already-scheduled obsid then the parameters
    ``att``, ``t_ccd``, ``date``, and ``dither`` will
    be fetched via ``mica.starcheck`` if not explicitly provided here.

    :param obsid: obsid (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param t_ccd: ACA CCD temperature (degC)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither: dither size 2-element tuple: (dither_y, dither_z) (float, arcsec)
    :param n_guide: number of guide stars to attempt to get
    :param fids: selected fids (used for guide star exclusion)
    :param stars: astropy.Table of AGASC stars (will be fetched from agasc if None)
    :param include_ids: list of AGASC IDs of stars to include in guide catalog
    :param exclude_ids: list of AGASC IDs of stars to exclude from guide catalog
    :param dark: ACAImage of dark map (fetched based on time and t_ccd if None)
    :param print_log: print the run log to stdout (default=False)

    :returns: GuideTable of acquisition stars
    """
    STAR_PAIR_DIST_CACHE.clear()

    guides = GuideTable()
    guides.set_attrs_from_kwargs(obsid=obsid, **kwargs)
    guides.set_stars()

    # Process monitor window requests, converting them into fake stars that
    # are added to the include_ids list.
    guides.process_monitors_pre()

    # Do a first cut of the stars to get a set of reasonable candidates
    guides.cand_guides = guides.get_initial_guide_candidates()

    # Process guide-from-monitor requests by finding corresponding star in
    # cand_guides and adding to the include_ids list.
    # guides.process_monitors_pre2()

    # Run through search stages to select stars
    selected = guides.run_search_stages()

    # Transfer to table (which at this point is an empty table)
    guides.add_columns(list(selected.columns.values()))

    if len(guides) < guides.n_guide:
        guides.log(
            f"Selected only {len(guides)} guide stars versus requested {guides.n_guide}",
            warning=True,
        )

    img_size = guides.get_img_size()
    if len(guides) > 0:
        guides["sz"] = f"{img_size}x{img_size}"
        guides["type"] = "GUI"
        guides["maxmag"] = (guides["mag"] + 1.5).clip(None, ACA.max_maxmag)
        guides["halfw"] = 25
        guides["dim"], guides["res"] = get_dim_res(guides["halfw"])

        guides.process_monitors_post()

    guides["idx"] = np.arange(len(guides))

    return guides


def get_guide_candidates_mask(
    stars: "Table",
    t_ccd: float = -10.0,
    dyn_bgd: bool = True,
) -> np.ndarray:
    """Get filter on ``stars`` for acceptable guide candidates.

    This does not include spatial filtering.

    Parameters
    ----------
    stars : Table
        Table of stars
    t_ccd : float
        ACA CCD temperature (degC)
    dyn_bgd : bool
        Dynamic background enabled flag

    Returns
    -------
    ok : np.array of bool
        Mask of acceptable stars
    """
    # Scale the reference ACA faint mag limit to the CCD temperature keeping
    # expected signal-to-noise constant.
    faint_mag_limit = snr_mag_for_t_ccd(
        t_ccd, ref_mag=GUIDE.ref_faint_mag, ref_t_ccd=GUIDE.ref_faint_mag_t_ccd
    )

    # Without dynamic background, ensure faint limit is no larger (fainter)
    # than ref_faint_mag (10.3 mag).
    if not dyn_bgd:
        faint_mag_limit = min(faint_mag_limit, GUIDE.ref_faint_mag)

    # Allow for stars from proseco StarsTable.from_stars() with `mag` and `mag_err`,
    # or native agasc.get_agasc_cone() stars table.
    mag = stars["mag"] if "mag" in stars.colnames else stars["MAG_ACA"]
    mag_err = (
        stars["mag_err"] if "mag_err" in stars.colnames else stars["MAG_ACA_ERR"] / 100
    )

    ok = (
        (stars["CLASS"] == 0)
        & (mag > 5.2)
        & (mag < faint_mag_limit)
        & (mag_err < 1.0)
        & (stars["COLOR1"] != 0.7)
        & (stars["ASPQ1"] < 20)  # Less than 1 arcsec offset from nearby spoiler
        & (stars["ASPQ2"] == 0)  # Unknown proper motion, or PM < 500 milli-arcsec/year
        & (stars["POS_ERR"] < 1250)  # Position error < 1.25 arcsec
        & ((stars["VAR"] == -9999) | (stars["VAR"] == 5))  # Not known to vary > 0.2 mag
    )
    return ok


class ImgSizeMetaAttribute(MetaAttribute):
    def __set__(self, instance, value):
        if value not in (4, 6, 8, None):
            raise ValueError("img_size must be 4, 6, 8, or None")
        instance.meta[self.name] = value


class GuideTable(ACACatalogTable):
    # Define base set of allowed keyword args to __init__. Subsequent MetaAttribute
    # or AliasAttribute properties will add to this.
    allowed_kwargs = ACACatalogTable.allowed_kwargs | set(["fids"])

    # Catalog type when plotting (None | 'FID' | 'ACQ' | 'GUI')
    catalog_type = "GUI"

    # Elements of meta that should not be directly serialized to pickle.
    # (either too big or requires special handling).
    pickle_exclude = ("stars", "dark")

    # Name of table.  Use to define default file names where applicable.
    # (e.g. `obs19387/guide.pkl`).
    name = "guide"

    # Required attributes
    required_attrs = ("att", "t_ccd_guide", "date", "dither_guide", "n_guide")

    cand_guides = MetaAttribute(is_kwarg=False)
    reject_info = MetaAttribute(default=[], is_kwarg=False)
    img_size = ImgSizeMetaAttribute()

    def reject(self, reject):
        """
        Add a reject dict to self.reject_info
        """
        reject_info = self.reject_info
        reject_info.append(reject)

    t_ccd = AliasAttribute()  # Maps t_ccd to t_ccd_guide attribute from base class
    dither = AliasAttribute()  # .. and likewise.
    include_ids = AliasAttribute()
    exclude_ids = AliasAttribute()

    def process_monitors_pre(self):
        # No action required if there are no monitor stars requested
        if self.mons is None or self.mons.monitors is None:
            return

        monitors = self.mons.monitors
        is_mon = np.isin(monitors["function"], [MonFunc.MON_FIXED, MonFunc.MON_TRACK])
        if np.any(is_mon):
            # For MON fixed and tracked, the dark cal map gets hacked to make a
            # local blob of hot pixels. Make a copy to avoid modifying the global
            # dark map which is shared by reference between acqs, guides and fids.
            self.dark = self.dark.copy()

        for monitor in monitors[is_mon]:
            # Make a square blob of saturated pixels that will keep away any
            # star selection.
            is_track = monitor["function"] == MonFunc.MON_TRACK
            dr = int(4 + np.ceil(self.dither.row if is_track else 0))
            dc = int(4 + np.ceil(self.dither.col if is_track else 0))
            row, col = int(monitor["row"]) + 512, int(monitor["col"]) + 512
            self.dark[row - dr : row + dr, col - dc : col + dc] = (
                ACA.bad_pixel_dark_current
            )

            # Reduce n_guide for each MON. On input the n_guide arg is the
            # number of GUI + MON, but for guide selection we need to make the
            # MON slots unavailable.
            self.n_guide -= 1
            if self.n_guide < 2:
                raise ValueError("too many MON requests leaving < 2 guide stars")

        is_guide = monitors["function"] == MonFunc.GUIDE
        for monitor in monitors[is_guide]:
            self.include_ids.append(monitor["id"])

    def process_monitors_post(self):
        """Post-processing of monitor windows.

        - Clean up fake stars from stars and cand_guides
        - Restore the dark current map
        - Set type, sz, res, dim columns for relevant entries to reflect
          status as monitor or guide-from-monitor in the guides catalog.
        """
        # No action required if there are no monitor stars requested
        if self.mons is None or self.mons.monitors is None:
            return

        # Mon window processing munges the dark cal to impose a keep-out zone.
        # Put the dark current back to the standard one if possible
        if self.acqs is not None:
            self.dark = self.acqs.dark

        # Find the guide stars that are actually from a MON request (converted
        # to guide) and set the size and type.
        for monitor in self.mons.monitors:
            if monitor["function"] == MonFunc.GUIDE:
                row = self.get_id(monitor["id"])
                row["sz"] = "8x8"
                row["type"] = "GFM"  # Guide From Monitor, changed in merge_catalog

    def get_img_size(self, n_fids=None):
        """Get guide image readout size from ``img_size`` and ``n_fids``.

        If img_size is None (typical case) then this uses the default rules
        defined in ``core.get_img_size()``, namely 8x8 for all guide stars
        unless overridden by the PROSECO_OR_IMAGE_SIZE environment variable.

        This requires that the ``fids`` attribute has been set, normally by
        providing the table as an arg to ``get_guide_catalog()``.

        Parameters
        ----------
        n_fids : int or None
            Number of fids in the catalog

        Returns
        -------
        int
            Guide star image readout size to be used in a catalog
        """
        if n_fids is None:
            n_fids = 0 if self.fids is None else len(self.fids)

        img_size = self.img_size

        if img_size is None:
            # Call the function from core as the single source for the rule.
            img_size = get_img_size(n_fids)

        return img_size

    def make_report(self, rootdir="."):
        """
        Make summary HTML report for guide selection process and outputs.

        Output is in ``<rootdir>/obs<obsid>/guide/index.html`` plus related images
        in that directory.

        :param rootdir: root directory for outputs

        """
        from .report_guide import make_report

        make_report(self, rootdir=rootdir)

    def run_search_stages(self):
        """
        Run search stages as necessary to select up to the requested number of guide stars.

        This routine first force selects any guide stars supplied as ``include_ids``, marking
        these as selected in stage 0 in the 'stage' column in the candidate table, and then
        runs search stages.

        In each "search stage", checks are run on each candidate to exclude candidates that
        do not meet the selection criteria in the stage.  Candidates that satisfy the
        stage requirements are marked with the allowed stage of selection (by its integer id)
        in the 'stage' column in the candidate table.

        The run_search_stages method loops over the search stages until at least the
        ``n_guide`` requested stars are marked with a stage >= 0 or until all stage checks have
        been exhausted.  This run_search_stages routine then sorts the candidates by
        selected stage and magnitude and returns up to the n requested guide stars as
        the selected stars.

        """
        cand_guides = self.cand_guides
        self.log("Starting search stages")
        if len(cand_guides) == 0:
            self.log("There are no candidates to check in stages.  Exiting")
            # Since there are no candidate stars, this returns the empty set of
            # cand_guides as the 'selected' stars.
            return cand_guides
        cand_guides["stage"] = -1
        # Force stars in include_ids to be selected at stage 0
        for star_id in self.include_ids:
            cand_guides["stage"][cand_guides["id"] == star_id] = 0
            self.log(f"{star_id} selected in stage 0 via include_ids", level=1)
        n_guide = self.n_guide
        for idx, stage in enumerate(GUIDE.stages, 1):
            already_selected = np.count_nonzero(cand_guides["stage"] != -1)

            # If we don't have enough stage-selected candidates, keep going.
            # Enough is defined as the requested n_guide + a "surplus" value
            # in characteristics
            if already_selected < (n_guide + GUIDE.surplus_stars):
                stage_ok = self.search_stage(stage)
                sel = cand_guides["stage"] == -1
                cand_guides["stage"][stage_ok & sel] = idx
                stage_selected = np.count_nonzero(stage_ok & sel)
                self.log(f"{stage_selected} stars selected in stage {idx}", level=1)
            else:
                self.log(
                    f"Quitting after stage {idx - 1} with {already_selected} stars",
                    level=1,
                )
                break
        self.log("Done with search stages")
        stage_cands = cand_guides[cand_guides["stage"] != -1]
        stage_cands.sort(["stage", "mag"])
        stage_cands = self.exclude_overlaps(stage_cands)
        guides = self.select_catalog(stage_cands[0 : n_guide + GUIDE.surplus_stars])

        if self.dyn_bgd_n_faint > 0:
            self.drop_excess_bonus_stars(guides)

        if len(guides) < self.n_guide:
            self.log(
                f"Could not find {self.n_guide} candidates after all search stages"
            )
        return guides

    def drop_excess_bonus_stars(self, guides):
        """Drop excess dynamic background faint "bonus" stars if necessary.

        For dyn bgd with dyn_bgd_n_faint > 0, candidates fainter then the
        nominal faint limit can be selected. However, only at most
        dyn_bgd_n_faint of these bonus faint stars are allowed in the final
        catalog (unless more are force-included).

        This method removes faint bonus stars (in-place within ``guides``) in
        excess of the allowed dyn_bgd_n_faint number. It is assumed that the
        catalog order is by star preference ('stage', 'mag'), so bonus stars
        that come first are kept.

        Force included stars are not removed.

        :param guides: Table of guide stars
        """
        # Compute the non-bonus faint_mag_limit
        faint_mag_limit = snr_mag_for_t_ccd(
            self.t_ccd, ref_mag=GUIDE.ref_faint_mag, ref_t_ccd=GUIDE.ref_faint_mag_t_ccd
        )
        n_faint = 0
        idxs_drop = []
        for idx in range(len(guides)):
            if guides["mag"][idx] > faint_mag_limit:
                n_faint += 1
                # If we have more than the allowed number of faint bonus stars
                # and the star is not force-included, mark it for removal.
                if (
                    n_faint > self.dyn_bgd_n_faint
                    and guides["id"][idx] not in self.include_ids
                ):
                    idxs_drop.append(idx)
        if idxs_drop:
            guides.remove_rows(idxs_drop)

    def exclude_overlaps(self, stage_cands):
        """
        Review the stars selected at any stage and exclude stars that overlap in
        tracking space with another. Overlap is defined as being within 12 pixels.
        """
        self.log("Checking for guide star overlap in stage-selected stars")
        nok = np.zeros(len(stage_cands)).astype(bool)
        for idx, star in enumerate(stage_cands):
            # If the star was manually-selected, don't bother checking to possibly exclude it.
            if star["id"] in self.include_ids:
                continue

            for jdx, other_star in enumerate(stage_cands):
                # The stage_cands are supplied in the order of preference (currently by mag)
                # Check and exclude a guide star only if it would spoil a lower index (better) star.
                if idx <= jdx:
                    continue
                drow = other_star["row"] - star["row"]
                dcol = other_star["col"] - star["col"]
                if np.abs(drow) <= 12 and np.abs(dcol) <= 12:
                    self.log(
                        f"Rejecting star {star['id']} with track overlap (12 pixels) "
                        f"with star {other_star['id']}"
                    )
                    self.reject(
                        {
                            "id": star["id"],
                            "type": "overlap",
                            "stage": 0,
                            "text": f"Cand {star['id']} has track overlap (12 pixels) "
                            f"with star {other_star['id']}",
                        }
                    )
                    nok[idx] = True
        return stage_cands[~nok]

    def select_catalog(self, stage_cands):
        """
        Select a catalog from the candidates.

        For the candidates selected at any stage, select the first combination that satisfies
        either all of the additional checks or the most checks.  The checks are manually weighted
        so that the jupiter check is weighted more heavily than the cluster checks (but
        the jupiter check only applies if jupiter is present).

        :param stage_cands: Table of stage-selected candidates

        """
        self.log(f"Selecting catalog from {len(stage_cands)} stage-selected stars")

        def index_combinations(n, m):
            seen = set()
            for n_tmp in range(m, n + 1):
                for comb in combinations(range(n_tmp), m):
                    if comb not in seen:
                        seen.add(comb)
                        yield comb

        choose_m = min(len(stage_cands), self.n_guide)
        n_tries = 0

        # Try to find the combination with the highest weighted sum of passing checks
        best_score = -1
        best_cands = None

        # The best possible score is 3 for 3 cluster checks or 8 for 3 cluster checks plus
        # a weighted-5 jupiter check
        best_possible_score = 8 if self.jupiter else 3

        for comb in index_combinations(len(stage_cands), choose_m):
            cands = stage_cands[list(comb)]

            # If there are any include_ids, then the selected stars must include them.
            # If they aren't included, skip this combination.
            if self.include_ids and not set(self.include_ids).issubset(cands["id"]):
                continue

            n_tries += 1

            score = 0

            cluster_weights = 1
            cluster_check_status = run_cluster_checks(cands)
            score += np.sum(cluster_check_status) * cluster_weights

            if len(self.jupiter) > 0:
                jupiter_weight = 5
                from proseco.jupiter import jupiter_distribution_check

                jupiter_check_status = jupiter_distribution_check(cands, self.jupiter)
                score += np.sum([jupiter_check_status]) * jupiter_weight

            if score > best_score:
                best_score = score
                best_cands = cands

            if best_score == best_possible_score:
                break

        if best_cands is not None:
            self.log(
                f"Selected stars with weighted score {best_score}",
                warning=False,
                tried_combinations=n_tries,
            )
            return best_cands

        # Fallback: return the first choose_m stars if nothing else worked.
        # The code should only get here if there are more include_ids than n_guide.
        self.log(
            "No combination satisfied any checks, returning first available set",
            warning=True,
            tried_combinations=n_tries,
        )
        return stage_cands[:choose_m]

    def search_stage(self, stage):
        """
        Review the candidates with the criteria defined in ``stage`` and return a mask that
        marks the candidates that are "ok" for this search stage.  This is used to then
        annotate the candidate table when called from run_search_stages to mark the stars that could
        be selected in a stage.  As candidate stars are "rejected" in the stage, details
        of that rejection are also added to self.reject_info .

        Additionally, candidate stars that do not pass a check in the stage get their error
        bitmask in 'stat_{n_stage}' marked up with the error.

        Candidates are not removed from the tble in this routine; the length of the candidate
        table should not change after get_initial_guide_candidates (called before this routine).
        Instead, as mentioned, this routine marks up informational columns in the table and returns
        a mask of the "ok" stars in the stage.

        **Details**:

        In each stage, the dictionary of characteristics for the stage is used to run the checks to
        see which candidates would be allowed. For example, the Stage 1 dictionary looks like::

          {"Stage": 1,
            "SigErrMultiplier": 3,
            "ASPQ1Lim": 0,
            "MagLimit": [5.6, 10.2],
            "DoBminusVcheck": 1,
            "Spoiler": {
             "BgPixThresh": 25,
             "RegionFrac": .05,
             },
            "Imposter": {
             "CentroidOffsetLim": .2,
             }}

        In each stage these checks are run:

        1. Candidate star magnitude is checked to see if it is within the range plus error.
        For stage 1 for example , each candidate star is checked to see if it is within the
        5.6 to 10.2 mag range, minus padding for error (SigErrMultiplier * cand['mag_err']).

        2. Candidates with ASPQ1 > stage ASPQ1Lim are marked for exclusion.

        3. Candidates with local dark current that suggests that the candidate centroid could
        be shifted more that CentroidOffsetLim (in arcsecs) are marked for exclusion.  The value of
        the local dark current was previously saved during "get_initial_guide_candidates" as
        "imp_mag".

        4. check_mag_spoilers is used to mark candidate stars that have spoilers that are too
        close for the magnitude delta to the candidate as defined by the "line" used by that check.
        The "line" basically works as a check that requires a minimum position separation for
        a delta magnitude.  A candidate star with a spoiler of the same magnitude would require
        that spoiler star to be Intercept (9 pixels) away.  A candidate star with a spoiler with
        no separation would require the spoiler to be 18 mags fainter::

          (9 pixels +
          (cand['mag'] - spoiler['mag'] + SigErrMultiplier * mag_err_sum) * 0.5 pix / dmag)

        The parameters of the check are not stage dependent, but the mag err there is defined as
        SigErrMultiplier * mag_err_sum (MAG_ACA_ERR of candidate and spoiler added in quadrature).

        5. check_spoil_contrib is used to mark candidate stars that have spoilers that contribute
        too much light onto either the the 8x8 image window or to the 8 background pixels.  "Too
        much" is defined by the Spoiler parameters of the stage, where in stage 1 a candidate
        star will be excluded if either:

           - the sum of the light from spoiler stars in the 8x8 contribute more than
              (candidate star magnitude in counts) * (RegionFrac).

           - the sum of the light from spoiler stars on any background pixel is greater
             that the BgPixThresh percentile of the current dark current.

        6. check_column_spoilers marks candidates for exclusion if they have column spoilers
        defined as spoilers between the candidate and the readout register, within
        characterstics.col_spoiler_pix_sep (10 columns), and 4.5 mags brighter (minus error)
        than the candidate.  The error is the stage-dependent term where it is set as
        SigErrMultiplier * (the candidate MAG_ACA_ERR and the spoiler MAG_ACA_ERR added in
        quadrature).

        7. Candidates are then screened for "bad" color (COLOR1 == 0.700) if DoBMinusVCheck
        is set for the stage.

        :param stage: dictionary of search stage parameters
        :returns: bool mask of the length of self.meta.cand_guides with true set for "ok" stars
        """

        cand_guides = self.cand_guides
        stars = self.stars
        dark = self.dark
        ok = np.ones(len(cand_guides)).astype(bool)

        # Adopt the SAUSAGE convention of a bit array for errors
        # Not all items will be checked for each star (allow short circuit)
        scol = "stat_{}".format(stage["Stage"])
        cand_guides[scol] = 0

        n_sigma = stage["SigErrMultiplier"]

        # And for bright stars, use a local mag_err that is lower bounded at 0.1
        # for the mag selection.
        mag_err = cand_guides["mag_err"]
        bright = cand_guides["mag"] < 7.0
        mag_err[bright] = mag_err[bright].clip(0.1)
        # Also set any color=0.7 stars to have lower bound mag err of 0.5
        bad_color = np.isclose(cand_guides["COLOR1"], 0.7)
        mag_err[bad_color] = mag_err[bad_color].clip(0.5)

        # Check reasonable mag
        bright_lim = stage["MagLimit"][0]
        faint_lim = stage["MagLimit"][1]
        # Confirm that the star mag is not outside the limits when padded by error.
        # For the bright end of the check, set a lower bound to always use at least 1
        # mag_err, but do not bother with this bound at the faint end of the check.
        # Also explicitly confirm that the star is not within 2 * mag_err of the hard
        # bright limit (which is basically 5.2, but if bright lim set to less than 5.2
        # in the stage, take that).
        bad_mag = (
            ((cand_guides["mag"] - max(n_sigma, 1) * mag_err) < bright_lim)
            | ((cand_guides["mag"] + n_sigma * mag_err) > faint_lim)
            | ((cand_guides["mag"] - 2 * mag_err) < min(bright_lim, 5.2))
        )
        for idx in np.flatnonzero(bad_mag):
            self.reject(
                {
                    "id": cand_guides["id"][idx],
                    "type": "mag outside range",
                    "stage": stage["Stage"],
                    "bright_lim": bright_lim,
                    "faint_lim": faint_lim,
                    "cand_mag": cand_guides["mag"][idx],
                    "cand_mag_err_times_sigma": n_sigma * mag_err[idx],
                    "text": (
                        f"Cand {cand_guides['id'][idx]} rejected with "
                        "mag outside range for stage"
                    ),
                }
            )
        cand_guides[scol][bad_mag] += GUIDE.errs["mag range"]
        ok = ok & ~bad_mag

        # Check stage ASPQ1
        bad_aspq1 = cand_guides["ASPQ1"] > stage["ASPQ1Lim"]
        for idx in np.flatnonzero(bad_aspq1):
            self.reject(
                {
                    "id": cand_guides["id"][idx],
                    "type": "aspq1 outside range",
                    "stage": stage["Stage"],
                    "aspq1_lim": stage["ASPQ1Lim"],
                    "text": (
                        f"Cand {cand_guides['id'][idx]} rejected with "
                        f"aspq1 > {stage['ASPQ1Lim']}"
                    ),
                }
            )
        cand_guides[scol][bad_aspq1] += GUIDE.errs["aspq1"]
        ok = ok & ~bad_aspq1

        # Check for bright pixels
        pixmag_lims = get_pixmag_for_offset(
            cand_guides["mag"], stage["Imposter"]["CentroidOffsetLim"]
        )
        # Which candidates have an 'imposter' brighter than the limit for this stage
        imp_spoil = cand_guides["imp_mag"] < pixmag_lims
        for idx in np.flatnonzero(imp_spoil):
            cand = cand_guides[idx]
            cen_limit = stage["Imposter"]["CentroidOffsetLim"]
            self.reject(
                {
                    "id": cand["id"],
                    "stage": stage["Stage"],
                    "type": "hot pixel",
                    "centroid_offset_thresh": cen_limit,
                    "pseudo_mag_for_thresh": pixmag_lims[idx],
                    "imposter_mag": cand["imp_mag"],
                    "imp_row_start": cand["imp_r"],
                    "imp_col_start": cand["imp_c"],
                    "text": (
                        f"Cand {cand['id']} mag {cand['mag']:.1f} imposter with "
                        f'"mag" {cand["imp_mag"]:.1f} '
                        f"limit {pixmag_lims[idx]:.2f} with offset lim {cen_limit} at stage"
                    ),
                }
            )
        cand_guides[scol][imp_spoil] += GUIDE.errs["hot pix"]
        ok = ok & ~imp_spoil

        # Check for 'direct catalog search' spoilers
        mag_spoil, mag_rej = check_mag_spoilers(cand_guides, ok, stars, n_sigma)
        for rej in mag_rej:
            rej["stage"] = stage["Stage"]
            self.reject(rej)
        cand_guides[scol][mag_spoil] += GUIDE.errs["spoiler (line)"]
        ok = ok & ~mag_spoil

        # Check for star spoilers (by light) background and edge
        if stage["ASPQ1Lim"] > 0:
            bg_pix_thresh = np.percentile(dark, stage["Spoiler"]["BgPixThresh"])
            reg_frac = stage["Spoiler"]["RegionFrac"]
            bg_spoil, reg_spoil, light_rej = check_spoil_contrib(
                cand_guides, ok, stars, reg_frac, bg_pix_thresh
            )
            for rej in light_rej:
                rej["stage"] = stage["Stage"]
                self.reject(rej)
            cand_guides[scol][bg_spoil] += GUIDE.errs["spoiler (bgd)"]
            cand_guides[scol][reg_spoil] += GUIDE.errs["spoiler (frac)"]
            ok = ok & ~bg_spoil & ~reg_spoil

        # Check for column spoiler
        col_spoil, col_rej = check_column_spoilers(cand_guides, ok, stars, n_sigma)
        for rej in col_rej:
            rej["stage"] = stage["Stage"]
            self.reject(rej)
        cand_guides[scol][col_spoil] += GUIDE.errs["col spoiler"]
        ok = ok & ~col_spoil

        if stage["DoBminusVcheck"] == 1:
            bad_color = np.isclose(cand_guides["COLOR1"], 0.7, atol=1e-6, rtol=0)
            for idx in np.flatnonzero(bad_color):
                self.reject(
                    {
                        "id": cand_guides["id"][idx],
                        "type": "bad color",
                        "stage": stage["Stage"],
                        "text": f"Cand {cand_guides['id'][idx]} has bad color (0.7)",
                    }
                )
            cand_guides[scol][bad_color] += GUIDE.errs["bad color"]
            ok = ok & ~bad_color
        return ok

    def process_include_ids(self, cand_guides, stars):
        """Ensure that the cand_guides table has stars that were forced to be included.

        :param cand_guides: candidate guide stars table
        :param stars: stars table

        """
        row_max = CCD["row_max"] - CCD["row_pad"] - CCD["window_pad"]
        col_max = CCD["col_max"] - CCD["col_pad"] - CCD["window_pad"]

        ok = (np.abs(stars["row"]) < row_max) & (np.abs(stars["col"]) < col_max)

        super().process_include_ids(cand_guides, stars[ok])

    def get_candidates_mask(self, stars):
        """Get base filter for acceptable candidates.

        This does not include spatial filtering.

        :param stars: StarsTable
        :returns: bool mask of acceptable stars

        """
        # If dyn_bgd is active then apply the T_ccd bonus to the effective
        # CCD temperature. This will bring in fainter stars to the candidates.
        t_ccd = self.t_ccd
        if self.dyn_bgd_n_faint > 0:
            t_ccd += self.dyn_bgd_dt_ccd

        dyn_bgd = self.dyn_bgd_n_faint > 0
        return get_guide_candidates_mask(stars, t_ccd, dyn_bgd)

    def get_initial_guide_candidates(self):
        """
        Create a candidate list from the available stars in the field.

        As the initial list of candidates is created this:

        1. Runs get_candidate_mask to limit the "ok" mask to the list of candidates to only those
        that at least mee the minimum filter (based fields of the AGASC).

        2. Sets the "ok" mask to limit to stars that are on the CCD (with minimum edge padding
        but no dither padding)

        3. Adds a column ('offchip') to the table of all stars to mark those that aren't on the
        CCD at all (offchip = True).

        4. Makes one more position filter of the possible candidate stars that limits to those
        stars that are on the CCD with edge minimum padding, readout window padding, and padding
        for dither.

        5. Filters the supplied star table (self.stars) using the filters in 1), 2) and 4), to
        produce a candidate list.

        6. Filters the candidate list to remove any that are spoiled by bad pixel.

        7. Filters the candidate list to remove any that are in the bad stars list.

        8. Filters the candidates to remove any that have a spoiler in a NxN box centered
        on the star brighter than M magnitudes fainter than the candidate (N is 5 and M is -4
        from the box_spoiler section of guide characteristics).  This check uses the
        `has_spoiler_in_box` method and is the first of many spoiler checks.  This spoiler
        check is not stage dependent.

        9. Filters the candidates to remova any that are spoiled by the fid trap (using
        `check_fid_trap` method).

        10. Puts any force include candidates from the include_ids parameter back in the candidate
        list if they were filtered out by an earlier filter in this routine.

        11. Filters/removes any candidates that are force excluded (in exclude_ids).

        12. Uses the local dark current around each candidate to calculate an "imposter mag"
        describing the brightest 2x2 in the region the star would dither over.  This is
        saved to the candidate star table.


        """
        stars = self.stars
        dark = self.dark

        # Mark stars that are off chip
        offchip = (np.abs(stars["row"]) > CCD["row_max"]) | (
            np.abs(stars["col"]) > CCD["col_max"]
        )
        stars["offchip"] = offchip

        # Add a filter for stars that are too close to the chip edge including dither
        r_dith_pad = self.dither.row
        c_dith_pad = self.dither.col
        row_max = CCD["row_max"] - (
            CCD["row_pad"] + CCD["window_pad"] + CCD["guide_extra_pad"] + r_dith_pad
        )
        col_max = CCD["col_max"] - (
            CCD["col_pad"] + CCD["window_pad"] + CCD["guide_extra_pad"] + c_dith_pad
        )
        outofbounds = (np.abs(stars["row"]) > row_max) | (
            np.abs(stars["col"]) > col_max
        )

        # Set the candidates to be the set of stars that is both *not* out of bounds
        # and satisfies the rules in 'get_candidates_mask' (which uses the primary
        # selection filter from acq, but allows bad color and limits to brighter stars).
        ok = self.get_candidates_mask(stars) & ~outofbounds
        cand_guides = stars[ok]
        self.log("Filtering on CLASS, mag, row/col, mag_err, ASPQ1/2, POS_ERR:")
        self.log(
            f"Reduced star list from {len(stars)} to "
            f"{len(cand_guides)} candidate guide stars"
        )

        bp, bp_rej = spoiled_by_bad_pixel(cand_guides, self.dither)
        for rej in bp_rej:
            rej["stage"] = 0
            self.reject(rej)
        cand_guides = cand_guides[~bp]
        self.log("Filtering on candidates near bad (not just bright/hot) pixels")
        self.log(
            f"Reduced star list from {len(bp)} to "
            f"{len(cand_guides)} candidate guide stars"
        )

        bs = self.in_bad_star_list(cand_guides)
        for idx in np.flatnonzero(bs):
            self.reject(
                {
                    "id": cand_guides["id"][idx],
                    "stage": 0,
                    "type": "bad star list",
                    "text": f"Cand {cand_guides['id'][idx]} in bad star list",
                }
            )
        cand_guides = cand_guides[~bs]
        self.log("Filtering stars on bad star list")
        self.log(
            f"Reduced star list from {len(bs)} to "
            f"{len(cand_guides)} candidate guide stars"
        )

        box_spoiled, box_rej = has_spoiler_in_box(
            cand_guides,
            stars,
            halfbox=GUIDE.box_spoiler["halfbox"],
            magdiff=GUIDE.box_spoiler["magdiff"],
        )
        for rej in box_rej:
            rej["stage"] = 0
            self.reject(rej)
        cand_guides = cand_guides[~box_spoiled]
        self.log("Filtering stars that have bright spoilers with centroids near/in 8x8")
        self.log(
            f"Reduced star list from {len(box_spoiled)} to "
            f"{len(cand_guides)} candidate guide stars"
        )

        fid_trap_spoilers, fid_rej = check_fid_trap(
            cand_guides, fids=self.fids, dither=self.dither
        )
        for rej in fid_rej:
            rej["stage"] = 0
            self.reject(rej)
        cand_guides = cand_guides[~fid_trap_spoilers]

        if len(self.jupiter) > 0:
            from proseco.jupiter import check_spoiled_by_jupiter

            # Exclude candidates within 15 columns of Jupiter
            spoiled_by_jupiter, jupiter_rej = check_spoiled_by_jupiter(
                cand_guides, self.jupiter
            )
            for rej in jupiter_rej:
                rej["stage"] = 0
                self.reject(rej)
            cand_guides = cand_guides[~spoiled_by_jupiter]

        # Deal with include_ids by putting them back in candidate table if necessary
        self.process_include_ids(cand_guides, stars)

        # Deal with exclude_ids by cutting from the candidate list
        for star_id in self.exclude_ids:
            if star_id in cand_guides["id"]:
                self.reject(
                    {
                        "stage": 0,
                        "type": "exclude_id",
                        "id": star_id,
                        "text": f"Cand {star_id} rejected.  In exclude_ids",
                    }
                )
                cand_guides = cand_guides[cand_guides["id"] != star_id]

        # Get the brightest 2x2 in the dark map for each candidate and save value and location
        imp_mag, imp_row, imp_col = get_imposter_mags(cand_guides, dark, self.dither)
        cand_guides["imp_mag"] = imp_mag
        cand_guides["imp_r"] = imp_row
        cand_guides["imp_c"] = imp_col
        self.log("Getting pseudo-mag of brightest pixel 2x2 in candidate region")

        return cand_guides

    def in_bad_star_list(self, cand_guides):
        """
        Mark star bad if candidate AGASC ID in bad star list.

        :param cand_guides: Table of candidate stars
        :returns: boolean mask where True means star is in bad star list
        """
        bad = np.in1d(cand_guides["id"], list(ACA.bad_star_set))

        # Set any matching bad stars as bad for plotting
        for bad_id in cand_guides["id"][bad]:
            idx = self.stars.get_id_idx(bad_id)
            self.bad_stars_mask[idx] = True

        return bad


def check_fid_trap(cand_stars, fids, dither):
    """
    Search for guide stars that would cause the fid trap issue and mark as spoilers.

    :param cand_stars: candidate star Table
    :param fids: fid Table
    :param dither: dither ACABox
    :returns: mask on cand_stars of fid trap spoiled stars, list of rejection info dicts
    """

    spoilers = np.zeros(len(cand_stars)).astype(bool)
    rej = []

    if fids is None or len(fids) == 0:
        return spoilers, []

    bad_row = GUIDE.fid_trap["row"]
    bad_col = GUIDE.fid_trap["col"]
    fid_margin = GUIDE.fid_trap["margin"]

    # Check to see if the fid is in the zone that's a problem for the trap and if there's
    # a star that can cause the effect in the readout regiser
    for fid in fids:
        incol = abs(fid["col"] - bad_col) < fid_margin
        inrow = fid["row"] < 0 and fid["row"] > bad_row
        if incol and inrow:
            fid_dist_to_trap = fid["row"] - bad_row
            star_dist_to_register = 512 - abs(cand_stars["row"])
            spoils = abs(fid_dist_to_trap - star_dist_to_register) < (
                fid_margin + dither.row
            )
            spoilers = spoilers | spoils
            for idx in np.flatnonzero(spoils):
                cand = cand_stars[idx]
                rej.append(
                    {
                        "id": cand["id"],
                        "type": "fid trap effect",
                        "fid_id": fid["id"],
                        "fid_dist_to_trap": fid_dist_to_trap,
                        "star_dist_to_register": star_dist_to_register[idx],
                        "text": f"Cand {cand['id']} in trap zone for fid {fid['id']}",
                    }
                )
    return spoilers, rej


def check_spoil_contrib(cand_stars, ok, stars, regfrac, bgthresh):
    """
    Check that there are no spoiler stars contributing more than a fraction
    of the candidate star to the candidate star's 8x8 pixel region or more than bgthresh
    to any of the background pixels.

    :param cand_stars: candidate star Table
    :param ok: mask on cand_stars of candidates that are still 'ok'
    :param stars: Table of agasc stars for this field
    :param regfrac: fraction of candidate star mag that may fall on the 8x8 due to spoilers
                    A sum above this fraction will mark the cand_star as spoiled
    :param bgthresh: background pixel threshold (in e-/sec).  If spoilers contribute more
                    than this value to any background pixel, mark the cand_star as spoiled.
    :returns: reg_spoiled, bg_spoiled, rej - two masks on cand_stars and a list of reject
              debug dicts
    """
    fraction = regfrac
    bg_spoiled = np.zeros_like(ok)
    reg_spoiled = np.zeros_like(ok)
    bgpix = CCD["bgpix"]
    rej = []
    for cand in cand_stars[ok]:
        if cand["ASPQ1"] == 0:
            continue
        spoilers = (np.abs(cand["row"] - stars["row"]) < 9) & (
            np.abs(cand["col"] - stars["col"]) < 9
        )

        # If there is only one match, it is the candidate so there's nothing to do
        if np.count_nonzero(spoilers) == 1:
            continue
        cand_counts = mag_to_count_rate(cand["mag"])

        # Get a reasonable AcaImage for the location of the 8x8 for the candidate
        cand_img_region = ACAImage(
            np.zeros((8, 8)),
            row0=int(round(cand["row"])) - 4,
            col0=int(round(cand["col"])) - 4,
        )
        on_region = cand_img_region
        for spoil in stars[spoilers]:
            if spoil["id"] == cand["id"]:
                continue
            spoil_img = APL.get_psf_image(
                row=spoil["row"],
                col=spoil["col"],
                pix_zero_loc="edge",
                norm=mag_to_count_rate(spoil["mag"]),
            )
            on_region = on_region + spoil_img.aca

        # Consider it spoiled if the star contribution on the 8x8 is over a fraction
        frac_limit = cand_counts * fraction
        sum_in_region = np.sum(on_region)
        if sum_in_region > frac_limit:
            reg_spoiled[cand_stars["id"] == cand["id"]] = True
            rej.append(
                {
                    "id": cand_stars["id"],
                    "type": "region sum spoiled",
                    "limit_for_star": frac_limit,
                    "fraction": fraction,
                    "sum_in_region": sum_in_region,
                    "text": (
                        f"Cand {cand_stars['id']} has too much contribution "
                        "to region from spoilers"
                    ),
                }
            )
            continue

        # Or consider it spoiled if the star contribution to any background pixel
        # is more than the Nth percentile of the dark current
        for pixlabel in bgpix:
            val = on_region[pixlabel == chandra_aca.aca_image.EIGHT_LABELS][0]
            if val > bgthresh:
                bg_spoiled[cand_stars["id"] == cand["id"]] = True
                rej.append(
                    {
                        "id": cand["id"],
                        "type": "region background spoiled",
                        "bg_thresh": bgthresh,
                        "bg_pix_val": val,
                        "pix_id": pixlabel,
                        "text": f"Cand {cand['id']} has bg pix spoiled by spoilers",
                    }
                )
                continue

    return bg_spoiled, reg_spoiled, rej


def check_mag_spoilers(cand_stars, ok, stars, n_sigma):
    """
    Use the slope-intercept mag-spoiler relationship to exclude all
    stars that have a "mag spoiler".  This is basically equivalent to the
    "direct catalog search" for spoilers in SAUSAGE, but does not forbid
    all stars within 7 pixels (spoilers must be faint to be close).

    The n-sigma changes by stage for the mags/magerrs used in the check.

    :param cand_stars: Table of candidate stars
    :param ok: mask on cand_stars describing those that are still "ok"
    :param stars: Table of AGASC stars in this field
    :param n_sigma: multiplier use for MAG_ACA_ERR when reviewing spoilers
    :returns: bool mask of length cand_stars marking mag_spoiled stars, list of reject debug dicts
    """
    intercept = GUIDE.mag_spoiler["Intercept"]
    spoilslope = GUIDE.mag_spoiler["Slope"]
    magdifflim = GUIDE.mag_spoiler["MagDiffLimit"]

    # If there are already no candidates, there isn't anything to do
    if not np.any(ok):
        return np.zeros(len(ok)).astype(bool), []

    mag_spoiled = np.zeros(len(ok)).astype(bool)
    rej = []
    cand_idxs = np.flatnonzero(ok)

    for cand_idx in cand_idxs:
        cand = cand_stars[cand_idx]
        spoil_idxs = np.flatnonzero(
            (np.abs(cand["row"] - stars["row"]) < 10)
            & (np.abs(cand["col"] - stars["col"]) < 10)
        )

        # If there is only one match, it is the candidate so there's nothing to do
        if len(spoil_idxs) == 1:
            continue

        for spoil_idx in spoil_idxs:
            spoil = stars[spoil_idx]
            if spoil["id"] == cand["id"]:
                continue
            if (cand["mag"] - spoil["mag"]) < magdifflim:
                continue
            mag_err_sum = np.sqrt(
                (cand["MAG_ACA_ERR"] * 0.01) ** 2 + (spoil["MAG_ACA_ERR"] * 0.01) ** 2
            )
            delmag = cand["mag"] - spoil["mag"] + n_sigma * mag_err_sum
            thsep = intercept + delmag * spoilslope
            dist = np.sqrt(
                ((cand["row"] - spoil["row"]) ** 2)
                + ((cand["col"] - spoil["col"]) ** 2)
            )
            if dist < thsep:
                rej.append(
                    {
                        "id": cand["id"],
                        "spoiler": spoil["id"],
                        "spoiler_mag": spoil["mag"],
                        "dmag_with_err": delmag,
                        "min_dist_for_dmag": thsep,
                        "actual_dist": dist,
                        "type": "spoiler by distance-mag line",
                        "text": (
                            f"Cand {cand['id']} spoiled by {spoil['id']}, "
                            f"too close ({dist:.1f}) pix for magdiff ({delmag:.1f})"
                        ),
                    }
                )
                mag_spoiled[cand["id"] == cand_stars["id"]] = True
                continue

    return mag_spoiled, rej


def check_column_spoilers(cand_stars, ok, stars, n_sigma):
    """
    For each candidate, check for stars 'MagDiff' brighter and within 'Separation' columns
    between the star and the readout register, i.e. Column Spoilers.

    :param cand_stars: Table of candidate stars
    :param ok: mask on cand_stars describing still "ok" candidates
    :param stars: Table of AGASC stars
    :param n_sigma: multiplier used when checking mag with MAG_ACA_ERR
    :returns: bool mask on cand_stars marking column spoiled stars, list of debug reject dicts
    """
    column_spoiled = np.zeros_like(ok)
    rej = []
    for idx, cand in enumerate(cand_stars):
        if not ok[idx]:
            continue

        # Get possible column spoiling stars by position that are that are
        # on the same side of the CCD as the candidate
        # AND between the candidate star and readout register
        # AND in the column "band" for the candidate
        pos_spoil = (
            (np.sign(cand["row"]) == np.sign(stars["row"][~stars["offchip"]]))
            & (np.abs(cand["row"]) < np.abs(stars["row"][~stars["offchip"]]))
            & (
                np.abs(cand["col"] - stars["col"][~stars["offchip"]])
                <= ACA.col_spoiler_pix_sep
            )
        )
        if not np.count_nonzero(pos_spoil) >= 1:
            continue

        mag_errs = n_sigma * np.sqrt(
            (cand["MAG_ACA_ERR"] * 0.01) ** 2
            + (stars["MAG_ACA_ERR"][~stars["offchip"]][pos_spoil] * 0.01) ** 2
        )
        dm = cand["mag"] - stars["mag"][~stars["offchip"]][pos_spoil] + mag_errs
        spoils = dm > ACA.col_spoiler_mag_diff
        if np.any(spoils):
            column_spoiled[idx] = True
            spoiler = stars[~stars["offchip"]][pos_spoil][spoils][0]
            rej.append(
                {
                    "id": cand["id"],
                    "type": "column spoiled",
                    "spoiler": spoiler["id"],
                    "spoiler_mag": spoiler["mag"],
                    "dmag_with_err": dm[spoils][0],
                    "dmag_lim": ACA.col_spoiler_mag_diff,
                    "dcol": cand["col"] - spoiler["col"],
                    "text": (
                        f"Cand {cand['id']} has column spoiler {spoiler['id']} "
                        f"at ({spoiler['row']:.1f}, {spoiler['row']:.1f}), "
                        f"mag {spoiler['mag']:.2f}"
                    ),
                }
            )
    return column_spoiled, rej


def get_ax_range(rc, extent):
    """
    Given a float pixel row or col value and an "extent" in float pixels,
    generally 4 + 1.6 for 8" dither and 4 + 5.0 for 20" dither,
    return a range for the row or col that is divisible by 2 and contains
    at least the requested extent.

    :param rc: row or col float value (edge pixel coords)
    :param extent: half of desired range from n (should include pixel dither)
    :returns: tuple of range as (minus, plus)
    """
    minus = int(np.floor(rc - extent))
    plus = int(np.ceil(rc + extent))
    # If there isn't an even range of pixels, add or subtract one from the range
    if (plus - minus) % 2 != 0:
        # If the "rc" value in on the 'right' side of a pixel, add one to the plus
        if rc - np.floor(rc) > 0.5:
            plus += 1
        # Otherwise subtract one from the minus
        else:
            minus -= 1
    return minus, plus


def get_imposter_mags(cand_stars, dark, dither):
    """
    Get "pseudo-mag" of max pixel value in each candidate star region

    :param cand_stars: Table of candidate stars
    :param dark: full CCD dark map
    :param dither: observation dither to be used to determine pixels a star could use
    :returns: np.array pixmags, np.array pix_r, np.array pix_c all of length cand_stars
    """

    pixmags = []
    pix_r = []
    pix_c = []

    # Define the 1/2 pixel region as half the 8x8 plus a pad plus dither
    row_extent = 4 + GUIDE.dither_pix_pad + dither.row
    col_extent = 4 + GUIDE.dither_pix_pad + dither.col
    for cand in cand_stars:
        rminus, rplus = get_ax_range(cand["row"], row_extent)
        cminus, cplus = get_ax_range(cand["col"], col_extent)
        pix = np.array(dark[rminus + 512 : rplus + 512, cminus + 512 : cplus + 512])
        pixmax = 0
        max_r = None
        max_c = None
        # Check the 2x2 bins for the max 2x2 region.  Search the "offset" versions as well
        for pix_chunk, row_off, col_off in zip(
            (pix, pix[1:-1, :], pix[:, 1:-1], pix[1:-1, 1:-1]),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        ):
            bin_image = bin2x2(pix_chunk)
            pixsum = np.max(bin_image)
            if pixsum > pixmax:
                pixmax = pixsum
                idx = np.unravel_index(np.argmax(bin_image), bin_image.shape)
                max_r = rminus + row_off + idx[0] * 2
                max_c = cminus + col_off + idx[1] * 2
        # Get the mag equivalent to pixmax.  If pixmax is zero (for a synthetic dark map)
        # clip lower bound at 1.0 to avoid 'inf' mag and warnings from chandra_aca.transform
        pixmax_mag = count_rate_to_mag(np.clip(pixmax, 1.0, None))
        pixmags.append(pixmax_mag)
        pix_r.append(max_r)
        pix_c.append(max_c)
    return np.array(pixmags), np.array(pix_r), np.array(pix_c)


def get_pixmag_for_offset(cand_mag, offset):
    """
    Determine the magnitude an individual bad pixel would need to spoil the centroid of
    the candidate star by ``offset``.  This just constructs the worst case as the bad pixel
    3 pixels away (edge of box).  The offset in this case would be
    offset = spoil_cnts * 3 * 5 / (spoil_cnts + cand_counts), where 3 is the distance to
    the edge pixel in pixels and 5 is the conversion to arcsec.
    Solving for spoil_cnts gives:
    spoil_cnts = mag_to_count_rate(cand_mag) * (offset / (15 - offset))

    :param cand_mag: candidate star magnitude
    :param offset: centroid offset in arcsec
    :returns: 'magnitude' of bad pixel needed to for offset
    """
    spoil_cnts = mag_to_count_rate(cand_mag) * (offset / (15 - offset))
    return count_rate_to_mag(spoil_cnts)


def has_spoiler_in_box(cand_guides, stars, halfbox=5, magdiff=-4):
    """
    Check each candidate star for spoilers that would fall in a box centered on the star.
    Mark candidate spoiled if there's a spoiler in the box and brighter than magdiff fainter
    than the candidate's mag.

    :param cand_guides: Table of candidate stars
    :param stars: Table of AGASC stars in the field
    :param halfbox: half of the length of a side of the box used for check (pixels)
    :param magdiff: magnitude difference threshold
    :returns: mask on cand_guides set to True if star spoiled by another star in the pixel box,
              and a list of dicts with reject debug info
    """
    box_spoiled = np.zeros(len(cand_guides)).astype(bool)
    rej = []
    for idx, cand in enumerate(cand_guides):
        dr = np.abs(cand["row"] - stars["row"])
        dc = np.abs(cand["col"] - stars["col"])
        inbox = (dr <= halfbox) & (dc <= halfbox)
        itself = stars["id"] == cand["id"]
        box_spoilers = ~itself & inbox & (cand["mag"] - stars["mag"] > magdiff)
        if np.any(box_spoilers):
            box_spoiled[idx] = True
            n = np.count_nonzero(box_spoilers)
            boxsize = halfbox * 2
            bright = np.argmin(stars[box_spoilers]["mag"])
            spoiler = stars[box_spoilers][bright]
            rej.append(
                {
                    "id": cand["id"],
                    "type": "in-box spoiler star",
                    "boxsize": boxsize,
                    "magdiff_thresh": magdiff,
                    "spoiler": spoiler["id"],
                    "dmag": cand["mag"] - spoiler["mag"],
                    "n": n,
                    "text": (
                        f"Cand {cand['id']} spoiled by {n} stars in {boxsize}x{boxsize} "
                        f" including {spoiler['id']}"
                    ),
                }
            )
    return box_spoiled, rej


def spoiled_by_bad_pixel(cand_guides, dither):
    """
    Mark star bad if spoiled by a bad pixel in the bad pixel list (not hot)

    :param cand_guides: Table of candidate stars
    :param dither: dither ACABox
    :returns: boolean mask on cand_guides where True means star is spoiled by bad pixel,
              list of dicts of reject debug info
    """

    raw_bp = np.array(ACA.bad_pixels)
    bp_row = []
    bp_col = []

    # Bad pixel entries are [row_min, row_max, col_min, col_max]
    # Convert this to lists of the row and col coords of the bad pixels
    for row in raw_bp:
        for rr in range(row[0], row[1] + 1):
            for cc in range(row[2], row[3] + 1):
                bp_row.append(rr)
                bp_col.append(cc)
    bp_row = np.array(bp_row)
    bp_col = np.array(bp_col)

    # Then for the pixel region of each candidate, see if there is a bad
    # pixel in the region.
    spoiled = np.zeros(len(cand_guides)).astype(bool)
    # Also save an array of rejects to pass back
    rej = []
    row_extent = np.ceil(4 + dither.row)
    col_extent = np.ceil(4 + dither.col)
    for idx, cand in enumerate(cand_guides):
        rminus = int(np.floor(cand["row"] - row_extent))
        rplus = int(np.ceil(cand["row"] + row_extent + 1))
        cminus = int(np.floor(cand["col"] - col_extent))
        cplus = int(np.ceil(cand["col"] + col_extent + 1))

        # If any bad pixel is in the guide star pixel region, mark as spoiled
        bps = (
            (bp_row >= rminus)
            & (bp_row <= rplus)
            & (bp_col >= cminus)
            & (bp_col <= cplus)
        )
        if np.any(bps):
            spoiled[idx] = True
            rej.append(
                {
                    "id": cand["id"],
                    "type": "bad pixel",
                    "pixel": (bp_row[bps][0], bp_col[bps][0]),
                    "n_bad": np.count_nonzero(bps),
                    "text": (
                        f"Cand {cand['id']} spoiled by {np.count_nonzero(bps)} bad pixels "
                        f"including {(bp_row[bps][0], bp_col[bps][0])}"
                    ),
                }
            )
    return spoiled, rej


def dist2(g1, g2):
    """
    Calculate squared distance between a pair of stars in a star table.

    This local version of the method uses a cache.
    """
    if (g1["id"], g2["id"]) in STAR_PAIR_DIST_CACHE:
        return STAR_PAIR_DIST_CACHE[(g1["id"], g2["id"])]
    out = (g1["yang"] - g2["yang"]) ** 2 + (g1["zang"] - g2["zang"]) ** 2
    STAR_PAIR_DIST_CACHE[(g1["id"], g2["id"])] = out
    return out


def check_single_cluster(cand_guide_set, threshold, n_minus):
    """
    Confirm that a set of stars satisfies the minimum separation threshold when n_minus
    stars are removed from the set.  For example, for a threshold of 1000, n_minus of 1,
    and an input set of candidates stars with 5 candidates, confirm that, for any 4 of
    the 5 stars (5 minus n_minus = 1), one pair of those stars is separated by at
    least 1000 arcsecs.

    :returns: bool (True for check passing threshold)
    """
    min_dist = threshold
    min_dist2 = min_dist**2
    guide_idxs = np.arange(len(cand_guide_set))
    n_guide = len(guide_idxs)
    for idxs in combinations(guide_idxs, n_guide - n_minus):
        for idx0, idx1 in combinations(idxs, 2):
            if dist2(cand_guide_set[idx0], cand_guide_set[idx1]) > min_dist2:
                break
        else:
            return False
    return True


def run_cluster_checks(cand_guide_set):
    """
    Perform cluster checks on a combination of candidate guide stars.

    This performs a set of specific cluster checks on the provided cand_guide_set
    and returns a list of bools, one for each cluster check run.

    :returns: list of bools, one for each cluster check run
    """
    test_status = []
    for n_minus, threshold in enumerate(GUIDE.cluster_thresholds):
        if n_minus < len(cand_guide_set) + 1:
            test_status.append(
                check_single_cluster(
                    cand_guide_set, threshold=threshold, n_minus=n_minus
                )
            )
        else:
            test_status.append(False)

    return test_status
