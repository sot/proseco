# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Get a catalog of fid lights.
"""

import os
import weakref

import numpy as np
from chandra_aca.drift import get_fid_offset
from chandra_aca.transform import yagzag_to_pixels
from cxotime import CxoTimeLike

from . import characteristics as ACA
from . import characteristics_acq as ACQ
from . import characteristics_fid as FID
from . import guide
from .core import ACACatalogTable, AliasAttribute, MetaAttribute
from .jupiter import is_spoiled_by_jupiter


def get_fid_catalog(obsid=0, **kwargs):
    """
    Get a catalog of fid lights.

    This is the initial selection and only returns a catalog that is "perfect":

    - No fids are spoiled by a star
    - No fids spoil an acq star (via ``acqs``)

    If no such fid catalog is available this function returns a zero-length
    FidTable.  In this case a subsequent concurrent optimization of fid lights
    and acq stars is peformed.

    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param focus_offset: SIM focus offset [steps] (default=0)
    :param sim_offset: SIM translation offset from nominal [steps] (default=0)
    :param acqs: AcqTable catalog.  Optional but needed for actual fid selection.
    :param guide_cands: GuideTable of initial guide candidates (used for star spoilers)
    :param stars: stars table.  Defaults to acqs.stars if available.
    :param dither_acq: acq dither size (2-element sequence (y, z), arcsec)
    :param dither_guide: guide dither size (2-element sequence (y, z), arcsec)
    :param include_ids: fid ids to force include. If no possible sets of fids include
                        the id (aka index), no fids will be selected.
    :param exclude_ids: fid ids to exclude
    :param n_fid: number of desired fid lights
    :param print_log: print log to stdout (default=False)

    :returns: fid catalog (FidTable)
    """
    # If no fids are requested then just initialize an empty table
    # here, set the attributes and return the table.  No need to go
    # through the trouble of getting candidate fids.
    fids = FidTable()
    fids.set_attrs_from_kwargs(obsid=obsid, **kwargs)

    if fids.n_fid == 0:
        empty_fids = FidTable.empty()
        empty_fids.meta = fids.meta
        return empty_fids

    fids.set_stars(acqs=fids.acqs)
    fids.cand_fids = fids.get_fid_candidates()

    # Set list of available fid_set's, accounting for n_fid and cand_fids.
    fids.cand_fid_sets = fids.get_cand_fid_sets()

    # Set initial fid catalog if possible to a set for which no field stars
    # spoiler any of the fid lights and no fid lights is a search spoilers for
    # any current acq star.  If not possible then the table is still zero
    # length and we need to fall through to the optimization process.
    fids.set_initial_catalog()

    # Add a `slot` column that makes sense
    fids.set_slot_column()
    return fids


class FidTable(ACACatalogTable):
    # Define base set of allowed keyword args to __init__. Subsequent MetaAttribute
    # or AliasAttribute properties will add to this.
    allowed_kwargs = ACACatalogTable.allowed_kwargs | set(["acqs", "guide_cands"])

    # Catalog type when plotting (None | 'FID' | 'ACQ' | 'GUI')
    catalog_type = "FID"

    # Name of table.  Use to define default file names where applicable.
    # (e.g. `obs19387/fids.pkl`).
    name = "fids"

    cand_fids = MetaAttribute(is_kwarg=False)
    cand_fid_sets = MetaAttribute(is_kwarg=False)
    guide_cands = MetaAttribute(default=None)

    required_attrs = (
        "att",
        "detector",
        "sim_offset",
        "focus_offset",
        "t_ccd_acq",
        "t_ccd_guide",
        "date",
        "dither_acq",
        "dither_guide",
    )

    include_ids = AliasAttribute()
    exclude_ids = AliasAttribute()

    @property
    def acqs(self):
        return self._acqs() if hasattr(self, "_acqs") else None

    @acqs.setter
    def acqs(self, val):
        """Note some subtlety here - this is using a standard __dict__ attribute
        ``_acqs`` instead of a MetaAttribute, and thus the weakref here is
        ignored in pickling because Table does not pickle the __dict__, it only
        pickles the columns and the ``meta`` attribute.

        This overrides the base class definition which *does* store the value in
        ``.meta`` so it gets into the pickle.

        """
        self._acqs = weakref.ref(val)

    @property
    def t_ccd(self):
        # For fids use the guide CCD temperature
        return self.t_ccd_guide

    @t_ccd.setter
    def t_ccd(self, value):
        self.t_ccd_guide = value
        self.t_ccd_acq = value

    def set_fid_set(self, fid_ids):
        if len(self) > 0:
            self.remove_rows(np.arange(len(self)))
        for fid_id in sorted(fid_ids):
            self.add_row(self.cand_fids.get_id(fid_id))

    def get_cand_fid_sets(self):
        """
        Get a list of candidate fid-sets that can be selected given the list
        of candidate fids that are available.  It also ensure that the fid sets
        are compatible with the specified n_fid.
        """
        cand_fids = self.cand_fids

        if self.n_fid > 3 or self.n_fid < 0:
            raise ValueError("number of fids n_fid must be between 0 and 3 inclusive")

        # Final number of fids is the max of n_fid and the number of candidate fids.
        actual_n_fid = min(self.n_fid, len(cand_fids))

        if actual_n_fid == 0:
            cand_fid_sets = []

        elif actual_n_fid == 1:
            cand_fid_sets = [set([fid_id]) for fid_id in cand_fids["id"]]

        elif actual_n_fid == 2:
            # Make a list of available pairs sorted in order of radial separation
            # (largest first).
            dist2s = []
            fid_ids0 = []
            fid_ids1 = []
            for idx0 in range(len(cand_fids)):
                for idx1 in range(idx0 + 1, len(cand_fids)):
                    fid0 = cand_fids[idx0]
                    fid1 = cand_fids[idx1]
                    dist2 = -(
                        (fid0["yang"] - fid1["yang"]) ** 2
                        + (fid0["zang"] - fid1["zang"]) ** 2
                    )
                    fid_ids0.append(fid0["id"])
                    fid_ids1.append(fid1["id"])
                    dist2s.append(dist2)

            sort_idx = np.argsort(dist2s)
            fid_ids0 = np.array(fid_ids0)[sort_idx]
            fid_ids1 = np.array(fid_ids1)[sort_idx]

            cand_fid_sets = [
                set([fid_id0, fid_id1]) for fid_id0, fid_id1 in zip(fid_ids0, fid_ids1)
            ]

        elif actual_n_fid == 3:
            cand_fids_ids = set(cand_fids["id"])
            cand_fid_sets = [
                fid_set
                for fid_set in FID.fid_sets[self.detector]
                if fid_set <= cand_fids_ids
            ]

        # Restrict candidate fid sets to those that entirely contain the include_ids_set
        include_ids_set = set(self.include_ids_fid)
        cand_fid_sets = [
            fid_set for fid_set in cand_fid_sets if fid_set >= include_ids_set
        ]
        self.log(
            f"Reducing fid sets to those that include fid ids {self.include_ids_fid}"
        )

        return cand_fid_sets

    def set_slot_column(self):
        """
        Set the `slot` column.
        """
        self["slot"] = np.arange(len(self), dtype=np.int64)

        # Add slot to cand_fids table, putting in -99 if not selected as acq.
        # This is for convenience in downstream reporting or introspection.
        cand_fids = self.cand_fids
        slots = [
            self.get_id(fid["id"])["slot"] if fid["id"] in self["id"] else -99
            for fid in cand_fids
        ]
        cand_fids["slot"] = np.array(slots, dtype=np.int64)

    def set_initial_catalog(self):
        """Set initial fid catalog (fid set) if possible to the first set which is
        "perfect":

        - No field stars spoil any of the fid lights
        - Fid lights are not search spoilers for any of the current acq stars
        - The fid set does not include any fid lights in the trap region if there
        are guide candidates that would trigger the fid trap effect.

        If not possible then the table is still zero length and we will need to
        fall through to the optimization process.

        """
        # Start by getting the id of every fid that has a zero spoiler score and
        # is not affected by fid trap, meaning no star spoils the fid as set
        # previously in get_initial_candidates.
        cand_fids = self.cand_fids
        unspoiled_fid_ids = set(
            fid["id"]
            for fid in cand_fids
            if fid["spoiler_score"] == 0 and not fid["fid_trap_spoiler"]
        )

        # Get list of fid_sets that are consistent with candidate fids. These
        # fid sets are the combinations of 3 (or 2) fid lights in preferred
        # order.
        ok_fid_sets = [
            fid_set for fid_set in self.cand_fid_sets if fid_set <= unspoiled_fid_ids
        ]

        # If no fid_sets are possible, return a zero-length table with correct columns
        if not ok_fid_sets:
            fid_set = ()
            self.log("No acceptable fid sets (off-CCD or spoiled by field stars)")
        # If no stars then just pick the first allowed fid set.
        elif self.acqs is None and self.guide_cands is None:
            fid_set = ok_fid_sets[0]
            self.log(f"No acq/guide stars available, using first OK fid set {fid_set}")

        else:
            spoils_any_acq = {}
            spoils_any_guide_cand = {}
            for fid_set in ok_fid_sets:
                self.log(f"Checking fid set {fid_set} for acq star spoilers", level=1)
                for fid_id in fid_set:
                    if self.acqs is not None:
                        if fid_id not in spoils_any_acq:
                            fid = cand_fids.get_id(fid_id)
                            spoils_any_acq[fid_id] = any(
                                self.spoils(fid, acq, acq["halfw"]) for acq in self.acqs
                            )
                    if self.guide_cands is not None:
                        if fid_id not in spoils_any_guide_cand:
                            fid = cand_fids.get_id(fid_id)
                            spoils_any_guide_cand[fid_id] = any(
                                self.spoils(fid, guide, 25)
                                for guide in self.guide_cands
                            )
                            fid_trap, _ = guide.check_fid_trap(
                                self.guide_cands, [fid], self.dither_guide
                            )
                            if np.any(fid_trap):
                                spoils_any_guide_cand[fid_id] = True
                    if fid_id in spoils_any_acq and spoils_any_acq[fid_id]:
                        # Loser, don't bother with the rest.
                        self.log(f"Fid {fid_id} spoils an acq star", level=2)
                        break
                    if (
                        fid_id in spoils_any_guide_cand
                        and spoils_any_guide_cand[fid_id]
                    ):
                        # Loser, don't bother with the rest.
                        self.log(f"Fid {fid_id} spoils a guide candidate", level=2)
                        break
                else:
                    # We have a winner, none of the fid_ids in current fid set
                    # will spoil any acquisition star.  Break out of loop with
                    # fid_set as the winner.
                    self.log(
                        f"Fid set {fid_set} is fine for acq stars and guide candidates"
                    )
                    break
            else:
                # Tried every set and none were acceptable.
                fid_set = ()
                self.log("No acceptable fid set found")

        # Transfer fid set columns to self (which at this point is an empty
        # table)
        idxs = [cand_fids.get_id_idx(fid_id) for fid_id in sorted(fid_set)]
        for name, col in cand_fids.columns.items():
            self[name] = col[idxs]
        self.cand_fids = cand_fids

    def spoils(self, fid, acq, box_size):
        """
        Return true if ``fid`` could be within ``acq`` search box.

        Includes:
        - 20" (4 pix) positional err on fid light
        - 4 pixel readout halfw for fid light
        - 2 pixel PSF of fid light that could creep into search box
        - Acq search box half-width
        - Dither amplitude (since OBC adjusts search box for dither)

        :param fid: fid light (FidTable Row)
        :param acq: acq star (AcqTable Row)
        :param box_size: box size (arcsec)

        :returns: True if ``fid`` could be within ``acq`` search box
        """
        spoiler_margin = FID.spoiler_margin + self.dither_acq + box_size
        dy = np.abs(fid["yang"] - acq["yang"])
        dz = np.abs(fid["zang"] - acq["zang"])
        return dy < spoiler_margin.y and dz < spoiler_margin.z

    def get_fid_candidates(self):
        """
        Get all fids for this detector that are on the CCD (with margin) and are not
        impacted by a bad pixel.

        This also finds fid spoiler stars and computes the spoiler_score.

        Result is updating self.cand_fids.
        """
        yang, zang = get_fid_positions(
            self.detector,
            self.focus_offset,
            self.sim_offset,
            t_ccd=self.t_ccd_acq,
            date=self.date,
        )
        row, col = yagzag_to_pixels(yang, zang, allow_bad=True)
        ids = np.arange(len(yang), dtype=np.int64) + 1  # E.g. 1 to 6 for ACIS

        # Set up candidate fids table (which copies relevant meta data) and add
        # columns.
        cand_fids = FidTable(
            [ids, yang, zang, row, col], names=["id", "yang", "zang", "row", "col"]
        )
        shape = (len(cand_fids),)
        cand_fids["mag"] = np.full(shape, FID.fid_mag)  # 7.000
        cand_fids["spoilers"] = np.full(shape, None)  # Filled in with Table of spoilers
        cand_fids["spoiler_score"] = np.full(shape, 0, dtype=np.int64)
        cand_fids["fid_trap_spoiler"] = np.full(shape, False, dtype=bool)

        self.log(f"Initial candidate fid ids are {cand_fids['id'].tolist()}")

        # First check that any manually included fid ids are valid by seeing if
        # the supplied fid is in the initial ids for this detector.
        if id_diff := set(self.include_ids_fid) - set(cand_fids["id"]):
            raise ValueError(f"included fid ids {id_diff} are not valid")

        # Then reject candidates that are off CCD, have a bad pixel, are spoiled,
        # or are manually excluded, unless the candidates are forced/manually included.
        # Check for spoilers only against stars that are bright enough and on CCD
        # (within dither).
        idx_bads = []
        stars_mask = (self.stars["mag"] < FID.fid_mag - ACA.col_spoiler_mag_diff) & (
            np.abs(self.stars["row"]) < 512 + self.dither_guide.row
        )
        for idx, fid in enumerate(cand_fids):
            excluded = (
                self.off_ccd(fid)
                or self.near_hot_or_bad_pixel(fid)
                or self.has_column_spoiler(fid, self.stars, stars_mask)
                or self.is_excluded(fid)
                or is_spoiled_by_jupiter(fid, self.jupiter)
            )
            included = fid["id"] in self.include_ids_fid
            if not included and excluded:
                idx_bads.append(idx)

        if idx_bads:
            cand_fids.remove_rows(idx_bads)

        cand_fids["idx"] = np.arange(len(cand_fids), dtype=np.int64)

        # If stars or guide candidates are available then find stars that are bad for fid.
        if self.stars or self.guide_cands is not None:
            for fid in cand_fids:
                self.set_spoilers_score(fid)

        return cand_fids

    def off_ccd(self, fid):
        """Return True if ``fid`` is outside allowed CCD region.

        :param fid: FidTable Row of candidate fid
        """
        if (np.abs(fid["row"]) + FID.ccd_edge_margin > ACA.max_ccd_row) or (
            np.abs(fid["col"]) + FID.ccd_edge_margin > ACA.max_ccd_col
        ):
            self.log(
                f"Rejecting fid id={fid['id']} row,col="
                f"({fid['row']:.1f}, {fid['col']:.1f}) off CCD",
                level=1,
            )
            return True
        else:
            return False

    def is_excluded(self, fid):
        """Return True if fid id is in exclude_ids_fid manual list

        :param fid: FidTable Row of candidate fid light
        """
        if fid["id"] in self.exclude_ids_fid:
            self.log(f"Rejecting fid {fid['id']}: manually excluded by exclude_ids_fid")
            return True
        else:
            return False

    def near_hot_or_bad_pixel(self, fid):
        """Return True if fid has a bad pixel too close

        :param fid: FidTable Row of candidate fid light
        """
        dp = FID.spoiler_margin / 5
        r0 = int(fid["row"] - dp)
        c0 = int(fid["col"] - dp)
        r1 = int(fid["row"] + dp) + 1
        c1 = int(fid["col"] + dp) + 1
        dark = self.dark[r0 + 512 : r1 + 512, c0 + 512 : c1 + 512]

        bad = dark > FID.hot_pixel_spoiler_limit
        if np.any(bad):
            idxs = np.flatnonzero(bad)
            rows, cols = np.unravel_index(idxs, dark.shape)
            vals = dark[rows, cols]
            self.log(
                f"Rejecting fid {fid['id']}: near hot or bad pixel(s) "
                f"rows={rows + r0} cols={cols + c0} vals={vals}"
            )
            return True
        else:
            return False

    def set_spoilers_score(self, fid):
        """
        Get stars within FID.spoiler_margin (50 arcsec) + dither and check for fid trap.

        Starcheck uses 25" but this seems small: 20" (4 pix) positional err + 4 pixel
        readout halfw + 2 pixel PSF width of spoiler star.

        This sets the 'spoilers' column value to a table of spoilers stars (usually empty).

        It also sets the 'spoiler_score' based on:
        - 1 for yellow spoiler (4 <= star_mag - fid_mag < 5)
        - 4 for red spoiler (star_mag - fid_mag < 4)

        Additionally, if the fid is in the fid_trap region and there is a potential guide
        candidate that would trigger the fid trap effect, the 'fid_trap_spoiler' flag is set.

        The spoiler score is used later to choose an acceptable set of fids and acq stars.

        :param fid: fid light (FidTable Row)
        """

        stars = self.stars[ACQ.spoiler_star_cols]
        dither = self.dither_guide

        # Run guide star fid_trap checks if guide candidates are available
        if self.guide_cands is not None:
            fid_trap, _ = guide.check_fid_trap(self.guide_cands, [fid], dither)
            if np.any(fid_trap):
                fid["fid_trap_spoiler"] = True
                self.log(
                    f"Fid {fid['id']} spoiled by fid trap potential from guide candidates",
                    level=1,
                )

        # Potential spoiler by position
        spoil = (
            np.abs(stars["yang"] - fid["yang"]) < FID.spoiler_margin + dither.y
        ) & (np.abs(stars["zang"] - fid["zang"]) < FID.spoiler_margin + dither.z)

        if not np.any(spoil):
            # Make an empty table with same columns
            fid["spoilers"] = []
        else:
            stars = stars[spoil]

            # Now check mags
            red = stars["mag"] - fid["mag"] < 4.0
            yellow = (stars["mag"] - fid["mag"] >= 4.0) & (
                stars["mag"] - fid["mag"] < 5.0
            )

            spoilers = stars[red | yellow]
            spoilers.sort("mag")
            spoilers["warn"] = np.where(red[red | yellow], "red", "yellow")
            fid["spoilers"] = spoilers

            if np.any(red):
                fid["spoiler_score"] = 4
            elif np.any(yellow):
                fid["spoiler_score"] = 1

            if fid["spoiler_score"] != 0:
                self.log(
                    f"Set fid {fid['id']} spoiler score to {fid['spoiler_score']}",
                    level=1,
                )


def get_fid_positions(
    detector: str,
    focus_offset: float,
    sim_offset: float,
    t_ccd: float | None = None,
    date: CxoTimeLike | None = None,
) -> tuple:
    """Calculate the fid light positions for all fids for ``detector``.

    This is adapted from the Matlab
    MissionScheduling/stars/StarSelector/@StarSelector/private/fidPositions.m::

      %Convert focus steps to meters
      table = characteristics.Science.Sim.focus_table;
      xshift = interp1(table(:,1),table(:,2),focus,'*linear');
      if(isnan(xshift))
          error('Focus is out of range');
      end

      %find translation from sim offset
      stepsize = characteristics.Science.Sim.stepsize;
      zshift = SIfield.fidpos(:,2)-simoffset*stepsize;

      yfid=-SIfield.fidpos(:,1)/(SIfield.focallength-xshift);
      zfid=-zshift/(SIfield.focallength-xshift);

    This function also applies a temperature-dependent fid offset if ``t_ccd`` and ``date``
    are supplied and the ``PROSECO_ENABLE_FID_OFFSET`` env var is ``"True"`` or not set.

    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param focus_offset: SIM focus offset [steps]
    :param sim_offset: SIM translation offset from nominal [steps]
    :param t_ccd: CCD temperature (C)
    :param date: date (CxoTime compatible)

    :returns: yang, zang where each is a np.array of angles [arcsec]
    """
    # Table of (step, fa_pos [m]) pairs, used to interpolate from FA offset
    # in step to FA offset in meters.
    focus_offset_table = np.array(FID.focus_table)
    steps = focus_offset_table[:, 0]  # Absolute FA step position
    fa_pos = focus_offset_table[:, 1]  # Focus offset in meters
    xshift = np.interp(focus_offset, steps, fa_pos, left=np.nan, right=np.nan)
    if np.isnan(xshift):
        raise ValueError("focus_offset is out of range")

    # Y and Z position of fids on focal plane in meters.
    # Apply SIM Z translation from sim offset to the nominal Z position.
    ypos = FID.fidpos[detector][:, 0]
    zpos = FID.fidpos[detector][:, 1] - sim_offset * FID.tsc_stepsize

    # Calculate angles.  (Should these be atan2?  Does it matter?)
    yfid = -ypos / (FID.focal_length[detector] - xshift)
    zfid = -zpos / (FID.focal_length[detector] - xshift)

    yang, zang = np.degrees(yfid) * 3600, np.degrees(zfid) * 3600

    enable_fid_offset_env = os.environ.get("PROSECO_ENABLE_FID_OFFSET")
    if enable_fid_offset_env not in ("True", "False", None):
        raise ValueError(
            f'PROSECO_ENABLE_FID_OFFSET env var must be either "True", "False", or not set, '
            f"got {enable_fid_offset_env}"
        )

    has_tccd_and_date = t_ccd is not None and date is not None

    if enable_fid_offset_env == "True" and not has_tccd_and_date:
        raise ValueError(
            "t_ccd_acq and date must be provided if PROSECO_ENABLE_FID_OFFSET is 'True'"
        )

    if has_tccd_and_date and enable_fid_offset_env != "False":
        dy, dz = get_fid_offset(date, t_ccd)
        yang += dy
        zang += dz

    return yang, zang
