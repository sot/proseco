# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Get a catalog of fid lights.
"""

import weakref

import numpy as np

from chandra_aca.transform import yagzag_to_pixels

from . import characteristics_fid as FID
from . import characteristics as ACQ

from .core import ACACatalogTable


def get_fid_catalog(*, detector=None, focus_offset=0, sim_offset=0,
                    acqs=None, stars=None, dither=None,
                    print_log=None):
    """
    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param focus_offset: SIM focus offset [steps] (default=0)
    :param sim_offset: SIM translation offset from nominal [steps] (default=0)
    :param acqs: AcqTable catalog.  Optional but needed for actual fid selection.
    :param stars: stars table.  Defaults to acqs.meta['stars'] if available.
    :param dither: dither (float or 2-element sequence (dither_y, dither_z), [arcsec]
                   Defaults to acqs.meta['dither'] if available.
    :param print_log: print log to stdout (default=False)
    """
    fids = FidTable(detector=detector, focus_offset=focus_offset,
                    sim_offset=sim_offset, acqs=acqs, stars=stars,
                    dither=dither, print_log=print_log)

    fids.meta['cand_fids'] = fids.get_fid_candidates()

    # Set initial fid catalog if possible to a set for which no field stars
    # spoiler any of the fid lights and no fid lights is a search spoilers for
    # any current acq star.  If not possible then the table is still zero
    # length and we need to fall through to the optimization process.
    fids.set_initial_catalog()

    # Add a `slot` column that makes sense
    fids.set_slot_column()

    # Set fid thumbs_up to just be True for now
    fids.thumbs_up = True

    return fids


class FidTable(ACACatalogTable):
    # Elements of meta that should not be directly serialized to YAML
    # (either too big or requires special handling).
    yaml_exclude = ('cand_fids', 'stars', 'dark')

    # Name of table.  Use to define default file names where applicable.
    # (e.g. `obs19387/fids.yaml`).
    name = 'fids'

    def __init__(self, data=None, *,  # Keyword only from here
                 detector=None, focus_offset=0, sim_offset=0,
                 acqs=None, stars=None, dither=None,
                 print_log=None, **kwargs):
        """
        Table of fid lights, with methods for selection.

        :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
        :param focus_offset: SIM focus offset [steps] (default=0)
        :param sim_offset: SIM translation offset from nominal [steps] (default=0)
        :param acqs: AcqTable catalog.  Optional but needed for actual fid selection.
        :param stars: stars table.  Defaults to acqs.meta['stars'] if available.
        :param dither: dither (float or 2-element sequence (dither_y, dither_z), [arcsec]
                       Defaults to acqs.meta['dither'] if available.
        :param print_log: print log to stdout (default=False)
        :param **kwargs: any other kwargs for Table init
        """
        # If acqs (acq catalog) supplied then make a weak reference since that
        # may have a ref to this fid catalog.
        if acqs is not None:
            if stars is None:
                stars = acqs.meta['stars']
            if dither is None:
                dither = acqs.meta['dither']
            if detector is None:
                detector = acqs.meta['detector']
            if sim_offset is None:
                sim_offset = acqs.meta['sim_offset']
            if focus_offset is None:
                focus_offset = acqs.meta['focus_offset']
            if print_log is None:
                print_log = acqs.print_log
            self.acqs = acqs

        # TO DO: fix this temporary stub put in for the 1.0 release.  This converts
        # a 2-element dither (y, z) to a single value which is currently needed for acq
        # selection.
        try:
            dither = max(dither[0], dither[1])
        except TypeError:
            # Catches only the case where dither is not subscriptable.  Anything else
            # should raise.
            pass

        super().__init__(data=data, print_log=print_log, **kwargs)

        self.meta.update({'detector': detector,
                          'focus_offset': focus_offset,
                          'sim_offset': sim_offset,
                          'acqs': acqs,
                          'dither': dither,
                          'stars': stars})

    @property
    def acqs(self):
        return self._acqs() if hasattr(self, '_acqs') else None

    @acqs.setter
    def acqs(self, val):
        self._acqs = weakref.ref(val)

    def set_slot_column(self):
        """
        Set the `slot` column.
        """
        not_sel = '...'  # Not selected

        if len(self) > 0:
            # Sort to make order match the original candidate list order (by
            # increasing mag), and assign a slot.
            self['slot'] = np.arange(len(self))

            # Add slot to cand_fids table, putting in '...' if not selected as acq.
            # This is for convenience in downstream reporting or introspection.
            cand_fids = self.meta['cand_fids']
            slots = [str(self.get_id(fid['id'])['slot']) if fid['id'] in self['id'] else not_sel
                     for fid in cand_fids]
            cand_fids['slot'] = slots
        else:
            self.meta['cand_fids']['slot'] = '...'

    def set_initial_catalog(self):
        """Set initial fid catalog (fid set) if possible to the first set which is
        "perfect":

        - No field stars spoiler any of the fid lights
        - Fid lights are not search spoilers for any of the current acq stars

        If not possible then the table is still zero length and we will need to
        fall through to the optimization process.

        """
        # Start by getting the id of every fid that has a zero spoiler score,
        # meaning no star spoils the fid as set previously in get_initial_candidates.
        cand_fids = self.meta['cand_fids']
        cand_fids_set = set(fid['id'] for fid in cand_fids if fid['spoiler_score'] == 0)

        # Get list of fid_sets that are consistent with candidate fids. These
        # fid sets are the combinations of 3 (or 2) fid lights in preferred
        # order.
        fid_sets = FID.fid_sets[self.meta['detector']]
        ok_fid_sets = [fid_set for fid_set in fid_sets if fid_set <= cand_fids_set]

        # If no fid_sets are possible, leave self as a zero-length table
        if not ok_fid_sets:
            self.log('No acceptable fid sets (off-CCD or spoiled by field stars)')
            return

        # If no acq stars then just pick the first allowed fid set.
        if self.acqs is None:
            fid_set = ok_fid_sets[0]
            self.log(f'No acq stars available, using first OK fid set {fid_set}')

        else:
            spoils_any_acq = {}

            for fid_set in ok_fid_sets:
                self.log(f'Checking fid set {fid_set} for acq star spoilers', level=1)
                for fid_id in fid_set:
                    if fid_id not in spoils_any_acq:
                        fid = cand_fids.get_id(fid_id)
                        spoils_any_acq[fid_id] = any(self.spoils(fid, acq)
                                                     for acq in self.acqs)
                    if spoils_any_acq[fid_id]:
                        # Loser, don't bother with the rest.
                        self.log(f'Fid {fid_id} spoils an acq star', level=2)
                        break
                else:
                    # We have a winner, none of the fid_ids in current fid set
                    # will spoil any acquisition star.  Break out of loop with
                    # fid_set as the winner.
                    self.log(f'Fid set {fid_set} is fine for acq stars')
                    break
            else:
                # Tried every set and none were acceptable.
                fid_set = None
                self.log('No acceptable fid set found')

        if fid_set is not None:
            # Transfer fid set columns to self (which at this point is an empty
            # table)
            idxs = [cand_fids.get_id_idx(fid_id) for fid_id in sorted(fid_set)]
            for name, col in cand_fids.columns.items():
                self[name] = col[idxs]


    def spoils(self, fid, acq):
        """
        Return true if ``fid`` could be within ``acq`` search box.

        Includes:
        - 20" (4 pix) positional err on fid light
        - 4 pixel readout halfw for fid light
        - 2 pixel PSF of fid light that could creep into search box
        - Acq search box half-width
        - Dither amplitude (since OBC adjusts search box for dither)
        """
        spoiler_margin = (FID.spoiler_margin +
                          self.acqs.meta['dither'] +
                          acq['halfw'])
        dy = np.abs(fid['yang'] - acq['yang'])
        dz = np.abs(fid['zang'] - acq['zang'])
        return (dy < spoiler_margin and dz < spoiler_margin)

    def get_fid_candidates(self):
        """
        Get all fids for this detector that are on the CCD (with margin) and are not
        impacted by a bad pixel.

        This also finds fid spoiler stars and computes the spoiler_score.

        Result is updating self.meta['cand_fids'].
        """
        yang, zang = get_fid_positions(self.meta['detector'],
                                       self.meta['focus_offset'],
                                       self.meta['sim_offset'])
        row, col = yagzag_to_pixels(yang, zang, allow_bad=True)
        ids = np.arange(len(yang), dtype=np.int64) + 1  # E.g. 1 to 6 for ACIS

        # Set up candidate fids table (which copies relevant meta data) and add
        # columns.
        cand_fids = FidTable([ids, yang, zang, row, col],
                             names=['id', 'yang', 'zang', 'row', 'col'])
        self.log(f'Initial candidate fid ids are {cand_fids["id"].tolist()}')

        # Reject (entirely) candidates that are off CCD or have a bad pixel
        self.reject_off_ccd(cand_fids)
        self.reject_bad_pixel(cand_fids)

        cand_fids['mag'] = FID.fid_mag  # 7.000
        cand_fids['spoilers'] = None  # Filled in with Table of spoilers
        cand_fids['spoiler_score'] = np.int64(0)

        # If stars are available then find stars that are bad for fid.
        if self.meta['stars']:
            for fid in cand_fids:
                self.set_spoilers_score(fid)

        return cand_fids

    def reject_off_ccd(self, cand_fids):
        """Filter out candidates that are outside allowed CCD region.

        This updates ``cand_fids`` in place.

        :param cand_fids: table of candidate fid lights (on CCD)
        """
        n_cand_fids = len(cand_fids)
        idx_bads = []

        for idx, fid in enumerate(cand_fids):
            if ((np.abs(fid['row']) + FID.ccd_edge_margin > ACQ.max_ccd_row) or
                    (np.abs(fid['col']) + FID.ccd_edge_margin > ACQ.max_ccd_col)):
                # If this would result in fewer than 2 fid lights in catalog then
                # just stop rejecting.  It will be picked up in review.
                if n_cand_fids - len(idx_bads) <= 2:
                    self.log('ERROR: fewer than 2 good fids found, accepting '
                             'off-CCD fid light(s)', level=1)
                    break

                self.log(f'Rejecting fid id={fid["id"]} row,col='
                         f'({fid["row"]:.1f}, {fid["col"]:.1f}) off CCD',
                         level=1)
                idx_bads.append(idx)

        if idx_bads:
            cand_fids.remove_rows(idx_bads)

    def reject_bad_pixel(self, cand_fids):
        """Filter out candidates that have a bad pixel too close"""
        n_cand_fids = len(cand_fids)
        idx_bads = []
        for idx, fid in enumerate(cand_fids):
            # TO BE IMPLEMENTED
            if False:
                # If this would result in fewer than 2 fid lights in catalog then
                # just stop.  It will be picked up in review.
                if n_cand_fids - len(idx_bads) <= 2:
                    self.log('ERROR: fewer than 2 good fids found, accepting bad fid light(s)')
                    break

                self.log(f'Rejecting fid {fid["id"]}: near bad pixel')
                idx_bads.append(idx)

        if idx_bads:
            cand_fids.remove_rows(idx_bads)

    def set_spoilers_score(self, fid):
        """Get stars within FID.spoiler_margin (50 arcsec) + dither.  Starcheck uses
        25" but this seems small: 20" (4 pix) positional err + 4 pixel readout
        halfw + 2 pixel PSF width of spoiler star.

        This sets the 'spoilers' column value to a table of spoilers stars (usually empty).

        If also sets the 'spoiler_score' to 1 if there is a yellow spoiler
        (4 <= star_mag - fid_mag < 5) or 4 for red spoiler (star_mag - fid_mag < 4).
        The spoiler score is used later to choose an acceptable set of fids and acq stars.

        """
        stars = self.meta['stars'][ACQ.spoiler_star_cols]
        dither = self.meta['dither']

        # Potential spoiler by position
        spoil = ((np.abs(stars['yang'] - fid['yang']) < FID.spoiler_margin + dither) &
                 (np.abs(stars['zang'] - fid['zang']) < FID.spoiler_margin + dither))

        if not np.any(spoil):
            # Make an empty table with same columns
            fid['spoilers'] = []
        else:
            stars = stars[spoil]

            # Now check mags
            red = (stars['mag'] - fid['mag'] < 4.0)
            yellow = ((stars['mag'] - fid['mag'] >= 4.0) &
                      (stars['mag'] - fid['mag'] < 5.0))

            spoilers = stars[red | yellow]
            spoilers.sort('mag')
            spoilers['warn'] = np.where(red[red | yellow], 'red', 'yellow')
            fid['spoilers'] = spoilers

            if np.any(red):
                fid['spoiler_score'] = 4
            elif np.any(yellow):
                fid['spoiler_score'] = 1

            if fid['spoiler_score'] != 0:
                self.log(f'Set fid {fid["id"]} spoiler score to {fid["spoiler_score"]}',
                         level=1)


def get_fid_positions(detector, focus_offset, sim_offset):
    """Calculate the fid light positions for all fids for ``detector``.

    This is adapted from the Matlab
    MissionScheduling/stars/StarSelector/@StarSelector/private/fidPositions.m

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

    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param focus_offset: SIM focus offset [steps]
    :param sim_offset: SIM translation offset from nominal [steps]

    :returns: yang, zang where each is a np.array of angles [arcsec]
    """
    # Table of (step, fa_pos [m]) pairs, used to interpolate from FA offset
    # in step to FA offset in meters.
    focus_offset_table = np.array(FID.focus_table)
    steps = focus_offset_table[:, 0]  # Absolute FA step position
    fa_pos = focus_offset_table[:, 1]  # Focus offset in meters
    xshift = np.interp(focus_offset, steps, fa_pos, left=np.nan, right=np.nan)
    if np.isnan(xshift):
        raise ValueError('focus_offset is out of range')

    # Y and Z position of fids on focal plane in meters.
    # Apply SIM Z translation from sim offset to the nominal Z position.
    ypos = FID.fidpos[detector][:, 0]
    zpos = FID.fidpos[detector][:, 1] - sim_offset * FID.tsc_stepsize

    # Calculate angles.  (Should these be atan2?  Does it matter?)
    yfid = -ypos / (FID.focal_length[detector] - xshift)
    zfid = -zpos / (FID.focal_length[detector] - xshift)

    yang, zang = np.degrees(yfid) * 3600, np.degrees(zfid) * 3600

    return yang, zang
