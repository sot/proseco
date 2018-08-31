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
    :param dither: dither [arcsec].  Defaults to acqs.meta['dither'] if available.
    :param print_log: print log to stdout (default=False)
    """
    fids = FidTable(detector=detector, focus_offset=focus_offset,
                    sim_offset=sim_offset, acqs=acqs, stars=stars,
                    dither=dither, print_log=print_log)

    fids.meta['cand_fids'] = fids.get_fid_candidates()

    # Get an initial fid catalog based on the rank of field stars spoiling fids
    # and fids spoiling current acq star within current acq search box.
    fids.get_initial_catalog()

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
        :param dither: dither [arcsec].  Defaults to acqs.meta['dither'] if available.
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
            if print_log is None:
                print_log = acqs.print_log
            acqs = weakref.ref(acqs)

        super().__init__(data=data, print_log=print_log, **kwargs)

        self.meta.update({'detector': detector,
                          'focus_offset': focus_offset,
                          'sim_offset': sim_offset,
                          'acqs': acqs,
                          'dither': dither,
                          'stars': stars})

    def get_initial_catalog(self):
        cand_fids = self.meta['cand_fids']
        cand_fids_set = set(cand_fids['id'])

        for fid_set in self.meta['fid_sets']:
            if fid_set <= cand_fids_set:
                # If no acqs then just take the first allowed fid set and jump out
                if self.acqs is None:
                    cand_fids_ids = cand_fids['id'].tolist()
                    idxs = [cand_fids.iloc(fid_id) for fid_id in sorted(fid_set)]
                    # Transfer to self (which at this point is an empty table)
                    for name, col in cand_fids.columns.items():
                        self[name] = col[idxs]
                    return

                for fid_id in fid_set:
                    fid = cand_fids.get_id(fid_id)




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

        # Set up candidate fids table and add columns
        cand_fids = FidTable([ids, yang, zang, row, col],
                             names=['id', 'yang', 'zang', 'row', 'col'])

        # Copy meta from parent FidTable
        # cand_fids.meta.update(self.meta)

        # Reject (entirely) candidates that are off CCD or have a bad pixel
        self.reject_off_ccd(cand_fids)
        self.reject_bad_pixel(cand_fids)

        cand_fids['mag'] = FID.fid_mag  # 7.000
        cand_fids['spoilers'] = None  # Filled in with Table of spoilers
        cand_fids['spoiler_score'] = np.int64(0)

        # Make sure cand_fids meta doesn't have its own cand_fids (in case of
        # non-standard call order)
        if 'cand_fids' in self.meta:
            del self.meta['cand_fids']

        # If acqs table available that means stars and dark are available to
        # find stars or bad pixels (imposters) that are bad for fid.
        if self.meta['acqs']:
            for fid in cand_fids:
                fid['spoilers'] = self.get_spoiler_stars(fid)

        return cand_fids

    def reject_off_ccd(self, cand_fids):
        """Filter out candidates that are outside allowed CCD region"""
        n_cand_fids = len(cand_fids)
        idx_bad = []
        for idx, fid in enumerate(cand_fids):
            if ((np.abs(fid['row']) + FID.ccd_edge_margin > ACQ.max_ccd_row) or
                    (np.abs(fid['col']) + FID.ccd_edge_margin > ACQ.max_ccd_col)):
                # If this would result in fewer than 2 fid lights in catalog then
                # just stop rejecting.  It will be picked up in review.
                if n_cand_fids - len(idx_bad) <= 2:
                    self.log('ERROR: fewer than 2 good fids found, accepting off-CCD fid light(s)')
                    break

                self.log(f'Rejecting fid id={fid["id"]} row,col='
                         f'({fid["row"]:.1f}, {fid["col"]:.1f}) off CCD')
                idx_bad.append(idx)

        if idx_bad:
            cand_fids.remove_rows(idx_bad)

    def reject_bad_pixel(self, cand_fids):
        """Filter out candidates that have a bad pixel too close"""
        n_cand_fids = len(cand_fids)
        idx_bad = []
        for idx, fid in enumerate(cand_fids):
            # TO BE IMPLEMENTED
            if False:
                # If this would result in fewer than 2 fid lights in catalog then
                # just stop.  It will be picked up in review.
                if n_cand_fids - len(idx_bad) <= 2:
                    self.log('ERROR: fewer than 2 good fids found, accepting bad fid light(s)')
                    break

                self.log(f'Rejecting fid {fid["id"]}: near bad pixel')
                idx_bad.append(idx)

        if idx_bad:
            cand_fids.remove_rows(idx_bad)

    def get_spoiler_stars(self, fid):
        """Get stars within FID.spoiler_margin (10 pixels) + dither.  Starcheck uses 25" but this
        seems small: 20" (4 pix) positional err + 4 pixel readout halfw + 2 pixel PSF
        width of spoiler star.

        This sets the 'spoilers' column value to a table of spoilers stars (usually empty).

        If also sets the 'spoiler_score' to 1 if there is a yellow spoiler
        (4 <= star_mag - fid_mag < 5) or 4 for red spoiler (star_mag - fid_mag < 4).
        The spoiler score is used later to choose an acceptable set of fids and acq stars.
        """
        stars = self.meta['stars'][ACQ.spoiler_star_cols]
        dither_pix = self.meta['dither'] / 5

        # Potential spoiler by position
        spoil = ((np.abs(stars['row'] - fid['row']) < FID.spoiler_margin + dither_pix) &
                 (np.abs(stars['col'] - fid['col']) < FID.spoiler_margin + dither_pix))

        if not np.any(spoil):
            # Make an empty table with same columns
            fid['spoilers'] = stars[[]]
        else:
            stars = stars[spoil]

            # Now check mags
            red = (stars['mag'] - fid['mag'] < 4.0)
            yellow = ((stars['mag'] - fid['mag'] >= 4.0) &
                      (stars['mag'] - fid['mag'] < 5.0))

            spoilers = stars[red | yellow]
            spoilers.sort('mag')
            spoilers['warn'] = np.where(red, 'red', 'yellow')
            fid['spoilers'] = spoilers

            if np.any(red):
                fid['spoiler_score'] = 4
            elif np.any(yellow):
                fid['spoiler_score'] = 1


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
