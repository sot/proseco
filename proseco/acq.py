# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Get a catalog of acquisition stars using the algorithm described in
https://docs.google.com/presentation/d/1VtFKAW9he2vWIQAnb6unpK4u1bVAVziIdX9TnqRS3a8
"""

import numpy as np
from scipy import ndimage, stats
from scipy.interpolate import interp1d

from chandra_aca.star_probs import acq_success_prob, prob_n_acq
from chandra_aca.transform import (pixels_to_yagzag, mag_to_count_rate,
                                   snr_mag_for_t_ccd)

from . import characteristics as ACA
from . import characteristics_acq as ACQ
from .core import (get_mag_std, ACACatalogTable, bin2x2,
                   get_image_props, pea_reject_image, ACABox,
                   MetaAttribute, AliasAttribute, calc_spoiler_impact)


def get_acq_catalog(obsid=0, **kwargs):
    """
    Get a catalog of acquisition stars using the algorithm described in
    https://docs.google.com/presentation/d/1VtFKAW9he2vWIQAnb6unpK4u1bVAVziIdX9TnqRS3a8

    If ``obsid`` corresponds to an already-scheduled obsid then the parameters
    ``att``, ``man_angle``, ``t_ccd``, ``date``, and ``dither`` will
    be fetched via ``mica.starcheck`` if not explicitly provided here.

    :param obsid: obsid (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param n_acq: desired number of acquisition stars (default=8)
    :param man_angle: maneuver angle (deg)
    :param t_ccd: ACA CCD temperature (degC)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither: dither size (float or 2-element sequence (dither_y, dither_z), arcsec)
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

    :returns: AcqTable of acquisition stars
    """

    # Make an empty AcqTable object, mostly for logging.  It gets populated
    # after selecting initial an inital catalog of potential acq stars.
    acqs = AcqTable()
    acqs.set_attrs_from_kwargs(obsid=obsid, **kwargs)
    acqs.set_stars()

    # Only allow imposters that are statistical outliers and are brighter than
    # this (temperature-dependent) threshold.  See characterisics.py for more
    # explanation.
    acqs.imposters_mag_limit = snr_mag_for_t_ccd(acqs.t_ccd,
                                                 ref_mag=ACQ.imposter_mag_lim_ref_mag,
                                                 ref_t_ccd=ACQ.imposter_mag_lim_ref_t_ccd)

    acqs.log(f'getting dark cal image at date={acqs.date} t_ccd={acqs.t_ccd:.1f}')

    # Probability of man_err for this observation with a given man_angle.  Used
    # for marginalizing probabilities over different man_errs.
    acqs.p_man_errs = np.array([get_p_man_err(man_err, acqs.man_angle)
                                for man_err in ACQ.man_errs])

    acqs.cand_acqs = acqs.get_acq_candidates(acqs.stars)

    # Fill in the entire acq['probs'].p_acqs table (which is actual a dict of keyed by
    # (box_size, man_err) tuples).
    for acq in acqs.cand_acqs:
        acq['probs'] = AcqProbs(acqs, acq, acqs.dither, acqs.stars, acqs.dark,
                                acqs.t_ccd, acqs.date)

    acqs.get_initial_catalog()

    if acqs.optimize:
        acqs.optimize_catalog(acqs.verbose)

    # Set p_acq column to be the marginalized probabilities
    acqs.update_p_acq_column(acqs)

    # Sort to make order match the original candidate list order (by
    # increasing mag), and assign a slot.  Sadly astropy 3.1 has a real
    # performance bug here and doing the sort makes 6 deepcopy's of the
    # meta, which in this case is substantial (mostly stars).  So temporarily
    # clear out the meta before sorting and then restore from a (light) copy.
    acqs_meta_copy = acqs.meta.copy()
    acqs.meta.clear()
    acqs.sort('idx')
    acqs.meta.update(acqs_meta_copy)

    acqs['slot'] = np.arange(len(acqs), dtype=np.int64)

    # Add slot to cand_acqs table, putting in -99 if not selected as acq.
    # This is for convenience in downstream reporting or introspection.
    slots = [acqs.get_id(acq['id'])['slot'] if acq['id'] in acqs['id'] else -99
             for acq in acqs.cand_acqs]
    acqs.cand_acqs['slot'] = np.array(slots, dtype=np.int64)

    if len(acqs) < acqs.n_acq:
        acqs.log(f'Selected only {len(acqs)} acq stars versus requested {acqs.n_acq}',
                 warning=True)

    return acqs


class AcqTable(ACACatalogTable):
    """
    Catalog of acquisition stars
    """
    # Catalog type when plotting (None | 'FID' | 'ACQ' | 'GUI')
    catalog_type = 'ACQ'

    # Elements of meta that should not be directly serialized to pickle
    # (either too big or requires special handling).
    pickle_exclude = ('stars', 'dark', 'bad_stars')

    # Name of table.  Use to define default file names where applicable.
    # (e.g. `obs19387/acqs.pkl`).
    name = 'acqs'

    # Required attributes
    required_attrs = ('att', 'man_angle', 't_ccd_acq', 'date', 'dither_acq')

    t_ccd = AliasAttribute()  # Maps t_ccd to t_ccd_acq base attribute
    dither = AliasAttribute()  # .. and likewise.
    include_ids = AliasAttribute()
    include_halfws = AliasAttribute()
    exclude_ids = AliasAttribute()

    p_man_errs = MetaAttribute(is_kwarg=False)
    cand_acqs = MetaAttribute(is_kwarg=False)
    p_safe = MetaAttribute(is_kwarg=False)
    _fid_set = MetaAttribute(is_kwarg=False, default=())
    imposters_mag_limit = MetaAttribute(is_kwarg=False, default=20.0)

    @classmethod
    def empty(cls):
        """
        Return a minimal ACACatalogTable which satisfies API requirements.  For AcqTable
        it should have 'id' and 'halfw' columns.

        :returns: StarsTable of stars (empty)
        """
        out = super().empty()
        out['halfw'] = np.full(fill_value=0, shape=(0,), dtype=np.int64)
        return out

    @property
    def fid_set(self):
        if not hasattr(self, '_fid_set'):
            self._fid_set = ()
        return self._fid_set

    @fid_set.setter
    def fid_set(self, fid_ids):
        # No action required if fid_set is already fid_ids
        if self.fid_set == tuple(fid_ids):
            return

        if self.fids is None:
            raise ValueError('cannot set fid_set before setting fids')

        cand_fids = self.fids.cand_fids
        if cand_fids is None:
            raise ValueError('cannot set fid_set before selecting candidate fids')

        self._fid_set = ()
        cand_fids_ids = list(cand_fids['id'])
        for fid_id in sorted(fid_ids):
            if fid_id in cand_fids_ids:
                self._fid_set += (fid_id,)
            else:
                self.log(f'Fid {fid_id} is not in available candidate '
                         f'fid ids {cand_fids_ids}, ignoring',
                         warning=True)

        # Update marginalized p_acq and p_safe.  The underlying probability
        # functions know about fid_set and new values are computed on-demand.
        self.update_p_acq_column(self)
        self.calc_p_safe()

    @property
    def thumbs_up(self):
        if self.n_acq == 0:
            out = 1
        elif len(self) < 2:
            out = 0
        else:
            self.update_p_acq_column(self)
            out = int(self.get_log_p_2_or_fewer() <= np.log10(ACQ.acq_prob))
        return out

    def make_report(self, rootdir='.'):
        """
        Make summary HTML report for acq selection process and outputs.

        Output is in ``<rootdir>/obs<obsid>/acq/index.html`` plus related images
        in that directory.

        :param rootdir: root directory for outputs

        """
        from .report_acq import make_report
        make_report(self, rootdir=rootdir)

    def update_p_acq_column(self, acqs):
        """
        Update (in-place) the marginalized acquisition probability column
        'p_acq'.  This is typically called after a change in catalog or
        change in the fid set.  The acq['probs'].p_acq_marg() method will
        pick up the new fid set.
        :param acqs:
        :param acqs:
        """
        for acq in self:
            acq['p_acq'] = acq['probs'].p_acq_marg(acq['halfw'], acqs)

    def update_idxs_halfws(self, idxs, halfws):
        """
        Update the rows of self to match the specified ``agasc_ids``
        and half widths.  These two input lists must match the length
        of self and correspond to stars in self.cand_acqs.

        :param agasc_ids: list of AGASC IDs
        :param halfws: list of search box half widths
        """
        if len(idxs) != len(self) or len(halfws) != len(self):
            raise ValueError('input lists must match length of acqs')

        for acq, idx, halfw in zip(self, idxs, halfws):
            if acq['idx'] != idx:
                acq_new = self.cand_acqs[idx]
                for name in self.colnames:
                    acq[name] = acq_new[name]
            acq['halfw'] = halfw

    def get_log_p_2_or_fewer(self):
        """
        Return the starcheck acquisition merit function of the probability of
        acquiring two or fewer stars.

        :returns: log10(probability) (float)
        """
        n_or_fewer_probs = prob_n_acq(self['p_acq'])[1]
        if len(n_or_fewer_probs) > 2:
            p_2_or_fewer = n_or_fewer_probs[2]
        else:
            p_2_or_fewer = 1.0
        return np.log10(p_2_or_fewer)

    def get_obs_info(self):
        """
        Convenience method to return the parts of meta that are needed
        for test_common OBS_INFO.

        :returns: dict of observation information
        """
        keys = ('obsid', 'att', 'date', 't_ccd_acq', 't_ccd_guide', 'man_angle',
                'dither_acq', 'dither_guide',
                'detector', 'sim_offset', 'focus_offset')
        return {key: getattr(self, key) for key in keys}

    def get_candidates_mask(self, stars):
        """Get base filter for acceptable candidates.

        This does not include spatial filtering.

        :param stars: StarsTable
        :returns: bool mask of acceptable stars

        """
        ok = ((stars['CLASS'] == 0) &
              (stars['mag'] > 5.9) &
              (stars['mag'] < 11.0) &
              (~np.isclose(stars['COLOR1'], 0.7)) &
              (stars['mag_err'] < 1.0) &  # Mag err < 1.0 mag
              (stars['ASPQ1'] < 40) &  # Less than 2 arcsec offset from nearby spoiler
              (stars['ASPQ2'] == 0) &  # Proper motion less than 0.5 arcsec/yr
              (stars['POS_ERR'] < 3000) &  # Position error < 3.0 arcsec
              ((stars['VAR'] == -9999) | (stars['VAR'] == 5))  # Not known to vary > 0.2 mag
              )
        return ok

    def get_acq_candidates(self, stars, max_candidates=20):
        """
        Get candidates for acquisition stars from ``stars`` table.

        This allows for candidates right up to the useful part of the CCD.
        The p_acq will be accordingly penalized.

        :param stars: list of stars in the field
        :param max_candidates: maximum candidate acq stars

        :returns: Table of candidates, indices of rejected stars
        """
        ok = (self.get_candidates_mask(stars) &
              (np.abs(stars['row']) < ACA.max_ccd_row) &  # Max usable row
              (np.abs(stars['col']) < ACA.max_ccd_col)  # Max usable col
              )

        cand_acqs = stars[ok]

        cand_acqs.sort('mag')
        self.log('Filtering on CLASS, mag, COLOR1, row/col, '
                 'mag_err, ASPQ1/2, POS_ERR:')
        self.log(f'Reduced star list from {len(stars)} to '
                 f'{len(cand_acqs)} candidate acq stars')

        # Reject any candidate with a spoiler or bad star.  Collect a list of
        # good (not rejected) candidates and stop when there are
        # max_candidates.  Check for col spoilers only against stars that are
        # bright enough and on CCD
        goods = []
        stars_mask = stars['mag'] < 11.5 - ACA.col_spoiler_mag_diff
        for ii, acq in enumerate(cand_acqs):
            if (self.in_bad_star_set(acq) or
                    self.has_nearby_spoiler(acq, stars) or
                    self.has_column_spoiler(acq, stars, stars_mask)):
                continue

            goods.append(ii)
            if len(goods) == max_candidates:
                break

        cand_acqs = cand_acqs[goods]
        self.log('Selected {} candidates with no spoiler (star within 3 mag and 30 arcsec)'
                 .format(len(cand_acqs)))

        # If any include_ids (stars forced to be in catalog) ensure that the
        # star is in the cand_acqs table.  Need to re-sort as well.
        if self.include_ids or self.include_halfws:
            self.process_include_ids(cand_acqs, stars)
            cand_acqs.sort('mag')

        cand_acqs.rename_column('COLOR1', 'color')
        # Drop all the other AGASC columns.  No longer useful.
        names = [name for name in cand_acqs.colnames if not name.isupper()]
        cand_acqs = AcqTable(cand_acqs[names])

        box_sizes_list = self.get_box_sizes(cand_acqs)
        halfws = [box_sizes[0] for box_sizes in box_sizes_list]

        # Make this suitable for plotting
        n_cand = len(cand_acqs)
        cand_acqs['idx'] = np.arange(n_cand, dtype=np.int64)
        cand_acqs['type'] = np.full(n_cand, 'ACQ')
        cand_acqs['halfw'] = np.array(halfws, dtype=np.int64)

        # Acq prob for box_size=halfw, marginalized over man_err
        cand_acqs['p_acq'] = np.full(n_cand, -999.0)
        cand_acqs['probs'] = np.full(n_cand, None)  # Filled in with AcqProb objects
        cand_acqs['spoilers'] = np.full(n_cand, None)  # Filled in with Table of spoilers
        cand_acqs['imposters'] = np.full(n_cand, None)  # Filled in with Table of imposters
        # Cached value of box_size + man_err for spoilers
        cand_acqs['spoilers_box'] = np.full(n_cand, None)
        # Cached value of box_size + dither for imposters
        cand_acqs['imposters_box'] = np.full(n_cand, None)
        cand_acqs['box_sizes'] = box_sizes_list

        return cand_acqs

    def get_box_sizes(self, cand_acqs):
        """Get the available box sizes for each cand_acq as all those with size <= the
        largest man_error with non-zero probability.  E.g. in the 5-20 deg man
        angle bin the 80-100" row is 0.1 and the 100-120" row is 0.0.  So this
        will will limit the box sizes to 60, 80, and 100.

        An exception to the box size limit is for bright stars.  For stars
        brighter than 8.0 mag (referenced to t_ccd=-10), the box size is
        allowed to go up to at least 100 arcsec.  For stars brighter than 9.0
        mag it can go up to at least 80 arcsec.  At these bright mags the
        larger search boxes have no impact on acquisition probability.

        This is particularly relevant to man_angle < 5 deg, where the max
        maneuver error is 60 arcsec.  In this case, bright stars can still have
        80 or 100 arcsec boxes.  In the case of a creep-away observation where
        the initial bias might be bad, this gives a bit more margin.

        :param cand_acqs: AcqTable of candidate acq stars
        :return: list of box-size arrays corresponding to cand_acqs table
        """
        box_sizes_list = []
        max_man_err = np.max(ACQ.man_errs[self.p_man_errs > 0])

        # Get the effective equivalent of 8.0 and 9.0 mag for the current t_ccd
        mag_8 = snr_mag_for_t_ccd(self.t_ccd, ref_mag=8.0, ref_t_ccd=-10.0)
        mag_9 = snr_mag_for_t_ccd(self.t_ccd, ref_mag=9.0, ref_t_ccd=-10.0)

        for cand_acq in cand_acqs:
            mag = cand_acq['mag']
            if mag < mag_8:
                max_box_size = max(max_man_err, 100)
            elif mag < mag_9:
                max_box_size = max(max_man_err, 80)
            else:
                max_box_size = max_man_err
            box_sizes = ACQ.box_sizes[ACQ.box_sizes <= max_box_size]
            box_sizes_list.append(box_sizes)

        return box_sizes_list

    def in_bad_star_set(self, acq):
        """
        Returns True if ``acq`` is in the bad star set.

        :param acq: AcqTable Row
        :returns: bool
        """
        if acq['id'] in ACA.bad_star_set:
            self.log(f'Rejecting star {acq["id"]} which is in bad star list', id=acq['id'])
            idx = self.stars.get_id_idx(acq['id'])
            self.bad_stars_mask[idx] = True

            return True
        else:
            return False

    def has_nearby_spoiler(self, acq, stars):
        """
        Returns True if ``acq`` has a nearby star that could spoil acquisition.

        :param acq: AcqTable Row
        :param stars: StarsTable
        :returns: bool
        """
        if acq['ASPQ1'] == 0:
            return False

        dy, dz, frac_norm = calc_spoiler_impact(acq, stars)
        if np.abs(dy) > 1.5 or np.abs(dz) > 1.5 or frac_norm < 0.95:
            self.log(f'Candidate acq star {acq["id"]} rejected due to nearby spoiler(s) '
                     f'dy={dy:.1f} dz={dz:.1f} frac_norm={frac_norm:.2f}',
                     id=acq['id'])
            return True
        else:
            return False

    def process_include_ids(self, cand_acqs, stars):
        """Ensure that the cand_acqs table has stars that were forced to be included.

        Also do validation of include_ids and include_halfws.

        :param cand_acqs: candidate acquisition stars table
        :param stars: stars table

        """
        if len(self.include_ids) != len(self.include_halfws):
            raise ValueError('include_ids and include_halfws must have same length')

        # Ensure values are valid box_sizes
        grid_func = interp1d(ACQ.box_sizes, ACQ.box_sizes,
                             kind='nearest', fill_value='extrapolate')
        self.include_halfws = grid_func(self.include_halfws).tolist()

        super().process_include_ids(cand_acqs, stars)

    def select_best_p_acqs(self, cand_acqs, min_p_acq, acq_indices, box_sizes):
        """
        Find stars with the highest acquisition probability according to the
        algorithm below.  ``p_acqs`` is the same-named column from candidate
        acq stars and it contains a dict keyed by (box_size, man_err).  This
        algorithm uses the assumption of man_err=box_size.

        - Loop over box sizes in descending order (160, ..., 60)
        - Sort in descending order the p_acqs corresponding to that box size
          (where largest p_acqs come first)
        - Loop over the list and add any stars with p_acq > min_p_acq to the
          list of accepted stars.
        - If the list is ``n_acq`` long (completely catalog) then stop

        This function can be called multiple times with successively smaller
        min_p_acq to fill out the catalog.  The acq_indices and box_sizes
        arrays are appended in place in this process.

        :param cand_acqs: AcqTable of candidate acquisition stars
        :param min_p_acq: minimum p_acq to include in this round (float)
        :param acq_indices: list of indices into cand_acqs of selected stars
        :param box_sizes: list of box sizes of selected stars
        """
        self.log(f'Find stars with best acq prob for min_p_acq={min_p_acq}')
        self.log(f'Current catalog: acq_indices={acq_indices} box_sizes={box_sizes}')

        for box_size in ACQ.box_sizes:
            # Get array of marginalized (over man_err) p_acq values corresponding
            # to box_size for each of the candidate acq stars.  For acq's where
            # the current box_size is not in the available list then set the
            # probability to zero.  This happens for small maneuver angles where
            # acq.box_sizes might be only [60] or [60, 80].
            p_acqs_for_box = np.zeros(len(cand_acqs))
            my_box_sizes = cand_acqs['box_sizes']
            my_probs = cand_acqs['probs']
            for idx in range(len(cand_acqs)):
                if box_size in my_box_sizes[idx]:
                    p_acqs_for_box[idx] = my_probs[idx].p_acq_marg(box_size, self)

            self.log(f'Trying search box size {box_size} arcsec', level=1)

            if np.all(p_acqs_for_box < min_p_acq):
                self.log(f'No acceptable candidates (probably small man angle)', level=2)
                continue

            indices = np.argsort(-p_acqs_for_box, kind='mergesort')
            for acq_idx in indices:
                if acq_idx in acq_indices:
                    continue

                acq = cand_acqs[acq_idx]

                # Don't consider any stars in the exclude list
                if acq['id'] in self.exclude_ids:
                    continue

                p_acq = p_acqs_for_box[acq_idx]
                accepted = p_acq > min_p_acq
                status = 'ACCEPTED' if accepted else 'rejected'
                self.log(f'Star idx={acq_idx:2d} id={acq["id"]:10d} '
                         f'box={box_size:3d} mag={acq["mag"]:5.1f} p_acq={p_acq:.3f} '
                         f'{status}',
                         id=acq['id'],
                         level=2)

                if accepted:
                    acq_indices.append(acq_idx)
                    box_sizes.append(box_size)

                if len(acq_indices) == self.n_acq:
                    self.log(f'Found {self.n_acq} acq stars, done')
                    return

    def get_initial_catalog(self):
        """
        Get the initial catalog of up to ``n_acq`` candidate acquisition stars.  This
        updates the current AcqTable (self) in place to add selected stars.

        TO DO: these should all just be taken from self

        :param cand_acqs: AcqTable of candidate acquisition stars
        :param stars: StarsTable of stars in or near the ACA FOV
        :param dark: dark current image (ndarray, e-/sec)
        :param dither: dither (float, arcsec)
        :param t_ccd: CCD temperature (float, degC)
        :param date: observation date
        """
        cand_acqs = self.cand_acqs

        self.log(f'Getting initial catalog from {len(cand_acqs)} candidates')

        # Start the lists of acq indices and box sizes with the values from
        # the include lists.  Usually these will be empty.
        acq_indices = [cand_acqs.get_id_idx(id) for id in self.include_ids]
        box_sizes = self.include_halfws[:]  # make a copy

        # Accumulate indices and box sizes of candidate acq stars that meet
        # successively less stringent minimum p_acq.
        for min_p_acq in (0.75, 0.5, 0.25, 0.05):
            if len(acq_indices) < self.n_acq:
                # Updates acq_indices, box_sizes in place
                self.select_best_p_acqs(cand_acqs, min_p_acq, acq_indices, box_sizes)

            if len(acq_indices) == self.n_acq:
                break

        # Make all the not-accepted candidate acqs have halfw=120 as a reasonable
        # default and then set the accepted acqs to the best box_size.  Then set
        # p_acq to the marginalized acquisition probability.
        cand_acqs['halfw'] = np.minimum(120, cand_acqs['halfw'])
        cand_acqs['halfw'][acq_indices] = box_sizes
        cand_acqs.update_p_acq_column(self)

        # Finally select the initial catalog
        acqs_init = cand_acqs[acq_indices]

        # Transfer to acqs (which at this point is an empty table)
        self.add_columns(acqs_init.columns.values())

    def calc_p_brightest(self, acq, box_size, man_err=0, bgd=0):
        """
        Calculate the probability that the `acq` star is the brightest
        candidate in the search box.

        This caches the spoiler and imposter stars in the acqs table (the row
        corresponding to ``acq``).  It is required that the first time this is
        called that the box_size and man_err be the maximum, and this is checked.

        :param acq: acq stars (AcqTable Row)
        :param box_size: box size (float, arcsec)
        :param man_err: maneuver error (float, arcsec, default=0)
        :param bgd: assume background for imposters (float, e-sec, default=0)

        :returns: probability that acq is the brightest (float)
        """
        stars = self.stars
        dark = self.dark
        dither = self.dither

        # Spoilers
        ext_box_size = box_size + man_err
        kwargs = dict(stars=stars, acq=acq, box_size=ext_box_size)
        spoilers = get_intruders(acq, ext_box_size, 'spoilers',
                                 n_sigma=2.0,  # TO DO: put to characteristics
                                 get_func=get_spoiler_stars, kwargs=kwargs)

        # Imposters
        ext_box_size = box_size + dither
        kwargs = dict(star_row=acq['row'], star_col=acq['col'],
                      maxmag=acq['mag'] + acq['mag_err'],
                      box_size=ext_box_size,
                      dark=dark,
                      bgd=bgd,  # TO DO deal with this
                      mag_limit=self.imposters_mag_limit
                      )
        imposters = get_intruders(acq, ext_box_size, 'imposters',
                                  n_sigma=1.0,  # TO DO: put to characteristics
                                  get_func=get_imposter_stars, kwargs=kwargs)

        mags = np.concatenate([spoilers['mag'], imposters['mag']])
        mag_errs = np.concatenate([spoilers['mag_err'], imposters['mag_err']])
        prob = calc_p_brightest_compare(acq, mags, mag_errs)

        return prob

    def calc_p_safe(self, verbose=False):
        """
        Calculate the probability of a safing action resulting from failure
        to acquire at least two (2) acquisition stars.

        This uses the probability of 2 or fewer stars => "conservative" p_fail at this
        man_err.  This uses 2 stars instead of 1 or fewer (which is the actual criteria
        for a safing action).  This allows for one star to be dropped for reasons not
        reflected in the acq model probability and makes the optimization dig a bit deeper
        in to the catalog beyond the brightest stars.

        :returns: p_safe (float)

        """

        p_no_safe = 1.0

        self_halfws = self['halfw']
        self_probs = self['probs']

        for man_err, p_man_err in zip(ACQ.man_errs, self.p_man_errs):
            if p_man_err == 0.0:
                continue

            p_acqs = [prob.p_acqs(halfw, man_err, self)
                      for halfw, prob in zip(self_halfws, self_probs)]

            p_n_cum = prob_n_acq(p_acqs)[1]  # This returns (p_n, p_n_cum)

            # Probability of 2 or fewer stars => conservative fail criteria
            p2 = p_n_cum[2]

            if verbose:
                self.log(f'man_err = {man_err}, p_man_err = {p_man_err}')
                self.log('p_acqs =' + ' '.join(['{:.3f}'.format(val) for val in p_acqs]))
                self.log('log10(p 2_or_fewer) = {:.2f}'.format(np.log10(p2)))

            p_no_safe *= (1 - p_man_err * p2)

        p_safe = 1 - p_no_safe
        self.p_safe = p_safe

        return p_safe

    def optimize_acq_halfw(self, idx, p_safe, verbose=False):
        """
        Optimize the box size (halfw) for the acq star ``idx`` in the current acqs
        table.  Assume current ``p_safe``.

        :param idx: acq star index
        :param p_safe: current value of p_safe
        :param verbose: include extra information in the run log
        :returns improved, p_safe: whether p_safe was improved and the new value
        """
        acq = self[idx]
        orig_halfw = acq['halfw']
        orig_p_acq = acq['probs'].p_acq_marg(acq['halfw'], self)

        self.log(f'Optimizing halfw for idx={idx} id={acq["id"]}', id=acq['id'])

        # Compute p_safe for each possible halfw for the current star
        p_safes = []
        box_sizes = acq['box_sizes']
        for box_size in box_sizes:
            new_p_acq = acq['probs'].p_acq_marg(box_size, self)
            # Do not reduce marginalized p_acq to below 0.1.  It can happen that p_safe
            # goes down very slightly with an increase in box size from the original,
            # and then the box size gets stuck there because of the deadband for later
            # reducing box size.
            if new_p_acq < 0.1 and new_p_acq < orig_p_acq:
                self.log(f'Skipping halfw {box_size}: new marg p_acq < 0.1 and new < orig'
                         f' ({new_p_acq:.3f} < {orig_p_acq:.3f})')
                p_safes.append(p_safe)
            else:
                acq['halfw'] = box_size
                p_safes.append(self.calc_p_safe(verbose))

        # Find best p_safe
        min_idx = np.argmin(p_safes)
        min_p_safe = p_safes[min_idx]
        min_halfw = box_sizes[min_idx]

        # If p_safe went down, then consider this an improvement if either:
        #   - acq halfw is increased (bigger boxes are better)
        #   - p_safe went down by at least 10%
        # So avoid reducing box sizes for only small improvements in p_safe.
        improved = ((min_p_safe < p_safe) and
                    ((min_halfw > orig_halfw) or (min_p_safe / p_safe < 0.9)))

        p_safes_strs = [f'{np.log10(p):.2f} ({box_size}")'
                        for p, box_size in zip(p_safes, box_sizes)]
        self.log('p_safes={}'.format(', '.join(p_safes_strs)), level=1, id=acq['id'])
        self.log('min_p_safe={:.2f} p_safe={:.2f} min_halfw={} orig_halfw={} improved={}'
                 .format(np.log10(min_p_safe), np.log10(p_safe),
                         min_halfw, orig_halfw, improved),
                 level=1, id=acq['id'])

        if improved:
            self.log(f'Update acq idx={idx} halfw from {orig_halfw} to {min_halfw}',
                     level=1, id=acq['id'])
            p_safe = min_p_safe
            acq['halfw'] = min_halfw
        else:
            acq['halfw'] = orig_halfw

        return p_safe, improved

    def optimize_acqs_halfw(self, verbose=False):
        """
        Optimize the box_size (halfw) for the acq stars in the current catalog.
        This cycles through each star and optimizes the box size for that star
        using the ``optimize_acq_halfw()`` method.

        :param verbose: include additional information in the run log
        """
        p_safe = self.calc_p_safe()
        idxs = self['p_acq'].argsort()

        # Any updates made?
        any_improved = False

        for idx in idxs:
            # Don't optimize halfw for a star that is specified for inclusion
            if self['id'][idx] in self.include_ids:
                continue

            p_safe, improved = self.optimize_acq_halfw(idx, p_safe, verbose)
            any_improved |= improved

        return p_safe, any_improved

    def optimize_catalog(self, verbose=False):
        """
        Optimize the current acquisition catalog.

        :param verbose: include additional information in the run log
        """
        # If every acq star is specified as included, then no optimization
        if all(acq['id'] in self.include_ids for acq in self):
            return

        p_safe = self.calc_p_safe(verbose=True)
        self.log('initial log10(p_safe)={:.2f}'.format(np.log10(p_safe)))

        # Start by optimizing the half-widths of the initial catalog
        for _ in range(5):
            p_safe, improved = self.optimize_acqs_halfw(verbose)
            if not improved:
                break

        self.log(f'After optimizing initial catalog p_safe = {p_safe:.5f}')

        # Now try to swap in a new star from the candidate list and see if
        # it can improve p_safe.  Skips candidates already in the catalog
        # or specifically excluded.
        skip_acq_ids = set(self['id']) | set(self.exclude_ids)
        for cand_acq in self.cand_acqs:
            cand_id = cand_acq['id']
            if cand_id in skip_acq_ids:
                continue

            # Get the index of the worst p_acq in the catalog, excluding acq stars
            # that are in include_ids (since they are not to be replaced).
            ok = [acq['id'] not in self.include_ids for acq in self]
            # acqs = self[ok]
            acqs_probs_ok = self['probs'][ok]
            acqs_halfw_ok = self['halfw'][ok]
            acqs_id_ok = self['id'][ok]

            # Sort by the marginalized acq probability for the current box size
            p_acqs = [acq_probs.p_acq_marg(acq_halfw, self)
                      for acq_probs, acq_halfw in zip(acqs_probs_ok, acqs_halfw_ok)]
            # TODO: performance?
            idx_worst = np.argsort(p_acqs, kind='mergesort')[0]

            idx = self.get_id_idx(acqs_id_ok[idx_worst])

            self.log('Trying to use {} mag={:.2f} to replace idx={} with p_acq={:.3f}'
                     .format(cand_id, cand_acq['mag'], idx, p_acqs[idx_worst]), id=cand_id)

            # Make a copy of the row (acq star) as a numpy void (structured array row)
            orig_acq = self[idx].as_void()

            # Stub in the new candidate and get the best halfw (and corresponding new p_safe)
            self[idx] = cand_acq
            new_p_safe, improved = self.optimize_acq_halfw(idx, p_safe, verbose)

            # If the new star is noticably better (regardless of box size), OR
            # comparable but with a bigger box, then accept it and do one round of
            # full catalog box-size optimization.
            improved = ((new_p_safe / p_safe < 0.9) or
                        (new_p_safe < p_safe and self['halfw'][idx] > orig_acq['halfw']))
            if improved:
                p_safe, improved = self.optimize_acqs_halfw(verbose)
                self.calc_p_safe(verbose=True)
                self.log(f'  accepted, new p_safe = {p_safe:.5f}', id=cand_id)
            else:
                self[idx] = orig_acq


def get_spoiler_stars(stars, acq, box_size):
    """
    Get acq spoiler stars, i.e. any star in the specified box_size (which
    would normally be an extended box including man_err).

    OBC adjusts search box position based on the difference between estimated
    and target attitude (which is the basis for yang/zang in catalog).  Dither
    is included in the adjustment, so the only remaining term is the
    maneuver error, which is included via the ``man_err`` box extension.
    Imagine a 500 arcsec dither pattern.  OBC adjusts search box for that,
    so apart from actual man err the box will be centered on the acq star.

    See this ref for information on how well the catalog mag errors correlate
    with observed.  Answer: not exactly, but probably good enough.  Plots all
    the way at the bottom are key::

      http://nbviewer.jupyter.org/url/cxc.harvard.edu/mta/ASPECT/
             ipynb/ssawg/2018x03x21/star-mag-uncertainties.ipynb

    TO DO: consider mag uncertainties at the faint end related to
    background subtraction and warm pixel corruption of background.

    :param stars: StarsTable of stars for this field
    :param acq: acquisition star (AcqTable Row)
    :param box_size: box size (float, arcsec)

    :returns: numpy structured array of spoiler stars
    """
    stars = stars.as_array()
    # 1-sigma of difference of stars['mag'] - acq['mag']
    # TO DO: lower limit clip?
    mag_diff_err = np.sqrt(stars['mag_err'] ** 2 + acq['mag_err'] ** 2)

    # Stars in extended box and within 3-sigma (99.7%)
    ok = ((np.abs(stars['yang'] - acq['yang']) < box_size) &
          (np.abs(stars['zang'] - acq['zang']) < box_size) &
          (stars['mag'] - acq['mag'] < 3 * mag_diff_err) &
          (stars['id'] != acq['id'])
          )
    spoilers = stars[ok]
    spoilers.sort(order=['mag'])

    return spoilers


def get_imposter_stars(dark, star_row, star_col, thresh=None,
                       maxmag=11.5, box_size=120, bgd=40, mag_limit=20.0, test=False):
    """
    Note: current alg purposely avoids using the actual flight background
    calculation because this is unstable to small fluctuations in values
    and often over-estimates background.  Using this can easily miss a
    search hit that the flight ACA will detect.  So just use a mean
    dark current ``bgd``.

    :param dark: dark current image (ndarray, e-/sec)
    :param star_row: row of acq star (float)
    :param star_col: col of acq star (float)
    :param thresh: PEA search hit threshold for a 2x2 block (e-/sec)
    :param maxmag: Max mag (alternate way to specify search hit ``thresh``)
    :param box_size: box size (arcsec)
    :param bgd: assumed flat background (float, e-/sec)
    :param mag_limit: Max mag for imposter (using 6x6 readout)
    :param test: hook for convenience in algorithm testing

    :returns: numpy structured array of imposter stars
    """
    # Convert row/col to array index coords unless testing.
    rc_off = 0 if test else 512
    acq_row = int(star_row + rc_off)
    acq_col = int(star_col + rc_off)
    box_row = int(box_size.row)
    box_col = int(box_size.col)

    # Make sure box is within CCD
    box_r0 = np.clip(acq_row - box_row, 0, 1024)
    box_r1 = np.clip(acq_row + box_row, 0, 1024)
    box_c0 = np.clip(acq_col - box_col, 0, 1024)
    box_c1 = np.clip(acq_col + box_col, 0, 1024)

    # Make sure box has even number of pixels on each edge.  Increase
    # box by one if needed.
    #
    # TO DO: Test the clipping and shrinking code
    #
    if (box_r1 - box_r0) % 2 == 1:
        if box_r1 == 1024:
            box_r0 -= 1
        else:
            box_r1 += 1
    if (box_c1 - box_c0) % 2 == 1:
        if box_c1 == 1024:
            box_c0 -= 1
        else:
            box_c1 += 1

    # Get bgd-subtracted dark current image corresponding to the search box
    # and bin in 2x2 blocks.
    dc2x2 = bin2x2(dark[box_r0:box_r1, box_c0:box_c1]) - bgd * 4
    if test:
        print(dc2x2)

    # PEA search hit threshold for a 2x2 block based on count_rate(MAXMAG) / 4
    if thresh is None:
        thresh = mag_to_count_rate(maxmag) / 4  # e-/sec

    # Get an image ``dc_labeled`` which same shape as ``dc2x2`` but has
    # contiguous regions above ``thresh`` labeled with a unique index.
    # This is a one-line way of doing the PEA merging process, roughly.
    dc_labeled, n_hits = ndimage.label(dc2x2 > thresh)
    if test:
        print(dc_labeled)

    # If no hits just return empty list
    if n_hits == 0:
        return []

    outs = []
    for idx in range(n_hits):
        # Get row and col index vals for each merged region of search hits
        rows, cols = np.where(dc_labeled == idx + 1)
        vals = dc2x2[rows, cols]

        # Centroid row, col in 2x2 binned coords.  Since we are using edge-based
        # coordinates, we need to at 0.5 pixels to coords for FM centroid calc.
        # A single pixel at coord (0, 0) has FM centroid (0.5, 0.5).
        rows = rows + 0.5
        cols = cols + 0.5
        vals_sum = np.sum(vals)
        r2x2 = np.sum(rows * vals) / vals_sum
        c2x2 = np.sum(cols * vals) / vals_sum

        # Integer centroid row/col (center of readout image 8x8 box)
        c_row = int(np.round(box_r0 + 2 * r2x2))
        c_col = int(np.round(box_c0 + 2 * c2x2))

        # Reject if too close to CCD edge
        if (c_row < 4 or c_row > dark.shape[0] - 4 or
                c_col < 4 or c_col > dark.shape[1] - 4):
            continue

        img, img_sum, mag, row, col = get_image_props(dark, c_row, c_col, bgd)

        if mag > mag_limit:
            continue

        if pea_reject_image(img):
            continue

        # Revert to ACA coordinates (row,col => -512:512) unless testing, where
        # it is more convenient to just use normal array index coords.
        if not test:
            row -= 512
            col -= 512
            c_row -= 512
            c_col -= 512

        yang, zang = pixels_to_yagzag(row, col, allow_bad=True)

        out = (row,
               col,
               row - star_row,
               col - star_col,
               yang,
               zang,
               c_row - 4,
               c_col - 4,
               img,
               img_sum,
               mag,
               get_mag_std(mag).item(),
               )
        outs.append(out)

    if len(outs) > 0:
        dtype = [('row', '<f8'), ('col', '<f8'), ('d_row', '<f8'), ('d_col', '<f8'),
                 ('yang', '<f8'), ('zang', '<f8'), ('row0', '<i8'), ('col0', '<i8'),
                 ('img', 'f8', (8, 8)), ('img_sum', '<f8'), ('mag', '<f8'), ('mag_err', '<f8')]
        outs = np.rec.fromrecords(outs, dtype=dtype)
        outs.sort(order=['mag'])

    return outs


def calc_p_brightest_compare(acq, mags, mag_errs):
    """
    For given ``acq`` star and intruders mag, mag_err,
    do the probability calculation to see if the acq star is brighter
    than all of them.

    :param acq: acquisition star (AcqTable Row)
    :param mags: iterable of mags
    :param mag_errs: iterable of mag errors

    :returns: probability that acq stars is brighter than all mags
    """
    if len(mags) == 0:
        return 1.0

    n_pts = 100
    x0, x1 = stats.norm.ppf([0.001, 0.999], loc=acq['mag'], scale=acq['mag_err'])
    x = np.linspace(x0, x1, n_pts)
    dx = (x1 - x0) / (n_pts - 1)

    acq_pdf = stats.norm.pdf(x, loc=acq['mag'], scale=acq['mag_err'])

    sp_cdfs = []
    for mag, mag_err in zip(mags, mag_errs):
        # Compute prob intruder is fainter than acq (so sp_mag > x).
        # CDF is prob that sp_mag < x, so take 1-CDF.
        sp_cdf = stats.norm.cdf(x, loc=mag, scale=mag_err)
        sp_cdfs.append(1 - sp_cdf)
    prod_sp_cdf = np.prod(sp_cdfs, axis=0).clip(1e-30)

    # Do the integral  ∫ dθ p(θ|t) Πm≠t p(θ<θt|m)
    prob = np.sum(acq_pdf * prod_sp_cdf * dx)

    return prob


def get_intruders(acq, box_size, name, n_sigma, get_func, kwargs):
    """
    Get intruders table for name='spoilers' or 'imposters' from ``acq``.
    If not already in acq then call ``get_func(**kwargs)`` to get it.

    :param acq: acq stars (AcqTable Row)
    :param box_size: box size (float, arcsec)
    :param name: intruder name ('spoilers' | 'imposters')
    :param n_sigma: sigma threshold for comparisons
    :param get_func: function to actually get spoilers or imposters
    :param kwargs: kwargs to pass to get_func()

    :returns: dict with keys yang, zang, mag, mag_err.
    """
    name_box = name + '_box'
    intruders = acq[name]
    box_size = ACABox(box_size)

    if intruders is None:
        intruders = get_func(**kwargs)
        acq[name_box] = box_size

        if len(intruders) > 0:
            # Clip to within n_sigma.  d_mag < 0 for intruder brighter than acq
            d_mag = intruders['mag'] - acq['mag']
            d_mag_err = np.sqrt(intruders['mag_err'] ** 2 + acq['mag_err'] ** 2)
            ok = d_mag < n_sigma * d_mag_err
            intruders = intruders[ok]
        acq[name] = intruders

    else:
        # Ensure cached spoilers cover the current case.
        if box_size > acq[name_box]:
            raise ValueError(f'box_size is greater than {name_box}')

    colnames = ['yang', 'zang', 'mag', 'mag_err']
    if len(intruders) == 0:
        intruders = {name: np.array([], dtype=np.float64) for name in colnames}
    else:
        ok = ((np.abs(intruders['yang'] - acq['yang']) < box_size.y) &
              (np.abs(intruders['zang'] - acq['zang']) < box_size.z))
        intruders = {name: intruders[name][ok] for name in ['mag', 'mag_err']}

    return intruders


def calc_p_on_ccd(row, col, box_size):
    """
    Calculate the probability that star and initial tracked readout box
    are fully within the usable part of the CCD.

    Note that ``box_size`` here is not a search box size, it is normally
    ``man_err + dither`` and reflects the size of the box where the star can
    land on the CCD.  This is independent of the search box size, but does
    assume that man_err < search box size.  This is always valid because
    this function only gets called in that case (otherwise p_acq is just
    set to 0.0 in calc_p_safe.  Dither does not enter into the
    ``man_err < search box size`` relation because the OBC accounts for
    dither when setting the search box position.

    This uses a simplistic calculation which assumes that ``p_on_ccd`` is
    just the fraction of box area that is within the effective usable portion
    of the CCD.

    :param row: row coordinate of star (float)
    :param col: col coordinate of star (float)
    :param box_size: box size (ACABox)

    :returns: probability the star is on usable part of CCD (float)
    """
    p_on_ccd = 1.0

    # Require that the readout box when candidate acq star is evaluated
    # by the PEA (via a normal 8x8 readout) is fully on the CCD usable area.
    # Do so by reducing the effective CCD usable area by the readout
    # halfwidth (noting that there is a leading row before 8x8).
    max_ccd_row = ACA.max_ccd_row - 5
    max_ccd_col = ACA.max_ccd_col - 4

    for rc, max_rc, half_width in ((row, max_ccd_row, box_size.row),
                                   (col, max_ccd_col, box_size.col)):

        # Pixel boundaries are symmetric so just take abs(row/col)
        rc1 = abs(rc) + half_width

        full_width = half_width * 2
        pix_off_ccd = rc1 - max_rc
        if pix_off_ccd > 0:
            # Reduce p_on_ccd by fraction of pixels inside usable area.
            pix_inside = full_width - pix_off_ccd
            if pix_inside > 0:
                p_on_ccd *= pix_inside / full_width
            else:
                p_on_ccd = 0.0

    return p_on_ccd


class AcqProbs:
    def __init__(self, acqs, acq, dither, stars, dark, t_ccd, date):
        """Calculate probabilities related to acquisition, in particular an element
        in the ``p_acqs`` matrix which specifies star acquisition probability
        for given search box size and maneuver error.

        This sets these attributes:

        - ``p_brightest``: probability this star is the brightest in box (function
            of ``box_size`` and ``man_err``)
        - ``p_acq_model``: probability of acquisition from the chandra_aca model
            (function of ``box_size``)
        - ``p_on_ccd``: probability star is on the usable part of the CCD (function
            of ``man_err`` and ``dither``)
        - ``p_acqs``: product of the above three

        Since chandra_aca 4.24 with the grid-floor model, the acquisition
        probability model value is multiplied here by 0.985 in order to help
        the optimization algorithm converge to a good solution.  The grid-floor
        model has accurate p_fail values for bright stars, in the range of
        0.005, but for catalogs with some bright stars this ends up skewing the
        p_safe calculation to where the box size of fainter stars does not make
        enough impact to get optimized.

        :param acqs: acqs table (AcqTable)
        :param acq: acq star (AcqTable Row) in the candidate acqs table
        :param dither: dither (float, arcsec)
        :param stars: stars table
        :param dark: dark current map
        :param t_ccd: CCD temperature (float, degC)
        :param date: observation date

        """
        self._p_brightest = {}
        self._p_acq_model = {}
        self._p_on_ccd = {}
        self._p_acqs = {}
        self._p_acq_marg = {}
        self._p_fid_spoiler = {}
        self._p_fid_id_spoiler = {}

        # Convert table row to plain dict for persistence
        self.acq = {key: acq[key] for key in ('yang', 'zang')}

        for box_size in ACQ.box_sizes:
            # Need to iterate over man_errs in reverse order because calc_p_brightest
            # caches interlopers based on first call, so that needs to have the largest
            # box sizes.
            for man_err in ACQ.man_errs[::-1]:
                if man_err > box_size:
                    p_brightest = self._p_brightest[box_size, man_err] = 0.0
                else:
                    # Prob of being brightest in box (function of box_size and
                    # man_err, independently because imposter prob is just a
                    # function of box_size not man_err).  Technically also a
                    # function of dither, but that does not vary here.
                    p_brightest = acqs.calc_p_brightest(acq, box_size=box_size,
                                                        man_err=man_err)
                    self._p_brightest[box_size, man_err] = p_brightest

        # Acquisition probability model value (function of box_size only)
        for box_size in ACQ.box_sizes:
            p_acq_model = acq_success_prob(date=date, t_ccd=t_ccd,
                                           mag=acq['mag'], color=acq['color'],
                                           spoiler=False, halfwidth=box_size)
            self._p_acq_model[box_size] = p_acq_model * 0.985

        # Probability star is in acq box (function of man_err and dither only)
        for man_err in ACQ.man_errs:
            p_on_ccd = calc_p_on_ccd(acq['row'], acq['col'], box_size=man_err + dither)
            self._p_on_ccd[man_err] = p_on_ccd

    def p_on_ccd(self, man_err):
        return self._p_on_ccd[man_err]

    def p_brightest(self, box_size, man_err, acqs):
        assert acqs.cand_acqs is not None
        return self._p_brightest[box_size, man_err]

    def p_acq_model(self, box_size):
        return self._p_acq_model[box_size]

    def p_acqs(self, box_size, man_err, acqs):
        assert acqs.cand_acqs is not None
        fid_set = acqs.fid_set

        try:
            return self._p_acqs[box_size, man_err, fid_set]
        except KeyError:
            p_acq = (self.p_brightest(box_size, man_err, acqs) *
                     self.p_acq_model(box_size) *
                     self.p_on_ccd(man_err) *
                     self.p_fid_spoiler(box_size, acqs))
            self._p_acqs[box_size, man_err, fid_set] = p_acq
            return p_acq

    def p_acq_marg(self, box_size, acqs):
        assert acqs.cand_acqs is not None
        fid_set = acqs.fid_set
        try:
            return self._p_acq_marg[box_size, fid_set]
        except KeyError:
            p_acq_marg = 0.0
            for man_err, p_man_err in zip(ACQ.man_errs, acqs.p_man_errs):
                p_acq_marg += self.p_acqs(box_size, man_err, acqs) * p_man_err
            self._p_acq_marg[box_size, fid_set] = p_acq_marg
            return p_acq_marg

    def p_fid_spoiler(self, box_size, acqs):
        """
        Return the probability multiplier based on any fid in the current fid set spoiling
        this acq star (within ``box_size``).  The current fid set is a property of the
        ``fids`` table.  The output value will be 1.0 for no spoilers and 0.0 for one or
        more spoiler (normally there can be at most one fid spoiler).

        This caches the values in a dict for subsequent access.

        :param acqs:
        :param box_size: search box size in arcsec
        :returns: probability multiplier (0 or 1)
        """
        assert acqs.cand_acqs is not None
        fid_set = acqs.fid_set
        try:
            return self._p_fid_spoiler[box_size, fid_set]
        except KeyError:
            p_fid_spoiler = 1.0

            # If there are fids then multiplier the individual fid spoiler probs
            for fid_id in fid_set:
                p_fid_spoiler *= self.p_fid_id_spoiler(box_size, fid_id, acqs)
            self._p_fid_spoiler[box_size, fid_set] = p_fid_spoiler

            return p_fid_spoiler

    def p_fid_id_spoiler(self, box_size, fid_id, acqs):
        """
        Return the probability multiplier for fid ``fid_id`` spoiling this acq star (within
        ``box_size``).  The output value will be 0.0 if this fid spoils this acq, otherwise
        set to 1.0 (no impact).

        This caches the values in a dict for subsequent access.

        :param acqs:
        :param box_size: search box size in arcsec
        :returns: probability multiplier (0 or 1)
        """
        assert acqs.cand_acqs is not None
        try:
            return self._p_fid_id_spoiler[box_size, fid_id]
        except KeyError:
            fids = acqs.fids
            if fids is None:
                acqs.add_warning('Requested fid spoiler probability without '
                                 'setting acqs.fids first')
                return 1.0

            p_fid_id_spoiler = 1.0
            try:
                fid = fids.cand_fids.get_id(fid_id)
            except (KeyError, IndexError, AssertionError):
                # This should not happen, but ignore with a warning in any case.  Non-candidate
                # fid cannot spoil an acq star.
                acqs.add_warning(f'Requested fid spoiler probability for fid '
                                 f'{acqs.detector}-{fid_id} but it is '
                                 f'not a candidate')
            else:
                if fids.spoils(fid, self.acq, box_size):
                    p_fid_id_spoiler = 0.0

            self._p_fid_id_spoiler[box_size, fid_id] = p_fid_id_spoiler

            return p_fid_id_spoiler


def get_p_man_err(man_err, man_angle):
    """
    Probability for given ``man_err`` given maneuver angle ``man_angle``.

    :param man_err: maneuver error (float, arcsec)
    :param man_angle: maneuver angle (float, deg)

    :returns: probability of man_error for given man_angle
    """
    pmea = ACQ.p_man_errs_angles  # [0, 5, 20, 40, 60, 80, 100, 120, 180]
    pme = ACQ.p_man_errs
    man_angle_idx = np.searchsorted(pmea, man_angle) if (man_angle > 0) else 1
    name = '{}-{}'.format(pmea[man_angle_idx - 1], pmea[man_angle_idx])

    man_err_idx = np.searchsorted(pme['man_err_hi'], man_err)
    if man_err_idx == len(pme):
        raise ValueError(f'man_err must be <= {pme["man_err_hi"]}')

    return pme[name][man_err_idx]
