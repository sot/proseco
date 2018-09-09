# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Get a catalog of acquisition stars using the algorithm described in
https://docs.google.com/presentation/d/1VtFKAW9he2vWIQAnb6unpK4u1bVAVziIdX9TnqRS3a8
"""
import warnings

import numpy as np
from scipy import ndimage, stats
from astropy.table import Table

from chandra_aca.star_probs import acq_success_prob, prob_n_acq
from chandra_aca.transform import (pixels_to_yagzag, mag_to_count_rate)
from mica.archive.aca_dark.dark_cal import get_dark_cal_image

from . import characteristics as CHAR
from .core import (get_mag_std, StarsTable, ACACatalogTable, bin2x2,
                   get_image_props, pea_reject_image)


def get_acq_catalog(obsid=0, att=None,
                    man_angle=None, t_ccd=None, date=None, dither=None,
                    detector=None, sim_offset=None, focus_offset=None, stars=None,
                    include_ids=None, include_halfws=None, exclude_ids=None,
                    optimize=True, verbose=False, print_log=False):
    """
    Get a catalog of acquisition stars using the algorithm described in
    https://docs.google.com/presentation/d/1VtFKAW9he2vWIQAnb6unpK4u1bVAVziIdX9TnqRS3a8

    If ``obsid`` corresponds to an already-scheduled obsid then the parameters
    ``att``, ``man_angle``, ``t_ccd``, ``date``, and ``dither`` will
    be fetched via ``mica.starcheck`` if not explicitly provided here.

    :param obsid: obsid (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param man_angle: maneuver angle (deg)
    :param t_ccd: ACA CCD temperature (degC)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither: dither size (float, arcsec)
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
    acqs = AcqTable(print_log=print_log)

    # If an explicit obsid is not provided to all getting parameters via mica
    # then all other params must be supplied.
    all_pars = all(x is not None for x in (att, man_angle, t_ccd, date, dither))
    if obsid == 0 and not all_pars:
        raise ValueError('if `obsid` not supplied then all other params are required')

    # If not all params supplied then get via mica for the obsid.
    if not all_pars:
        from mica.starcheck import get_starcheck_catalog
        acqs.log(f'getting starcheck catalog for obsid {obsid}')

        obs = get_starcheck_catalog(obsid)
        obso = obs['obs']

        if att is None:
            att = [obso['point_ra'], obso['point_dec'], obso['point_roll']]
        if date is None:
            date = obso['mp_starcat_time']
        if t_ccd is None:
            t_ccd = obs['pred_temp']
        if man_angle is None:
            man_angle = obs['manvr']['angle_deg'][0]
        if detector is None:
            detector = obso['sci_instr']
        if sim_offset is None:
            sim_offset = obso['sim_z_offset_steps']
        if focus_offset is None:
            focus_offset = 0

        # TO DO: deal with non-square dither pattern, esp. 8 x 64.
        dither_y_amp = obso.get('dither_y_amp')
        dither_z_amp = obso.get('dither_z_amp')
        if dither_y_amp is not None and dither_z_amp is not None:
            dither = max(dither_y_amp, dither_z_amp)

        if dither is None:
            dither = 20

    acqs.log(f'getting dark cal image at date={date} t_ccd={t_ccd:.1f}')
    dark = get_dark_cal_image(date=date, select='before', t_ccd_ref=t_ccd)

    acqs.meta = {'obsid': obsid,
                 'att': att,
                 'date': date,
                 't_ccd': t_ccd,
                 'man_angle': man_angle,
                 'dither': dither,
                 'detector': detector,
                 'sim_offset': sim_offset,
                 'focus_offset': focus_offset,
                 'include_ids': include_ids or [],
                 'include_halfws': include_halfws or [],
                 'exclude_ids': exclude_ids or []}

    # Probability of man_err for this observation with a given man_angle.  Used
    # for marginalizing probabilities over different man_errs.
    acqs.meta['p_man_errs'] = np.array([get_p_man_err(man_err, acqs.meta['man_angle'])
                                        for man_err in CHAR.man_errs])

    if stars is None:
        stars = StarsTable.from_agasc(att, date=date, logger=acqs.log)
    else:
        stars = StarsTable.from_stars(att, stars, logger=acqs.log)

    cand_acqs, bad_stars = acqs.get_acq_candidates(stars)

    # Fill in the entire acq['probs'].p_acqs table (which is actual a dict of keyed by
    # (box_size, man_err) tuples).
    for acq in cand_acqs:
        acq['probs'] = AcqProbs(acqs, acq, dither, stars, dark, t_ccd, date,
                                acqs.meta['man_angle'])

    acqs.get_initial_catalog(cand_acqs, stars=stars, dark=dark, dither=dither,
                             t_ccd=t_ccd, date=date)

    acqs.meta.update({'cand_acqs': cand_acqs,
                      'stars': stars,
                      'dark': dark,
                      'bad_stars': bad_stars})

    if optimize:
        acqs.optimize_catalog(verbose)

    # Set p_acq column to be the marginalized probabilities
    acqs['p_acq'] = [acq['probs'].p_acq_marg[acq['halfw']] for acq in acqs]

    # Sort to make order match the original candidate list order (by
    # increasing mag), and assign a slot.
    acqs.sort('idx')
    acqs['slot'] = np.arange(len(acqs))

    # Add slot to cand_acqs table, putting in '...' if not selected as acq.
    # This is for convenience in downstream reporting or introspection.
    slots = [str(acqs.get_id(acq['id'])['slot']) if acq['id'] in acqs['id'] else '...'
             for acq in cand_acqs]
    cand_acqs['slot'] = slots

    return acqs


class AcqTable(ACACatalogTable):
    """
    Catalog of acquisition stars
    """
    # Elements of meta that should not be directly serialized to YAML
    # (either too big or requires special handling).
    yaml_exclude = ('stars', 'cand_acqs', 'dark', 'bad_stars')
    pickle_exclude = ('stars', 'dark', 'bad_stars')

    # Name of table.  Use to define default file names where applicable.
    # (e.g. `obs19387/acqs.yaml`).
    name = 'acqs'

    def get_obs_info(self):
        """
        Convenience method to return the parts of meta that are needed
        for test_common OBS_INFO.

        """
        keys = ('obsid', 'att', 'date', 't_ccd', 'man_angle', 'dither',
                'detector', 'sim_offset', 'focus_offset')
        return {key: self.meta[key] for key in keys}

    def get_acq_candidates(self, stars, max_candidates=20):
        """
        Get candidates for acquisition stars from ``stars`` table.

        This allows for candidates right up to the useful part of the CCD.
        The p_acq will be accordingly penalized.

        :param stars: list of stars in the field
        :param max_candidates: maximum candidate acq stars
        :returns: Table of candidates, indices of rejected stars
        """
        ok = ((stars['CLASS'] == 0) &
              (stars['mag'] > 5.9) &
              (stars['mag'] < 11.0) &
              (~np.isclose(stars['COLOR1'], 0.7)) &
              (np.abs(stars['row']) < CHAR.max_ccd_row) &  # Max usable row
              (np.abs(stars['col']) < CHAR.max_ccd_col) &  # Max usable col
              (stars['mag_err'] < 1.0) &  # Mag err < 1.0 mag
              (stars['ASPQ1'] < 20) &  # Less than 1 arcsec offset from nearby spoiler
              (stars['ASPQ2'] == 0) &  # Proper motion less than 0.5 arcsec/yr
              (stars['POS_ERR'] < 3000) &  # Position error < 3.0 arcsec
              ((stars['VAR'] == -9999) | (stars['VAR'] == 5))  # Not known to vary > 0.2 mag
              )
        # TO DO: column and row-readout spoilers (BONUS points: Mars and Jupiter)
        # Note see email with subject including "08220 (Jupiter)" about an
        # *upstream* spoiler from Jupiter.

        bads = ~ok
        cand_acqs = stars[ok]

        # If any include_ids (stars forced to be in catalog) ensure that the
        # star is in the cand_acqs table
        if self.meta.get('include_ids'):
            self.add_include_ids(cand_acqs, stars)

        cand_acqs.sort('mag')
        self.log('Filtering on CLASS, mag, COLOR1, row/col, '
                 'mag_err, ASPQ1/2, POS_ERR:')
        self.log(f'Reduced star list from {len(stars)} to '
                 f'{len(cand_acqs)} candidate acq stars')

        # Reject any candidate with a spoiler that is within a 30" HW box
        # and is within 3 mag of the candidate in brightness.  Collect a
        # list of good (not rejected) candidates and stop when there are
        # max_candidates.
        goods = []
        for ii, acq in enumerate(cand_acqs):
            bad = ((np.abs(acq['yang'] - stars['yang']) < 30) &
                   (np.abs(acq['zang'] - stars['zang']) < 30) &
                   (stars['mag'] - acq['mag'] < 3))
            if np.count_nonzero(bad) == 1:  # Self always matches
                goods.append(ii)
            if len(goods) == max_candidates:
                break

        cand_acqs = cand_acqs[goods]
        self.log('Selected {} candidates with no spoiler (star within 3 mag and 30 arcsec)'
                 .format(len(cand_acqs)))

        cand_acqs.rename_column('COLOR1', 'color')
        # Drop all the other AGASC columns.  No longer useful.
        names = [name for name in cand_acqs.colnames if not name.isupper()]
        cand_acqs = AcqTable(cand_acqs[names])

        # Make this suitable for plotting
        cand_acqs['idx'] = np.arange(len(cand_acqs), dtype=np.int64)
        cand_acqs['type'] = 'ACQ'
        cand_acqs['halfw'] = np.full(len(cand_acqs), 120, dtype=np.int64)

        # Set up columns needed for catalog initial selection and optimization
        def empty_dicts():
            return [{} for _ in range(len(cand_acqs))]

        cand_acqs['p_acq'] = -999.0  # Acq prob for box_size=halfw, marginalized over man_err
        cand_acqs['probs'] = empty_dicts()  # Dict keyed on (box_size, man_err) (mult next 3)

        cand_acqs['spoilers'] = None  # Filled in with Table of spoilers
        cand_acqs['imposters'] = None  # Filled in with Table of imposters
        cand_acqs['spoilers_box'] = -999.0  # Cached value of box_size + man_err for spoilers
        cand_acqs['imposters_box'] = -999.0  # Cached value of box_size + dither for imposters

        return cand_acqs, bads

    def add_include_ids(self, cand_acqs, stars):
        """Ensure that the cand_acqs table has stars that were forced to be included.

        :param cand_acqs: candidate acquisition stars table
        :param stars: stars table

        """
        for include_id in self.meta['include_ids']:
            if include_id not in cand_acqs['id']:
                try:
                    star = stars.get_id(include_id)
                except KeyError:
                    warnings.warn(f'Tried including star id={include_id} but this is not '
                                  f'a valid star on the ACA field of view')
                else:
                    cand_acqs.add_row(star)

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
          - If the list is 8 long (completely catalog) then stop

        This function can be called multiple times with successively smaller
        min_p_acq to fill out the catalog.  The acq_indices and box_sizes
        arrays are appended in place in this process.
        """
        self.log(f'Find stars with best acq prob for min_p_acq={min_p_acq}')
        self.log(f'Current catalog: acq_indices={acq_indices} box_sizes={box_sizes}')

        for box_size in CHAR.box_sizes:
            # Get array of marginalized (over man_err) p_acq values corresponding
            # to box_size for each of the candidate acq stars.
            p_acqs_for_box = np.array([acq['probs'].p_acq_marg[box_size] for acq in cand_acqs])

            self.log(f'Trying search box size {box_size} arcsec', level=1)

            indices = np.argsort(-p_acqs_for_box)
            for acq_idx in indices:
                if acq_idx in acq_indices:
                    continue

                acq = cand_acqs[acq_idx]
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

                if len(acq_indices) == 8:
                    self.log('Found 8 acq stars, done')
                    return

    def get_initial_catalog(self, cand_acqs, stars, dark, dither=20, t_ccd=-11.0, date=None):
        """
        Get the initial catalog of up to 8 candidate acquisition stars.
        """
        self.log(f'Getting initial catalog from {len(cand_acqs)} candidates')

        # Accumulate indices and box sizes of candidate acq stars that meet
        # successively less stringent minimum p_acq.
        acq_indices = []
        box_sizes = []
        for min_p_acq in (0.75, 0.5, 0.25, 0.05):
            # Updates acq_indices, box_sizes in place
            self.select_best_p_acqs(cand_acqs, min_p_acq, acq_indices, box_sizes)

            if len(acq_indices) == 8:
                break

        # Make all the not-accepted candidate acqs have halfw=120 as a reasonable
        # default and then set the accepted acqs to the best box_size.  Then set
        # p_acq to the marginalized acquisition probability.
        cand_acqs['halfw'] = 120
        cand_acqs['halfw'][acq_indices] = box_sizes
        for acq in cand_acqs:
            acq['p_acq'] = acq['probs'].p_acq_marg[acq['halfw']]

        # Finally select the initial catalog
        acqs_init = cand_acqs[acq_indices]

        # Transfer to acqs (which at this point is an empty table)
        for name, col in acqs_init.columns.items():
            self[name] = col

    def calc_p_safe(self, verbose=False):
        p_no_safe = 1.0

        for man_err, p_man_err in zip(CHAR.man_errs, self.meta['p_man_errs']):
            if p_man_err == 0.0:
                continue

            p_acqs = [acq['probs'].p_acqs[acq['halfw'], man_err] for acq in self]

            p_n_cum = prob_n_acq(p_acqs)[1]
            if verbose:
                self.log(f'man_err = {man_err}, p_man_err = {p_man_err}')
                self.log('p_acqs =' + ' '.join(['{:.3f}'.format(val) for val in p_acqs]))
                self.log('log10(p 1_or_fewer) = {:.2f}'.format(np.log10(p_n_cum[1])))
            p_01 = p_n_cum[1]  # 1 or fewer => p_fail at this man_err

            p_no_safe *= (1 - p_man_err * p_01)

        p_safe = 1 - p_no_safe
        self.meta['p_safe'] = p_safe

        return p_safe

    def optimize_acq_halfw(self, idx, p_safe, verbose=False):
        """
        Optimize the box size (halfw) for the acq star ``idx`` in the current acqs
        table.  Assume current ``p_safe``.
        """
        acq = self[idx]
        orig_halfw = acq['halfw']
        orig_p_acq = acq['probs'].p_acq_marg[acq['halfw']]

        self.log(f'Optimizing halfw for idx={idx} id={acq["id"]}', id=acq['id'])

        # Compute p_safe for each possible halfw for the current star
        p_safes = []
        for box_size in CHAR.box_sizes:
            new_p_acq = acq['probs'].p_acq_marg[box_size]
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
        min_halfw = CHAR.box_sizes[min_idx]

        # If p_safe went down, then consider this an improvement if either:
        #   - acq halfw is increased (bigger boxes are better)
        #   - p_safe went down by at least 10%
        # So avoid reducing box sizes for only small improvements in p_safe.
        improved = ((min_p_safe < p_safe) and
                    ((min_halfw > orig_halfw) or (min_p_safe / p_safe < 0.9)))

        p_safes_strs = [f'{np.log10(p):.2f} ({box_size}")'
                        for p, box_size in zip(p_safes, CHAR.box_sizes)]
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
        p_safe = self.calc_p_safe()
        idxs = self.argsort('p_acq')

        # Any updates made?
        any_improved = False

        for idx in idxs:
            p_safe, improved = self.optimize_acq_halfw(idx, p_safe, verbose)
            any_improved |= improved

        return p_safe, any_improved

    def optimize_catalog(self, verbose=False):
        p_safe = self.calc_p_safe(verbose=True)
        self.log('initial log10(p_safe)={:.2f}'.format(np.log10(p_safe)))

        # Start by optimizing the half-widths of the initial catalog
        for _ in range(5):
            p_safe, improved = self.optimize_acqs_halfw(verbose)
            if not improved:
                break

        self.log(f'After optimizing initial catalog p_safe = {p_safe:.5f}')

        # Now try to swap in a new star from the candidate list and see if
        # it can improve p_safe.
        acq_ids = set(self['id'])
        for cand_acq in self.meta['cand_acqs']:
            cand_id = cand_acq['id']
            if cand_id in acq_ids:
                continue
            else:
                acq_ids.add(cand_id)

            # Get the index of the worst p_acq in the catalog
            p_acqs = [acq['probs'].p_acq_marg[acq['halfw']] for acq in self]
            idx = np.argsort(p_acqs)[0]

            self.log('Trying to use {} mag={:.2f} to replace idx={} with p_acq={:.3f}'
                     .format(cand_id, cand_acq['mag'], idx, p_acqs[idx]), id=cand_id)

            # Make a copy of the row (acq star) as a numpy void (structured array row)
            orig_acq = self[idx].as_void()

            # Stub in the new candidate and get the best halfw (and corresponding new p_safe)
            self[idx] = cand_acq
            new_p_safe, improved = self.optimize_acq_halfw(idx, p_safe, verbose)

            # If the new star is noticably better (regardless of box size), OR
            # comparable but with a bigger box, then accept it and do one round of
            # full catalog optimization
            improved = ((new_p_safe / p_safe < 0.9) or
                        (new_p_safe < p_safe and self['halfw'][idx] > orig_acq['halfw']))
            if improved:
                p_safe, improved = self.optimize_acqs_halfw(verbose)
                self.calc_p_safe(verbose=True)
                self.log(f'  accepted, new p_safe = {p_safe:.5f}', id=cand_id)
            else:
                self[idx] = orig_acq

    def to_yaml_custom(self, out):
        """
        Defined by subclass to do custom process prior to YAML serialization
        and possible writing to file.  This method is called by self.to_yaml()
        and should modify the ``out`` dict in place.

        :param out: dict of pure-Python output to be serialized to YAML
        """
        # Convert cand_acqs table to a pure-Python structure
        out['cand_acqs'] = self.meta['cand_acqs'].to_struct()

    def from_yaml_custom(self, obj):
        """
        Custom processing on the dict ``obj`` which is the raw result of
        loading the YAML representation.  This gets called from the class method
        ``ACACatalogTable.from_yaml()``.

        :param obj: dict of pure-Python input from YAML to construct AcqTable
        """
        # Create candidate acqs AcqTable from pure-Python structure
        cand_acqs = obj.pop('cand_acqs')
        self.meta['cand_acqs'] = self.__class__.from_struct(cand_acqs)


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
    the way at the bottom are key.
      http://nbviewer.jupyter.org/url/cxc.harvard.edu/mta/ASPECT/
             ipynb/ssawg/2018x03x21/star-mag-uncertainties.ipynb

    TO DO: consider mag uncertainties at the faint end related to
    background subtraction and warm pixel corruption of background.
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
                       maxmag=11.5, box_size=120, bgd=40, test=False):
    """
    Note: current alg purposely avoids using the actual flight background
    calculation because this is unstable to small fluctuations in values
    and often over-estimates background.  Using this can easily miss a
    search hit that the flight ACA will detect.  So just use a mean
    dark current ``bgd``.
    """
    # Convert row/col to array index coords unless testing.
    rc_off = 0 if test else 512
    acq_row = int(star_row + rc_off)
    acq_col = int(star_col + rc_off)
    box_hw = int(box_size) // 5

    # Make sure box is within CCD
    box_r0 = np.clip(acq_row - box_hw, 0, 1024)
    box_r1 = np.clip(acq_row + box_hw, 0, 1024)
    box_c0 = np.clip(acq_col - box_hw, 0, 1024)
    box_c1 = np.clip(acq_col + box_hw, 0, 1024)

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

        out = {'row': row,
               'col': col,
               'd_row': row - star_row,
               'd_col': col - star_col,
               'yang': yang,
               'zang': zang,
               'row0': c_row - 4,
               'col0': c_col - 4,
               'img': img,
               'img_sum': img_sum,
               'mag': mag,
               'mag_err': get_mag_std(mag).item(),
               }
        outs.append(out)

    if len(outs) > 0:
        outs = Table(outs).as_array()
        outs.sort(order=['mag'])

    return outs


def calc_p_brightest_compare(acq, mags, mag_errs):
    """
    For given ``acq`` star and intruders mag, mag_err,
    do the probability calculation to see if the acq star is brighter
    than all of them.
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
    prod_sp_cdf = np.prod(sp_cdfs, axis=0)

    # Do the integral  ∫ dθ p(θ|t) Πm≠t p(θ<θt|m)
    prob = np.sum(acq_pdf * prod_sp_cdf * dx)

    return prob


def get_intruders(acq, box_size, name, n_sigma, get_func, kwargs):
    """
    Get intruders table for name='spoilers' or 'imposters') from ``acq``.
    If not already in acq then call get_func(**kwargs) to get it.

    Returns Table with cols yang, zang, mag, mag_err.
    """
    name_box = name + '_box'
    intruders = acq[name]

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
        # Ensure cached spoilers cover the current case
        if box_size > acq[name_box]:
            raise ValueError(f'box_size is greater than {name_box}')

    colnames = ['yang', 'zang', 'mag', 'mag_err']
    if len(intruders) == 0:
        intruders = {name: np.array([], dtype=np.float64) for name in colnames}
    else:
        ok = ((np.abs(intruders['yang'] - acq['yang']) < box_size) &
              (np.abs(intruders['zang'] - acq['zang']) < box_size))
        intruders = {name: intruders[name][ok] for name in ['mag', 'mag_err']}

    return intruders


def calc_p_brightest(acq, box_size, stars, dark, man_err=0,
                     dither=20, bgd=0):
    """
    Calculate the probability that the `acq` star is the brightest
    candidate in the search box.

    This caches the spoiler and imposter stars in the acqs table (the row
    corresponding to ``acq``).  It is required that the first time this is
    called that the box_size and man_err be the maximum, and this is checked.
    """
    # Spoilers
    ext_box_size = box_size + man_err
    kwargs = dict(stars=stars, acq=acq, box_size=ext_box_size)
    spoilers = get_intruders(acq, ext_box_size, 'spoilers',
                             n_sigma=2.0,  # TO DO: put to characteristics
                             get_func=get_spoiler_stars, kwargs=kwargs)

    # Imposters
    ext_box_size = box_size + dither
    kwargs = dict(star_row=acq['row'], star_col=acq['col'],
                  maxmag=acq['mag'] + acq['mag_err'],  # + 1.5, TO DO: put to characteristics
                  box_size=ext_box_size,
                  dark=dark,
                  bgd=bgd,  # TO DO deal with this
                  )
    imposters = get_intruders(acq, ext_box_size, 'imposters',
                              n_sigma=1.0,  # TO DO: put to characteristics
                              get_func=get_imposter_stars, kwargs=kwargs)

    mags = np.concatenate([spoilers['mag'], imposters['mag']])
    mag_errs = np.concatenate([spoilers['mag_err'], imposters['mag_err']])
    prob = calc_p_brightest_compare(acq, mags, mag_errs)

    return prob


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
    """
    p_on_ccd = 1.0
    half_width = box_size / 5  # half width of box in pixels
    full_width = half_width * 2

    # Require that the readout box when candidate acq star is evaluated
    # by the PEA (via a normal 8x8 readout) is fully on the CCD usable area.
    # Do so by reducing the effective CCD usable area by the readout
    # halfwidth (noting that there is a leading row before 8x8).
    max_ccd_row = CHAR.max_ccd_row - 5
    max_ccd_col = CHAR.max_ccd_col - 4

    for rc, max_rc in ((row, max_ccd_row),
                       (col, max_ccd_col)):

        # Pixel boundaries are symmetric so just take abs(row/col)
        rc1 = abs(rc) + half_width

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
    def __init__(self, acqs, acq, dither, stars, dark, t_ccd, date, man_angle):
        """
        Calculate probabilities related to acquisition, in particular an element
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

        :param acqs: acqs table
        :param acq: row in the candidate acqs table
        :param dither: dither (arcsec)
        :param stars: stars table
        :param dark: dark current map
        :param t_ccd: CCD temperature (degC)
        :param date: observation date
        """
        self.p_brightest = {}
        self.p_acq_model = {}
        self.p_on_ccd = {}
        self.p_acqs = {}
        self.p_acq_marg = {}

        for box_size in CHAR.box_sizes:
            # Need to iterate over man_errs in reverse order because calc_p_brightest
            # caches interlopers based on first call, so that needs to have the largest
            # box sizes.
            for man_err in CHAR.man_errs[::-1]:
                if man_err > box_size:
                    p_brightest = self.p_brightest[box_size, man_err] = 0.0
                else:
                    # Prob of being brightest in box (function of box_size and
                    # man_err, independently because imposter prob is just a
                    # function of box_size not man_err).  Technically also a
                    # function of dither, but that does not vary here.
                    p_brightest = calc_p_brightest(acq, box_size=box_size, stars=stars,
                                                   dark=dark, man_err=man_err, dither=dither)
                    self.p_brightest[box_size, man_err] = p_brightest

        # Acquisition probability model value (function of box_size only)
        for box_size in CHAR.box_sizes:
            p_acq_model = acq_success_prob(date=date, t_ccd=t_ccd,
                                           mag=acq['mag'], color=acq['color'],
                                           spoiler=False, halfwidth=box_size)
            self.p_acq_model[box_size] = p_acq_model

        # Probability star is in acq box (function of man_err and dither only)
        for man_err in CHAR.man_errs:
            p_on_ccd = calc_p_on_ccd(acq['row'], acq['col'], box_size=man_err + dither)
            self.p_on_ccd[man_err] = p_on_ccd

        # All together now!
        for box_size in CHAR.box_sizes:
            p_acq_marg = 0.0
            for man_err, p_man_err in zip(CHAR.man_errs, acqs.meta['p_man_errs']):
                self.p_acqs[box_size, man_err] = (self.p_brightest[box_size, man_err] *
                                                  self.p_acq_model[box_size] *
                                                  self.p_on_ccd[man_err])
                p_acq_marg += self.p_acqs[box_size, man_err] * p_man_err
            self.p_acq_marg[box_size] = p_acq_marg


def get_p_man_err(man_err, man_angle):
    """
    Probability for given ``man_err`` given maneuver angle ``man_angle``.
    """
    pmea = CHAR.p_man_errs_angles  # [0, 5, 20, 40, 60, 80, 100, 120, 180]
    pme = CHAR.p_man_errs
    man_angle_idx = np.searchsorted(pmea, man_angle) if (man_angle > 0) else 1
    name = '{}-{}'.format(pmea[man_angle_idx - 1], pmea[man_angle_idx])

    man_err_idx = np.searchsorted(pme['man_err_hi'], man_err)
    if man_err_idx == len(pme):
        raise ValueError(f'man_err must be <= {pme["man_err_hi"]}')

    return pme[name][man_err_idx]
