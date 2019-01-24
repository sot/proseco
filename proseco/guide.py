# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import chandra_aca.aca_image
from chandra_aca.transform import mag_to_count_rate, count_rate_to_mag
from chandra_aca.aca_image import ACAImage, AcaPsfLibrary
from chandra_aca.star_probs import guide_count

from . import characteristics as CHAR
from . import characteristics_guide as GUIDE_CHAR

from .core import bin2x2, ACACatalogTable, MetaAttribute, AliasAttribute

CCD = GUIDE_CHAR.CCD


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

    guides = GuideTable()
    guides.set_attrs_from_kwargs(obsid=obsid, **kwargs)
    guides.set_stars()

    # Do a first cut of the stars to get a set of reasonable candidates
    guides.cand_guides = guides.get_initial_guide_candidates()

    # Run through search stages to select stars
    selected = guides.run_search_stages()

    # Transfer to table (which at this point is an empty table)
    for name, col in selected.columns.items():
        guides[name] = col

    guides['idx'] = np.arange(len(guides))

    if len(guides) < guides.n_guide:
        guides.log(f'Selected only {len(guides)} guide stars versus requested {guides.n_guide}',
                   warning=True)

    return guides


class GuideTable(ACACatalogTable):

    # Elements of meta that should not be directly serialized to pickle.
    # (either too big or requires special handling).
    pickle_exclude = ('stars', 'dark')

    # Name of table.  Use to define default file names where applicable.
    # (e.g. `obs19387/guide.pkl`).
    name = 'guide'

    allowed_kwargs = ACACatalogTable.allowed_kwargs | set(['fids'])

    # Required attributes
    required_attrs = ('att', 't_ccd_guide', 'date', 'dither_guide', 'n_guide')

    cand_guides = MetaAttribute(is_kwarg=False)
    reject_info = MetaAttribute(default=[], is_kwarg=False)

    def reject(self, reject):
        """
        Add a reject dict to self.reject_info
        """
        reject_info = self.reject_info
        reject_info.append(reject)

    t_ccd = AliasAttribute()
    dither = AliasAttribute()
    include_ids = AliasAttribute()
    exclude_ids = AliasAttribute()

    @property
    def thumbs_up(self):
        if self.n_guide == 0:
            # If no guides were requested then always OK
            out = 1
        elif len(self) == 0:
            out = 0
        else:
            # Evaluate guide catalog quality for thumbs_up
            count = guide_count(self['mag'], self.t_ccd)
            out = int(count >= GUIDE_CHAR.min_guide_count)
        return out

    def run_search_stages(self):
        """
        Run through search stages to select stars with priority given to "better"
        stars in the stages.
        """
        cand_guides = self.cand_guides
        self.log("Starting search stages")
        if len(cand_guides) == 0:
            self.log("There are no candidates to check in stages.  Exiting")
            # Since there are no candidate stars, this returns the empty set of
            # cand_guides as the 'selected' stars.
            return cand_guides
        cand_guides['stage'] = -1
        # Force stars in include_ids to be selected at stage 0
        for star_id in self.include_ids:
            cand_guides['stage'][cand_guides['id'] == star_id] = 0
            self.log(f'{star_id} selected in stage 0 via include_ids', level=1)
        n_guide = self.n_guide
        for idx, stage in enumerate(GUIDE_CHAR.stages, 1):
            already_selected = np.count_nonzero(cand_guides['stage'] != -1)

            # If we don't have enough stage-selected candidates, keep going
            if already_selected < n_guide:
                stage_ok = self.search_stage(stage)
                sel = cand_guides['stage'] == -1
                cand_guides['stage'][stage_ok & sel] = idx
                stage_selected = np.count_nonzero(stage_ok & sel)
                self.log(f'{stage_selected} stars selected in stage {idx}', level=1)
            else:
                self.log(f'Quitting after stage {idx - 1} with {already_selected} stars', level=1)
                break
        self.log('Done with search stages')
        selected = cand_guides[cand_guides['stage'] != -1]
        selected.sort(['stage', 'mag'])
        if len(selected) >= self.n_guide:
            return selected[0:n_guide]
        if len(selected) < self.n_guide:
            self.log(f'Could not find {self.n_guide} candidates after all search stages')
            return selected

    def search_stage(self, stage):
        """
        Review the candidates with the criteria defined in ``stage`` and return a mask that
        marks the candidates that are "ok" for this search stage.

        This also marks up self.meta.cand_guides with a new column 'stat_{n_stage}' with a bit
        mask of the errors that the star accrued during this search stage.

        :param stage: dictionary of search stage parameters (needs to be defined)
        :returns: bool mask of the length of self.meta.cand_guides
        """

        cand_guides = self.cand_guides
        stars = self.stars
        dark = self.dark
        ok = np.ones(len(cand_guides)).astype(bool)

        # Adopt the SAUSAGE convention of a bit array for errors
        # Not all items will be checked for each star (allow short circuit)
        scol = 'stat_{}'.format(stage['Stage'])
        cand_guides[scol] = 0

        n_sigma = stage['SigErrMultiplier']

        # Check reasonable mag
        bright_lim = stage['MagLimit'][0]
        faint_lim = stage['MagLimit'][1]
        bad_mag = (((cand_guides['mag'] - n_sigma * cand_guides['mag_err']) < bright_lim) |
                   ((cand_guides['mag'] + n_sigma * cand_guides['mag_err']) > faint_lim))
        for idx in np.flatnonzero(bad_mag):
            self.reject({'id': cand_guides['id'][idx],
                         'type': 'mag outside range',
                         'stage': stage['Stage'],
                         'bright_lim': bright_lim,
                         'faint_lim': faint_lim,
                         'cand_mag': cand_guides['mag'][idx],
                         'cand_mag_err_times_sigma': n_sigma * cand_guides['mag_err'][idx],
                         'text': f'Cand {cand_guides["id"][idx]} rejected with mag outside range for stage'})
        cand_guides[scol][bad_mag] += GUIDE_CHAR.errs['mag range']
        ok = ok & ~bad_mag

        # Check stage ASPQ1
        bad_aspq1 = cand_guides['ASPQ1'] > stage['ASPQ1Lim']
        for idx in np.flatnonzero(bad_aspq1):
            self.reject({'id': cand_guides['id'][idx],
                         'type': 'aspq1 outside range',
                         'stage': stage['Stage'],
                         'aspq1_lim': stage['ASPQ1Lim'],
                         'text': f'Cand {cand_guides["id"][idx]} rejected with aspq1 > {stage["ASPQ1Lim"]}'})
        cand_guides[scol][bad_aspq1] += GUIDE_CHAR.errs['aspq1']
        ok = ok & ~bad_aspq1

        # Check for bright pixels
        pixmag_lims = get_pixmag_for_offset(cand_guides['mag'],
                                            stage['Imposter']['CentroidOffsetLim'])
        # Which candidates have an 'imposter' brighter than the limit for this stage
        imp_spoil = cand_guides['imp_mag'] < pixmag_lims
        for idx in np.flatnonzero(imp_spoil):
            cand = cand_guides[idx]
            cen_limit = stage['Imposter']['CentroidOffsetLim']
            self.reject({
                'id': cand['id'],
                'stage': stage['Stage'],
                'type': 'hot pixel',
                'centroid_offset_thresh': cen_limit,
                'pseudo_mag_for_thresh': pixmag_lims[idx],
                'imposter_mag': cand['imp_mag'],
                'imp_row_start': cand['imp_r'],
                'imp_col_start': cand['imp_c'],
                'text': (f'Cand {cand["id"]} mag {cand["mag"]:.1f} imposter with "mag" {cand["imp_mag"]:.1f} '
                         f'limit {pixmag_lims[idx]:.2f} with offset lim {cen_limit} at stage')})
        cand_guides[scol][imp_spoil] += GUIDE_CHAR.errs['hot pix']
        ok = ok & ~imp_spoil

        # Check for 'direct catalog search' spoilers
        mag_spoil, mag_rej = check_mag_spoilers(cand_guides, ok, stars, n_sigma)
        for rej in mag_rej:
            rej['stage'] = stage['Stage']
            self.reject(rej)
        cand_guides[scol][mag_spoil] += GUIDE_CHAR.errs['spoiler (line)']
        ok = ok & ~mag_spoil

        # Check for star spoilers (by light) background and edge
        if stage['ASPQ1Lim'] > 0:
            bg_pix_thresh = np.percentile(dark, stage['Spoiler']['BgPixThresh'])
            reg_frac = stage['Spoiler']['RegionFrac']
            bg_spoil, reg_spoil, light_rej = check_spoil_contrib(cand_guides, ok, stars,
                                                                 reg_frac, bg_pix_thresh)
            for rej in light_rej:
                rej['stage'] = stage['Stage']
                self.reject(rej)
            cand_guides[scol][bg_spoil] += GUIDE_CHAR.errs['spoiler (bgd)']
            cand_guides[scol][reg_spoil] += GUIDE_CHAR.errs['spoiler (frac)']
            ok = ok & ~bg_spoil & ~reg_spoil

        # Check for column spoiler
        col_spoil, col_rej = check_column_spoilers(cand_guides, ok, stars, n_sigma)
        for rej in col_rej:
            rej['stage'] = stage['Stage']
            self.reject(rej)
        cand_guides[scol][col_spoil] += GUIDE_CHAR.errs['col spoiler']
        ok = ok & ~col_spoil

        if stage['DoBminusVcheck'] == 1:
            bad_color = np.isclose(cand_guides['COLOR1'], 0.7, atol=1e-6, rtol=0)
            for idx in np.flatnonzero(bad_color):
                self.reject({'id': cand_guides['id'][idx],
                             'type': 'bad color',
                             'stage': stage['Stage'],
                             'text': f'Cand {cand_guides["id"][idx]} has bad color (0.7)'})
            cand_guides[scol][bad_color] += GUIDE_CHAR.errs['bad color']
            ok = ok & ~bad_color
        return ok

    def process_include_ids(self, cand_guides, stars):
        """Ensure that the cand_guides table has stars that were forced to be included.

        Also do validation of include_ids

        :param cand_guides: candidate guide stars table
        :param stars: stars table

        """
        for include_id in self.include_ids:
            if include_id not in cand_guides['id']:
                try:
                    star = stars.get_id(include_id)
                    if ((star['CLASS'] != 0) |
                        (np.abs(star['row']) >= CHAR.max_ccd_row) |
                        (np.abs(star['col']) >= CHAR.max_ccd_col)):
                        raise ValueError("Not a valid candidate")
                except (ValueError, KeyError):
                    raise ValueError(f'cannot include star id={include_id} that is not '
                                     f'a valid star in the ACA field of view')
                else:
                    cand_guides.add_row(star)
                    self.log(f'Included star id={include_id} put in cand_guides')

    def get_initial_guide_candidates(self):
        """
        Create a candidate list from the available stars in the field.
        """
        stars = self.stars
        dark = self.dark

        # Use the primary selection filter from acq, but allow bad color
        # and limit to brighter stars
        ok = ((stars['CLASS'] == 0) &
              (stars['mag'] > 5.9) &
              (stars['mag'] < 10.3) &
              (np.abs(stars['row']) < CHAR.max_ccd_row) &  # Max usable row
              (np.abs(stars['col']) < CHAR.max_ccd_col) &  # Max usable col
              (stars['mag_err'] < 1.0) &  # Mag err < 1.0 mag
              (stars['ASPQ1'] < 20) &  # Less than 1 arcsec offset from nearby spoiler
              (stars['ASPQ2'] == 0) &  # Proper motion less than 0.5 arcsec/yr
              (stars['POS_ERR'] < 3000) &  # Position error < 3.0 arcsec
              ((stars['VAR'] == -9999) | (stars['VAR'] == 5))  # Not known to vary > 0.2 mag
              )

        # Mark stars that are off chip
        offchip = (np.abs(stars['row']) > CCD['row_max']) | (np.abs(stars['col']) > CCD['col_max'])
        stars['offchip'] = offchip

        # Add a filter for stars that are too close to the chip edge including dither
        r_dith_pad = self.dither.row
        c_dith_pad = self.dither.col
        row_max = CCD['row_max'] - (CCD['row_pad'] + CCD['window_pad'] + CCD['guide_extra_pad'] +
                                    r_dith_pad)
        col_max = CCD['col_max'] - (CCD['col_pad'] + CCD['window_pad'] + CCD['guide_extra_pad'] +
                                    c_dith_pad)
        outofbounds = (np.abs(stars['row']) > row_max) | (np.abs(stars['col']) > col_max)

        cand_guides = stars[ok & ~outofbounds]
        self.log('Filtering on CLASS, mag, row/col, '
                 'mag_err, ASPQ1/2, POS_ERR:')
        self.log(f'Reduced star list from {len(stars)} to '
                 f'{len(cand_guides)} candidate guide stars')

        bp, bp_rej = spoiled_by_bad_pixel(cand_guides, self.dither)
        for rej in bp_rej:
            rej['stage'] = 0
            self.reject(rej)
        cand_guides = cand_guides[~bp]
        self.log('Filtering on candidates near bad (not just bright/hot) pixels')
        self.log(f'Reduced star list from {len(bp)} to '
                 f'{len(cand_guides)} candidate guide stars')

        bs = in_bad_star_list(cand_guides)
        for idx in np.flatnonzero(bs):
            self.reject({'id': cand_guides['id'][idx],
                         'stage': 0,
                         'type': 'bad star list',
                         'text': f'Cand {cand_guides["id"][idx]} in bad star list'})
        cand_guides = cand_guides[~bs]
        self.log('Filtering stars on bad star list')
        self.log(f'Reduced star list from {len(bs)} to '
                 f'{len(cand_guides)} candidate guide stars')

        box_spoiled, box_rej = has_spoiler_in_box(cand_guides, stars,
                                                  halfbox=GUIDE_CHAR.box_spoiler['halfbox'],
                                                  magdiff=GUIDE_CHAR.box_spoiler['magdiff'])
        for rej in box_rej:
            rej['stage'] = 0
            self.reject(rej)
        cand_guides = cand_guides[~box_spoiled]
        self.log('Filtering stars that have bright spoilers with centroids near/in 8x8')
        self.log(f'Reduced star list from {len(box_spoiled)} to '
                 f'{len(cand_guides)} candidate guide stars')

        fid_trap_spoilers, fid_rej = check_fid_trap(cand_guides, fids=self.fids,
                                                    dither=self.dither)
        for rej in fid_rej:
            rej['stage'] = 0
            self.reject(rej)
        cand_guides = cand_guides[~fid_trap_spoilers]

        # Deal with include_ids by putting them back in candidate table if necessary
        self.process_include_ids(cand_guides, stars)

        # Deal with exclude_ids by cutting from the candidate list
        for star_id in self.exclude_ids:
            if star_id in cand_guides['id']:
                self.reject({'stage': 0,
                             'type': 'exclude_id',
                             'id': star_id,
                             'text': f'Cand {star_id} rejected.  In exclude_ids'})
                cand_guides = cand_guides[cand_guides['id'] != star_id]

        # Get the brightest 2x2 in the dark map for each candidate and save value and location
        imp_mag, imp_row, imp_col = get_imposter_mags(cand_guides, dark, self.dither)
        cand_guides['imp_mag'] = imp_mag
        cand_guides['imp_r'] = imp_row
        cand_guides['imp_c'] = imp_col
        self.log('Getting pseudo-mag of brightest pixel 2x2 in candidate region')

        return cand_guides


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

    bad_row = GUIDE_CHAR.fid_trap['row']
    bad_col = GUIDE_CHAR.fid_trap['col']
    fid_margin = GUIDE_CHAR.fid_trap['margin']

    # Check to see if the fid is in the zone that's a problem for the trap and if there's
    # a star that can cause the effect in the readout regiser
    for fid in fids:
        incol = abs(fid['col'] - bad_col) < fid_margin
        inrow = fid['row'] < 0 and fid['row'] > bad_row
        if incol and inrow:
            fid_dist_to_trap = fid['row'] - bad_row
            star_dist_to_register = 512 - abs(cand_stars['row'])
            spoils = abs(fid_dist_to_trap - star_dist_to_register) < (fid_margin + dither.row)
            spoilers = spoilers | spoils
            for idx in np.flatnonzero(spoils):
                cand = cand_stars[idx]
                rej.append({'id': cand['id'],
                            'type': 'fid trap effect',
                            'fid_id': fid['id'],
                            'fid_dist_to_trap': fid_dist_to_trap,
                            'star_dist_to_register': star_dist_to_register[idx],
                            'text': f'Cand {cand["id"]} in trap zone for fid {fid["id"]}'})
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
    :returns: reg_spoiled, bg_spoiled, rej - two masks on cand_stars and a list of reject debug dicts
    """
    fraction = regfrac
    APL = AcaPsfLibrary()
    bg_spoiled = np.zeros_like(ok)
    reg_spoiled = np.zeros_like(ok)
    bgpix = CCD['bgpix']
    rej = []
    for cand in cand_stars[ok]:
        if cand['ASPQ1'] == 0:
            continue
        spoilers = ((np.abs(cand['row'] - stars['row']) < 9) &
                    (np.abs(cand['col'] - stars['col']) < 9))

        # If there is only one match, it is the candidate so there's nothing to do
        if np.count_nonzero(spoilers) == 1:
            continue
        cand_counts = mag_to_count_rate(cand['mag'])

        # Get a reasonable AcaImage for the location of the 8x8 for the candidate
        cand_img_region = ACAImage(np.zeros((8, 8)),
                                   row0=int(round(cand['row'])) - 4,
                                   col0=int(round(cand['col'])) - 4)
        on_region = cand_img_region
        for spoil in stars[spoilers]:
            if spoil['id'] == cand['id']:
                continue
            spoil_img = APL.get_psf_image(row=spoil['row'], col=spoil['col'], pix_zero_loc='edge',
                                          norm=mag_to_count_rate(spoil['mag']))
            on_region = on_region + spoil_img.aca

        # Consider it spoiled if the star contribution on the 8x8 is over a fraction
        frac_limit = cand_counts * fraction
        sum_in_region = np.sum(on_region)
        if sum_in_region > frac_limit:
            reg_spoiled[cand_stars['id'] == cand['id']] = True
            rej.append({'id': cand_stars['id'],
                        'type': 'region sum spoiled',
                        'limit_for_star': frac_limit,
                        'fraction': fraction,
                        'sum_in_region': sum_in_region,
                        'text': f'Cand {cand_stars["id"]} has too much contribution to region from spoilers'})
            continue

        # Or consider it spoiled if the star contribution to any background pixel
        # is more than the Nth percentile of the dark current
        for pixlabel in bgpix:
            val = on_region[pixlabel == chandra_aca.aca_image.EIGHT_LABELS][0]
            if val > bgthresh:
                bg_spoiled[cand_stars['id'] == cand['id']] = True
                rej.append({'id': cand['id'],
                            'type': 'region background spoiled',
                            'bg_thresh': bgthresh,
                            'bg_pix_val': val,
                            'pix_id': pixlabel,
                            'text': f'Cand {cand["id"]} has bg pix spoiled by spoilers'})
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
    intercept = GUIDE_CHAR.mag_spoiler['Intercept']
    spoilslope = GUIDE_CHAR.mag_spoiler['Slope']
    magdifflim = GUIDE_CHAR.mag_spoiler['MagDiffLimit']

    # If there are already no candidates, there isn't anything to do
    if not np.any(ok):
        return np.zeros(len(ok)).astype(bool), []

    mag_spoiled = np.zeros(len(ok)).astype(bool)
    rej = []
    for cand in cand_stars[ok]:
        pix_dist = np.sqrt(((cand['row'] - stars['row']) ** 2) +
                           ((cand['col'] - stars['col']) ** 2))
        spoilers = ((np.abs(cand['row'] - stars['row']) < 10) &
                    (np.abs(cand['col'] - stars['col']) < 10))

        # If there is only one match, it is the candidate so there's nothing to do
        if np.count_nonzero(spoilers) == 1:
            continue

        for spoil, dist in zip(stars[spoilers], pix_dist[spoilers]):
            if spoil['id'] == cand['id']:
                continue
            if (cand['mag'] - spoil['mag']) < magdifflim:
                continue
            mag_err_sum = np.sqrt((cand['MAG_ACA_ERR'] * 0.01) ** 2 +
                                  (spoil['MAG_ACA_ERR'] * 0.01) ** 2)
            delmag = cand['mag'] - spoil['mag'] + n_sigma * mag_err_sum
            thsep = intercept + delmag * spoilslope
            if dist < thsep:
                rej.append({'id': cand['id'],
                            'spoiler': spoil['id'],
                            'spoiler_mag': spoil['mag'],
                            'dmag_with_err': delmag,
                            'min_dist_for_dmag': thsep,
                            'actual_dist': dist,
                            'type': 'spoiler by distance-mag line',
                            'text': (f'Cand {cand["id"]} spoiled by {spoil["id"]}, '
                                     f'too close ({dist:.1f}) pix for magdiff ({delmag:.1f})')})
                mag_spoiled[cand['id'] == cand_stars['id']] = True
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
            (np.sign(cand['row']) == np.sign(stars['row'][~stars['offchip']])) &
            (np.abs(cand['row']) < np.abs(stars['row'][~stars['offchip']])) &
            (np.abs(cand['col'] - stars['col'][~stars['offchip']]) <= CHAR.col_spoiler_pix_sep))
        if not np.count_nonzero(pos_spoil) >= 1:
            continue

        mag_errs = (n_sigma *
                    np.sqrt((cand['MAG_ACA_ERR'] * 0.01) ** 2 +
                            (stars['MAG_ACA_ERR'][~stars['offchip']][pos_spoil] * 0.01) ** 2))
        dm = (cand['mag'] - stars['mag'][~stars['offchip']][pos_spoil] + mag_errs)
        spoils = dm > CHAR.col_spoiler_mag_diff
        if np.any(spoils):
            column_spoiled[idx] = True
            spoiler = stars[~stars['offchip']][pos_spoil][spoils][0]
            rej.append({'id': cand['id'],
                        'type': 'column spoiled',
                        'spoiler': spoiler['id'],
                        'spoiler_mag': spoiler['mag'],
                        'dmag_with_err': dm[spoils][0],
                        'dmag_lim': CHAR.col_spoiler_mag_diff,
                        'dcol': cand['col'] - spoiler['col'],
                        'text': (f'Cand {cand["id"]} has column spoiler {spoiler["id"]} '
                                 f'at ({spoiler["row"]:.1f}, {spoiler["row"]:.1f}), '
                                 f'mag {spoiler["mag"]:.2f}')})
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
    row_extent = 4 + GUIDE_CHAR.dither_pix_pad + dither.row
    col_extent = 4 + GUIDE_CHAR.dither_pix_pad + dither.col
    for cand in cand_stars:
        rminus, rplus = get_ax_range(cand['row'], row_extent)
        cminus, cplus = get_ax_range(cand['col'], col_extent)
        pix = np.array(dark.aca[rminus:rplus, cminus:cplus])
        pixmax = 0
        max_r = None
        max_c = None
        # Check the 2x2 bins for the max 2x2 region.  Search the "offset" versions as well
        for pix_chunk, row_off, col_off in zip((pix, pix[1:-1, :], pix[:, 1:-1], pix[1:-1, 1:-1]),
                                               (0, 1, 0, 1),
                                               (0, 0, 1, 1)):
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
        dr = np.abs(cand['row'] - stars['row'])
        dc = np.abs(cand['col'] - stars['col'])
        inbox = (dr <= halfbox) & (dc <= halfbox)
        itself = stars['id'] == cand['id']
        box_spoilers = ~itself & inbox & (cand['mag'] - stars['mag'] > magdiff)
        if np.any(box_spoilers):
            box_spoiled[idx] = True
            n = np.count_nonzero(box_spoilers)
            boxsize = halfbox * 2
            bright = np.argmin(stars[box_spoilers]['mag'])
            spoiler = stars[box_spoilers][bright]
            rej.append({'id': cand['id'],
                        'type': 'in-box spoiler star',
                        'boxsize': boxsize,
                        'magdiff_thresh': magdiff,
                        'spoiler': spoiler['id'],
                        'dmag': cand['mag'] - spoiler['mag'],
                        'n': n,
                        'text': (f'Cand {cand["id"]} spoiled by {n} stars in {boxsize}x{boxsize} '
                                 f' including {spoiler["id"]}')
                    })
    return box_spoiled, rej


def in_bad_star_list(cand_guides):
    """
    Mark star bad if candidate AGASC ID in bad star list.

    :param cand_guides: Table of candidate stars
    :returns: boolean mask where True means star is in bad star list
    """
    bad = [cand_guide['id'] in CHAR.bad_star_set for cand_guide in cand_guides]
    return np.array(bad)


def spoiled_by_bad_pixel(cand_guides, dither):
    """
    Mark star bad if spoiled by a bad pixel in the bad pixel list (not hot)

    :param cand_guides: Table of candidate stars
    :param dither: dither ACABox
    :returns: boolean mask on cand_guides where True means star is spoiled by bad pixel,
              list of dicts of reject debug info
    """

    raw_bp = np.array(CHAR.bad_pixels)
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
        rminus = int(np.floor(cand['row'] - row_extent))
        rplus = int(np.ceil(cand['row'] + row_extent + 1))
        cminus = int(np.floor(cand['col'] - col_extent))
        cplus = int(np.ceil(cand['col'] + col_extent + 1))

        # If any bad pixel is in the guide star pixel region, mark as spoiled
        bps = ((bp_row >= rminus) & (bp_row <= rplus) & (bp_col >= cminus) & (bp_col <= cplus))
        if np.any(bps):
            spoiled[idx] = True
            rej.append({'id': cand['id'],
                        'type': 'bad pixel',
                        'pixel': (bp_row[bps][0], bp_col[bps][0]),
                        'n_bad': np.count_nonzero(bps),
                        'text': (f'Cand {cand["id"]} spoiled by {np.count_nonzero(bps)} bad pixels '
                                 f'including {(bp_row[bps][0], bp_col[bps][0])}')})
    return spoiled, rej
