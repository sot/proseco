# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import chandra_aca.aca_image
from chandra_aca.transform import mag_to_count_rate, count_rate_to_mag
from chandra_aca.aca_image import ACAImage, AcaPsfLibrary
from chandra_aca.star_probs import guide_count

from . import characteristics as CHAR
from . import characteristics_guide as GUIDE_CHAR

from .core import bin2x2, ACACatalogTable, ACABox, MetaAttribute

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
    :param stars: astropy.Table of AGASC stars (will be fetched from agasc if None)
    :param dark: ACAImage of dark map (fetched based on time and t_ccd if None)
    :param print_log: print the run log to stdout (default=False)

    :returns: GuideTable of acquisition stars
    """

    guides = GuideTable()
    guides.set_attrs_from_kwargs(obsid=obsid, **kwargs)
    guides.set_stars()

    if guides.dark is None:
        from mica.archive.aca_dark import get_dark_cal_image
        guides.log(f'getting dark cal image at date={guides.date} t_ccd={guides.t_ccd:.1f}')
        guides.dark = get_dark_cal_image(date=guides.date, select='before',
                                         t_ccd_ref=guides.t_ccd, aca_image=True)
    else:
        guides.log('Using supplied dark map (ignores t_ccd)')

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

    # Elements of meta that should not be directly serialized to YAML
    # (either too big or requires special handling).
    yaml_exclude = ('stars', 'cand_guides', 'dark')
    pickle_exclude = ('stars', 'dark')

    # Name of table.  Use to define default file names where applicable.
    # (e.g. `obs19387/guide.yaml`).
    name = 'guide'

    # Required attributes
    required_attrs = ('att', 't_ccd_guide', 'date', 'dither_guide')

    dark = MetaAttribute()
    cand_guides = MetaAttribute(is_kwarg=False)
    reject_info = MetaAttribute(default=[], is_kwarg=False)

    def reject(self, reject):
        """
        Add a reject dict to self.reject_info
        """
        reject_info = self.reject_info
        reject_info.append(reject)

    @property
    def t_ccd(self):
        return self.t_ccd_guide

    @t_ccd.setter
    def t_ccd(self, value):
        self.t_ccd_guide = value

    @property
    def dither(self):
        return self.dither_guide

    @dither.setter
    def dither(self, value):
        self.dither_guide = value

    @property
    def thumbs_up(self):
        if self.n_guide == 0:
            # If no guides were requested then always OK
            out = 1
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
        cand_guides[scol][bad_mag] += GUIDE_CHAR.errs['mag range']
        ok = ok & ~bad_mag

        # Check stage ASPQ1
        bad_aspq1 = cand_guides['ASPQ1'] > stage['ASPQ1Lim']
        cand_guides[scol][bad_aspq1] += GUIDE_CHAR.errs['aspq1']
        ok = ok & ~bad_aspq1

        # Check for bright pixels
        pixmag_lims = get_pixmag_for_offset(cand_guides['mag'],
                                            stage['Imposter']['CentroidOffsetLim'])
        # Which candidates have an 'imposter' brighter than the limit for this stage
        imp_spoil = cand_guides['imp_mag'] < pixmag_lims
        cand_guides[scol][imp_spoil] += GUIDE_CHAR.errs['hot pix']
        ok = ok & ~imp_spoil

        # Check for 'direct catalog search' spoilers
        mag_spoil = check_mag_spoilers(cand_guides, ok, stars, n_sigma)
        cand_guides[scol][mag_spoil] += GUIDE_CHAR.errs['spoiler (line)']
        ok = ok & ~mag_spoil

        # Check for star spoilers (by light) background and edge
        if stage['ASPQ1Lim'] > 0:
            bg_pix_thresh = np.percentile(dark, stage['Spoiler']['BgPixThresh'])
            reg_frac = stage['Spoiler']['RegionFrac']
            bg_spoil, reg_spoil = check_spoil_contrib(cand_guides, ok, stars,
                                                      reg_frac, bg_pix_thresh)
            cand_guides[scol][bg_spoil] += GUIDE_CHAR.errs['spoiler (bgd)']
            cand_guides[scol][reg_spoil] += GUIDE_CHAR.errs['spoiler (frac)']
            ok = ok & ~bg_spoil & ~reg_spoil

        # Check for column spoiler
        col_spoil = check_column_spoilers(cand_guides, ok, stars, n_sigma)
        cand_guides[scol][col_spoil] += GUIDE_CHAR.errs['col spoiler']
        ok = ok & ~col_spoil

        if stage['DoBminusVcheck'] == 1:
            bad_color = np.isclose(cand_guides['COLOR1'], 0.7, atol=1e-6, rtol=0)
            cand_guides[scol][bad_color] += GUIDE_CHAR.errs['bad color']
            ok = ok & ~bad_color
        return ok

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
        row_max = CCD['row_max'] - (CCD['row_pad'] + CCD['window_pad'] + r_dith_pad)
        col_max = CCD['col_max'] - (CCD['col_pad'] + CCD['window_pad'] + c_dith_pad)
        outofbounds = (np.abs(stars['row']) > row_max) | (np.abs(stars['col']) > col_max)

        cand_guides = stars[ok & ~outofbounds]
        self.log('Filtering on CLASS, mag, row/col, '
                 'mag_err, ASPQ1/2, POS_ERR:')
        self.log(f'Reduced star list from {len(stars)} to '
                 f'{len(cand_guides)} candidate guide stars')

        bp = spoiled_by_bad_pixel(cand_guides, self.dither)
        cand_guides = cand_guides[~bp]
        self.log('Filtering on candidates near bad (not just bright/hot) pixels')
        self.log(f'Reduced star list from {len(bp)} to '
                 f'{len(cand_guides)} candidate guide stars')

        bs = in_bad_star_list(cand_guides)
        cand_guides = cand_guides[~bs]
        self.log('Filtering stars on bad star list')
        self.log(f'Reduced star list from {len(bs)} to '
                 f'{len(cand_guides)} candidate guide stars')

        box_spoiled = has_spoiler_in_box(cand_guides, stars,
                                         halfbox=GUIDE_CHAR.box_spoiler['halfbox'],
                                         magdiff=GUIDE_CHAR.box_spoiler['magdiff'])
        cand_guides = cand_guides[~box_spoiled]
        self.log('Filtering stars that have bright spoilers with centroids near/in 8x8')
        self.log(f'Reduced star list from {len(box_spoiled)} to '
                 f'{len(cand_guides)} candidate guide stars')

        cand_guides['imp_mag'] = get_imposter_mags(cand_guides, dark, self.dither)
        self.log('Getting pseudo-mag of brightest pixel in candidate region')

        return cand_guides


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
    :returns: reg_spoiled, bg_spoiled - two masks on cand_stars.
    """
    fraction = regfrac
    APL = AcaPsfLibrary()
    bg_spoiled = np.zeros_like(ok)
    reg_spoiled = np.zeros_like(ok)
    bgpix = CCD['bgpix']
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
        if np.sum(on_region) > (cand_counts * fraction):
            reg_spoiled[cand_stars['id'] == cand['id']] = True
            continue

        # Or consider it spoiled if the star contribution to any background pixel
        # is more than the Nth percentile of the dark current
        for pixlabel in bgpix:
            val = on_region[pixlabel == chandra_aca.aca_image.EIGHT_LABELS][0]
            if val > bgthresh:
                bg_spoiled[cand_stars['id'] == cand['id']] = True

    return bg_spoiled, reg_spoiled


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
    :returns: bool mask of length cand_stars marking mag_spoiled stars
    """
    intercept = GUIDE_CHAR.mag_spoiler['Intercept']
    spoilslope = GUIDE_CHAR.mag_spoiler['Slope']
    magdifflim = GUIDE_CHAR.mag_spoiler['MagDiffLimit']

    # If there are already no candidates, there isn't anything to do
    if not np.any(ok):
        return np.zeros(len(ok)).astype(bool)

    mag_spoiled = np.zeros(len(ok)).astype(bool)

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
                mag_spoiled[cand['id'] == cand_stars['id']] = True
                continue

    return mag_spoiled


def check_column_spoilers(cand_stars, ok, stars, n_sigma):
    """
    For each candidate, check for stars 'MagDiff' brighter and within 'Separation' columns
    between the star and the readout register, i.e. Column Spoilers.

    :param cand_stars: Table of candidate stars
    :param ok: mask on cand_stars describing still "ok" candidates
    :param stars: Table of AGASC stars
    :param n_sigma: multiplier used when checking mag with MAG_ACA_ERR
    :returns: bool mask on cand_stars marking column spoiled stars
    """
    column_spoiled = np.zeros_like(ok)
    for idx, cand in enumerate(cand_stars):
        if not ok[idx]:
            continue
        dcol = cand['col'] - stars['col'][~stars['offchip']]
        direction = stars['row'][~stars['offchip']] / cand['row']
        pos_spoil = ((np.abs(dcol) <= (GUIDE_CHAR.col_spoiler['Separation'])) &
                     (direction > 1.0))
        if not np.count_nonzero(pos_spoil) >= 1:
            continue
        mag_errs = (n_sigma *
                    np.sqrt((cand['MAG_ACA_ERR'] * 0.01) ** 2 +
                            (stars['MAG_ACA_ERR'][~stars['offchip']][pos_spoil] * 0.01) ** 2))
        dm = (cand['mag'] - stars['mag'][~stars['offchip']][pos_spoil] + mag_errs)
        if np.any(dm > GUIDE_CHAR.col_spoiler['MagDiff']):
            column_spoiled[idx] = True
    return column_spoiled


def get_imposter_mags(cand_stars, dark, dither):
    """
    Get "pseudo-mag" of max pixel value in each candidate star region

    :param cand_stars: Table of candidate stars
    :param dark: full CCD dark map
    :param dither: observation dither to be used to determine pixels a star could use
    :returns: array of magnitudes of length cand_stars
    """
    def get_ax_range(r, extent):

        # Should come back to this and do something smarter
        # but right now I just want things that bin nicely 2x2
        rminus = int(np.floor(r - row_extent))
        rplus = int(np.ceil(r + row_extent))
        if (np.floor(r) != np.ceil(r)):
            if r - np.floor(r) > .5:
                rplus += 1
            else:
                rminus -= 1
        return rminus, rplus

    pixmags = []

    # Define the 1/2 pixel region as half the 8x8 plus dither
    row_extent = np.ceil(4 + dither.row)
    col_extent = np.ceil(4 + dither.col)
    for idx, cand in enumerate(cand_stars):
        rminus, rplus = get_ax_range(cand['row'], row_extent)
        cminus, cplus = get_ax_range(cand['col'], col_extent)
        pix = np.array(dark.aca[rminus:rplus, cminus:cplus])
        pixmax = max(np.max(bin2x2(pix)),
                     np.max(bin2x2(pix[1:-1])),
                     np.max(bin2x2(pix[:, 1:-1])),
                     np.max(bin2x2(pix[1:-1, 1:-1])))
        pixmax_mag = count_rate_to_mag(pixmax)
        pixmags.append(pixmax_mag)
    return np.array(pixmags)


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
    :returns: mask on cand_guides set to True if star spoiled by another star in the pixel box
    """
    box_spoiled = np.zeros(len(cand_guides)).astype(bool)
    for idx, cand in enumerate(cand_guides):
        dr = np.abs(cand['row'] - stars['row'])
        dc = np.abs(cand['col'] - stars['col'])
        inbox = (dr <= halfbox) & (dc <= halfbox)
        itself = stars['id'] == cand['id']
        box_spoilers = ~itself & inbox & (cand['mag'] - stars['mag'] > magdiff)
        if np.any(box_spoilers):
            box_spoiled[idx] = True
    return box_spoiled


def in_bad_star_list(cand_guides):
    """
    Mark star bad if candidate AGASC ID in bad star list.

    :param cand_guides: Table of candidate stars
    :returns: boolean mask where True means star is in bad star list
    """
    bad = np.zeros(len(cand_guides)).astype(bool)
    for star in CHAR.bad_star_list:
        bad[cand_guides['id'] == star] = True
    return bad


def spoiled_by_bad_pixel(cand_guides, dither):
    """
    Mark star bad if spoiled by a bad pixel in the bad pixel list (not hot)

    :param cand_guides: Table of candidate stars
    :param dither: tuple or list of dither where dither[0] is Y dither peak ampl
                   in arcsecs and dither[1] is Z dither peak ampl in arcsecs
    :returns: boolean mask on cand_guides where True means star is spoiled by bad pixel
    """
    if not isinstance(dither, ACABox):
        dither = ACABox(dither)

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
    row_extent = np.ceil(4 + dither.row)
    col_extent = np.ceil(4 + dither.col)
    for idx, cand in enumerate(cand_guides):
        rminus = int(np.floor(cand['row'] - row_extent))
        rplus = int(np.ceil(cand['row'] + row_extent + 1))
        cminus = int(np.floor(cand['col'] - col_extent))
        cplus = int(np.ceil(cand['col'] + col_extent + 1))

        # If any bad pixel is in the guide star pixel region, mark as spoiled
        if np.any((bp_row >= rminus) & (bp_row <= rplus) & (bp_col >= cminus) & (bp_col <= cplus)):
            spoiled[idx] = True
    return spoiled
