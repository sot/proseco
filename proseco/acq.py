# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import division, print_function, absolute_import  # For Py2 compatibility

import inspect
import time
import numpy as np
from collections import namedtuple

from chandra_aca.star_probs import acq_success_prob, prob_n_acq
from mica.archive.aca_dark.dark_cal import get_dark_cal_image
from Ska.quatutil import radec2yagzag
from chandra_aca.transform import (yagzag_to_pixels, pixels_to_yagzag,
                                   mag_to_count_rate, count_rate_to_mag)
from agasc import get_agasc_cone
from astropy.table import Table, Column, vstack
from Quaternion import Quat
from scipy import ndimage, stats
from scipy.interpolate import interp1d
from mica.cache import lru_cache

from . import characteristics as CHAR


Event = namedtuple('Event', ['dt', 'func', 'descr', 'level'])


# Global used to collect diagnostic information during acq star selection.
# This gets cleared at the beginning of get_acq_catalog.  Perhaps this
# will go away if/when processing here is refactored into a container class.
class Info(dict):
    def __init__(self):
        self['events'] = []
        self.time0 = time.time()

    def clear(self):
        super(Info, self).clear()
        self.__init__()

    def log(self, descr, level='info'):
        calling_func_name = inspect.currentframe().f_back.f_code.co_name
        dt = time.time() - self.time0
        self['events'].append(Event(dt=round(dt, 3),
                                    func=calling_func_name,
                                    descr=descr,
                                    level=level))


INFO = Info()


class AcqTable(Table):
    @property
    def _non_object_table(self):
        names = [name for name in self.colnames
                 if self[name].dtype.kind != 'O']
        return self[names]

    def _repr_html_(self):
        return Table._repr_html_(self._non_object_table)

    def __repr__(self):
        return Table.__repr__(self._non_object_table)

    def __str__(self):
        return Table.__str__(self._non_object_table)

    def __bytes__(self):
        return Table.__bytes__(self._non_object_table)

    def to_struct(self):
        """Turn table into a list of dict (rows)"""
        rows = []
        colnames = self.colnames
        for row in self:
            outrow = {}
            for name in colnames:
                val = row[name]
                if isinstance(val, np.ndarray):
                    if val.ndim == 1:
                        val = val.item()
                    else:
                        # Skip non-scalar values
                        continue
                elif isinstance(val, Table):
                    val = AcqTable.to_struct(val)
                outrow[name] = val
            rows.append(outrow)
        return rows

    @classmethod
    def from_struct(cls, rows):
        out = cls(rows)
        for name in out.colnames:
            col = out[name]
            if col.dtype.kind == 'O':
                for idx, val in enumerate(col):
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        col[idx] = Table(val)
        return out


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
        raise ValueError('man_err must be <= {}'.format(pme['man_err_hi']))

    return pme[name][man_err_idx]


# AGASC columns not needed (at this moment) for acq star selection.
# Change as needed.
AGASC_COLS_DROP = [
    'RA',
    'DEC',
    'POS_CATID',
    'EPOCH',
    'PM_CATID',
    'PLX',
    'PLX_ERR',
    'PLX_CATID',
    'MAG',
    'MAG_ERR',
    'MAG_BAND',
    'MAG_CATID',
    'C1_CATID',
    'COLOR2',
    'COLOR2_ERR',
    'C2_CATID',
    'RSV1',
    'RSV2',
    'RSV3',
    'VAR_CATID',
    'ACQQ1',
    'ACQQ2',
    'ACQQ3',
    'ACQQ4',
    'ACQQ5',
    'ACQQ6',
    'XREF_ID1',
    'XREF_ID2',
    'XREF_ID3',
    'XREF_ID4',
    'XREF_ID5',
    'RSV4',
    'RSV5',
    'RSV6',
]

# Define a function that returns the observed standard deviation in ACA mag
# as a function of catalog mag.  This is about the std-dev for samples
# within an observation, NOT the error on the catalog mag (which is about
# the error on mean of the sample, not the std-dev of the sample).
#
# See skanb/star_selection/star-mag-std-dev.ipynb for the source of numbers.
#
# Note that get_mag_std is a function that is called with get_mag_std(mag_aca).
get_mag_std = interp1d(x=[-10, 6.7, 7.3, 7.8, 8.3, 8.8, 9.2, 9.7, 10.1, 11, 20],  # mag
                       y=[0.015, 0.015, 0.022, 0.029, 0.038, 0.055, 0.077,
                          0.109, 0.141, 0.23, 1.0],  # std-dev
                       kind='linear')


def get_stars(att, date=None, radius=1.2):
    """
    Get AGASC stars in the ACA FOV.  This uses the mini-AGASC, so only stars
    within 3-sigma of 11.5 mag.  TO DO: maybe use the full AGASC, for faint
    candidate acq stars with large uncertainty.
    """
    q_att = Quat(att)
    INFO.log('getting stars at ra={:.5f} dec={:.4f} ..'
             .format(q_att.ra, q_att.dec))
    stars = get_agasc_cone(q_att.ra, q_att.dec, radius=radius, date=date)
    INFO.log(' got {} stars'.format(len(stars)))
    yag, zag = radec2yagzag(stars['RA_PMCORR'], stars['DEC_PMCORR'], q_att)
    yag *= 3600
    zag *= 3600
    row, col = yagzag_to_pixels(yag, zag, allow_bad=True, pix_zero_loc='edge')

    stars.remove_columns(AGASC_COLS_DROP)

    stars.add_column(Column(stars['RA_PMCORR'], name='ra'), index=1)
    stars.add_column(Column(stars['DEC_PMCORR'], name='dec'), index=2)
    stars.add_column(Column(yag, name='yang'), index=3)
    stars.add_column(Column(zag, name='zang'), index=4)
    stars.add_column(Column(row, name='row'), index=5)
    stars.add_column(Column(col, name='col'), index=6)

    stars.add_column(Column(stars['MAG_ACA'], name='mag'), index=7)  # For convenience

    # Mag_err column is the RSS of the catalog mag err (i.e. systematic error in
    # the true star mag) and the sample uncertainty in the ACA readout mag
    # for a star with mag=MAG_ACA.  The latter typically dominates above 9th mag.
    mag_aca_err = stars['MAG_ACA_ERR'] * 0.01
    mag_std_dev = get_mag_std(stars['MAG_ACA'])
    mag_err = np.sqrt(mag_aca_err ** 2 + mag_std_dev ** 2)
    stars.add_column(Column(mag_err, name='mag_err'), index=8)

    # Make printed table look nicer
    for name in ('yang', 'zang', 'row', 'col', 'mag', 'mag_err'):
        stars[name].format = '.2f'
    for name in ('ra', 'dec', 'RA_PMCORR', 'DEC_PMCORR'):
        stars[name].format = '.6f'

    # Filter stars in or near ACA FOV
    rcmax = 512.0 + 200 / 5  # 200 arcsec padding around CCD edge
    ok = (row > -rcmax) & (row < rcmax) & (col > -rcmax) & (col < rcmax)
    stars = stars[ok]

    INFO.log('finished star processing', level='verbose')

    return stars


def get_acq_candidates(stars, max_candidates=20):
    """
    Get candidates for acquisition stars from ``stars`` table.

    This allows for candidates right up to the useful part of the CCD.
    The p_acq will be accordingly penalized.
    """
    ok = ((stars['CLASS'] == 0) &
          (stars['MAG_ACA'] > 5.9) &
          (stars['MAG_ACA'] < 11.0) &
          (~np.isclose(stars['COLOR1'], 0.7)) &
          (np.abs(stars['row']) < CHAR.max_ccd_row) &  # Max usable row
          (np.abs(stars['col']) < CHAR.max_ccd_col) &  # Max usable col
          (stars['MAG_ACA_ERR'] < 100) &  # Mag err < 1.0 mag
          (stars['ASPQ1'] < 20) &  # Less than 1 arcsec offset from nearby spoiler
          (stars['ASPQ2'] == 0) &  # Proper motion less than 0.5 arcsec/yr
          (stars['POS_ERR'] < 3000)  # Position error < 3.0 arcsec
          )
    # TO DO: column and row-readout spoilers (BONUS points: Mars and Jupiter)
    # Note see email with subject including "08220 (Jupiter)" about an
    # *upstream* spoiler from Jupiter.

    bads = ~ok
    acqs = stars[ok]
    acqs.sort('MAG_ACA')
    INFO.log('filtering on CLASS, MAG_ACA, COLOR1, row/col, '
             'MAG_ACA_ERR, ASPQ1/2, POS_ERR:\n'
             'reduced star list from {} to {} candidate acq stars'
             .format(len(stars), len(acqs)))

    # Reject any candidate with a spoiler that is within a 30" HW box
    # and is within 3 mag of the candidate in brightness.  Collect a
    # list of good (not rejected) candidates and stop when there are
    # max_candidates.
    goods = []
    for ii, acq in enumerate(acqs):
        bad = ((np.abs(acq['yang'] - stars['yang']) < 30) &
               (np.abs(acq['zang'] - stars['zang']) < 30) &
               (stars['MAG_ACA'] - acq['MAG_ACA'] < 3))
        if np.count_nonzero(bad) == 1:  # Self always matches
            goods.append(ii)
        if len(goods) == max_candidates:
            break

    acqs = acqs[goods]
    INFO.log('selected {} candidates with no spoiler (star within 3 mag and 30 arcsec)'
             .format(len(acqs)))

    acqs.rename_column('AGASC_ID', 'id')
    acqs.rename_column('COLOR1', 'color')
    # Drop all the other AGASC columns.  No longer useful.
    names = [name for name in acqs.colnames if not name.isupper()]
    acqs = AcqTable(acqs[names])
    # acqs['AGASC_ID'] = acqs['id']

    # Make this suitable for plotting
    acqs['idx'] = np.arange(len(acqs))
    acqs['type'] = 'ACQ'
    acqs['halfw'] = 120  # "Official" acq box_size for catalog

    # Set up columns needed for catalog initial selection and optimization
    def empty_dicts():
        return [{} for _ in range(len(acqs))]

    acqs['p_acq'] = -999.0  # Acq prob for star for box_size=halfw, marginalized over man_err
    acqs['p_acqs'] = empty_dicts()  # Dict keyed on (box_size, man_err) (product of next three)
    acqs['p_brightest'] = empty_dicts()  # Dict keyed on (box_size, man_err)
    acqs['p_acq_model'] = empty_dicts()  # Dict keyed on box_size
    acqs['p_in_box'] = empty_dicts()  # Dict keyed on box_size

    acqs['spoilers'] = None  # Filled in with Table of spoilers
    acqs['imposters'] = None  # Filled in with Table of imposters
    acqs['spoilers_box'] = -999.0  # Cached value of box_size + man_err for spoilers
    acqs['imposters_box'] = -999.0  # Cached value of box_size + dither for imposters

    acqs['p_acq'].format = '.3f'
    acqs['color'].format = '.2f'

    return acqs, bads


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
    # 1-sigma of difference of stars['MAG_ACA'] - acq['MAG_ACA']
    # TO DO: lower limit clip?
    mag_diff_err = np.sqrt(stars['mag_err'] ** 2 + acq['mag_err'] ** 2)

    # Stars in extended box and within 3-sigma (99.7%)
    ok = ((np.abs(stars['yang'] - acq['yang']) < box_size) &
          (np.abs(stars['zang'] - acq['zang']) < box_size) &
          (stars['mag'] - acq['mag'] < 3 * mag_diff_err) &
          (stars['AGASC_ID'] != acq['id'])
          )
    spoilers = stars[ok]
    spoilers.sort('mag')

    return spoilers


def bin2x2(arr):
    """Bin 2-d ``arr`` in 2x2 blocks.  Requires that ``arr`` has even shape sizes"""
    shape = (arr.shape[0] // 2, 2, arr.shape[1] // 2, 2)
    return arr.reshape(shape).sum(-1).sum(1)


RC_6x6 = np.array([10, 11, 12, 13,
                   17, 18, 19, 20, 21, 22,
                   25, 26, 27, 28, 29, 30,
                   33, 34, 35, 36, 37, 38,
                   41, 42, 43, 44, 45, 46,
                   50, 51, 52, 53],
                  dtype=np.int64)

ROW_6x6 = np.array([1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5,
                    6, 6, 6, 6],
                   dtype=np.float64) + 0.5

COL_6x6 = np.array([2, 3, 4, 5,
                    1, 2, 3, 4, 5, 6,
                    1, 2, 3, 4, 5, 6,
                    1, 2, 3, 4, 5, 6,
                    1, 2, 3, 4, 5, 6,
                    2, 3, 4, 5],
                   dtype=np.float64) + 0.5


def get_image_props(ccd_img, c_row, c_col, bgd=None):
    """
    Do a pseudo-read and compute the background-subtracted magnitude
    of the image.  Returns the 8x8 image and mag.

    :param ccd_image: full-frame CCD image (e.g. dark current, with or without stars).
    :param c_row: int row at center (readout row-4 to row+4)
    :param c_col: int col at center
    :param bgd: background to subtract at each pixel.  If None then compute flight bgd.

    :returns: 8x8 image (ndarray), image mag
    """
    img = ccd_img[c_row - 4:c_row + 4, c_col - 4:c_col + 4]

    if bgd is None:
        raise NotImplementedError('no calc_flight_background here')
        # bgd = calc_flight_background(img)  # currently not defined

    img = img - bgd
    img_6x6 = img.flatten()[RC_6x6]
    img_sum = np.sum(img_6x6)
    r = np.sum(img_6x6 * ROW_6x6) / img_sum
    c = np.sum(img_6x6 * COL_6x6) / img_sum
    row = r + c_row - 4
    col = c + c_col - 4
    mag = count_rate_to_mag(img_sum)

    return img, img_sum, mag, row, col


def pea_reject_image(img):
    """
    Check if PEA would reject image (too narrow, too peaky, etc)
    """
    # To be implemented
    return False


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
    if ((box_r1 - box_r0) % 2 == 1):
        if box_r1 == 1024:
            box_r0 -= 1
        else:
            box_r1 += 1
    if ((box_c1 - box_c0) % 2 == 1):
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
               'mag_err': get_mag_std(mag),
               }
        outs.append(out)

    if len(outs) > 0:
        outs = Table(outs)
        outs.sort('mag')

    return outs


def calc_p_brightest_compare(acq, intruders):
    """
    For given ``acq`` star and intruders mag, mag_err,
    do the probability calculation to see if the acq star is brighter
    than all of them.
    """
    if len(intruders) == 0:
        return 1.0

    n_pts = 100
    x0, x1 = stats.norm.ppf([0.001, 0.999], loc=acq['mag'], scale=acq['mag_err'])
    x = np.linspace(x0, x1, n_pts)
    dx = (x1 - x0) / (n_pts - 1)

    acq_pdf = stats.norm.pdf(x, loc=acq['mag'], scale=acq['mag_err'])

    sp_cdfs = []
    for sp in intruders:
        # Compute prob intruder is fainter than acq (so sp_mag > x).
        # CDF is prob that sp_mag < x, so take 1-CDF.
        sp_cdf = stats.norm.cdf(x, loc=sp['mag'], scale=sp['mag_err'])
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
    if acq[name] is None:
        acq[name] = get_func(**kwargs)
        acq[name_box] = box_size

        if len(acq[name]) > 0:
            # Clip to within n_sigma.  d_mag < 0 for intruder brighter than acq
            d_mag = acq[name]['mag'] - acq['mag']
            d_mag_err = np.sqrt(acq[name]['mag_err'] ** 2 + acq['mag_err'] ** 2)
            ok = d_mag < n_sigma * d_mag_err
            acq[name] = acq[name][ok]

    else:
        # Ensure cached spoilers cover the current case
        if box_size > acq[name_box]:
            raise ValueError('box_size is greater than {}'.format(name_box))

    intruders = acq[name]
    colnames = ['yang', 'zang', 'mag', 'mag_err']
    if len(intruders) == 0:
        intruders = Table(names=colnames)  # zero-length table with float cols
    else:
        ok = ((np.abs(intruders['yang'] - acq['yang']) < box_size) &
              (np.abs(intruders['zang'] - acq['zang']) < box_size))
        intruders = intruders[colnames][ok]

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

    intruders = vstack([spoilers, imposters])
    prob = calc_p_brightest_compare(acq, intruders)

    return prob


def broadcast_arrays(*args):
    is_scalar = all(np.array(arg).ndim == 0 for arg in args)
    args = np.atleast_1d(*args)
    if not isinstance(args, list):
        args = [args]
    outs = [is_scalar] + np.broadcast_arrays(*args)
    return outs


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

    This uses a simplistic calculation which assumes that ``p_in_box`` is
    just the fraction of box area that is within the effective usable portion
    of the CCD.
    """
    p_in_box = 1.0
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
            # Reduce p_in_box by fraction of pixels inside usable area.
            pix_inside = full_width - pix_off_ccd
            if pix_inside > 0:
                p_in_box *= pix_inside / full_width
            else:
                p_in_box = 0.0

    return p_in_box


def select_best_p_acqs(p_acqs, min_p_acq, acq_indices, box_sizes):
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
    INFO.log('find stars with best acq prob for min_p_acq={}'.format(min_p_acq))
    INFO.log('current: acq_indices={} box_sizes={}'.format(acq_indices, box_sizes),
             level='verbose')

    for box_size in CHAR.box_sizes:
        # Get array of p_acq values corresponding to box_size and
        # man_err=box_size, for each of the candidate acq stars.
        p_acqs_for_box = np.array([p_acq[box_size, box_size] for p_acq in p_acqs])

        indices = np.argsort(-p_acqs_for_box)
        for acq_idx in indices:
            if p_acqs_for_box[acq_idx] > min_p_acq and acq_idx not in acq_indices:
                INFO.log('  adding idx={} with box_size={} and p_acq={}'
                         .format(acq_idx, box_size, round(p_acqs_for_box[acq_idx], 3)),
                         level='verbose')
                acq_indices.append(acq_idx)
                box_sizes.append(box_size)

            if len(acq_indices) == 8:
                INFO.log('  found 8 acq stars, exiting', level='verbose')
                return

    return


def calc_acq_p_vals(acq, box_size, man_err, dither, stars, dark, t_ccd, date):
    if (box_size, man_err) not in acq['p_acqs']:
        # Prob of being brightest in box (function of box_size and man_err,
        # independently because imposter prob is just a function of box_size not man_err).
        # Technically also a function of dither, but that does not vary here.
        p_brightest = calc_p_brightest(acq, box_size=box_size, stars=stars, dark=dark,
                                       man_err=man_err, dither=dither)
        acq['p_brightest'][box_size, man_err] = p_brightest

        # Acquisition probability model value (function of box_size only)
        p_acq_model = acq_success_prob(date=date, t_ccd=t_ccd,
                                       mag=acq['mag'], color=acq['color'],
                                       spoiler=False, halfwidth=box_size)
        acq['p_acq_model'][box_size] = p_acq_model

        # Probability star is in acq box (function of box_size only)
        p_in_box = calc_p_on_ccd(acq['row'], acq['col'], box_size=man_err + dither)
        acq['p_in_box'][box_size] = p_in_box

        # All together now!
        acq['p_acqs'][box_size, man_err] = p_brightest * p_acq_model * p_in_box


def get_initial_catalog(cand_acqs, stars, dark, dither=20, t_ccd=-11.0, date=None):
    """
    Get the initial catalog of up to 8 candidate acquisition stars.
    """
    INFO.log('calculating initial acq_p_vals for {} candidates'.format(len(cand_acqs)))
    for acq in cand_acqs:
        for box_size in CHAR.box_sizes:
            # For initial catalog set man_err to box_size.  TO DO: set man_err to 160 here??
            man_err = box_size
            calc_acq_p_vals(acq, box_size, man_err, dither, stars, dark, t_ccd, date)

    acq_indices = []
    box_sizes = []
    for min_p_acq in (0.75, 0.5, 0.25, 0.05):
        # Updates acq_indices, box_sizes in place
        select_best_p_acqs(cand_acqs['p_acqs'], min_p_acq, acq_indices, box_sizes)

        if len(acq_indices) == 8:
            break

    acqs = cand_acqs[acq_indices]
    acqs['halfw'] = box_sizes
    for acq, box_size in zip(acqs, box_sizes):
        acq['p_acq'] = acq['p_acqs'][box_size, box_size]

    return acqs


def calc_p_safe(acqs, verbose=False):
    p_no_safe = 1.0

    for man_err in CHAR.p_man_errs['man_err_hi']:
        p_man_err = get_p_man_err(man_err, acqs.meta['man_angle'])
        p_acqs = []
        for acq in acqs:
            box_size = acq['halfw']

            if man_err > box_size:
                p_acq = 0.0
            else:
                calc_acq_p_vals(acq, box_size, man_err,
                                dither=acqs.meta['dither'],
                                stars=acqs.meta['stars'],
                                dark=acqs.meta['dark'],
                                t_ccd=acqs.meta['t_ccd'],
                                date=acqs.meta['date'])
                p_acq = acq['p_acqs'][box_size, man_err]

            p_acqs.append(p_acq)

        p_n, p_n_cum = prob_n_acq(p_acqs)
        if verbose:
            INFO.log('man_err = {}'.format(man_err), level='verbose')
            INFO.log('  p_acqs =' + ' '.join(['{:.3f}'.format(val) for val in p_acqs]),
                     level='verbose')
            INFO.log('  p N_or_fewer=' + ' '.join(['{:.3f}%'.format(val * 100)
                                                   for val in p_n_cum[:4]]),
                     level='verbose')
        p_01 = p_n_cum[1]  # 1 or fewer => p_fail at this man_err

        p_no_safe *= (1 - p_man_err * p_01)

    p_safe = 1 - p_no_safe
    acqs.meta['p_safe'] = p_safe

    return p_safe


def optimize_acq_halfw(acqs, idx, p_safe, verbose=False):
    """
    Optimize the box size (halfw) for the acq star ``idx`` in the ``acqs`` list.
    Assume current ``p_safe``.
    """
    acq = acqs[idx]
    orig_halfw = acq['halfw']

    # Compute p_safe for each possible halfw for the current star
    p_safes = []
    for box_size in CHAR.box_sizes:
        acq['halfw'] = box_size
        p_safes.append(calc_p_safe(acqs))

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
    if verbose:
        INFO.log('for idx={} id={}'.format(idx, acq['id']))
        p_safes_strs = ['{:.2f}'.format(np.log10(p)) for p in p_safes]
        INFO.log('  p_safes={}'.format(' '.join(p_safes_strs)))
        INFO.log('  min_p_safe={:.2f} p_safe={:.2f} min_halfw={} orig_halfw={} improved={}'
                 .format(np.log10(min_p_safe), np.log10(p_safe),
                         min_halfw, orig_halfw, improved),
                 level='verbose')

    if improved:
        if verbose:
            INFO.log('  update acq idx={} halfw from {} to {}'
                     .format(idx, orig_halfw, min_halfw))
        p_safe = min_p_safe
        acq['halfw'] = min_halfw
    else:
        acq['halfw'] = orig_halfw

    return p_safe, improved


def optimize_acqs_halfw(acqs, verbose=False):
    p_safe = calc_p_safe(acqs)
    idxs = acqs.argsort('p_acq')

    # Any updates made?
    any_improved = False

    for idx in idxs:
        p_safe, improved = optimize_acq_halfw(acqs, idx, p_safe, verbose)
        any_improved |= improved

    return p_safe, any_improved


def optimize_catalog(acqs, verbose=False):
    p_safe = calc_p_safe(acqs, verbose=True)
    INFO.log('initial p_safe={:.5f}'.format(p_safe))

    # Start by optimizing the half-widths of the initial catalog
    for _ in range(5):
        p_safe, improved = optimize_acqs_halfw(acqs, verbose)
        if not improved:
            break

    INFO.log('After optimizing initial catalog p_safe = {:.5f}'.format(p_safe))
    # calc_p_safe(acqs, verbose=True)  # TO DO: need to figure out what to do here

    # Now try to swap in a new star from the candidate list and see if
    # it can improve p_safe.
    acq_ids = set(acqs['id'])
    for cand_acq in acqs.meta['cand_acqs']:
        cand_id = cand_acq['id']
        if cand_id in acq_ids:
            continue
        else:
            acq_ids.add(cand_id)

        # Get the index of the worst p_acq in the catalog
        p_acqs = [acq['p_acqs'][acq['halfw'], acq['halfw']] for acq in acqs]
        idx = np.argsort(p_acqs)[0]

        INFO.log('trying to use {} mag={:.2f} to replace idx={} with p_acq={:.3f}'
                 .format(cand_id, cand_acq['mag'], idx, p_acqs[idx]), level='verbose')

        # Make a copy of the row (acq star) as a numpy void (structured array row)
        orig_acq = acqs[idx].as_void()

        # Stub in the new candidate and get the best halfw (and corresponding new p_safe)
        acqs[idx] = cand_acq
        new_p_safe, improved = optimize_acq_halfw(acqs, idx, p_safe, verbose)

        # If the new star is noticably better (regardless of box size), OR
        # comparable but with a bigger box, then accept it and do one round of
        # full catalog optimization
        improved = ((new_p_safe / p_safe < 0.9) or
                    (new_p_safe < p_safe and acqs['halfw'][idx] > orig_acq['halfw']))
        if improved:
            p_safe, improved = optimize_acqs_halfw(acqs, verbose)
            calc_p_safe(acqs, verbose=True)
            INFO.log('  accepted, new p_safe = {:.5f}'.format(p_safe))
        else:
            acqs[idx] = orig_acq


@lru_cache(300)
def get_acq_catalog(obsid=None, att=None,
                    man_angle=None, t_ccd=None, date=None, dither=None,
                    optimize=True, verbose=False):
    INFO.clear()

    if obsid is not None:
        from mica.starcheck import get_starcheck_catalog
        INFO.log('getting starcheck catalog for obsid {}'.format(obsid))

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

        # TO DO: deal with non-square dither pattern, esp. 8 x 64.
        dither_y_amp = obso.get('dither_y_amp')
        dither_z_amp = obso.get('dither_z_amp')
        if dither_y_amp is not None and dither_z_amp is not None:
            dither = max(dither_y_amp, dither_z_amp)
        if dither is None:
            dither = 20

    INFO.log('getting dark cal image at date={} t_ccd={:.1f}'
             .format(date, t_ccd))
    dark = get_dark_cal_image(date=date, select='nearest', t_ccd_ref=t_ccd)

    stars = get_stars(att)
    cand_acqs, bads = get_acq_candidates(stars)
    acqs = get_initial_catalog(cand_acqs, stars=stars, dark=dark, dither=dither,
                               t_ccd=t_ccd, date=date)
    acqs.meta = {'att': att,
                 'date': date,
                 't_ccd': t_ccd,
                 'man_angle': man_angle,
                 'dither': dither,
                 'cand_acqs': cand_acqs,
                 'stars': stars,
                 'dark': dark,
                 'bads': bads,
                 }

    if optimize:
        optimize_catalog(acqs, verbose)

    acqs['slot'] = np.arange(len(acqs))

    return acqs
