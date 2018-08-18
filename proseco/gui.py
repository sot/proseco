import os
import warnings
from itertools import count

import agasc
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from Quaternion import Quat
from Ska.quatutil import radec2yagzag
import chandra_aca
from .gui_char import CHAR, CCD

# Ignore known numexpr.necompiler and table.conditions warning
warnings.filterwarnings(
    'ignore',
    message="using `oa_ndim == 0` when `op_axes` is NULL is deprecated.*",
    category=DeprecationWarning)


STAR_CHAR = CHAR["Stars"]
# Update the characteristics to change the bright mag hard limit for guide stars to 5.9 from 5.8
#for stage in CHAR['Guide']:
#    stage['Inertial']['MagLimit'][0] = 5.9


PIX_2_ARC = STAR_CHAR["General"]["Pix2Arc"]
ARC_2_PIX = 1.0 / PIX_2_ARC
RAD_2_PIX = 180/np.pi*3600*ARC_2_PIX


MANVR_ERROR = 60
FIELD_ERROR_PAD = 0



def set_manvr_error(manvr_error):
    global MANVR_ERROR
    MANVR_ERROR = manvr_error


def check_mag(cone_stars, opt, label):
    magOneSigError = cone_stars['mag_one_sig_err']
    mag = cone_stars['MAG_ACA']
    magNSig = opt['Spoiler']['SigErrMultiplier'] * magOneSigError
    too_bright = ((mag - magNSig - opt['Inertial']['MagErrSyst'])
                  < np.min(opt['Inertial']['MagLimit']))
    too_dim = ((mag + magNSig + opt['Inertial']['MagErrSyst'])
               > np.max(opt['Inertial']['MagLimit']))
    nomag = mag == -9999
    #cone_stars['too_bright_{}'.format(label)] = too_bright
    #cone_stars['too_dim_{}'.format(label)] = too_dim
    #cone_stars['nomag_{}'.format(label)] = nomag
    return ~too_bright & ~too_dim & ~nomag


def check_mag_spoilers(cone_stars, ok, opt):
    stype = opt['Type']
    stderr2 = cone_stars['mag_one_sig_err2']
    fidpad = FIELD_ERROR_PAD * ARC_2_PIX
    maxsep = STAR_CHAR['General']['Spoiler']['MaxSep'] + fidpad
    intercept = STAR_CHAR['General']['Spoiler']['Intercept'] + fidpad
    spoilslope = STAR_CHAR['General']['Spoiler']['Slope']
    nSigma = opt['Spoiler']['SigErrMultiplier']
    magdifflim = STAR_CHAR['General']['Spoiler']['MagDiffLimit']
    if magdifflim == '-_Inf_':
        magdifflim = -1 * np.inf
    mag_col = 'mag_spoiled_{}_{}'.format(nSigma, stype)
    mag_spoil_check = 'mag_spoil_check_{}_{}'.format(nSigma, stype)
    if mag_col in cone_stars.columns:
        # if ok for stage and not previously mag spoiler checked for this
        # nsigma
        ok = ok & ~cone_stars[mag_spoil_check]
    if not np.any(ok):
        return np.zeros_like(ok), ok
    coords = SkyCoord(cone_stars['RA_PMCORR'], cone_stars['DEC_PMCORR'],
                      unit='deg')
    maxsep_arcs = maxsep * PIX_2_ARC
    cand_idx_in_ok, cat_idx, spoil_dist, dist3d = search_around_sky(
        coords[ok],
        coords,
        seplimit=u.arcsec * maxsep_arcs)
    # index the candidates back into the full list so we have one
    # index to work with, shared betweek cand_idx and cat_idx
    cand_idx = np.flatnonzero(ok)[cand_idx_in_ok]
    # and try to find the spoilers in a vectorized way
    cand_mags = cone_stars['MAG_ACA'][cand_idx]
    spoiler_mags = cone_stars['MAG_ACA'][cat_idx]
    itself = cand_idx == cat_idx
    too_dim = (cand_mags - spoiler_mags) < magdifflim
    delmag = (cand_mags - spoiler_mags
              + nSigma * np.sqrt(stderr2[cand_idx] + stderr2[cat_idx]))
    thsep = intercept + delmag * spoilslope
    spoils = (spoil_dist < u.arcsec * thsep * PIX_2_ARC) & ~itself & ~too_dim
    # this is now indexed over the cat/cand idxs so, re-index again
    spoiled = np.zeros_like(ok)
    spoiled[np.unique(cand_idx[spoils])] = True
    # and include any previous spoilers
    spoiled = spoiled | cone_stars[mag_col]
    return spoiled, ok


def check_bad_pixels(cone_stars, not_bad, dither, opt):
    bp = opt['Body']['Pixels']['BadPixels']
    # start with big distances
    distance = np.ones(len(cone_stars[not_bad])) * 9999
    if bp is None:
        full_distance = np.ones(len(cone_stars)) * 9999
        full_distance[not_bad] = distance
        return full_distance
    row, col = cone_stars['row'], cone_stars['col']
    bp = np.array(bp)
    # Loop over the stars to check each distance to closest bad pixel
    rpad = .5 * dither[0] * ARC_2_PIX
    cpad = .5 * dither[1] * ARC_2_PIX
    for i, rs, cs in zip(count(), row[not_bad], col[not_bad]):
        in_reg_r = (rs >= (bp[:,0] - rpad)) & (rs <= (bp[:,1] + rpad))
        in_reg_c = (cs >= (bp[:,2] - cpad)) & (cs <= (bp[:,3] + cpad))
        r_dist = np.min(np.abs(
                [rs - (bp[:,0] - rpad), rs - (bp[:,1] + rpad)]), axis=0)
        r_dist[in_reg_r] = 0
        c_dist = np.min(np.abs(
                [cs - (bp[:,2] - cpad), cs - (bp[:,3] + cpad)]), axis=0)
        c_dist[in_reg_c] = 0
        # For the nearest manhattan distance we want the max in each axis
        maxes = np.max(np.vstack([r_dist, c_dist]), axis=0)
        # And then the minimum of the maxes
        idxmatch = np.argmin(maxes)
        distance[i] = maxes[idxmatch]
    full_distance = np.ones(len(cone_stars)) * 9999
    full_distance[not_bad] = distance
    # TODO fix this hard limit
    return full_distance < 6


def check_column_spoilers(cone_stars, ok, opt):
    Column = STAR_CHAR['General']['Body']['Column']
    nSigma = opt['Spoiler']['SigErrMultiplier']
    magerr2 = cone_stars['mag_one_sig_err2']
    column_pad = FIELD_ERROR_PAD * ARC_2_PIX
    column_spoiled = np.zeros_like(ok)
    # For the remaining candidates
    for cand in cone_stars[ok]:
        dm = (cand['MAG_ACA'] - cone_stars['MAG_ACA'][~cone_stars['offchip']] +
              nSigma * np.sqrt(cand['mag_one_sig_err2']
                               + cone_stars['mag_one_sig_err2'][~cone_stars['offchip']]))
        # if there are no stars ~ MagDiff (4.5 mags) brighter than the candidate we're done
        if not np.any(dm > Column['MagDiff']):
            continue
        dcol = cand['col'] - cone_stars['col'][~cone_stars['offchip']]
        direction = cone_stars['row'][~cone_stars['offchip']] / cand['row']
        spoilers = ((dm > Column['MagDiff'])
                    & (np.abs(dcol) <= (Column['Separation'] + column_pad))
                    & (direction > 1.0))
        if np.any(spoilers):
            column_spoiled[cone_stars['AGASC_ID'] == cand['AGASC_ID']] = True
    return column_spoiled


def check_stage(cone_stars, not_bad, dither, opt, label):
    stype = opt['Type']
    mag_ok = check_mag(cone_stars, opt, label)
    ok = mag_ok & not_bad
    if not np.any(ok):
        return ok
    bp = check_bad_pixels(cone_stars, ok, dither, opt)
    ok = ok & ~bp

    nSigma = opt['Spoiler']['SigErrMultiplier']
    mag_spoiled = ~ok.copy()
    mag_check_col = 'mag_spoil_check_{}_{}'.format(nSigma, stype)
    if mag_check_col not in cone_stars.columns:
        cone_stars[mag_check_col] = np.zeros_like(not_bad)
        cone_stars['mag_spoiled_{}_{}'.format(nSigma, stype)] = np.zeros_like(not_bad)
    mag_spoiled, checked = check_mag_spoilers(cone_stars, ok, opt)
    cone_stars[mag_check_col] = cone_stars[mag_check_col] | checked
    cone_stars['mag_spoiled_{}_{}'.format(nSigma, stype)] = (
        cone_stars['mag_spoiled_{}_{}'.format(nSigma, stype)] | mag_spoiled)

    ok = ok & ~mag_spoiled


    #cone_stars['bad_box_{}'.format(stype)] = bad_box
    if opt['SearchSettings']['DoColumnRegisterCheck']:
        badcr = check_column_spoilers(cone_stars, ok, opt)
        ok = ok & ~badcr
#    if opt['SearchSettings']['DoBminusVCheck']:
#        badbv = check_bv(cone_stars, ok & ~mag_spoiled & ~bad_dist)
#        ok = ok & ~badbv

    return ok


def get_mag_errs(cone_stars, opt):
    caterr = cone_stars['MAG_ACA_ERR'] / 100.
    caterr = np.min([np.ones(len(caterr))*opt['Inertial']['MaxMagError'], caterr], axis=0)
    randerr = opt['Inertial']['MagErrRand']
    magOneSigError = np.sqrt(randerr*randerr + caterr*caterr)
    return magOneSigError, magOneSigError**2


def select_stage_stars(ra, dec, roll, dither, cone_stars):

    stype = 'Guide'
    opt = STAR_CHAR[stype][0]
    opt['Type'] = stype
    opt['Stage'] = 0

    if 'mag_one_sig_err' not in cone_stars.columns:
        cone_stars['mag_one_sig_err'], cone_stars['mag_one_sig_err2'] = get_mag_errs(cone_stars, opt)

    q = Quat((ra, dec, roll))
    yag_deg, zag_deg = radec2yagzag(cone_stars['RA_PMCORR'], cone_stars['DEC_PMCORR'], q)
    row, col = chandra_aca.yagzag_to_pixels(yag_deg * 3600,
                                            zag_deg * 3600, allow_bad=True)
    cone_stars['row'] = row
    cone_stars['col'] = col
    # Filter in place the stars outside the FOV
    wayoffchip = ((cone_stars['row'] > 512 + 40) | (cone_stars['row'] < -512 - 40)
                  | (cone_stars['col'] > 512 + 40) | (cone_stars['col'] < -512 - 40))
    cone_stars = cone_stars[~wayoffchip]

    # Mark the ones that are offchip by smaller amounts (can still be spoiler with dither)
    offchip = ((cone_stars['row'] > CCD['row_max'])
               | (cone_stars['row'] < CCD['row_min'])
               | (cone_stars['col'] > CCD['col_max'])
               | (cone_stars['col'] < CCD['col_min']))
    cone_stars['offchip'] = offchip

    r_dith_pad = dither[0] * ARC_2_PIX
    c_dith_pad = dither[1] * ARC_2_PIX
    row_min = CCD['row_min'] + (CCD['row_pad'] + CCD['window_pad'] + r_dith_pad)
    row_max = CCD['row_max'] - (CCD['row_pad'] + CCD['window_pad'] + r_dith_pad)
    col_min = CCD['col_min'] + (CCD['col_pad'] + CCD['window_pad'] + c_dith_pad)
    col_max = CCD['col_max'] - (CCD['col_pad'] + CCD['window_pad'] + c_dith_pad)
    outofbounds = ((cone_stars['row'] > row_max)
                   | (cone_stars['row'] < row_min)
                   | (cone_stars['col'] > col_max)
                   | (cone_stars['col'] < col_min))
    cone_stars['outofbounds'] = outofbounds

    bad_mag_error = cone_stars['MAG_ACA_ERR'] > STAR_CHAR["General"]['MagErrorTol']
    #cone_stars['bad_mag_error_{}'.format(stype)] = bad_mag_error

    bad_pos_error = cone_stars['POS_ERR'] > STAR_CHAR['General']['PosErrorTol']
    #cone_stars['bad_pos_error_{}'.format(stype)] = bad_pos_error

    bad_aspq1 = ((cone_stars['ASPQ1'] > np.max(STAR_CHAR['General']['ASPQ1Lim']))
                  | (cone_stars['ASPQ1'] < np.min(STAR_CHAR['General']['ASPQ1Lim'])))
    #cone_stars['bad_aspq1_{}'.format(stype)] = bad_aspq1

    bad_aspq2 = ((cone_stars['ASPQ2'] > np.max(STAR_CHAR['General']['ASPQ2Lim']))
                  | (cone_stars['ASPQ2'] < np.min(STAR_CHAR['General']['ASPQ2Lim'])))
    #cone_stars['bad_aspq2_{}'.format(stype)] = bad_aspq2

    bad_aspq3 = ((cone_stars['ASPQ3'] > np.max(STAR_CHAR['General']['ASPQ3Lim']))
                  | (cone_stars['ASPQ3'] < np.min(STAR_CHAR['General']['ASPQ3Lim'])))
    #cone_stars['bad_aspq3_{}'.format(stype)] = bad_aspq3


    nonstellar = cone_stars['CLASS'] != 0
    #cone_stars['nonstellar'] = nonstellar

    not_bad = (~outofbounds & ~bad_mag_error & ~bad_pos_error
                & ~nonstellar & ~bad_aspq1 & ~bad_aspq2 & ~bad_aspq3)

    # Set some column defaults that will be updated in check_stage
    cone_stars['{}_stage'.format(stype)] = -1
    ncand = STAR_CHAR['General']['Select']['NMaxSelect'] + STAR_CHAR['General']['Select']['nSurplus']
    for idx, stage_char in enumerate(STAR_CHAR[stype], 1):
        # Save the type in the characteristics
        stage_char['Type'] = stype
        stage_char['Stage'] = idx
        if np.count_nonzero(cone_stars['{}_stage'.format(stype)] != -1) < ncand:
            stage  = check_stage(cone_stars,
                                 not_bad & ~(cone_stars['{}_stage'.format(stype)] != -1),
                                 dither=dither,
                                 opt=stage_char, label="{}_{}".format(stype, str(idx)))
            cone_stars['{}_stage'.format(stype)][stage] = idx
    selected = cone_stars[cone_stars['{}_stage'.format(stype)] != -1]
    return selected



def select_guide_stars(ra, dec, roll, dither=(8, 8), n=5, cone_stars=None, date=None):
    if cone_stars is None:
        cone_stars = agasc.get_agasc_cone(ra, dec, radius=1.4, date=date,
                                          agasc_file='/proj/sot/ska/data/agasc/agasc1p6.h5')
    selected = select_stage_stars(ra, dec, roll, dither=dither, cone_stars=cone_stars)
    # Ignore guide star code to use ACA matrix etc to optimize selection of stars in the last
    # stage and just take these by stage and then magnitude
    selected.sort(['Guide_stage', 'MAG_ACA'])
    #cone_stars['Guide_selected'] = False
    #for agasc_id in selected[0:n]['AGASC_ID']:
    #    cone_stars['Guide_selected'][cone_stars['AGASC_ID'] == agasc_id] = True
    return selected[0:n]

