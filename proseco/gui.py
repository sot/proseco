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
from mica.archive.aca_dark import get_dark_cal_image


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

FIELD_ERROR_PAD = 0



def check_mag(stars, opt, label):
    magOneSigError = stars['mag_one_sig_err']
    mag = stars['MAG_ACA']
    magNSig = opt['Spoiler']['SigErrMultiplier'] * magOneSigError
    too_bright = ((mag - magNSig - opt['Inertial']['MagErrSyst'])
                  < np.min(opt['Inertial']['MagLimit']))
    too_dim = ((mag + magNSig + opt['Inertial']['MagErrSyst'])
               > np.max(opt['Inertial']['MagLimit']))
    nomag = mag == -9999
    #stars['too_bright_{}'.format(label)] = too_bright
    #stars['too_dim_{}'.format(label)] = too_dim
    #stars['nomag_{}'.format(label)] = nomag
    return ~too_bright & ~too_dim & ~nomag


def check_mag_spoilers(stars, ok, opt):
    stype = opt['Type']
    stderr2 = stars['mag_one_sig_err2']
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
    if mag_col in stars.columns:
        # if ok for stage and not previously mag spoiler checked for this
        # nsigma
        ok = ok & ~stars[mag_spoil_check]
    if not np.any(ok):
        return np.zeros_like(ok), ok
    coords = SkyCoord(stars['RA_PMCORR'], stars['DEC_PMCORR'],
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
    cand_mags = stars['MAG_ACA'][cand_idx]
    spoiler_mags = stars['MAG_ACA'][cat_idx]
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
    spoiled = spoiled | stars[mag_col]
    return spoiled, ok


def check_bad_pixels(stars, not_bad, dither, opt):
    bp = opt['Body']['Pixels']['BadPixels']
    # start with big distances
    distance = np.ones(len(stars[not_bad])) * 9999
    if bp is None:
        full_distance = np.ones(len(stars)) * 9999
        full_distance[not_bad] = distance
        return full_distance
    row, col = stars['row'], stars['col']
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
    full_distance = np.ones(len(stars)) * 9999
    full_distance[not_bad] = distance
    # TODO fix this hard limit
    return full_distance < 6


def check_imposters(stars, ok, dark, dither, opt):
    """
    Check for any pixel with a value greater than the imposter thresh
    for each candidate and stage.
    """
    imp = np.zeros_like(ok)
    for cand in stars[ok]:
        row_extent = 4 + dither[0] * ARC_2_PIX
        col_extent = 4 + dither[1] * ARC_2_PIX
        rminus = int(np.floor(cand['row'] - row_extent))
        rplus = int(np.ceil(cand['row'] + row_extent + 1))
        cminus = int(np.floor(cand['col'] - col_extent))
        cplus = int(np.ceil(cand['col'] + col_extent + 1))
        pix = dark.aca[rminus:rplus, cminus:cplus]
        cand_counts = chandra_aca.mag_to_count_rate(cand['MAG_ACA'])
        #print("{} {} {}".format(cand['AGASC_ID'], np.max(pix) * 1.0 / cand_counts,
        #      opt['Imposter']['Thresh']))
        if np.max(pix) > (cand_counts * opt['Imposter']['Thresh']):
            imp[stars['AGASC_ID'] == cand['AGASC_ID']] = True
    return imp


def check_column_spoilers(stars, ok, opt):
    Column = STAR_CHAR['General']['Body']['Column']
    nSigma = opt['Spoiler']['SigErrMultiplier']
    magerr2 = stars['mag_one_sig_err2']
    column_pad = FIELD_ERROR_PAD * ARC_2_PIX
    column_spoiled = np.zeros_like(ok)
    # For the remaining candidates
    for cand in stars[ok]:
        dm = (cand['MAG_ACA'] - stars['MAG_ACA'][~stars['offchip']] +
              nSigma * np.sqrt(cand['mag_one_sig_err2']
                               + stars['mag_one_sig_err2'][~stars['offchip']]))
        # if there are no stars ~ MagDiff (4.5 mags) brighter than the candidate we're done
        if not np.any(dm > Column['MagDiff']):
            continue
        dcol = cand['col'] - stars['col'][~stars['offchip']]
        direction = stars['row'][~stars['offchip']] / cand['row']
        spoilers = ((dm > Column['MagDiff'])
                    & (np.abs(dcol) <= (Column['Separation'] + column_pad))
                    & (direction > 1.0))
        if np.any(spoilers):
            column_spoiled[stars['AGASC_ID'] == cand['AGASC_ID']] = True
    return column_spoiled


def check_stage(stars, not_bad, dither, dark, opt, label):
    stype = opt['Type']
    mag_ok = check_mag(stars, opt, label)
    ok = mag_ok & not_bad
    if not np.any(ok):
        return ok
    bp = check_bad_pixels(stars, ok, dither, opt)
    stars['bp_spoiled'] = bp
    ok = ok & ~bp
    imp = check_imposters(stars, ok, dark, dither, opt)
    ok = ok & ~imp
    stars['imp_spoiled'] = imp

    nSigma = opt['Spoiler']['SigErrMultiplier']
    mag_spoiled = ~ok.copy()
    mag_check_col = 'mag_spoil_check_{}_{}'.format(nSigma, stype)
    if mag_check_col not in stars.columns:
        stars[mag_check_col] = np.zeros_like(not_bad)
        stars['mag_spoiled_{}_{}'.format(nSigma, stype)] = np.zeros_like(not_bad)
    mag_spoiled, checked = check_mag_spoilers(stars, ok, opt)
    stars[mag_check_col] = stars[mag_check_col] | checked
    stars['mag_spoiled_{}_{}'.format(nSigma, stype)] = (
        stars['mag_spoiled_{}_{}'.format(nSigma, stype)] | mag_spoiled)

    ok = ok & ~mag_spoiled


    #stars['bad_box_{}'.format(stype)] = bad_box
    if opt['SearchSettings']['DoColumnRegisterCheck']:
        badcr = check_column_spoilers(stars, ok, opt)
        ok = ok & ~badcr
#    if opt['SearchSettings']['DoBminusVCheck']:
#        badbv = check_bv(stars, ok & ~mag_spoiled & ~bad_dist)
#        ok = ok & ~badbv

    return ok


def get_mag_errs(stars, opt):
    caterr = stars['MAG_ACA_ERR'] / 100.
    caterr = np.min([np.ones(len(caterr))*opt['Inertial']['MaxMagError'], caterr], axis=0)
    randerr = opt['Inertial']['MagErrRand']
    magOneSigError = np.sqrt(randerr*randerr + caterr*caterr)
    return magOneSigError, magOneSigError**2


def select_stage_stars(ra, dec, roll, dither, dark, stars):

    stype = 'Guide'
    opt = STAR_CHAR[stype][0]
    opt['Type'] = stype
    opt['Stage'] = 0

    if 'mag_one_sig_err' not in stars.columns:
        stars['mag_one_sig_err'], stars['mag_one_sig_err2'] = get_mag_errs(stars, opt)

    q = Quat((ra, dec, roll))
    yag_deg, zag_deg = radec2yagzag(stars['RA_PMCORR'], stars['DEC_PMCORR'], q)
    row, col = chandra_aca.yagzag_to_pixels(yag_deg * 3600,
                                            zag_deg * 3600, allow_bad=True)
    stars['row'] = row
    stars['col'] = col
    # Filter in place the stars outside the FOV
    wayoffchip = ((stars['row'] > 512 + 40) | (stars['row'] < -512 - 40)
                  | (stars['col'] > 512 + 40) | (stars['col'] < -512 - 40))
    stars = stars[~wayoffchip]

    # Mark the ones that are offchip by smaller amounts (can still be spoiler with dither)
    offchip = ((stars['row'] > CCD['row_max'])
               | (stars['row'] < CCD['row_min'])
               | (stars['col'] > CCD['col_max'])
               | (stars['col'] < CCD['col_min']))
    stars['offchip'] = offchip

    r_dith_pad = dither[0] * ARC_2_PIX
    c_dith_pad = dither[1] * ARC_2_PIX
    row_min = CCD['row_min'] + (CCD['row_pad'] + CCD['window_pad'] + r_dith_pad)
    row_max = CCD['row_max'] - (CCD['row_pad'] + CCD['window_pad'] + r_dith_pad)
    col_min = CCD['col_min'] + (CCD['col_pad'] + CCD['window_pad'] + c_dith_pad)
    col_max = CCD['col_max'] - (CCD['col_pad'] + CCD['window_pad'] + c_dith_pad)
    outofbounds = ((stars['row'] > row_max)
                   | (stars['row'] < row_min)
                   | (stars['col'] > col_max)
                   | (stars['col'] < col_min))
    stars['outofbounds'] = outofbounds

    bad_mag_error = stars['MAG_ACA_ERR'] > STAR_CHAR["General"]['MagErrorTol']
    #stars['bad_mag_error_{}'.format(stype)] = bad_mag_error

    bad_pos_error = stars['POS_ERR'] > STAR_CHAR['General']['PosErrorTol']
    #stars['bad_pos_error_{}'.format(stype)] = bad_pos_error

    bad_aspq1 = ((stars['ASPQ1'] > np.max(STAR_CHAR['General']['ASPQ1Lim']))
                  | (stars['ASPQ1'] < np.min(STAR_CHAR['General']['ASPQ1Lim'])))
    #stars['bad_aspq1_{}'.format(stype)] = bad_aspq1

    bad_aspq2 = ((stars['ASPQ2'] > np.max(STAR_CHAR['General']['ASPQ2Lim']))
                  | (stars['ASPQ2'] < np.min(STAR_CHAR['General']['ASPQ2Lim'])))
    #stars['bad_aspq2_{}'.format(stype)] = bad_aspq2

    bad_aspq3 = ((stars['ASPQ3'] > np.max(STAR_CHAR['General']['ASPQ3Lim']))
                  | (stars['ASPQ3'] < np.min(STAR_CHAR['General']['ASPQ3Lim'])))
    #stars['bad_aspq3_{}'.format(stype)] = bad_aspq3


    nonstellar = stars['CLASS'] != 0
    #stars['nonstellar'] = nonstellar

    not_bad = (~outofbounds & ~bad_mag_error & ~bad_pos_error
                & ~nonstellar & ~bad_aspq1 & ~bad_aspq2 & ~bad_aspq3)

    # Set some column defaults that will be updated in check_stage
    stars['{}_stage'.format(stype)] = -1
    ncand = STAR_CHAR['General']['Select']['NMaxSelect'] + STAR_CHAR['General']['Select']['nSurplus']
    for idx, stage_char in enumerate(STAR_CHAR[stype], 1):
        # Save the type in the characteristics
        stage_char['Type'] = stype
        stage_char['Stage'] = idx
        if np.count_nonzero(stars['{}_stage'.format(stype)] != -1) < ncand:
            stage  = check_stage(stars,
                                 not_bad & ~(stars['{}_stage'.format(stype)] != -1),
                                 dither=dither, dark=dark,
                                 opt=stage_char, label="{}_{}".format(stype, str(idx)))
            stars['{}_stage'.format(stype)][stage] = idx
    selected = stars[stars['{}_stage'.format(stype)] != -1]
    return selected



def select_guide_stars(ra, dec, roll, dither=(8, 8), n=5, date=None, t_ccd=None,
	                   stars=None, dark=None):
    if stars is None:
        stars = agasc.get_agasc_cone(ra, dec, radius=1.4, date=date,
                                          agasc_file='/proj/sot/ska/data/agasc/agasc1p6.h5')
    if dark is None:
        dark  = get_dark_cal_image(date=date, t_ccd_ref=t_ccd, aca_image=True)

    selected = select_stage_stars(ra, dec, roll, dither=dither, dark=dark, stars=stars)
    # Ignore guide star code to use ACA matrix etc to optimize selection of stars in the last
    # stage and just take these by stage and then magnitude
    selected.sort(['Guide_stage', 'MAG_ACA'])
    #stars['Guide_selected'] = False
    #for agasc_id in selected[0:n]['AGASC_ID']:
    #    stars['Guide_selected'][stars['AGASC_ID'] == agasc_id] = True
    return selected[0:n]

