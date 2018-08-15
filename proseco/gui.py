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
from . gui_char import CHAR

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


DITHER = 8
MANVR_ERROR = 60
FIELD_ERROR_PAD = 0

def set_dither(dither):
    global DITHER
    DITHER = dither


def set_manvr_error(manvr_error):
    global MANVR_ERROR
    MANVR_ERROR = manvr_error


def check_off_chips(cone_stars, opt):
    ypos = cone_stars['row']
    zpos = cone_stars['col']
    yPixLim = STAR_CHAR['General']['Body']['Pixels']['YPixLim']
    zPixLim = STAR_CHAR['General']['Body']['Pixels']['ZPixLim']
    edgeBuffer = STAR_CHAR['General']['Body']['Pixels']['EdgeBuffer']
    pad = DITHER * ARC_2_PIX
    yn = (yPixLim[0] + (pad + edgeBuffer))
    yp = (yPixLim[1] - (pad + edgeBuffer))
    zn = (zPixLim[0] + (pad + edgeBuffer))
    zp = (zPixLim[1] - (pad + edgeBuffer))
    ydist = np.min([yp - ypos, ypos - yn], axis=0)
    zdist = np.min([zp - zpos, zpos - zn], axis=0)
    chip_edge_dist = np.min([ydist, zdist], axis=0)
    offchip = chip_edge_dist < 0
    yag = cone_stars['yang']
    zag = cone_stars['zang']
    arcsec_pad = DITHER
    yArcSecLim = STAR_CHAR['General']['FOV']['YArcSecLim']
    ZArcSecLim = STAR_CHAR['General']['FOV']['ZArcSecLim']
    arcsec_yn = yArcSecLim[0] + arcsec_pad
    arcsec_yp = yArcSecLim[1] - arcsec_pad
    arcsec_zn = ZArcSecLim[0] + arcsec_pad
    arcsec_zp = ZArcSecLim[1] - arcsec_pad
    arcsec_ydist = np.min([arcsec_yp - yag, yag - arcsec_yn], axis=0)
    arcsec_zdist = np.min([arcsec_zp - zag, zag - arcsec_zn], axis=0)
    fov_edge_dist = np.min([arcsec_ydist, arcsec_zdist], axis=0)
    outofbounds = (yag < arcsec_yn) | (yag > arcsec_yp) | (zag < arcsec_zn) | (zag > arcsec_zp)
    return chip_edge_dist, fov_edge_dist, offchip, outofbounds


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


def check_bad_pixels(cone_stars, not_bad, opt):
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
    pad = .5 + DITHER * ARC_2_PIX
    for i, rs, cs in zip(count(), row[not_bad], col[not_bad]):
        in_reg_r = (rs >= (bp[:,0] - pad)) & (rs <= (bp[:,1] + pad))
        in_reg_c = (cs >= (bp[:,2] - pad)) & (cs <= (bp[:,3] + pad))
        r_dist = np.min(np.abs(
                [rs - (bp[:,0] - pad), rs - (bp[:,1] + pad)]), axis=0)
        r_dist[in_reg_r] = 0
        c_dist = np.min(np.abs(
                [cs - (bp[:,2] - pad), cs - (bp[:,3] + pad)]), axis=0)
        c_dist[in_reg_c] = 0
        # For the nearest manhattan distance we want the max in each axis
        maxes = np.max(np.vstack([r_dist, c_dist]), axis=0)
        # And then the minimum of the maxes
        idxmatch = np.argmin(maxes)
        distance[i] = maxes[idxmatch]
    full_distance = np.ones(len(cone_stars)) * 9999
    full_distance[not_bad] = distance
    return full_distance


def dist_to_bright_spoiler(cone_stars, ok, nSigma, opt):
    magOneSigError = cone_stars['mag_one_sig_err']
    row, col = cone_stars['row'], cone_stars['col']
    mag = cone_stars['MAG_ACA']
    magerr2 = cone_stars['mag_one_sig_err2']
    errorpad = (FIELD_ERROR_PAD + DITHER) * ARC_2_PIX
    dist = np.ones(len(cone_stars)) * 9999
    for cand, cand_magerr, idx in zip(cone_stars[ok],
                                      magOneSigError[ok],
                                      np.flatnonzero(ok)):
        mag_diff = (cand['MAG_ACA']
                    - mag
                    + nSigma * np.sqrt(cand_magerr ** 2 + magerr2))
        brighter = mag_diff > 0
        brighter[idx] = False
        if not np.any(brighter):
            continue
        cand_row, cand_col = row[idx], col[idx]
        rdiff = np.abs(cand_row - row[brighter])
        cdiff = np.abs(cand_col - col[brighter])
        match = np.argmin(np.max([rdiff, cdiff], axis=0))
        dist[idx] = np.max([rdiff, cdiff], axis=0)[match] - errorpad
    return dist


def check_column(cone_stars, not_bad, opt, chip_pos):
    zpixlim = STAR_CHAR['General']['Body']['Pixels']['ZPixLim']
    ypixlim = STAR_CHAR['General']['Body']['Pixels']['YPixLim']
    center = STAR_CHAR['General']['Body']['Pixels']['Center']
    Column = STAR_CHAR['General']['Body']['Column']
    Register = opt['Body']['Register']
    nSigma = opt['Spoiler']['SigErrMultiplier']
    row, col = chip_pos
    starmag = cone_stars['MAG_ACA']
    magerr2 = cone_stars['mag_one_sig_err2']
    register_pad = DITHER * ARC_2_PIX
    column_pad = FIELD_ERROR_PAD * ARC_2_PIX
    pass


def check_stage(cone_stars, not_bad, opt, label):
    stype = opt['Type']
    mag_ok = check_mag(cone_stars, opt, label)
    ok = mag_ok & not_bad
    if not np.any(ok):
        return ok
    nSigma = opt['Spoiler']['SigErrMultiplier']
    mag_spoiled = ~ok.copy()
    mag_check_col = 'mag_spoil_check_{}_{}'.format(nSigma, stype)
    if mag_check_col not in cone_stars.columns:
        cone_stars[mag_check_col] = np.zeros_like(not_bad)
        cone_stars['mag_spoiled_{}_{}'.format(nSigma, stype)] = np.zeros_like(not_bad)
    mag_spoiled, checked= check_mag_spoilers(cone_stars, ok, opt)
    cone_stars[mag_check_col] = cone_stars[mag_check_col] | checked
    cone_stars['mag_spoiled_{}_{}'.format(nSigma, stype)] = (
        cone_stars['mag_spoiled_{}_{}'.format(nSigma, stype)] | mag_spoiled)
    bad_pix_dist = check_bad_pixels(cone_stars, ok & ~mag_spoiled, opt)
    cone_stars['bad_pix_dist_{}'.format(stype)] = bad_pix_dist

    # these star distance checks are in pixels, so just do them for
    # every roll
    cone_stars['star_dist_{}_{}'.format(nSigma, stype)] = 9999
    star_dist = dist_to_bright_spoiler(cone_stars, ok & ~mag_spoiled, nSigma, opt)
    cone_stars['star_dist_{}_{}'.format(nSigma, stype)] = np.min(
                [cone_stars['star_dist_{}_{}'.format(nSigma, stype)], star_dist],
                axis=0)

    maxBoxArc = STAR_CHAR['General']['Select']['MaxSearchBox'] # 25
    minBoxArc = STAR_CHAR['General']['Select']['MinSearchBox']

    starBox = np.min([cone_stars['star_dist_{}_{}'.format(nSigma, stype)],
                      cone_stars['bad_pix_dist_{}'.format(stype)],
                      cone_stars['chip_edge_dist_{}'.format(stype)],
                      cone_stars['fov_edge_dist_{}'.format(stype)] * ARC_2_PIX], axis=0)
    box_size_arc = ((starBox * PIX_2_ARC) // 5) * 5
    box_size_arc[box_size_arc > maxBoxArc] = maxBoxArc
    cone_stars['box_size_arc_{}'.format(stype)] = box_size_arc
    bad_box = starBox < (minBoxArc * ARC_2_PIX)
    #cone_stars['bad_box_{}'.format(stype)] = bad_box
#    if opt['SearchSettings']['DoColumnRegisterCheck']:
#        badcolumn = check_column(cone_stars, ok & ~mag_spoiled & ~bad_dist, opt, chip_pos)
#        ok = ok & ~badcolumn
#    if opt['SearchSettings']['DoBminusVCheck']:
#        badbv = check_bv(cone_stars, ok & ~mag_spoiled & ~bad_dist)
#        ok = ok & ~badbv
    return ok & ~bad_box & ~mag_spoiled


def get_mag_errs(cone_stars, opt):
    caterr = cone_stars['MAG_ACA_ERR'] / 100.
    caterr = np.min([np.ones(len(caterr))*opt['Inertial']['MaxMagError'], caterr], axis=0)
    randerr = opt['Inertial']['MagErrRand']
    magOneSigError = np.sqrt(randerr*randerr + caterr*caterr)
    return magOneSigError, magOneSigError**2


def select_stage_stars(ra, dec, roll, cone_stars):

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
    # update these for every new roll
    cone_stars['yang'] = yag_deg * 3600
    cone_stars['zang'] = zag_deg * 3600
    cone_stars['row'] = row
    cone_stars['col'] = col

    # none of these appear stage dependent, but they could be type (guide/acq) dependent
    chip_edge_dist, fov_edge_dist, offchip, outofbounds = check_off_chips(cone_stars, opt)
    #cone_stars['offchip_{}'.format(stype)] = offchip
    #cone_stars['outofbounds_{}'.format(stype)] = outofbounds
    cone_stars['chip_edge_dist_{}'.format(stype)] = chip_edge_dist
    cone_stars['fov_edge_dist_{}'.format(stype)] = fov_edge_dist

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

    not_bad = (~offchip & ~outofbounds & ~bad_mag_error & ~bad_pos_error
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
                                 stage_char, "{}_{}".format(stype, str(idx)))
            cone_stars['{}_stage'.format(stype)][stage] = idx
    selected = cone_stars[cone_stars['{}_stage'.format(stype)] != -1]
    return selected



def select_guide_stars(ra, dec, roll, dither=8, n=5, cone_stars=None, date=None):
    if cone_stars is None:
        cone_stars = agasc.get_agasc_cone(ra, dec, radius=2, date=date,
                                          agasc_file='/proj/sot/ska/data/agasc/agasc1p6.h5')
    set_dither(dither)
    selected = select_stage_stars(ra, dec, roll, cone_stars)
    # Ignore guide star code to use ACA matrix etc to optimize selection of stars in the last
    # stage and just take these by stage and then magnitude
    selected.sort(['Guide_stage', 'MAG_ACA'])
    #cone_stars['Guide_selected'] = False
    #for agasc_id in selected[0:n]['AGASC_ID']:
    #    cone_stars['Guide_selected'][cone_stars['AGASC_ID'] == agasc_id] = True
    return selected[0:n]

