from itertools import count

import numpy as np
import numba

from chandra_aca.aca_image import AcaPsfLibrary
from chandra_aca.transform import mag_to_count_rate, pixels_to_yagzag
from chandra_aca.attitude import calc_roll_pitch_yaw

APL = AcaPsfLibrary()

ROWB = np.array([0, 0, 0, 0, 7, 7, 7, 7])
COLB = np.array([0, 1, 6, 7, 0, 1, 6, 7])
RCB = ROWB * 8 + COLB
MIN_FLOAT_64 = np.finfo(np.float64).min


@numba.jit(nopython=True)
def nanargmax(arr):
    max_val = MIN_FLOAT_64
    argmax = -1
    for ii in range(len(arr)):
        val = arr[ii]
        if not np.isnan(val) and val > max_val:
            max_val = val
            argmax = ii
    return argmax


@numba.jit(nopython=True)
def calc_flight_background(image):
    """
    Calculate average background of the input 8x8 ``image`` using
    the flight PEA background algorithm.
    :param image: 8x8 image (ndarray)
    :returns: bgd_avg
    """
    vals = image.flatten()[RCB]
    while True:
        avg = np.round(np.nanmean(vals))
        sigma = np.maximum(avg * 1.5, 10.0)
        dev = np.abs(vals - avg)
        imax = nanargmax(dev)
        if dev[imax] > sigma:
            vals[imax] = np.nan
        else:
            break

    return avg


def lofi(guides, n_read=300, dt=4.1, dark=None):
    if dark is None:
        dark = guides.dark
    dark = dark.astype(np.float64)  # Req'd for numba calc_flight_background()

    times = np.arange(n_read) * dt
    omega_r = 2 * np.pi / 1000
    omega_c = 2 * np.pi / 707
    dither_rs = guides.dither.row * np.sin(omega_r * times)
    dither_cs = guides.dither.col * np.sin(omega_c * times)

    sdrs = []
    for guide in guides:
        sdr = star_track(guide, dither_rs, dither_cs, dark=dark, stars=guides.stars)
        sdrs.append(sdr)

    return sdrs


def star_track(guide, dither_rs, dither_cs, dark, stars):
    # Find all stars with centroid within a 9-pixel halfw box of guide
    # Note pix_zero_loc = 'edge' for all these
    guide_row0 = guide['row']
    guide_col0 = guide['col']
    ok = ((np.abs(stars['row'] - guide_row0) < 9) &
          (np.abs(stars['col'] - guide_col0) < 9))
    star_row0s = stars['row'][ok]
    star_col0s = stars['col'][ok]
    star_norms = mag_to_count_rate(stars['mag'][ok])

    img_row = guide_row0
    img_col = guide_col0

    # Initial rate
    rate_row = 0.0
    rate_col = 0.0

    cent_rows = np.zeros_like(dither_rs)
    cent_cols = np.zeros_like(dither_rs)
    norms = np.zeros_like(dither_rs)
    img_row0s = np.zeros_like(dither_rs)
    img_col0s = np.zeros_like(dither_rs)

    for idx, dither_r, dither_c in zip(count(), dither_rs, dither_cs):
        # Next image location center as floats
        img_row = (img_row + rate_row).clip(-512 + 4, 512 - 4)
        img_col = (img_col + rate_col).clip(-512 + 4, 512 - 4)

        # Image readout lower left corner
        img_row0 = int(round(img_row)) - 4
        img_col0 = int(round(img_col)) - 4
        img_row0s[idx] = img_row0
        img_col0s[idx] = img_col0

        # Definitely optimize this later, making new ACAImage is expensive
        img = dark.aca[img_row0:img_row0 + 8, img_col0:img_col0 + 8].copy()

        # Shine star images onto img
        for star_row0, star_col0, star_norm in zip(star_row0s, star_col0s, star_norms):
            star_row = star_row0 + dither_r
            star_col = star_col0 + dither_c
            img_star = APL.get_psf_image(star_row, star_col, star_norm, pix_zero_loc='edge')
            img += img_star.aca

        bgd = calc_flight_background(np.asarray(img))
        cent_rows[idx], cent_cols[idx], norms[idx] = img.aca.centroid_fm(
            bgd=bgd, pix_zero_loc='edge', norm_clip=10)
        # print(f'cent_row={cent_rows[idx]:.3f}')

        rate_row = cent_rows[idx] - img_row
        rate_col = cent_cols[idx] - img_col

    out = dict(cent_row=cent_rows, cent_col=cent_cols, norm=norms,
               img_row0=img_row0s, img_col0=img_col0s)

    return out


def calc_lofi_roll_pitch_yaw(guides, sdrs):
    yag = guides['yang']
    zag = guides['zang']
    yag_obs = np.zeros(shape=(len(sdrs[0]['cent_row']), len(guides)))
    zag_obs = np.zeros_like(yag_obs)
    for idx, sdr in enumerate(sdrs):
        yags, zags = pixels_to_yagzag(sdr['cent_row'], sdr['cent_col'], pix_zero_loc='edge')
        yag_obs[:, idx] = yags
        zag_obs[:, idx] = zags
    roll, pitch, yaw = calc_roll_pitch_yaw(yag, zag, yag_obs, zag_obs)
    
    return roll * 3600, pitch * 3600, yaw * 3600
