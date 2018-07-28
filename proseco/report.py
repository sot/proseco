import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from chandra_aca.aca_image import ACAImage

from . import characteristics as CHAR


def plot_acq(acq, vmin=100, vmax=2000, figsize=(8, 8), r=None, c=None, filename=None):
    """
    Plot dark current, relevant boxes, imposters and spoilers.
    """
    acqs = acq.table  # Parent table of this row
    drc = int(np.max(CHAR.box_sizes) / 5 * 2) + 5

    # Make an image of `dark` centered on acq row, col.  Use ACAImage
    # arithmetic to handle the corner case where row/col are near edge
    # and a square image goes off the edge.
    if r is None:
        r = int(np.round(acq['row']))
        c = int(np.round(acq['col']))
    img = ACAImage(np.zeros(shape=(drc * 2, drc * 2)), row0=r - drc, col0=c - drc)
    dark = ACAImage(acqs.meta['dark'], row0=-512, col0=-512)
    img += dark.aca

    # Show the dark current image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(img.transpose(), interpolation='none', cmap='hot', origin='lower',
              vmin=vmin, vmax=vmax)

    # Imposter stars
    for idx, imp in enumerate(acq['imposters']):
        r = imp['row0'] - img.row0
        c = imp['col0'] - img.col0
        patch = patches.Rectangle((r + 0.5, c + 0.5), 6, 6,
                                  edgecolor="y", facecolor="none", lw=1.5)
        ax.add_patch(patch)
        plt.text(r + 7, c + 7, str(idx), color='y', fontsize='large', fontweight='bold')

    # Box regions
    rc = img.shape[0] // 2 + 0.5
    cc = img.shape[1] // 2 + 0.5
    exts = np.array([0, 60, 120, 160])
    labels = ['Search', 'Normal', 'Big', 'Anomalous']
    for ext, label in zip(exts, labels):
        hw = acq['halfw'] + ext
        hwp = hw / 5
        lw = 3 if ext == 0 else 1
        patch = patches.Rectangle((rc - hwp, cc - hwp), hwp * 2, hwp * 2, edgecolor='r',
                                  facecolor='none', lw=lw, alpha=1)
        ax.add_patch(patch)
        plt.text(rc - hwp + 1, cc - hwp + 1, label, color='y', fontsize='large', fontweight='bold')

    # Field stars
    stars = acqs.meta['stars']
    ok = ((stars['row'] > img.row0) & (stars['row'] < img.row0 + img.shape[0]) &
          (stars['col'] > img.col0) & (stars['col'] < img.col0 + img.shape[1]))
    if np.any(ok):
        stars = stars[ok]
        d_mags = stars['mag'] - acq['mag']
        d_mag_errs = np.sqrt(stars['mag_err'] ** 2 + acq['mag_err'] ** 2)
        d_sigmas = d_mags / d_mag_errs
        rads = np.clip(2 - d_sigmas, 1, 4)
        rads_dict = {}

        for color, sig0, sig1, maxmag in (('r', -100, 0.5, 15),  # Spoiler is brighter or close
                                          ('y', 0.5, 2, 15),  # Spoiler within 0.5 to 1.5 sigma
                                          ('g', 2, 100, 11.5)):  # Spoiler between 1.5 to 2 sigma
            ok = (d_sigmas > sig0) & (d_sigmas <= sig1) & (stars['mag'] < maxmag)
            for star, rad in zip(stars[ok], rads[ok]):
                r = star['row'] - img.row0
                c = star['col'] - img.col0
                patch = patches.Circle((r + 0.5, c + 0.5), rad, edgecolor=color,
                                       facecolor="none", lw=2.5)
                ax.add_patch(patch)
                rads_dict[star['AGASC_ID']] = rad

    # Spoiler stars
    for idx, sp in enumerate(acq['spoilers']):
        r = sp['row'] - img.row0
        c = sp['col'] - img.col0
        rad = rads_dict[sp['AGASC_ID']] / np.sqrt(2)
        plt.text(r + rad + 1, c + rad + 1, str(idx), color='y', fontweight='bold')

    # Hack to fix up ticks to have proper row/col coords.  There must be a
    # correct way to do this.
    xticks = [str(int(label) + img.row0) for label in ax.get_xticks().tolist()]
    ax.set_xticklabels(xticks)
    ax.set_xlabel('Row')
    yticks = [str(int(label) + img.col0) for label in ax.get_yticks().tolist()]
    ax.set_yticklabels(yticks)
    ax.set_ylabel('Column')

    if filename is not None:
        plt.savefig(filename)

    return img, ax
