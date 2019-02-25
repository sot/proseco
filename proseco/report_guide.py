# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import copy
from pathlib import Path

import matplotlib
matplotlib.use('agg')

from jinja2 import Template
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from chandra_aca.aca_image import ACAImage, AcaPsfLibrary
from chandra_aca.transform import mag_to_count_rate
from mica.archive.aca_dark.dark_cal import get_dark_cal_image

from .core import StarsTable, table_to_html
from .guide import GuideTable, get_ax_range, GUIDE

# Do reporting for at-most MAX_CAND candidates
MAX_CAND = 50
COLS = ['id', 'stage', 'forced',
        'mag', 'mag_err',
        'yang', 'zang', 'row', 'col',
        'COLOR1', 'ASPQ1', 'MAG_ACA_ERR']
FILEDIR = Path(__file__).parent
APL = AcaPsfLibrary()


def make_report(obsid, rootdir='.'):
    """
    Make summary HTML report for guide star selection.

    The first arg ``obsid`` can be an obsid (int) or a ``GuideTable`` object.
    For ``int`` the guide table is read from ``<rootdir>/obs<obsid>/guide.pkl``.

    Output is in ``<rootdir>/obs<obsid>/guide/index.html`` plus related images
    in that directory.

    :param obsid: int obsid or GuideTable instance
    :param rootdir: root directory for outputs

    :returns: GuideTable object (mostly for testing)
    """
    rootdir = Path(rootdir)

    if isinstance(obsid, GuideTable):
        guides = obsid
        obsid = guides.obsid
    else:
        guides = GuideTable.from_pickle(obsid, rootdir)

    # Define and make directories as needed
    obsdir = rootdir / f'obs{obsid:05}'
    outdir = obsdir / 'guide'
    outdir.mkdir(exist_ok=True, parents=True)

    cand_guides = guides.cand_guides.copy()
    cand_guides['sort_stage'] = cand_guides['stage']
    cand_guides['sort_stage'][cand_guides['stage'] == -1] = 1000
    cand_guides.sort(['sort_stage', 'mag'])
    if len(cand_guides) > MAX_CAND:
        cand_guides = cand_guides[0:MAX_CAND]

    context = copy(guides.meta)
    context['include_ids'] = ", ".join([str(val) for val in guides.include_ids])
    context['exclude_ids'] = ", ".join([str(val) for val in guides.exclude_ids])

    # Get stars if not already set (e.g. if guides is coming from a pickle).  If
    # guides.stars is already correctly defined this does nothing.
    guides.set_stars()

    # For include/exclude stars, add some bookkeeping (the forced column)
    cand_guides['forced'] = False
    for force_include in guides.include_ids_guide:
        cand_guides['forced'][cand_guides['id'] == force_include] = True
    for ex_cand_id in guides.exclude_ids_guide:
        # Stars that were excluded via exclude_ids are not kept as candidates
        # and have no debugging information.  Add them back at least to see
        # some of their properties in the report.
        match = guides.stars[guides.stars['id'] == ex_cand_id]
        if len(match) > 0:
            ex_cand = match[0]
        cand_guides.add_row()
        for col in cand_guides.colnames:
            if col in ex_cand.colnames:
                cand_guides[col][-1] = ex_cand[col]
        cand_guides['stage'][-1] = -1
        cand_guides['forced'][-1] = True

    make_cand_report(guides, cand_guides, context, outdir)

    # Guide star table
    cols = COLS.copy()
    cols.remove('forced')
    cols.remove('MAG_ACA_ERR')
    guides_table = guides[cols]
    guides_table['id'] = ['<a href=#{0}>{0}</a>'.format(guide['id'])
                          for guide in guides_table]
    context['guides_table'] = table_to_html(guides_table)

    # Candidate guides table
    cand_guides_table = cand_guides[COLS]
    cand_guides_table['id'] = ['<a href=#{0}>{0}</a>'.format(cand_guide['id'])
                               for cand_guide in cand_guides]
    context['cand_guides_table'] = table_to_html(cand_guides_table)

    # Make the HTML
    template_file = FILEDIR / GUIDE.index_template_file
    template = Template(open(template_file, 'r').read())
    out_html = template.render(context)
    out_filename = outdir / 'index.html'
    with open(out_filename, 'w') as fh:
        fh.write(out_html)

    return guides


def make_cand_report(guides, cand_guides, context, obsdir):

    n_stages = np.max(cand_guides['stage'])
    context['cand_guides'] = []
    for ii, guide in enumerate(cand_guides):
        select = 'SELECTED' if guide['id'] in guides['id'] else 'not selected'

        # Add debugging information if the star was in include/exclude lists
        if (guide['id'] in guides.include_ids_guide) or (guide['id'] in guides.exclude_ids_guide):
            select = select + ' - forced with include/exclude'
        rep = {
            'id': guide['id'],
            'selected': select
        }
        log = reject_info_to_report(guide['id'], guides.reject_info)
        if guide['stage'] != -1:
            if guide['stage'] == 0:
                log.append(f"FORCE Selected in stage 0")
            else:
                log.append(f"Selected in stage {guide['stage']}")
            if guide['id'] not in guides['id']:
                log.append(f"Did not make final cut of {len(guides)} stars")
        else:
            log.append(f"Not selected in any stage ({n_stages} stages run)")
        rep['log'] = log
        guide_table = cand_guides[COLS][ii:ii + 1].copy()
        guide_table['id'] = ['<a href="http://kadi.cfa.harvard.edu/star_hist/?agasc_id={0}" '
                             'target="_blank">{0}</a>'
                             .format(g['id']) for g in guide_table]
        rep['guide_table'] = table_to_html(guide_table)

        # Make the star detail plot
        basename = f'guide_candidate_{guide["id"]}.png'
        filename = obsdir / basename
        rep['candidate_plot'] = basename
        if not filename.exists():
            plot(guide, guides, filename=filename)
        context['cand_guides'].append(rep)


def reject_info_to_report(starid, reject_info):
    """
    For a given agasc_id, get all of the related "reject" info in an array
    """
    log = []
    for entry in reject_info:
        if entry['id'] != starid:
            continue
        log.append(f"Not selected stage {entry['stage']}: {entry['text']}")
    return log


def plot(guide, cand_guides, filename=None):
    """
    For a given candidate guide, make a plot of with a one subplot of possible spoiler stars
    and another subplot of the region of the dark map that would be dithered over.
    """
    # NOTE: all coordinates in plot funcs are "edge" coordinates.  Indexing an
    # ACAImage pixel using `.aca` coordinates implies using the row, col
    # coordinate of the lower-left corner of that pixel.

    fig = plt.figure(figsize=(8, 4))

    # First plot: star spoilers
    ax = fig.add_subplot(1, 2, 1)
    plot_spoilers(guide, cand_guides, ax)

    # Second plot: hot pixel imposters
    ax = fig.add_subplot(1, 2, 2)
    plot_imposters(guide, cand_guides, ax)

    plt.tight_layout()
    if filename is not None:
        # When Ska3 has matplotlib 2.2+ then just use `filename`
        plt.savefig(str(filename), pad_inches=0.0)

    plt.close(fig)


def plot_spoilers(guide, cand_guides, ax):
    """
    Make the spoilers plot for a given `guide` candidate.
    """
    # Define bounds of spoiler image plot: halfwidth, center, lower-left corner
    hw = 10
    rowc = int(round(guide['row']))
    colc = int(round(guide['col']))
    row0 = rowc - hw
    col0 = colc - hw

    # Pixel image canvas for plot
    region = ACAImage(np.zeros((hw * 2, hw * 2)),
                      row0=row0,
                      col0=col0)

    # Get spoilers
    stars = cand_guides.stars
    ok = ((np.abs(guide['row'] - stars['row']) < hw) &
          (np.abs(guide['col'] - stars['col']) < hw) &
          (stars['id'] != guide['id']))
    spoilers = stars[ok]

    # Add to image
    for spoil in spoilers:
        spoil_img = APL.get_psf_image(row=spoil['row'], col=spoil['col'], pix_zero_loc='edge',
                                      norm=mag_to_count_rate(spoil['mag']))
        region = region + spoil_img.aca

    # Get vmax from max pix value, but round to nearest 100, and create the image
    vmax = np.max(region).clip(100)
    vmax = np.ceil(vmax / 100) * 100
    ax.imshow(region.transpose(), cmap='hot', origin='lower', vmin=1, vmax=vmax,
              extent=(row0, row0 + hw * 2, col0, col0 + hw * 2))

    # Plot a box showing the 8x8 image boundaries
    x = rowc - 4
    y = colc - 4
    patch = patches.Rectangle((x, y), 8, 8, edgecolor='y', facecolor='none', lw=1)
    ax.add_patch(patch)

    # Make an empty box (no spoilers) or a cute 8x8 grid
    if len(spoilers) == 0:
        plt.text(rowc, colc, "No Spoilers", color='y', fontweight='bold',
                 ha='center', va='center')
    else:
        for i in range(1, 8):
            ax.plot([x, x + 8], [y + i, y + i], color='y', lw=1)
            ax.plot([x + i, x + i], [y, y + 8], color='y', lw=1)

    # Add mag text to each spoiler
    for spoil in spoilers:
        # Mag label above or below to ensure the label is in the image
        dcol = -3 if (spoil['col'] > colc) else 3
        plt.text(spoil['row'], spoil['col'] + dcol, f"mag={spoil['mag']:.1f}",
                 color='y', fontweight='bold', ha='center', va='center')

    plt.title(f"Spoiler stars (vmax={vmax:.0f})")
    ax.set_xlabel('Row')
    ax.set_ylabel('Column')


def plot_imposters(guide, cand_guides, ax):
    """
    Make the hot pixel imposters plot for a given `guide` candidate.
    """
    # Figure out pixel region for dithered-over-pixels plot
    row_extent = np.ceil(4 + cand_guides.dither.row)
    col_extent = np.ceil(4 + cand_guides.dither.col)
    rminus, rplus = get_ax_range(guide['row'], row_extent)
    cminus, cplus = get_ax_range(guide['col'], col_extent)

    # Pixel region of the guide
    img = ACAImage(np.zeros(shape=(rplus - rminus, cplus - cminus)),
                   row0=rminus, col0=cminus)
    dark = ACAImage(cand_guides.dark, row0=-512, col0=-512)
    img += dark.aca

    # Pixel region of the guide plus some space for annotation
    row0 = rminus - 4
    col0 = cminus - 4
    drow = (rplus - rminus) + 8
    dcol = (cplus - cminus) + 8
    canvas = ACAImage(np.zeros(shape=(drow, dcol)), row0=row0, col0=col0)
    canvas += img.aca

    ax.imshow(canvas.transpose(), interpolation='none', cmap='hot', origin='lower',
              vmin=50, vmax=3000, extent=(row0, row0 + drow, col0, col0 + dcol))

    # If force excluded, will not have imposter mag
    if not (guide['forced'] and guide['stage'] == -1):
        # Add max region with "mag"
        x = guide['imp_r']
        y = guide['imp_c']
        patch = patches.Rectangle((x, y), 2, 2, edgecolor='y', facecolor='none', lw=1.5)
        ax.add_patch(patch)
        plt.text(row0 + drow / 2, col0 + dcol - 1, f"box 'mag' {guide['imp_mag']:.1f}",
                 ha='center', va='center', color='y', fontweight='bold')

    # Plot a box showing the 8x8 image boundaries
    x = row0 + drow / 2 - 4
    y = col0 + dcol / 2 - 4
    patch = patches.Rectangle((x, y), 8, 8, edgecolor='y', facecolor='none', lw=1)
    ax.add_patch(patch)

    # Plot a box showing the 8x8 image boundaries
    patch = patches.Rectangle((rminus, cminus), rplus - rminus, cplus - cminus,
                              edgecolor='g', facecolor='none', lw=1)
    ax.add_patch(patch)

    ax.set_xlabel('Row')
    ax.set_ylabel('Column')
    plt.title("Dark Current in dither region")
