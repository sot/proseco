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

from .core import StarsTable
from .guide import GuideTable, get_ax_range
from .report import table_to_html

# Do reporting for at-most MAX_CAND candidates
MAX_CAND = 50
COLS = ['id', 'stage', 'forced',
        'mag', 'mag_err', 'MAG_ACA', 'MAG_ACA_ERR',
        'yang', 'zang', 'row', 'col', 'COLOR1']
FILEDIR = Path(__file__).parent
APL = AcaPsfLibrary()


def make_report(obsid, rootdir='.'):
    rootdir = Path(rootdir)
    print(f'Processing obsid {obsid}')

    obsdir = rootdir / f'obs{obsid:05}'
    guides = GuideTable.from_pickle(obsid, rootdir)

    cand_guides = guides.cand_guides
    cand_guides['sort_stage'] = cand_guides['stage']
    cand_guides['sort_stage'][cand_guides['stage'] == -1] = 1000
    cand_guides.sort(['sort_stage', 'mag'])
    if len(cand_guides) > MAX_CAND:
        cand_guides = cand_guides[0:MAX_CAND]

    context = copy(guides.meta)
    context['str_include_ids'] = ",".join([str(sid) for sid in guides.include_ids_guide])
    context['str_exclude_ids'] = ",".join([str(sid) for sid in guides.exclude_ids_guide])
    # Get information that is not stored in the acqs pickle for space reasons
    guides.stars = StarsTable.from_agasc(guides.att, date=guides.date)
    guides.dark = get_dark_cal_image(date=guides.date, select='before',
                                     t_ccd_ref=guides.t_ccd)

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

    make_cand_report(guides, cand_guides, context, obsdir)

    cand_guides_table = cand_guides[COLS]
    cand_guides_table['id'] = ['<a href=#{0}>{0}</a>'.format(cand_guide['id'])
                               for cand_guide in cand_guides]
    context['cand_guides_table'] = table_to_html(cand_guides_table)
    template_file = FILEDIR / 'guide_index_template.html'
    template = Template(open(template_file, 'r').read())
    out_html = template.render(context)
    out_filename = obsdir / 'guide_index.html'
    with open(out_filename, 'w') as fh:
        fh.write(out_html)


def make_cand_report(guides, cand_guides, context, obsdir):

    n_stages = np.max(cand_guides['stage'])
    context['cand_guides'] = []
    for ii, guide in enumerate(cand_guides):
        print('Doing detail for star {}'.format(guide['id']))
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


def plot(star, startable, filename=None):
    """
    For a given candidate star, make a plot of with a one subplot of possible spoiler stars
    and another subplot of the region of the dark map that would be dithered over.
    """
    hw = 10  # pixel halfwidth for spoiler plot
    region = ACAImage(np.zeros((hw * 2, hw * 2)),
                      row0=int(round(star['row'])) - hw,
                      col0=int(round(star['col'])) - hw)
    # Get spoilers
    stars = startable.stars
    ok = ((np.abs(star['row'] - stars['row']) < hw) &
          (np.abs(star['col'] - stars['col']) < hw) &
          (stars['id'] != star['id']))
    spoilers = stars[ok]

    # Add to image
    for spoil in spoilers:
        spoil_img = APL.get_psf_image(row=spoil['row'], col=spoil['col'], pix_zero_loc='edge',
                                      norm=mag_to_count_rate(spoil['mag']))
        region = region + spoil_img.aca

    # Plot spoilers in first subplot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 2, 1)
    vmax = max(np.max(region), 200)
    vmax = np.round(vmax / 100) * 100
    ax.imshow(region.transpose(), interpolation='none', cmap='hot', origin='lower',
              vmin=1, vmax=vmax)
    x = hw - 4 - 0.5
    y = hw - 4 - 0.5
    patch = patches.Rectangle((x, y), 8, 8, edgecolor='y', facecolor='none', lw=1)
    ax.add_patch(patch)
    if len(spoilers) == 0:
        plt.text(6.5, 9.5, "No Spoilers", color='y', fontweight='bold')
    else:
        # Add a cute 8x8 grid
        for i in range(1, 8):
            patch = patches.Rectangle((x + i, y + i),
                                      8 - i, 8 - i, edgecolor='y', facecolor='none', lw=1)
            ax.add_patch(patch)
            patch = patches.Rectangle((x, y),
                                      8 - i, 8 - i, edgecolor='y', facecolor='none', lw=1)
            ax.add_patch(patch)
    plt.title(f"Spoiler stars (vmax={vmax:.0f})")
    # Borrow Tom's hack for row/column ticks from acq reporting.
    # Hack to fix up ticks to have proper row/col coords.  There must be a
    # correct way to do this.
    xticks = [str(int(label) + region.row0) for label in ax.get_xticks().tolist()]
    ax.set_xticklabels(xticks)
    ax.set_xlabel('Row')
    yticks = [str(int(label) + region.col0) for label in ax.get_yticks().tolist()]
    ax.set_yticklabels(yticks)
    ax.set_ylabel('Column')

    # Figure out pixel region for dithered-over-pixels plot
    row_extent = np.ceil(4 + startable.dither.row)
    col_extent = np.ceil(4 + startable.dither.col)
    rminus, rplus = get_ax_range(star['row'], row_extent)
    cminus, cplus = get_ax_range(star['col'], col_extent)

    # Pixel region of the star
    img = ACAImage(np.zeros(shape=(rplus - rminus, cplus - cminus)),
                   row0=rminus, col0=cminus)
    dark = ACAImage(startable.dark, row0=-512, col0=-512)
    img += dark.aca

    # Pixel region of the star plus some space for annotation
    canvas = ACAImage(np.zeros(shape=((rplus - rminus) + 8, (cplus - cminus) + 8)),
                      row0=rminus - 4, col0=cminus - 4)
    canvas += img.aca

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(canvas.transpose(), interpolation='none', cmap='hot', origin='lower',
              vmin=50, vmax=3000)

    # If force excluded, will not have imposter mag
    if not (star['forced'] and star['stage'] == -1):
        # Add max region with "mag"
        x = star['imp_r'] - canvas.row0 - .5
        y = star['imp_c'] - canvas.col0 - .5
        patch = patches.Rectangle((x, y), 2, 2, edgecolor='y', facecolor='none', lw=1.5)
        ax.add_patch(patch)
        plt.text(row_extent - 2, cplus + 1 - canvas.col0, f"box 'mag' {star['imp_mag']:.1f}",
                 color='y', fontweight='bold')

    # Borrow Tom's hack for row/column ticks from acq reporting.
    # Hack to fix up ticks to have proper row/col coords.  There must be a
    # correct way to do this.
    xticks = [str(int(label) + canvas.row0) for label in ax.get_xticks().tolist()]
    ax.set_xticklabels(xticks)
    ax.set_xlabel('Row')
    yticks = [str(int(label) + canvas.col0) for label in ax.get_yticks().tolist()]
    ax.set_yticklabels(yticks)
    ax.set_ylabel('Column')
    plt.title("Dark Current in dither region")

    if filename is not None:
        # When Ska3 has matplotlib 2.2+ then just use `filename`
        plt.savefig(str(filename), pad_inches=0.0)

    plt.tight_layout()
    plt.close(fig)
