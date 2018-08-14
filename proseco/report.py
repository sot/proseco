# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import division, print_function, absolute_import  # For Py2 compatibility

import os
from copy import copy

import matplotlib
matplotlib.use('agg')
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from astropy.table import Table, Column

from chandra_aca.aca_image import ACAImage

from . import characteristics as CHAR
from .acq import AcqTable, get_stars, get_acq_candidates
from chandra_aca.plot import plot_stars
from mica.archive.aca_dark.dark_cal import get_dark_cal_image


FILEDIR = os.path.dirname(__file__)


def table_to_html(tbl):
    out = tbl._base_repr_(html=True, max_width=-1,
                          show_dtype=False, descr_vals=[],
                          max_lines=-1, tableclass='table-striped')
    return out


def get_p_acqs_table(acq, p_name):
    """
    Make HTML tables for an acq star for the following:

    - ``p_brightest``: probability this star is the brightest in box (function
        of ``box_size`` and ``man_err``)
    - ``p_acq_model``: probability of acquisition from the chandra_aca model
        (function of ``box_size``)
    - ``p_on_ccd``: probability star is on the usable part of the CCD (function
        of ``man_err`` and ``dither``)
    - ``p_acqs``: product of the above three
    """
    man_errs = CHAR.p_man_errs['man_err_hi']
    box_sizes = sorted(CHAR.box_sizes)
    names = ['box \ man_err'] + ['{}"'.format(man_err) for man_err in man_errs]
    cols = {}
    cols['box \ man_err'] = ['{}"'.format(box_size) for box_size in box_sizes]
    for man_err in man_errs:
        name = '{}"'.format(man_err)
        cols[name] = [round(acq[p_name].get((box_size, man_err), 0.0), 3)
                      for box_size in box_sizes]

    return table_to_html(Table(cols, names=names))


def get_p_on_ccd_table(acq):
    """
    Make HTML tables for an acq star for the following:

    - ``p_brightest``: probability this star is the brightest in box (function
        of ``box_size`` and ``man_err``)
    - ``p_acq_model``: probability of acquisition from the chandra_aca model
        (function of ``box_size``)
    - ``p_on_ccd``: probability star is on the usable part of the CCD (function
        of ``man_err`` and ``dither``)
    - ``p_acqs``: product of the above three
    """
    man_errs = CHAR.p_man_errs['man_err_hi']
    names = ['man_err'] + ['{}"'.format(man_err) for man_err in man_errs]
    cols = {}
    cols['man_err'] = ['']
    for man_err in man_errs:
        name = '{}"'.format(man_err)
        cols[name] = [round(acq['p_on_ccd'][man_err], 3)]

    return table_to_html(Table(cols, names=names))


def get_p_acq_model_table(acq):
    """
    Make HTML tables for an acq star for the following:

    - ``p_acq_model``: probability of acquisition from the chandra_aca model
        (function of ``box_size``)
    """
    box_sizes = sorted(CHAR.box_sizes)
    names = ['box size'] + ['{}"'.format(box_size) for box_size in box_sizes]
    cols = {}
    cols['box size'] = ['']
    for box_size in box_sizes:
        name = '{}"'.format(box_size)
        cols[name] = [round(acq['p_acq_model'][box_size], 3)]

    return table_to_html(Table(cols, names=names))


def select_events(events, funcs, **select):
    outs = [event for event in events
            if event['func'] in funcs and
            all(event.get(key) == val for key, val in select.items())]
    return outs


def make_events(acqs):
    # Set up global events
    events = copy(acqs.log_info['events'])
    for event in events:
        event['func_disp'] = event['func']
    last = events[0]
    for event in events[1:]:
        if event['dt'] == last['dt'] and event['func'] == last['func']:
            event['dt'] = ''
            event['func_disp'] = ''
        else:
            last = event
        event['data'] = '&nbsp;' * event.get('level', 0) * 4 + event['data']

    return events


def make_p_man_errs_report(context):
    tbl = CHAR.p_man_errs.copy()
    man_err = ['{}-{}"'.format(lo, hi)
               for lo, hi in zip(tbl['man_err_lo'],
                                 tbl['man_err_hi'])]
    del tbl['man_err_lo']
    del tbl['man_err_hi']

    for name in tbl.colnames:
        tbl.rename_column(name, name + '°')

    for col in tbl.columns.values():
        col[:] = np.round(col, 4)

    tbl.add_column(Column(man_err, name='err \ angle'), 0)

    context['p_man_errs_table'] = table_to_html(tbl)


def make_cand_acqs_report(acqs, cand_acqs, events, context, obsdir):
    ######################################################
    # Candidate acquisition stars and initial catalog
    ######################################################
    # Start with table
    cand_acq_cols = ['idx', 'id', 'yang', 'zang',
                     'row', 'col', 'mag', 'mag_err', 'color']
    context['cand_acqs_table'] = table_to_html(cand_acqs[cand_acq_cols])

    context['cand_acqs_events'] = select_events(events, ('get_acq_catalog',
                                                         'get_stars',
                                                         'get_acq_candidates'))

    # Get information that is not stored in the info.yaml for space reasons
    acqs.meta['stars'] = get_stars(acqs, acqs.meta['att'])
    _, acqs.meta['bad_stars'] = get_acq_candidates(acqs, acqs.meta['stars'])

    # Now plot figure
    filename = os.path.join(obsdir, 'candidate_stars.png')
    if not os.path.exists(filename):
        print('Making candidate stars plot {}'.format(filename))

        # Pull a fast-one and mark the final selected ACQ stars as BOT so they
        # get a circle in the plot.  This might be confusing and need fixing
        # later, but for now it is an easy way to show the winning candidates.
        for acq in acqs.meta['cand_acqs']:
            if acq['id'] in acqs['id']:
                acq['type'] = 'BOT'

        fig = plot_stars(acqs.meta['att'], stars=acqs.meta['stars'],
                         catalog=acqs.meta['cand_acqs'], bad_stars=acqs.meta['bad_stars'])
        fig.savefig(filename)


def make_initial_cat_report(events, context):
    context['initial_cat_events'] = select_events(events, ('get_initial_catalog',
                                                           'select_best_p_acqs'))


def make_acq_star_details_report(acqs, cand_acqs, events, context, obsdir):
    ######################################################
    # Candidate acq star detail sections
    ######################################################
    acqs.meta['dark'] = get_dark_cal_image(date=acqs.meta['date'], select='nearest',
                                           t_ccd_ref=acqs.meta['t_ccd'])

    context['cand_acqs'] = []
    for ii, acq in enumerate(cand_acqs):
        print('Doing detail for star {}'.format(acq['id']))
        # Local context dict for each cand_acq star
        cca = {'id': acq['id'],
               'selected': 'SELECTED' if acq['id'] in acqs['id'] else 'not selected'}

        # Events related to this ACQ ID
        cca['initial_selection_events'] = select_events(events, 'select_best_p_acqs', id=acq['id'])
        cca['optimize_events'] = select_events(events,
                                               ('optimize_catalog', 'optimize_acq_halfw'),
                                               id=acq['id'])
        # Make a dict copy of everything in ``acq``
        names = ('idx', 'id', 'ra', 'dec', 'yang', 'zang', 'row', 'col',
                 'mag', 'mag_err')
        cca['acq_table'] = table_to_html(cand_acqs[names][ii:ii + 1])

        cca['p_brightest_table'] = get_p_acqs_table(acq, 'p_brightest')
        cca['p_acqs_table'] = get_p_acqs_table(acq, 'p_acqs')
        cca['p_acq_model_table'] = get_p_acq_model_table(acq)
        cca['p_on_ccd_table'] = get_p_on_ccd_table(acq)

        # Make the acq detail plot with spoilers and imposters
        basename = 'star_{}.png'.format(acq['id'])
        filename = os.path.join(obsdir, basename)
        cca['acq_plot'] = basename
        if not os.path.exists(filename):
            plot_acq(acq, acqs.meta['dark'], acqs.meta['stars'], filename=filename)

        if len(acq['imposters']) > 0:
            names = ('row0', 'col0', 'd_row', 'd_col', 'img_sum', 'mag', 'mag_err')
            fmts = ('d', 'd', '.0f', '.0f', '.0f', '.2f', '.2f')
            imposters = acq['imposters'][names]
            for name, fmt in zip(names, fmts):
                imposters[name].info.format = fmt

            idx = Column(np.arange(len(acq['imposters'])), name='idx')
            imposters.add_column(idx, index=0)

            cca['imposters_table'] = table_to_html(imposters)
        else:
            cca['imposters_table'] = '<em> No imposters </em>'

        if len(acq['spoilers']) > 0:
            names = ('id', 'yang', 'zang', 'mag', 'mag_err')
            fmts = ('d', '.1f', '.1f', '.2f', '.2f')
            spoilers = acq['spoilers'][names]
            for name, fmt in zip(names, fmts):
                spoilers[name].info.format = fmt

            idx = Column(np.arange(len(spoilers)), name='idx')
            d_yang = Column(spoilers['yang'] - acq['yang'], name='d_yang', format='.1f')
            d_zang = Column(spoilers['zang'] - acq['zang'], name='d_zang', format='.1f')
            spoilers.add_column(d_zang, index=3)
            spoilers.add_column(d_yang, index=3)
            spoilers.add_column(idx, index=0)

            cca['spoilers_table'] = table_to_html(spoilers)
        else:
            cca['spoilers_table'] = '<em> No spoilers </em>'

        context['cand_acqs'].append(cca)


def make_optimize_catalog_report(events, context):
    context['optimize_events'] = select_events(events, ('calc_p_safe',
                                                        'optimize_catalog',
                                                        'optimize_acq_halfw'))


def make_report(obsid, rootdir='.'):
    print('Processing obsid {}'.format(obsid))

    obsdir = os.path.join(rootdir, 'obs{:05}'.format(obsid))
    filename = os.path.join(obsdir, 'acqs.yaml')
    print('Reading and parsing {}'.format(filename))
    with open(filename, 'r') as fh:
        yaml_str = fh.read()
    acqs = AcqTable.from_yaml(yaml_str)
    cand_acqs = acqs.meta['cand_acqs']
    context = copy(acqs.meta)

    events = make_events(acqs)
    context['events'] = events

    make_p_man_errs_report(context)
    make_cand_acqs_report(acqs, cand_acqs, events, context, obsdir)
    make_initial_cat_report(events, context)
    make_acq_star_details_report(acqs, cand_acqs, events, context, obsdir)
    make_optimize_catalog_report(events, context)

    template_file = os.path.join(FILEDIR, 'index_template.html')
    template = Template(open(template_file, 'r').read())
    out_html = template.render(context)
    out_filename = os.path.join(obsdir, 'index.html')
    with open(out_filename, 'w') as fh:
        fh.write(out_html)

    return acqs


def plot_acq(acq, dark, stars, vmin=100, vmax=2000,
             figsize=(8, 8), r=None, c=None, filename=None):
    """
    Plot dark current, relevant boxes, imposters and spoilers.
    """
    drc = int(np.max(CHAR.box_sizes) / 5 * 2) + 5

    # Make an image of `dark` centered on acq row, col.  Use ACAImage
    # arithmetic to handle the corner case where row/col are near edge
    # and a square image goes off the edge.
    if r is None:
        r = int(np.round(acq['row']))
        c = int(np.round(acq['col']))
    img = ACAImage(np.zeros(shape=(drc * 2, drc * 2)), row0=r - drc, col0=c - drc)
    dark = ACAImage(dark, row0=-512, col0=-512)
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
                rads_dict[star['id']] = rad

    # Spoiler stars
    for idx, sp in enumerate(acq['spoilers']):
        r = sp['row'] - img.row0
        c = sp['col'] - img.col0
        rad = rads_dict[sp['id']] / np.sqrt(2)
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
