# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import difflib

import numpy as np
from astropy.table import vstack

"""Output diff of catalog or catalogs
"""


def get_lines(cat, names=None, label=None):
    if names is None:
        names = 'idx id slot type sz dim res halfw'
    if isinstance(names, str):
        names = names.split()

    ok = np.in1d(cat['type'], ['FID'])
    fids = cat[ok]
    fids.sort('id')
    fids['dim'] = 1
    fids['res'] = 1
    fids['halfw'] = 25
    # Translate fid IDs in the range 11-14 => 1-4, 7-10 => 1-4. The fully
    # independent fid ID comes from starcheck typically.
    ok = fids['id'] > 10
    fids['id'][ok] -= 10
    ok = fids['id'] > 6
    fids['id'][ok] -= 6

    ok = np.in1d(cat['type'], ['GUI', 'BOT'])
    guides = cat[ok]
    guides.sort('id')
    guides['dim'] = 1
    guides['res'] = 1
    guides['halfw'] = 25
    guides['type'][guides['type'] == 'BOT'] = 'GU*'

    # Make MON catalog and convert slot in dim (designated track star) to an ID
    # so comparison is independent of slot differences.
    ok = np.in1d(cat['type'], ['MON'])
    mons = cat[ok]
    mons['halfw'] = 20

    ok = np.in1d(cat['type'], ['ACQ', 'BOT'])
    acqs = cat[ok]
    acqs['type'] = acqs['type'].astype('U4')
    acqs.sort('id')
    acqs['type'][acqs['type'] == 'BOT'] = 'AC*'

    out = vstack([fids, guides, mons, acqs], metadata_conflicts='silent')
    out = out[names]

    # Text representation of table with separator lines between GUI, MON, and
    # ACQ sections.
    table_lines = out.pformat_all()
    # Look for diffs in the first two characters of FID, GUI, GU*, MON, ACQ, AC*
    types = np.array(out['type'], dtype='U2')
    idxs = np.flatnonzero(types[:-1] != types[1:])
    for idx in reversed(idxs):
        table_lines.insert(idx + 3, table_lines[1])

    lines = []
    if label is not None:
        sep = '=' * max(40, len(label))
        lines.append(sep)
        lines.append(label)
        lines.append(sep)
        lines.append('')

    lines.extend(table_lines)

    return lines


class DiffHTML:
    def __init__(self, html):
        self.html = html

    def _repr_html_(self):
        return self.html


def diff_html(cats1, cats2, out_file=None, labels=None, names=None):
    """
    Calculate the diff of ACA catalogs ``cats1`` and ``cats2`` as HTML.

    If ``out_file`` is supplied the output is written to that file name, otherwise
    the output is returned in a ``DiffHTML`` object that will display the
    diff in Jupyter notebook, or return the HTML via the ``html`` attribute.

    :param cat1: Table or list of Table, first ACA catalog(s)
    :param cat2: Table or list of Table, second ACA catalog(s)
    :param out_file: str, output file name or Path
    :param names: list of column names (either as list or space-delimited str).
        Default = 'idx id slot type sz dim res halfw'
    :returns: None or DiffHTML object
    """
    if not isinstance(cats1, list):
        cats1 = [cats1]

    if not isinstance(cats2, list):
        cats2 = [cats2]

    lines1 = []
    lines2 = []
    if labels is None:
        labels = [f'Catalog {ii + 1}' for ii in range(len(cats1))]

    for cats, lines in ((cats1, lines1),
                        (cats2, lines2)):
        for cat, label in zip(cats, labels):
            lines.extend(get_lines(cat, names, label))
            lines.append('')

    differ = difflib.HtmlDiff()
    diff_html = differ.make_file(lines1, lines2)

    if out_file is not None:
        with open(out_file, 'w') as fh:
            fh.write(diff_html)
    else:
        return DiffHTML(diff_html)
