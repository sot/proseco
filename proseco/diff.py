# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import difflib

import numpy as np
from astropy.table import vstack

"""Output diff of catalog or catalogs
"""


def get_lines(cat, names=None, label=None, section_lines=True):
    """Get list of lines representing catalog suitable for diffing.

    This function sorts the catalog into fids, guides, monitors,and acquisition
    stars and outputs the table into a separate list of lines. In addition:

    - Translate HRC fid ids 7-10 and 11-14 to 1-4. Values in the range 7-14
      come from starcheck, but proseco uses 1-4.
    - Change the MON halfwidth to 25. Starcheck uses 20 to match what effective
      happens on the ACA, but in reality this parameter is fictional.
    - For guides, set dim=1, res=1, and halfw=25. The guide search parameters
      are hardwired in the OBC so the catalog values do not matter.
    - Optionally add a banner at the top with a label identifying the catalog,
      e.g. with the obsid. This banner is useful for diffing all the catalogs
      from a load, and helps to ensure that the diff algorithm stays on track.
    - Guide stars that are BOT are labeled as GU*, and conversely acq stars
      that are BOT are labeled as AC*. This gives some visibility into BOT stars
      while keeping the diffs clean.

    :param cat: Table, star catalog
    :param names: str, list of column names in the output lines for diffing
    :param label: str, label for catalog used in banner at top of lines
    :param section_lines: bool, add separator lines between types (default=True)
    :returns: list
    """
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

    # Make MON catalog and convert slot in dim (designated track star).
    ok = np.in1d(cat['type'], ['MON'])
    mons = cat[ok]
    mons['halfw'] = 25

    ok = np.in1d(cat['type'], ['ACQ', 'BOT'])
    acqs = cat[ok]
    acqs.sort('id')
    acqs['type'][acqs['type'] == 'BOT'] = 'AC*'

    out = vstack([fids, guides, mons, acqs], metadata_conflicts='silent')
    out = out[names]

    # Text representation of table with separator lines between GUI, MON, and
    # ACQ sections.
    table_lines = out.pformat_all()

    # Optionally divide the fid, guide, mon, and acq sections
    if section_lines:
        # Look for diffs in the first two characters of FID, GUI, GU*, MON, ACQ, AC*
        types = np.array(out['type'], dtype='U2')
        idxs = np.flatnonzero(types[:-1] != types[1:])
        for idx in reversed(idxs):
            table_lines.insert(idx + 3, table_lines[1])

    # Finally make a banner and add the table lines
    lines = []
    if label is not None:
        table_width = max(len(line) for line in table_lines)
        sep = '=' * max(table_width, len(label))
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


def diff_html(cats1, cats2, out_file=None, names=None, labels=None, section_lines=True):
    """
    Return the diff of ACA catalogs ``cats1`` and ``cats2`` as HTML.

    If ``out_file`` is supplied the output is written to that file name, otherwise
    the output is returned in a ``DiffHTML`` object that will display the
    diff in Jupyter notebook, or return the HTML via the ``html`` attribute.

    :param cat1: Table or list of Table, first ACA catalog(s)
    :param cat2: Table or list of Table, second ACA catalog(s)
    :param out_file: str, output file name or Path
    :param names: list of column names (either as list or space-delimited str).
        Default = 'idx id slot type sz dim res halfw'
    :param label: str, None, label for catalog used in banner at top of lines
    :param section_lines: bool, add separator lines between types (default=True)
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
