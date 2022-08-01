# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import difflib

import numpy as np
from astropy.table import vstack

"""Output diff of catalog or catalogs
"""

# The header and footer are copied from the output of difflib.HtmlDiff.make_file.
# Note that Jupyter notebook has a default td padding of 0.5em which looks
# terrible, so override that for table diff cells (the "table.diff td" style
# is added to the make_file() output).

HTML_HEADER = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html>
<head>
    <meta http-equiv="Content-Type"
          content="text/html; charset=utf-8" />
    <title></title>
    <style type="text/css">
        table.diff {font-family:Courier; border:medium;}
        table.diff td {padding-top: 0em; padding-bottom: 0em;
                       padding-right: 0.5em; padding-left: 0.5em}
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
    </style>
</head>
<body>
"""
HTML_FOOTER = """
        <table class="diff" summary="Legends">
        <tr> <th colspan="2"> Legends </th> </tr>
        <tr> <td> <table border="" summary="Colors">
                      <tr><th> Colors </th> </tr>
                      <tr><td class="diff_add">&nbsp;Added&nbsp;</td></tr>
                      <tr><td class="diff_chg">Changed</td> </tr>
                      <tr><td class="diff_sub">Deleted</td> </tr>
                  </table></td>
             <td> <table border="" summary="Links">
                      <tr><th colspan="2"> Links </th> </tr>
                      <tr><td>(f)irst change</td> </tr>
                      <tr><td>(n)ext change</td> </tr>
                      <tr><td>(t)op</td> </tr>
                  </table></td> </tr>
    </table>
</body>

</html>
"""

__all__ = ['get_catalog_lines', 'catalog_diff', 'CatalogDiff']


def get_catalog_lines(
    cat, names=None, section_lines=True, sort_name='id', formats=None
):
    """Get list of lines representing catalog suitable for diffing.

    This function sorts the catalog into fids, guides, monitors,and acquisition
    stars and outputs the table into a separate list of lines. In addition:

    - Translate HRC fid ids 7-10 and 11-14 to 1-4. Values in the range 7-14
      come from starcheck, but proseco uses 1-4.
    - Optionally add a banner at the top with a label identifying the catalog,
      e.g. with the obsid. This banner is useful for diffing all the catalogs
      from a load, and helps to ensure that the diff algorithm stays on track.
    - Guide stars that are BOT are labeled as GU*, and conversely acq stars
      that are BOT are labeled as AC*. This gives some visibility into BOT stars
      while keeping the diffs clean.

    :param cat: Table
        Star catalog
    :param names: str, list
        Column names in the output lines for diffing
    :param section_lines: bool
        Add separator lines between types (default=True)
    :param formats: dict, None
        Dict of column names and format specifiers
    :returns: list
    """
    if names is None:
        names = 'idx id slot type mag sz dim res halfw'
    if isinstance(names, str):
        names = names.split()

    ok = np.in1d(cat['type'], ['FID'])
    fids = cat[ok]
    fids.sort(sort_name)
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
    guides.sort(sort_name)
    bot = guides['type'] == 'BOT'
    guides['dim'][bot] = 1
    guides['res'][bot] = 1
    guides['halfw'][bot] = 25
    guides['type'][bot] = 'GU*'

    # Make MON catalog and convert slot in dim (designated track star).
    ok = np.in1d(cat['type'], ['MON'])
    mons = cat[ok]
    mons.sort(sort_name)

    ok = np.in1d(cat['type'], ['ACQ', 'BOT'])
    acqs = cat[ok]
    acqs.sort(sort_name)
    acqs['type'][acqs['type'] == 'BOT'] = 'AC*'

    out = vstack([fids, guides, mons, acqs], metadata_conflicts='silent')
    from .core import ACACatalogTable

    out = ACACatalogTable(out[names])

    # Handle odd case of catalog from starcheck with None in field
    for name in out.colnames:
        col = out[name]
        if col.dtype.kind == 'O':
            col[col == None] = 0  # noqa
            out[name] = col.tolist()

    # Allow specifying custom formats. For instance starcheck uses 3 digits for
    # mag. Rounding to the default 2 from proseco causes spuriuous diffs when
    # comparing to starcheck, so go back to 3 for diffs. Need the silly hack of
    # calling repr first to do the format auto-initialize.
    if formats:
        repr(out)
        for name in formats:
            out[name].format = formats[name]

    # Text representation of table with separator lines between GUI, MON, and
    # ACQ sections.
    lines = out.pformat_all()

    # Optionally divide the fid, guide, mon, and acq sections
    if section_lines:
        # Look for diffs in the first two characters of FID, GUI, GU*, MON, ACQ, AC*
        types = np.array(out['type'], dtype='U2')
        idxs = np.flatnonzero(types[:-1] != types[1:])
        for idx in reversed(idxs):
            lines.insert(idx + 3, lines[1])

    return lines


class CatalogDiff:
    """Represent an ACA catalog diff as either HTML or plain text"""

    def __init__(self, text, is_html=False):
        self.text = text
        self.is_html = is_html

    def _repr_html_(self):
        if self.is_html:
            return self.text
        else:
            return f'<pre>{self.text}</pre>'

    def __repr__(self):
        return self.text

    def write(self, out_file):
        """Write output to ``out_file``

        :param out_file: str, Path
            Output file
        """
        with open(out_file, 'w') as fh:
            fh.write(self.text)


def catalog_diff(
    cats1,
    cats2,
    style='html',
    names=None,
    labels=None,
    formats=None,
    sort_name='id',
    section_lines=True,
    n_context=3,
):
    """
    Return the diff of ACA catalogs ``cats1`` and ``cats2``.

    The output is returned in a ``CatalogDiff`` object that will display the
    formatted diff in Jupyter notebook, or return the diff text via the ``text``
    attribute. The difference text can be written to file with the
    ``CatalogDiff.write`` method

    :param cats1: Table, list of Table
        First ACA catalog(s)
    :param cats2: Table, list of Table
        Second ACA catalog(s)
    :param style: str
        Diff style, either 'html', 'content', or 'unified'
    :param names: list, str
        Column names in output as list or space-delimited str.
        Default = 'idx id slot type sz dim res halfw'
    :param labels: list, None
        Label for catalog used in banner at top of lines
    :param formats: dict, None
        Dict of column names and format specifiers e.g. {'mag': '6.3f'}
    :param sort_name: str
        Column name for sorting catalog within sections (default='id')
    :param section_lines: bool
        Add separator lines between types (default=True)
    :param n_context: int
        Number of context lines for unified, context diffs (default=3)
    :returns: CatalogDiff
    """
    if not isinstance(cats1, list):
        cats1 = [cats1]

    if not isinstance(cats2, list):
        cats2 = [cats2]

    if len(cats1) != len(cats2):
        raise ValueError('different number of catalogs in cat1 and cat2')

    header = HTML_HEADER if style == 'html' else ''
    footer = HTML_FOOTER if style == 'html' else ''

    if labels is None:
        ok = len(cats1) > 1
        labels = [(f'Catalog {ii + 1}' if ok else None) for ii in range(len(cats1))]

    text_all = ''
    for cat1, cat2, label in zip(cats1, cats2, labels):
        lines1 = []
        lines2 = []

        for cat, lines in ((cat1, lines1), (cat2, lines2)):
            cat_lines = get_catalog_lines(
                cat,
                names,
                section_lines=section_lines,
                sort_name=sort_name,
                formats=formats,
            )
            lines.extend(cat_lines)

        if style == 'html':
            differ = difflib.HtmlDiff()
            text = differ.make_table(lines1, lines2)
            if label:
                text = f'<h3>{label}</h3>' + os.linesep + text
        elif style in ('context', 'unified'):
            func = getattr(difflib, f'{style}_diff')
            ls = os.linesep
            text = ls.join(
                func(
                    lines1,
                    lines2,
                    fromfile='catalog 1',
                    tofile='catalog 2',
                    n=n_context,
                )
            )
            if label:
                sep = '=' * max(30, len(label))
                text = sep + ls + label + ls + sep + ls + text + ls + ls
        else:
            raise ValueError("style arg must be one of 'html', 'unified', 'context'")

        text_all += text

    return CatalogDiff(header + text_all + footer, is_html=(style == 'html'))
