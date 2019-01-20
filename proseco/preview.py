from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Quaternion import Quat
from jinja2 import Template
from chandra_aca.star_probs import guide_count
from chandra_aca.transform import yagzag_to_pixels
from astropy.table import Column

from proseco.catalog import ACATable
from proseco.core import StarsTable
import proseco.characteristics as CHAR

CACHE = {}


def get_acas(rootname):
    filename = rootname + '.pkl'
    acas = pickle.load(open(filename, 'rb'))
    return acas


def make_starcat_plot(self, rootdir='jan2819'):
    rootdir = Path(rootdir)
    rootdir.mkdir(exist_ok=True, parents=True)
    plotname = f'cat{self.obsid}.png'
    outfile = rootdir / plotname
    self.context['catalog_plot'] = plotname

    if outfile.exists():
        return

    stars = StarsTable.from_agasc(self.att, date=self.date)
    self.stars = stars

    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(1, 1, 1)
    self.acqs.stars
    self.plot(ax=ax)
    plt.tight_layout()
    fig.savefig(str(outfile))
    plt.close(fig)

ACATable.make_starcat_plot = make_starcat_plot


def get_text_pre(self):
    """Get pre-formatted text for report."""

    P2 = -np.log10(self.acqs.calc_p_safe())
    att = Quat(self.att)
    self._base_repr_()
    catalog = '\n'.join(self.pformat(max_width=-1))
    n_acq = np.sum(self.acqs['p_acq'])
    n_guide = guide_count(self.guides['mag'], self.guides.t_ccd)

    message_text = self.get_formatted_messages()

    text_pre = f"""\
{self.detector} SIM-Z offset: {self.sim_offset}
RA, Dec, Roll (deg): {att.ra:.6f} {att.dec:.5f} {att.roll:.5f}
Dither acq: Y_amp= {self.dither_acq.y:.1f}  Z_amp={self.dither_acq.z:.1f}
Dither gui: Y_amp= {self.dither_guide.y:.1f}  Z_amp={self.dither_guide.z:.1f}
Maneuver Angle: {self.man_angle:.2f}
Date: {self.date}

{catalog}

{message_text}\
Probability of acquiring 2 or fewer stars (10^-x): {P2:.2f}
Acquisition Stars Expected: {n_acq:.2f}
Guide Stars count: {n_guide:.2f}
Predicted Guide CCD temperature (max): {self.t_ccd_guide:.1f}
Predicted Acq CCD temperature (init) : {self.t_ccd_acq:.1f}"""

    return text_pre

ACATable.get_text_pre = get_text_pre


def stylize(text, category):
    """Stylize ``text``.

    Currently ``category`` of critical, warning, caution, or info are supported
    in the CSS span style.

    """
    out = f'<span class="{category}">{text}</span>'
    return out


def get_formatted_messages(self):
    """Format message dicts into pre-formatted lines for the preview report"""

    lines = []
    for message in self.messages:
        category = message['category']
        idx_str = f"[{message['idx']}] " if ('idx' in message) else ''
        line = f">> {category.upper()}: {idx_str}{message['text']}"
        line = stylize(line, category)
        lines.append(line)

    out = '\n'.join(lines) + '\n\n' if lines else ''
    return out

ACATable.get_formatted_messages = get_formatted_messages


def add_row_col(self):
    """Add row and col columns if not present"""
    if 'row' in self.colnames:
        return

    row, col = yagzag_to_pixels(self['yang'], self['zang'], allow_bad=True)
    index = self.colnames.index('zang') + 1
    self.add_column(Column(row, name='row'), index=index)
    self.add_column(Column(col, name='col'), index=index + 1)

ACATable.add_row_col = add_row_col


def preview(self):
    """Monkey patch method for catalog pre-review from proseco pickle"""
    self.make_starcat_plot()
    self.add_row_col()
    self.check_catalog()

    self.context['text_pre'] = self.get_text_pre()

ACATable.preview = preview


def preview_load(rootname='jan2819'):
    if rootname in CACHE:
        acas = CACHE[rootname]
    else:
        acas = get_acas(rootname)
        CACHE[rootname] = acas

    for obsid, aca in acas.items():
        aca.obsid = obsid
        aca.context = {}
        aca.messages = []
        aca.preview()

    context = {}

    # Probably sort by date
    context['acas'] = [aca for aca in acas.values()]

    template_file = 'index_template_preview.html'
    template = Template(open(template_file, 'r').read())
    out_html = template.render(context)

    out_filename = Path(rootname) / 'index.html'
    with open(out_filename, 'w') as fh:
        fh.write(out_html)


def check_catalog(self):
    for entry in self:
        self.check_position_on_ccd(entry)

ACATable.check_catalog = check_catalog


def check_position_on_ccd(self, entry):
    entry_type = entry['type']

    # Shortcuts and translate y/z to yaw/pitch
    dither_acq_y = self.dither_acq.y
    dither_acq_p = self.dither_acq.z
    dither_guide_y = self.dither_guide.y
    dither_guide_p = self.dither_guide.z

    # Set "dither" for FID to be pseudodither of 5.0 to give 1 pix margin
    # Set "track phase" dither for BOT GUI to max guide dither over interval or 20.0 if undefined.
    # TO DO: hand the guide guide dither
    dither_track_y = 5.0 if (entry_type == 'FID') else dither_guide_y
    dither_track_p = 5.0 if (entry_type == 'FID') else dither_guide_p

    row_lim = CHAR.max_ccd_row - CHAR.CCD['window_pad']
    col_lim = CHAR.max_ccd_col - CHAR.CCD['window_pad']

    def sign(axis):
        """Return sign of the corresponding entry value.  Note that np.sign returns 0
        if the value is 0.0, not the right thing here.
        """
        return -1 if (entry[axis] < 0) else 1

    track_lims = {'row': (row_lim - dither_track_y * CHAR.ARC_2_PIX) * sign('row'),
                  'col': (col_lim - dither_track_p * CHAR.ARC_2_PIX) * sign('col')}

    if entry_type in ('GUI', 'BOT', 'FID'):
        for axis in ('row', 'col'):
            track_delta = abs(track_lims[axis]) - abs(entry[axis])
            for delta_lim, category in ((2.5, 'critical'),
                                        (5.0, 'warning')):
                if track_delta < delta_lim:
                    text = (f"Less than {delta_lim} pix edge margin {axis} "
                            f"lim {track_lims[axis]:.1f} "
                            f"val {entry[axis]:.1f} "
                            f"delta {track_delta:.1f}")
                    self.add_message(text, idx=entry['idx'], category=category)
                    break

    # For acq stars, the distance to the row/col padded limits are also confirmed,
    # but code to track which boundary is exceeded (row or column) is not present.
    # Note from above that the pix_row_pad used for row_lim has 7 more pixels of padding
    # than the pix_col_pad used to determine col_lim.
    # acq_edge_delta = min((row_lim - dither_acq_y / ang_per_pix) - abs(pixel_row),
    #                          (col_lim - dither_acq_p / ang_per_pix) - abs(pixel_col))
    # if ((entry_type =~ /BOT|ACQ/) and (acq_edge_delta < (-1 * 12))){
    #     push @orange_warn, sprintf "alarm [%2d] Acq Off (padded) CCD by > 60 arcsec.\n",i
    # }
    # elsif ((entry_type =~ /BOT|ACQ/) and (acq_edge_delta < 0)){
    #     push @{self->{fyi}}, sprintf "alarm [%2d] Acq Off (padded) CCD (P_ACQ should be < .5)\n",i
    # }

ACATable.check_position_on_ccd = check_position_on_ccd


def add_message(self, text, category, **kwargs):
    message = {'text': text, 'category': category}
    message.update(kwargs)
    self.messages.append(message)

ACATable.add_message = add_message
