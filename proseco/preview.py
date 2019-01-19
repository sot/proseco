from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Quaternion import Quat
from jinja2 import Template
from chandra_aca.star_probs import guide_count

from proseco.catalog import ACATable
from proseco.core import StarsTable

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


def preview(self):
    """Monkey patch method for catalog pre-review from proseco pickle"""
    self.make_starcat_plot()

    P2 = -np.log10(self.acqs.calc_p_safe())
    att = Quat(self.att)
    self._base_repr_()
    catalog = '\n'.join(self.pformat(max_width=-1))
    n_acq = np.sum(self.acqs['p_acq'])
    n_guide = guide_count(self.guides['mag'], self.guides.t_ccd)

    text_pre = f"""\
{self.detector} SIM-Z offset: {self.sim_offset}
RA, Dec, Roll (deg): {att.ra:.6f} {att.dec:.5f} {att.roll:.5f}
Dither acq: Y_amp= {self.dither_acq.y:.1f}  Z_amp={self.dither_acq.z:.1f}
Dither gui: Y_amp= {self.dither_guide.y:.1f}  Z_amp={self.dither_guide.z:.1f}
Maneuver Angle: {self.man_angle:.2f}
Date: {self.date}

{catalog}

Probability of acquiring 2 or fewer stars (10^-x): {P2:.2f}
Acquisition Stars Expected: {n_acq:.2f}
Guide Stars count: {n_guide:.2f}
Predicted Guide CCD temperature (max): {self.t_ccd_guide:.1f}
Predicted Acq CCD temperature (init) : {self.t_ccd_acq:.1f}"""

    self.context['text_pre'] = text_pre
    self.context['messages'] = []


def preview_load(rootname='jan2819'):
    if rootname in CACHE:
        acas = CACHE[rootname]
    else:
        acas = get_acas(rootname)
        CACHE[rootname] = acas

    for obsid, aca in acas.items():
        aca.obsid = obsid
        aca.context = {}
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


ACATable.preview = preview
ACATable.make_starcat_plot = make_starcat_plot
