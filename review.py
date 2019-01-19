from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from proseco.core import StarsTable


def get_acas(filename='jan2819.pkl'):
    acas = pickle.load(open(filename, 'rb'))
    return acas


def make_starcat_plot(aca, obsid, rootdir='jan2819'):
    rootdir = Path(rootdir)
    rootdir.mkdir(exist_ok=True, parents=True)
    outfile = rootdir / f'cat{obsid}.png'
    if outfile.exists():
        return

    stars = StarsTable.from_agasc(aca.att, date=aca.date)
    aca.stars = stars
    P2 = -np.log10(aca.acqs.calc_p_safe())

    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    aca.acqs.stars
    aca.plot(ax=ax)
    ax.set_title(f'Obsid {obsid} P2={P2:.2f} thumbs={aca.thumbs_up}')
    fig.savefig(str(outfile))
    plt.close(fig)


def make_starcat_plots(acas, rootdir='jan2819'):
    for obsid, aca in acas.items():
        make_starcat_plot(aca, obsid, rootdir)
