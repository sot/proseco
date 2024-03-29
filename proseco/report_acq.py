# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import copy, deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Column, Table
from chandra_aca import plot as plot_aca
from chandra_aca.aca_image import ACAImage
from jinja2 import Template
from matplotlib import patches
from matplotlib.ticker import FixedLocator

from . import characteristics_acq as ACQ
from .acq import AcqTable
from .core import table_to_html

FILEDIR = Path(__file__).parent
ACQ_COLS = [
    "idx",
    "slot",
    "id",
    "yang",
    "zang",
    "row",
    "col",
    "mag",
    "mag_err",
    "color",
    "halfw",
    "p_acq",
]


def get_p_acqs_table(acqs, acq, p_name):
    """
    Make HTML tables for an acq star for the following:

    - ``p_brightest``: probability this star is the brightest in box (function
        of ``box_size`` and ``man_err``)
    - ``p_acqs``: product of all probability bits
    """
    man_errs = ACQ.p_man_errs["man_err_hi"]
    box_sizes = sorted(ACQ.box_sizes)
    names = [r"box \ man_err"] + [f'{man_err}"' for man_err in man_errs]
    cols = {}
    cols[r"box \ man_err"] = [f'{box_size}"' for box_size in box_sizes]
    for man_err in man_errs:
        name = f'{man_err}"'
        cols[name] = [
            round(getattr(acq["probs"], p_name)(box_size, man_err, acqs), 3)
            for box_size in box_sizes
        ]

    return table_to_html(Table(cols, names=names))


def get_p_on_ccd_table(acq):
    """
    Make HTML tables for an acq star for the following:

    - ``p_on_ccd``: probability star is on the usable part of the CCD (function
        of ``man_err`` and ``dither``)
    """
    man_errs = ACQ.p_man_errs["man_err_hi"]
    names = ["man_err"] + [f'{man_err}"' for man_err in man_errs]
    cols = {}
    cols["man_err"] = [""]
    for man_err in man_errs:
        name = f'{man_err}"'
        cols[name] = [round(acq["probs"].p_on_ccd(man_err), 3)]

    return table_to_html(Table(cols, names=names))


def get_p_acq_model_table(acq):
    """
    Make HTML tables for an acq star for the following:

    - ``p_acq_model``: probability of acquisition from the chandra_aca model
        (function of ``box_size``)
    """
    box_sizes = sorted(ACQ.box_sizes)
    names = ["box size"] + [f'{box_size}"' for box_size in box_sizes]
    cols = {}
    cols["box size"] = [""]
    for box_size in box_sizes:
        name = f'{box_size}"'
        cols[name] = [round(acq["probs"].p_acq_model(box_size), 3)]

    return table_to_html(Table(cols, names=names))


def select_events(events, funcs, **select):
    outs = [
        event
        for event in events
        if event["func"] in funcs
        and all(event.get(key) == val for key, val in select.items())
    ]
    return outs


def make_events(acqs):
    # Set up global events
    events = deepcopy(acqs.log_info["events"])
    for event in events:
        event["func_disp"] = event["func"]
    last = events[0]
    for event in events[1:]:
        if event["dt"] == last["dt"] and event["func"] == last["func"]:
            event["dt"] = ""
            event["func_disp"] = ""
        else:
            last = event
        event["data"] = "&nbsp;" * event.get("level", 0) * 4 + event["data"]

    return events


def make_p_man_errs_report(context):
    tbl = ACQ.p_man_errs.copy()
    man_err = [
        f'<b>{lo}-{hi}"</b>' for lo, hi in zip(tbl["man_err_lo"], tbl["man_err_hi"])
    ]
    del tbl["man_err_lo"]
    del tbl["man_err_hi"]

    for name in tbl.colnames:
        tbl.rename_column(name, name + "°")

    for col in tbl.columns.values():
        col[:] = np.round(col, 4)

    tbl.add_column(Column(man_err, name=r"err \ angle"), 0)

    context["p_man_errs_table"] = table_to_html(tbl)


def make_cand_acqs_report(acqs, cand_acqs, events, context, obsdir):
    ######################################################
    # Candidate acquisition stars and initial catalog
    ######################################################
    # Start with table
    cand_acqs_table = cand_acqs[ACQ_COLS]
    # Probably won't work in astropy 1.0
    cand_acqs_table["id"] = [
        "<a href=#{0}>{0}</a>".format(cand_acq["id"]) for cand_acq in cand_acqs
    ]
    context["cand_acqs_table"] = table_to_html(cand_acqs_table)

    context["cand_acqs_events"] = select_events(
        events, ("get_acq_catalog", "from_agasc", "get_acq_candidates", "get_box_sizes")
    )

    # Now plot figure
    filename = obsdir / "candidate_stars.png"
    if not filename.exists():
        # Pull a fast-one and mark the final selected ACQ stars as BOT so they
        # get a circle in the plot.  This might be confusing and need fixing
        # later, but for now it is an easy way to show the winning candidates.
        for acq in acqs.cand_acqs:
            if acq["id"] in acqs["id"]:
                acq["type"] = "BOT"

        fig = plot_aca.plot_stars(
            acqs.att,
            stars=acqs.stars,
            catalog=acqs.cand_acqs,
            bad_stars=acqs.bad_stars_mask,
        )
        # When Ska3 has matplotlib 2.2+ then just use `filename`
        fig.savefig(str(filename))
        plt.close(fig)

        # Restore original type designation
        acqs.cand_acqs["type"] = "ACQ"


def make_initial_cat_report(events, context):
    context["initial_cat_events"] = select_events(
        events, ("get_initial_catalog", "select_best_p_acqs")
    )


def make_acq_star_details_report(acqs, cand_acqs, events, context, obsdir):
    ######################################################
    # Candidate acq star detail sections
    ######################################################

    context["cand_acqs"] = []

    for ii, acq in enumerate(cand_acqs):
        # Local context dict for each cand_acq star
        cca = {
            "id": acq["id"],
            "selected": "SELECTED" if acq["id"] in acqs["id"] else "not selected",
        }

        # Events related to this ACQ ID
        cca["initial_selection_events"] = select_events(
            events, "select_best_p_acqs", id=acq["id"]
        )
        cca["optimize_events"] = select_events(
            events, ("optimize_catalog", "optimize_acq_halfw"), id=acq["id"]
        )
        # Make a dict copy of everything in ``acq``
        acq_table = cand_acqs[ACQ_COLS][ii : ii + 1].copy()
        acq_table["id"] = [
            '<a href="http://kadi.cfa.harvard.edu/star_hist/?agasc_id={0}" '
            'target="_blank">{0}</a>'.format(aq["id"])
            for aq in acq_table
        ]
        cca["acq_table"] = table_to_html(acq_table)

        cca["p_brightest_table"] = get_p_acqs_table(acqs, acq, "p_brightest")
        cca["p_acqs_table"] = get_p_acqs_table(acqs, acq, "p_acqs")
        cca["p_acq_model_table"] = get_p_acq_model_table(acq)
        cca["p_on_ccd_table"] = get_p_on_ccd_table(acq)

        # Make the star detail plot
        basename = f'spoilers_{acq["id"]}.png'
        filename = obsdir / basename
        cca["spoilers_plot"] = basename
        if not filename.exists():
            plot_spoilers(acq, acqs, filename=filename)

        # Make the acq detail plot with spoilers and imposters
        basename = f'imposters_{acq["id"]}.png'
        filename = obsdir / basename
        cca["imposters_plot"] = basename
        if not filename.exists():
            plot_imposters(acq, acqs.dark, acqs.dither, filename=filename)

        if len(acq["imposters"]) > 0:
            if not isinstance(acq["imposters"], Table):
                acq["imposters"] = Table(acq["imposters"])

            names = ("row0", "col0", "d_row", "d_col", "img_sum", "mag", "mag_err")
            fmts = ("d", "d", ".0f", ".0f", ".0f", ".2f", ".2f")
            imposters = acq["imposters"][names]
            for name, fmt in zip(names, fmts):
                imposters[name].info.format = fmt

            idx = Column(np.arange(len(acq["imposters"])), name="idx")
            imposters.add_column(idx, index=0)

            cca["imposters_table"] = table_to_html(imposters)
        else:
            cca["imposters_table"] = ""

        if len(acq["spoilers"]) > 0:
            if not isinstance(acq["spoilers"], Table):
                acq["spoilers"] = Table(acq["spoilers"])

            names = ("id", "yang", "zang", "mag", "mag_err")
            fmts = ("d", ".1f", ".1f", ".2f", ".2f")
            spoilers = acq["spoilers"][names]
            for name, fmt in zip(names, fmts):
                spoilers[name].info.format = fmt

            idx = Column(np.arange(len(spoilers)), name="idx")
            d_yang = Column(spoilers["yang"] - acq["yang"], name="d_yang", format=".1f")
            d_zang = Column(spoilers["zang"] - acq["zang"], name="d_zang", format=".1f")
            spoilers.add_column(d_zang, index=3)
            spoilers.add_column(d_yang, index=3)
            spoilers.add_column(idx, index=0)

            d_mags = spoilers["mag"] - acq["mag"]
            d_mag_errs = np.sqrt(spoilers["mag_err"] ** 2 + acq["mag_err"] ** 2)
            d_sigmas = d_mags / d_mag_errs
            spoilers["d_sigma"] = d_sigmas
            spoilers["d_sigma"].info.format = ".1f"

            cca["spoilers_table"] = table_to_html(spoilers)
        else:
            cca["spoilers_table"] = ""

        context["cand_acqs"].append(cca)


def make_optimize_catalog_report(events, context):
    context["optimize_events"] = select_events(
        events, ("calc_p_safe", "optimize_catalog", "optimize_acq_halfw")
    )


def make_obsid_summary(acqs, events, context, obsdir):  # noqa: ARG001
    acqs_table = acqs[ACQ_COLS]
    acqs_table["id"] = [
        '<a href="#{0}">{0}</a>'.format(acq["id"]) for acq in acqs_table
    ]
    context["acqs_table"] = table_to_html(acqs_table)

    basename = "acq_stars.png"
    filename = obsdir / "acq" / basename
    context["acq_stars_plot"] = basename
    if not filename.exists():
        fig = plt.figure(figsize=(4, 4))
        fig.subplots_adjust(top=0.95)
        ax = fig.add_subplot(1, 1, 1)
        plot_aca.plot_stars(
            acqs.att,
            stars=acqs.stars,
            catalog=acqs,
            bad_stars=acqs.bad_stars_mask,
            ax=ax,
        )
        # When Ska3 has matplotlib 2.2+ then just use `filename`
        plt.savefig(str(filename))
        plt.close(fig)


def make_report(obsid, rootdir="."):
    """
    Make summary HTML report for acq star selection.

    The first arg ``obsid`` can be an obsid (int) or a ``AcqTable`` object.
    For ``int`` the acq table is read from ``<rootdir>/obs<obsid>/acq.pkl``.

    Output is in ``<rootdir>/obs<obsid>/acq/index.html`` plus related images
    in that directory.

    :param obsid: int obsid or AcqTable instance
    :param rootdir: root directory for outputs

    :returns: AcqTable object (mostly for testing)
    """
    rootdir = Path(rootdir)
    if isinstance(obsid, AcqTable):
        acqs = obsid
        obsid = acqs.obsid
    else:
        acqs = AcqTable.from_pickle(obsid, rootdir)
    cand_acqs = acqs.cand_acqs

    # Define and make directories as needed
    obsdir = rootdir / f"obs{obsid:05}"
    outdir = obsdir / "acq"
    outdir.mkdir(exist_ok=True, parents=True)

    context = copy(acqs.meta)
    context["include_ids"] = ", ".join([str(val) for val in acqs.include_ids])
    context["include_halfws"] = ", ".join([str(val) for val in acqs.include_halfws])
    context["exclude_ids"] = ", ".join([str(val) for val in acqs.exclude_ids])

    # Get stars if not already set (e.g. if acqs is coming from a pickle).  If
    # acqs.stars is already correctly defined this does nothing.
    acqs.set_stars()

    events = make_events(acqs)
    context["events"] = events

    make_obsid_summary(acqs, events, context, obsdir)
    make_p_man_errs_report(context)
    make_cand_acqs_report(acqs, cand_acqs, events, context, outdir)
    make_initial_cat_report(events, context)
    make_acq_star_details_report(acqs, cand_acqs, events, context, outdir)
    make_optimize_catalog_report(events, context)

    template_file = FILEDIR / ACQ.index_template_file
    template = Template(open(template_file, "r").read())
    out_html = template.render(context)
    out_filename = outdir / "index.html"
    with open(out_filename, "w") as fh:
        fh.write(out_html)

    return acqs


def local_symsize(mag):
    # map mags to figsizes, defining
    # mag 6 as 40 and mag 11 as 3
    # interp should leave it at the bounding value outside
    # the range
    return np.interp(mag, [6.0, 11.0], [200.0, 20.0])


def plot_spoilers(acq, acqs, filename=None):
    # Clip figure to region around acq star
    plot_hw = 360  # arcsec
    hwp = plot_hw / 5  # Halfwidth of plot in pixels

    # Get stars
    stars = acqs.stars
    ok = (np.abs(stars["yang"] - acq["yang"]) < plot_hw) & (
        np.abs(stars["zang"] - acq["zang"]) < plot_hw
    )
    stars = stars[ok]
    bad_stars = acqs.bad_stars_mask[ok]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(acq["row"] - hwp, acq["row"] + hwp)
    ax.set_ylim(acq["col"] - hwp, acq["col"] + hwp)

    # Box regions
    for box_size in np.arange(120, 321, 40):
        hwp = box_size / 5  # half width in pixels
        r0 = acq["row"] - hwp
        c0 = acq["col"] - hwp
        patch = patches.Rectangle(
            (r0, c0), hwp * 2, hwp * 2, edgecolor="g", facecolor="none", lw=1, alpha=0.3
        )
        ax.add_patch(patch)
        plt.text(r0 + 1, c0 + 1, f'{box_size}"', fontsize="small", color="g")

    # Plot search boxes for all acq stars (most will get clipped later)
    for acq0 in acqs:
        hwp = acq0["halfw"] / 5
        r0 = acq0["row"] - hwp
        c0 = acq0["col"] - hwp
        patch = patches.Rectangle(
            (r0, c0), hwp * 2, hwp * 2, edgecolor="b", facecolor="none", lw=2, alpha=0.5
        )
        ax.add_patch(patch)

    # Plot the CCD box and readout register indicator
    b1hw = 512
    box1 = plt.Rectangle((b1hw, -b1hw), -2 * b1hw, 2 * b1hw, fill=False)
    ax.add_patch(box1)
    b2w = 520
    box2 = plt.Rectangle((b2w, -b1hw), -4 + -2 * b2w, 2 * b1hw, fill=False)
    ax.add_patch(box2)

    # Monkey patch the symsize function to make the stars bigger
    orig_symsize = plot_aca.symsize
    plot_aca.symsize = local_symsize
    plot_aca._plot_field_stars(ax, stars=stars, attitude=acqs.att, bad_stars=bad_stars)
    for star in stars:
        plt.text(
            star["row"],
            star["col"] - 3,
            f'{star["mag"]:.1f}±{star["mag_err"]:.1f}',
            verticalalignment="top",
            horizontalalignment="center",
            fontsize="small",
        )
    plot_aca.symsize = orig_symsize

    # Plot spoiler star indices
    for idx, spoiler in enumerate(acq["spoilers"]):
        plt.text(spoiler["row"] + 3, spoiler["col"] + 3, str(idx))

    ax.set_xlabel("Row")
    ax.set_ylabel("Col")
    ax.set_title("Green boxes show search box + man err")

    plt.tight_layout()
    if filename is not None:
        # When Ska3 has matplotlib 2.2+ then just use `filename`
        plt.savefig(str(filename), pad_inches=0.0)
    plt.close(fig)


def plot_imposters(
    acq,
    dark,
    dither,
    vmin=100,
    vmax=2000,
    figsize=(5, 5),
    r=None,
    c=None,
    filename=None,
):
    """
    Plot dark current, relevant boxes, imposters and spoilers.
    """
    drc = int(np.max(ACQ.box_sizes) / 5 + 5 + dither.max() / 5)

    # Make an image of `dark` centered on acq row, col.  Use ACAImage
    # arithmetic to handle the corner case where row/col are near edge
    # and a square image goes off the edge.
    if r is None:
        r = int(np.round(acq["row"]))
        c = int(np.round(acq["col"]))
    img = ACAImage(np.zeros(shape=(drc * 2, drc * 2)), row0=r - drc, col0=c - drc)
    dark = ACAImage(dark, row0=-512, col0=-512)
    img += dark.aca

    # Show the dark current image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(
        img.transpose(),
        interpolation="none",
        cmap="hot",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    # CCD edge
    r = -512.5 - img.row0
    c = -512.5 - img.col0
    patch = patches.Rectangle(
        (r, c), 1025, 1025, edgecolor="g", facecolor="none", lw=2.5
    )
    ax.add_patch(patch)

    # Imposter stars
    for idx, imp in enumerate(acq["imposters"]):
        r = imp["row0"] - img.row0
        c = imp["col0"] - img.col0
        patch = patches.Rectangle(
            (r + 0.5, c + 0.5), 6, 6, edgecolor="y", facecolor="none", lw=1.5
        )
        ax.add_patch(patch)
        plt.text(r + 7, c + 7, str(idx), color="y", fontsize="large", fontweight="bold")

    # Box regions
    rc = img.shape[0] // 2 + 0.5
    cc = img.shape[1] // 2 + 0.5
    for hw in ACQ.p_man_errs["man_err_hi"]:
        hwpr = hw / 5 + dither.row
        hwpc = hw / 5 + dither.col
        patch = patches.Rectangle(
            (rc - hwpr, cc - hwpc),
            hwpr * 2,
            hwpc * 2,
            edgecolor="r",
            facecolor="none",
            lw=1,
            alpha=1,
        )
        ax.add_patch(patch)
        plt.text(rc - hwpr + 1, cc - hwpc + 1, f'{hw}"', color="y", fontweight="bold")

    # Hack to fix up ticks to have proper row/col coords.  There must be a
    # correct way to do this.
    xticks_loc = ax.get_xticks().tolist()
    xticks = [str(int(label) + img.row0) for label in xticks_loc]
    ax.xaxis.set_major_locator(FixedLocator(xticks_loc))
    ax.set_xticklabels(xticks)
    ax.set_xlabel("Row")
    yticks_loc = ax.get_yticks().tolist()
    yticks = [str(int(label) + img.col0) for label in yticks_loc]
    ax.yaxis.set_major_locator(FixedLocator(yticks_loc))
    ax.set_yticklabels(yticks)
    ax.set_ylabel("Column")
    ax.set_title("Red boxes show search box size + dither")

    plt.tight_layout()
    if filename is not None:
        # When Ska3 has matplotlib 2.2+ then just use `filename`
        plt.savefig(str(filename), pad_inches=0.0)
    plt.close(fig)

    return img, ax
