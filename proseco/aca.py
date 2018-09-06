import numpy as np

from .acq import get_acq_catalog
from .fid import get_fid_catalog


def get_acq_fid_catalogs(obsid=0, att=None,
                         man_angle=None, t_ccd=None, date=None, dither=None,
                         detector=None, sim_offset=None, focus_offset=None, n_fids=None,
                         optimize=True, verbose=False, print_log=False):
    """
    Get a catalog of acquisition and fid lights.

    If ``obsid`` corresponds to an already-scheduled obsid then the parameters
    ``att``, ``man_angle``, ``t_ccd``, ``date``, and ``dither`` will
    be fetched via ``mica.starcheck`` if not explicitly provided here.

    :param obsid: obsid (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param man_angle: maneuver angle (deg)
    :param t_ccd: ACA CCD temperature (degC)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither: dither size (float, arcsec)
    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param focus_offset: SIM focus offset [steps] (default=0)
    :param sim_offset: SIM translation offset from nominal [steps] (default=0)
    :param optimize: optimize star catalog after initial selection (default=True)
    :param verbose: provide extra logging info (mostly calc_p_safe) (default=False)
    :param print_log: print the run log to stdout (default=False)

    :returns: AcqTable of acquisition stars
    """
    acqs = get_acq_catalog(obsid=obsid, att=att,
                           man_angle=man_angle, t_ccd=t_ccd, date=date, dither=dither,
                           detector=detector, sim_offset=sim_offset, focus_offset=focus_offset,
                           optimize=optimize, verbose=verbose, print_log=print_log)

    fids = get_fid_catalog(acqs=acqs, n_fids=n_fids)
    acqs.fids = fids

    # If no fids returned then some co-optimization of acq stars and fid set needs
    # to be done.
    if len(fids) == 0:
        optimize_acq_fid_catalogs(acqs, fids)

    return acqs, fids


def optimize_acq_fid_catalogs(acqs, fids):
    """
    Select optimal acquisition stars and fid lights using the process described in
    https://docs.google.com/presentation/d/1j89Zy2RRB4McV32ue-EpQThaEpiONw_EAilqcOqz45g

    This requires that ``acqs`` and ``fids`` have already been selected independently.
    This function updates each of those in place and returns None.

    :param acqs: AcqTable
    :param fids: FidTable
    """
    pass
