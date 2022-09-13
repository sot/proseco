import ska_helpers

__version__ = ska_helpers.get_version(__package__)


def get_aca_catalog(*args, **kwargs):
    """Get a catalog of guide stars, acquisition stars and fid lights.

    If ``obsid`` is supplied and is a string, then it is taken to be starcheck
    text with required info.  User-supplied kwargs take precedence, however
    (e.g.  one can override the dither from starcheck).

    In this situation if the ``obsid`` text includes the string
    ``--force-catalog`` anywhere then the final proseco guide and acq catalogs
    will be forced to match the input starcheck catalog.  This can be done by
    appending this string, e.g. with ``obs_text + '--force-catalog'`` in the
    call to ``get_aca_catalog``.

    The input ``n_guide`` parameter represents the number of slots available for
    the combination of guide stars and monitor windows (including both fixed and
    tracking monitor windows). In most normal situations, ``n_guide`` is equal
    to ``8 - n_fid``. The ``n_guide`` parameter is confusingly named but this is
    because the actual number of guide stars is not known in advance in the case
    of auto-conversion from a monitor request to a guide star. In actual
    practice, what is normally known is how many slots are available for the
    combination of guide stars and monitor windows, so this makes the call to
    catalog creation simpler.

    NOTE on API:

    Keywords that have ``_acq`` and/or ``_guide`` suffixes are handled with
    the AliasAttribute in core.py.  If one calls get_aca_catalog() with e.g.
    ``t_ccd=-10`` then that will set the CCD temperature for both acq and
    guide selection.  This is not part of the public API but is a private
    feature of the implementation that works for now.

    :param obsid: obsid (int) or starcheck text (str) (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param n_acq: desired number of acquisition stars (default=8)
    :param n_fid: desired number of fid lights (req'd unless obsid spec'd)
    :param n_guide: desired number of guide stars + monitor windows (req'd unless obsid spec'd)
    :param monitors: N x 5 float array specifying monitor windows
    :param man_angle: maneuver angle (deg)
    :param man_angle_next: maneuver angle to next attitude after this observation
                           (deg, default=180)
    :param t_ccd_acq: ACA CCD temperature for acquisition (degC)
    :param t_ccd_guide: ACA CCD temperature for guide (degC)
    :param t_ccd_penalty_limit: ACA CCD penalty limit for planning (degC). If not
        provided this defaults to value from the ACA xija thermal model.
    :param t_ccd_eff_acq: ACA CCD effective temperature for acquisition (degC)
    :param t_ccd_eff_guide: ACA CCD effective temperature for guide (degC)
    :param dark: 1024x1024 dark image (e-/sec, default=None => auto-fetch)
    :param dark_date: Date of dark cal (str, optional)
    :param duration: duration of observation (sec)
    :param target_name: name of target (str)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither_acq: acq dither size (2-element sequence (y, z), arcsec)
    :param dither_guide: guide dither size (2-element sequence (y, z), arcsec)
    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param sim_offset: SIM translation offset from nominal [steps] (default=0)
    :param focus_offset: SIM focus offset [steps] (default=0)
    :param target_offset: (y, z) target offset including dynamical offset
                          (2-element sequence (y, z), deg)
    :param dyn_bgd_n_faint: number of faint stars to apply the dynamic background
        temperature bonus ``dyn_bgd_dt_ccd`` (default=0)
    :param dyn_bgd_dt_ccd: dynamic background T_ccd temperature bonus (default=-4.0, degC)
    :param stars: table of AGASC stars (will be fetched from agasc if None)
    :param include_ids_acq: list of AGASC IDs of stars to include in acq catalog
    :param include_halfws_acq: list of acq halfwidths corresponding to ``include_ids``.
                               For values of ``0`` proseco chooses the best halfwidth(s).
    :param exclude_ids_acq: list of AGASC IDs of stars to exclude from acq catalog
    :param include_ids_fid: list of fiducial lights to include by index.  If no possible
                            sets of fids include the id, no fids will be selected.
    :param exclude_ids_fid: list of fiducial lights to exclude by index
    :param include_ids_guide: list of AGASC IDs of stars to include in guide catalog
    :param exclude_ids_guide: list of AGASC IDs of stars to exclude from guide catalog
    :param img_size_guide: readout window size for guide stars (6, 8, or ``None``).
                           For default value ``None`` use 8 for no fids, 6 for fids.
    :param optimize: optimize star catalog after initial selection (default=True)
    :param verbose: provide extra logging info (mostly calc_p_safe) (default=False)
    :param print_log: print the run log to stdout (default=False)
    :param raise_exc: raise exception if it occurs in processing (default=True)

    :returns: ACATable of stars and fids

    """
    from .catalog import get_aca_catalog

    return get_aca_catalog(*args, **kwargs)


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr

    return testr.test(*args, **kwargs)
