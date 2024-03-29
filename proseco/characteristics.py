import json

import agasc
from ska_helpers import chandra_models
from ska_helpers.utils import LazyDict

CCD = {
    "row_min": -512.0,
    "row_max": 512.0,
    "col_min": -512.0,
    "col_max": 512.0,
    "fov_pad": 40.0,  # Padding *outside* CCD (filtering AGASC stars in/near FOV in set_stars)
    "window_pad": 7,
    "row_pad": 8,
    "col_pad": 1,
    "guide_extra_pad": 3,
    "bgpix": ["A1", "B1", "G1", "H1", "I4", "J4", "O4", "P4"],
}

PIX_2_ARC = 4.96289
ARC_2_PIX = 1.0 / PIX_2_ARC


class MonFunc:
    AUTO = 0
    GUIDE = 1
    MON_TRACK = 2
    MON_FIXED = 3


class MonCoord:
    RADEC = 0
    ROWCOL = 1
    YAGZAG = 2


# Maximum value of star catalog MAXMAG parameter.  Clip value and implications
# of clipping discussed in emails circa June 7, 2019 with search key "maxmag".
max_maxmag = 11.2
min_delta_maxmag = 0.5
max_delta_maxmag = 1.5

# Monitor window MAXMAG. This value comes from the commanded MAXMAG parameter
# in the backstop AOSTRCAT command for obsid 23382 in MAR0821A. This also
# corresponds to the max mag value that the ACA can put into telemetry.
monitor_maxmag = 13.9375

# Convenience characteristics
max_ccd_row = CCD["row_max"] - CCD["row_pad"]  # Max allowed row for stars (SOURCE?)
max_ccd_col = CCD["col_max"] - CCD["col_pad"]  # Max allow col for stars (SOURCE?)

# Column spoiler rules
col_spoiler_mag_diff = 4.5
col_spoiler_pix_sep = 10  # pixels

# Dark current that corresponds to a 5.0 mag star in a single pixel.  Apply
# this value to the region specified by bad_pixels.
bad_pixel_dark_current = 700_000

# Bad pixels.
# Fid trap: http://cxc.cfa.harvard.edu/mta/ASPECT/aca_weird_pixels/
# Diminished response: https://chandramission.slack.com/archives/G01LN40AXPG/p1624543385000700
#   and subsequent discussion in nearby aspect-team threads.
#             [row0 row1 col0 col1] (where values are inclusive on both ends)
bad_pixels = [
    [-245, 0, 454, 454],  # Bad column
    [-374, -374, 347, 347],  # Fid trap
    [-319, -317, -299, -296],  # Diminished response
]


def _load_bad_star_set():
    # Add in entries from the AGASC supplement file, if possible, warn otherwise
    out = agasc.get_supplement_table("bad", as_dict=True)
    return out


bad_star_set = LazyDict(_load_bad_star_set)

# READ-ONLY variable which gives the version of chandra_models being use for the ACA
# xija model from which the planning and penalty limits are extracted. This module
# variable is set on demand to the chandra_models repo version. To select a specific
# version set the CHANDRA_MODELS_DEFAULT_VERSION environment variable.
#
# chandra_models_version

# The next two characteristics are lazily defined to ensure import succeeds.

# aca_t_ccd_penalty_limit : ACA T_ccd penalty limit (degC).
# Above this limit a temperature "penalty" is applied via get_effective_t_ccd()

# aca_t_ccd_planning_limit : ACA T_ccd planning limit (degC).
# Predicted ACA CCD temperatures must be below this limit.

ACA_SPEC_ATTRS = {
    "aca_t_ccd_penalty_limit": "planning.penalty.high",
    "aca_t_ccd_planning_limit": "planning.warning.high",
    "chandra_models_version": None,
}


def __getattr__(name):
    """Lazily define module attributes for the ACA planning and penalty limits"""
    if name in ACA_SPEC_ATTRS:
        attrs = _get_aca_spec_attrs()
        return attrs[name]
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


@chandra_models.chandra_models_cache
def _get_aca_spec_attrs():
    """Set global variables for ACA thermal planning and penalty limits"""
    spec_txt, info = chandra_models.get_data("chandra_models/xija/aca/aca_spec.json")
    spec = json.loads(spec_txt)

    out = {"chandra_models_version": info["version"]}
    for name, spec_name in ACA_SPEC_ATTRS.items():
        if spec_name is not None:
            out[name] = spec["limits"]["aacccdpt"].get(spec_name)

    return out


# Make sure module-level `dir()` includes the lazy attributes.

# Grab the module attributes before defining __dir__
_attrs = dir()


def __dir__():
    return sorted(_attrs + list(ACA_SPEC_ATTRS))
