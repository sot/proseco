from ska_helpers.utils import LazyDict
import agasc


CCD = {'row_min': -512.0,
       'row_max': 512.0,
       'col_min': -512.0,
       'col_max': 512.0,
       'fov_pad': 40.0,  # Padding *outside* CCD (filtering AGASC stars in/near FOV in set_stars)
       'window_pad': 7,
       'row_pad': 8,
       'col_pad': 1,
       'guide_extra_pad': 3,
       'bgpix': ['A1', 'B1', 'G1', 'H1', 'I4', 'J4', 'O4', 'P4']}

PIX_2_ARC = 4.96289
ARC_2_PIX = 1.0 / PIX_2_ARC

# Maximum value of star catalog MAXMAG parameter.  Clip value and implications
# of clipping discussed in emails circa June 7, 2019 with search key "maxmag".
max_maxmag = 11.2

# Convenience characteristics
max_ccd_row = CCD['row_max'] - CCD['row_pad']  # Max allowed row for stars (SOURCE?)
max_ccd_col = CCD['col_max'] - CCD['col_pad']  # Max allow col for stars (SOURCE?)

# Column spoiler rules
col_spoiler_mag_diff = 4.5
col_spoiler_pix_sep = 10  # pixels

# ACA T_ccd penalty limit (degC).
# Above this limit a temperature "penalty" is applied via get_effective_t_ccd()
aca_t_ccd_penalty_limit = -7.5

# Dark current that corresponds to a 5.0 mag star in a single pixel.  Apply
# this value to the region specified by bad_pixels.
bad_pixel_dark_current = 700_000

# Bad pixels.
# Fid trap: http://cxc.cfa.harvard.edu/mta/ASPECT/aca_weird_pixels/
# Diminished response: https://chandramission.slack.com/archives/G01LN40AXPG/p1624543385000700
#   and subsequent discussion in nearby aspect-team threads.
#             [row0 row1 col0 col1] (where values are inclusive on both ends)
bad_pixels = [[-245, 0, 454, 454],  # Bad column
              [-374, -374, 347, 347],  # Fid trap
              [-319, -317, -299, -296]  # Diminished response
              ]


def _load_bad_star_set():
    # Add in entries from the AGASC supplement file, if possible, warn otherwise
    out = agasc.get_supplement_table('bad', as_dict=True)
    return out


bad_star_set = LazyDict(_load_bad_star_set)
