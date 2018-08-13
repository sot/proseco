from __future__ import print_function, division, absolute_import

import numpy as np
from astropy.table import Table

box_sizes = np.array([160, 140, 120, 100, 80, 60])  # MUST be descending order
max_ccd_row = 512 - 8  # Max allowed row for stars (SOURCE?)
max_ccd_col = 512 - 1  # Max allow col for stars (SOURCE?)


# Maneuver angle bins used in the p_man_errs table below [deg].
p_man_errs_angles = np.array([0, 5, 20, 40, 60, 80, 100, 120, 180], dtype=int)

# P_man_errs table from text output copied from proseco-man-err-distribution
# notebook in skanb/star_selection.
#
# P_man_errs is a table that defines the probability of a maneuver error being
# within a defined lo/hi bin [arcsec] given a maneuver angle [deg] within a
# bin range.  The first two columns specify the maneuver error bins and the
# subsequent column names give the maneuver angle bin in the format
# <angle_lo>-<angle_hi>.  The source table does not include the row for
# the 0-60 man err case: this is generated automatically from the other
# values so the sum is 1.0.
#
_p_man_errs_text = """
man_err_lo man_err_hi 0-5 5-20 20-40 40-60 60-80 80-100 100-120 120-180
---------- ---------- --- ---- ----- ----- ----- ------ ------- -------
        60         80 0.1  0.2   0.5   0.6   1.6    4.0     8.0     8.0
        80        100 0.0  0.1   0.2   0.3   0.5    1.2     2.4     2.4
       100        120 0.0  0.0   0.1   0.2   0.3    0.8     0.8     0.8
       120        140 0.0  0.0  0.05  0.05   0.2    0.4     0.4     0.4
       140        160 0.0  0.0  0.05  0.05   0.2    0.2     0.2     0.4
"""
p_man_errs = Table.read(_p_man_errs_text, format='ascii.fixed_width_two_line')
# Generate the initial row
zero_row = [0, p_man_errs['man_err_lo'][0]]
for ang0, ang1 in zip(p_man_errs_angles[:-1], p_man_errs_angles[1:]):
    name = '{}-{}'.format(ang0, ang1)
    col = p_man_errs[name]
    col /= 100
    zero_row.append(1 - np.sum(col))
p_man_errs.insert_row(0, zero_row)