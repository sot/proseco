from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.interpolate import interp1d
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
        60         80 0.0  0.2   0.5   0.6   1.6    4.0     8.0     8.0
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

#
# Possible maneuver errors in the table for use in acq.py
#
man_errs = p_man_errs['man_err_hi']


# Minimal set of columns to store for spoiler stars
spoiler_star_cols = ['id', 'yang', 'zang', 'row', 'col', 'mag', 'mag_err']


bad_pixels = [[-245, 0, 454, 454]]

bad_star_list = [36178592,
                 39980640,
                 185871616,
                 188751528,
                 190977856,
                 260968880,
                 260972216,
                 261621080,
                 296753512,
                 300948368,
                 301078152,
                 301080376,
                 301465776,
                 335025128,
                 335028096,
                 414324824,
                 444743456,
                 465456712,
                 490220520,
                 502793400,
                 509225640,
                 570033768,
                 614606480,
                 637144600,
                 647632648,
                 650249416,
                 656409216,
                 690625776,
                 692724384,
                 788418168,
                 849226688,
                 914493824,
                 956175008,
                 989598624,
                 1004817824,
                 1016736608,
                 1044122248,
                 1117787424,
                 1130635848,
                 1130649544,
                 1161827976,
                 1196953168,
                 1197635184]

# Minimum acquisition probability thresholds from starcheck for thumbs_up
acq_prob_n = 2
acq_prob = 8e-3


def _get_fid_acq_stages():
    fid_acqs = Table.read("""

   warns score P2=0.0 P2=2.0 P2=2.5 P2=3.0 P2=4.0 P2=5.0 P2=6.0 P2=8.0 P2=99.0
   ----- ----- ------ ------ ------ ------ ------ ------ ------ ------ -------
       -     0      0    1.9    2.4    2.8    3.6    4.5    5.0    6.0     6.0
       Y     1      0    1.9    2.4    2.8    3.6    4.5    5.0    6.0     6.0
      YY     2      0    1.9    2.4   2.75    3.4    4.2    4.2    4.5     4.5
     YYY     3      0    1.9    2.4    2.7    3.2    3.5    3.5    3.5     3.5
       R     4      0    1.7    2.2    2.5    3.1    3.4    3.4    3.4     3.4
      RY     5      0    1.7    2.2    2.4    3.0    3.3    3.3    3.3     3.3
     RYY     6      0    1.7    2.2    2.3    2.5    2.5    2.5    2.5     2.5
      RR     8      0    1.5    2.0    2.0    2.0    2.0    2.0    2.0     2.0
     RRY     9      0    1.5    2.0    2.0    2.0    2.0    2.0    2.0     2.0
     RRR    12     -1   -1.0   -1.0   -1.0   -1.0   -1.0   -1.0   -1.0    -1.0

    """, format='ascii.fixed_width_two_line')

    P2s = [float(name[3:]) for name in fid_acqs.colnames
           if name.startswith('P2=')]
    funcs = []
    for fid_acq in fid_acqs:
        vals = [fid_acq[name] for name in fid_acqs.colnames
                if name.startswith('P2=')]
        funcs.append(interp1d(P2s, vals))

    out = Table([fid_acqs['score'], funcs],
                names=['spoiler_score', 'min_P2'])
    out.add_index('spoiler_score')
    return out

fid_acq_stages = _get_fid_acq_stages()
