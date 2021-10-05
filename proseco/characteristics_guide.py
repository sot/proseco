import numpy as np
from astropy.io import ascii

# Fid trap effect
# http://cxc.cfa.harvard.edu/mta/ASPECT/aca_weird_pixels/
fid_trap = {'row': -374,
            'col': 347,
            'margin': 8}

# Add this padding to region checked for bad pixels (in addition to dither)
dither_pix_pad = 0.4

# Reference faint magnitude limit and CCD temperature for guide star candidate
# selection.  The faint mag limit is scaled via the actual CCD temperature in
# the selection process (maintaining equivalent SNR relative to dark current).
#
# The reference temperature is set to -7.8 to reflect that circa Jan 2020 we had
# not be applying any scaling and nominally allowing stars as faint as 10.3 mag
# at temperatures up to -7.8 C.  In practice other constraints may kick in, but
# for the initial box criteria for candidate selection this is used.
ref_faint_mag_t_ccd = -7.8
ref_faint_mag = 10.3

# Error / check labeling
errs = {'mag range': 1,
        'aspq1': 2,
        'hot pix': 4,
        'spoiler (frac)': 8,
        'spoiler (bgd)': 16,
        'spoiler (line)': 32,
        'col spoiler': 64,
        'bad color': 128}
err_map = {v: k for k, v in errs.items()}


# Box spoiler check
box_spoiler = {'halfbox': 5,
               'magdiff': -4}

# Spoiler max FM centroid offsets using the results of the right plot in Fig 2
# of https://cxc.harvard.edu/mta/ASPECT/Papers/radial_spoilers.ps. The curves
# below represent the 0.05 arcsec and 0.5 arcsec offset contours respectively.
# Data were read off the plots using the app https://apps.automeris.io/wpd/. The
# 0.0 arcsec offset is the same as the legacy SAUSAGE "direct catalog check" of
# 9.0 mag - 0.5 * dist (pix).
spoiler_fm_offset = {}

spoiler_fm_offset[0.0] = ascii.read(
    """dmag, sep
       9.0, 0.0
       -1.0, 100.0""")

spoiler_fm_offset[0.05] = ascii.read(
    """dmag, sep
       3.0, 0.0
       4.08, 1.75
       5.05, 5.19
       5.45, 8.29""")

spoiler_fm_offset[0.5] = ascii.read(
    """dmag, sep
       1.0, 0.0
       2.97, 7.61
       4.13, 13.27
       4.04, 19.20
       3.41, 25.34
       1.85, 29.31
       -0.01, 33.09
       -1.13, 39.22
       -4.24, 50.0""")

# Search Stages
stages = [{"Stage": 1,
           "SigErrMultiplier": 3,
           "ASPQ1Lim": 0,
           "MagLimit": [6.2, 10.2],
           "DoBminusVcheck": 1,
           "Spoiler": {
               "BgPixThresh": 25,
               "RegionFrac": .05,
               "FirstMomentOffset": 0.0,  # arcsec
           },
           "Imposter": {
               "CentroidOffsetLim": .2,
           }},
          {"Stage": 2,
           "SigErrMultiplier": 2,
           "ASPQ1Lim": 0,
           "MagLimit": [6.2, 10.2],
           "DoBminusVcheck": 1,
           "Spoiler": {
               "BgPixThresh": 25,
               "RegionFrac": .05,
               "FirstMomentOffset": 0.0,
           },
           "Imposter": {
               "CentroidOffsetLim": .5,
           }},
          {"Stage": 3,
           "SigErrMultiplier": 1,
           "ASPQ1Lim": 10,
           "MagLimit": [6.1, 10.3],
           "DoBminusVcheck": 1,
           "Spoiler": {
               "BgPixThresh": 25,
               "RegionFrac": .05,
               "FirstMomentOffset": 0.0,
           },
           "Imposter": {
               "CentroidOffsetLim": 1.0,
           }},
          {"Stage": 4,
           "SigErrMultiplier": 0,
           "ASPQ1Lim": 20,
           "MagLimit": [6.0, 10.3],
           "DoBminusVcheck": 1,
           "Spoiler": {
               "BgPixThresh": 25,
               "RegionFrac": .05,
               "FirstMomentOffset": 0.05,
           },
           "Imposter": {
               "CentroidOffsetLim": 2.0,
           }},
          {"Stage": 5,
           "SigErrMultiplier": 0,
           "ASPQ1Lim": 20,
           "MagLimit": [5.9, 10.3],
           "DoBminusVcheck": 0,
           "Spoiler": {
               "BgPixThresh": 25,
               "RegionFrac": .05,
               "FirstMomentOffset": 0.5,
           },
           "Imposter": {
               "CentroidOffsetLim": 3.5,
           }},
          {"Stage": 6,
           "SigErrMultiplier": 0,
           "ASPQ1Lim": 20,
           "MagLimit": [5.6, 10.3],
           "DoBminusVcheck": 0,
           "Spoiler": {
               "BgPixThresh": 25,
               "RegionFrac": .05,
               "FirstMomentOffset": 0.5,
           },
           "Imposter": {
               "CentroidOffsetLim": 4.0,
           },
           }
          ]


# Guide cluster checks.
# Index is the "n_minus" value, so n - 0 stars need minimum 2500 max separation
# n - 1 stars require 1000 arcsec separation.
# n - 2 stars require 500 arcsec minimum max separation in the pairs.
cluster_thresholds = [2500, 1000, 500]
surplus_stars = 8


# Index template file name
index_template_file = 'index_template_guide.html'
