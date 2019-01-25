import numpy as np

# Fid trap effect
# http://cxc.cfa.harvard.edu/mta/ASPECT/aca_weird_pixels/
fid_trap = {'row': -374,
            'col': 347,
            'margin': 8}

# Minimum scaled guide count for thumbs_up
min_guide_count = 4.0

# Add this padding to region checked for bad pixels (in addition to dither)
dither_pix_pad = 0.4

CCD = {'row_min': -512.0,
       'row_max': 512.0,
       'col_min': -512.0,
       'col_max': 512.0,
       'window_pad': 7,
       'row_pad': 8,
       'col_pad': 1,
       'bgpix': ['A1', 'B1', 'G1', 'H1', 'I4', 'J4', 'O4', 'P4']}


# For the check that ERs must have 3 stars brighter than 9th mag
er_n_bright_req = 3
er_bright_thresh = 9.0

PIX_2_ARC = 4.96289
ARC_2_PIX = 1.0 / PIX_2_ARC

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

# Mag spoiler line rules
mag_spoiler = {
    "MinSep": 7,
    "MaxSep": 11,
    "Intercept": 9,
    "Slope": 0.5,
    "MagDiffLimit": -1 * np.inf,
    }

# Search Stages
stages = [{"Stage": 1,
           "SigErrMultiplier": 3,
           "ASPQ1Lim": 0,
           "MagLimit": [5.9, 10.2],
           "DoBminusVcheck": 1,
           "Spoiler": {
            "BgPixThresh": 25,
            "RegionFrac": .05,
            },
           "Imposter": {
            "CentroidOffsetLim": .2,
            }},
          {"Stage": 2,
           "SigErrMultiplier": 2,
           "ASPQ1Lim": 0,
           "MagLimit": [5.9, 10.2],
           "DoBminusVcheck": 1,
           "Spoiler": {
            "BgPixThresh": 25,
            "RegionFrac": .05,
            },
           "Imposter": {
            "CentroidOffsetLim": .5,
            }},
          {"Stage": 3,
           "SigErrMultiplier": 1,
           "ASPQ1Lim": 10,
           "MagLimit": [5.9, 10.3],
           "DoBminusVcheck": 1,
           "Spoiler": {
            "BgPixThresh": 25,
            "RegionFrac": .05,
            },
           "Imposter": {
            "CentroidOffsetLim": 1.0,
            }},
          {"Stage": 4,
           "SigErrMultiplier": 0,
           "ASPQ1Lim": 20,
           "MagLimit": [5.9, 10.3],
           "DoBminusVcheck": 1,
           "Spoiler": {
            "BgPixThresh": 25,
            "RegionFrac": .05,
            },
           "Imposter": {
            "CentroidOffsetLim": 2.5,
            }},
          {"Stage": 5,
           "SigErrMultiplier": 0,
           "ASPQ1Lim": 20,
           "MagLimit": [5.9, 10.3],
           "DoBminusVcheck": 0,
           "Spoiler": {
            "BgPixThresh": 25,
            "RegionFrac": .05,
            },
           "Imposter": {
            "CentroidOffsetLim": 4.5,
            },
           }
          ]
