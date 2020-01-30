import os
import warnings
from pathlib import Path
from astropy.table import Table

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
aca_t_ccd_penalty_limit = -8.1

# Dark current that corresponds to a 5.0 mag star in a single pixel.  Apply
# this value to the region specified by bad_pixels.
bad_pixel_dark_current = 700_000

# Bad pixels.
# Fid trap: http://cxc.cfa.harvard.edu/mta/ASPECT/aca_weird_pixels/
bad_pixels = [[-245, 0, 454, 454],  # Bad column
              [-374, -374, 347, 347]]  # Fid trap

bad_star_set = set([36178592,
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
                    350392600,
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
                    1197635184,
                    1158290496])

# Add in entries from the AGASC supplement file, if possible, warn otherwise
agasc_supplement_file = Path(os.environ['SKA']) / 'data' / 'agasc' / 'agasc_supplement.h5'
if agasc_supplement_file.exists():
    bad_star_table = Table.read(str(agasc_supplement_file), path='bad')
    bad_star_set.update(set(bad_star_table['agasc_id'].tolist()))
else:
    warnings.warn(f'Unable to find {agasc_supplement_file}, using limited bad star set')
