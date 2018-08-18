# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function  # For Py2 compatibility

import numpy as np
import pytest
from astropy.table import Table

from Quaternion import Quat
import chandra_aca
from Ska.quatutil import radec2yagzag
import agasc


from ..gui import select_guide_stars



def test_select():
    # "random" ra/dec/roll
    selected = select_guide_stars(10, 20, 3)
    expected_star_ids = [156384720, 156376184, 156381600, 156379416, 156377856]
    assert selected['AGASC_ID'].tolist() == expected_star_ids


def test_obsid_19461():
    # overall poor star field
    selected = select_guide_stars(150.969600, 45.064045, 95.988189, t_ccd=-20, date='2017:362')
    expected_star_ids = [450103048, 450101704, 394003312, 450109160, 450109016]
    assert selected['AGASC_ID'].tolist() == expected_star_ids


def test_common_column_obsid_19904():
    # Should not select 1091705224 which has a column spoiler
    # Limit the star field to just a handful of stars including the star
    # and the column spoiler
    limited_stars = [1091709256, 1091698696, 1091705224, 1091702440, 1091704824]
    date = '2018:001'
    star_recs = [agasc.get_star(s, date=date) for s in limited_stars]
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    selected = select_guide_stars(248.515786,   -47.373203,   238.665124,
                                  date=date, t_ccd=-20,
                                  stars=stars)
    # Assert the column spoiled one isn't in the list
    assert 1091705224 not in selected['AGASC_ID'].tolist()
    assert selected['AGASC_ID'].tolist() == [1091702440, 1091698696, 1091704824]


def test_avoid_trap():
    # Set up a scenario where a star is selected fine at one roll, and then
    # confirm that it is not selected when roll places it on the trap
    limited_stars = [156384720, 156376184, 156381600, 156379416, 156384304]
    date = '2018:001'
    ra1 = 9.769
    dec1 = 20.147
    roll1= 295.078
    star_recs = [agasc.get_star(s, date=date) for s in limited_stars]
    stars = Table(rows=star_recs, names=star_recs[0].colnames)
    selected1 = select_guide_stars(ra1, dec1, roll1, date=date, t_ccd=-15,
                                  stars=stars)
    assert selected1['AGASC_ID'].tolist() == limited_stars
    # Roll so that 156381600 is on the trap
    ra2 = 9.769
    dec2 = 20.147
    roll2 = 297.078
    selected2 = select_guide_stars(ra2, dec2, roll2, date=date, t_ccd=-15,
                                   stars=stars)
    assert 156381600 not in selected2['AGASC_ID'].tolist()
