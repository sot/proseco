# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for guide candidate awareness in fid selection and acq-fid-guide optimization.
"""

import numpy as np
import pytest

from proseco import get_aca_catalog
from proseco.fid import FidTable, get_fid_catalog
from proseco.guide import MIN_DYN_BGD_ANCHOR_STARS, get_guide_catalog, get_t_ccds_bonus
from proseco.tests.test_common import mod_std_info

from ..core import StarsTable


def test_get_t_ccds_bonus_1():
    mags = [1, 10, 2, 11, 3, 4]
    t_ccd = 10

    # Temps corresponding to two faintest stars are smaller.
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=2, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 9, 10, 9, 10, 10])

    # Temps corresponding to three faintest stars are smaller.
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=3, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 9, 10, 9, 10, 9])

    # Temps corresponding to just the three faintest stars are smaller because of the
    # minimum number of anchor stars = 3.
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=4, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 9, 10, 9, 10, 9])

    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=0, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10, 10, 10, 10, 10, 10])


def test_get_t_ccds_bonus_min_anchor():
    mags = [1, 10, 2]
    t_ccd = 10
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=2, dyn_bgd_dt_ccd=-1)
    # Assert that there are at least MIN_DYN_BGD_ANCHOR_STARS without bonus
    assert np.count_nonzero(t_ccds == t_ccd) >= MIN_DYN_BGD_ANCHOR_STARS

    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=4, dyn_bgd_dt_ccd=-1)
    assert np.count_nonzero(t_ccds == t_ccd) >= MIN_DYN_BGD_ANCHOR_STARS


def test_get_t_ccds_bonus_small_catalog():
    mags = [1]
    t_ccd = 10
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=2, dyn_bgd_dt_ccd=-1)
    assert np.all(t_ccds == [10])


def test_get_t_ccds_bonus_no_bonus_stars():
    """When dyn_bgd_n_faint=0, all t_ccds should equal input t_ccd."""
    mags = np.array([8.0, 9.0, 10.0, 10.5])
    t_ccd = -10.0
    t_ccds = get_t_ccds_bonus(mags, t_ccd, dyn_bgd_n_faint=0, dyn_bgd_dt_ccd=5.0)
    assert np.allclose(t_ccds, -10.0)


# Create a synthetic test for fid trap scoring and detection
@pytest.mark.parametrize("row", [-200, 200])
def test_fid_trap_scoring(row):
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=4)
    # For the nominal ACIS-I FID 6 position here of -69.87 345.27
    # a star will cause the fid trap if it has about the same distance
    # to the register as the fid distance to the trap.
    # Either positive or negative row works so this test is parameterized.
    stars.add_fake_star(id=1001, mag=7.0, row=row, col=400)
    args = mod_std_info(detector="ACIS-I")
    fids = get_fid_catalog(stars=stars, guide_cands=stars, **args)
    assert fids.cand_fids.get_id(6)["spoiler_score"] >= 10
    # And for these cases, confirm 6 is not selected
    assert 6 not in fids["id"]


def test_fid_trap_optimize():
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=4, mag=[8, 9, 9.5, 10])
    # For the nominal ACIS-I FID 6 position here of -69.87 345.27
    # a star will cause the fid trap if it has about the same distance
    # to the register as the fid distance to the trap.
    # Either positive or negative row works so this test is parameterized.
    stars.add_fake_star(id=1001, mag=7.0, row=200, col=400)
    kwargs = mod_std_info(detector="ACIS-I")

    initial_fids = get_fid_catalog(stars=stars, **kwargs)
    # Now, spoil all of the candidate fids on purpose with bright stars
    for fid in initial_fids.cand_fids:
        stars.add_fake_star(
            id=2000 + fid["id"],
            mag=8.0,
            row=fid["row"],
            col=fid["col"],
        )

    fids = get_fid_catalog(stars=stars, guide_cands=stars, **kwargs)
    # No fids should be selected
    assert len(fids) == 0
    assert fids.cand_fids.get_id(6)["spoiler_score"] >= 10
    assert np.all(fids.cand_fids["spoiler_score"] >= 4)

    # However, the full catalog selection should select a fid catalog
    # via optimization.
    aca = get_aca_catalog(stars=stars, **kwargs, optimize=True)
    assert len(aca.fids) == kwargs["n_fid"]

    # And the fid trap fid 6 should not be selected
    assert 6 not in aca.fids["id"]

    # And none of the fid spoiler stars for the 3 selected fids should be in the catalog
    for fid in aca.fids:
        dys = np.abs(fid["yang"] - aca.guides["yang"])
        dzs = np.abs(fid["zang"] - aca.guides["zang"])
        spoiled = (dys < 5) & (dzs < 5)
        assert not np.any(spoiled)

    # And the log should show that optimization occurred
    log_opt = [
        rec["data"]
        for rec in aca.log_info["events"]
        if "optimize_acqs_fids" == rec["func"]
    ]
    assert len(log_opt) == 14
    assert "fid_ids=(1, 3, 5) N opt runs=10" in log_opt[-1]


def test_fid_spoil_short_circuits_optimization():
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=4, mag=[8, 9, 9.5, 10])
    # For the nominal ACIS-I FID 6 position here of -69.87 345.27
    # a star will cause the fid trap if it has about the same distance
    # to the register as the fid distance to the trap.
    # Either positive or negative row works so this test is parameterized.
    stars.add_fake_star(id=1001, mag=7.0, row=200, col=400)
    kwargs = mod_std_info(detector="ACIS-I")

    aca = get_aca_catalog(stars=stars, **kwargs, optimize=True)
    assert 6 not in aca.fids["id"]

    # And the log should show that the optimization function was called but not needed
    log_opt = [
        rec["data"]
        for rec in aca.log_info["events"]
        if "optimize_acqs_fids" == rec["func"]
    ]
    assert "No acq-fid optimization required" == log_opt[-1]


guide_test_cases = [
    {"mags": [6, 6, 7, 7, 8, 8], "expected_fids": [3, 5, 6]},
    {"mags": [10, 6, 6, 10, 9, 9], "expected_fids": [1, 4, 5]},
    {"mags": [8, 8, 8, 8, 8, 8], "expected_fids": [1, 5, 6]},
    {"mags": [10, 10, 10, 6, 6, 6], "expected_fids": [1, 2, 3]},
]


@pytest.mark.parametrize("case", guide_test_cases)
def test_guide_fid_optimization(case):
    stars = StarsTable.empty()
    kwargs = mod_std_info(detector="ACIS-I")
    initial_fids = get_fid_catalog(stars=stars, **kwargs)
    # Now, spoil all of the candidate fids on purpose with bright stars
    for fid, mag in zip(initial_fids.cand_fids, case["mags"]):
        stars.add_fake_star(
            id=2000 + fid["id"],
            mag=mag,
            row=fid["row"],
            col=fid["col"],
        )
    aca = get_aca_catalog(stars=stars, **kwargs, optimize=True)
    assert len(aca.fids) == kwargs["n_fid"]
    assert set(aca.fids["id"]) == set(case["expected_fids"])


def test_backward_compatibility_no_guide_cands():
    """Verify that fid selection works without guide_cands (backward compat)."""
    kwargs = mod_std_info(n_fid=3)

    # Call without guide_cands argument (old behavior)
    fids = get_fid_catalog(**kwargs)

    assert isinstance(fids, FidTable)


def test_guide_catalog_without_guides_arg():
    """Verify get_guide_catalog works without initial_guide_cands argument (backward compat)."""
    kwargs = mod_std_info(n_guide=5)

    # Call without guides argument (old behavior)
    guides = get_guide_catalog(**kwargs)

    assert len(guides) > 0 or len(guides.cand_guides) > 0
