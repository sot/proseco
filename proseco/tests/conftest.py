import agasc
import pytest
from agasc import get_agasc_filename


@pytest.fixture(autouse=True)
def clear_star_dist_cache():
    from proseco.guide import STAR_PAIR_DIST_CACHE
    STAR_PAIR_DIST_CACHE.clear()


@pytest.fixture()
def disable_fid_offsets(monkeypatch):
    monkeypatch.setenv("PROSECO_ENABLE_FID_OFFSET", "False")


@pytest.fixture(autouse=True)
def use_fixed_chandra_models(monkeypatch):
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")


@pytest.fixture(autouse=True)
def disable_agasc_supplement(monkeypatch):
    monkeypatch.setenv(agasc.SUPPLEMENT_ENABLED_ENV, "False")


@pytest.fixture()
def disable_overlap_penalty(monkeypatch):
    monkeypatch.setenv("PROSECO_DISABLE_OVERLAP_PENALTY", "True")


# By default test with the latest AGASC version available including release candidates
@pytest.fixture(autouse=True)
def proseco_agasc_rc(monkeypatch):
    agasc_file = get_agasc_filename("proseco_agasc_*", allow_rc=True)
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)


@pytest.fixture()
def proseco_agasc_1p7(monkeypatch):
    agasc_file = get_agasc_filename("proseco_agasc_*", version="1p7")
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)


@pytest.fixture()
def miniagasc_1p7(monkeypatch):
    agasc_file = get_agasc_filename("miniagasc_*", version="1p7")
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)
