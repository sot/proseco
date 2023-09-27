import pytest
from agasc import get_agasc_filename


@pytest.fixture(autouse=True)
def use_fixed_chandra_models(monkeypatch):
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")


@pytest.fixture()
def proseco_agasc_1p7(monkeypatch):
    agasc_file = get_agasc_filename("proseco_agasc_*", version="1p7")
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)


@pytest.fixture()
def miniagasc_1p7(monkeypatch):
    agasc_file = get_agasc_filename("miniagasc_*", version="1p7")
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)


@pytest.fixture(autouse=True)
def proseco_agasc_rc(monkeypatch):
    agasc_file = get_agasc_filename("proseco_agasc_*", allow_rc=True)
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)
