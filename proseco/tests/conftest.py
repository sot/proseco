import pytest


@pytest.fixture(autouse=True)
def use_fixed_chandra_models(monkeypatch):
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")
