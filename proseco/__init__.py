import ska_helpers

__version__ = ska_helpers.get_version(__package__)


def get_aca_catalog(*args, **kwargs):
    from .catalog import get_aca_catalog
    return get_aca_catalog(*args, **kwargs)


def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
