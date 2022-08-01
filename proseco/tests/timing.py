# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Provide a function to check for timing regressions for any changes.
"""

import sys
import time

import numpy as np

from proseco import get_aca_catalog
from proseco.tests.test_common import mod_std_info


def time_get_aca_catalog(n_samples=100):
    """
    Get n_samples catalogs for standard parameters (all at the same 2018:001) date.
    """

    np.random.seed(0)

    ras = np.random.uniform(0, 360, size=n_samples)
    decs = np.random.uniform(-90, 90, size=n_samples)
    rolls = np.random.uniform(0, 360, size=n_samples)

    # Get rid of initial imports or one-time startup stuff (e.g. in AGASC)
    get_aca_catalog(**mod_std_info())

    t0 = time.time()
    for ra, dec, roll in zip(ras, decs, rolls):
        print('.', end='')
        sys.stdout.flush()
        get_aca_catalog(**mod_std_info(att=(ra, dec, roll)))

    t1 = time.time()
    print()

    print(f'Got {n_samples} catalogs in {(t1 - t0) / n_samples:.3f} secs (mean)')
