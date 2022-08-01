# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from chandra_aca.transform import (
    pixels_to_yagzag,
    radec_to_yagzag,
    yagzag_to_pixels,
    yagzag_to_radec,
)

import proseco.characteristics as ACA
from proseco.characteristics import MonCoord, MonFunc
from proseco.core import ACACatalogTable


class BadMonitorError(ValueError):
    pass


def get_mon_catalog(obsid=0, **kwargs):
    """
    Get a catalog of monitor stars

    If ``obsid`` corresponds to an already-scheduled obsid then the parameters
    ``att``, ``t_ccd``, ``date``, and ``dither`` will
    be fetched via ``mica.starcheck`` if not explicitly provided here.

    :param obsid: obsid (default=0)
    :param att: attitude (any object that can initialize Quat)
    :param t_ccd: ACA CCD temperature (degC)
    :param date: date of acquisition (any DateTime-compatible format)
    :param dither: dither size 2-element tuple: (dither_y, dither_z) (float, arcsec)
    :param n_guide: number of guide stars to attempt to get
    :param fids: selected fids (used for guide star exclusion)
    :param stars: astropy.Table of AGASC stars (will be fetched from agasc if None)
    :param include_ids: list of AGASC IDs of stars to include in guide catalog
    :param exclude_ids: list of AGASC IDs of stars to exclude from guide catalog
    :param dark: ACAImage of dark map (fetched based on time and t_ccd if None)
    :param print_log: print the run log to stdout (default=False)

    :returns: MonTable of acquisition stars
    """
    mons = MonTable()
    mons.set_attrs_from_kwargs(obsid=obsid, **kwargs)
    mons.set_stars()

    # Pre-populate mons with an empty table with the right names and columns
    names = {
        'slot': np.int64,
        'idx': np.int64,
        'id': np.int64,
        'type': 'U3',
        'sz': 'U3',
        'p_acq': np.float64,
        'mag': np.float64,
        'maxmag': np.float64,
        'yang': np.float64,
        'zang': np.float64,
        'row': np.float64,
        'col': np.float64,
        'ra': np.float64,
        'dec': np.float64,
        'dim': np.int64,
        'res': np.int64,
        'halfw': np.int64,
    }

    for name, dtype in names.items():
        mons[name] = np.empty(0, dtype=dtype)

    # Process monitor window requests, converting them into fake stars that
    # are added to the include_ids list.
    mons.process_monitors()

    return mons


class MonTable(ACACatalogTable):
    # Define base set of allowed keyword args to __init__. Subsequent MetaAttribute
    # or AliasAttribute properties will add to this.
    allowed_kwargs = ACACatalogTable.allowed_kwargs

    # Catalog type when plotting (None | 'FID' | 'ACQ' | 'GUI' | 'MON')
    catalog_type = 'MON'

    # Elements of meta that should not be directly serialized to pickle.
    # (either too big or requires special handling).
    pickle_exclude = ('stars', 'dark')

    # Name of table.  Use to define default file names where applicable.
    name = 'mon'

    # Required attributes
    required_attrs = ('att', 'date')

    @property
    def t_ccd(self):
        # For fids use the guide CCD temperature
        return self.t_ccd_guide

    @t_ccd.setter
    def t_ccd(self, value):
        self.t_ccd_guide = value

    def process_monitors(self):
        """Process monitor window requests"""
        if self.monitors is None:
            return

        # Add columns for each of the three coordinate representations. The
        # original list input for monitors has been turned into a Table by the
        # Meta processing.
        monitors = self.monitors
        monitors['id'] = 0
        monitors['ra'] = 0.0
        monitors['dec'] = 0.0
        monitors['yang'] = 0.0
        monitors['zang'] = 0.0
        monitors['row'] = 0.0
        monitors['col'] = 0.0

        for monitor in monitors:
            if monitor['coord_type'] == MonCoord.RADEC:
                # RA, Dec
                monitor['ra'], monitor['dec'] = monitor['coord0'], monitor['coord1']
                monitor['yang'], monitor['zang'] = radec_to_yagzag(
                    monitor['ra'], monitor['dec'], self.att
                )
                monitor['row'], monitor['col'] = yagzag_to_pixels(
                    monitor['yang'], monitor['zang'], allow_bad=True
                )

            elif monitor['coord_type'] == MonCoord.ROWCOL:
                # Row, col
                monitor['row'], monitor['col'] = monitor['coord0'], monitor['coord1']
                monitor['yang'], monitor['zang'] = pixels_to_yagzag(
                    monitor['row'], monitor['col'], allow_bad=True, flight=True
                )
                monitor['ra'], monitor['dec'] = yagzag_to_radec(
                    monitor['yang'], monitor['zang'], self.att
                )

            elif monitor['coord_type'] == MonCoord.YAGZAG:
                # Yag, zag
                monitor['yang'], monitor['zang'] = monitor['coord0'], monitor['coord1']
                monitor['row'], monitor['col'] = yagzag_to_pixels(
                    monitor['yang'], monitor['zang'], allow_bad=True
                )
                monitor['ra'], monitor['dec'] = yagzag_to_radec(
                    monitor['yang'], monitor['zang'], self.att
                )

        # Process bona fide monitor windows according to function
        mon_id = 1000
        for monitor in self.monitors:
            if monitor['function'] in (MonFunc.GUIDE, MonFunc.MON_TRACK):
                # Try to get star at MON position
                dist = np.linalg.norm(
                    [
                        self.stars['yang'] - monitor['yang'],
                        self.stars['zang'] - monitor['zang'],
                    ],
                    axis=0,
                )
                idx = np.argmin(dist)
                if dist[idx] < 2.0:
                    star = self.stars[idx]
                    monitor['id'] = star['id']
                    monitor['mag'] = star['mag']
                elif monitor['function'] == MonFunc.GUIDE:
                    raise BadMonitorError(
                        'no acceptable AGASC star within '
                        '2 arcsec of monitor position'
                    )

            if monitor['function'] in (MonFunc.MON_FIXED, MonFunc.MON_TRACK):
                if monitor['id'] == 0:
                    monitor['id'] = mon_id
                    mon_id += 1

                # Make a stub row for a MON entry using zero everywhere. This
                # also works for str (giving '0').
                mon = {col.name: col.dtype.type(0) for col in self.itercols()}
                # These type codes get fixed later in merge_catalog
                mon['type'] = (
                    'MFX' if monitor['function'] == MonFunc.MON_FIXED else 'MTR'
                )
                mon['sz'] = '8x8'
                mon[
                    'dim'
                ] = -999  # Set an obviously bad value for DTS, gets fixed later.
                mon['res'] = 0
                mon['halfw'] = 20
                mon['maxmag'] = ACA.monitor_maxmag
                for name in ('id', 'mag', 'yang', 'zang', 'row', 'col', 'ra', 'dec'):
                    mon[name] = monitor[name]

                # Finally add the MON as a row in table
                self.add_row(mon)

            elif monitor['function'] != MonFunc.GUIDE:
                raise ValueError(f'unexpected monitor function {monitor["function"]}')
