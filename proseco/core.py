import os
import inspect
import time
from copy import copy

import numpy as np
import yaml
from scipy.interpolate import interp1d
from astropy.table import Table, Column

from chandra_aca.transform import yagzag_to_pixels
from Ska.quatutil import radec2yagzag
from agasc import get_agasc_cone
from Quaternion import Quat


def to_python(val):
    try:
        val = val.tolist()
    except AttributeError:
        pass
    return val


class AcqTable(Table):
    def __init__(self, *args, print_log=False, **kwargs):
        super(AcqTable, self).__init__(*args, **kwargs)
        self.log_info = {}
        self.log_info['events'] = []
        self.log_info['time0'] = time.time()
        self.print_log = print_log

        # Make printed table look nicer.  This is defined in advance
        # and will be applied the first time the table is represented.
        self._default_formats = {'p_acq': '.3f'}
        for name in ('yang', 'zang', 'row', 'col', 'mag', 'mag_err', 'color'):
            self._default_formats[name] = '.2f'
        for name in ('ra', 'dec', 'RA_PMCORR', 'DEC_PMCORR'):
            self._default_formats[name] = '.6f'

    def log(self, data, **kwargs):
        # Name of calling functions, starting from top (outermost) and
        # ending with function that called log()
        func = inspect.currentframe().f_back.f_code.co_name

        # Possible extension later to include calling context, but this
        # currently includes a whole bunch more, need to filter to just
        # this module.
        # framerecs = inspect.stack()[1:-1]
        # funcs = [framerec[3] for framerec in reversed(framerecs)]
        # func = funcs[-1]

        dt = time.time() - self.log_info['time0']
        kwargs = {key: to_python(val) for key, val in kwargs.items()}
        event = dict(dt=round(dt, 4),
                     func=func,
                     # funcs=funcs,
                     data=data,
                     **kwargs)
        self.log_info['events'].append(event)
        if self.print_log:
            data_str = ' ' * event.get('level', 0) * 4 + str(event['data'])
            print(f'{dt:7.3f} {func:20s}: {data_str}')

    def _base_repr_(self, *args, **kwargs):
        names = [name for name in self.colnames
                 if self[name].dtype.kind != 'O']

        # Apply default formats to applicable columns and delete
        # that default format so it doesn't get set again next time.
        for name in list(self._default_formats.keys()):
            if name in names:
                self[name].format = self._default_formats.pop(name)

        return super(AcqTable, self[names])._base_repr_(*args, **kwargs)

    def to_yaml(self, rootdir=None):
        """
        Serialize table as YAML and return string.  If ``rootdir`` is set then
        the table YAML is output to ``<rootdir>/obs<obsid>/acqs.yaml``.
        """
        out = {}
        exclude = ('stars', 'cand_acqs', 'dark', 'bad_stars')
        for par in self.meta:
            if par not in exclude:
                out[par] = to_python(self.meta[par])
        out['acqs'] = self.to_struct()
        out['cand_acqs'] = self.meta['cand_acqs'].to_struct()
        out['log_info'] = self.log_info

        yml = yaml.dump(out)

        if rootdir is not None:
            outdir = os.path.join(rootdir, 'obs{:05}'.format(self.meta['obsid']))
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outfile = os.path.join(outdir, 'acqs.yaml')
            with open(outfile, 'w') as fh:
                fh.write(yml)

        return yml

    @classmethod
    def from_yaml(cls, yaml_str):
        """
        Construct table from YAML string
        """
        obj = yaml.load(yaml_str)
        acqs = cls.from_struct(obj.pop('acqs'))
        acqs.meta['cand_acqs'] = cls.from_struct(obj.pop('cand_acqs'))
        acqs.log_info.update(obj.pop('log_info'))

        for par in obj:
            acqs.meta[par] = copy(obj[par])
        return acqs

    def to_struct(self):
        """Turn table into a dict structure with keys:

        - names: column names in order
        - dtype: column dtypes as strings
        - rows: list of dict with row values

        This takes pains to remove any numpy objects so the YAML serialization
        is tidy (pure-Python only).
        """
        rows = []
        colnames = self.colnames
        dtypes = [col.dtype.str for col in self.itercols()]
        out = {'names': colnames, 'dtype': dtypes}

        for row in self:
            outrow = {}
            for name in colnames:
                val = row[name]
                if isinstance(val, np.ndarray) and val.dtype.names:
                    val = Table(val)

                if isinstance(val, Table):
                    val = AcqTable.to_struct(val)

                elif isinstance(val, dict):
                    new_val = {}
                    for key, item in val.items():
                        if isinstance(key, tuple):
                            key = tuple(to_python(k) for k in key)
                        else:
                            key = to_python(key)
                        item = to_python(item)
                        new_val[key] = item
                    val = new_val

                else:
                    val = to_python(val)

                outrow[name] = val
            rows.append(outrow)

        # Only include rows=[..] kwarg if there are rows.  Table initializer is unhappy
        # with a zero-length list for rows, but is OK with just names=[..] dtype=[..].
        if rows:
            out['rows'] = rows

        return out

    @classmethod
    def from_struct(cls, struct):
        out = cls(**struct)
        for name in out.colnames:
            col = out[name]
            if col.dtype.kind == 'O':
                for idx, val in enumerate(col):
                    if isinstance(val, dict) and 'dtype' in val.keys():
                        col[idx] = Table(**val)
        return out


# AGASC columns not needed (at this moment) for acq star selection.
# Change as needed.
AGASC_COLS_DROP = [
    'RA',
    'DEC',
    'POS_CATID',
    'EPOCH',
    'PM_CATID',
    'PLX',
    'PLX_ERR',
    'PLX_CATID',
    'MAG',
    'MAG_ERR',
    'MAG_BAND',
    'MAG_CATID',
    'C1_CATID',
    'COLOR2',
    'COLOR2_ERR',
    'C2_CATID',
    'RSV1',
    'RSV2',
    'RSV3',
    'VAR_CATID',
    'ACQQ1',
    'ACQQ2',
    'ACQQ3',
    'ACQQ4',
    'ACQQ5',
    'ACQQ6',
    'XREF_ID1',
    'XREF_ID2',
    'XREF_ID3',
    'XREF_ID4',
    'XREF_ID5',
    'RSV4',
    'RSV5',
    'RSV6',
]

# Define a function that returns the observed standard deviation in ACA mag
# as a function of catalog mag.  This is about the std-dev for samples
# within an observation, NOT the error on the catalog mag (which is about
# the error on mean of the sample, not the std-dev of the sample).
#
# See skanb/star_selection/star-mag-std-dev.ipynb for the source of numbers.
#
# Note that get_mag_std is a function that is called with get_mag_std(mag_aca).
get_mag_std = interp1d(x=[-10, 6.7, 7.3, 7.8, 8.3, 8.8, 9.2, 9.7, 10.1, 11, 20],  # mag
                       y=[0.015, 0.015, 0.022, 0.029, 0.038, 0.055, 0.077,
                          0.109, 0.141, 0.23, 1.0],  # std-dev
                       kind='linear')


def get_stars(att, date=None, radius=1.2, logger=None):
    """
    Get AGASC stars in the ACA FOV.  This uses the mini-AGASC, so only stars
    within 3-sigma of 11.5 mag.  TO DO: maybe use the full AGASC, for faint
    candidate acq stars with large uncertainty.
    """
    if logger is None:
        def logger(*args, **kwargs):
            return None

    q_att = Quat(att)
    logger('Getting stars at ra={:.5f} dec={:.4f}'.format(q_att.ra, q_att.dec))
    stars = get_agasc_cone(q_att.ra, q_att.dec, radius=radius, date=date)
    logger('Got {} stars'.format(len(stars)), level=1)
    yag, zag = radec2yagzag(stars['RA_PMCORR'], stars['DEC_PMCORR'], q_att)
    yag *= 3600
    zag *= 3600
    row, col = yagzag_to_pixels(yag, zag, allow_bad=True, pix_zero_loc='edge')

    stars.remove_columns(AGASC_COLS_DROP)

    stars.rename_column('AGASC_ID', 'id')
    stars.add_column(Column(stars['RA_PMCORR'], name='ra'), index=1)
    stars.add_column(Column(stars['DEC_PMCORR'], name='dec'), index=2)
    stars.add_column(Column(yag, name='yang'), index=3)
    stars.add_column(Column(zag, name='zang'), index=4)
    stars.add_column(Column(row, name='row'), index=5)
    stars.add_column(Column(col, name='col'), index=6)

    stars.add_column(Column(stars['MAG_ACA'], name='mag'), index=7)  # For convenience

    # Mag_err column is the RSS of the catalog mag err (i.e. systematic error in
    # the true star mag) and the sample uncertainty in the ACA readout mag
    # for a star with mag=MAG_ACA.  The latter typically dominates above 9th mag.
    mag_aca_err = stars['MAG_ACA_ERR'] * 0.01
    mag_std_dev = get_mag_std(stars['MAG_ACA'])
    mag_err = np.sqrt(mag_aca_err ** 2 + mag_std_dev ** 2)
    stars.add_column(Column(mag_err, name='mag_err'), index=8)

    # Filter stars in or near ACA FOV
    rcmax = 512.0 + 200 / 5  # 200 arcsec padding around CCD edge
    ok = (row > -rcmax) & (row < rcmax) & (col > -rcmax) & (col < rcmax)
    stars = stars[ok]

    logger('Finished star processing', level=1)

    return stars
