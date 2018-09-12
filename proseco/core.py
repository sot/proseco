import warnings
import pickle
import inspect
import time
from copy import copy
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import interp1d
from astropy.table import Table, Column

from chandra_aca.transform import yagzag_to_pixels, count_rate_to_mag, pixels_to_yagzag
from Ska.quatutil import radec2yagzag, yagzag2radec
import agasc
from Quaternion import Quat


def yagzag_to_radec(yag, zag, att):
    """
    Convert yag, zag [arcsec] to ra, dec [deg] for attitude ``att``.

    :param yag: y-angle [arcsec]
    :param zag: z-angle [arcsec]
    :returns: ra, dec [deg]
    """
    return yagzag2radec(yag / 3600, zag / 3600, att)


def radec_to_yagzag(ra, dec, att):
    """
    Convert ra, dec [deg] to yag, zag [arcsec] for attitude ``att``.

    :param ra: RA [deg]
    :param dec: Dec [deg]
    :returns: yag, zag [arcsec]
    """
    yag, zag = radec2yagzag(ra, dec, att)
    return yag * 3600, zag * 3600


def to_python(val):
    try:
        val = val.tolist()
    except AttributeError:
        pass
    return val


class ACACatalogTable(Table):
    """
    Base class for representing ACA catalogs in star selection.  This
    can apply to acq, guide and fid entries.

    Features:
    - Inherits from astropy Table
    - Logging
    - Serialization to/from YAML
    - Predefined formatting for nicer representations
    """
    # Elements of meta that should not be directly serialized to YAML (either
    # too big or requires special handling) or pickle.  Should be set by
    # subclass.
    yaml_exclude = ()
    pickle_exclude = ()

    # Name of table.  Use to define default file names where applicable.
    # Should be set by subclass, e.g. ``name = 'acqs'`` for AcqTable.
    name = 'aca_cat'

    def __init__(self, data=None, print_log=False, **kwargs):
        super().__init__(data=data, **kwargs)
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

    def make_index(self):
        # Low-tech index to quickly get a row or the row index by `id` column.
        self._id_index = {}

        for idx, row in enumerate(self):
            self._id_index[row['id']] = idx

    def get_id(self, id):
        """
        Return row corresponding to ``id`` in id column.

        :param id: row ``id`` column value
        :returns: table Row
        """
        return self[self.get_id_idx(id)]

    def get_id_idx(self, id):
        """
        Return row corresponding to ``id`` in id column.

        :param id: row ``id`` column value
        :returns: table row index (int)
        """
        if not hasattr(self, '_index'):
            self.make_index()

        try:
            idx = self._id_index[id]
            assert self['id'][idx] == id
        except (KeyError, IndexError, AssertionError):
            self.make_index()
            try:
                idx = self._id_index[id]
                assert self['id'][idx] == id
            except (KeyError, IndexError, AssertionError):
                raise KeyError(f'{id} is not in table')

        return idx

    @property
    def acqs(self):
        return self.meta.get('acqs')

    @acqs.setter
    def acqs(self, val):
        self.meta['acqs'] = val

    @property
    def fids(self):
        return self.meta.get('fids')

    @fids.setter
    def fids(self, val):
        self.meta['fids'] = val

    @property
    def guides(self):
        return self.meta.get('guides')

    @guides.setter
    def guides(self, val):
        self.meta['guides'] = val

    @property
    def thumbs_up(self):
        return self.meta.get('thumbs_up')

    @thumbs_up.setter
    def thumbs_up(self, val):
        self.meta['thumbs_up'] = val

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

        return super(ACACatalogTable, self[names])._base_repr_(*args, **kwargs)

    def __getstate__(self):
        columns, meta = super().__getstate__()
        meta = meta.copy()
        meta['log_info'] = self.log_info
        for attr in self.pickle_exclude:
            if attr in meta:
                del meta[attr]
        return columns, meta

    def __setstate__(self, state):
        super().__setstate__(state)
        self.log_info = self.meta['log_info']
        del self.meta['log_info']

    def to_pickle(self, rootdir='.'):
        """
        Write the catalog table as pickle to:
          ``<rootdir>/obs<obsid>/<name>.pkl``

        :param rootdir: root directory (default='.')
        """
        rootdir = Path(rootdir)
        outdir = rootdir / f'obs{self.meta["obsid"]:05}'
        if not outdir.exists():
            outdir.mkdir()
        outfile = outdir / f'{self.name}.pkl'
        with open(outfile, 'wb') as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_pickle(cls, obsid, rootdir='.'):
        """
        Construct table from pickle file in
          ``<rootdir>/obs<obsid>/<name>.pkl``

        :param obsid: Obsid
        :param rootdir: root directory (default='.')
        :returns: catalog table
        """
        rootdir = Path(rootdir)
        infile = rootdir / f'obs{obsid:05}' / f'{cls.name}.pkl'
        with open(infile, 'rb') as fh:
            return pickle.load(fh)

    def to_yaml_custom(self, out):
        """
        Defined by subclass to do custom process prior to YAML serialization
        and possible writing to file.  This method should modify the ``out``
        dict in place.
        """
        pass

    def to_yaml(self, rootdir=None):
        """
        Serialize table as YAML and return string.  If ``rootdir`` is set then
        the table YAML is output to ``<rootdir>/obs<obsid>/acqs.yaml``.
        """
        out = {}
        for par in self.meta:
            if par not in self.yaml_exclude:
                out[par] = to_python(self.meta[par])

        # Store table itself and log info
        out[self.name] = self.to_struct()
        out['log_info'] = self.log_info

        # Custom processing
        self.to_yaml_custom(out)

        yml = yaml.dump(out)

        if rootdir is not None:
            rootdir = Path(rootdir)
            outdir = rootdir / f'obs{self.meta["obsid"]:05}'
            if not outdir.exists():
                outdir.mkdir()
            outfile = outdir / f'{self.name}.yaml'
            with open(outfile, 'w') as fh:
                fh.write(yml)

        return yml

    def from_yaml_custom(self, obj):
        """
        Custom processing on the dict ``obj`` which is the raw result of
        loading the YAML representation.  ``self`` is the ACACatalogTable
        subclass that is receiving info from ``obj``, so this method allows
        custom processing of that.
        """
        pass

    @classmethod
    def from_yaml(cls, obsid, rootdir='.'):
        """
        Construct table from YAML string
        """
        rootdir = Path(rootdir)
        infile = rootdir / f'obs{obsid:05}' / f'{cls.name}.yaml'
        with open(infile, 'r') as fh:
            yaml_str = fh.read()

        obj = yaml.load(yaml_str)

        # Construct the table itself and log info
        tbl = cls.from_struct(obj.pop(cls.name))
        tbl.log_info.update(obj.pop('log_info'))

        # Custom stuff
        tbl.from_yaml_custom(obj)

        # Anything else gets dumped into meta dict
        for par in obj:
            tbl.meta[par] = copy(obj[par])
        return tbl

    def to_struct(self):
        """Turn table ``tbl`` into a dict structure with keys:

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
                    val = ACACatalogTable.to_struct(val)

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


class StarsTable(ACACatalogTable):
    @staticmethod
    def get_logger(logger):
        if logger is not None:
            return logger
        else:
            def null_logger(*args, **kwargs):
                pass
            return null_logger

    @classmethod
    def from_agasc(cls, att, date=None, radius=1.2, logger=None):
        """
        Get AGASC stars in the ACA FOV.  This uses the mini-AGASC, so only stars
        within 3-sigma of 11.5 mag.
        TO DO: maybe use the full AGASC, for faint candidate acq stars with
        large uncertainty.
        TO DO: AGASC version selector?

        :param att: any Quat-compatible attitude
        :param date: DateTime compatible date for star proper motion (default=NOW)
        :param radius: star cone radius [deg] (default=1.2)
        :param logger: logger object (default=None)

        :returns: StarsTable of stars
        """
        q_att = Quat(att)
        agasc_stars = agasc.get_agasc_cone(q_att.ra, q_att.dec, radius=radius, date=date)
        stars = StarsTable.from_stars(att, agasc_stars, copy=False)

        logger = StarsTable.get_logger(logger)
        logger(f'Got {len(stars)} stars from AGASC at '
               'ra={q_att.ra:.5f} dec={q_att.dec:.4f}',
               level=1)

        return stars

    @classmethod
    def from_agasc_ids(cls, att, agasc_ids, date=None, logger=None):
        """
        Get AGASC stars in the ACA FOV using a list of AGASC IDs.

        :param att: any Quat-compatible attitude
        :param agasc_ids: sequence of AGASC ID values
        :param date: DateTime compatible date for star proper motion (default=NOW)
        :param logger: logger object (default=None)

        :returns: StarsTable of stars
        """
        agasc_stars = []
        for agasc_id in agasc_ids:
            try:
                star = agasc.get_star(agasc_id, date=date)
            except Exception:
                raise ValueError(f'failed to get AGASC ID={agasc_id}')
            else:
                agasc_stars.append(star)
        agasc_stars = Table(rows=agasc_stars, names=agasc_stars[0].colnames)
        return StarsTable.from_stars(att, stars=agasc_stars)

    @classmethod
    def from_stars(cls, att, stars, logger=None, copy=True):
        """
        Return a StarsTable from an existing AGASC stars query.  This just updates
        columns in place.

        :param att: any Quat-compatible attitude
        :param stars: Table of stars
        :param logger: logger object (default=None)
        :param copy: copy ``stars`` table columns

        :returns: StarsTable of stars
        """
        logger = StarsTable.get_logger(logger)

        if isinstance(stars, StarsTable):
            logger('stars is a StarsTable, assuming positions are correct for att')
            return stars

        stars = cls(stars, copy=copy)
        logger(f'Updating star columns for attitude and convenience')

        q_att = stars.meta['q_att'] = Quat(att)
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

    @classmethod
    def empty(cls, att=(0, 0, 0)):
        """
        Return an empty StarsTable suitable for generating synthetic tables.

        :param att: any Quat-compatible attitude
        :returns: StarsTable of stars (empty)
        """
        return cls.from_agasc(att, radius=-1)

    def add_agasc_id(self, agasc_id):
        """
        Add a AGASC star to the current StarsTable.

        :param agasc_id: AGASC ID of the star to add
        """
        stars = StarsTable.from_agasc_ids(self.meta['q_att'], [agasc_id])
        self.add_row(stars[0])

    def add_fake_constellation(self, n_stars=8, size=1500, mag=7.0, **attrs):
        """
        Add a fake constellation of up to 8 stars consisting of a cross and square

                *
              *   *
            *       *
              *   *
                *

        yangs = [1,  0, -1,  0, 0.5,  0.5, -0.5, -0.5] * size
        zangs = [0,  1,  0, -1, 0.5, -0.5,  0.5, -0.5] * size

        Additional star table attributes can be specified as keyword args.  All
        attributes are broadcast as needed.

        Example::

          >>> stars = StarsTable.empty()
          >>> stars.add_fake_constellation(n_stars=4, size=1000, mag=7.5, ASPQ1=[0, 1, 0, 20])
          >>> stars['id', 'yang', 'zang', 'mag', 'ASPQ1']
          <StarsTable length=4>
            id    yang     zang     mag   ASPQ1
          int32 float64  float64  float32 int16
          ----- -------- -------- ------- -----
            100  1000.00     0.00    7.50     0
            101     0.00  1000.00    7.50     1
            102 -1000.00     0.00    7.50     0
            103     0.00 -1000.00    7.50    20

        :param n_stars: number of stars (default=8, max=8)
        :param size: size of constellation [arcsec] (default=2000)
        :param mag: star magnitudes (default=7.0)
        :param **attrs: other star table attributes
        """
        if n_stars > 8:
            raise ValueError('max value of n_stars is 8')

        ids = len(self) + 100 + np.arange(n_stars)
        yangs = np.array([1, 0, -1, 0, 0.5, 0.5, -0.5, -0.5][:n_stars]) * size
        zangs = np.array([0, 1, 0, -1, 0.5, -0.5, 0.5, -0.5][:n_stars]) * size

        arrays = [ids, yangs, zangs, mag]
        names = ['id', 'yang', 'zang', 'mag']
        for name, array in attrs.items():
            names.append(name)
            arrays.append(array)

        arrays = np.broadcast_arrays(*arrays)
        for vals in zip(*arrays):
            self.add_fake_star(**{name: val for name, val in zip(names, vals)})

    def add_fake_star(self, **star):
        """
        Add a star to the current StarsTable.

        The input kwargs must have at least:
        - One of yang/zang, ra/dec, or row/col
        - mag
        - mag_err (defaults to 0.1)

        Mag_err is set to 0.1 if not provided.  Yang/zang, ra/dec, and row/col
        RA/DEC_PMCORR, MAG_ACA, MAG_ACA_ERR will all be set according to the
        primary inputs unless explicitly provided.  All the rest will be set
        to default "good"" values that will preclude initial exclusion of the star.

        :param **star: keyword arg attributes corresponding to StarTable columns
        """
        names = ['id', 'ra', 'dec', 'yang', 'zang', 'row', 'col', 'mag', 'mag_err',
                 'POS_ERR', 'PM_RA', 'PM_DEC', 'MAG_ACA',
                 'MAG_ACA_ERR', 'CLASS', 'COLOR1', 'COLOR1_ERR', 'VAR', 'ASPQ1',
                 'ASPQ2', 'ASPQ3', 'RA_PMCORR', 'DEC_PMCORR']

        defaults = {'id': len(self) + 100,
                    'mag_err': 0.1,
                    'VAR': -9999}
        out = {name: star.get(name, defaults.get(name, 0)) for name in names}

        q_att = self.meta['q_att']
        if 'ra' in star and 'dec' in star:
            out['yang'], out['zang'] = radec_to_yagzag(out['ra'], out['dec'], q_att)
            out['row'], out['col'] = yagzag_to_pixels(out['yang'], out['zang'],
                                                      allow_bad=True)

        elif 'yang' in star and 'zang' in star:
            out['ra'], out['dec'] = yagzag_to_radec(out['yang'], out['zang'], q_att)
            out['row'], out['col'] = yagzag_to_pixels(out['yang'], out['zang'],
                                                      allow_bad=True)

        elif 'row' in star and 'col' in star:
            out['yang'], out['zang'] = pixels_to_yagzag(out['row'], out['col'],
                                                        allow_bad=True)
            out['ra'], out['dec'] = yagzag_to_radec(out['yang'], out['zang'], q_att)

        reqd_names = ('ra', 'dec', 'row', 'col', 'yang', 'zang', 'mag', 'mag_err')
        for name in reqd_names:
            if name not in out:
                raise ValueError(f'incomplete star data did not get {name} data')

        out['RA_PMCORR'] = out['ra']
        out['DEC_PMCORR'] = out['dec']
        out['MAG_ACA'] = out['mag']

        self.add_row(out)


def bin2x2(arr):
    """Bin 2-d ``arr`` in 2x2 blocks.  Requires that ``arr`` has even shape sizes"""
    shape = (arr.shape[0] // 2, 2, arr.shape[1] // 2, 2)
    return arr.reshape(shape).sum(-1).sum(1)


RC_6x6 = np.array([10, 11, 12, 13,
                   17, 18, 19, 20, 21, 22,
                   25, 26, 27, 28, 29, 30,
                   33, 34, 35, 36, 37, 38,
                   41, 42, 43, 44, 45, 46,
                   50, 51, 52, 53],
                  dtype=np.int64)

ROW_6x6 = np.array([1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5,
                    6, 6, 6, 6],
                   dtype=np.float64) + 0.5

COL_6x6 = np.array([2, 3, 4, 5,
                    1, 2, 3, 4, 5, 6,
                    1, 2, 3, 4, 5, 6,
                    1, 2, 3, 4, 5, 6,
                    1, 2, 3, 4, 5, 6,
                    2, 3, 4, 5],
                   dtype=np.float64) + 0.5


def get_image_props(ccd_img, c_row, c_col, bgd=None):
    """
    Do a pseudo-read and compute the background-subtracted magnitude
    of the image.  Returns the 8x8 image and mag.

    :param ccd_image: full-frame CCD image (e.g. dark current, with or without stars).
    :param c_row: int row at center (readout row-4 to row+4)
    :param c_col: int col at center
    :param bgd: background to subtract at each pixel.  If None then compute flight bgd.

    :returns: 8x8 image (ndarray), image mag
    """
    img = ccd_img[c_row - 4:c_row + 4, c_col - 4:c_col + 4]

    if bgd is None:
        raise NotImplementedError('no calc_flight_background here')
        # bgd = calc_flight_background(img)  # currently not defined

    img = img - bgd
    img_6x6 = img.flatten()[RC_6x6]
    img_sum = np.sum(img_6x6)
    r = np.sum(img_6x6 * ROW_6x6) / img_sum
    c = np.sum(img_6x6 * COL_6x6) / img_sum
    row = r + c_row - 4
    col = c + c_col - 4
    mag = count_rate_to_mag(img_sum)

    return img, img_sum, mag, row, col


def pea_reject_image(img):
    """
    Check if PEA would reject image (too narrow, too peaky, etc)
    """
    # To be implemented
    return False
