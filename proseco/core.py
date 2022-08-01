# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import functools
import inspect
import os
import pickle
import re
import time
import warnings
from copy import copy
from pathlib import Path

import agasc
import numpy as np
from astropy.table import Column, Table
from chandra_aca.aca_image import AcaPsfLibrary
from chandra_aca.transform import (
    count_rate_to_mag,
    mag_to_count_rate,
    pixels_to_yagzag,
    radec_to_yagzag,
    yagzag_to_pixels,
    yagzag_to_radec,
)
from mica.archive.aca_dark import get_dark_cal_id, get_dark_cal_image
from Quaternion import Quat
from scipy.interpolate import interp1d

from . import characteristics as ACA

# For testing this is used to cache fid tables for a detector
FIDS_CACHE = {}
APL = AcaPsfLibrary()

# Cache recently retrieved images which are called with the same args/kwargs
get_dark_cal_image = functools.lru_cache(maxsize=6)(get_dark_cal_image)
get_dark_cal_id = functools.lru_cache(maxsize=6)(get_dark_cal_id)


def to_python(val):
    try:
        val = val.tolist()
    except AttributeError:
        pass
    return val


# Row and column indices of ACA image background pixels
_ROWB = np.array([0, 0, 0, 0, 7, 7, 7, 7])
_COLB = np.array([0, 1, 6, 7, 0, 1, 6, 7])


def table_to_html(tbl):
    """
    Make an HTML representation of a table

    :param tbl: astropy Table
    :returns: str
    """
    out = tbl._base_repr_(
        html=True,
        max_width=-1,
        show_dtype=False,
        descr_vals=[],
        max_lines=-1,
        tableclass="table-striped",
    )
    # Undo HTML sanitizing to allow raw HTML in table elements
    out = re.sub(r"&quot;", '"', out)
    out = re.sub(r"&lt;", "<", out)
    out = re.sub(r"&gt;", ">", out)

    return out


def _get_bgd(img):
    """
    Helper function to compute the mean of the two largest values in
    the background pixels.  This is a very conservative upper limit and
    imagines that there are other warm/hot pixels that allow the top-2
    seen here to not get rejected in the flight background algorithm.
    """
    bgds = np.sort(np.array(img)[_ROWB, _COLB])
    bgd = (bgds[6] + bgds[7]) / 2
    return bgd


def calc_spoiler_impact(star, stars, dark_bgd=40, debug=False):
    """
    Calculate the centroid shift and relative image brightness change
    for ``star`` when nearby ``stars`` are included.  The assumption
    is that ``star`` has ASPQ1 > 0 so it is known to have some spoiler.

    The returned norm_frac is the ratio of the summed (mouse-bitten)
    counts in the original image over the spoiled image, so a value less
    than 1.0 is a problem since it indicates a high background.

    :param star: object with row, col, mag keys/cols
    :param stars: StarsTable
    :returns: d_yang, d_zang, norm_frac
    """
    row = star["row"]
    col = star["col"]
    mag = star["mag"]

    s_rows = stars["row"]
    s_cols = stars["col"]
    s_mags = stars["mag"]

    # Find potential spoilers with centroid within a 9-pixel halfw box
    ok = (
        (np.abs(s_rows - row) < 9)
        & (np.abs(s_cols - col) < 9)
        & (star["id"] != stars["id"])
    )
    if not np.any(ok):
        return 0.0, 0.0, 1.0

    # Make an image of the candidate star
    norm = mag_to_count_rate(mag)
    img = APL.get_psf_image(row=row, col=col, norm=norm, pix_zero_loc="edge")
    img += dark_bgd

    # Centroid the candidate star image, using an upper-limit to the
    # flight background algorithm.
    bgd = _get_bgd(img)
    row0, col0, norm0 = img.centroid_fm(bgd=bgd)

    # Select spoilers
    s_rows = s_rows[ok]
    s_cols = s_cols[ok]
    s_mags = s_mags[ok]

    # If there are multiple spoilers, put them all in the same quadrant
    # to guard against canceling centroid offsets.
    s_rows = np.sign(s_rows[0]) * np.abs(s_rows)
    s_cols = np.sign(s_cols[0]) * np.abs(s_cols)

    # Shine spoiler stars onto image
    for s_row, s_col, s_mag in zip(s_rows, s_cols, s_mags):
        s_norm = mag_to_count_rate(s_mag)
        s_img = APL.get_psf_image(
            row=s_row, col=s_col, norm=s_norm, pix_zero_loc="edge"
        )
        img += s_img.aca

    # Centroid the candidate + spoilers image.  This might raise an exception
    # if the background is high enough to result in a negative norm.  In that
    # case return values that will certainly exceed any spoiler thresholds.
    bgd = _get_bgd(img)
    if debug:
        print("bgd", bgd, count_rate_to_mag(bgd * 32))
        print(repr(img))
    try:
        row1, col1, norm1 = img.centroid_fm(bgd=bgd)
    except ValueError as err:
        if "non-positive image norm" in str(err):
            return 99, 99, -99
        else:
            # Some other code other, not expected
            raise

    # Final results
    dy = (row1 - row0) * 5
    dz = (col1 - col0) * 5
    frac_norm = norm1 / norm0

    return dy, dz, frac_norm


@functools.lru_cache(maxsize=8)
def get_starcheck_catalog(obsid):
    from mica.starcheck import get_starcheck_catalog

    return get_starcheck_catalog(obsid)


class ACABox:
    def __init__(self, size=None):
        """
        Class to represent a box half width amplitude.  Can be initialized with
        either a two-element sequence (y, z) or a scalar (which applies
        to both y and z.

        :param size: scalar or 2-element sequence (y, z), (arcsec, default=20)

        """
        if isinstance(size, ACABox):
            self.y = size.y
            self.z = size.z
            return

        try:
            assert len(size) == 2
        except TypeError:
            # len(scalar) raises TypeError
            if size is None:
                size = 20
            self.y = size
            self.z = size
        except AssertionError:
            # Has a length but it is not 2
            raise ValueError(
                "size arg must be either a scalar or a two-element sequence"
            )
        else:
            self.y = size[0]
            self.z = size[1]

    @property
    def row(self):
        return self.y / 5

    @property
    def col(self):
        return self.z / 5

    def max(self):
        return max(self.y, self.z)

    def __eq__(self, other):
        if not isinstance(other, ACABox):
            other = ACABox(other)
        return self.y == other.y and self.z == other.z

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        """
        Returns True if either component is larger.  This is somewhat
        specific to how boxes are used in intruder processing.
        """
        if not isinstance(other, ACABox):
            other = ACABox(other)
        return self.y > other.y or self.z > other.z

    def __add__(self, other):
        if not isinstance(other, ACABox):
            other = ACABox(other)
        return ACABox((self.y + other.y, self.z + other.z))

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return f"<ACABox y={self.y} z={self.z}>"


class AliasAttribute:
    """Descriptor to define class attributes which can be accessed via
    either <name> or <name>_<subclass>.

    For instance the ``dither`` attribute is found in both AcqTable and
    GuideTable, but in general these can be different for the two cases.

    The ``get_aca_catalog()`` function handles this by accepting ``dither_acq``
    and ``dither_guide``.  In order to be able to generically pass the
    same commmon kwargs to ``get_acq_catalog`` and ``get_guide_catalog``,
    we define the AliasAttribute to allow set/get of either ``dither``
    or ``dither_acq`` (or ``dither_guide``).

    The <subclass> name is the lower case version of everything before
    ``Table`` in the subclass name, so GuideTable => 'guide'.
    """

    def __get__(self, instance, owner):
        if instance is None:
            # When called without an instance, return self to allow access
            # to descriptor attributes.
            return self
        else:
            return getattr(instance, self.alias)

    def __set__(self, instance, value):
        setattr(instance, self.alias, value)

    def __set_name__(self, owner, name):
        if owner.__name__.endswith("Table"):
            self.alias = name + "_" + owner.__name__[:-5].lower()
            owner.allowed_kwargs.add(name)
        else:
            raise ValueError("can only be used in classes named *Table")

    def __repr__(self):
        return f"<{self.__class__.__name__} alias={self.alias}"


class MetaAttribute:
    def __init__(self, default=None, is_kwarg=True, pickle=True):
        """
        Descriptor to define ACACatalogTable class attributes which get stored in
        the table meta dict and have defined defaults.  Also includes a flag to
        specify whether the attribute will be included when pickling.  Some
        attributes (like dark current map) are too big and not necessary.

        :param default: default value
        :param is_kwarg: include in list of allow kwargs to class init
        :param pickle: store correspoding meta attribute in pickle

        """
        self.default = default
        self.is_kwarg = is_kwarg
        self.pickle = pickle

    def __get__(self, instance, owner):
        try:
            return instance.meta[self.name]
        except AttributeError:
            # When called without an instance, return self to allow access
            # to descriptor attributes.
            # AttributeError: 'NoneType' object has no attribute 'meta'
            return self
        except KeyError:
            if self.default is not None:
                instance.meta[self.name] = copy(self.default)
            return instance.meta.get(self.name)

    def __set__(self, instance, value):
        instance.meta[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name
        if self.is_kwarg:
            owner.allowed_kwargs.add(name)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} name={self.name} default={self.default} "
            f"pickle={self.pickle}>"
        )


class MonitorsMetaAttribute(MetaAttribute):
    def __set__(self, instance, value):
        if isinstance(value, Table):
            colnames = ["coord0", "coord1", "coord_type", "mag", "function"]
            if not set(colnames) <= set(value.colnames):
                raise ValueError(
                    f'monitors input table must have {", ".join(colnames)} columns'
                )
        else:
            value = np.asarray(value)
            if value.ndim != 2 or value.shape[1] != 5:
                raise ValueError("monitors input must have shape (N, 5)")
            out = Table()
            out["coord0"] = value[:, 0].astype(np.float64)
            out["coord1"] = value[:, 1].astype(np.float64)
            out["coord_type"] = value[:, 2].astype(np.uint8)
            out["mag"] = value[:, 3].astype(np.float64)
            out["function"] = value[:, 4].astype(np.uint8)
            value = out

            if np.any(out["coord_type"] < 0) or np.any(out["coord_type"] > 2):
                raise ValueError("monitors coord type must be 0, 1, or 2")

            if np.any(out["function"] < 0) or np.any(out["function"] > 3):
                raise ValueError("monitors function must be 0, 1, 2, or 3")

        instance.meta[self.name] = value


class QuatMetaAttribute(MetaAttribute):
    def __set__(self, instance, value):
        if not isinstance(value, Quat):
            value = Quat(value)
        instance.meta[self.name] = value


class IntMetaAttribute(MetaAttribute):
    def __set__(self, instance, value):
        instance.meta[self.name] = int(value)


class IntListMetaAttribute(MetaAttribute):
    def __set__(self, instance, value):
        instance.meta[self.name] = np.atleast_1d(value).astype(np.int64).tolist()


class BaseCatalogTable(Table):
    """Base class for representing star catalogs or ACA (guide/acq/fid) catalogs.

    Features:
    - Inherits from astropy Table
    - Predefined formatting for nicer representations
    - Indexing on 'id' column

    """

    # Catalog type when plotting (None | 'FID' | 'ACQ' | 'GUI')
    catalog_type = None

    def __init__(self, data=None, **kwargs):
        super().__init__(data=data, **kwargs)

        # Make printed table look nicer.  This is defined in advance
        # and will be applied the first time the table is represented.
        self._default_formats = {}
        formats = {
            ("p_acq",): "5.3f",  # 0.912
            ("yang", "zang"): "8.2f",  # -2500.12
            ("mag", "maxmag", "mag_err", "color", "COLOR1"): "5.2f",  # -1.12
            ("row", "col"): "7.2f",  # -500.12
            ("ra", "dec", "RA_PMCORR", "DEC_PMCORR"): "11.6f",  # -160.123561
            ("dec", "DEC_PMCORR"): "10.5f",  # -89.12345
            ("idx", "dim"): "2d",
            ("halfw",): "3d",
        }

        for names, format in formats.items():
            for name in names:
                self._default_formats[name] = format

    def __getstate__(self):
        """Set the pickle state from self.

        The main functionality here is munging ``meta`` to exclude any
        MetaAttributes that are marked for exclusion from pickling (e.g. stars,
        dark).

        """
        columns, meta = super().__getstate__()
        meta = meta.copy()
        cls = self.__class__

        # For everything in meta that is a MetaAttribute, check if it should
        # be pickled.
        pickle_excludes = []
        for attr in meta:
            descr = getattr(cls, attr, None)
            if isinstance(descr, MetaAttribute) and not descr.pickle:
                pickle_excludes.append(attr)

        for attr in pickle_excludes:
            del meta[attr]

        return columns, meta

    def __setstate__(self, state):
        """Restore object from pickle state.

        This fixes an upstream issue in astropy.table (as of 3.1) where the
        Table __setstate__ does ``self.__init__(columns, meta)``, which makes a
        deepcopy of ``meta``.  That is inefficient but more importantly does
        not preserve the ``acqs`` weakref in cand_acqs['probs'].

        """
        columns, meta = state
        self.__init__(columns)
        self.meta.update(meta)

    def make_index(self):
        # Low-tech index to quickly get a row or the row index by `id` column.
        self._id_index = {}
        self._id_index_mon = {}

        has_type = "type" in self.colnames
        for idx, row in enumerate(self):
            if has_type and row["type"] == "MON":
                self._id_index_mon[row["id"]] = idx
            else:
                self._id_index[row["id"]] = idx

    def get_id(self, id, mon=False):
        """
        Return row corresponding to ``id`` in id column.

        :param id: int
            row ``id`` column value
        :param mon: bool
            return ID for corresponding MON star (default=False)
        :returns: table Row
        """
        return self[self.get_id_idx(id, mon)]

    def get_id_idx(self, id, mon=False):
        """
        Return row corresponding to ``id`` in id column.

        :param id: int
            row ``id`` column value
        :param mon: bool
            return ID for corresponding MON star (default=False)
        :returns: table row index (int)
        """
        if not hasattr(self, "_id_index"):
            self.make_index()

        try:
            idx = (self._id_index_mon if mon else self._id_index)[id]
            assert self["id"][idx] == id
        except (KeyError, IndexError, AssertionError):
            self.make_index()
            try:
                idx = (self._id_index_mon if mon else self._id_index)[id]
                assert self["id"][idx] == id
            except (KeyError, IndexError, AssertionError):
                raise KeyError(f"{id} is not in table")

        return idx

    def get_candidates_mask(self, stars):
        """Return a boolean mask indicating which ``stars`` are acceptable candidates
        for the parent class.  This basically applies to guide and acq, though
        ACACatalog uses a hybrid of those two.

        This is used directly in selection of candidates.  For the base class all
        stars are considered acceptable.

        :param stars: StarsTable of stars
        :returns: bool mask

        """
        return np.ones(len(stars), dtype=bool)

    @property
    def bad_stars_mask(self):
        """Mask of stars that are not acceptable candidates for the parent class.  This
        basically applies to guide and acq, though ACACatalog uses a hybrid of
        those two.

        This is mostly used for plotting, corresponding to the ``bad_stars``
        arg of plot_stars where bad stars are plotting in a distinctive color.

        """
        if not hasattr(self, "_bad_stars_mask"):
            self._bad_stars_mask = ~self.get_candidates_mask(self.stars)
        return self._bad_stars_mask

    @bad_stars_mask.setter
    def bad_stars_mask(self, value):
        self._bad_stars_mask = value

    @property
    def bad_stars(self):
        """Back-compatibility property for bad_stars_mask"""
        return self.bad_stars_mask

    def plot(self, ax=None, **kwargs):
        """
        Plot the catalog and background stars.

        :param ax: matplotlib axes object for plotting to (optional)
        :param kwargs: other keyword args for plot_stars
        """
        import matplotlib.pyplot as plt
        from chandra_aca.plot import plot_stars

        kwargs.setdefault("stars", np.array([]))

        # Explicitly set bad_stars so that plot_stars does not apply its
        # internal version (which is no lot always the right answer).
        kwargs.setdefault("bad_stars", np.zeros(len(kwargs["stars"]), dtype=bool))

        fig = plot_stars(attitude=self.att, ax=ax, **kwargs)
        if "agg" not in plt.get_backend().lower():
            plt.show()

        return fig


class ACACatalogTable(BaseCatalogTable):
    """
    Base class for representing ACA catalogs in star selection.  This
    can apply to acq, guide and fid entries.

    Features:
    - Inherits from astropy Table
    - Logging
    - Serialization to/from pickle
    - Predefined formatting for nicer representations
    """

    # Name of table.  Use to define default file names where applicable.
    # Should be set by subclass, e.g. ``name = 'acqs'`` for AcqTable.
    name = "aca_cat"

    # Catalog attributes, gets set in MetaAttribute or AliasAttribute
    allowed_kwargs = set(["dark"])

    required_attrs = ("dither_acq", "dither_guide", "date")

    obsid = MetaAttribute(default=0)
    att = QuatMetaAttribute()
    n_acq = MetaAttribute(default=8)
    n_guide = MetaAttribute()
    n_fid = MetaAttribute()
    monitors = MonitorsMetaAttribute()
    man_angle = MetaAttribute()
    man_angle_next = MetaAttribute(default=180.0)
    t_ccd_acq = MetaAttribute()
    t_ccd_guide = MetaAttribute()
    date = MetaAttribute()
    dark_date = MetaAttribute()
    dither_acq = MetaAttribute()
    dither_guide = MetaAttribute()
    detector = MetaAttribute()
    sim_offset = MetaAttribute()
    focus_offset = MetaAttribute()
    target_offset = MetaAttribute(default=(0.0, 0.0))
    dyn_bgd_n_faint = MetaAttribute(default=0)
    dyn_bgd_dt_ccd = MetaAttribute(default=-4.0)
    stars = MetaAttribute(pickle=False)
    include_ids_acq = IntListMetaAttribute(default=[])
    include_halfws_acq = IntListMetaAttribute(default=[])
    exclude_ids_acq = IntListMetaAttribute(default=[])
    include_ids_guide = IntListMetaAttribute(default=[])
    exclude_ids_guide = IntListMetaAttribute(default=[])
    include_ids_fid = IntListMetaAttribute(default=[])
    exclude_ids_fid = IntListMetaAttribute(default=[])
    optimize = MetaAttribute(default=True)
    verbose = MetaAttribute(default=False)
    print_log = MetaAttribute(default=False)
    log_info = MetaAttribute(default={}, is_kwarg=False)

    @property
    def dark(self):
        if hasattr(self, "_dark"):
            return self._dark

        # If self.dark_date is already set (most likely after unpickling an existing
        # catalog), then use that instead of the observation date for getting the dark map.
        if self.dark_date:
            dark_id = get_dark_cal_id(date=self.dark_date, select="nearest")
            self.log(
                f"Found dark cal image {dark_id} nearest self.dark_date={self.dark_date}"
            )

            # Warn if the pickled dark_date is not the same as we would have gotten based
            # on the observation date.
            dark_id_for_date = get_dark_cal_id(date=self.date, select="before")
            if dark_id != dark_id_for_date:
                warnings.warn(
                    f"Unexpected dark_date: "
                    f"dark_id nearest dark_date={self.dark_date} is {dark_id} "
                    f"but dark_id before obs date={self.date} is {dark_id_for_date}"
                )
        else:
            dark_id = get_dark_cal_id(date=self.date, select="before")
            self.log(f"Found dark cal image {dark_id} before self.date={self.date}")

        # In all cases set self.dark_date to be the actual date of the dark cal
        # we are using, not what might have been set on input.
        self.dark_date = dark_id[:4] + ":" + dark_id[4:]

        # Dark current map handling.  If asking for `dark` attribute without having set
        # it then auto-fetch from mica.  Note that get_dark_cal_image caches returned values
        # using LRU cache on all params, so this is efficient for the different
        # catalog tables.
        self.log(
            f"Getting dark cal image at date={self.dark_date} t_ccd={self.t_ccd:.1f} "
        )
        self.dark = get_dark_cal_image(
            date=self.dark_date, select="nearest", t_ccd_ref=self.t_ccd, aca_image=False
        )

        return self._dark

    @dark.setter
    def dark(self, value):
        assert value.shape == (1024, 1024)
        self._dark = value

        # Set pixel regions from ACA.bad_pixels to have acqs.dark=700000 (5.0 mag
        # star) per pixel.
        for r0, r1, c0, c1 in ACA.bad_pixels:
            self._dark[
                r0 + 512 : r1 + 513, c0 + 512 : c1 + 513
            ] = ACA.bad_pixel_dark_current

    def set_attrs_from_kwargs(self, **kwargs):
        for name, val in kwargs.items():
            # If name is an allowed keyword arg or a settable property on the class
            # then set now.
            if name in self.allowed_kwargs or inspect.isdatadescriptor(
                getattr(self.__class__, name, None)
            ):
                setattr(self, name, val)
            else:
                raise ValueError(f'unexpected keyword argument "{name}"')

        # If an explicit obsid is not provided to all getting parameters via mica
        # then all other params must be supplied.
        missing = [attr for attr in self.required_attrs if getattr(self, attr) is None]
        if self.obsid == 0 and missing:
            raise ValueError(f"missing required parameters {missing}")

        # If not all params supplied then get via mica for the obsid.
        if missing:
            self.log(f"getting starcheck catalog for obsid {self.obsid}")

            try:
                obs = get_starcheck_catalog(self.obsid)
                obso = obs["obs"]
            except Exception:
                # Probably AttributeError for a non-existent obsid, where
                # get_starcheck_catalog returns None. But catch anything in case
                # something else happened. Exception chaining will help here.
                raise ValueError(
                    f"missing required parameters {missing} and "
                    f"could not fill them in from obsid {self.obsid}"
                )

            if self.att is None:
                self.att = [obso["point_ra"], obso["point_dec"], obso["point_roll"]]
            if self.date is None:
                self.date = obso["mp_starcat_time"]
            if self.t_ccd is None:
                self.t_ccd = obs.get("pred_temp", -15.0)
            if self.man_angle is None:
                self.man_angle = obs["manvr"]["angle_deg"][0]
            if self.detector is None:
                self.detector = obso["sci_instr"]
            if self.sim_offset is None:
                self.sim_offset = obso["sim_z_offset_steps"]
            if self.focus_offset is None:
                self.focus_offset = 0

            if self.n_fid is None:
                self.n_fid = np.count_nonzero(obs["cat"]["type"] == "FID")

            n_mon = np.count_nonzero(obs["cat"]["type"] == "MON")
            if n_mon > 0:
                warnings.warn(f"found {n_mon} MON stars in catalog, ignoring them")

            if self.n_guide is None:
                # Replicate the number of guide stars in the original catalog.
                self.n_guide = 8 - self.n_fid - n_mon

            if self.detector is None:
                self.n_fid = 0
                self.detector = "HRC-S"

            for dither_attr in ("dither_acq", "dither_guide"):
                if getattr(self, dither_attr) is None:
                    dither_y_amp = obso.get("dither_y_amp")
                    dither_z_amp = obso.get("dither_z_amp")
                    if dither_y_amp is not None and dither_z_amp is not None:
                        setattr(self, dither_attr, ACABox((dither_y_amp, dither_z_amp)))

                        # Special rule for handling big dither from mica.starcheck,
                        # which does not yet know about dither_acq vs. dither_guide.
                        # Use the most common dither size for ACIS / HRC.
                        if dither_attr == "dither_acq" and self.dither_acq.max() > 30:
                            dither = 8 if self.detector.startswith("ACIS") else 20
                            self.dither_acq = ACABox(dither)

            self.starcheck_catalog = obs

        for dither_attr in ("dither_acq", "dither_guide"):
            dither = getattr(self, dither_attr)
            if not isinstance(dither, ACABox):
                setattr(self, dither_attr, ACABox(dither))

    def set_stars(self, acqs=None, filter_near_fov=True):
        """Set the object ``stars`` attribute to an appropriate StarsTable object.

        If ``acqs`` is defined that will be a previously computed AcqTable with
        ``stars`` already available, so use that.

        :param acqs: AcqTable for same observation
        :param filter_near_fov: include only stars in or near ACA FOV
        """
        if acqs is None:
            if self.stars is None:
                stars = StarsTable.from_agasc(self.att, date=self.date, logger=self.log)
            else:
                stars = StarsTable.from_stars(self.att, self.stars, logger=self.log)
        else:
            stars = acqs.stars

        if filter_near_fov:
            ok = (np.abs(stars["row"]) < ACA.CCD["row_max"] + ACA.CCD["fov_pad"]) & (
                np.abs(stars["col"]) < ACA.CCD["col_max"] + ACA.CCD["fov_pad"]
            )
            if not np.all(ok):
                stars = stars[ok]

        self.stars = stars

    def get_catalog_for_plot(self):
        """Return value of ``catalog`` for plot_stars() call for this object

        For ACACatalogTable this is the self Table object but with the possibility
        of changing all the `type` column values to self.catalog_type (e.g. 'ACQ').

        """
        if self.catalog_type:
            self["type"] = self.catalog_type
        catalog = self.as_array()
        return catalog

    def plot(self, ax=None, **kwargs):
        """
        Plot the catalog and background stars.

        :param ax: matplotlib axes object for plotting to (optional)
        :param kwargs: other keyword args for plot_stars
        """
        kwargs.setdefault("catalog", self.get_catalog_for_plot())
        if self.stars is None:
            self.set_stars()
        kwargs.setdefault("stars", self.stars)
        kwargs.setdefault("bad_stars", self.bad_stars_mask)
        if self.date is not None:
            kwargs.setdefault("starcat_time", self.date)
        if getattr(self, "duration", None) is not None:
            kwargs.setdefault("duration", self.duration)
        return super().plot(ax, **kwargs)

    @property
    def dither(self):
        return None

    @dither.setter
    def dither(self, value):
        self.dither_acq = value
        self.dither_guide = value

    @property
    def t_ccd(self):
        # Subclasses must implement.
        raise NotImplementedError()

    @t_ccd.setter
    def t_ccd(self, value):
        raise NotImplementedError()

    @classmethod
    def empty(cls):
        """
        Return a minimal ACACatalogTable which satisfies API requirements.  Currently
        this means that it has an 'id' column which can be examined for length.

        :returns: StarsTable of stars (empty)
        """
        out = cls()
        out["id"] = np.full(fill_value=0, shape=(0,), dtype=np.int64)
        out["idx"] = np.full(fill_value=0, shape=(0,), dtype=np.int64)
        return out

    @property
    def acqs(self):
        return self.meta.get("acqs")

    @acqs.setter
    def acqs(self, val):
        self.meta["acqs"] = val

    @property
    def fids(self):
        return self.meta.get("fids")

    @fids.setter
    def fids(self, val):
        self.meta["fids"] = val

    @property
    def guides(self):
        return self.meta.get("guides")

    @guides.setter
    def guides(self, val):
        self.meta["guides"] = val

    @property
    def mons(self):
        return self.meta.get("mons")

    @mons.setter
    def mons(self, val):
        self.meta["mons"] = val

    def process_include_ids(self, cand_stars, stars):
        """Ensure that the candidate acqs/guides table has stars that were forced to be included.

        Updates ``cand_stars`` in place.

        :param cand_stars: candidate acquisition or guide stars table
        :param stars: stars table

        """
        for include_id in self.include_ids:
            if include_id not in cand_stars["id"]:
                try:
                    star = stars.get_id(include_id)
                except KeyError:
                    raise ValueError(
                        f"cannot include star id={include_id} that is not "
                        f"a valid star in the ACA field of view"
                    )
                else:
                    cand_stars.add_row(star)
                    self.log(f"Included star id={include_id} in candidates table")

    def log(self, data, **kwargs):
        """
        Add a log event to self.log_info['events'] include ``data`` (which
        is typically, but not required to be, a string).
        """
        # Name of calling functions, starting from top (outermost) and
        # ending with function that called log()
        func = inspect.currentframe().f_back.f_code.co_name

        # Possible extension later to include calling context, but this
        # currently includes a whole bunch more, need to filter to just
        # this module.
        # framerecs = inspect.stack()[1:-1]
        # funcs = [framerec[3] for framerec in reversed(framerecs)]
        # func = funcs[-1]
        log_info = self.log_info

        tm = time.time()
        dt = tm - log_info.setdefault("unix_run_time", tm)

        kwargs = {key: to_python(val) for key, val in kwargs.items()}
        event = dict(dt=round(dt, 4), func=func, data=data, **kwargs)
        if event.get("warning") and isinstance(data, str):
            event["data"] = "WARNING: " + event["data"]

        if "events" not in log_info:
            log_info["events"] = []
        log_info["events"].append(event)

        if self.print_log:
            data_str = " " * event.get("level", 0) * 4 + str(event["data"])
            print(f"{dt:7.3f} {func:20s}: {data_str}")

    @property
    def warnings(self):
        """
        Return list of warnings in the event log.  This returns just the
        string message without context.
        """
        warns = [
            event["data"] for event in self.log_info["events"] if event.get("warning")
        ]
        return warns

    @property
    def exception(self):
        """
        Return string traceback of a top-level caught exception, or "" (empty
        string) if no exception has been caught.  Mostly for use in Matlab
        interface.
        """
        return self.meta.get("exception", "")

    @exception.setter
    def exception(self, val):
        self.meta["exception"] = val

    def _set_formats(self):
        # Apply default formats to applicable columns and delete
        # that default format so it doesn't get set again next time.
        names = self.colnames
        for name in list(self._default_formats.keys()):
            if name in names:
                self[name].format = self._default_formats.pop(name)

    def pformat(self, *args, **kwargs):
        self._set_formats()
        return super().pformat(*args, **kwargs)

    def pprint(self, *args, **kwargs):
        self._set_formats()
        return super().pprint(*args, **kwargs)

    def _base_repr_(self, *args, **kwargs):
        self._set_formats()

        names = [name for name in self.colnames if self[name].dtype.kind != "O"]

        tmp = self.__class__([self[name] for name in names], names=names, copy=False)
        return super(BaseCatalogTable, tmp)._base_repr_(*args, **kwargs)

    def to_pickle(self, rootdir="."):
        """
        Write the catalog table as pickle to:
          ``<rootdir>/obs<obsid>/<name>.pkl``

        :param rootdir: root directory (default='.')
        """
        rootdir = Path(rootdir)
        outdir = rootdir / f'obs{self.meta["obsid"]:05}'
        if not outdir.exists():
            outdir.mkdir()
        outfile = outdir / f"{self.name}.pkl"
        with open(outfile, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_pickle(cls, obsid, rootdir="."):
        """
        Construct table from pickle file in
          ``<rootdir>/obs<obsid>/<name>.pkl``

        :param obsid: Obsid
        :param rootdir: root directory (default='.')
        :returns: catalog table
        """
        rootdir = Path(rootdir)
        infile = rootdir / f"obs{obsid:05}" / f"{cls.name}.pkl"
        with open(infile, "rb") as fh:
            return pickle.load(fh)

    def has_column_spoiler(self, cand, stars, mask=None):
        """
        Returns True if ``cand`` has a column spoiler downstream toward readout
        register which is:
        - at least 4.5 mag brighter (ACA.col_spoiler_mag_diff)
        - within 10 pixels in column (ACA.col_spoiler_pix_sep)

        Also records a log entry for detected spoilers

        :param cand: object with 'row', 'col', and 'mag' items
        :param stars: StarsTable
        :param mask: boolean mask of stars to check (default=None => no mask)
        :returns: bool
        """
        if mask is None:
            mask = ()  # No mask

        # Star further from readout register than candidate
        # AND on same side of CCD
        # AND within column separation limit
        # AND within mag limit
        bads = (
            (np.abs(cand["row"]) < abs(stars["row"][mask]))
            & (np.sign(cand["row"]) == np.sign(stars["row"][mask]))
            & (np.abs(cand["col"] - stars["col"][mask]) < ACA.col_spoiler_pix_sep)
            & (cand["mag"] - stars["mag"][mask] > ACA.col_spoiler_mag_diff)
        )

        if np.any(bads):
            self.log(
                f'Candidate object id={cand["id"]} rejected due to column spoiler(s) '
                f'{stars["id"][mask][bads].tolist()}',
                id=cand["id"],
            )
            return True
        else:
            return False


# AGASC columns not needed (at this moment) for acq star selection.
# Change as needed.
AGASC_COLS_DROP = [
    "RA",
    "DEC",
    "POS_CATID",
    "EPOCH",
    "PM_CATID",
    "PLX",
    "PLX_ERR",
    "PLX_CATID",
    "MAG",
    "MAG_ERR",
    "MAG_BAND",
    "MAG_CATID",
    "C1_CATID",
    "COLOR2",
    "COLOR2_ERR",
    "C2_CATID",
    "RSV1",
    "RSV2",
    "RSV3",
    "VAR_CATID",
    "ACQQ1",
    "ACQQ2",
    "ACQQ3",
    "ACQQ4",
    "ACQQ5",
    "ACQQ6",
    "XREF_ID1",
    "XREF_ID2",
    "XREF_ID3",
    "XREF_ID4",
    "XREF_ID5",
    "RSV4",
    "RSV5",
    "RSV6",
]

# Define a function that returns the observed standard deviation in ACA mag
# as a function of catalog mag.  This is about the std-dev for samples
# within an observation, NOT the error on the catalog mag (which is about
# the error on mean of the sample, not the std-dev of the sample).
#
# See ASPECT/ipynb/star_selection/star-mag-std-dev.ipynb for the source of numbers.
#
# Note that get_mag_std is a function that is called with get_mag_std(mag_aca).
get_mag_std = interp1d(
    x=[-10, 6.7, 7.3, 7.8, 8.3, 8.8, 9.2, 9.7, 10.1, 11, 20],  # mag
    y=[
        0.015,
        0.015,
        0.022,
        0.029,
        0.038,
        0.055,
        0.077,
        0.109,
        0.141,
        0.23,
        1.0,
    ],  # std-dev
    kind="linear",
)


class StarsTable(BaseCatalogTable):
    """Table of stars for use in proseco.

    This is meant to be created only with available class methods:

    - from_agasc() : calls agasc.get_agasc_cone()
    - from_stars() : init from the results of a call to get_agasc_cone()
    - from_agasc_ids() : calls agasc.get_stars()
    - empty() : empty catalog, then use add_* methods to add stars

    """

    att = QuatMetaAttribute()

    # StarsTable attributes, gets set in MetaAttribute or AliasAttribute
    allowed_kwargs = set()

    @staticmethod
    def get_logger(logger):
        if logger is not None:
            return logger
        else:

            def null_logger(*args, **kwargs):
                pass

            return null_logger

    def get_catalog_for_plot(self):
        """Return value of ``catalog`` for plot_stars() call for this object

        For StarsTable this is None (no catalog).
        """
        return None

    def plot(self, ax=None, **kwargs):
        """
        Plot the star field.

        :param ax: matplotlib axes object for plotting to (optional)
        :param kwargs: other keywords for plot_stars
        """
        kwargs.setdefault("stars", self)
        return super().plot(ax, **kwargs)

    @classmethod
    def from_agasc(cls, att, date=None, radius=1.2, logger=None):
        """
        Get AGASC stars in the ACA FOV.  This uses the mini-AGASC, so only stars
        within 3-sigma of 11.5 mag.

        :param att: any Quat-compatible attitude
        :param date: DateTime compatible date for star proper motion (default=NOW)
        :param radius: star cone radius [deg] (default=1.2)
        :param logger: logger object (default=None)

        :returns: StarsTable of stars
        """
        q_att = Quat(att)
        agasc_file = Path(os.environ["SKA"], "data", "agasc", "proseco_agasc_1p7.h5")
        agasc_stars = agasc.get_agasc_cone(
            q_att.ra, q_att.dec, radius=radius, date=date, agasc_file=agasc_file
        )
        stars = StarsTable.from_stars(att, agasc_stars, copy=False)

        logger = StarsTable.get_logger(logger)
        logger(
            f"Got {len(stars)} stars from AGASC at "
            f"ra={q_att.ra:.5f} dec={q_att.dec:.4f}",
            level=1,
        )

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
                raise ValueError(f"failed to get AGASC ID={agasc_id}")
            else:
                agasc_stars.append(star)
        agasc_stars = Table(rows=agasc_stars, names=agasc_stars[0].colnames)
        return StarsTable.from_stars(att, stars=agasc_stars)

    @classmethod
    def from_stars(cls, att, stars, logger=None, copy=True):
        """
        Return a StarsTable from an existing AGASC stars query.  This just updates
        columns in place.

        If ``stars`` is a StarsTable, the attitude of that object ``stars.att`` must
        match ``att`` to within 0.001 arcsec in pitch, yaw, and roll.

        :param att: any Quat-compatible attitude
        :param stars: Table of stars
        :param logger: logger object (default=None)
        :param copy: copy ``stars`` table columns

        :returns: StarsTable of stars

        """
        logger = StarsTable.get_logger(logger)

        q_att = Quat(att)

        if isinstance(stars, StarsTable):
            # Check for consistency between stars att and supplied att
            # if att is not the same object as stars.att.
            if att is not stars.att:
                q_att_stars = Quat(stars.att)
                dq = q_att.dq(q_att_stars)
                lim = 0.001 / 3600  # 0.001 arcsec
                if any(
                    abs(getattr(dq, attr)) > lim for attr in ("pitch", "yaw", "roll0")
                ):
                    raise ValueError(
                        f"supplied att {att} does not match stars att {stars.att}"
                    )
            return stars

        stars = cls(stars, copy=copy)
        stars.att = att

        logger(f"Updating star columns for attitude and convenience")

        stars.meta["q_att"] = q_att
        yag, zag = radec_to_yagzag(stars["RA_PMCORR"], stars["DEC_PMCORR"], q_att)
        row, col = yagzag_to_pixels(yag, zag, allow_bad=True, pix_zero_loc="edge")

        stars.remove_columns(
            [name for name in AGASC_COLS_DROP if name in stars.colnames]
        )

        stars.rename_column("AGASC_ID", "id")
        stars.add_column(Column(stars["RA_PMCORR"], name="ra"), index=1)
        stars.add_column(Column(stars["DEC_PMCORR"], name="dec"), index=2)
        stars.add_column(Column(yag, name="yang"), index=3)
        stars.add_column(Column(zag, name="zang"), index=4)
        stars.add_column(Column(row, name="row"), index=5)
        stars.add_column(Column(col, name="col"), index=6)

        stars.add_column(
            Column(stars["MAG_ACA"], name="mag"), index=7
        )  # For convenience

        # Mag_err column is the RSS of the catalog mag err (i.e. systematic error in
        # the true star mag) and the sample uncertainty in the ACA readout mag
        # for a star with mag=MAG_ACA.  The latter typically dominates above 9th mag.
        mag_aca_err = stars["MAG_ACA_ERR"] * 0.01

        # For color=1.5 stars set a lower limit of 0.3 on the catalog mag error.
        # This is based on analysis in ipynb/star_selection/star-mag-std-dev.ipynb
        # which shows that for color=1.5 stars the observed mag error is
        # does not correlate well with mag_err below around 0.3 and so setting a
        # floor of 0.3 is a reasonable way to capture the uncertainty.
        ok = stars["COLOR1"] == 1.5
        mag_aca_err[ok] = mag_aca_err[ok].clip(0.3)

        mag_std_dev = get_mag_std(stars["MAG_ACA"])
        mag_err = np.sqrt(mag_aca_err**2 + mag_std_dev**2)
        stars.add_column(Column(mag_err, name="mag_err"), index=8)

        logger("Finished star processing", level=1)

        return stars

    @classmethod
    def empty(cls, att=(0, 0, 0)):
        """
        Return an empty StarsTable suitable for generating synthetic tables.

        :param att: any Quat-compatible attitude
        :returns: StarsTable of stars (empty)
        """
        stars = cls.from_agasc(att, radius=-1)
        stars.att = att

        return stars

    def add_agasc_id(self, agasc_id):
        """
        Add a AGASC star to the current StarsTable.

        :param agasc_id: AGASC ID of the star to add
        """
        stars = StarsTable.from_agasc_ids(self.meta["q_att"], [agasc_id])
        self.add_row(stars[0])

    def add_fake_constellation(self, n_stars=8, size=1500, mag=7.0, **attrs):
        r"""
        Add a fake constellation of up to 8 stars consisting of a cross and square::

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
        :param \**attrs: other star table attributes
        """
        if n_stars > 8:
            raise ValueError("max value of n_stars is 8")

        ids = len(self) + 100 + np.arange(n_stars)
        yangs = (
            np.array([1, 0, -1, 0, 0.5, 0.5, -0.5, -0.5][:n_stars], dtype=np.float64)
            * size
        )
        zangs = (
            np.array([0, 1, 0, -1, 0.5, -0.5, 0.5, -0.5][:n_stars], dtype=np.float64)
            * size
        )

        arrays = [ids, yangs, zangs, mag]
        names = ["id", "yang", "zang", "mag"]
        for name, array in attrs.items():
            names.append(name)
            arrays.append(array)

        arrays = np.broadcast_arrays(*arrays)
        for vals in zip(*arrays):
            self.add_fake_star(**{name: val for name, val in zip(names, vals)})

    def add_fake_star(self, **star):
        r"""
        Add a star to the current StarsTable.

        The input kwargs must have at least:
        - One of yang/zang, ra/dec, or row/col
        - mag
        - mag_err (defaults to 0.1)

        Mag_err is set to 0.1 if not provided.  Yang/zang, ra/dec, and row/col
        RA/DEC_PMCORR, MAG_ACA, MAG_ACA_ERR will all be set according to the
        primary inputs unless explicitly provided.  All the rest will be set
        to default "good"" values that will preclude initial exclusion of the star.

        :param \**star: keyword arg attributes corresponding to StarTable columns
        """
        names = [
            "id",
            "ra",
            "dec",
            "yang",
            "zang",
            "row",
            "col",
            "mag",
            "mag_err",
            "POS_ERR",
            "PM_RA",
            "PM_DEC",
            "MAG_ACA",
            "MAG_ACA_ERR",
            "CLASS",
            "COLOR1",
            "COLOR1_ERR",
            "VAR",
            "ASPQ1",
            "ASPQ2",
            "ASPQ3",
            "RA_PMCORR",
            "DEC_PMCORR",
        ]

        defaults = {"id": len(self) + 100, "mag_err": 0.1, "VAR": -9999}
        out = {
            name: star.get(name, defaults.get(name, 0))
            for name in names
            if name in self.colnames
        }

        q_att = self.meta["q_att"]
        if "ra" in star and "dec" in star:
            out["yang"], out["zang"] = radec_to_yagzag(out["ra"], out["dec"], q_att)
            out["row"], out["col"] = yagzag_to_pixels(
                out["yang"], out["zang"], allow_bad=True
            )

        elif "yang" in star and "zang" in star:
            out["ra"], out["dec"] = yagzag_to_radec(out["yang"], out["zang"], q_att)
            out["row"], out["col"] = yagzag_to_pixels(
                out["yang"], out["zang"], allow_bad=True
            )

        elif "row" in star and "col" in star:
            out["yang"], out["zang"] = pixels_to_yagzag(
                out["row"], out["col"], allow_bad=True
            )
            out["ra"], out["dec"] = yagzag_to_radec(out["yang"], out["zang"], q_att)

        reqd_names = ("ra", "dec", "row", "col", "yang", "zang", "mag", "mag_err")
        for name in reqd_names:
            if name not in out:
                raise ValueError(f"incomplete star data did not get {name} data")

        out["RA_PMCORR"] = out["ra"]
        out["DEC_PMCORR"] = out["dec"]
        out["MAG_ACA"] = out["mag"]

        # The fake_code attribute is used for monitor window handling to
        # indicate that this star has been added as a fake star so that guide
        # selection can select it as a force-include star. Subsequently the
        # code is used to infer the provenance of that star.
        if "fake_code" in star:
            if "fake_code" not in self.colnames:
                self["fake_code"] = 0
            out["fake_code"] = star["fake_code"]

        self.add_row(out)

    def add_fake_stars_from_fid(
        self,
        fid_id=1,
        offset_y=0,
        offset_z=0,
        mag=7.0,
        id=None,
        detector="ACIS-S",
        sim_offset=0,
    ):
        try:
            fids = FIDS_CACHE[detector, sim_offset]
        except KeyError:
            from .fid import get_fid_catalog

            fids = get_fid_catalog(
                att=(0, 0, 0),
                n_fid=3,
                detector=detector,
                sim_offset=sim_offset,
                focus_offset=0,
                t_ccd=-10.0,
                date="2018:001",
                dither=8.0,
            )
            FIDS_CACHE[detector, sim_offset] = fids

        arrays = np.broadcast_arrays(fid_id, offset_y, offset_z, mag, id)

        for fid_id, offset_y, offset_z, mag, id in zip(*arrays):
            fid = fids.cand_fids.get_id(fid_id)
            kwargs = dict(
                yang=fid["yang"] + offset_y, zang=fid["zang"] + offset_z, mag=mag
            )
            if id is not None:
                kwargs["id"] = id
            self.add_fake_star(**kwargs)


def bin2x2(arr):
    """Bin 2-d ``arr`` in 2x2 blocks.  Requires that ``arr`` has even shape sizes"""
    shape = (arr.shape[0] // 2, 2, arr.shape[1] // 2, 2)
    return arr.reshape(shape).sum(-1).sum(1)


RC_6x6 = np.array(
    [
        10,
        11,
        12,
        13,
        17,
        18,
        19,
        20,
        21,
        22,
        25,
        26,
        27,
        28,
        29,
        30,
        33,
        34,
        35,
        36,
        37,
        38,
        41,
        42,
        43,
        44,
        45,
        46,
        50,
        51,
        52,
        53,
    ],
    dtype=np.int64,
)

ROW_6x6 = (
    np.array(
        [
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
        ],
        dtype=np.float64,
    )
    + 0.5
)

COL_6x6 = (
    np.array(
        [
            2,
            3,
            4,
            5,
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
            5,
            6,
            2,
            3,
            4,
            5,
        ],
        dtype=np.float64,
    )
    + 0.5
)


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
    img = ccd_img[c_row - 4 : c_row + 4, c_col - 4 : c_col + 4]

    if bgd is None:
        raise NotImplementedError("no calc_flight_background here")
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


def get_kwargs_from_starcheck_text(obs_text, include_cat=False, force_catalog=False):
    """
    Get proseco kwargs using the exact catalog from the starcheck output text
    ``obs_text``.  Mostly copied from annie/annie (should move into mica.starcheck
    one day).

    :param obs_text: text copied from starcheck output
    :param include_cat: include the original catalog in returned kwargs
    :param force_catalog: force proseco catalog to match the original catalog
    :returns: dict of keyword args corresponding to proseco args
    """
    import re
    import textwrap

    from mica.starcheck.starcheck_parser import (
        get_catalog,
        get_coords,
        get_dither,
        get_manvrs,
        get_pred_temp,
        get_starcat_header,
        get_targ,
    )

    # Remove common leading whitespace
    obs_text = textwrap.dedent(obs_text)

    # Set up defaults
    kw = {
        "dither": (8.0, 8.0),
        "t_ccd": -10.0,
        "date": "2018:001",
        "n_guide": 8,
        "detector": "ACIS-S",
        "sim_offset": 0,
        "focus_offset": 0,
    }
    try:
        kw["obsid"] = int(re.search(r"OBSID:\s(\d+).*", obs_text).group(1))
    except Exception:
        # Nothing else will work so raise an exception
        raise ValueError(
            "text does not have OBSID: <obsid>, does not look like appropriate text"
        )

    try:
        out = get_coords(obs_text)
        kw["att"] = [out["point_ra"], out["point_dec"], out["point_roll"]]
    except Exception:
        try:
            out = get_manvrs(obs_text)[-1]
            kw["att"] = [
                out["target_Q1"],
                out["target_Q2"],
                out["target_Q3"],
                out["target_Q4"],
            ]
        except Exception:
            pass

    try:
        out = get_manvrs(obs_text)[-1]
        kw["man_angle"] = out["angle_deg"]
    except Exception:
        pass

    try:
        out = get_dither(obs_text)
        kw["dither"] = (out["dither_y_amp"], out["dither_z_amp"])
    except Exception:
        pass

    try:
        ccd_temp = get_pred_temp(obs_text)
        kw["t_ccd"] = ccd_temp
    except Exception:
        pass

    try:
        starcat_hdr = get_starcat_header(obs_text)
        kw["date"] = starcat_hdr["mp_starcat_time"]
    except Exception:
        pass

    try:
        cat = Table(get_catalog(obs_text))
    except Exception:
        cat = None

    if cat is not None:
        monitors = get_monitors(kw["obsid"], cat)
        if monitors:
            kw["monitors"] = monitors

        kw["n_fid"] = np.count_nonzero(cat["type"] == "FID")

        if include_cat:
            kw["cat"] = cat

        if force_catalog:
            ok = np.in1d(cat["type"], ("ACQ", "BOT"))
            kw["n_acq"] = np.count_nonzero(ok)
            kw["include_ids_acq"] = cat["id"][ok].tolist()
            kw["include_halfws_acq"] = cat["halfw"][ok].tolist()

            ok = np.in1d(cat["type"], ("GUI", "BOT"))
            # n_guide kwarg needs to include guide stars + MON stars
            kw["n_guide"] = np.count_nonzero(ok) + np.count_nonzero(
                cat["type"] == "MON"
            )
            kw["include_ids_guide"] = cat["id"][ok].tolist()

            ok = np.in1d(cat["type"], "FID")
            kw["include_ids_fid"] = cat["id"][ok].tolist()
        else:
            kw["n_acq"] = 8
            kw["n_guide"] = 8 - kw["n_fid"]

    try:
        targ = get_targ(obs_text)
        kw["detector"] = targ["sci_instr"]
        kw["sim_offset"] = targ["sim_z_offset_steps"]
        kw["focus_offset"] = 0
    except Exception:
        pass

    return kw


def get_monitors(obsid, cat):
    """
    Generate a list of monitor requests for an existing catalog

    :param obsid: int, obsid
    :param cat: starcheck catalog from get_catalog
    :returns: list of monitor requests
    """
    monitors = []

    # Sort catalog by slot and index, and then iterate over catalog in reverse
    # order. This guarantees that any GFM or MON in slot 7 comes first so that
    # `monitors` is in the correct order. In particular the first entry is
    # always put into slot 7.
    cat = cat.copy()
    cat.sort(["slot", "idx"])

    for row in cat[::-1]:
        # Will need to remove when we start scheduling OR's with 8x8 guides.
        if (
            row["type"] in ("GUI", "BOT")
            and row["slot"] == 7
            and row["sz"] == "8x8"
            and obsid <= 38000
        ):
            # Guide from MON (GFM) so force scheduling as GUI
            monitor = [
                row["yang"],
                row["zang"],
                ACA.MonCoord.YAGZAG,
                row["mag"],
                ACA.MonFunc.GUIDE,
            ]
            monitors.append(monitor)

        elif row["type"] == "MON":
            # Bona fide MON entry
            mon_func = (
                ACA.MonFunc.MON_FIXED
                if row["slot"] == row["dim"]
                else ACA.MonFunc.MON_TRACK
            )
            monitor = [
                row["yang"],
                row["zang"],
                ACA.MonCoord.YAGZAG,
                row["maxmag"],
                mon_func,
            ]
            monitors.append(monitor)

    return monitors


def includes_for_obsid(obsid):
    """
    Return a dict with the include_ids_acq, include_halfws_acq and include_ids_guide
    lists for the as-run flight catalog of ``obsid``.  Useful for forcing catalog.

    Example::

      >>> from proseco import get_aca_catalog
      >>> from proseco.core import includes_for_obsid
      >>> obsid = 8008
      >>> aca = get_aca_catalog(obsid, **includes_for_obsid(obsid))

    :param obsid: int, obsid
    """
    from mica.starcheck import get_starcheck_catalog

    obs = get_starcheck_catalog(obsid)

    cat = obs["cat"]
    out = {}

    ok = np.in1d(cat["type"], ("ACQ", "BOT"))
    out["include_ids_acq"] = cat["id"][ok].tolist()
    out["include_halfws_acq"] = cat["halfw"][ok].tolist()

    ok = np.in1d(cat["type"], ("GUI", "BOT"))
    out["include_ids_guide"] = cat["id"][ok].tolist()

    ok = np.in1d(cat["type"], ("FID"))
    out["include_ids_fid"] = cat["id"][ok].tolist()

    return out


def get_dim_res(halfws):
    """Get ACA search command ``dim`` and ``res`` corresponding to ``halfws``

    From DM11, where ``dim`` is a 6-bit uint and ``res`` is a 1-bit uint with
    0 => "L" and 1 => H".

    Define the search region (``dim``) for any image data slot to be a square
    centered at the commanded angular Z and Y coordinates. When the high
    resolution flag (H) (``res``) in command word 3 is true (=1), the half width
    of the square, in arc seconds, is (20 + 5D) where D is the dimension field
    in command word 2. When the H flag is false (=0), the half width, is (20 +
    40D) arc seconds.

    :param halfws: array of int
        Half-widths (arcsec)
    :returns: tuple (int array, int array)
        ACA search DIM and RES entries corresponding to halfws
    """
    # Max value of dim is 2**6 = 64, so high res is up to 20 + 5 * 63 = 335.
    # But values up to 335 + 2.5 would have dim=63 because of rounding. Select
    # the high-res and low-res values at that cutoff.
    ok = halfws < 335 + 2.5

    dim = np.zeros_like(halfws)
    res = np.zeros_like(halfws)
    # High-resolution halfws
    res[ok] = 1
    dim[ok] = np.round((halfws[ok] - 20) / 5)
    # Low-resolution halfws
    res[~ok] = 0
    dim[~ok] = np.round((halfws[~ok] - 20) / 40)

    if np.any((dim < 0) | (dim > 63)):
        raise ValueError(f"halfws {halfws} leading to bad value(s) of dim {dim}")

    return dim, res


def get_img_size(n_fids):
    """Get guide image readout size from ``n_fids``.

    This is the core definition for the default rule that OR's (with fids) get
    6x6 and ER's (no fids) get 8x8. This might change in the future.

    Parameters
    ----------
    n_fids : int
        Number of fids in the catalog

    Returns
    -------
    int
        Guide star image readout size to be used in a catalog (6 or 8)
    """
    return 6 if n_fids > 0 else 8
