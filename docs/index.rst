Proseco
=======

Proseco is a Python package to allow for selection of acquisition stars,
guide stars, and fiducial lights for the Chandra ACA star tracker during
operations.

For acquisition stars and fid lights, proseco uses a probabilistic
evalation of the catalog to minimize the chance of a safing event and
loss of science.  For guide stars, proseco uses a staged approach to
select optimal stars.

Getting Started
---------------

From a top-level perspective the main interface into proseco is a single
function :func:`~proseco.catalog.get_aca_catalog`.

Arguments
+++++++++

As can be seen from the docstring, :func:`~proseco.catalog.get_aca_catalog`
takes a long list of arguments which are required to fully specify:

**Required**

============= =========================================================
Argument      Description
============= =========================================================
obsid         obsid (default=0)
                **OR**
att           attitude (any object that can initialize Quat)
n_acq         desired number of acquisition stars (default=8)
n_fid         desired number of fid lights (req'd unless obsid spec'd)
n_guide       desired number of guide stars + monitor windows (see note below)
man_angle     maneuver angle (deg)
t_ccd_acq     ACA CCD temperature for acquisition (degC)
t_ccd_guide   ACA CCD temperature for guide (degC)
date          date of acquisition (any DateTime-compatible format)
dither_acq    acq dither size (2-element sequence (y, z), arcsec)
dither_guide  guide dither size (2-element sequence (y, z), arcsec)
detector      'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
============= =========================================================

**Optional**

=================== =================================================================
Argument            Description
=================== =================================================================
target_name         Target name, e.g. from the OR list (str)
duration            Observation duration (secs)
man_angle_next      Manvr angle to next attitude after observation (deg, default=180)
sim_offset          SIM translation offset from nominal [steps] (default=0)
focus_offset        SIM focus offset [steps] (default=0)
target_offset       target offset including dynamical offset (y, z, deg)
include_ids_acq     int or list of AGASC IDs of stars to include in acq catalog
include_halfws_acq  int or list of acq halfwidths corresponding to ``include_ids``
exclude_ids_acq     int or list of AGASC IDs of stars to exclude from acq catalog
include_ids_guide   int or list of AGASC IDs of stars to include in guide catalog
exclude_ids_guide   int or list of AGASC IDs of stars to exclude from guide catalog
include_ids_fid     int or list of fid IDs to include from fid catalog
exclude_ids_fid     int or list of fid IDs to exclude from fid catalog
img_size_guide      readout window size for guide stars (6, 8, or ``None``)
stars               table of AGASC stars (will be fetched from agasc if None)
monitors            N x 5 array of monitor star specifications (see `Monitor stars`_)
t_ccd_eff_acq       ACA CCD effective temperature for acquisition (degC)
t_ccd_eff_guide     ACA CCD effective temperature for guide (degC)
dark                1024x1024 dark image (e-/sec, default=None => auto-fetch)
dark_date           date of dark cal (str)
=================== =================================================================

The input ``n_guide`` parameter represents the number of slots available for the
combination of guide stars and monitor windows (including both fixed and
tracking monitor windows). In most normal situations, ``n_guide`` is equal to
``8 - n_fid``. The ``n_guide`` parameter is confusingly named but this is
because the actual number of guide stars is not known in advance in the case of
auto-conversion from a monitor request to a guide star. In actual practice,
what is normally known is how many slots are available for the combination of
guide stars and monitor windows, so this makes the call to catalog creation
simpler.

Within the ``include_halfws_acq`` list, one can supply the value ``0`` for a
star instead of a typical legal value such as ``60`` or ``120``.  In that case
proseco will run the normal optimization and choose the best halfwidth for that
included star.  If the ``include_halfws_acq`` argument is not supplied or set
to ``[]`` then all halfwidths will be chosen by proseco.

The two optional effective temperatures are normally not supplied and will be
computed based on the current ACA penalty limit. Likewise, the ``dark_date`` is
normally derived as the date of the most recent dark cal before ``date``.
However, for reproducing catalog selection it may be required to provide all of
those arguments.

**Debug**

============== ================================================================
Argument       Description
============== ================================================================
optimize       optimize star catalog after initial selection (default=True)
verbose        provide extra logging info (mostly calc_p_safe) (default=False)
print_log      print the run log to stdout (default=False)'
raise_exc      raise exception if it occurs in processing (default=True)
============== ================================================================

Examples
^^^^^^^^

For generating a catalog corresponding to an actual scheduled observation, the
easiest strategy is to simply provide the ``obsid`` as a parameter.  Proseco
will then fetch all necessary data via the ``mica.starcheck`` archive.  This
requires installation of the ``cmd_states.db3`` and ``starcheck.db3`` data
files into the correct locations.  See `Data requirements`_ for details.

::

  >>> from proseco.catalog import get_aca_catalog
  >>> aca = get_aca_catalog(19387)
  >>> aca
  <ACACatalogTable length=11>
   slot  idx     id    type  sz   p_acq    mag    maxmag   yang     zang    dim   res  halfw
  int64 int64  int64   str3 str3 float64 float64 float64 float64  float64  int64 int64 int64
  ----- ----- -------- ---- ---- ------- ------- ------- -------- -------- ----- ----- -----
      0     1        2  FID  8x8   0.000    7.00    8.00  -773.20 -1742.03     1     1    25
      1     2        4  FID  8x8   0.000    7.00    8.00  2140.23   166.63     1     1    25
      2     3        5  FID  8x8   0.000    7.00    8.00 -1826.28   160.17     1     1    25
      3     4 38280776  BOT  6x6   0.985    8.77   10.27 -2254.09 -2172.43    20     1   160
      4     5 37879960  BOT  6x6   0.984    9.20   10.70  -567.34  -632.27    20     1    80
      5     6 37882072  BOT  6x6   0.956   10.16   11.66  2197.62  1608.89    20     1    80
      6     7 37879992  ACQ  6x6   0.933   10.41   11.91   318.47 -1565.92    20     1    60
      7     8 37882416  ACQ  6x6   0.901   10.41   11.91   481.80  2204.44    20     1    80
      0     9 37880176  ACQ  6x6   0.584   10.62   12.12   121.33 -1068.25    20     1    60
      1    10 37881728  ACQ  6x6   0.057   10.76   12.26  2046.89  1910.79    20     1   100
      2    11 37880376  ACQ  6x6   0.084   10.80   12.30 -1356.71  1071.32    20     1   100

The catalog represented by the ``aca`` object is the "merged" version of the
acquisition, guide, and fid catalogs.  It corresponds to the uplinked catalog
and what is reviewed in starcheck.

The component catalogs are available in ``acqs``, ``guides``, and ``fids``
attributes::

  >>> aca.acqs
  <AcqTable length=8>
     id        ra       dec      yang     zang     row   ...  slot  maxmag  dim   sz   res
   int32    float64   float64  float64  float64  float64 ... int64 float32 int64 str3 int64
  -------- ---------- -------- -------- -------- ------- ... ----- ------- ----- ---- -----
  38280776 188.538857 3.077618 -2254.09 -2172.43  460.83 ...     3   10.27    20  6x6     1
  37879960 188.579307 2.444460  -567.34  -632.27  119.53 ...     4   10.70    20  6x6     1
  37882072 188.584100 1.455829  2197.62  1608.89 -436.73 ...     5   11.66    20  6x6     1
  37879992 188.222720 2.414842   318.47 -1565.92  -58.45 ...     6   11.91    20  6x6     1
  37882416 189.011634 1.723922   481.80  2204.44  -90.05 ...     7   11.91    20  6x6     1
  37880176 188.364941 2.371052   121.33 -1068.25  -18.64 ...     0   12.12    20  6x6     1
  37881728 188.675737 1.435994  2046.89  1910.79 -406.36 ...     1   12.26    20  6x6     1
  37880376 189.086022 2.319189 -1356.71  1071.32  278.91 ...     2   12.30    20  6x6     1

  >>> aca.guides
  <GuideTable length=3>
     id        ra       dec      yang     zang     row   ...  maxmag p_acq  dim   res  halfw  sz
   int32    float64   float64  float64  float64  float64 ... float32 int64 int64 int64 int64 str3
  -------- ---------- -------- -------- -------- ------- ... ------- ----- ----- ----- ----- ----
  38280776 188.538857 3.077618 -2254.09 -2172.43  460.83 ...   10.27 0.000     1     1    25  6x6
  37879960 188.579307 2.444460  -567.34  -632.27  119.53 ...   10.70 0.000     1     1    25  6x6
  37882072 188.584100 1.455829  2197.62  1608.89 -436.73 ...   11.66 0.000     1     1    25  6x6

  >>> aca.fids
  <GuideTable length=3>
     id        ra       dec      yang     zang     row   ...  maxmag p_acq  dim   res  halfw  sz
   int32    float64   float64  float64  float64  float64 ... float32 int64 int64 int64 int64 str3
  -------- ---------- -------- -------- -------- ------- ... ------- ----- ----- ----- ----- ----
  38280776 188.538857 3.077618 -2254.09 -2172.43  460.83 ...   10.27 0.000     1     1    25  6x6
  37879960 188.579307 2.444460  -567.34  -632.27  119.53 ...   10.70 0.000     1     1    25  6x6
  37882072 188.584100 1.455829  2197.62  1608.89 -436.73 ...   11.66 0.000     1     1    25  6x6

Each of the individual catalogs also has a ``warnings`` attribute that is a
list of any warnings which occurred in processing::

  >>> aca.guides.warnings
  ['WARNING: Selected only 3 guide stars versus requested 5']

Finally, in the event of an unhandled exception within the acq, guide, and fid
selection code, there is an ``exception`` attribute on the top-level ``aca``
object.  If an exception occurs then the output table will have zero length
with a single ``id`` column::

  >>> aca = get_aca_catalog(19387, detector='FAIL')
  >>> len(aca['id'])
  0

  >>> print(aca.exception)
  Traceback (most recent call last):
    File "/Users/aldcroft/git/proseco/proseco/catalog.py", line 43, in get_aca_catalog
      aca = _get_aca_catalog(obsid=obsid, **kwargs)
    File "/Users/aldcroft/git/proseco/proseco/catalog.py", line 102, in _get_aca_catalog
      aca.fids = get_fid_catalog(acqs=aca.acqs, **fid_kwargs)
    File "/Users/aldcroft/git/proseco/proseco/fid.py", line 51, in get_fid_catalog
      fids.meta['cand_fids'] = fids.get_fid_candidates()
    File "/Users/aldcroft/git/proseco/proseco/fid.py", line 258, in get_fid_candidates
      self.meta['sim_offset'])
    File "/Users/aldcroft/git/proseco/proseco/fid.py", line 416, in get_fid_positions
      ypos = FID.fidpos[detector][:, 0]
  KeyError: 'FAIL'

Monitor stars
-------------

Monitor stars in a catalog are specified withe the ``monitors`` keyword.
This can accept a list of lists (or 2-d array of floats) in the form::

  [
    [coord0, coord1, coord_type, mag, function],  # Monitor window 1 (req'd)
    [coord0, coord1, coord_type, mag, function],  # Monitor windows 2, 3, ... N (optional)
    ..
  ]


If passed as a list of lists, the input will be converted to a numpy 2-d array
(``np.array(monitors)``) which will have shape ``(N, 5)`` where ``N`` is the number of
monitor windows. Enumeration codes related to the ``monitors`` keyword are
defined in the module ``proseco.characteristics``::

  class MonFunc:
      AUTO = 0
      GUIDE = 1
      MON_TRACK = 2
      MON_FIXED = 3

  class MonCoord:
      RADEC = 0
      ROWCOL = 1
      YAGZAG = 2

The 5 values ``[coord0, coord1, coord_type, mag, function]`` in a monitor window
specification are:

- ``coord0``: first coordinate: RA (deg), row (pixel), or Y angle (arcsec))
- ``coord1``: second coordinate: Dec (deg), column (pixel), or Z angle (arcsec)
- ``coord_type``:

  - ``MonCoord.RADEC``: RA/Dec (deg)
  - ``MonCoord.ROWCOL``: row/column (pixel)
  - ``MonCoord.YAGZAG``: Y/Z angle (arcsec)

- ``mag``: star magnitude (used for commanded MAXMAG for monitor window requests)
- ``function``:

  - ``MonFunc.AUTO``: Schedule as a guide star if it meets requirements and does not introduce
    critical warnings, otherwise schedule as a monitor window that tracks the brightest star.
  - ``MonFunc.GUIDE``: Schedule as a guide star if a corresponding star is found
    in the AGASC within 2 arcsec, otherwise raise an exception.
  - ``MonFunc.MON_TRACK``: Schedule as monitor window that tracks the brightest star.
  - ``MonFunc.MON_FIXED``: Schedule as monitor window that is fixed on the CCD
     with no tracking.

The typical use cases are as follows:

- OR-specified monitor target with ``function=MonFunc.AUTO``

  - This is the most common configuration and lets proseco decide how to schedule the OR.
  - Coordinates provided as RA, Dec.

- OR-specified monitor target ``function=MonFunc.GUIDE``

  - Force scheduling as a guide star in a case where proseco would normally
    schedule as a monitor window.
  - For instance a star where the observer provides information on mag or
    variability that is different from the AGASC (e.g. brighter).

- OR-specified monitor target ``function=MonFunc.MON_TRACK``

  - Force scheduling as a monitor window in a case where proseco
    would normally schedule as a guide star.
  - For instance a star where the observer provides information on mag or
    variability that is different from the AGASC (e.g. fainter or more variable).

- Fixed-location ER monitor window ``function=MonFunc.MON_FIXED``

  - Used for engineering investigation, e.g. looking at a specific flickering pixel.
  - ``coord0,1`` as Y/Z angle or (most commonly) row/column.

Data requirements
-----------------
Required data files are:

::

  $SKA/data/agasc/proseco_agasc_<latest>.h5
  $SKA/data/mica/archive/aca_dark/<YYYYddd>/image.fits
  $SKA/data/mica/archive/aca_dark/<YYYYddd>/properties.json

The following optional data files are needed for specifying only an
``obsid`` in the call to :func:`~proseco.catalog.get_aca_catalog`::

  $SKA/data/cmd_states/cmd_states.db3
  $SKA/data/mica/archive/starcheck/starcheck.db3

See `Syncing Ska data
<https://github.com/sot/skare3/wiki/Ska3-runtime-environment-for-users#ska-data>`_
for details on automatically syncing these files for a standalone linux or Mac
environment.  For Matlab tools on Windows a separate mechanism will be provided.

Environment Variables
---------------------

The following environment variables are used by proseco:

- ``AGASC_DIR``: path to AGASC directory for getting AGASC star data. This
  overrides the default value of ``$SKA/data/agasc``.
- ``AGASC_HDF5_FILE``: path to AGASC HDF5 file for getting AGASC star data. This
  overrides the default value of ``<default_agasc_dir>/proseco_agasc_<latest>.h5``,
  where ``<default_agasc_dir> = $AGASC_DIR or $SKA/data/agasc``.
  If this is a relative path then it is relative to ``<default_agasc_dir>``.
- ``AGASC_SUPPLEMENT_ENABLED``: set to ``"False"`` to disable using the AGASC
   supplement. This is for testing and should not be used in production.
- ``PROSECO_DISABLE_OVERLAP_PENALTY``: if set to ``"True"`` then disable the
  overlap penalty in the acq star selection. This is for testing and should not
  be used in production.
- ``PROSECO_ENABLE_FID_OFFSET``: controls application of time and temperature dependent fid
  light position offsets (from the ACA drift model) in :ref:`~proseco.fid.get_fid_positions`:
  - Not set: apply offsets if time and temperature are provided (as is done in ``proseco`` fid
    selection since version 5.12.0)
  - ``"True"``: require that time and temperature be provided and apply offsets.
  - ``"False"``: do not apply offsets (typically used in regression testing not production).
- ``PROSECO_IGNORE_MAXAGS_CONSTRAINTS``: if set then do not update ``maxmag`` in the
  catalog to prevent search hits clipping.
- ``PROSECO_OR_IMAGE_SIZE``: override the default OR image size of 8x8. Can be one of
  "4", "6", or "8".
- ``PROSECO_PRINT_OBC_CAT``: if set then create and print a debug catalog while doing
  catalog merging.
- ``SKA``: root directory that contains 'data' directory

API docs
--------

.. toctree::
   :maxdepth: 3

   api

Unit Testing
------------

.. toctree::
   :maxdepth: 3

   unit_tests
