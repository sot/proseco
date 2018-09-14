.. proseco documentation master file, created by
   sphinx-quickstart on Thu Sep 13 18:24:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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
man_angle     maneuver angle (deg)
t_ccd_acq     ACA CCD temperature for acquisition (degC)
t_ccd_guide   ACA CCD temperature for guide (degC)
date          date of acquisition (any DateTime-compatible format)
dither_acq    acq dither size (2-element sequence (y, z), arcsec)
dither_guide  guide dither size (2-element sequence (y, z), arcsec)
detector      'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
sim_offset    SIM translation offset from nominal [steps] (default=0)
focus_offset  SIM focus offset [steps] (default=0)
============= =========================================================

**Optional**

============== =========================================================
Argument       Description
============== =========================================================
include_ids    list of AGASC IDs of stars to include in selected catalog
include_halfws list of acq halfwidths corresponding to ``include_ids``
exclude_ids    list of AGASC IDs of stars to exclude from selected catalog
stars          table of AGASC stars (will be fetched from agasc if None)
============== =========================================================

**Debug**

============== =========================================================
Argument       Description
============== =========================================================
optimize       optimize star catalog after initial selection (default=True)
verbose        provide extra logging info (mostly calc_p_safe) (default=False)
print_log      print the run log to stdout (default=False)
============== =========================================================

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

The ``aca`` object provides a ``thumbs_up`` attribute which is
a rough indicator that the catalog will likely pass or fail ACA review.
The value of this is ``0`` for NOT OK and ``1`` for OK.  Note in
this case that too few guide stars have been selected::

  >>> aca.thumbs_up
  0

  >>> aca.acqs.thumbs_up
  1
  >>> aca.guides.thumbs_up
  0
  >>> aca.fids.thumbs_up
  1

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

Data requirements
-----------------
Required data files are:

::

  $SKA/data/agasc/miniagasc.h5
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

API docs
--------

.. toctree::
   :maxdepth: 3

   api
