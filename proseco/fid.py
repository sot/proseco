import numpy as np

from . import characteristics_fid as FID


def get_fid_positions(detector, focus, sim_offset):
    """Calculate the fid light positions for all fids for ``detector``.

    This is adapted from the Matlab
    MissionScheduling/stars/StarSelector/@StarSelector/private/fidPositions.m

      %Convert focus steps to meters
      table = characteristics.Science.Sim.focus_table;
      xshift = interp1(table(:,1),table(:,2),focus,'*linear');
      if(isnan(xshift))
          error('Focus is out of range');
      end

      %find translation from sim offset
      stepsize = characteristics.Science.Sim.stepsize;
      zshift = SIfield.fidpos(:,2)-simoffset*stepsize;

      yfid=-SIfield.fidpos(:,1)/(SIfield.focallength-xshift);
      zfid=-zshift/(SIfield.focallength-xshift);

    :param detector: 'ACIS-S' | 'ACIS-I' | 'HRC-S' | 'HRC-I'
    :param focus: SIM focus [steps]
    :param sim_offset: SIM translation offset from nominal [steps]

    :returns: yang, zang where each is a np.array of angles [arcsec]
    """
    # Table of (step, fa_pos [m]) pairs, used to interpolate from step to
    # fa_pos [m].
    focus_table = np.array(FID.focus_table)
    steps = focus_table[:, 0]  # Absolute FA step position
    fa_pos = focus_table[:, 1]  # meters
    xshift = np.interp(focus, steps, fa_pos, left=np.nan, right=np.nan)
    if np.isnan(xshift):
        raise ValueError('focus is out of range')

    # Y and Z position of fids on focal plane in meters.
    # Apply SIM Z translation from sim offset to the nominal Z position.
    ypos = FID.fidpos[detector][:, 0]
    zpos = FID.fidpos[detector][:, 1] - sim_offset * FID.tsc_stepsize

    # Calculate angles.  (Should these be atan2?  Does it matter?)
    yfid = -ypos / (FID.focal_length[detector] - xshift)
    zfid = -zpos / (FID.focal_length[detector] - xshift)

    yang, zang = np.degrees(yfid) * 3600, np.degrees(zfid) * 3600

    return yang, zang
