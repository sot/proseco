import numpy as np

from proseco.core import get_kwargs_from_starcheck_text

# Vanilla observation info
STD_INFO = dict(att=(0, 0, 0),
                detector='ACIS-S',
                sim_offset=0,
                focus_offset=0,
                date='2018:001',
                n_guide=5,
                n_fid=3,
                t_ccd=-11,
                man_angle=90,
                dither=8.0)


def mod_std_info(**kwargs):
    std_info = STD_INFO.copy()
    std_info.update(kwargs)
    return std_info


# Flat dark current map
DARK40 = np.full(shape=(1024, 1024), fill_value=40)


# Parameters for test cases (to avoid starcheck.db3 dependence)
OBS_INFO = {}

OBS_INFO[19387] = dict(obsid=19387,
                       detector='ACIS-S',
                       sim_offset=0,
                       focus_offset=0,
                       att=[188.617671, 2.211623, 231.249803],
                       date='2017:182:22:06:22.744',
                       t_ccd=-14.1,
                       man_angle=1.74,
                       dither=4.0)

OBS_INFO[21007] = dict(obsid=21007,
                       detector='ACIS-S',
                       sim_offset=0,
                       focus_offset=0,
                       att=[184.371121, 17.670062, 223.997765],
                       date='2018:159:11:20:52.162',
                       t_ccd=-11.3,
                       man_angle=60.39,
                       dither=8.0)

OBS_INFO[20603] = dict(obsid=20603,
                       detector='ACIS-S',
                       sim_offset=0,
                       focus_offset=0,
                       att=[201.561783, 7.748784, 205.998301],
                       date='2018:120:19:06:28.154',
                       t_ccd=-11.2,
                       man_angle=111.95,
                       dither=8.0)

OBS_INFO[19605] = {'att': [350.897404, 58.836913, 75.068745],
                   'date': '2018:135:15:52:08.898',
                   'detector': 'ACIS-S',
                   'dither': 8.0,
                   'focus_offset': 0,
                   'man_angle': 79.15,
                   'obsid': 19605,
                   'sim_offset': 0,
                   't_ccd': -10.8}


def get_starcheck_obs_kwargs(filename):
    """
    Parse the starcheck.txt file to get keyword arg dicts for get_aca_catalog()

    :param filename: file name of starcheck.txt in load products
    :returns: dict (by obsid) of kwargs for get_aca_catalog()

    """
    delim = "==================================================================================== "
    with open(filename, 'r') as fh:
        text = fh.read()
    chunks = text.split(delim)
    outs = {}
    for chunk in chunks:
        if "No star catalog for obsid" in chunk:
            continue
        try:
            out = get_kwargs_from_starcheck_text(chunk, include_cat=True)

        except ValueError:
            continue
        else:
            outs[out['obsid']] = out

    return outs
