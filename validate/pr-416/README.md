# PR #416 validation: fid search spoilers

These scripts check the fid search spoiler change in PR #416 against flight load
products and telemetry.

| Script | Purpose |
|---|---|
| `rerun_loads.py` | Reselect every catalog in load `*_proseco.pkl.gz` files with the flight `call_args` and dark cal (`dark_date`), using whichever proseco is importable. |
| `diff_loads.py` | Compare two `rerun_loads.py` runs and report changed catalogs, fid sets, acq P2 and fid search spoilers. |
| `check_fid_acq.py` | For one obsid and fid, show the fid slot telemetry at fid search and the bright AGASC stars near the fid, to tell whether the fid or a star was acquired. |

## Requirements

- A ska3 environment with cheta telemetry access (for `check_fid_acq.py`).
- Load products in `$SKA/data/mpcrit1/mplogs` and dark cals for the loads.

## Reproduce

From the proseco repo root on the `fid-search-spoiler` branch:

```bash
LOADS=$SKA/data/mpcrit1/mplogs/2026/*/oflsa/output/*_proseco.pkl.gz
OUT=/tmp/pr-416   # any directory outside the repo

# Reference: proseco master (5.19.0) in a separate worktree
git fetch origin
git worktree add --detach $OUT/proseco-master origin/master

# PYTHONPATH selects the proseco under test (script dir is first on sys.path, so
# PYTHONPATH wins over the repo checkout).
env PYTHONPATH=$OUT/proseco-master python validate/pr-416/rerun_loads.py \
    --out-dir $OUT/runs/master $LOADS
env PYTHONPATH=$PWD python validate/pr-416/rerun_loads.py \
    --out-dir $OUT/runs/pr416 $LOADS

python validate/pr-416/diff_loads.py $OUT/runs/master $OUT/runs/pr416

# Flight fid acquisition for each changed catalog
M=$SKA/data/mpcrit1/mplogs/2026
python validate/pr-416/check_fid_acq.py $M/APR2026/oflsa/output/APR2026A_proseco.pkl.gz 31516 3
python validate/pr-416/check_fid_acq.py $M/AUG3126/oflsa/output/AUG3126A_proseco.pkl.gz 32521 4
python validate/pr-416/check_fid_acq.py $M/OCT1226/oflsa/output/OCT1226A_proseco.pkl.gz 31770 6
python validate/pr-416/check_fid_acq.py $M/OCT1226/oflsa/output/OCT1226A_proseco.pkl.gz 33497 6
```

Each `rerun_loads.py` run takes about 1 minute with 10 processes on an M-series Mac.
Each output pickle records the proseco version, file and git commit that were used.

## Expected output

`diff_loads.py` for the 45 loads JAN1226A to OCT1926A. The first two lines, which give
the proseco version, git commit and path for each run, are omitted. The reference is
`origin/master` at 4926c55 (5.19.0).

```
Loads: 45  catalogs: 1894  with fids: 1196
Reference != flight: 647 catalogs in 43 loads
New != reference: 4 catalogs

APR2026A  31516 2026:113 HRC-I  Cyg X-2                fids [1, 2, 3] -> [1, 2, 4]  P2 6.57 -> 6.57  n_other_changed=0
    fid 3: 417350448 mag=5.32 dy=+70 dz=-83
AUG3126A  32521 2026:248 ACIS-S GJ 2006 B              fids [2, 4, 5] -> [3, 5, 6]  P2 4.42 -> 4.42  n_other_changed=0
    fid 4: 916330040 mag=5.31 dy=+125 dz=-87
    fid 4: 916332864 mag=5.82 dy=+120 dz=-93
    fid 4: 916332936 mag=6.21 dy=+123 dz=-92
OCT1226A  31770 2026:285 ACIS-S RXJ0821.0+0752         fids [1, 4, 6] -> [1, 3, 4]  P2 7.10 -> 7.12  n_other_changed=2
    fid 6: 104205176 mag=6.94 dy=+66 dz=-87
OCT1226A  33497 2026:291 ACIS-S RXJ0821.0+0752         fids [3, 5, 6] -> [3, 4, 5]  P2 6.08 -> 6.08  n_other_changed=0
    fid 6: 104205176 mag=6.94 dy=+29 dz=-68
```

"Reference != flight" is expected. Loads before JUN2226A were made with proseco
5.17 or 5.18, and 5.19.0 changed the ACA transform (PR #414). The OCT1226A loads were
made with 5.19.0, and all 51 of their reference catalogs match flight. This confirms
that reselecting with `call_args` and `dark_date` reproduces the flight catalogs.

`check_fid_acq.py` for obsid 33497 shows the 6.94 mag star tracked instead of fid 6.
The tracked position (431.9, 761.1) matches the star position at the estimated
attitude (433.2, 761.0), and it drifts as the attitude error is corrected:

```
Obsid 33497 fid 6 slot 2: commanded yang=400.4 zang=802.8
obs_start=2026:291:05:29:58.388
         date         AOACFID2 AOACFCT2 AOACYAN2 AOACZAN2 AOACMAG2
--------------------- -------- -------- -------- -------- --------
2026:291:05:31:21.829     FID      SRCH -3276.80 -3276.80    13.94
2026:291:05:31:30.029     FID      TRAK   431.88   761.05     6.94
2026:291:05:31:31.054     FID      TRAK   431.88   761.05     6.94
2026:291:05:32:01.804     FID      TRAK   430.43   755.28     6.94
2026:291:05:32:32.554     FID      TRAK   428.68   752.03     6.94
2026:291:05:33:03.304     FID      TRAK   426.68   750.60     7.00
2026:291:05:33:34.054     FID      TRAK   424.75   749.70     6.94
2026:291:05:34:04.804     FID      TRAK   422.35   749.45     7.00
2026:291:05:34:35.554     FID      TRAK   420.48   749.03     6.94

First fid TRAK at 2026:291:05:31:30.029
AGASC stars with MAG_ACA < 7.25 within 200 arcsec of fid:
  dy_cat, dz_cat: offset from fid at catalog attitude
  yang_est, zang_est: position at estimated attitude at first fid TRAK
 agasc_id mag_aca dy_cat dz_cat yang_est zang_est
--------- ------- ------ ------ -------- --------
104205176    6.94  29.43 -67.90   433.19   761.03
```
