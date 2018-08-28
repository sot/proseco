# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Star selection characteristics from MATLAB

CCD = {'row_min': -512.5,
       'row_max': 511.5,
       'col_min': -512.5,
       'col_max': 511.5,
       'window_pad': 7,
       'row_pad': 8,
       'col_pad': 1}

CHAR = { "Stars": {
           "General": {
            "Pix2Arc": 4.9628899999999998,
            "BadStarList": [36178592,39980640,185871616,188751528,190977856,260968880,260972216,261621080,296753512,300948368,301078152,301080376,301465776,335025128,335028096,414324824,444743456,465456712,490220520,502793400,509225640,570033768,614606480,637144600,647632648,650249416,656409216,690625776,692724384,788418168,849226688,914493824,956175008,989598624,1004817824,1016736608,1044122248,1117787424,1130635848,1130649544,1161827976,1196953168,1197635184],
            "ASPQ1Lim": [0,0],
            "ASPQ2Lim": [0,0],
            "ASPQ3Lim": [0,999],
            "MagErrorTol": 100,
            "PosErrorTol": 3000,
            "Traps": {
                "Column": 347,
                "Row": -374,
                "DeltaColumn": 3,
                "ExclusionZone": {
                    "Neg": [-6,-2],
                    "Pos": [3,7]
                    }
                },
            "Body": {
                "Column": {
                    "MagDiff": 4.5,
                    "Separation": 10,
                    },
                },
            "Select": {
                "NMaxSelect": 8,
                "nSurplus": 1
                },
            "Spoiler": {
                "MinSep": 7,
                "MaxSep": 11,
                "Intercept": 9,
                "Slope": 0.5,
                "MagDiffLimit": "-_Inf_"
                },
            },
        "Guide": [
            {
                "Stage": 1,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 1
                    },
                "Inertial": {
                    "MagLimit": [5.9,10.199999999999999],
                    "VARIABLELim": -9999,
                    "MagErrRand": 0.26000000000000001,
                    "MagErrSyst": 0,
                    "MaxMagError": 1.5
                    },
                "Body": {
                    "Pixels": {
                        "BadPixels": [
                            [-245,0,454,454],
                            ]
                        },
                    },
                "Spoiler": {
                    "SigErrMultiplier": 3,
                    },
                "Imposter": {
                    "Thresh": 3.5,
                    },
                },
            {
                "Stage": 2,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 1
                    },
                "Inertial": {
                    "MagLimit": [5.9,10.199999999999999],
                    "VARIABLELim": -9999,
                    "MagErrRand": 0.14999999999999999,
                    "MagErrSyst": 0,
                    "MaxMagError": 1
                    },
                "Body": {
                    "Pixels": {
                        "BadPixels": [
                            [-245,0,454,454],
                            ]
                        },
                    },
                "Spoiler": {
                    "SigErrMultiplier": 2,
                    },
                "Imposter": {
                    "Thresh": 3.0,
                    },
                },
            {
                "Stage": 3,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 1
                    },
                "Inertial": {
                    "MagLimit": [5.9,10.300000000000001],
                    "VARIABLELim": -9999,
                    "MagErrRand": 0,
                    "MagErrSyst": 0,
                    "MaxMagError": 0.5
                    },
                "Body": {
                    "Pixels": {
                        "BadPixels": [
                            [-245,0,454,454],
                            ]
                        },
                    },
                "Spoiler": {
                    "SigErrMultiplier": 1,
                    },
                "Imposter": {
                    "Thresh": 3.0,
                    },
                },
            {
                "Stage": 4,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 1
                    },
                "Inertial": {
                    "MagLimit": [5.9,10.300000000000001],
                    "VARIABLELim": -9999,
                    "MagErrRand": 0,
                    "MagErrSyst": 0,
                    "MaxMagError": 0.5
                    },
                "Body": {
                    "Pixels": {
                        "BadPixels": [
                            [-245,0,454,454],
                            ]
                        },
                    },
                "Spoiler": {
                    "SigErrMultiplier": 0,
                    },
                "Imposter": {
                    "Thresh": 2.5,
                    },
                },
            {
                "Stage": 5,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 0
                    },
                "Inertial": {
                    "MagLimit": [5.9,10.300000000000001],
                    "VARIABLELim": -9999,
                    "MagErrRand": 0,
                    "MagErrSyst": 0,
                    "MaxMagError": 0.5
                    },
                "Body": {
                    "Pixels": {
                        "BadPixels": [
                            [-245,0,454,454],
                            ]
                        },
                    },
                "Spoiler": {
                    "SigErrMultiplier": 0,
                    },
                "Imposter": {
                    "Thresh": 1.75,
                    },
                }
            ],
        },
}
