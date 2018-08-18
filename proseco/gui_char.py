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
            "Aca2Pix": [
                [6.0884049557694304,4.9261856391646699],
                [0.00037618178860977601,0.200203020554239],
                [-0.200249577514165,0.00033228418325504598],
                [-2.7052897180601e-09,-5.3570209777220199e-09],
                [9.7557263803716508e-10,1.91073314224894e-08],
                [-2.9486515531677799e-08,-4.8576658185286602e-09],
                [8.3119801831277497e-13,2.0109207042849501e-10],
                [-1.96043819238097e-10,5.0972154587641396e-16],
                [5.1413424477146302e-13,1.99339355492595e-10],
                [-1.97282476269237e-10,2.5273983431918399e-14]
                ],
            "Pix2Arc": 4.9628899999999998,
            "CatalogRadius": 1.1000000000000001,
            "CatalogMinMag": 14.5,
            "ManvrErrorSigmaInit": [4.8481368110953597e-05,4.8481368110953605e-07,4.8481368110953605e-07],
            "ManvrErrorSearchBoxMargin": 9.6962736221907193e-05,
            "BadStarList": [36178592,39980640,185871616,188751528,190977856,260968880,260972216,261621080,296753512,300948368,301078152,301080376,301465776,335025128,335028096,414324824,444743456,465456712,490220520,502793400,509225640,570033768,614606480,637144600,647632648,650249416,656409216,690625776,692724384,788418168,849226688,956175008,989598624,1004817824,1016736608,1044122248,1117787424,1130635848,1130649544,1161827976,1196953168,1197635184],
            "ACAFaintCommandLimit": 13.98,
            "minCmdMagErrorBar": 1.5,
            "StarAcqTime": 270,
            "ERImageSize": 2,
            "Dark_Map": {
                "Scale_4c": 1.5900000000000001
                },
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
            "FOV": {
                "YArcSecLim": [-2410,2473],
                "ZArcSecLim": [-2504,2450]
                },
            "Body": {
                "Column": {
                    "MagDiff": 4.5,
                    "Separation": 10,
                    },
                "Register": {
                    "MagDiff": 5,
                    "Separation": 4,
                    "Width": 2
                    },
                "Pixels": {
                    "ZPixLim": [-512.5,511.5],
                    "YPixLim": [-512.5,511.5],
                    "Center": [-0.5,-0.5],
                    "EdgeBuffer": 5,
                    }
                },
            "Select": {
                "NMaxSelect": 8,
                "MaxSearchBox": 25,
                "MinSearchBox": 25,
                "LeverArm": 0.017453292519943295,
                "NDirectSearch": 792,
                "C_10": 1444,
                "CCDIntTime": 1.3999999999999999,
                "Sig_P1": 16.199999999999999,
                "Sig_P2": 0.5,
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
                    "MagLimit": [5.7999999999999998,10.199999999999999],
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
                    "Thresh": .025,
                    },
                },
            {
                "Stage": 2,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 1
                    },
                "Inertial": {
                    "MagLimit": [5.7999999999999998,10.199999999999999],
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
                    "Thresh": .05,
                    },
                },
            {
                "Stage": 3,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 1
                    },
                "Inertial": {
                    "MagLimit": [5.7999999999999998,10.300000000000001],
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
                    "Thresh": .05,
                    },
                },
            {
                "Stage": 4,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 1
                    },
                "Inertial": {
                    "MagLimit": [5.7999999999999998,10.300000000000001],
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
                    "Thresh": .075,
                    },
                },
            {
                "Stage": 5,
                "SearchSettings": {
                    "DoColumnRegisterCheck": 1,
                    "DoBminusVcheck": 0
                    },
                "Inertial": {
                    "MagLimit": [5.7999999999999998,10.300000000000001],
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
                    "Thresh": .10,
                    },
                }
            ],
        },
}
