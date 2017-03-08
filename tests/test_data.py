#!/usr/bin/env python3
# System modules

# External modules
import numpy as np

# Internal modules

#######################################
### Constants and example test data ###
#######################################

# Weibull test data
WEIBULL_TEST_DATA = {
"scale": np.array([ 1,         1,      0.5]),
"shape": np.array([ 1,         1,      0.5]),
"lag"  : np.array([-1,         1,      0.5]),
"res"  : np.array([ 0,np.exp(-1),np.exp(-1)]),
}

# Linear test data
LINEAR_TEST_DATA = {
"intercept": np.array([  1,  1,   1,  1,   1,  2,   2]),
"slope":     np.array([ -1, -1,  -1, -1,  -1, -1,  -1]),
"lag"  :     np.array([ -1,  0, 0.5,  1, 1.1,  1, 2.1]),
"res"  :     np.array([  0,  1, 0.5,  0,   0,  1,   0]),
}

# Constant test data
CONSTANT_TEST_DATA = {
"constant": np.array([  1,  1,   1,  1,   1,  2,   2]),
"lag"  :    np.array([ -1,  0, 0.5,  1, 1.1,  1, 2.1]),
"res"  :    np.array([  0,  1,   1,  1,   0,  2,   0]),
"max_lag":  np.array([  0,  0,   0,  0,   1,  0,   2]),
}

# convolution test data
CONVOLUTION_TEST_DATA = { 
"x": [ 
    np.array([0,1,2]),
    np.array([0,1,2,3,4]),
    np.array([0,1,2,3,4]),
    np.array([0,1,2,3,4]),
    ],
"filter": [ 
    np.array([1,2,1]),
    np.array([1,2,1]),
    np.array([0]),
    np.array([0,0,0]),
    ],
"res": [ 
    np.array([4]),
    np.array([4,8,12]),
    np.array([0,0,0,0,0]),
    np.array([0,0,0]),
    ],
}

# convolution matrix test data
CONVOLUTION_MATRIX_TEST_DATA = { 
"length": [ 
    1,2,3,4,5,6
    ],
"filter": [ 
    np.array([1]),
    np.array([1,2]),
    np.array([1,2,3]),
    np.array([1]),
    np.array([1,2]),
    np.array([1,2,3]),
    ],
"res": [ 
    np.array([[1]]),
    np.array([[ 1.,  2. ]]),
    np.array([[ 1.,  2.,  3. ]]),
    np.array([[ 1.,  0.,  0.,  0. ], 
              [ 0.,  1.,  0.,  0. ], 
              [ 0.,  0.,  1.,  0. ], 
              [ 0.,  0.,  0.,  1. ]]),
    np.array([[ 1.,  2.,  0.,  0.,  0. ], 
              [ 0.,  1.,  2.,  0.,  0. ],
              [ 0.,  0.,  1.,  2.,  0. ], 
              [ 0.,  0.,  0.,  1.,  2. ]]),
    np.array([[ 1.,  2.,  3.,  0.,  0.,  0. ],
              [ 0.,  1.,  2.,  3.,  0.,  0. ],
              [ 0.,  0.,  1.,  2.,  3.,  0. ],
              [ 0.,  0.,  0.,  1.,  2.,  3. ]]),
    ],
"convres": [ 
    np.array([0.]),
    np.array([2.]),
    np.array([8.]),
    np.array([ 0.,  1.,  2.,  3. ]),
    np.array([  2.,   5.,   8.,  11. ]),
    np.array([  8.,  14.,  20.,  26. ]),
    ]
}


