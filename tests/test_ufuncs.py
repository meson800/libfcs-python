import numpy as np
from numpy import NaN

import _libfcs_ext

# Tests from 
# https://master.dl.sourceforge.net/project/flowcyt/Gating-ML/Gating-ML%202.0/GatingML_2.0_Specification.20130122.pdf

def test_flin():
    x = np.array([-100,-10,0,10,100,120,890,1000]).reshape((8,1))
    T = np.array([1000,1000,1024]).reshape((1,3))
    A = np.array([0,100,256]).reshape((1,3))
    expected = np.array([
        [-0.1, 0.0, 0.121875],
        [-0.01, 0.081818, 0.1921875],
        [0, 0.090909, 0.2],
        [0.01, 0.1, 0.2078125],
        [0.1, 0.181818, 0.278125],
        [0.12, 0.2, 0.29375],
        [0.89, 0.9, 0.8953125],
        [1.0, 1.0, 0.98125]
    ])
    np.testing.assert_allclose(_libfcs_ext.flin(x, T, A), expected, rtol=2e-5)

def test_flog():
    x = np.array([-1, 0, 0.5, 1, 10, 100, 1000, 1023, 10000, 100000, 262144]).reshape((11,1))
    T = np.array([10000,1023,262144]).reshape((1,3))
    M = np.array([5, 4.5, 4.5]).reshape((1,3))
    expected = np.array([
        [NaN, NaN, NaN],
        [NaN, NaN, NaN],
        [0.139794, 0.264243, -0.271016],
        [0.2, 0.331139, -0.204120],
        [0.4, 0.553361, 0.018102],
        [0.6, 0.775583, 0.240324],
        [0.8, 0.997805, 0.462547],
        [0.801975, 1.0, 0.464741],
        [1.0, 1.220028, 0.684768],
        [1.2, 1.442250, 0.906991],
        [1.283708, 1.535259, 1.0]
    ])
    print(_libfcs_ext.flog(x,T,M))
    np.testing.assert_allclose(_libfcs_ext.flog(x, T, M), expected, rtol=2e-5)