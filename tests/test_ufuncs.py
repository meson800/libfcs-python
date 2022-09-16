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

def test_fasinh():
    x = np.array([-10, -5, -1, 0, 0.3, 1, 3, 10, 100, 1000]).reshape((10,1))
    T = np.array([1000, 1000, 1000]).reshape((1,3))
    M = np.array([4, 5, 3]).reshape((1,3))
    A = np.array([1, 0, 2]).reshape((1,3))
    expected = np.array([
        [-0.200009, -0.6, 0.199144],
        [-0.139829, -0.539794, 0.256923],
        [-0.00085584, -0.400009, 0.358203],
        [0.2, 0, 0.4],
        [0.303776, 0.295521, 0.412980],
        [0.400856, 0.400009, 0.441797],
        [0.495521, 0.495425, 0.503776],
        [0.600009, 0.6, 0.600856],
        [0.8, 0.8, 0.800009],
        [1, 1, 1]
    ])
    print(_libfcs_ext.fasinh(x,T,M,A))
    np.testing.assert_allclose(_libfcs_ext.fasinh(x, T, M, A), expected, rtol=2e-5)

def test_logicle():
    x = np.array([-10, -5, -1, 0, 0.3, 1, 3, 10, 100, 1000]).reshape((10,1))
    T = np.array([1000, 1000, 1000]).reshape((1,3))
    W = np.array([1, 1, 0]).reshape((1,3))
    M = np.array([4, 4, 4]).reshape((1,3))
    A = np.array([0, 1, 1]).reshape((1,3))
    tol = np.array([1e-9, 1e-9, 1e-9]).reshape((1,3))
    expected = np.array([
        [0.067574, 0.254059, -0.200009],
        [0.147986, 0.318389, -0.139829],
        [0.228752, 0.383001, -0.0008558414],
        [0.25, 0.4, 0.2],
        [0.256384, 0.405107, 0.303776],
        [0.271248, 0.416999, 0.400856],
        [0.312897, 0.450318, 0.495521],
        [0.432426, 0.545941, 0.600009],
        [0.739548, 0.791638, 0.8],
        [1, 1, 1]
    ])
    actual = _libfcs_ext.logicle(x,T,W,M,A,tol)
    print(actual)
    np.testing.assert_allclose(actual, expected, rtol=2e-5)

def test_hyperlog():
    x = np.array([-10, -5, -1, 0, 0.3, 1, 3, 10, 100, 1000]).reshape((10,1))
    T = np.array([1000, 1000, 1000]).reshape((1,3))
    W = np.array([1, 1, 0.01]).reshape((1,3))
    M = np.array([4, 4, 4]).reshape((1,3))
    A = np.array([0, 1, 1]).reshape((1,3))
    tol = np.array([1e-9, 1e-9, 1e-9]).reshape((1,3))
    expected = np.array([
        [0.083554, 0.266843, 0.017447],
        [0.155868, 0.324695, 0.106439],
        [0.229477, 0.383581, 0.182593],
        [0.25, 0.4, 0.202],
        [0.256239, 0.404991, 0.207833],
        [0.270523, 0.416419, 0.221407],
        [0.309091, 0.447273, 0.259838],
        [0.416446, 0.533157, 0.386553],
        [0.731875, 0.7855, 0.774211],
        [1, 1, 1]
    ])
    actual = _libfcs_ext.hyperlog(x,T,W,M,A,tol)
    print(actual)
    np.testing.assert_allclose(actual, expected, atol=1e-6)