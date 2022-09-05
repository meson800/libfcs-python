import numpy as np

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
    assert np.allclose(_libfcs_ext.flin(x, T, A), expected)