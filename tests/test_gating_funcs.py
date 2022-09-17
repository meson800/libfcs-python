import numpy as np
import _libfcs_ext

def test_polygon_gate():
    # Test self-intersecting gate (Figure 12)
    polygon = np.array([
        [0,0], [2,0], [2,3], [3,3], [3,2],
        [1,2], [1,1], [4,1], [4,4], [0,4]
    ])
    print(polygon)
    test_x, test_y = np.meshgrid(
        np.arange(0.5, 4.5, 1.0),
        np.arange(0.5, 4.5, 1.0)
        )
    test_points = np.column_stack((test_x.flatten(), test_y.flatten()))
    print(test_points)
    expected = np.array([
        True, True, False, False,
        True, False, True, True,
        True, True, False, True,
        True, True, True, True
    ])
    actual = _libfcs_ext.polygon_gate(test_points, polygon)
    np.testing.assert_array_equal(actual, expected)