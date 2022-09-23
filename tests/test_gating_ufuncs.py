import numpy as np
import libfcs._libfcs_ext as _libfcs_ext

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
    print(actual)
    np.testing.assert_array_equal(actual, expected)

    # Test simple triangle gate (Figure 13)
    polygon = np.array([[0,0], [300,400], [400,0]])
    test_points = np.array([[50, 200], [200, 50]])
    expected = np.array([False,True])
    actual = _libfcs_ext.polygon_gate(test_points, polygon)
    np.testing.assert_array_equal(actual, expected)

    # Test simple pentagon gate (Figure 14)
    polygon = np.array([
        [0.2, 50], [0.6, 50], [0.6, 150],
        [0.2, 150], [0.4, 100]
    ])
    test_points = np.array([[0.3, 100], [0.5, 100], [0.3, 55], [0.3, 140]])
    expected = np.array([False, True, True, True])
    actual = _libfcs_ext.polygon_gate(test_points, polygon)
    np.testing.assert_array_equal(actual, expected)