"""
Convenience gate functions
"""
import numpy as np

import _libfcs_ext


class Gate:
    """
    Gates output an object that can be either be:
    1) described
    2) filtered
    3) tagged
    """
    pass

def polygon_gate(events: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    return _libfcs_ext.polygon_gate(events, polygon)