"""
Convenience gate functions
"""
from typing import Dict, List
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from . import _libfcs_ext
from .transforms import Transform
from .exceptions import GateError


class Gate(object):
    """
    Gates output an object that can be either be:
    1) described
    2) filtered
    3) tagged

    To do these operations, a Gate object has two functions,
    a inclusion() function and a label() function.
    Gates must define the label() function and can optionally
    override the inclusion() function if needed. Otherwise,
    it will default to coercion to direct comparison to the given value.

    NOTE: remove the col_map complexity. Have users decide which names they want in the FlowEvents constructor
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def label(self, df: pd.DataFrame) -> np.ndarray:
        '''To override'''
        pass

    def inclusion(self, df: pd.DataFrame, *, val=True) -> np.ndarray:
        return self.label(df) == val

class RangeGate(Gate):
    """
    Represents a (possibly unbounded in one direction)
    unidimensional gate.
    """
    x: Transform
    left: float
    right: float
    def __init__(self, x: Transform, y: Transform, *, left: float = -np.inf, right: float = np.inf):
        self.x = x
        self.left = left
        self.right = right
    
    def label(self, df: pd.DataFrame) -> np.ndarray:
        events = self.x.transform_df(df)
        return (events >= self.left) & (events < self.right)

class PolygonGate(Gate):
    """
    Represents a (possibly non-simple) polygon gate.
    """
    x: Transform
    y: Transform
    vertices: np.ndarray
    def __init__(self, x: Transform, y: Transform, *, vertices: np.ndarray):
        self.x = x
        self.y = y
        self.vertices = vertices
    
    def label(self, df: pd.DataFrame) -> np.ndarray:
        """
        Uses polygon inclusion-exclusion to define gate
        """
        events = np.column_stack([self.x.transform_df(df), self.y.transform_df(df)])
        return _libfcs_ext.polygon_gate(events, self.vertices)

class EllipsoidGate(Gate):
    """
    Represents a ellipsoid gate.
    """
    transforms: List[Transform]
    center: np.ndarray
    inv_covar_matrix: np.ndarray
    d_squared: float

    def __init__(self, *transforms, center: np.ndarray, covariance: np.ndarray, distance_squared: float):
        self.transforms = [t for t in transforms]
        n_dims = len(self.transforms)
        # Check dimensions of other variables
        if center.shape != (n_dims, 1):
            raise GateError(f"Ellipsoid center must be a column vector of shape {(n_dims,1)}")
        if covariance.shape != (n_dims, n_dims):
            raise GateError(f"Covariance matrix must be square and of shape {(n_dims, n_dims)}")
        if distance_squared <= 0:
            raise GateError("Ellipsoid distance must be a positive number!")
        
        # Check for positive definite covariance matrix
        try:
            _ = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            raise GateError("Covariance matrix must be positive semidefinite!")
        
        self.center = center
        self.inv_covar_matrix = np.linalg.inv(covariance)
        self.d_squared = distance_squared
    
    def label(self, df: pd.DataFrame) -> np.ndarray:
        # Define "n" the number of dimensions, "e" the number of events
        events = np.column_stack([t.transform_df(df) for t in self.transforms])[:, np.newaxis] # This is e x n x 1
        recentered_events = events - self.center # this is (e x n x 1) - (n x 1)
        transposed_recentered = np.transpose(recentered_events, (0, 2, 1)) # this is (e x 1 x n)
        # then do (e x 1 x n) @ (n x n) @ (e x n x 1) = (e x 1 x 1), then flatten to (e,)
        return np.matmul(
                np.matmul(transposed_recentered, self.inv_covar_matrix), recentered_events
            ).flatten() <= self.d_squared


class AndGate(Gate):
    """
    Represents a AND condition over many sub-gates.
    """
    subgates: List[Gate]
    def __init__(self, *args):
        if len(args) < 2:
            raise GateError("Must specify at least two gates to AND together!")
        self.subgates = [gate for gate in args]
    
    def label(self, df: pd.DataFrame) -> np.ndarray:
        result = self.subgates[0].label(df) & self.subgates[1].label(df)
        for gate in self.subgates[2:]:
            result = result & gate
        return result

class OrGate(Gate):
    """
    Represents an OR condition over many sub-gates.
    """
    subgates: List[Gate]
    def __init__(self, *args):
        if len(args) < 2:
            raise GateError("Must specify at least two gates to OR together!")
        self.subgates = [gate for gate in args]

    def label(self, df: pd.DataFrame) -> np.ndarray:
        result = self.subgates[0].label(df) | self.subgates[1].label(df)
        for gate in self.subgates[2:]:
            result = result | gate
        return result

class NotGate(Gate):
    """
    Represents a NOT condition with respect to a single sub-gate.
    """
    gate: Gate
    def __init__(self, gate):
        self.gate = gate
    
    def label(self, df: pd.DataFrame) -> np.ndarray:
        return np.logical_not(self.gate.label(df))