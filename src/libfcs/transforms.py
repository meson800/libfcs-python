"""
Transformation functions to use with data.
"""
from typing import Dict
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from libfcs.exceptions import TransformError

from . import _libfcs_ext

class Transform(object):
    """Abstract transformations of data"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform(self, x:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, x:np.ndarray) -> np.ndarray:
        pass

class UnidimensionalTransform(Transform):
    """
    Represents the wide class of transformations
    that only depend on one column of entries.
    """
    x_param: str
    def __init__(self, x: str):
        self.x_param = x
    
    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform(df[self.x_param])

class LinearTransform(UnidimensionalTransform):
    """
    Represents a parametrized linear transformation.
    """
    scale_bottom: float
    scale_top: float
    def __init__(self, x: str, *, bottom: float = 0, top: float = 1):
        if top <= 0:
            raise TransformError(f"Top-of-scale value ({top}) must be positive!")
        if bottom < 0 or bottom > top:
            raise TransformError(f"Bottom-of-scale value ({bottom}) must be between zero and the top-of-scale value!")
        self.scale_bottom = bottom
        self.scale_top = top
        super().__init__(x)

    def transform(self, x:np.ndarray) -> np.ndarray:
        # T = scale_top, A = scale_bottom
        return _libfcs_ext.flin(x, self.scale_top, self.scale_bottom)
    
    def inverse_transform(self, x:np.ndarray) -> np.ndarray:
        return _libfcs_ext.inv_flin(x, self.scale_top, self.scale_bottom)

class LogTransform(UnidimensionalTransform):
    """
    Represents a parametrized logarithmic transformation.
    """
    n_decades: float
    scale_top: float
    def __init__(self, x: str, *, decades: float = 1, top: float = 1):
        if top <= 0:
            raise TransformError(f"Top-of-scale value ({top}) must be positive!")
        if decades <= 0:
            raise TransformError(f"Number of decades ({decades}) must be positive!")
        self.n_decades = decades
        self.scale_top = top
        super().__init__(x)
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        # T = scale_top, M = n_decades
        return _libfcs_ext.flog(x, self.scale_top, self.n_decades)

    def inverse_transform(self, x:np.ndarray) -> np.ndarray:
        return _libfcs_ext.inv_flog(x, self.scale_top, self.n_decades)

class InverseHyperbolicSineTransform(UnidimensionalTransform):
    """
    Represents the parametrized inverse hyperbolic sine transform.
    """
    scale_top: float
    n_decades: float
    n_negative_decades: float
    def __init__(self, x: str, *, positive_decades: float = 4, negative_decades: float = -1, top: float = 2**18):
        if top <= 0:
            raise TransformError(f"Top-of-scale value ({top}) must be positive!")
        if positive_decades <= 0:
            raise TransformError(f"Number of positive decades ({positive_decades}) must be positive!")
        if negative_decades < 0 or negative_decades > positive_decades:
            raise TransformError(f"Number of negative decades ({negative_decades}) must be positive and less than the number of positive decades!")
        self.n_decades = positive_decades
        self.n_negative_decades = negative_decades
        self.scale_top = top
        super().__init__(x)
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        # T = scale_top, M = n_decades, A = n_negative_decades
        return _libfcs_ext.fasinh(x, self.scale_top, self.n_decades, self.n_negative_decades)
    
    def inverse_transform(self, x:np.ndarray) -> np.ndarray:
        return _libfcs_ext.inv_fasinh(x, self.scale_top, self.n_decades, self.n_negative_decades)

class LogicleTransform(UnidimensionalTransform):
    """
    Represents the Logicle transform
    """
    scale_top: float
    n_decades: float
    n_linear_decades: float
    n_negative_decades: float
    tol: float
    def __init__(self, x: str, *,
                 positive_decades: float = 4, linear_decades: float = 1, negative_decades: float = 0,
                 top: float = 2**18, tolerance: float=1e-8):
        if top <= 0:
            raise TransformError(f"Top-of-scale value ({top}) must be positive!")
        if positive_decades <= 0:
            raise TransformError(f"Number of positive decades ({positive_decades}) must be positive!")
        if linear_decades < 0 or linear_decades > positive_decades / 2:
            raise TransformError(f"Number of linear decades ({linear_decades}) must be between zero "
            f"and half of the number of positive decades ({positive_decades/2})")
        if negative_decades < -linear_decades or negative_decades > positive_decades - 2 * linear_decades:
            raise TransformError(f"Number of negative decades ({negative_decades}) must be between the "
            f"negative of the number of linear decades ({-linear_decades}) and the number of positive "
            f"non-linear decades ({positive_decades - 2 * linear_decades})!")
        if tolerance <= 0:
            raise TransformError(f"Accuracy tolerance ({tolerance}) must be positive!")
        self.scale_top = top
        self.n_decades = positive_decades
        self.n_linear_decades = linear_decades
        self.n_negative_decades = negative_decades
        self.tol = tolerance
        super().__init__(x)
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        # T = scale_top, W = linear_decades, M = decades, A = n_negative_decades
        return _libfcs_ext.logicle(x, self.scale_top, self.n_linear_decades, self.n_decades, self.n_negative_decades, self.tol)

    def inverse_transform(self, x:np.ndarray) -> np.ndarray:
        return _libfcs_ext.inv_logicle(x, self.scale_top, self.n_linear_decades, self.n_decades, self.n_negative_decades)

class HyperlogTransform(UnidimensionalTransform):
    """
    Represents the Hyperlog transform.
    """
    scale_top: float
    n_decades: float
    n_linear_decades: float
    n_negative_decades: float
    tol: float
    def __init__(self, x: str, *,
                 positive_decades: float = 4, linear_decades: float = 1, negative_decades: float = 0,
                 top: float = 2**18, tolerance: float=1e-8):
        if top <= 0:
            raise TransformError(f"Top-of-scale value ({top}) must be positive!")
        if positive_decades <= 0:
            raise TransformError(f"Number of positive decades ({positive_decades}) must be positive!")
        if linear_decades <= 0 or linear_decades > positive_decades / 2:
            raise TransformError(f"Number of linear decades ({linear_decades}) must be between zero "
            f"and half of the number of positive decades ({positive_decades/2})")
        if negative_decades < -linear_decades or negative_decades > positive_decades - 2 * linear_decades:
            raise TransformError(f"Number of negative decades ({negative_decades}) must be between the "
            f"negative of the number of linear decades ({-linear_decades}) and the number of positive "
            f"non-linear decades ({positive_decades - 2 * linear_decades})!")
        if tolerance <= 0:
            raise TransformError(f"Accuracy tolerance ({tolerance}) must be positive!")
        self.scale_top = top
        self.n_decades = positive_decades
        self.n_linear_decades = linear_decades
        self.n_negative_decades = negative_decades
        self.tol = tolerance
        super().__init__(x)
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        # T = scale_top, W = linear_decades, M = decades, A = n_negative_decades
        return _libfcs_ext.hyperlog(x, self.scale_top, self.n_linear_decades, self.n_decades, self.n_negative_decades, self.tol)

    def inverse_transform(self, x:np.ndarray) -> np.ndarray:
        return _libfcs_ext.inv_hyperlog(x, self.scale_top, self.n_linear_decades, self.n_decades, self.n_negative_decades)