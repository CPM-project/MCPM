# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["linear_least_squares"]

import numpy as np
from scipy import linalg


def linear_least_squares(A, y, yvar=None, l2=None):
    """
        Solve a linear system as fast as possible.
        
        :param A: ``(ndata, nbasis)``
        The basis matrix.
        
        :param y: ``(ndata)``
        The observations.
        
        :param yvar:
        The observational variance of the points ``y``.
        
        :param l2:
        The L2 regularization strength. Can be a scalar or a vector (of length
        ``A.shape[1]``).
        
    """
    # Check if A, l2, and yvar are the same as before, if so, then use 
    # previously remembered factor.
    if (linear_least_squares.A is not None
            and linear_least_squares.factor is not None):
        if (linear_least_squares.A.shape == A.shape
                    and linear_least_squares.l2 == l2
                    and linear_least_squares.yvar == yvar):
            if (linear_least_squares.A == A).all():
                if len(y.shape)==1:
                    y = y[:, None]
                if yvar is not None:
                    Ciy = y / yvar[:, None]
                else:
                    Ciy = y
                fac = linear_least_squares.factor
                AT = linear_least_squares.AT
                return linalg.cho_solve(fac, np.dot(AT, Ciy), overwrite_b=True)
    
    linear_least_squares.l2 = l2
    linear_least_squares.yvar = yvar
    linear_least_squares.A = A
    
    if len(y.shape)==1:
        y = y[:, None]
        
    # Incorporate the observational uncertainties.
    if yvar is not None:
        CiA = A / yvar[:, None]
        Ciy = y / yvar[:, None]
    else:
        CiA = A
        Ciy = y
    
    # Compute the pre-factor.
    AT = A.T
    linear_least_squares.AT = AT    
    ATA = np.dot(AT, CiA)
    
    # Incorporate any L2 regularization.
    if l2 is not None:
        if np.isscalar(l2):
            l2 = l2 + np.zeros(A.shape[1])
        ATA[np.diag_indices_from(ATA)] += l2
    
    # Solve the equations overwriting the temporary arrays for speed.
    factor = linalg.cho_factor(ATA, overwrite_a=True)
    linear_least_squares.factor = factor
    return linalg.cho_solve(factor, np.dot(AT, Ciy), overwrite_b=True)
    
linear_least_squares.A = None
linear_least_squares.factor = None
