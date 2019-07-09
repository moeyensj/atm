#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

__all__ = ["calcM"]

def calcM(mags, D, delta):
    """
    Calculates the pseudo-absolute magnitude.
    
    Parameters
    ----------
    mags : float or `~np.ndarray` (N, M)
        Array of N magnitudes per M filters.
    D : float or `~np.ndarray` (N)
        Diameter of the asteroid in m.
    delta : float or `~np.ndarray` (N)
        Distance between asteroid and the observatory in AU.

    Returns
    -------
    M : `np.ndarray` (N, M)
        Pseudo-absolute magnitude with shape N observations by
        M filters.
    """
    M = mags.T + 5 * np.log10(D/1000) - 5 * np.log10(delta)
    return M.T