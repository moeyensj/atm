#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

__all__ = ["calcH",
           "calcD",
           "calcPv"]


def calcH(D, p_v):
    """
    Calculate absolute H magnitude given asteroid diameter and geometric albedo.

    Parameters
    ----------
    D : float or `~numpy.ndarray` (N)
        Asteroid diameter in meters.
    p_v : float or `~numpy.ndarray` (N)
        Geometric albedo.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns asteroid absolute H magnitude.
    """
    return 5 * np.log10(1.329e6 / (D * np.sqrt(p_v)))


def calcD(H, p_v):
    """
    Calculate asteroid diameter given absolute H magnitude and geometric albedo.

    Parameters
    ----------
    H : float or `~numpy.ndarray` (N)
        Absolute H magnitude.
    p_v : float or `~numpy.ndarray` (N)
        Geometric albedo.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Asteroid diameter in meters.
    """
    return 1.329e6 / (np.sqrt(p_v) * 10**(H / 5))

def calcPv(D, H):
    """
    Calculate geometric albedo given asteroid diameter and H magnitude.

    Parameters
    ----------
    D : float or `~numpy.ndarray` (N)
        Asteroid diameter in meters.
    H : float or `~numpy.ndarray` (N)
        Absolute H magnitude.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Geometric albedo.
    """
    return (1.329e6 / (D * 10**(H / 5)))**2
