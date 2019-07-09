#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ..constants import Constants
from .hg import calcQ

__all__ = ["calcTss",
           "calcT1"]

S = Constants.SOLAR_CONSTANT
sigma = Constants.STEFAN_BOLTZMANN


def calcTss(r, p_v, eps, G, eta):
    """
    Calculate the subsolar temperature.

    Parameters
    ----------
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    p_v : float or `~numpy.ndarray` (N)
        Geometric albedo.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.
    G : float or `~numpy.ndarray` (N)
        HG slope parameter.
    eta: float or `~numpy.ndarray` (N)
        Beaming parameter.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns subsolar temperature in K.
    """
    return ((1 - p_v * calcQ(G)) * S / (eps * eta * sigma * (r**2)))**0.25


def calcT1(r, p_v, eps, G, eta):
    """
    Calculate the normalized subsolar temperature.

    See Myhrvold 2017 (https://doi.org/10.1016/j.icarus.2017.12.024).

    Parameters
    ----------
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    p_v : float or `~numpy.ndarray` (N)
        Geometric albedo.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.
    G : float or `~numpy.ndarray` (N)
        HG slope parameter.
    eta: float or `~numpy.ndarray` (N)
        Beaming parameter.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns normalized subsolar temperature in K.
    """
    return calcTss(r, p_v, eps, G, eta) * np.sqrt(r)
