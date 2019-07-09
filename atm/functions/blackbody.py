#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ..constants import Constants

__all__ = ["calcPlanckLambda",
           "calcPlanckNu"]

# Initialize constants
h = Constants.PLANCK
c = Constants.LIGHTSPEED
k_B = Constants.BOLTZMANN


def calcPlanckLambda(lambd, T):
    """
    The wavelength form of Planck's law. Given a wavelength or array
    of wavelengths, returns the blackbody distribution in SI units
    with characteristic temperature T in Kelvin.

    Where wavelength or temperature are less than zero, returns zero.

    Parameters
    ----------
    lambd : float or `~numpy.ndarray` (N)
        Wavelength in m.
    T : float or `~numpy.ndarray` (N)
        Temperature in K.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns Planck function in W sr^−1 m^−3.

    See Also
    --------
    calcPlanckNu : The frequency form of Planck's law
    """
    return np.where(
        (lambd <= 0.0) | (T <= 0.0),
        0.0,
        (2*h*c**2)/lambd**5 * 1/(np.exp((h*c)/(lambd*k_B*T)) - 1)
        )


def calcPlanckNu(nu, T):
    """
    The frequency form of Planck's law. Given a frequency or array
    of frequencies, returns the blackbody distribution in SI units
    with characteristic temperature T in Kelvin.

    Where frequency or temperature are less than zero, returns zero.

    Parameters
    ----------
    nu : float or `~numpy.ndarray` (N)
        Frequency in Hz.
    T : float or `~numpy.ndarray` (N)
        Temperature in K.

    Returns
    -------
    float or `~numpy.ndarray`
        Returns Planck function in W sr^-1 m^-2 Hz^−1.

    See Also
    --------
    calcPlanckLambda : The wavelength form of Planck's law
    """
    return np.where(
        (nu <= 0.0) | (T <= 0.0),
        0.0,
        (2*h*nu**3)/c**2 * 1/(np.exp((h*nu)/(k_B*T)) - 1)
        )
