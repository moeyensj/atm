#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy import integrate

A = np.array([3.332, 1.862])
B = np.array([0.631, 1.218])
C = np.array([0.986, 0.238])

__all__ = ["calcHG",
           "calcQ"]


def _calcW(alpha):
    """
    Calculate W component.

    Phi_i = W*Phi_iS + (1 - W)*Phi_iL; i = 1,2

    See Bowell et al. 1989 equation A4.
    """
    return np.exp(-90.56 * np.tan(1/2 * alpha)**2)


def _calcPhiS(C_i, alpha):
    """
    Calculate Phi_iS component.

    Phi_i = W*Phi_iS + (1 - W)*Phi_iL; i = 1,2

    See Bowell et al. 1989 equation A4.
    """
    return 1 - C_i*np.sin(alpha) / (0.119 + 1.341 * np.sin(alpha) - 0.754 * np.sin(alpha)**2)


def _calcPhiL(A_i, B_i, alpha):
    """
    Calculate Phi_iL component.

    Phi_i = W*Phi_iS + (1 - W)*phi_iL; i = 1,2

    See Bowell et al. 1989 equation A4.
    """
    return np.exp(-A_i * np.tan(1/2 * alpha)**B_i)


def _calcPhi(alpha, A_i, B_i, C_i):
    """
    Calculate phase functions Phi_1, Phi_2. Returns phase function values
    as a numpy array: [Phi_1, Phi_2]

    See Bowell et al. 1989 equation A4.
    """
    return (_calcW(alpha) * _calcPhiS(C_i, alpha) + (1 - _calcW(alpha)) * _calcPhiL(A_i, B_i, alpha))


def calcHG(alpha, G):
    """
    Calculate (1 - G) * Phi_1(alpha) + G * Phi_2(alpha).

    Only valid for alpha between 0 and 120 degrees.

    See Bowell et al. 1989 equation A4.

    Parameters
    ----------
    alpha : float or `~numpy.ndarray` (N)
        Phase angle in radians.
    G : float or `~numpy.ndarray` (N)
        HG slope parameter.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        The full HG function.
    """
    return (1 - G) * _calcPhi(alpha, A[0], B[0], C[0]) + G *_calcPhi(alpha, A[1], B[1], C[1])


def _constantTerm(alpha):
    """
    Calculate the constant component of the phase integral.
    """
    return calcHG(alpha, 0.0) * np.sin(alpha)


def _gTerm(alpha):
    """
    Calculate the multiplicative factor of the phase integral.
    """
    return (calcHG(alpha, 1.0) - calcHG(alpha, 0.0)) * np.sin(alpha)

constTerm = 2.0*integrate.quad(_constantTerm, 0, np.pi)[0]
gTerm = 2.0*integrate.quad(_gTerm, 0, np.pi)[0]


def calcQ(G):
    """
    Calculates the phase integral given the HG slope parameter.
    This relationship is only valid for 0 < G < 1.

    See Bowell et al. 1989 equation A7.

    Parameters
    ----------
    G : float or `~numpy.ndarray` (N)
        HG slope parameter.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        The phase integral for the given value of G.
    """
    return constTerm + G*gTerm
