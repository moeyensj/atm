#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from scipy import integrate

from ..config import Config
from ..functions import calcPlanckLambda
from .model import Model

__all__ = ["STM"]


class STM(Model):
    """
    Standard Thermal Model

    Reference
    ---------
   

    """
    def __init__(self,
                 name="Standard Thermal Model (STM)",
                 acronym="STM",
                 tableDir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables/stm/"),
                 verbose=Config.verbose):
        super(STM, self).__init__(name=acronym,
                                  acronym=acronym,
                                  tableDir=tableDir,
                                  verbose=verbose)

    def calcT(self, omega, T_ss):
        """
        Calculates the temperature at a specific angular position on the surface of
        an asteroid. The subsolar point is located at (omega) = (0, 0) 
        where the temperature is defined to be the subsolar temperature.

        Returns zero K for any angular positions on the night side of an asteroid
        (omega > pi/2).

        Parameters
        ----------
        omega : float or `~numpy.ndarray`
            Angular distance from the subsolar point in radians.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.

        Returns
        -------
        float or `~numpy.ndarray`
            Temperature at an angular specific point on the asteroid in K.
        """
        return T_ss * np.maximum(0.0, np.cos(omega))**0.25

    def calcFluxLambdaEmitted(self, omega, lambd, T_ss):
        """
        Calculates the flux emitted at a specific angular point on
        the surface of an asteroid of size unity as a function of angular position,
        wavelength and the subsolar temperature.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        omega : float or `~numpy.ndarray`
            Angular distance from the subsolar point.
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.

        Returns
        -------
        float or `~numpy.ndarray`
            Flux emitted at a specific angular point on the asteroid in units of W m^-3.
        """
        return np.pi * calcPlanckLambda(lambd, self.calcT(omega, T_ss))

    def calcTotalFluxLambdaEmitted(self, lambd, T_ss):
        """
        Calculates the total flux emitted at the surface of an asteroid
        of size unity as a function of wavelength and the subsolar temperature. 

        Returns flux in units of W m^-3.

        Parameters
        ----------
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.

        Returns
        -------
        float or `~numpy.ndarray`
            Total flux emitted in units of W m^-3.
        """
        return 2 * integrate.quad(lambda omega, lambd, T_ss:
                                  (self.calcFluxLambdaEmitted(omega, lambd, T_ss)
                                   * np.cos(omega) * np.sin(omega)),
                                  0, np.pi/2,
                                  args=(lambd, T_ss))[0]

    def calcFluxLambdaEmittedToObs(self, omega, lambd, T_ss, alpha,
                                   phaseAngleFluxCorrection=Config.phaseAngleFluxCorrection):
        """
        Calculates the flux emitted at a specific angular point on
        the surface of an asteroid of size unity towards an observer at
        phase angle alpha as a function of angular position,
        wavelength, subsolar temperature and phase angle.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        omega : float or `~numpy.ndarray`
            Angular distance in from the subsolar point in radians.
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.
        alpha : float or `~numpy.ndarray`
            Phase angle in radians.
        phaseAngleFluxCorrection : float, optional
            Phase angle flux correction.
            [Default = `atm.Config.phaseAngleFluxCorrection`]

        Returns
        -------
        float or `~numpy.ndarray`
            Flux in units of W m^-3 at a specific angular point on the asteroid emitted
            to an observer located in the direction of alpha.
        """
        return (np.pi * calcPlanckLambda(lambd, self.calcT(omega, T_ss)) 
                * 10**(-np.abs(np.degrees(alpha)) * phaseAngleFluxCorrection/2.5))

    def calcTotalFluxLambdaEmittedToObs(self, lambd, T_ss, alpha):
        """
        Calculates the total flux emitted at the surface of an asteroid 
        of size unity towards an observer at phase angle alpha 
        as a function of wavelength, subsolar temperature and phase angle.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.
        alpha : float or `~numpy.ndarray`
            Phase angle in radians.

        Returns
        -------
        float or `~numpy.ndarray`
            Total flux in units of W m^-3 emitted to an observer 
            located in the direction of alpha.
        """
        return 2 * integrate.quad(lambda omega, lambd, T_ss, alpha:
                                  self.calcFluxLambdaEmittedToObs(omega, lambd, T_ss, alpha) * np.cos(omega) * np.sin(omega), 
                                  0, np.pi/2,
                                  args=(lambd, T_ss, alpha))[0]

