#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from scipy import integrate

from ..config import Config
from ..functions import calcPlanckLambda
from ..frames import Geographic
from .model import Model

__all__ = ["FRM"]


class FRM(Model, Geographic):
    """
    Fast Rotating Thermal Model

    Reference
    ---------

    """
    def __init__(self,
                 name="Fast Rotating Thermal Model (FRM)",
                 acronym="FRM",
                 tableDir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables/frm/"),
                 verbose=Config.verbose):
        super(FRM, self).__init__(name=name,
                                  acronym=acronym,
                                  tableDir=tableDir,
                                  verbose=verbose)

    def calcT(self, theta, T_ss):
        """
        Calculates the temperature at a specific angular position on the surface of 
        an asteroid. The subsolar point is located at (theta, phi) = (0, 0) 
        where the temperature is defined to be the subsolar temperature.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Latitude in radians.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.

        Returns
        -------
        float or `~numpy.ndarray`
            Temperature at an angular specific point on the asteroid in K.
        """
        return T_ss * np.cos(theta)**0.25

    def calcFluxLambdaEmitted(self, theta, lambd, T_ss):
        """
        Calculates the flux emitted at a specific angular point on
        the surface of an asteroid of size unity as a function of angular
        position, wavelength and the subsolar temperature.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Latitude in radians.
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.

        Returns
        -------
        float or `~numpy.ndarray`
            Flux emitted at a specific angular point on the asteroid in units
            of W m^-3.
        """
        return np.pi * calcPlanckLambda(lambd, self.calcT(theta, T_ss))

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
        float
            Total flux emitted in units of W m^-3.
        """
        return 4 / np.pi * integrate.quad(lambda theta, lambd, T_ss:
                                          (self.calcFluxLambdaEmitted(theta, lambd, T_ss) 
                                           * self.surfaceElement(theta, 0) * np.cos(theta)),
                                          0,
                                          np.pi/2,
                                          args=(lambd, T_ss))[0]

    def calcFluxLambdaEmittedToObs(self, theta, lambd, T_ss, alpha):
        """
        Calculates the flux emitted at a specific angular point on
        the surface of an asteroid of size unity towards an observer at
        phase angle alpha as a function of angular position,
        wavelength, subsolar temperature and phase angle.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Latitude in radians.
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.
        alpha : float or `~numpy.ndarray`
            Phase angle in radians.

        Returns
        -------
        float or `~numpy.ndarray`
            Flux in units of W m^-3 at a specific angular point on the asteroid emitted 
            to an observer located in the direction of alpha.
        """
        return np.pi * calcPlanckLambda(lambd, self.calcT(theta, T_ss))

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
        return 4 / np.pi * integrate.quad(lambda theta, lambd, T_ss, alpha:
                                          (self.calcFluxLambdaEmittedToObs(theta, lambd, T_ss, alpha)
                                           * self.surfaceElement(theta, 0) * np.cos(theta)),
                                          0,
                                          np.pi/2,
                                          args=(lambd, T_ss, alpha))[0]

