#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from scipy import integrate

from ..config import Config
from ..functions import calcPlanckLambda
from ..frames import Geographic
from .model import Model

__all__ = ["NEATM"]


class NEATM(Model, Geographic):
    """
    Near Earth Asteroid Thermal Model

    Reference
    ---------
    Alan W. Harris, 1998: A Thermal Model for Near-Earth Asteroids
        https://www.sciencedirect.com/science/article/pii/S0019103597958656

    """
    def __init__(self,
                 name="Near Earth Asteroid Thermal Model (NEATM)",
                 acronym="NEATM",
                 tableDir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables/neatm/"),
                 verbose=Config.verbose):
        super(NEATM, self).__init__(name=acronym,
                                    acronym=acronym,
                                    tableDir=tableDir,
                                    verbose=verbose)

    def calcDirectionCosine(self, theta, phi, alpha):
        """
        Calculates the direction cosine to the observer. The observer and Sun are assumed 
        to be located in the equatorial plane of the asteroid. The Sun is located at phi = 0
        and the observer at phi = alpha.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Polar angle in radians.
        phi : float or `~numpy.ndarray`
            Azimuthal angle in radians.
        alpha : float or `~numpy.ndarray`
            Phase angle in radians.

        Returns
        -------
        float or `~numpy.ndarray`
            Direction cosine towards the observer.
        """
        return np.maximum(0.0, np.cos(theta) * np.cos(phi - alpha))

    def calcT(self, theta, phi, T_ss):
        """
        Calculates the temperature at a specific angular position on the surface of 
        an asteroid. The subsolar point is located at (theta, phi) = (0, 0)
        where the temperature is defined to be the subsolar temperature.

        Returns zero K for any angular positions on the night side of an asteroid
        (theta > pi/2).

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Latitude in radians.
        phi : float or `~numpy.ndarray`
            Longitude in radians.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.

        Returns
        -------
        float or `~numpy.ndarray`
            Temperature at an angular specific point on the asteroid in K.
        """
        return T_ss * np.maximum(0.0, np.cos(theta) * np.cos(phi))**0.25

    def calcFluxLambdaEmitted(self, theta, phi, lambd, T_ss):
        """
        Calculates the flux emitted at a specific angular point on
        the surface of an asteroid of size unity as a function of angular position,
        wavelength and the subsolar temperature.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Latitude in radians.
        phi : float or `~numpy.ndarray`
            Longitude in radians.
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.

        Returns
        -------
        float or `~numpy.ndarray`
            Flux emitted at a specific angular point on the asteroid in units of W m^-3.
        """
        return np.pi * calcPlanckLambda(lambd, self.calcT(theta, phi, T_ss))

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
        return integrate.dblquad(lambda theta, phi, lambd, T_ss:
                                 (self.calcFluxLambdaEmitted(theta, phi, lambd, T_ss)
                                  * self.surfaceElement(theta, phi)),
                                 -np.pi/2,
                                 np.pi/2,
                                 lambda theta: -np.pi/2,
                                 lambda theta: np.pi/2,
                                 args=[lambd, T_ss])[0]

    def calcFluxLambdaEmittedToObs(self, theta, phi, lambd, T_ss, alpha):
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
        phi : float or `~numpy.ndarray`
            Longitude in radians.
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
        return (calcPlanckLambda(lambd, self.calcT(theta, phi, T_ss))
                * self.calcDirectionCosine(theta, phi, alpha))

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
        float
            Total flux in units of W m^-3 emitted to an observer
            located in the direction of alpha.
        """
        return integrate.dblquad(lambda theta, phi, lambd, T_ss, alpha:
                                 (self.calcFluxLambdaEmittedToObs(theta, phi, lambd, T_ss, alpha)
                                  * self.surfaceElement(theta, phi)),
                                 -np.pi/2 + alpha,
                                 np.pi/2 + alpha,
                                 lambda theta: -np.pi/2,
                                 lambda theta: np.pi/2,
                                 args=[lambd, T_ss, alpha])[0]

    