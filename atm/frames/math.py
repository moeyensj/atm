#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from .coordinate_frame import CoordinateFrame

__all__ = ["Math"]


class Math(CoordinateFrame):
    """
    Math: The spherical coordinate system often used in mathematical
    applications.

    Theta is the azimuthal angle and ranges from 0 to 360 degrees
    and is measured from the x-axis.

    Phi is the polar angle and ranges from 0 to 180 degrees and is
    measured from the z-axis.

    """
    def __init__(self, frameName="Math"):
        self._frameName = frameName
        self._coordinateRanges = {
            "theta": np.array([0, 2*np.pi]),
            "phi": np.array([0, np.pi])
        }

    def convertToCartesian(self, theta, phi, R=1):
        """
        Convert angular coordinates to cartesian x, y, z.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Azimuthal angle in radians.
        phi : float or `~numpy.ndarray`
            Polar angle in radians.
        R : float or `~numpy.ndarray`, optional
            Radius in arbitrary units.
            [Default = 1]

        Returns
        -------
        float or `~numpy.ndarray`
            X coordinate in the units of R.
        float or `~numpy.ndarray`
            Y coordinate in the units of R.
        float or `~numpy.ndarray`
            Z coordinate in the units of R.
        """
        x = R*np.cos(theta)*np.sin(phi)
        y = R*np.sin(theta)*np.sin(phi)
        z = R*np.cos(phi)
        return x, y, z

    def surfaceElement(self, theta, phi, R=1):
        """
        Calculate the surface element.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Azimuthal angle in radians.
        phi : float or `~numpy.ndarray`
            Polar angle in radians.
        R : float or `~numpy.ndarray`, optional
            Radius in arbitrary units.
            [Default = 1]

        Returns
        -------
        float or `~numpy.ndarray`
            The surface element dA.
        """
        return R**2 * np.sin(phi)
