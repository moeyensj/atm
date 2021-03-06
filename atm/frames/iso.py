#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from .coordinate_frame import CoordinateFrame

__all__ = ["ISO"]


class ISO(CoordinateFrame):
    """
    ISO: The ISO spherical coordinate system often used in phyics.

    Theta is the polar angle and ranges from 0 to 180 degrees and is
    typically measured from the z-axis.

    Phi is the azimuthal angle and ranges from 0 to 360 degrees and
    is typically measured starting from the x-axis.

    """
    def __init__(self, frameName="ISO"):
        self._frameName = frameName
        self._coordinateRanges = {
            "theta": np.array([0, np.pi]),
            "phi": np.array([0, 2*np.pi])
            }

    def convertToCartesian(self, theta, phi, R=1):
        """
        Convert angular coordinates to cartesian x, y, z.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Polar angle in radians.
        phi : float or `~numpy.ndarray`
            Azimuthal angle in radians.
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
        x = R*np.cos(phi)*np.sin(theta)
        y = R*np.sin(phi)*np.sin(theta)
        z = R*np.cos(theta)
        return x, y, z

    def surfaceElement(self, theta, phi, R=1):
        """
        Calculate the surface element.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Polar angle in radians.
        phi : float or `~numpy.ndarray`
            Azimuthal angle in radians.
        R : float or `~numpy.ndarray`, optional
            Radius in arbitrary units.
            [Default = 1]

        Returns
        -------
        float or `~numpy.ndarray`
            The surface element dA.
        """
        return R**2 * np.sin(theta)
