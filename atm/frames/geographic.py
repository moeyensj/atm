#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from .coordinate_frame import CoordinateFrame

__all__ = ["Geographic"]


class Geographic(CoordinateFrame):
    """
    Geographic: The geographic coordinate system
    often used for planetary bodies.

    Theta is the coordinate of latitude and ranges from
    -90 to 90 degrees and is centered on the equator.

    Phi is the coordinate of longitude and ranges from
    -180 to 180 degrees and is centered on the intersection
    of the meridian and the equator.
    """
    def __init__(self, frameName="Geographic"):
        self._frameName = frameName
        self._coordinateRanges = {
            "theta": np.array([-np.pi/2, np.pi/2]),
            "phi":  np.array([-np.pi, np.pi])
            }

    def convertToCartesian(self, theta, phi, R=1):
        """
        Convert angular coordinates to cartesian x, y, z.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Latitude in radians.
        phi : float or `~numpy.ndarray`
            Longitude in radians.
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
        x = R*np.cos(phi)*np.cos(theta)
        y = R*np.sin(phi)*np.cos(theta)
        z = R*np.sin(theta)
        return x, y, z

    def surfaceElement(self, theta, phi, R=1):
        """
        Calculate the surface element.

        Parameters
        ----------
        theta : float or `~numpy.ndarray`
            Latitude in radians.
        phi : float or `~numpy.ndarray`
            Longitude in radians.
        R : float or `~numpy.ndarray`, optional
            Radius in arbitrary units.
            [Default = 1]

        Returns
        -------
        float or `~numpy.ndarray`
            The surface element dA.
        """
        return R**2 * np.cos(theta)
