#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__all__ = ["CoordinateFrame"]

class CoordinateFrame(object):
    """
    CoordinateFrame: Base coordinate frame class.

    Parameters
    ----------
    frameName : str
        Name of the coordinate system.
    coordinateRanges : dict
        Dictionary with coordinate names as keys and two-element lists
        containing the lower and upper limits for the coordinate
        as values.
    """
    def __init__(self, frameName="CoordinateFrame", coordinateRanges={}):
        self._frameName = frameName
        self._coordinateRanges = {}

    @property
    def frameName(self):
        return self._frameName

    @frameName.setter
    def frameName(self, value):
        self._frameName = value

    @frameName.deleter
    def frameName(self):
        del self._frameName

    @property
    def coordinateRanges(self):
        return self._coordinateRanges

    @coordinateRanges.setter
    def coordinateRanges(self, value):
        self._coordinateRanges = value

    @coordinateRanges.deleter
    def coordinateRanges(self):
        del self._coordinateRanges

    def convertToCartesian(self):
        """
        This method should convert the coordinates from this frame
        to cartesian coordinates. This method is particularly used for plotting
        purposes.

        Throws NotImplementedError if not defined.
        """
        raise NotImplementedError("convertToCartesian is not defined.")

    def surfaceElement(self):
        """
        This method should calculate the surface element for this
        coordinate frame. This method is used when integrating over
        the surface of an asteroid.

        Throws NotImplementedError if not defined.
        """
        raise NotImplementedError("surfaceElement is not defined.")
