#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ..constants import Constants
from .observatory import Observatory

__all__ = ["WISE"]

c = Constants.LIGHTSPEED


class WISE(Observatory):

    def __init__(self):
        super().__init__(
            name="Wide-Field Infrared Survey Explorer",
            acronym="WISE",
            filterNames=["W1", "W2", "W3", "W4"],
            filterEffectiveLambdas=np.array([
                3.368e-6,
                4.618e-6,
                12.082e-6,
                22.194e-6
            ]),
            filterQuadratureLambdas=np.array([
                3.0974e-6,
                3.6298e-6,
                4.3371e-6,
                4.9871e-6,
                #8.0145e-6, # Wright
                8.6109e-6, # NM
                #11.495e-6, # Wright
                11.786e-6, # NM
                #15.256e-6, # Wright
                15.277e-6, # NM
                21.150e-6,
                24.690e-6
            ]),
            filterQuadratureNus=np.array([
                9.6788e+13,
                8.2592e+13,
                6.9123e+13,
                6.0114e+13,
                3.7406e+13,
                2.6080e+13,
                1.96514e+13,
                1.4175e+13,
                1.2142e+13]),
            fluxNuNorm=np.array([
                306.682,
                170.663,
                31.3684,
                7.9525
            ]),
            # Conversion from Vega magnitudes to AB 
            # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
            deltaMagnitudes=np.array([
                2.699,
                3.339,
                5.174,
                6.620
            ])
        )

    def bandpassLambda(self, F, args=[]):
        """
        Computes the WISE bandpass throughput for any arbitrary function of wavelength.
        For more details, see: http://adsabs.harvard.edu/abs/2013AAS...22143905W

        Parameters
        ----------
        F : function
            Any function whose first argument is lambda in meters.
        args : list
            List of arguments to pass to the function F.

        Returns
        -------
        `~numpy.ndarray` (N, 4)
            The throughput for each filter.
        """
        return np.array([
            0.5117*F(3.0974e-6, *args) + 0.4795*F(3.6298e-6, *args),
            0.5811*F(4.3371e-6, *args) + 0.4104*F(4.9871e-6, *args),
            #0.1785*F(8.0145e-6, *args) + 0.4920*F(11.495e-6, *args) + 0.2455*F(15.256e-6, *args),   # Wright
            0.1414*F(8.6109e-6, *args) + 0.4412*F(11.786e-6, *args) + 0.4174*F(15.277e-6, *args),   # NM 
            0.7156*F(21.150e-6, *args) + 0.2753*F(24.690e-6, *args)])

    def bandpassNu(self, F, args=[]):
        """
        Computes the WISE bandpass throughput for any arbitrary function of frequency.
        For more details, see: http://adsabs.harvard.edu/abs/2013AAS...22143905W

        Parameters
        ----------
        F : function
            Any function whose first argument is frequency in Hz.
        args : list
            List of arguments to pass to the function F.

        Returns
        -------
        `~numpy.ndarray` (N, 4)
            The throughput for each filter.
        """
        return np.array([
            0.5117*F(9.6788e+13, *args) + 0.4795*F(8.2592e+13, *args),
            0.5811*F(6.9123e+13, *args) + 0.4104*F(6.0114e+13, *args),
            0.1785*F(3.7406e+13, *args) + 0.4920*F(2.6080e+13, *args) + 0.2455*F(1.96514e+13, *args),
            0.7156*F(1.4175e+13, *args) + 0.2753*F(1.2142e+13, *args)])
