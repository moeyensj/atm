#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__all__ = ["Constants"]


class Constants(object):
    """
    Constants: Stores required physical constants used
    throughout `atm`. These should not be changed.

    """
    # speed of light in in SI units
    LIGHTSPEED = 299792458.0

    # planck's constant in SI units
    PLANCK = 6.62607004e-34

    # boltzmann constant in SI units
    BOLTZMANN = 1.38064852e-23

    # stefan-boltzmann constant in SI units
    STEFAN_BOLTZMANN = 5.670367e-08

    # astronomical unit in SI units
    ASTRONOMICAL_UNIT = 1.495978707e11

    # solar radius in SI units
    SOLAR_RADIUS = 695700000

    # solar effective blackbody temperature in SI units
    SOLAR_TEMPERATURE = 5778

    # incident solar flux at 1 AU in SI units
    SOLAR_CONSTANT = 1.3608e3
