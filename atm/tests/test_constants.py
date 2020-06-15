#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from astropy.constants import astropyconst20 as c

from ..constants import Constants

# Test speed of light is consistent with Astropy
def test_lightspeed():
    np.testing.assert_equal(Constants.LIGHTSPEED, c.c.si.value)

# Test Planck's constant is consistent with Astropy
def test_planck():
    np.testing.assert_equal(Constants.PLANCK, c.h.si.value)

# Test Boltzmann's constant is consistent with Astropy
def test_boltzmann():
    np.testing.assert_equal(Constants.BOLTZMANN, c.k_B.si.value)

# Test Stefan-Boltzmann's constant is consistent with Astropy
def test_stefan_boltzmann():
    np.testing.assert_equal(Constants.STEFAN_BOLTZMANN, c.sigma_sb.si.value)

# Test an AU is consistent with Astropy
def test_astronomical_unit():
    np.testing.assert_equal(Constants.ASTRONOMICAL_UNIT, c.au.si.value)

# Test solar radius is consistent with Astropy
def test_constants_solar_radius():
    np.testing.assert_equal(Constants.SOLAR_RADIUS, c.R_sun.si.value)
