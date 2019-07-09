#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy import integrate

from ..iso import ISO 
from ..math import Math
from ..geographic import Geographic

xaxis = (1, 0, 0)
yaxis = (0, 1, 0)
zaxis = (0, 0, 1)

# Test ISO x, y, z locations
def test_ISO():
    system = ISO()
    np.testing.assert_almost_equal(xaxis,
                                   system.convertToCartesian(np.pi/2, 0, R=1))
    np.testing.assert_almost_equal(yaxis,
                                   system.convertToCartesian(np.pi/2, np.pi/2, R=1))
    np.testing.assert_almost_equal(zaxis,
                                   system.convertToCartesian(0, 0, R=1))

# Test ISO surface integration
def test_ISO_Integration():
    system = ISO()
    S = integrate.dblquad(system.surfaceElement,
                          system.coordinateRanges["phi"][0],
                          system.coordinateRanges["phi"][1],
                          lambda theta: system.coordinateRanges["theta"][0],
                          lambda theta: system.coordinateRanges["theta"][1])[0]
    np.testing.assert_almost_equal(4*np.pi, S) 

# Test ISO name
def test_ISO_Name():
    system = ISO()
    assert system.frameName == "ISO"
    system.frameName = "test"
    assert system.frameName == "test"

# Test Math x, y, z locations
def test_Math():
    system = Math()
    np.testing.assert_almost_equal(xaxis,
                                   system.convertToCartesian(0, np.pi/2, R=1))
    np.testing.assert_almost_equal(yaxis,
                                   system.convertToCartesian(np.pi/2, np.pi/2, R=1))
    np.testing.assert_almost_equal(zaxis,
                                   system.convertToCartesian(0, 0, R=1))

# Test Math surface integration
def test_Math_Integration():
    system = Math()
    S = integrate.dblquad(system.surfaceElement,
                          system.coordinateRanges["phi"][0],
                          system.coordinateRanges["phi"][1],
                          lambda theta: system.coordinateRanges["theta"][0],
                          lambda theta: system.coordinateRanges["theta"][1])[0]
    np.testing.assert_almost_equal(4*np.pi, S)
    
# Test Math name
def test_Math_Name():
    system = Math()
    assert system.frameName == "Math"
    system.frameName = "test"
    assert system.frameName == "test"

# Test Geograpic x, y, z locations
def test_Geographic():
    system = Geographic()
    np.testing.assert_almost_equal(xaxis,
                                   system.convertToCartesian(0, 0, R=1))
    np.testing.assert_almost_equal(yaxis,
                                   system.convertToCartesian(0, np.pi/2, R=1))
    np.testing.assert_almost_equal(zaxis,
                                   system.convertToCartesian(np.pi/2, np.pi/2, R=1))

# Test Geographic surface integration
def test_Geographic_Integration():
    system = Geographic()
    S = integrate.dblquad(system.surfaceElement,
                          system.coordinateRanges["phi"][0],
                          system.coordinateRanges["phi"][1],
                          lambda theta: system.coordinateRanges["theta"][0],
                          lambda theta: system.coordinateRanges["theta"][1])[0]
    np.testing.assert_almost_equal(4*np.pi, S)

# Test Geographic name
def test_Geographic_Name():
    system = Geographic()
    assert system.frameName == "Geographic"
    system.frameName = "test"
    assert system.frameName == "test"