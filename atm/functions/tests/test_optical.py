#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from numpy import testing as test

from ..optical import calcH
from ..optical import calcD
from ..optical import calcPv

# Test calcH
def test_calcH():
    # Data taken from:
    # Reference
    # ---------
    # Alan W. Harris and Alan W. Harris, 1997: On the Revision of Radiometric Albedos and Diameters of Asteroids
    #    https://www.sciencedirect.com/science/article/pii/S001910359695664X?via%3Dihub
    D = np.array([99.66e3, 19.7e3, 11.19e3, 101.60e3, 5.09e3])
    p_v = np.array([0.167, 0.220, 0.354, 0.062, 0.098])
    H = np.array([7.57, 10.79, 11.50, 8.60, 14.60])

    test.assert_allclose(H, calcH(D, p_v), rtol=0.01)

# Test calcD
def test_calcD():
    # Data taken from:
    # Reference
    # ---------
    # Alan W. Harris and Alan W. Harris, 1997: On the Revision of Radiometric Albedos and Diameters of Asteroids
    #    https://www.sciencedirect.com/science/article/pii/S001910359695664X?via%3Dihub
    D = np.array([99.66e3, 19.7e3, 11.19e3, 101.60e3, 5.09e3])
    p_v = np.array([0.167, 0.220, 0.354, 0.062, 0.098])
    H = np.array([7.57, 10.79, 11.50, 8.60, 14.60])

    test.assert_allclose(D, calcD(H, p_v), rtol=0.01)

# Test calcPv
def test_calcPv():
    # Data taken from:
    # Reference
    # ---------
    # Alan W. Harris and Alan W. Harris, 1997: On the Revision of Radiometric Albedos and Diameters of Asteroids
    #    https://www.sciencedirect.com/science/article/pii/S001910359695664X?via%3Dihub
    D = np.array([99.66e3, 19.7e3, 11.19e3, 101.60e3, 5.09e3])
    p_v = np.array([0.167, 0.220, 0.354, 0.062, 0.098])
    H = np.array([7.57, 10.79, 11.50, 8.60, 14.60])

    test.assert_allclose(p_v, calcPv(D, H), rtol=0.01)

   
