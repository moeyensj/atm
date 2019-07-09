#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from astropy import constants as C
from astropy import units as u
from astropy.modeling.blackbody import blackbody_lambda
from astropy.modeling.blackbody import blackbody_nu

from ..blackbody import calcPlanckLambda
from ..blackbody import calcPlanckNu

# Test calcPlanckLambda against Astropy
def test_calcPlanckLambda():
    lambd = np.linspace(400*10**-9, 700*10**-9, 1000)
    astropyPlanck = blackbody_lambda(lambd*u.m, 5778*u.K).to(u.W*u.rad**-2*u.m**-3).value
    np.testing.assert_allclose(calcPlanckLambda(lambd, 5778),
                               astropyPlanck,
                               rtol=1e-5)

# Test calcPlanckNu against Astropy
def test_calcPlanckNu():
    nu = np.linspace(C.c.value/(400*10**-9), C.c.value/(700*10**-9), 1000)
    astropyPlanck = blackbody_nu(nu*u.s**-1, 5778*u.K).to(u.W*u.rad**-2*u.m**-2*u.s).value
    np.testing.assert_allclose(calcPlanckNu(nu, 5778),
                         astropyPlanck, 
                         rtol=1e-5)