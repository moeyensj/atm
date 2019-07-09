#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from numpy import testing as test

from ...constants import Constants
from ...models import NEATM
from ...obs import WISE
from ..interpolate_flux_lambda import *
from ..flux_lambda import *
from ..hg import calcQ

c = Constants.LIGHTSPEED

# Test calcFluxAtObs for self-consistency
def test_interpFluxAtObs():
    # Prepare model and load tables
    model = NEATM()
    obs = WISE()
    lambd_interp = obs.filterQuadratureLambdas[0]
    model.loadLambdaTables([lambd_interp], verbose=False)

    # Create some fake observations
    num = 25
    delta = np.linspace(0.5, 5, num)
    r = np.linspace(0.5, 5, num)
    alpha = np.linspace(0.0, np.pi/2, num)
    D = np.linspace(10, 1e5, num)
    eps = np.linspace(0, 1, num)
    lambd_calc = np.ones(num) * lambd_interp
    T_ss = np.linspace(350, 500, num)
    test.assert_allclose(interpFluxLambdaAtObs(model, r, delta, lambd_interp, T_ss, D, alpha, eps), calcFluxLambdaAtObs(model, r, delta, lambd_calc, T_ss, D, alpha, eps))

    del model
    del obs
    return


# Test calcFluxAtObs for self-consistency
def test_interpFluxAtObsWithSunlight():
    # Prepare model and load tables
    model = NEATM()
    obs = WISE()
    lambd_interp = obs.filterQuadratureLambdas[0]
    model.loadLambdaTables([lambd_interp], verbose=False)

    # Create some fake observations
    num = 25
    delta = np.linspace(0.5, 5, num)
    r = np.linspace(0.5, 5, num)
    alpha = np.linspace(0.0, np.pi/2, num)
    D = np.linspace(10, 1e5, num)
    eps = np.linspace(0, 1, num)
    lambd_calc = np.ones(num) * lambd_interp
    T_ss = np.linspace(350, 500, num)
    G = np.ones(num) * 0.15
    p = (1 - eps) / calcQ(G)
    test.assert_allclose(
        interpFluxLambdaAtObsWithSunlight(model, r, delta, lambd_interp, T_ss, D, alpha, eps, p, G), 
        calcFluxLambdaAtObsWithSunlight(model, r, delta, lambd_calc, T_ss, D, alpha, eps, p, G))

    del model
    del obs
    return

# Test calcFluxObs for self-consistency
def test_interpFluxObs():
    # Prepare model and load tables
    model = NEATM()
    obs = WISE()
    model.loadLambdaTables(obs.filterQuadratureLambdas, verbose=False)

    # Create some fake observations
    num = 5
    delta = np.linspace(0.5, 5, num)
    r = np.linspace(0.5, 5, num)
    alpha = np.linspace(0.0, np.pi/2, num)
    D = np.linspace(10, 1e5, num)
    eps = np.linspace(0, 1, num)
    T_ss = np.linspace(350, 500, num)
    test.assert_allclose(interpFluxLambdaObs(model, obs, r, delta, T_ss, D, alpha, eps), calcFluxLambdaObs(model, obs, r, delta, T_ss, D, alpha, eps))

    del model
    del obs
    return


# Test calcFluxObs for self-consistency
def test_interpFluxObsWithSunlight():
    # Prepare model and load tables
    model = NEATM()
    obs = WISE()
    model.loadLambdaTables(obs.filterQuadratureLambdas, verbose=False)
    
    # Create some fake observations
    num = 5
    delta = np.linspace(0.5, 5, num)
    r = np.linspace(0.5, 5, num)
    alpha = np.linspace(0.0, np.pi/2, num)
    D = np.linspace(10, 1e5, num)
    eps = np.linspace(0, 1, num)
    T_ss = np.linspace(350, 500, num)
    G = np.ones(num) * 0.15
    p = (1 - eps) / calcQ(G)
    test.assert_allclose(
        interpFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G), 
        calcFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G))

    del model
    del obs
    return
