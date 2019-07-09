#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from numpy import testing as test

from ...constants import Constants
from ..model import Model
from ..stm import STM
from ..frm import FRM
from ..neatm import NEATM

c = Constants.LIGHTSPEED

# Test STM calcT for subsolarPoint
def test_calcT_STM():
    model = STM()
    test.assert_allclose(0.0, model.calcT(0.0, 0.0))
    test.assert_allclose(400.0, model.calcT(0.0, 400.0))

# Test FRM calcT for subsolarPoint
def test_calcT_FRM():
    model = FRM()
    test.assert_allclose(0.0, model.calcT(0.0, 0.0))
    test.assert_allclose(400.0, model.calcT(0.0, 400.0))

# Test FRM calcT for subsolarPoint
def test_calcT_NEATM():
    model = NEATM()
    test.assert_allclose(0.0, model.calcT(0.0, 0.0, 0.0))
    test.assert_allclose(400.0, model.calcT(0.0, 0.0, 400.0))

# Test buildLambdaTables, loadLambdaTables and interpTotalFluxLambdaEmitted
def test_lambdaTables():
    for modeli in [STM, FRM, NEATM]:
        for lambd in [1.1111e-6, 33.333e-6]:
            model = modeli()
            model.buildLambdaTables([lambd],
                                    tRange=[400, 500],
                                    tStep=25,
                                    alphaRange=[0, np.pi/2],
                                    alphaStep=np.pi/8,
                                    threads=4,
                                    verbose=True)

            model = modeli()
            model.loadLambdaTables([lambd])
            test.assert_allclose(model.interpTotalFluxLambdaEmittedToObs(lambd, 400, 0), model.calcTotalFluxLambdaEmittedToObs(lambd, 400, 0))
            test.assert_allclose(model.interpTotalFluxLambdaEmittedToObs(lambd, 450, np.pi/2), model.calcTotalFluxLambdaEmittedToObs(lambd, 450, np.pi/2))
            test.assert_allclose(model.interpTotalFluxLambdaEmittedToObs(lambd, 500, np.pi/4), model.calcTotalFluxLambdaEmittedToObs(lambd, 500, np.pi/4))

            # Remove the tables
            os.remove(os.path.join(model.tableDir, os.path.join(model.tableLambdaFiles[lambd])))

# Test base model class instantiation for user-defined functions
def test_baseModel():
    model = Model(name="test", acronym="t", tableDir="..")

    # Test instantiation
    assert model.name == "test"
    assert model.acronym == "t"
    assert model.tableDir == ".."

    # Test setters and deleters
    del model.name 
    assert model.name == None
    model.name = "test"
    assert model.name == "test"

    del model.acronym 
    assert model.acronym == None
    model.acronym = "t"
    assert model.acronym == "t"

    # Test user-defined functions
    with test.assert_raises(NotImplementedError):
        model.calcT()

    with test.assert_raises(NotImplementedError):
        model.calcFluxLambdaEmitted()

    with test.assert_raises(NotImplementedError):
        model.calcTotalFluxLambdaEmitted()

    with test.assert_raises(NotImplementedError):
        model.calcFluxLambdaEmittedToObs()

    with test.assert_raises(NotImplementedError):
        model.calcTotalFluxLambdaEmittedToObs()
