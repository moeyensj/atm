#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ...constants import Constants
from ..temperature import calcTss
from ..temperature import calcT1

# Test calcTss and calcT1 against each other
def test_calcTss_calcT1():
    T_ss = calcTss(2, 0, 1, 0.15, 1)
    T1 = calcT1(2, 0, 1, 0.15, 1)
    np.testing.assert_allclose(np.sqrt(2)*T_ss, T1)
    