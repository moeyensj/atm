#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ..hg import calcHG
from ..hg import calcQ

# Check zero case is equal to one
def test_calcHG():
    np.testing.assert_allclose(calcHG(0,0), 1.0)

# Check zero slope parameter case is approximately
# equal to constant term
def test_calcQ():
    constTerm = np.round(calcQ(0), decimals=4)
    np.testing.assert_allclose(constTerm, 0.2856)