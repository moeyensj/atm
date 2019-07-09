#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ..constants import Constants
from .observatory import Observatory

__all__ = ["SDSS"]

c = Constants.LIGHTSPEED


class SDSS(Observatory):

    def __init__(self):
        super().__init__(
            name="Sloan Digital Sky Survey",
            acronym="SDSS",
            filterNames=["u", "g", "r", "i",  "z"],
            filterEffectiveLambdas=np.array([
                3.543e-7,
                4.770e-7,
                6.231e-7,
                7.625e-7,
                9.134e-7,
            ])
        )