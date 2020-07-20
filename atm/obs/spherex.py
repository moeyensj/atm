#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ..constants import Constants
from .observatory import Observatory

__all__ = ["SPHEREx"]


class SPHEREx(Observatory):

    def __init__(self):
        super().__init__(
            name="Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer",
            acronym="SPHEREx",
            filterNames=["s{:02d}".format(i+1) for i in range(96)],
            filterQuadratureLambdas=np.array([
                7.500000e-07, 7.682927e-07, 7.870315e-07, 8.062274e-07,
                8.258915e-07, 8.460352e-07, 8.666702e-07, 8.878085e-07,
                9.094624e-07, 9.316444e-07, 9.543674e-07, 9.776447e-07,
                1.001490e-06, 1.025916e-06, 1.050939e-06, 1.076571e-06,
                1.137073e-06, 1.164807e-06, 1.193217e-06, 1.222319e-06,
                1.252132e-06, 1.282672e-06, 1.313957e-06, 1.346004e-06,
                1.378834e-06, 1.412464e-06, 1.446914e-06, 1.482205e-06,
                1.518356e-06, 1.555389e-06, 1.593325e-06, 1.632187e-06,
                1.680000e-06, 1.720976e-06, 1.762951e-06, 1.805949e-06,
                1.849997e-06, 1.895119e-06, 1.941341e-06, 1.988691e-06,
                2.037196e-06, 2.086883e-06, 2.137783e-06, 2.189924e-06,
                2.243337e-06, 2.298052e-06, 2.354102e-06, 2.411520e-06,
                2.489143e-06, 2.560261e-06, 2.633412e-06, 2.708652e-06,
                2.786042e-06, 2.865643e-06, 2.947519e-06, 3.031733e-06,
                3.118354e-06, 3.207450e-06, 3.299092e-06, 3.393351e-06,
                3.490304e-06, 3.590027e-06, 3.692600e-06, 3.798102e-06,
                3.854727e-06, 3.889770e-06, 3.925132e-06, 3.960815e-06,
                3.996822e-06, 4.033157e-06, 4.069822e-06, 4.106820e-06,
                4.144155e-06, 4.181829e-06, 4.219846e-06, 4.258208e-06,
                4.296919e-06, 4.335982e-06, 4.375400e-06, 4.415176e-06,
                4.454000e-06, 4.488262e-06, 4.522787e-06, 4.557577e-06,
                4.592636e-06, 4.627964e-06, 4.663563e-06, 4.699437e-06,
                4.735586e-06, 4.772014e-06, 4.808722e-06, 4.845712e-06,
                4.882987e-06, 4.920548e-06, 4.958398e-06, 4.996540e-06
            ]),
        )

    def bandpassLambda(self, F, args=[]):
        """
        Computes the SPHEREx bandpass throughput for any arbitrary function of wavelength.

        Parameters
        ----------
        F : function
            Any function whose first argument is lambda in meters.
        args : list
            List of arguments to pass to the function F.

        Returns
        -------
        `~numpy.ndarray` (96, N)
            The throughput for each filter.
        """
        tp = np.zeros((len(args[0]), len(self.filterQuadratureLambdas)))
        for i in range(len(self.filterQuadratureLambdas)):
            tp_i = F(self.filterQuadratureLambdas[i], *args)
            tp[:, i] = tp_i.T
        return tp.T