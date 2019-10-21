#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

__all__ = ["Config"]


class Config(object):
    """
    Config: Holds configuration settings.

    Parameters
    ----------
    fitParameters : list
        Parameters to fit.
    parameterPriors : dict
        Dictionary with parameters as keys, and a dictionary
        as the value for each key. This dictionary is called
        to setup the pymc3 priors for each parameter not in
        fitParameters.
    columnMapping : dict
        This dictionary should define the
        column names of the user's data relative to the
        internally used names.
    tableParameterLimits : dict
        This is dictionary is called
        when building model tables to set the grid in subsolar
        temperature and phase angle.  It should have 'T_ss' and
        'alpha' as keys. Values should be a list:
        the first element should be another list
        with the lower and upper bounds, the second element
        should be the step size.
    threads : int
        The number of threads to use when bulding model tables
        and running the multi-fit script.
    numSamples : int
        Number of samples to draw from the posterior distribution.
    numBurnIn : int
        Number of the drawn samples to discard from summary statistics
        and plotting.
    numChains : int
        Number of Markov chains to sample the posterior distribution.
    phaseAngleFluxCorrection : float
        The default value to correct for phase-angle effects in the
        Standard Thermal Model. The canonical value is 0.01.
    verbose : bool
        Print progress statements?
    """
    fitParameters = ["logT1", "logD", "eps"]
    
    parameterPriors = {
        "logD": {
            "lower": 1,
            "upper": 8,
            },
        "eps": {
            "lower": 0.0,
            "upper": 1.0},
        "logT1": {
            "lower": 0.01,
            "upper": 5,
            },
        "T_ss":  {
            "lower": 10,
            "upper": 1200.0
            },
        "alpha_rad": {
            "lower": 0,
            "upper": np.pi
            },
        "r_au": {
            "lower": 0,
            "upper": 10
            },
        "delta_au": {
            "lower": 0,
            "upper": 10
            },
        "G": {
            "lower": 0,
            "upper": 1},
        "p": {
            "lower": 0,
            "upper": 5
            },
        "eta": {
            "lower": 0,
            "upper": 10
            }
        }
    
    columnMapping = {
        "designation" : "designation",
        "obs_id": "obs_id",
        "exp_mjd": "mjd",
        "r_au": "r_au",
        "delta_au": "delta_au",
        "alpha_rad": "alpha_rad",
        "G": "G",
        "logD": "logD",
        "logT1" : "logT1",
        "eta": "eta",
        "eps": "eps",
        "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
        "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
        "mag" : ["mag_W1", "mag_W2", "mag_W3", "mag_W4"],
        "magErr" : ["magErr_W1", "magErr_W2", "magErr_W3", "magErr_W4"]
        }
    
    tableParameterLimits = {
        "T_ss": [[100.0, 1200.0], 0.5],
        "alpha": [[0.0, np.pi], np.pi/360]
        }
    
    threads = 10
    samples = 2500
    burnInSamples = 500
    chains = 20
    phaseAngleFluxCorrection = 0.01
    verbose = True
