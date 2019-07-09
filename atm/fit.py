#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import platform
if platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import os
import time
import corner
import warnings
import numpy as np
import pymc3 as pm
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
import theano.tensor as tt
from theano import shared
from theano.compile import as_op

from .config import Config
from .helpers import __handleParameters
from .functions import calcQ
from .functions import interpFluxLambdaObsWithSunlight
from .functions import interpFittedFluxLambdaAndMagObs
from .analysis import _median
from .analysis import _sigmaG

__all__ = ["fit",
           "multiFit"]

def fit(model, obs, data, 
        fitParameters=Config.fitParameters,
        fitFilters="all",
        emissivitySpecification=None,
        albedoSpecification="auto",
        chains=Config.chains,
        samples=Config.samples,
        burnInSamples=Config.burnInSamples,
        threads=Config.threads,
        priorFunction=pm.Uniform,
        parameterPriors=Config.parameterPriors,
        plotTrace=True,
        plotCorner=True,
        scaling=0.1,
        progressBar=False,
        fitCode=1,
        saveDir=None,
        returnFigs=False,
        figKwargs={"dpi": 200},
        columnMapping=Config.columnMapping,
        verbose=Config.verbose):
    """
    Fit observed data with a thermal model. 
    
    Missing flux values in data should be set to NaN, they will be replaced with zeros.
    Their corresponding errors should be set to NaN, they will be replaced with 100000x the maximum flux
    error found in data.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object with the approprate tables loaded into memory.
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    data : `~pandas.DataFrame`
        DataFrame containing the relevant data to fit. The user should define 
        the columnMapping dictionary which maps internally used variables to the 
        variables used in the user's DataFrame.
    fitParameters : list, optional
        The parameters that should be fit for. If a fit parameter is found to 
        exist inside the data, that column will be ignored for fitting purposes. 
        [Default = `~atm.Config.fitParameters`]
    fitFilters : {"all", list}, optional
        To fit using data from all filters set to "all". To fit using only data
        from certain filters, set this parameter to a list with the names of the filters
        to use.
        [Default = "all"]
    emissivitySpecification : {None, "perBand", dict, "auto"}, optional
        There are for different emissivity scenarios supported for fitting:
            1) Setting the emissivity specification to None forces the fitter to
            use a single epsilon across all bands.
            2) Setting the emissivity specification to "perBand" forces the fitter
            to use an epsilon per band. In the case of WISE, this would mean having
            four epsilons: eps_W1, eps_W2, eps_W3, eps_W4
            3) Setting the emissivity specification to a dictionary allows the fitter
            to use combination emissivities. For example, passing 
                {"eps_W1W2": ["W1", "W2"],
                 "eps_W3W4": ["W3", "W4"]}
            tells the fitter that there should be two epsilon parameters. One that constrains
            emissivity in filters W1 and W2, and one that constrains emissivity 
            in filters W3 and W4. The data format should be a dictionary with the combination 
            parameter(s) as key(s) and lists of the filter names as values. The combination
            parameter name should be formatted as in the example: "eps_{filter names}".
            (Again, using WISE as the example.)
            4) Setting the emissivity specification to "auto" forces fitting to use the albedo
            specification definition and creates emissivity parameters that obey Kirchoff's
            law. Both emissivity specification and albedo specification cannot be "auto".
        [Default = None]
    albedoSpecification : {None, "perBand", dict, "auto"} optional
        There are three different albedo scenarios supported for fitting:
            1) Setting the albedo specification to None forces the fitter to
            use a single p across all bands.
            2) Setting the albedo specification to "perBand" forces the fitter
            to use an epislon per band. In the case of WISE, this would mean having
            four epsilons: p_W1, p_W2, p_W3, p_W4
            3) Setting the albedo specification to a dictionary allows the fitter
            to use combination albedos. For example, passing 
                {"p_W1W2": ["W1", "W2"],
                 "p_W3W4": ["W3", "W4"]}
            tells the fitter that there should be two albedo parameters. One that constrains
            reflectivity in filters W1 and W2, and one that constrains reflectivity 
            in filters W3 and W4. The data format should be a dictionary with the combination 
            parameter(s) as key(s) and lists of the filter names as values. The combination
            parameter name should be formatted as in the example: "p_{filter names}".
            (Again, using WISE as the example.)
            4) Setting the albedo specification to "auto" forces fitting to use the emissivity
            specification definition and creates albedo parameters that obey Kirchoff's
            law. Both emissivity specification and albedo specification cannot be "auto".
        [Default = "auto"]
    chains : int, optional
        The number of chains to fit with. 
        [Default = `~atm.Config.chains`]
    samples : int, optional
        The number of samples to draw from the posterior. 
        [Default = `~atm.Config.samples`]
    burnInSamples : int, optional
        The number of samples to discard for the statistical summary
        and plotting.
        [Default = `~atm.Config.burnInSamples]
    threads : int, optional
            Number of processors to use.
            [Default = `atm.Config.threads`]
    priorFunction : `~pymc3.distributions`, optional
        Pymc3 distribution from which the priors should be drawn.
        [Default = `~pymc3.distributions.continuous.Uniform`]
    parameterPriors : dict, optional
        A nested dictionary with parameters as keys, and dictionaries as values.
        The inner dictionaries should have kwargs that belong to the priorFunction as keys
        and their desired values as values. For example, if logD is a fitParameter and parameterPriors
        is defined as below,  then a uniform prior will be initialized with a 
        lower bound of 1 and upper boundof 6. 
        parameterPriors = {
        "logD": {
            "lower": 1,
            "upper": 6,
            },
        "eps": {
            "lower": 0.0,
            "upper": 1.0},
        ...}
        [Default = `~Config.parameterPriors`]
    plotTrace : bool, optional
        Plot traces for each of the fit parameters.
        [Default = True]
    plotCorner : bool, optional
        Plot corner plot. 
        [Default = True]
    scaling : float, optional
        Scaling parameter for the MCMC sampler.
        [Default = 0.1]
    progressBar : bool, optional
        Display progress bar.
        [Default = False]
    fitCode : {int, float, str}, optional
        fitCode to destinguish between different fit runs. Used when
        multiFit calls this funcion. 
        [Default = 1]
    saveDir : {str, None}, optional
        If not None, will save plots inside saveDir.
        [Default = None]
    returnFigs : bool, optional
        Return Matplotlib figure objects.
        [Default = False]
    figKwargs : dict, optional
        Keyword arguments to pass to figure API.
        [Default = {"dpi" : 200}]
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    verbose : bool, optional
        Print progress statements?
        [Default = `~atm.Config.verbose`]

    Returns
    -------
    summary : `~pandas.DataFrame`
        A pandas DataFrame containing summary statistics (from pymc3)
    model_observations : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model.
    pymc_objs : (`~pymc3.model.Model`, `~pymc3.MultiTrace`)
        A tuple containing the pymc3 model and the pymc3 trace. 
    figures : ['~matplotlib.figure.Figure', `~matplotlib.figure.Figure`]
        Trace and corner Figure objects, only if returnFigs is True.
    """
    # Start timer
    time_start = time.time()

    # Load lambda tables
    model.loadLambdaTables(obs.filterQuadratureLambdas, verbose=verbose)
    
    # Handle parameters
    fitParametersSet, parametersSet, emissivityParameters, albedoParameters, dataParametersToIgnoreSet = __handleParameters(
        obs,
        fitParameters, 
        data.columns.tolist(), 
        emissivitySpecification=emissivitySpecification, 
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping)
    
    # Make sure there is only one unique object in data
    if data[columnMapping["designation"]].nunique() != 1:
        raise ValueError("More than one object found in observations! Cannot proceed.")
    designation = data[columnMapping["designation"]].unique()[0]
    
    print("Fitting {} with {}...".format(designation, model.acronym))
    print("Fit Code: {}".format(fitCode))
    print("Fit Parameters: {}".format(fitParameters))
    print("Emissivity Parameters: {}".format(np.unique(np.array(emissivityParameters))))
    print("Emissivity Specification: {}".format(emissivitySpecification))
    print("Albedo Parameters: {}".format(np.unique(np.array(albedoParameters))))
    print("Albedo Specification: {}".format(albedoSpecification))
    print("Number of observations: {}".format(len(data)))
    if len(dataParametersToIgnoreSet) != 0:
        print("Ignoring data parameters: {}".format(list(dataParametersToIgnoreSet)))
    print("Chains: {}".format(chains))
    print("Samples per Chain: {}".format(samples + burnInSamples))
    print("Burn-in Samples: {}".format(burnInSamples))
    print("Effective Samples per Chain: {}".format(samples))
    print("Threads: {}".format(threads))
    
    # Create filter toggle for slicing flux array
    if fitFilters == "all":
        filterToggle = np.arange(0, len(obs.filterNames))
    elif type(fitFilters) == list:
        filterToggle = []
        for i, f in enumerate(obs.filterNames):
            if f in set(fitFilters):
                filterToggle.append(i)
    else:
        raise ValueError("fitFilters should be either 'all' or a list of the filters (in order of increasing effective wavelength) to fit")

    # Get fluxes and flux errors
    #Y_toggle = data[columnMapping["flux_si"]].isna().values.astype(int)
    #Y = np.ma.array(data[columnMapping["flux_si"]].values, mask=Y_toggle)
    #Y_err = np.ma.array(data[columnMapping["fluxErr_si"]].values, mask=Y_toggle)
    Y = np.nan_to_num(data[columnMapping["flux_si"]].values) 
    Y_err = np.where(np.isnan(data[columnMapping["fluxErr_si"]].values),
                     100000*np.nanmax(data[columnMapping["fluxErr_si"]].values), 
                     data[columnMapping["fluxErr_si"]].values)

    # Test that flux and flux error arrays are the correct shape
    obs._testshape(Y)
    obs._testshape(Y_err)

    # Set up model
    pymc_model = pm.Model()

    with pymc_model:

        if "r_au" in parametersSet and "r_au" not in fitParametersSet:
            r = pm.Deterministic("r_au", shared(data[columnMapping["r_au"]].values))
        else: 
            r = priorFunction("r_au", shape=(len(Y),), **parameterPriors["r_au"])
            # If len(y) == 1, then the parameter wont cast to a vector, lets
            # explicitly do that
            r.type = tt.dvector

        if "delta_au" in parametersSet and "delta_au" not in fitParametersSet:
            delta = pm.Deterministic("delta_au", shared(data[columnMapping["delta_au"]].values))
        else: 
            delta = priorFunction("delta_au", shape=(len(Y),), **parameterPriors["delta_au"])
            # If len(y) == 1, then the parameter wont cast to a vector, lets
            # explicitly do that
            delta.type = tt.dvector

        if "alpha_rad" in parametersSet and "alpha_rad" not in fitParametersSet:
            alpha = pm.Deterministic("alpha_rad", shared(data[columnMapping["alpha_rad"]].values))
        else: 
            alpha = priorFunction("alpha_rad", shape=(len(Y),), **parameterPriors["alpha_rad"])
            # If len(y) == 1, then the parameter wont cast to a vector, lets
            # explicitly do that
            alpha.type = tt.dvector

        if "G" in parametersSet and "G" not in fitParametersSet:
            G = pm.Deterministic("G", shared(data[columnMapping["G"]].values[0]))
        else: 
            G = priorFunction("G", **parameterPriors["G"])
            # If len(y) == 1, then the parameter wont cast to a vector, lets
            # explicitly do that
            G.type = tt.dvector

        if "logD" in parametersSet and "logD" not in fitParametersSet:
            logD = pm.Deterministic("logD", shared(data[columnMapping["logD"]].values[0]))
        else: 
            logD = priorFunction("logD", **parameterPriors["logD"])

        if "logT1" in parametersSet and "logT1" not in fitParametersSet:
            logT1 = pm.Deterministic("logT1", shared(data[columnMapping["logT1"]].values[0]))
        else: 
            logT1 = priorFunction("logT1", **parameterPriors["logT1"])

        T_ss = pm.Deterministic("T_ss", 10**logT1 / np.sqrt(r))

        # Initialize emissivity parameter arrays and dictionaries
        eps = []
        emissivityParamTracker = {}
        p = []
        albedoParamTracker = {}

        if emissivityParameters == "eps" and emissivitySpecification != "auto":
            if "eps" in parametersSet and "eps" not in fitParametersSet:
                eps = pm.Deterministic("eps", shared(data[columnMapping["eps"]].values[0]))
            else: 
                eps = priorFunction("eps", **parameterPriors["eps"])

            if albedoSpecification == "auto":
                p = pm.Deterministic(albedoParameters, (1 - eps) / calcQ(G))

        # If the emissivityParameters is a list (this list needs to have length equal to the 
        # number of bandpasses in the observatory class)
        if type(emissivityParameters) is list and emissivitySpecification != "auto":
            for i, emissivity in enumerate(emissivityParameters):
                eps_p = None
                p_p = None

                # Is the emissivity parameter a fit parameter? 
                if emissivity in fitParametersSet:
                    # We have not defined the parameter yet, so lets do that
                    if emissivity not in emissivityParamTracker.keys():
                        eps_p = priorFunction(emissivity, **parameterPriors["eps"])
                        emissivityParamTracker[emissivity] = eps_p

                    # We have already created the parameter, lets grab it
                    else:
                        eps_p = emissivityParamTracker[emissivity]
                # The emissivity parameter is not a fit parameter
                else:
                    # We have not defined the parameter yet, so lets do that
                    if emissivity not in emissivityParamTracker.keys():
                        eps_p = pm.Deterministic(emissivity, shared(data[emissivity].values[0]))
                        emissivityParamTracker[emissivity] = eps_p
                    # We have already created the parameter, lets grab it
                    else:
                        eps_p = emissivityParamTracker[emissivity]

                if albedoSpecification == "auto":
                    albedo = albedoParameters[i]
                    if albedo not in albedoParamTracker.keys():
                        p_p = pm.Deterministic(albedo, (1 - eps_p) / calcQ(G))
                        albedoParamTracker[albedo] = p_p
                    else:
                        p_p = albedoParamTracker[albedo]

                    p.append([p_p])
                eps.append([eps_p])
            eps = tt.stack(eps)
            
            if albedoSpecification == "auto":
                p = tt.stack(p)

        if albedoParameters == "p" and albedoSpecification != "auto":
            if "p" in parametersSet and "p" not in fitParametersSet:
                p = pm.Deterministic("p", shared(data[columnMapping["p"]].values[0]))
            else: 
                p = priorFunction("p", **parameterPriors["p"])

            if emissivitySpecification == "auto":
                eps = pm.Deterministic(emissivityParameters, 1 - p * calcQ(G))

        # If the albedoParameters is a list (this list needs to have length equal to the 
        # number of bandpasses in the observatory class)
        if type(albedoParameters) is list and albedoSpecification != "auto":
            for albedo in albedoParameters:
                eps_p = None
                p_p = None

                # Is the albedo parameter a fit parameter? 
                if albedo in fitParametersSet:
                    # We have not defined the parameter yet, so lets do that
                    if albedo not in albedoParamTracker.keys():
                        p_p = priorFunction(albedo, **parameterPriors["p"])
                        albedoParamTracker[albedo] = p_p
                    # We have already created the parameter, lets grab it
                    else:
                        p_p = albedoParamTracker[albedo]
                # The albedo parameter is not a fit parameter
                else:
                    # We have not defined the parameter yet, so lets do that
                    if albedo not in albedoParamTracker.keys():
                        p_p = pm.Deterministic(albedo, shared(data[albedo].values[0]))
                        albedoParamTracker[albedo] = p_p
                    # We have already created the parameter, lets grab it
                    else:
                        p_p = albedoParamTracker[albedo]

                if emissivitySpecification == "auto":
                    emissivity = emissivityParameters[i]
                    if emissivity not in emissivityParamTracker.keys():
                        eps_p = pm.Deterministic(emissivity, 1 - p_p * calcQ(G))
                        emissivityParamTracker[emissivity] = eps_p
                    else:
                        eps_p = emissivityParamTracker[emissivity]

                    eps.append([eps_p])
                p.append([p_p])      
            p = tt.stack(p)
            
            if emissivitySpecification == "auto":
                eps = tt.stack(eps)

       
        @as_op(itypes=[r.type, delta.type, T_ss.type, logD.type, alpha.type, eps.type, p.type, G.type],
               otypes=[tt.dmatrix])
        def det_interpFluxLambdaObsWithSunlight(r, delta, T_ss, D, alpha, eps, p, G):
            flux = interpFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G).T
            return flux 

        y = det_interpFluxLambdaObsWithSunlight(r, delta, T_ss, 10**logD, alpha, eps, p, G)

        flux_obs = pm.Normal("flux", mu=y[:, filterToggle], sd=Y_err[:, filterToggle], observed=Y[:, filterToggle])

        step = pm.Metropolis(scaling=scaling)
        trace = pm.sample(samples, cores=threads, chains=chains, step=step, progressbar=progressBar)

    if plotTrace is True:
        fig, ax = plt.subplots(len(fitParameters), 2, figsize=(10, 2*len(fitParameters)), **figKwargs)
        trace_ax = pm.traceplot(trace, 
                                varnames=fitParameters, 
                                skip_first=burnInSamples,
                                ax=ax)
        if saveDir != None:
            fig.savefig(os.path.join(saveDir, "{}_{}_{}_{}_trace.png".format(designation, 
                                                                             fitCode, 
                                                                             model.acronym, 
                                                                             obs.acronym)),
                       bbox_inches='tight')
    
    if plotCorner is True:        
        stack = []
        truths = []
        fitParametersCorner = []

        for parameter in fitParameters:
            values = trace.get_values(parameter)
            # This applies to parameters that are actually arrays such as 
            # distances and angles
            if len(values.shape) > 1:
                for i in range(0, values.shape[1]):
                    stack.append(values[:, i])
                    fitParametersCorner.append("{}__{}".format(parameter, i))
                    if parameter in dataParametersToIgnoreSet:
                        truths.append(data[columnMapping[parameter]].values[i])
                    else:
                        truths.append(None)
            # Single value parameter such as logT1, D, G, different emissivities
            # and albedos
            else:
                fitParametersCorner.append(parameter)
                stack.append(values)

                if parameter in dataParametersToIgnoreSet:
                    if (type(emissivityParameters) == list) and (parameter in set(emissivityParameters)):
                        truths.append(data[parameter].values[0])
                    elif (type(albedoParameters) == list) and (parameter in set(albedoParameters)):
                        truths.append(data[parameter].values[0])
                    else:
                        truths.append(data[columnMapping[parameter]].values[0])
                    
                else:
                    truths.append(None)


        corner_fig = corner.corner(np.vstack(stack).T,
                                   labels=fitParametersCorner, 
                                   truths=truths, 
                                   quantiles=(0.16, 0.84),
                                   show_titles=True)
        if saveDir != None:
            corner_fig.savefig(os.path.join(saveDir, "{}_{}_{}_{}_corner.png".format(designation, 
                                                                                     fitCode, 
                                                                                     model.acronym, 
                                                                                     obs.acronym)),
                               dpi=figKwargs["dpi"],
                               bbox_inches='tight')

    summary = pm.summary(trace, varnames=fitParameters, start=burnInSamples, stat_funcs=[_median, _sigmaG], extend=True)
    summary["code"] = np.array([fitCode for i in range(0, len(summary))])
    summary["model"] = np.array([model.acronym for i in range(0, len(summary))])
    summary[columnMapping["designation"]] = np.array([data[columnMapping["designation"]].unique()[0] for i in range(0, len(summary))])
    summary.reset_index(inplace=True)
    summary.rename(columns={"index": "parameter"}, inplace=True)
    summary = summary[[columnMapping["designation"], "model", "code", "parameter", "median", "sigmaG", "mean", "sd", "mc_error", "n_eff", "Rhat", "hpd_2.5", "hpd_97.5"]]
    pymc_objs = (pymc_model, trace)
    
    longestFitParameter = 0
    for parameter in fitParameters:
        if len(parameter) > longestFitParameter:
            longestFitParameter = len(parameter)
   
    print("Found best fit parameters:")
    for parameter in fitParameters:
        median = summary[summary["parameter"] == parameter]["median"].values[0]
        sigmaG = summary[summary["parameter"] == parameter]["sigmaG"].values[0]
        print(" {}:".ljust(longestFitParameter - len(parameter) + 5).format(parameter) + "{:.3f} +- {:.3f}".format(median, sigmaG))
    
    # Calculate model fluxes and magnitudes
    model_observations = interpFittedFluxLambdaAndMagObs(model, obs, data, summary,
        fitParameters=fitParameters,
        fitFilters=fitFilters,
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping,
        verbose=verbose)
    model_observations["code"] = np.array([fitCode for i in range(0, len(model_observations))])
    
    # End timer
    time_end = time.time()
    print("Total time: {:.2f} seconds".format(time_end-time_start))
    print("Done.")
    print("")
    
    if returnFigs is True:
        return summary, model_observations, pymc_objs, [fig, corner_fig]
    else:
        return summary, model_observations, pymc_objs

def multiFit(model, obs, dataDict, fitDict, fitConfigDict, saveDir=None):
    """
    Fit one or multiple objects with a thermal model and with different fitting 
    assumptions (fitParameters, emissivitySpecification, etc...) if so desired.
    
    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object with the approprate tables loaded into memory.
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    dataDict : dict
        Dictionary keyed with different fit codes and with pandas.DataFrames containing 
        relevant data and assumed parameters as values.
    fitDict : dict
        Dictionary containing modeling assumptions: fitParameters, emissivitySpecification,
        albedoSpecification, fitFilters, columnMapping.
    fitConfigDict : dict
        Dictionary with keys and values describing the run configuration for all runs:
        samples, burnInSamples, chains, threads, progressBar, plotTrace, plotCorner.
    saveDir : {None, str}
        If saveDir is passed, multiFit will make the passed directory and create a
        directory structure with a folder per run. In each run folder, a database is added with the results. 
        If plotTrace or plotCorner are set to True inside fitConfigDict, then those plots are saved in this 
        directory too.
      
    Returns
    -------
    summary : `~pandas.DataFrame`
        A pandas DataFrame containing summary statistics (from pymc3). All runs are combined
        into a single DataFrame. 
    model_observations : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model. All runs are combined into a single DataFrame. 
    
    Example
    -------
    ```
    model = NEATM()
    obs = WISE()
    
    # Initialize data dictionary
    dataDict = {}
        dataDict["run2a"] = data.copy()
        dataDict["run2a"]["eps_W3"] = np.ones(len(data)) * 0.70
        dataDict["run2a"]["eps_W4"] = np.ones(len(data)) * 0.86
    
    # Initialize fit dictionary
    fitDict["run2a"] = {
        "fitParameters" : ["logT1", "logD", "eps_W1W2"],
        "emissivitySpecification" : {
                    "eps_W1W2" : ["W1","W2"],
                    "eps_W3" : ["W3"],
                    "eps_W4" : ["W4"]},
        "albedoSpecification": "auto",
        "fitFilters" : "all",
        "columnMapping" : {
                    "obs_id" : "obs_id",
                    "designation" : "designation",
                    "exp_mjd" : "mjd",
                    "r_au" : "r_au",
                    "delta_au" : "delta_au",
                    "alpha_rad" : "alpha_rad",
                    "eps" : ["eps_W3", "eps_W4"],
                    "p" : None,
                    "G" : "G",
                    "logT1" : None,
                    "logD" : None,
                    "flux_si" : ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
                    "fluxErr_si" : ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
                    "mag" : ["mag_W1", "mag_W2", "mag_W3", "mag_W4"],
                    "magErr" : ["magErr_W1", "magErr_W2", "magErr_W3", "magErr_W4"]
        }
    }
    
    # Set fit configuration
    fitConfigDict = {
            "chains" : 30,
            "samples" : 3000,
            "burnInSamples": 500,
            "threads": 30,
            "scaling": 0.01,
            "plotTrace" : True,
            "plotCorner" : True,
            "progressBar" : True,
    }
    
    summary, model_observations = multiFit(model, obs, dataDict, fitDict, fitConfigDict, saveDir=None)
    ```
    """
    # Make save directory
    if saveDir is not None:
        if os.path.isdir(saveDir) == False:
            os.makedirs(saveDir)
    
    run_summaries = []
    run_model_observations = []
    
    # Lets make sure the dictionaries have at least the same keys
    if len(set(fitDict.keys()) - set(dataDict.keys())) != 0:
        raise ValueError("fitDict and dataDict do not have the same keys.")
    
    __status = {}
    objectsToProcess = {}
    databases = {}
    numTotalFits = 0
    print("Calculating number of fits to run...")
    for code in dataDict.keys():
        print("Fit code: {}".format(code))

        # Grab designation column
        desigCol = fitDict[code]["columnMapping"]["designation"]

        # Grab observations
        observations = dataDict[code]

        # Initialize list of objects to process
        objectsProcessed = []

        # Initialize status DataFrame, if it is found in a results database from a 
        # previous run then it will be updated
        __status[code] = pd.DataFrame({
            "designation" : observations[desigCol].unique(),
            "completed" : np.zeros(observations[desigCol].nunique(), dtype=int)
        })

        if saveDir is not None:
            runDir = os.path.join(saveDir, code)
            databases[code] = os.path.join(runDir, "atm_results_{}.db".format(code))

            if os.path.isdir(runDir) == True:
                if os.path.isfile(databases[code]):
                    print("Found existing results database ({}).".format(databases[code]))
                    con = sql.connect(databases[code])

                    # Lets see if the results database has any useful results
                    try:
                        summary = pd.read_sql("""SELECT * FROM summary""", con)
                        __status_old = pd.read_sql("""SELECT * FROM __status""", con)
                        objectsProcessed = __status_old[__status_old["completed"] == 1]["designation"].unique()
                        __status[code].loc[__status[code]["designation"].isin(objectsProcessed), "completed"] = 1
                        print("Objects previously fitted: {}".format(len(objectsProcessed)))
                        print("Removing incomplete fits (sampling may have been interrupted).")
                        print("Removing incomplete fits (if any) from summary table...")
                        summary = summary[summary[desigCol].isin(objectsProcessed)]
                        summary.to_sql("summary", con, index=False, if_exists="replace")
                        __status[code].to_sql("__status", con, index=False, if_exists="replace")
                        
                        print("Removing incomplete fits (if any) from model_observations table...")
                        obsIdCol = fitDict[code]["columnMapping"]["obs_id"]
                        obsIds = observations[observations[desigCol].isin(objectsProcessed)][obsIdCol].unique()
                        model_observations = pd.read_sql("""SELECT * FROM model_observations""", con)
                        model_observations = model_observations[model_observations[obsIdCol].isin(obsIds)]
                        model_observations.to_sql("model_observations", con, index=False, if_exists="replace")
                        con.commit()

                        run_summaries.append(summary)
                        run_model_observations.append(model_observations)
                    # If it doesn't lets get ready to run
                    except:
                        print("No previously fitted objects were found.")
                        objectsProcessed = []
                    con.close()
                
        objectsToProcess[code] = dataDict[code][~dataDict[code][desigCol].isin(objectsProcessed)][desigCol].unique()
        numTotalFits += len(objectsToProcess[code])
        print("Number of fits to run: {}".format(len(objectsToProcess[code])))
        print("")

    print("Total number of fits to run: {}".format(numTotalFits))
    print("")
    
    if numTotalFits > 0:
        fitNumber = 1
        for code in fitDict.keys():
            # Make run directory
            print("Starting fit code: {}".format(code))
            print("")

            desigCol = fitDict[code]["columnMapping"]["designation"]
            
            if saveDir is not None:
                runDir = os.path.join(saveDir, code)
                if os.path.isdir(runDir) == False:
                    os.makedirs(runDir)

                con = sql.connect(databases[code])
            
            # Select observations and designations for this run
            observations = dataDict[code]
            designations = objectsToProcess[code]

            for i, designation in enumerate(designations):
                print("Fitting object {} ({}/{})...".format(designation, i + 1, len(designations)))
                print("Fit number: {}/{}".format(fitNumber, numTotalFits))
                print("")
                # Set completion flag to zero 
                __completed = 0 

                # Grab object's observations
                object_observations = observations[observations[desigCol] == designation]
                
                # Check if some plots need to be saved
                if saveDir is not None:
                    if fitConfigDict["plotTrace"] is True or fitConfigDict["plotCorner"] is True:
                        savePlotDir = os.path.join(runDir, "{}".format(designation))
                        if os.path.isdir(savePlotDir) == False:
                            os.makedirs(savePlotDir)
                    else:
                        savePlotDir = None
                else:
                    savePlotDir = None
                    
                # Fit thermal model to observations
                summary, model_observations, pymc_objs = fit(
                        model, 
                        obs, 
                        object_observations, 
                        fitCode=code,
                        **fitDict[code],
                        **fitConfigDict,
                        saveDir=savePlotDir, 
                        returnFigs=False,
                        verbose=False)

                trace = pymc_objs[1]
                # If the desired number of samples were drawn from the posterior distribution
                # set completed to 1
                if len(trace) == fitConfigDict["samples"]:
                    print("Completed. Traces have the desired number of samples.")
                    __completed = 1
                    __status[code].loc[__status[code]["designation"] == designation, "completed"] = 1
                    
                    run_summaries.append(summary)
                    run_model_observations.append(model_observations)
                else:
                    warnings.warn("""Traces do not have the desired number of samples. Discarding results for object {}.""".format(designation), UserWarning)
                
                if saveDir is not None:
                    if __completed == 1:
                        print("Saving results to database ({})...".format(databases[code]))
                        summary.to_sql("summary", con, index=False, if_exists="append")
                        model_observations.to_sql("model_observations", con, index=False, if_exists="append")
                        __status[code].to_sql("__status", con, index=False, if_exists="replace")
                        con.commit()

                print("Done.")
                print()
            
                fitNumber += 1

            if saveDir is not None:
                con.close()
    else:
        print("No fits to run.")
                
    summary = pd.concat(run_summaries)
    model_observations = pd.concat(run_model_observations)
    summary.reset_index(inplace=True, drop=True)
    model_observations.reset_index(inplace=True, drop=True)
    return summary, model_observations 

    