#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

from ..config import Config
from ..constants import Constants
from ..helpers import __handleParameters
from .flux_lambda import calcFluxLambdaSun
from .hg import calcHG
from .hg import calcQ

__all__ = ["interpFluxLambdaAtObs",
           "interpFluxLambdaAtObsWithSunlight",
           "interpFluxLambdaObs",
           "interpFluxLambdaObsWithSunlight",
           "interpFittedFluxLambdaAndMagObs"]

AU = Constants.ASTRONOMICAL_UNIT


def interpFluxLambdaAtObs(model, r, delta, lambd, T_ss, D, alpha, eps):
    """
    Interpolate the flux at an observer or observatory from an asteroid without
    reflected sunlight.

    Requires `model` to have tables loaded.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object.
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    delta : float or `~numpy.ndarray` (N)
        Distance between asteroid and the observatory in AU.
    lambd : float
        Wavelength in m.
    T_ss : float or `~numpy.ndarray` (N)
        Subsolar temperature in K.
    D : float or `~numpy.ndarray` (N)
        Asteroid diameter in m.
    alpha : float or `~numpy.ndarray` (N)
        Phase angle in radians.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns array of fluxes at an observer or observatory.
    """
    return D**2 / (4 * delta**2 * AU**2) * (eps * model.interpTotalFluxLambdaEmittedToObs(lambd, T_ss, alpha))


def interpFluxLambdaAtObsWithSunlight(model, r, delta, lambd, T_ss, D, alpha, eps, p, G):
    """
    Interpolate the flux at an observer or observatory from an asteroid with
    reflected sunlight.

    Requires `model` to have tables loaded.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object.
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    delta : float or `~numpy.ndarray` (N)
        Distance between asteroid and the observatory in AU.
    lambd : float
        Wavelength in m.
    T_ss : float or `~numpy.ndarray` (N)
        Subsolar temperature in K.
    D : float or `~numpy.ndarray` (N)
        Asteroid diameter in meters.
    alpha : float or `~numpy.ndarray` (N)
        Phase angle in radians.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.
    p : float or `~numpy.ndarray` (N)
        Albedo.
    G : float or `~numpy.ndarray` (N)
        HG slope parameter.

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns array of fluxes at an observer or observatory with
        reflected sunlight.
    """
    return (interpFluxLambdaAtObs(model, r, delta, lambd, T_ss, D, alpha, eps)
            + (D**2 / (4 * delta**2 * AU**2)) * p * calcHG(alpha, G)
            * calcFluxLambdaSun(lambd, r))


def interpFluxLambdaObs(model, obs, r, delta, T_ss, D, alpha, eps):
    """
    Interpolate the observed flux from an asteroid without
    reflected sunlight.

    Requires `model` to have tables loaded.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object.
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    delta : float or `~numpy.ndarray` (N)
        Distance between asteroid and the observatory in AU.
    T_ss : float or `~numpy.ndarray` (N)
        Subsolar temperature in K.
    D : float or `~numpy.ndarray` (N)
        Asteroid diameter in meters.
    alpha : float or `~numpy.ndarray` (N)
        Phase angle in radians.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.

    Returns
    -------
    float or `~numpy.ndarray` (N, M)
        Returns an array of fluxes with shape N observations by
        M filters.
    """
    return D**2 / (4 * delta**2 * AU**2) * (eps * obs.bandpassLambda(model.interpTotalFluxLambdaEmittedToObs, args=[T_ss, alpha]))


def interpFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G):
    """
    Interpolate the observed flux from an asteroid with
    reflected sunlight.

    Requires `model` to have tables loaded.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object.
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    delta : float or `~numpy.ndarray` (N)
        Distance between asteroid and the observatory in AU.
    T_ss : float or `~numpy.ndarray` (N)
        Subsolar temperature in K.
    D : float or `~numpy.ndarray` (N)
        Asteroid diameter in meters.
    alpha : float or `~numpy.ndarray` (N)
        Phase angle in radians.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.
    p : float or `~numpy.ndarray` (N)
        Albedo.
    G : float or `~numpy.ndarray` (N)
        HG slope parameter.

    Returns
    -------
    float or `~numpy.ndarray` (N, M)
        Returns an array of fluxes with shape N observations by
        M filters.
    """
    return (interpFluxLambdaObs(model, obs, r, delta, T_ss, D, alpha, eps)
            + (D**2 / (4 * delta**2 * AU**2)) * p * calcHG(alpha, G)
            * obs.bandpassLambda(calcFluxLambdaSun, args=[r]))

def interpFittedFluxLambdaAndMagObs(model, obs, data, summary,
        fitParameters=Config.fitParameters,
        fitFilters="all",
        emissivitySpecification=None,
        albedoSpecification="auto",
        columnMapping=Config.columnMapping,
        verbose=Config.verbose):
    """
    Calculate fluxes and magnitudes (model observations) for a single asteroid with fitting
    results from the summary DataFrame.

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
    summary : `~pandas.DataFrame`
        Summary DataFrame returned from fit function for a single object.
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
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    verbose : bool, optional
        Print progress statements?
        [Default = `~atm.Config.verbose`]

    Returns
    -------
    model_observations : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model.
    """
    
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
    
    
    if "r_au" in parametersSet and "r_au" not in fitParametersSet:
        r = data[columnMapping["r_au"]].values
    else: 
        r = summary[summary["parameter"].isin(["r_au__{}".format(i) for i in range(0, len(data))])]["median"].values     

    if "delta_au" in parametersSet and "delta_au" not in fitParametersSet:
        delta = data[columnMapping["delta_au"]].values
    else: 
        delta = summary[summary["parameter"].isin(["delta_au__{}".format(i) for i in range(0, len(data))])]["median"].values     

    if "alpha_rad" in parametersSet and "alpha_rad" not in fitParametersSet:
        alpha = data[columnMapping["alpha_rad"]].values
    else: 
        alpha = summary[summary["parameter"].isin(["alpha_rad__{}".format(i) for i in range(0, len(data))])]["median"].values     
        
    if "G" in parametersSet and "G" not in fitParametersSet:
        G = data[columnMapping["G"]].values[0]
    else: 
        G = summary[summary["parameter"] == "G"]["median"].values[0]

    if "logD" in parametersSet and "logD" not in fitParametersSet:
        logD = data[columnMapping["logD"]].values[0]
    else: 
        logD = summary[summary["parameter"] == "logD"]["median"].values[0]

    if "logT1" in parametersSet and "logT1" not in fitParametersSet:
        logT1 = data[columnMapping["logT1"]].values[0]
    else: 
        logT1 = summary[summary["parameter"] == "logT1"]["median"].values[0]

    T_ss = 10**logT1 / np.sqrt(r)

    if emissivityParameters == "eps" and emissivitySpecification != "auto":
        if "eps" in parametersSet and "eps" not in fitParametersSet:
            eps = data[columnMapping["eps"]].values[0]
        else: 
            eps = summary[summary["parameter"] == "eps"]["median"].values[0]

        if albedoSpecification == "auto":
            p = (1 - eps) / calcQ(G)
            
    if type(emissivityParameters) is list and emissivitySpecification != "auto":
        eps = np.zeros_like([emissivityParameters], dtype=float)
        for i, parameter in enumerate(emissivityParameters):
            if parameter in parametersSet and parameter not in fitParametersSet:
                eps[0, i] = data[parameter].values[0]
            else:
                eps[0, i] = summary[summary["parameter"] == parameter]["median"].values[0]
                
        eps = eps.T
            
        if albedoSpecification == "auto":
            p = (1 - eps) / calcQ(G)
        

    if albedoParameters == "p" and albedoSpecification != "auto":
        if "p" in parametersSet and "p" not in fitParametersSet:
            p = data[columnMapping["p"]].values[0]
        else: 
            p = summary[summary["parameter"] == "p"]["median"].values[0]

        if emissivitySpecification == "auto":
            eps = 1 - p * calcQ(G)

    if type(albedoParameters) is list and albedoSpecification != "auto":
        p = np.zeros_like([albedoParameters], dtype=float)
        for i, parameter in enumerate(albedoParameters):
            if parameter in parametersSet and parameter not in fitParametersSet:
                p[0, i] = data[parameter].values[0]
            else:
                p[0, i] = summary[summary["parameter"] == parameter]["median"].values[0]
                
        p = p.T
            
        if emissivitySpecification == "auto":
            eps = 1 - p * calcQ(G)
    
    fitted_flux = interpFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, 10**logD, alpha, eps, p, G).T
    fitted_magnitudes = obs.convertFluxLambdaToMag(fitted_flux)
       
    columns = [columnMapping["obs_id"]]
    columns += ["model_flux_si_{}".format(f) for f in obs.filterNames] 
    columns += ["model_mag_{}".format(f) for f in obs.filterNames]
    model_observations = pd.DataFrame(np.concatenate([data[columnMapping["obs_id"]].values.reshape(-1, 1), fitted_flux, fitted_magnitudes], axis=1), columns=columns)
    model_observations[columnMapping["obs_id"]] = model_observations[columnMapping["obs_id"]].astype(int, inplace=True)
    return model_observations
