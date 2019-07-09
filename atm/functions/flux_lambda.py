#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

from ..config import Config
from ..constants import Constants
from ..helpers import __handleParameters
from .hg import calcHG
from .hg import calcQ
from .temperature import calcTss
from .blackbody import calcPlanckLambda


__all__ = ["calcFluxLambdaSun",
           "calcFluxLambdaAtObs",
           "calcFluxLambdaAtObsWithSunlight",
           "calcFluxLambdaObs",
           "calcFluxLambdaObsWithSunlight",
           "calcFluxLambdaSED"]

R_Sun = Constants.SOLAR_RADIUS
T_Sun = Constants.SOLAR_TEMPERATURE
AU = Constants.ASTRONOMICAL_UNIT


def calcFluxLambdaSun(lambd, r, T=T_Sun):
    """
    Calculate solar flux at wavelength lambd in meters
    at heliocentric distance r in AU.

    Parameters
    ----------
    lambd : float or `~numpy.ndarray` (N)
        Wavelength in m.
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    T : float, optional
        Solar temperature in K.
        [Default = `~atm.Constants.SOLAR_TEMPERATURE`]

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns solar flux at lambd wavelength.
    """
    return (np.pi * R_Sun**2) / (r**2 * AU**2) * calcPlanckLambda(lambd, T)


def calcFluxLambdaAtObs(model, r, delta, lambd, T_ss, D, alpha, eps, threads=Config.threads):
    """
    Calculate the flux at an observer or observatory from an asteroid without
    reflected sunlight.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object.
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    delta : float or `~numpy.ndarray` (N)
        Distance between asteroid and the observatory in AU.
    lambd : float or `~numpy.ndarray` (N)
        Wavelength in m.
    T_ss : float or `~numpy.ndarray` (N)
        Subsolar temperature in K.
    D : float or `~numpy.ndarray` (N)
        Asteroid diameter in m.
    alpha : float or `~numpy.ndarray` (N)
        Phase angle in radians.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.
    threads : int, optional
        Number of processors to use.
        [Default = `atm.Config.threads`]

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns array of fluxes at an observer or observatory.
    """
    return (D**2 / (4 * delta**2 * AU**2) *
            (eps * model.calcTotalFluxLambdaEmittedToObsMany(lambd, T_ss, alpha)))


def calcFluxLambdaAtObsWithSunlight(model, r, delta, lambd, T_ss, D, alpha, eps, p, G, threads=Config.threads):
    """
    Calculate the flux at an observer or observatory from an asteroid with
    reflected sunlight.

    Parameters
    ----------
    model : `~atm.models.Model`
        Flux model object.
    r : float or `~numpy.ndarray` (N)
        Distance between asteroid and the Sun in AU.
    delta : float or `~numpy.ndarray` (N)
        Distance between asteroid and the observatory in AU.
    lambd : float or `~numpy.ndarray` (N)
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
    threads : int, optional
        Number of processors to use.
        [Default = `atm.Config.threads`]

    Returns
    -------
    float or `~numpy.ndarray` (N)
        Returns array of fluxes at an observer or observatory with
        reflected sunlight.
    """
    return (calcFluxLambdaAtObs(model, r, delta, lambd, T_ss, D, alpha, eps, threads=threads) 
            + (D**2 / (4 * delta**2 * AU**2)) * p * calcHG(alpha, G)
            * calcFluxLambdaSun(lambd, r))


def calcFluxLambdaObs(model, obs, r, delta, T_ss, D, alpha, eps, threads=Config.threads):
    """
    Calculate the observed flux from an asteroid without
    reflected sunlight.

    This function is multi-processed to make calculations faster since 
    doing many integrations can take a while. If you are looking to do any 
    sort of fitting it is recommended to use the interpolated version of this 
    function.

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
    threads : int, optional
        Number of processors to use.
        [Default = `atm.Config.threads`]

    Returns
    -------
    float or `~numpy.ndarray` (N, M)
        Returns an array of fluxes with shape N observations by
        M filters.
    """
    return (D**2 / (4 * delta**2 * AU**2) * (eps * obs.bandpassLambda(model.calcTotalFluxLambdaEmittedToObsMany, args=[T_ss, alpha])))


def calcFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G, threads=Config.threads):
    """
    Calculate the observed flux from an asteroid with
    reflected sunlight.

    This function is multi-processed to make calculations faster since 
    doing many integrations can take a while. If you are looking to do any 
    sort of fitting it is recommended to use the interpolated version of this 
    function.

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
    alpha : float v=or `~numpy.ndarray` (N)
        Phase angle in radians.
    eps : float or `~numpy.ndarray` (N)
        Emissivity.
    p : float or `~numpy.ndarray` (N)
        Albedo.
    G : float or `~numpy.ndarray` (N)
        HG slope parameter.
    threads : int, optional
        Number of processors to use.
        [Default = `atm.Config.threads`]

    Returns
    -------
    float or `~numpy.ndarray` (N, M)
        Returns an array of fluxes with shape N observations by
        M filters.
    """
    return (calcFluxLambdaObs(model, obs, r, delta, T_ss, D, alpha, eps, threads=threads)
            + (D**2 / (4 * delta**2 * AU**2)) * p * calcHG(alpha, G)
            * obs.bandpassLambda(calcFluxLambdaSun, args=[r]))

def calcFluxLambdaSED(model, obs, data, 
        summary=None,
        lambdaRange=[1.5e-6, 30e-6],
        lambdaNum=200,
        lambdaEdges=[3.9e-6, 6.5e-6, 18.5e-6],
        linearInterpolation=True,
        fitParameters=[],
        emissivitySpecification=None,
        albedoSpecification="auto",
        threads=4,
        columnMapping=Config.columnMapping):
    """
    Calculate flux between lambdaRange[0] and lambdaRange[1] for a single asteroid thermal model. If fitParameters is an
    empty list, this function will look for all the required parameters in data. If fitParameters is not an empty 
    list, then this function will look for the fitParameters in the summary DataFrame. 
    Uses the median observation geometry from the data to plot the calculated best-fit SED.
    If the fitting scenario allowed for emissivity or albedo to change as a function of wavelength or bandpass, then the user
    has the option to linearly interpolate albedo and emissivity between calculated or assumed values.

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
    summary : `~pandas.DataFrame`, optional
        Summary DataFrame returned from fit function for a single object.
    lambdaRange : list, optional
        Minimum and maximum wavelength in meters at which to calculate model flux.
        [Default = [1.5e-6, 30e-6]]
    lambdaNum : int, optional
        Number of wavelength points between (and including) lambdaRange[0] and lambdaRange[1]. 
        [Default = 200]
    lambdaEdges : {None, list}, optional
        If emissivity or albedo are not constant over the wavelength range, then set the lambdaEdges to a list
        with the interior boundaries where emissivity and/or albedo should change. For example, set lambda edges to 
        the wavelengths between two filters. Do not include lambdaRange[0] or lambdaRange[1]. If linearInterpolation 
        is set to True, these edges are ignored in favor of a more robust linear interpolation.
        [Default = [3.9e-6, 6.5e-6, 18.5e-6]]
    linearInterpolation : bool, optional
        Linearly interpolate emissivity and albedo values between calculated best fit values or values from the data. 
        [Default = True]
    fitParameters : list, optional
        The parameters that should be fit for. If a fit parameter is found to 
        exist inside the data, that column will be ignored for fitting purposes. 
        [Default = `~atm.Config.fitParameters`]
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

    Returns
    -------
    model_observations : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model.
    """
    # Handle parameters
    fitParametersSet, parametersSet, emissivityParameters, albedoParameters, dataParametersToIgnoreSet = __handleParameters(
        obs,
        fitParameters, 
        data.columns.tolist(), 
        emissivitySpecification=emissivitySpecification, 
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping)
    
    lambd = np.linspace(lambdaRange[0], lambdaRange[-1], num=lambdaNum)

    if "r_au" in parametersSet and "r_au" not in fitParametersSet:
        r = np.median(data[columnMapping["r_au"]].values) * np.ones(len(lambd))
    else: 
        r = summary[summary["parameter"].isin(["r_au__{}".format(i) for i in range(0, len(data))])]["median"].values * np.ones(len(lambd))    

    if "delta_au" in parametersSet and "delta_au" not in fitParametersSet:
        delta = np.median(data[columnMapping["delta_au"]].values) * np.ones(len(lambd))
    else: 
        delta = summary[summary["parameter"].isin(["delta_au__{}".format(i) for i in range(0, len(data))])]["median"].values * np.ones(len(lambd))      

    if "alpha_rad" in parametersSet and "alpha_rad" not in fitParametersSet:
        alpha = np.median(data[columnMapping["alpha_rad"]].values) * np.ones(len(lambd))
    else: 
        alpha = summary[summary["parameter"].isin(["alpha_rad__{}".format(i) for i in range(0, len(data))])]["median"].values * np.ones(len(lambd))     
        
    if "G" in parametersSet and "G" not in fitParametersSet:
        G = data[columnMapping["G"]].values[0] * np.ones(len(lambd))
    else: 
        G = summary[summary["parameter"] == "G"]["median"].values[0] * np.ones(len(lambd))

    if "logD" in parametersSet and "logD" not in fitParametersSet:
        logD = data[columnMapping["logD"]].values[0] * np.ones(len(lambd))
    else: 
        logD = summary[summary["parameter"] == "logD"]["median"].values[0] * np.ones(len(lambd))

    if "logT1" in parametersSet and "logT1" not in fitParametersSet:
        logT1 = data[columnMapping["logT1"]].values[0] * np.ones(len(lambd))
    else: 
        logT1 = summary[summary["parameter"] == "logT1"]["median"].values[0] * np.ones(len(lambd))

    T_ss = 10**logT1 / np.sqrt(r)

    if emissivityParameters == "eps" and emissivitySpecification != "auto":
        if "eps" in parametersSet and "eps" not in fitParametersSet:
            eps = data[columnMapping["eps"]].values[0] * np.ones(len(lambd))
        else: 
            eps = summary[summary["parameter"] == "eps"]["median"].values[0] * np.ones(len(lambd))

        if albedoSpecification == "auto":
            p = (1 - eps) / calcQ(G)
            
    if type(emissivityParameters) is list and emissivitySpecification != "auto":
        eps_values = np.zeros_like(emissivityParameters, dtype=float)
        for i, parameter in enumerate(emissivityParameters):
            if parameter in parametersSet and parameter not in fitParametersSet:
                eps_values[i] = data[parameter].values[0]
            else:
                eps_values[i] = summary[summary["parameter"] == parameter]["median"].values[0]

        eps = np.zeros_like(lambd)
        if linearInterpolation is True:
            eps = np.interp(lambd, obs.filterEffectiveLambdas, eps_values)
        else:
            for i, (edge_start, edge_end) in enumerate(zip([lambdaRange[0]] + lambdaEdges, 
                                                   lambdaEdges + [lambdaRange[-1]])):
                eps = np.where((lambd >= edge_start) & (lambd <= edge_end), eps_values[i], eps)
            
        if albedoSpecification == "auto":
            p = (1 - eps) / calcQ(G)

    if albedoParameters == "p" and albedoSpecification != "auto":
        if "p" in parametersSet and "p" not in fitParametersSet:
            p = data[columnMapping["p"]].values[0] * np.ones(len(lambd))
        else: 
            p = summary[summary["parameter"] == "p"]["median"].values[0] * np.ones(len(lambd))

        if emissivitySpecification == "auto":
            eps = 1 - p * calcQ(G)

    if type(albedoParameters) is list and albedoSpecification != "auto":
        p_values = np.zeros_like(albedoParameters, dtype=float)
        for i, parameter in enumerate(albedoParameters):
            if parameter in parametersSet and parameter not in fitParametersSet:
                p_values[i] = data[parameter].values[0]
            else:
                p_values[i] = summary[summary["parameter"] == parameter]["median"].values[0]
                
        p = np.zeros_like(lambd)
        if linearInterpolation is True:
            p = np.interp(lambd, obs.filterEffectiveLambdas, p_values)
        else:
            for i, (edge_start, edge_end) in enumerate(zip([lambdaRange[0]] + lambdaEdges, 
                                                   lambdaEdges + [lambdaRange[-1]])):
                p = np.where((lambd >= edge_start) & (lambd <= edge_end), p_values[i], p)
            
        if emissivitySpecification == "auto":
            eps = 1 - p * calcQ(G)
    
    fitted_flux = calcFluxLambdaAtObsWithSunlight(model, r, delta, lambd, T_ss, 10**logD, alpha, eps, p, G, threads=threads)
    df = pd.DataFrame(data={
        "lambda": lambd,
        "eps": eps,
        "p" : p,
        "flux": fitted_flux})
    return df


