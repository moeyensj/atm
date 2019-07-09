#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pytest
import numpy as np
import pandas as pd

from ..fit import __handleParameters
from ..fit import fit
from ..obs import WISE
from ..models import NEATM
from ..functions import interpFluxLambdaObsWithSunlight
from ..functions import calcQ
from ..functions import calcTss

def test_handleParameters():
    obs = WISE()
    
    ### Case 1: Test perBand emissivity and auto albedo
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
            "exp_mjd": "mjd",
            "r_au": "r_au",
            "delta_au": "delta_au",
            "alpha_rad": "alpha_rad",
            "G": "G",
            "logD": "logD",
            "logT1" : "logT1",
            "eta": "eta",
            "eps": ["eps_W1", "eps_W2", "eps_W4"],
            "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
            "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
            }

    columns = ["eps_W1", "eps_W2", "eps_W3", "eps_W4", "logD", "logT1", "r_au", "delta_au", "alpha_rad", "G"]
    fitParameters = ["logD", "logT1", "eps_W3"]
    emissivitySpecification = "perBand"
    albedoSpecification = "auto"

    fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
        obs,
        fitParameters, 
        columns, 
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping)

    np.testing.assert_equal(fitParameters_test, set(fitParameters))
    np.testing.assert_equal(emissivityParameters_test, ["eps_W1", "eps_W2", "eps_W3", "eps_W4"])
    np.testing.assert_equal(albedoParameters_test, ["p_W1", "p_W2", "p_W3", "p_W4"])
    np.testing.assert_equal(dataParametersToIgnoreSet, set(["eps_W3", "logT1", "logD"]))

    ### Case 2: Test None emissivity and auto albedo
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
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
            }

    columns = ["eps", "r_au", "delta_au", "alpha_rad", "G"]
    fitParameters = ["logD", "logT1", "eps"]
    emissivitySpecification = None
    albedoSpecification = "auto"

    fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
        obs,
        fitParameters, 
        columns, 
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping)

    np.testing.assert_equal(fitParameters_test, set(fitParameters))
    np.testing.assert_equal(emissivityParameters_test, "eps")
    np.testing.assert_equal(albedoParameters_test, "p")
    np.testing.assert_equal(dataParametersToIgnoreSet, set(["eps"]))

    ### Case 3: Test auto emissivity and auto albedo [should raise ValueError]
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
            "exp_mjd": "mjd",
            "r_au": "r_au",
            "delta_au": "delta_au",
            "alpha_rad": "alpha_rad",
            "G": "G",
            "logD": "logD",
            "logT1" : "logT1",
            "eta": "eta",
            "eps": "eps",
            "p" : "p",
            "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
            "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
            }

    columns = ["eps", "r_au", "delta_au", "alpha_rad", "G"]
    fitParameters = ["logD", "logT1", "eps"]
    emissivitySpecification = "auto"
    albedoSpecification = "auto"

    with pytest.raises(ValueError): 
        fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
            obs,
            fitParameters, 
            columns, 
            emissivitySpecification=emissivitySpecification,
            albedoSpecification=albedoSpecification,
            columnMapping=columnMapping)

    ### Case 4: Test None emissivity and None albedo [should warn with a RuntimeWarning]
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
            "exp_mjd": "mjd",
            "r_au": "r_au",
            "delta_au": "delta_au",
            "alpha_rad": "alpha_rad",
            "G": "G",
            "logD": "logD",
            "logT1" : "logT1",
            "eta": "eta",
            "eps": "eps",
            "p" : "p",
            "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
            "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
            }

    columns = ["eps", "r_au", "delta_au", "alpha_rad", "G"]
    fitParameters = ["logD", "logT1", "eps", "p"]
    emissivitySpecification = None
    albedoSpecification = None

    with pytest.warns(RuntimeWarning): 
        fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
            obs,
            fitParameters, 
            columns, 
            emissivitySpecification=emissivitySpecification,
            albedoSpecification=albedoSpecification,
            columnMapping=columnMapping)
        
        np.testing.assert_equal(fitParameters_test, set(fitParameters))
        np.testing.assert_equal(emissivityParameters_test, "eps")
        np.testing.assert_equal(albedoParameters_test, "p")
        np.testing.assert_equal(dataParametersToIgnoreSet, set(["eps"]))


    ### Case 5: Test combined filter emissivities and "auto" albedo
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
            "exp_mjd": "mjd",
            "r_au": "r_au",
            "delta_au": "delta_au",
            "alpha_rad": "alpha_rad",
            "G": "G",
            "logD": "logD",
            "logT1" : "logT1",
            "p" : "p",
            "eta": "eta",
            "eps": ["eps_W1W2", "eps_W3", "eps_W4"],
            "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
            "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
            }

    columns = ["eps", "r_au", "delta_au", "alpha_rad", "G", "eps_W1W2", "eps_W3", "eps_W4"]
    fitParameters = ["logD", "logT1", "eps_W4", "eps_W1W2"]
    emissivitySpecification = {"eps_W1W2" : ["W1", "W2"],
                               "eps_W3" : ["W3"],
                               "eps_W4": ["W4"]}
    albedoSpecification = "auto"


    fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
        obs,
        fitParameters, 
        columns, 
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping)

    np.testing.assert_equal(fitParameters_test, set(fitParameters))
    np.testing.assert_equal(emissivityParameters_test, ["eps_W1W2", "eps_W1W2", "eps_W3", "eps_W4"])
    np.testing.assert_equal(albedoParameters_test, ["p_W1W2", "p_W1W2", "p_W3", "p_W4"])
    np.testing.assert_equal(dataParametersToIgnoreSet, set(["eps_W1W2", "eps_W4"]))

    ### Case 6: Test combined filter albedos and "auto" emissivity
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
            "exp_mjd": "mjd",
            "r_au": "r_au",
            "delta_au": "delta_au",
            "alpha_rad": "alpha_rad",
            "G": "G",
            "logD": "logD",
            "logT1" : "logT1",
            "eta": "eta",
            "eps": ["eps_W1W2", "eps_W3", "eps_W4"],
            "p" : ["p_W1"],
            "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
            "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
            }

    columns = ["eps", "r_au", "delta_au", "alpha_rad", "G", "eps_W1W2", "eps_W3", "eps_W4", "p_W1"]
    fitParameters = ["logD", "logT1", "p_W2W3W4"]
    emissivitySpecification = "auto"
    albedoSpecification = {"p_W1" : ["W1"],
                           "p_W2W3W4": ["W2", "W3", "W4"]}

    fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
        obs,
        fitParameters, 
        columns, 
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping)

    np.testing.assert_equal(fitParameters_test, set(fitParameters))
    np.testing.assert_equal(emissivityParameters_test, ["eps_W1", "eps_W2W3W4", "eps_W2W3W4", "eps_W2W3W4"])
    np.testing.assert_equal(albedoParameters_test, ["p_W1", "p_W2W3W4", "p_W2W3W4", "p_W2W3W4"])
    np.testing.assert_equal(dataParametersToIgnoreSet, set())

    ### Case 7: Test combined filter albedos and emissivity
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
            "exp_mjd": "mjd",
            "r_au": "r_au",
            "delta_au": "delta_au",
            "alpha_rad": "alpha_rad",
            "G": "G",
            "logD": "logD",
            "logT1" : "logT1",
            "eta": "eta",
            "eps": ["eps_W1W2W3", "eps_W4"],
            "p" : ["p_W1"],
            "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
            "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
            }

    columns = ["eps", "r_au", "delta_au", "alpha_rad", "G", "eps_W1W2W3", "eps_W4", "p_W1"]
    fitParameters = ["logD", "logT1", "p_W2W3W4", "eps_W1W2W3", "p_W1"]
    emissivitySpecification = {"eps_W1W2W3" : ["W1", "W2", "W3"],
                               "eps_W4" : ["W4"]}
    albedoSpecification = {"p_W1" : ["W1"],
                           "p_W2W3W4": ["W2", "W3", "W4"]}

    with pytest.warns(RuntimeWarning): 
        fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
            obs,
            fitParameters, 
            columns, 
            emissivitySpecification=emissivitySpecification,
            albedoSpecification=albedoSpecification,
            columnMapping=columnMapping)

        np.testing.assert_equal(fitParameters_test, set(fitParameters))
        np.testing.assert_equal(emissivityParameters_test, ["eps_W1W2W3", "eps_W1W2W3", "eps_W1W2W3", "eps_W4"])
        np.testing.assert_equal(albedoParameters_test, ["p_W1", "p_W2W3W4", "p_W2W3W4", "p_W2W3W4"])
        np.testing.assert_equal(dataParametersToIgnoreSet, set(["eps_W1W2W3", "p_W1"]))
        
    ### Case 8: Test combined filter albedos and emissivity with weird columns
    columnMapping = {
            "designation" : "designation",
            "obs_id": "obsId",
            "exp_mjd": "mjd",
            "r_au": "r",
            "delta_au": "delta_au",
            "alpha_rad": "a213d",
            "G": "a213d",
            "logD": "logD10",
            "logT1" : "logT1",
            "eta": "eta",
            "eps": ["eps_W1W2W3", "eps_W4"],
            "p" : ["p_W1"],
            "flux_si": ["flux_W1_si", "flux_W2_si", "flux_W3_si", "flux_W4_si"],
            "fluxErr_si": ["fluxErr_W1_si", "fluxErr_W2_si", "fluxErr_W3_si", "fluxErr_W4_si"],
            }

    columns = ["eps", "r", "delta_au", "a213d", "a213d", "eps_W1W2W3", "eps_W4", "p_W1"]
    fitParameters = ["logD", "logT1", "p_W2W3W4", "eps_W1W2W3", "p_W1", "r_au"]
    emissivitySpecification = {"eps_W1W2W3" : ["W1", "W2", "W3"],
                               "eps_W4" : ["W4"]}
    albedoSpecification = {"p_W1" : ["W1"],
                           "p_W2W3W4": ["W2", "W3", "W4"]}

    with pytest.warns(RuntimeWarning): 
        fitParameters_test, requiredDataParametersSet_test, emissivityParameters_test, albedoParameters_test, dataParametersToIgnoreSet = __handleParameters(
            obs,
            fitParameters, 
            columns, 
            emissivitySpecification=emissivitySpecification,
            albedoSpecification=albedoSpecification,
            columnMapping=columnMapping)

        np.testing.assert_equal(fitParameters_test, set(fitParameters))
        np.testing.assert_equal(emissivityParameters_test, ["eps_W1W2W3", "eps_W1W2W3", "eps_W1W2W3", "eps_W4"])
        np.testing.assert_equal(albedoParameters_test, ["p_W1", "p_W2W3W4", "p_W2W3W4", "p_W2W3W4"])
        np.testing.assert_equal(dataParametersToIgnoreSet, set(["eps_W1W2W3", "p_W1", "r_au"]))
        
def test_fit_constant_emissivity():
    # Instantiate observatory and NEATM class for simulating data
    obs = WISE()
    model = NEATM(verbose=False)

    # Load WISE quadrature lookup tables into memory 
    model.loadLambdaTables(obs.filterQuadratureLambdas)

    # Create fake observing geometry 
    num_obs = 7
    r = np.random.normal(loc=3.0, scale=0.25, size=num_obs)
    delta = r - 1.0
    alpha = np.random.normal(loc=np.radians(2),
                             scale=np.radians(2)/100,
                             size=num_obs)

    # Create fake asteroid
    eta = 0.756 * np.ones(num_obs)
    G = 0.15 * np.ones(num_obs)
    D = 1000 * np.ones(num_obs)
    logD = np.log10(D)

    # Define percent error in fluxes
    err = 0.01

    eps = np.array(0.78)
    p = (1 - eps) / calcQ(G[0])
    T_ss = calcTss(r, p, eps, G, eta)
    T1 = T_ss * np.sqrt(r)
    logT1 = np.log10(T1)

    # Model flux with sunlight
    flux = interpFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G)
    #flux = np.random.normal(loc=flux, scale=err*flux)

    data = np.vstack([r, delta, alpha, G, D, logD, eta, T1, logT1, T_ss,
                      flux[0,:], flux[1,:], flux[2,:], flux[3,:],
                      err*flux[0,:], err*flux[1,:], err*flux[2,:], err*flux[3,:]])

    data = pd.DataFrame(data.T, columns=["r",
                                         "delta",
                                         "alpha",
                                         "G",
                                         "D",
                                         "logD",
                                         "eta",
                                         "T1",
                                         "logT1",
                                         "T_ss",
                                         "flux_W1",
                                         "flux_W2",
                                         "flux_W3",
                                         "flux_W4",
                                         "fluxErr_W1",
                                         "fluxErr_W2",
                                         "fluxErr_W3",
                                         "fluxErr_W4"])
    data["obs_id"] = np.arange(1, len(data) + 1)
    data["eps"] = eps * np.ones(num_obs)
    data["p"] = p * np.ones(num_obs)
    data["designation"] = ["0000" for i in range(len(data))]
    data = data[["obs_id", "designation", "r", "delta", "alpha", "G", "D", "logD", "eta", "T1", "logT1", "T_ss", "eps", "p",
                 "flux_W1", "flux_W2", "flux_W3", "flux_W4", "fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"]]

    mags = obs.convertFluxLambdaToMag(data[["flux_W1", "flux_W2", "flux_W3", "flux_W4"]].values)
    magErrs = obs.convertFluxLambdaErrToMagErr(data[["flux_W1", "flux_W2", "flux_W3", "flux_W4"]].values, data[["fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"]].values)
    mag_columns = ["mag_{}".format(f) for f in obs.filterNames]
    magErr_columns = ["magErr_{}".format(f) for f in obs.filterNames]
    for i, (j, k) in enumerate(zip(mag_columns, magErr_columns)):
        data[j] = mags[:, i]
        data[k] = magErrs[:, i]

    columnMapping = {
        "designation" : "designation",
        "obs_id" : "obs_id",
        "r_au" : "r",
        "delta_au" : "delta",
        "alpha_rad" : "alpha",
        "G" : "G",
        "logT1" : "logT1",
        "logD": "logD",
        "eta" : "eta",
        "eps" : "eps", 
        "flux_si" : ["flux_W1", "flux_W2", "flux_W3", "flux_W4"],
        "fluxErr_si" : ["fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"],  
        "mag" : ["mag_W1", "mag_W2", "mag_W3", "mag_W4"],
        "magErr" : ["magErr_W1", "magErr_W2", "magErr_W3", "magErr_W4"],
    }
    fitParameters = ["logT1", "logD", "eps"]
    fitFilters = "all"
    emissivitySpecification = None
    albedoSpecification = "auto"

    # Run NEATM
    model = NEATM(verbose=False)
    summary_neatm, model_observations_neatm, pymc_objs_neatm = fit(
        model, 
        obs, 
        data, 
        fitParameters=fitParameters, 
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping,
        samples=5000,
        threads=4)

    np.testing.assert_allclose(10**summary_neatm[summary_neatm["parameter"] == "logD"]["median"].values, [10**3.00], rtol=0.02)
    np.testing.assert_allclose(summary_neatm[summary_neatm["parameter"] == "eps"]["median"].values, [0.78], rtol=0.01)
    
def test_fit_combination_emissivity():
    # Instantiate observatory and NEATM class for simulating data
    obs = WISE()
    model = NEATM(verbose=False)

    # Load WISE quadrature lookup tables into memory 
    model.loadLambdaTables(obs.filterQuadratureLambdas)

    # Create fake observing geometry 
    num_obs = 7
    r = np.random.normal(loc=3.0, scale=0.25, size=num_obs)
    delta = r - 1.0
    alpha = np.random.normal(loc=np.radians(2),
                             scale=np.radians(2)/100,
                             size=num_obs)

    # Create fake asteroid
    eta = 0.756 * np.ones(num_obs)
    G = 0.15 * np.ones(num_obs)
    D = 1000 * np.ones(num_obs)
    logD = np.log10(D)

    # Define percent error in fluxes
    err = 0.01
    
    eps = np.array([[0.7, 0.7, 0.9, 0.9]])
    p = (1 - eps) / calcQ(G[0])
    T_ss = calcTss(r, p.T, eps.T, G, eta)
    T_ss = T_ss.T[:,0]
    T1 = T_ss * np.sqrt(r)
    logT1 = np.log10(T1)

    eps = eps.T
    p = p.T

    # Model flux with sunlight
    flux = interpFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G)

    data = np.vstack([r, delta, alpha, G, D, logD, eta, T1, logT1, T_ss,
                      flux[0,:], flux[1,:], flux[2,:], flux[3,:],
                      err*flux[0,:], err*flux[1,:], err*flux[2,:], err*flux[3,:]])

    data = pd.DataFrame(data.T, columns=["r",
                                         "delta",
                                         "alpha",
                                         "G",
                                         "D",
                                         "logD",
                                         "eta",
                                         "T1",
                                         "logT1",
                                         "T_ss",
                                         "flux_W1",
                                         "flux_W2",
                                         "flux_W3",
                                         "flux_W4",
                                         "fluxErr_W1",
                                         "fluxErr_W2",
                                         "fluxErr_W3",
                                         "fluxErr_W4"])
    data["obs_id"] = np.arange(1, len(data) + 1)
    data["eps_W1W2"] = eps[0] * np.ones(num_obs)
    data["eps_W3W4"] = eps[2] * np.ones(num_obs)
    data["p_W1W2"] = p[0] * np.ones(num_obs)
    data["p_W3W4"] = p[2] * np.ones(num_obs)
    data["designation"] = ["0000" for i in range(len(data))]
    data = data[["obs_id", "designation", "r", "delta", "alpha", "G", "D", "logD", "eta", "T1", "logT1", "T_ss", "eps_W1W2", "eps_W3W4", "p_W1W2", "p_W3W4",
                 "flux_W1", "flux_W2", "flux_W3", "flux_W4", "fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"]]
    
    mags = obs.convertFluxLambdaToMag(data[["flux_W1", "flux_W2", "flux_W3", "flux_W4"]].values)
    magErrs = obs.convertFluxLambdaErrToMagErr(data[["flux_W1", "flux_W2", "flux_W3", "flux_W4"]].values, data[["fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"]].values)
    mag_columns = ["mag_{}".format(f) for f in obs.filterNames]
    magErr_columns = ["magErr_{}".format(f) for f in obs.filterNames]
    for i, (j, k) in enumerate(zip(mag_columns, magErr_columns)):
        data[j] = mags[:, i]
        data[k] = magErrs[:, i]

    columnMapping = {
        "designation" : "designation",
        "obs_id" : "obs_id",
        "r_au" : "r",
        "delta_au" : "delta",
        "alpha_rad" : "alpha",
        "G" : "G",
        "logT1" : "logT1",
        "logD": "logD",
        "eta" : "eta",
        "eps" : ["eps_W1W2", "eps_W3W4"], 
        "flux_si" : ["flux_W1", "flux_W2", "flux_W3", "flux_W4"],
        "fluxErr_si" : ["fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"],  
        "mag" : ["mag_W1", "mag_W2", "mag_W3", "mag_W4"],
        "magErr" : ["magErr_W1", "magErr_W2", "magErr_W3", "magErr_W4"],
    }
    fitParameters = ["logT1", "logD", "eps_W1W2"]
    emissivitySpecification = {"eps_W1W2": ["W1", "W2"],
                               "eps_W3W4": ["W3", "W4"]}
    albedoSpecification="auto"

    # Run NEATM
    model = NEATM(verbose=False)
    summary_neatm, model_observations_neatm, pymc_objs_neatm = fit(
        model, 
        obs, 
        data, 
        fitParameters=fitParameters, 
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping,
        samples=5000,
        threads=4)

    np.testing.assert_allclose(10**summary_neatm[summary_neatm["parameter"] == "logD"]["median"].values, [10**3.00], rtol=0.02)
    np.testing.assert_allclose(summary_neatm[summary_neatm["parameter"] == "eps_W1W2"]["median"].values, [0.70], rtol=0.01)
    
def test_fit_combination_albedo():
    # Instantiate observatory and NEATM class for simulating data
    obs = WISE()
    model = NEATM(verbose=False)

    # Load WISE quadrature lookup tables into memory 
    model.loadLambdaTables(obs.filterQuadratureLambdas)

    # Create fake observing geometry 
    num_obs = 7
    r = np.random.normal(loc=3.0, scale=0.25, size=num_obs)
    delta = r - 1.0
    alpha = np.random.normal(loc=np.radians(2),
                             scale=np.radians(2)/100,
                             size=num_obs)

    # Create fake asteroid
    eta = 0.756 * np.ones(num_obs)
    G = 0.15 * np.ones(num_obs)
    D = 1000 * np.ones(num_obs)
    logD = np.log10(D)

    # Define percent error in fluxes
    err = 0.01

    eps = np.array([[0.9, 0.9, 0.9, 0.9]])
    p = (1 - eps) / calcQ(G[0])
    p[0, 2] = 0.0
    p[0, 3] = 0.0
    T_ss = calcTss(r, p.T, eps.T, G, eta)
    T_ss = T_ss.T[:,0]
    T1 = T_ss * np.sqrt(r)
    logT1 = np.log10(T1)

    eps = eps.T
    p = p.T

    # Model flux with sunlight
    flux = interpFluxLambdaObsWithSunlight(model, obs, r, delta, T_ss, D, alpha, eps, p, G)

    data = np.vstack([r, delta, alpha, G, D, logD, eta, T1, logT1, T_ss,
                      flux[0,:], flux[1,:], flux[2,:], flux[3,:],
                      err*flux[0,:], err*flux[1,:], err*flux[2,:], err*flux[3,:]])

    data = pd.DataFrame(data.T, columns=["r",
                                         "delta",
                                         "alpha",
                                         "G",
                                         "D",
                                         "logD",
                                         "eta",
                                         "T1",
                                         "logT1",
                                         "T_ss",
                                         "flux_W1",
                                         "flux_W2",
                                         "flux_W3",
                                         "flux_W4",
                                         "fluxErr_W1",
                                         "fluxErr_W2",
                                         "fluxErr_W3",
                                         "fluxErr_W4"])
    data["obs_id"] = np.arange(1, len(data) + 1)
    data["eps"] = eps[0] * np.ones(num_obs)
    data["p_W1W2"] = p[0] * np.ones(num_obs)
    data["p_W3W4"] = p[2] * np.ones(num_obs)
    data["designation"] = ["0000" for i in range(len(data))]
    data = data[["obs_id", "designation", "r", "delta", "alpha", "G", "D", "logD", "eta", "T1", "logT1", "T_ss", "eps", "p_W1W2", "p_W3W4",
                 "flux_W1", "flux_W2", "flux_W3", "flux_W4", "fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"]]

    mags = obs.convertFluxLambdaToMag(data[["flux_W1", "flux_W2", "flux_W3", "flux_W4"]].values)
    magErrs = obs.convertFluxLambdaErrToMagErr(data[["flux_W1", "flux_W2", "flux_W3", "flux_W4"]].values, data[["fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"]].values)
    mag_columns = ["mag_{}".format(f) for f in obs.filterNames]
    magErr_columns = ["magErr_{}".format(f) for f in obs.filterNames]
    for i, (j, k) in enumerate(zip(mag_columns, magErr_columns)):
        data[j] = mags[:, i]
        data[k] = magErrs[:, i]

    columnMapping = {
        "designation" : "designation",
        "obs_id" : "obs_id",
        "r_au" : "r",
        "delta_au" : "delta",
        "alpha_rad" : "alpha",
        "G" : "G",
        "logT1" : "logT1",
        "logD": "logD",
        "eta" : "eta",
        "eps" : "eps", 
        "p" : ["p_W3W4"],
        "flux_si" : ["flux_W1", "flux_W2", "flux_W3", "flux_W4"],
        "fluxErr_si" : ["fluxErr_W1", "fluxErr_W2", "fluxErr_W3", "fluxErr_W4"],  
        "mag" : ["mag_W1", "mag_W2", "mag_W3", "mag_W4"],
        "magErr" : ["magErr_W1", "magErr_W2", "magErr_W3", "magErr_W4"],
    }
    fitParameters = ["logT1", "logD", "p_W1W2"]
    fitFilters = "all"
    emissivitySpecification = None
    albedoSpecification = {"p_W1W2" : ["W1", "W2"],
                           "p_W3W4" : ["W3", "W4"]}

    # Run NEATM
    model = NEATM(verbose=False)
    summary_neatm, model_observations_neatm, pymc_objs_neatm = fit(
        model, 
        obs, 
        data, 
        fitParameters=fitParameters, 
        emissivitySpecification=emissivitySpecification,
        albedoSpecification=albedoSpecification,
        columnMapping=columnMapping,
        samples=5000,
        threads=4)

    np.testing.assert_allclose(10**summary_neatm[summary_neatm["parameter"] == "logD"]["median"].values, [10**3.00], rtol=0.02)
    np.testing.assert_allclose(summary_neatm[summary_neatm["parameter"] == "p_W1W2"]["median"].values, [0.26], rtol=0.01)