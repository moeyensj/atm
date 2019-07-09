#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

from ..config import Config

__all__ = ["calcMagResiduals",
           "calcMagChi2",
           "calcMagReducedChi2"]

def calcMagResiduals(obs, observations, model_observations, columnMapping=Config.columnMapping):
    """
    For each unique fit code in model observations, calculate the magnitude residuals (observations - model)
    and return them in a DataFrame.
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids.
    model_observations : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model.
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    
    Returns
    -------
    mag_residuals : `~pandas.DataFrame`
        The magnitude residuals in a DataFrame calculated for each different 
        fit code. 
    """
    mag_columns = columnMapping["mag"]
    model_mag_columns = ["model_mag_{}".format(f) for f in obs.filterNames]
    mag_residuals_columns = ["residual_{}".format(f) for f in obs.filterNames]
    print("Calculating magnitude residuals using these columns:")
    print("Observed magnitudes : {}".format(", ".join(mag_columns)))
    print("Model magnitudes : {}".format(", ".join(model_mag_columns)))
    print("Creating residuals: {}".format(", ".join(mag_residuals_columns)))
    
    mag_residuals = model_observations.merge(observations[[columnMapping["obs_id"]] + mag_columns], on=columnMapping["obs_id"])
    for m1, m2, r in zip(mag_columns, model_mag_columns, mag_residuals_columns):
        mag_residuals[r] = mag_residuals[m1].values - mag_residuals[m2].values
        
    mag_residuals = mag_residuals[[columnMapping["obs_id"], "code"] + mag_residuals_columns]
    mag_residuals.sort_values(by=["code", columnMapping["obs_id"]], inplace=True)
    mag_residuals.reset_index(inplace=True, drop=True)
    
    print("Done.")
    print("")
    return mag_residuals

def calcMagChi2(obs, observations, mag_residuals, columnMapping=Config.columnMapping):
    """
    For each unique fit code in model observations, calculate chi2 for each observation in each 
    band. (Calculates and also returns the magnitude residuals)
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids.
    mag_residuals : `~pandas.DataFrame`
        The magnitude residuals in a DataFrame calculated for each different 
        fit code. 
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    
    Returns
    -------
    chi2 : `~pandas.DataFrame`
        The chi squared for each observation in each band in a DataFrame calculated for each different 
        fit code. 
    """
    residuals_columns = ["residual_{}".format(f) for f in obs.filterNames]
    chi2_columns = ["chi2_{}".format(f) for f in obs.filterNames]
    print("Calculating chi squared using these columns:")
    print("Magnitude residuals : {}".format(", ".join(residuals_columns)))
    print("Magnitude errors : {}".format(", ".join(columnMapping["magErr"])))
    print("Creating chi2 columns: {}".format(", ".join(chi2_columns)))

    chi2 = observations.merge(mag_residuals, on=[columnMapping["obs_id"]])
    for r, e, c in zip(residuals_columns, columnMapping["magErr"], chi2_columns):
        chi2[c] = chi2[r].values**2 / chi2[e].values**2

    chi2 = chi2[[columnMapping["obs_id"], "code"] + chi2_columns]
    chi2["chi2"] = np.zeros(len(chi2))
    for c in chi2_columns:
        # Sets NaN values to 0 so they don't contribute to the sum
        chi2["chi2"] += np.nan_to_num(chi2[c].values)
    chi2.sort_values(by=["code", columnMapping["obs_id"]], inplace=True)
    chi2.reset_index(inplace=True, drop=True)
    
    print("Done.")
    print("")
    return chi2

from atm import Config

def calcMagReducedChi2(obs, chi2, observations, summary, columnMapping=Config.columnMapping):
    """
    Calculates the chi squared per degree of freedom (number of observations - the number of fit parameters) for each 
    filter and object, grouped by fit code.
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    chi2 : `~pandas.DataFrame`
        The chi squared for each observation in each band in a DataFrame calculated for each different 
        fit code. 
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids.
    summary : `~pandas.DataFrame`
        A pandas DataFrame containing summary statistics from the fit function.
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
        
    Returns
    -------
    reduced_chi2 : `~pandas.DataFrame`
        The reduced chi squared for each object (per-band and total) grouped by fit code.
    """
    chi2_columns = ["chi2_{}".format(f) for f in obs.filterNames]
    reduced_chi2_columns = ["reduced_{}".format(c) for c in chi2_columns]
    print("Calculating reduced chi squared using these columns:")
    print("Chi squared : {}".format(", ".join(chi2_columns)))
    
    # Count number of observations per band
    def _countObs(x):
        return x.notna().astype(int).sum()
    num_cols = ["num_{}".format(f) for f in obs.filterNames]
    num_obs = observations.groupby(by=[columnMapping["designation"]])[columnMapping["mag"]].agg(_countObs)
    num_obs.rename(columns={mag_col : num_col for mag_col, num_col in zip(columnMapping["mag"], num_cols)},
                   inplace=True)
    num_obs["num_obs"] = np.zeros(len(num_obs), dtype=int)
    for num_col in num_cols:
        num_obs[num_col] = num_obs[num_col].astype(int)
        num_obs["num_obs"] += num_obs[num_col].values

    merged = pd.merge(chi2, observations[[columnMapping["obs_id"], columnMapping["designation"]]], 
                      left_on=columnMapping["obs_id"], 
                      right_on=columnMapping["obs_id"])
    reduced_chi2_list = []

    for code in merged["code"].unique():
        merged_run = merged[merged["code"] == code].copy()
        numFitParameters = summary[summary["code"] == code]["parameter"].nunique()
        reduced_chi2 = pd.DataFrame(merged_run.groupby(by=[columnMapping["designation"], "code"])["chi2"].sum())
        reduced_chi2 = reduced_chi2.join(num_obs, on=[columnMapping["designation"]])
        reduced_chi2["reduced_chi2"] = reduced_chi2["chi2"] / (reduced_chi2["num_obs"] - numFitParameters)
        reduced_chi2_list.append(reduced_chi2)
    
    # Concatenate DataFrames, remove designation and code as indexes then sort
    # by designation and code and finally cleanup the index again
    reduced_chi2 = pd.concat(reduced_chi2_list)
    reduced_chi2.reset_index(inplace=True)
    reduced_chi2.sort_values(by=[columnMapping["designation"], "code"], inplace=True)
    reduced_chi2.reset_index(inplace=True, drop=True)
    
    print("Done.")
    print("")
    return reduced_chi2