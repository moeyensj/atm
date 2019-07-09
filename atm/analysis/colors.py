#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

from ..config import Config

__all__ = ["calcColors",
           "calcModelColors",
           "calcColorResiduals"]

def calcColors(obs, observations, columnMapping=Config.columnMapping):
    """
    Calculates colors using magnitudes. Filter order in obs.filterNames and magnitude order 
    in columnMapping["mag"] must be identical.
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids. This DataFrame
        should contain observed magnitudes.
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    
    Returns
    -------
    colors : `~pandas.DataFrame`
        DataFrame with the observation ID column from the observations DataFrame and the 
        corresponding colors.
    """
    observations_mag_columns = columnMapping["mag"]
    color_columns = ["{}-{}".format(f1, f2) for f1, f2 in zip(obs.filterNames[:-1], obs.filterNames[1:])]
    print("Calculating colors using these columns:")
    print("Magnitudes : {}".format(", ".join(observations_mag_columns)))
    print("Creating colors: {}".format(", ".join(color_columns)))
    
    color_dict = {}
    color_dict[columnMapping["obs_id"]] = observations[columnMapping["obs_id"]].values
    for f1, f2, c in zip(columnMapping["mag"][:-1], columnMapping["mag"][1:], color_columns):
        color_dict[c] = observations[f1].values - observations[f2].values
    
    print("Done.")
    print("")
    return pd.DataFrame(color_dict)

def calcModelColors(obs, model_observations, columnMapping=Config.columnMapping):
    """
    Calculates model colors using model magnitudes. Filter order in obs.filterNames and magnitude order 
    in columnMapping["mag"] must be identical.
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    model_observations : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model.
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
        
    Returns
    -------
    model_colors : `~pandas.DataFrame`
        DataFrame with the observation ID and code columns from the model_observations DataFrame and the 
        corresponding model colors.
    """
    model_mag_columns = ["model_mag_{}".format(f) for f in obs.filterNames]
    model_color_columns = ["model_{}-{}".format(f1, f2) for f1, f2 in zip(obs.filterNames[:-1], obs.filterNames[1:])]
    print("Calculating model colors using these columns:")
    print("Model magnitudes : {}".format(", ".join(model_mag_columns)))
    print("Creating model colors: {}".format(", ".join(model_color_columns)))
    
    color_dict = {}
    color_dict[columnMapping["obs_id"]] = model_observations[columnMapping["obs_id"]].values
    color_dict["code"] = model_observations["code"].values
    for f1, f2, c in zip(model_mag_columns[:-1], model_mag_columns[1:], model_color_columns):
        color_dict[c] = model_observations[f1].values - model_observations[f2].values
    
    print("Done.")
    print("")
    return pd.DataFrame(color_dict)

def calcColorResiduals(obs, colors, model_colors, columnMapping=Config.columnMapping):
    """
    Calculates color residuals (observations - model). Filter order in obs.filterNames and magnitude order 
    in columnMapping["mag"] must be identical.
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    colors : `~pandas.DataFrame`
        DataFrame with the observation ID column from the observations DataFrame and the 
        corresponding colors.
    model_colors : `~pandas.DataFrame`
        DataFrame with the observation ID and code columns from the model_observations DataFrame and the 
        corresponding model colors.
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
        
    Returns
    -------
    color_residuals : `~pandas.DataFrame`
        DataFrame with the observation ID and code columns from the model_observations DataFrame and the 
        corresponding color residuals.
    """
    color_columns = ["{}-{}".format(f1, f2) for f1, f2 in zip(obs.filterNames[:-1], obs.filterNames[1:])]
    model_color_columns = ["model_{}-{}".format(f1, f2) for f1, f2 in zip(obs.filterNames[:-1], obs.filterNames[1:])]
    residual_color_columns = ["residual_{}-{}".format(f1, f2) for f1, f2 in zip(obs.filterNames[:-1], obs.filterNames[1:])]
    print("Calculating color residuals using these columns:")
    print("Observed colors: {}".format(", ".join(color_columns)))
    print("Model colors : {}".format(", ".join(model_color_columns)))
    print("Creating residuals: {}".format(", ".join(residual_color_columns)))

    color_residuals = model_colors.merge(colors, on="obs_id")
    for c1, c2, r in zip(color_columns, model_color_columns, residual_color_columns):
        color_residuals[r] = color_residuals[c1].values - color_residuals[c2].values
    
    color_residuals = color_residuals[[columnMapping["obs_id"], "code"] + residual_color_columns]
    color_residuals.sort_values(by=["code", columnMapping["obs_id"]], inplace=True)
    color_residuals.reset_index(inplace=True, drop=True)
    
    print("Done.")
    print("")
    return color_residuals