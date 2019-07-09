#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sqlite3 as sql

from ..config import Config
from ..data_processing import modifyErrors
from .colors import calcColors
from .colors import calcModelColors
from .colors import calcColorResiduals
from .mags_and_fluxes import calcMagResiduals
from .mags_and_fluxes import calcMagChi2
from .mags_and_fluxes import calcMagReducedChi2

__all__ = ["_median",
           "_sigmaG",
           "calcStdDev",
           "postProcess",
           "postProcessDatabase"]

def _median(x):
    """
    Helper function that calculates the median of a series. 
    
    Ignores NaNs.
    """
    return pd.Series(np.nanmedian(x, axis=0), name='median')

def _sigmaG(x):
    """
    Helper function that computes the rank-based estimate of the standard deviation.
    See: http://www.astroml.org/modules/generated/astroML.stats.sigmaG.html
    
    Ignores NaNs.
    """
    q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
    return pd.Series((q75 - q25) * 0.7413, name='sigmaG')

def calcStdDev(values, robust=True):
    """
    Calculate the standard deviation or the rank-based estimate of the standard deviation of an array of values.
    
    Parameters
    ----------
    values : list or `~numpy.ndarray`
        Array of values 
    robust : bool, optional
        Calculate sigma G (rank-based estimate of the standard deviation) instead of the traditional sigma. 
        
    Returns
    -------
    float
        Standard deviation or rank-based estimate of the standard deviation
    """
    if robust is True:
        q25, q75 = np.nanpercentile(values, [25, 75])
        return (0.7413*(q75 - q25))
    else:
        return np.nansqrt(np.nanvar(values))
    
def postProcess(obs, observations, model_observations, summary, columnMapping=Config.columnMapping):
    """
    Computes the following:
        - magnitude residuals
        - observed colors
        - model colors
        - color residuals
        - median, sigmaG values for observations
        - median, sigmaG values for model observations
        - chi2 per band per model observation
        - reduced chi2 per object per run per band
    
    Lastly, this code erges the summary dataframe into the model_observations dataframe for easier
    access to results. 
        
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids used when fitting. This DataFrame
        should contain observed magnitudes.
    model_observations : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model.
    summary : `~pandas.DataFrame`
        A pandas DataFrame containing summary statistics from the fit function.
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    
    Returns
    -------
    observations_pp : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids. This DataFrame
        should contain observed magnitudes. Observed colors are added.
    model_observations_pp : `~pandas.DataFrame`
        A pandas DataFrame containing the predicted fluxes and magnitudes for a best fit
        model. Magnitude residuals, model colors, color residuals, chi2 per observation per band
        are added.
    observed_stats : `~pandas.DataFrame`
        Median and sigmaG values for all observed quantities in the observed DataFrame grouped 
        by object.
    model_stats : `~pandas.DataFrame`
        Median and sigmaG values for all quantities in the model observations DataFrame grouped 
        by object. Add is also the summary of best-fit parametes.
    """
    # Calculate observed colors
    colors = calcColors(obs, observations, columnMapping=columnMapping)
    
    # Calculate model colors
    model_colors = calcModelColors(obs, model_observations, columnMapping=columnMapping)

    # Caclulate color residuals
    color_residuals = calcColorResiduals(obs, colors, model_colors, columnMapping=columnMapping)
    
    # Calculate magnitude residuals
    mag_residuals = calcMagResiduals(obs, observations, model_observations, columnMapping=columnMapping)
    
    # Calculate magnitude chi2
    chi2 = calcMagChi2(obs, observations, mag_residuals, columnMapping=columnMapping)

    # Calculate reduced chi2
    reduced_chi2 = calcMagReducedChi2(obs, chi2, observations, summary, columnMapping=columnMapping)

    # Merge results into new post-processing DataFrames
    observations_pp = pd.merge(observations, 
                               colors, 
                               on=columnMapping["obs_id"])
    model_observations_pp = pd.merge(model_observations,
                                     mag_residuals, 
                                     on=["code", columnMapping["obs_id"]])
    model_observations_pp = pd.merge(model_observations_pp, 
                                     color_residuals, 
                                     on=["code", columnMapping["obs_id"]])
    model_observations_pp = pd.merge(model_observations_pp, 
                                     chi2, 
                                     on=["code", columnMapping["obs_id"]])
    model_observations_pp = model_observations_pp.merge(observations[[columnMapping["obs_id"], 
                                                                      columnMapping["designation"]]], 
                                                        left_on=columnMapping["obs_id"],
                                                        right_on=columnMapping["obs_id"])
    
    # Re-org columns to make it more user friendly
    cols = model_observations_pp.columns.to_list()
    cols.remove(columnMapping["obs_id"])
    cols.remove("code")
    cols.remove(columnMapping["designation"])
    model_observations_pp = model_observations_pp[[columnMapping["designation"], "code", columnMapping["obs_id"]] + cols]
    model_observations_pp.sort_values(by=[columnMapping["designation"], "code", columnMapping["obs_id"]], inplace=True)
    model_observations_pp.reset_index(inplace=True, drop=True)
    
    # Calculate median statistics on model observations
    print("Calculating median and sigmaG for model observations.")
    print("This may take a while...")
    cols = model_observations_pp.columns.to_list()
    cols.remove(columnMapping["obs_id"])
    model_stats = model_observations_pp[cols].groupby(by=[columnMapping["designation"], "code"]).agg([_median, _sigmaG])
    model_stats.rename(columns={"_median" : "median", "_sigmaG" : "sigmaG"}, inplace=True)
    model_stats.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in model_stats.columns]
    model_stats = reduced_chi2.merge(model_stats, on=[columnMapping["designation"], "code"])
    print("Done.")
    print("")
    
    # Calculate median statistics on observations
    print("Calculating median and sigmaG for observations.")
    print("This may take a while...")
    cols = observations_pp.columns.to_list()
    cols.remove(columnMapping["obs_id"])
    observed_stats = observations_pp[cols].groupby(by=columnMapping["designation"]).agg([_median, _sigmaG])
    observed_stats.rename(columns={"_median" : "median", "_sigmaG" : "sigmaG"}, inplace=True)
    observed_stats.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in observed_stats.columns]
    observed_stats.reset_index(inplace=True)
    print("Done.")
    print("")
    
    # Rotate summary dataframe and merge it with median model stats
    print("Merging parameter summary with model stats.")
    summary_rotated = pd.pivot_table(summary, 
                                 values=["median", "sigmaG"], 
                                 index=['designation', 'code'],
                                 columns=['parameter'])
    summary_rotated.columns = ['{1}_{0}'.format(col[0], col[1]) for col in summary_rotated.columns.values]
    model_stats = summary_rotated.merge(model_stats, on=[columnMapping["designation"], "code"])
    model_stats.reset_index(inplace=True, drop=True)
    print("Done.")
    print("")
    return observations_pp, model_observations_pp, observed_stats, model_stats

def postProcessDatabase(obs, observations, results_database, columnMapping=Config.columnMapping):
    """
    Run post-processing analysis on a results database and save the resulting
    analysis DataFrames to the results database.
    
    See `~atm.analysis.postProcess` for more details.
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids used when fitting. This DataFrame
        should contain observed magnitudes.
    results_database : str
        Path to a modeling run results database. This database should contain a 
        summary table and a model_observations table. 
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    
    Returns
    -------
    None
    """    
    con_results = sql.connect(results_database)
    try: 
        model_observations_pp = pd.read_sql("""SELECT * FROM model_observations_pp""", con_results)
        observations_pp = pd.read_sql("""SELECT * FROM observations_pp""", con_results)
        model_stats = pd.read_sql("""SELECT * FROM model_stats""", con_results)
        observed_stats = pd.read_sql("""SELECT * FROM observed_stats""", con_results)
        print("Post-processing tables have already been made for this database.")
    except:
        print("Failed to retrieve post-processing tables...")
        print("Running post-processing...")
        print("")
        
        model_observations = pd.read_sql("""SELECT * FROM model_observations""", con_results)
        summary = pd.read_sql("""SELECT * FROM summary""", con_results)

        observations_pp, model_observations_pp, observed_stats, model_stats = postProcess(
            obs, 
            observations, 
            model_observations, 
            summary, 
            columnMapping=columnMapping)

        print("Saving analyzed observations to results database...")
        observations_pp.to_sql("observations_pp", con_results, if_exists="replace", index=False)
        print("Saving analyzed model_observations to results database...")
        model_observations_pp.to_sql("model_observations_pp", con_results,  if_exists="replace", index=False)
        print("Saving observations statistics to results database...")
        observed_stats.to_sql("observed_stats", con_results, if_exists="replace", index=False)
        print("Saving model statistics to results database...")
        model_stats.to_sql("model_stats", con_results, if_exists="replace", index=False)
    print("Done.")
    print("")
    return