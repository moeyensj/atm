#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sqlite3 as sql

from ..config import Config

__all__ = ["crossmatchNEOWISE",
           "mergeResultsWithNEOWISE"]

def crossmatchNEOWISE(neowise, observations):
    """
    Attempts to crossmatch observations of minor planets with the NEOWISE results table. Adds a 'matched_designation' column to
    the NEOWISE results table with the matched designation from the observations table if found, if observations could not be matched
    to an asteroid in the NEOWISE table then 'matched_designation' will be set to NaN. 
    
    Tries to match designations in the following order: first asteroid number, then comet designation, then provisional designation
    and finally MPC packed name.
    
    Parameters
    ----------
    neowise : `~pandas.DataFrame`
        Pandas DataFrame contain the published NEOWISE results. These results can be filtered or unfiltered
        depending on the use case. 
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids.
    
    Returns
    -------
    None
    """
    # Attempt to match on asteroid number, set matched_designation to NaN if unnmatched
    neowise.loc[(~neowise["ASTEROID_NUMBER"].isna()), "matched_designation"] = neowise[(~neowise["ASTEROID_NUMBER"].isna())]["ASTEROID_NUMBER"].astype(int).astype(str)
    neowise.loc[(~neowise["matched_designation"].isin(observations["designation"].values)), "matched_designation"] = np.NaN
    
    # Attempt to match on comet designation, set matched_designation to NaN if unnmatched
    neowise.loc[(~neowise["COMET_DESIG"].isna()) & (neowise["matched_designation"].isna()), "matched_designation"] = neowise[(~neowise["COMET_DESIG"].isna())]["COMET_DESIG"].str.strip()
    neowise.loc[(~neowise["matched_designation"].isin(observations["designation"].values)), "matched_designation"] = np.NaN

    # Attempt to match on provisional designation, set matched_designation to NaN if unnmatched
    neowise.loc[(~neowise["PROV_DESIG"].isna()) & (neowise["matched_designation"].isna()), "matched_designation"] = neowise[(~neowise["PROV_DESIG"].isna())]["PROV_DESIG"].str.strip()
    neowise.loc[(~neowise["matched_designation"].isin(observations["designation"].values)), "matched_designation"] = np.NaN
    
    # Attempt to match on MPC packed name, set matched_designation to NaN if unnmatched
    neowise.loc[(~neowise["MPC_PACKED_NAME"].isna()) & (neowise["matched_designation"].isna()), "matched_designation"] = neowise[(~neowise["MPC_PACKED_NAME"].isna())]["MPC_PACKED_NAME"].str.strip()
    neowise.loc[(~neowise["matched_designation"].isin(observations["designation"].values)), "matched_designation"] = np.NaN

    print("Crossmatched {} unique designations from observations with NEOWISE table.".format(len(neowise[~neowise["matched_designation"].isna()]["matched_designation"].unique())))
    return 

def mergeResultsWithNEOWISE(observations_database, results_database, 
                            minObs=3, 
                            fitCodes=["DVBI"],
                            neowiseTable="neowise_v1",
                            columnMapping=Config.columnMapping):
    """
    Merges the post-processing observed_stats and model_stats DataFrames
    with the 2016 NEOWISE PDS release. 
    
    Only returns previously matched designations using the 'matched_designation'
    column in the neowise_v1 table. 
    
    Parameters
    ----------
    observations_database : str
        Path the database containing observations table and the neowise_v1 
        table. 
    results_database : 
        Path to a results database from a thermal modeling run.
    minObs : int, optional
        The minimum number of observations required in each band of the 
        NEOWISE table. 
        [Default = 3]
    fitCodes : list, optional
        Fit codes to extract from the NEOWISE table. 
        [Defaults = ["DVBI"]]
    neowiseTable : str, optional
        Name of the NEOWISE results table in the observations database 
        to use. 
        [Default = "neowise_v1"]
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
        
    Returns
    -------
    merged_results : `~pandas.DataFrame`
        Merged observed_stats, model_stats and neowise_v1 DataFrames.
    (observations_pp, model_observations_pp, observed_stats model_stats) : 
        (`~pandas.DataFrame`, `~pandas.DataFrame`, `~pandas.DataFrame`, `~pandas.DataFrame`)
        Post-processing tables from the results database.
    """
    # Connect to results database and grab post-processing tables
    con_results = sql.connect(results_database)  
    observations_pp = pd.read_sql("""SELECT * FROM observations_pp""", con_results)
    model_observations_pp = pd.read_sql("""SELECT * FROM model_observations_pp""", con_results)
    observed_stats = pd.read_sql("""SELECT * FROM observed_stats""", con_results)
    model_stats = pd.read_sql("""SELECT * FROM model_stats""", con_results)

    # Connect to observations database and grab NEOWISE results
    # Filter them as desired by the user.
    con = sql.connect(observations_database)
    neowise = pd.read_sql("""SELECT * FROM {}""".format(neowiseTable), con)
    print("There are {} fits for {} unique objects in the 2016 NEOWISE PDS table.".format(len(neowise), neowise["MPC_PACKED_NAME"].nunique()))
    print("Selecting only fits with at least {} observations in each band.".format(minObs))
    print("Selecting only fits with fit code(s): {}.".format(", ".join(fitCodes)))
    neowise = neowise[(neowise["FIT_CODE"].isin(fitCodes))
                  & (neowise["N_W1"] >= minObs)
                  & (neowise["N_W2"] >= minObs)
                  & (neowise["N_W3"] >= minObs)
                  & (neowise["N_W4"] >= minObs)]  
    print("There are {} fits for {} unique objects.".format(len(neowise), 
                                                            neowise["MPC_PACKED_NAME"].nunique()))
    neowise = neowise[~neowise["matched_designation"].isna()]
    print("{} fits have been matched with an object in observations.".format(len(neowise)))
    print("Sorting by number of observations and keeping the fits using the most observations...")
    neowise = neowise.sort_values(by=["N_W1", "N_W2", "N_W3", "N_W4"], ascending=False)
    neowise = neowise.drop_duplicates(subset=["matched_designation"], keep="first")
    
    print("Merging NEOWISE results with post-processed tables...")
    merged_results = model_stats.merge(observed_stats, on=columnMapping["designation"])
    merged_results = merged_results.merge(neowise, left_on=columnMapping["designation"], right_on="matched_designation")
    print("")
    print("{} fits have been found for {} unique objects in observations.".format(len(merged_results),              merged_results["matched_designation"].nunique()))
    print("Done.")
    print("")
    return merged_results, (observations_pp, model_observations_pp, observed_stats, model_stats)