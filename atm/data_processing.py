#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

from .config import Config

__all__ = ["clipObservations",
           "modifyErrors"]

def clipObservations(observations, obs, magCut=1.0, minObs=3, maxObsCut=0.3, columnMapping=Config.columnMapping):
    """
    Fits a linear time versus magnitude model and flags any observations sufficiently far away from the best-fit line of 
    magnitudes. If any observation in a filter is flagged, all other observations in different filters at the same epoch are also 
    flagged -- this may flag otherwise good data and is perhaps too stringent. 
    
    This function adds a 'keep' column which has values 1 or 0. If 1 then the observations passed the cutting criteria. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing the observations of asteroids. Needs at least a column with the exposure time of the observations, the 
        SI flux measurement for each band and the flux error as well, needs a column with the asteroid's designation.
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    magCut : float, optional
        The maximum deviation in magnitude from the median magnitude per band. This is used at two stages: first, the median
        magnitude is caculated for each object, and observations that fall within +- magCut are used to fit a linear 
        magnitude model per band. Second, once the linear model has been fit using the initial cut of points, the linear model 
        is then used to find any observations +- magCut from the model. Only the observations where all bands are within magCut 
        from the linear model pass the cut criteria.
        [Default = 1.0]
    minObs : int, optional
        The minimum number of observations an object should have post-cut. If an object has fewer than
        this number of observations the entire object is cut
        [Default = 3]
    maxObsCut : float, optional
        Maximum percentage of observations (as a fraction between 0 and 1) that can be cut. If more observations
        are cut than this fraction then the entire object is cut. 
        [Default = 0.3]
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
    
    Returns
    -------
    observations_flagged : `~pandas.DataFrame`
        DataFrame containing the observations of asteroids with a `keep` column added.
    """
    # Make sure observations are sorted by designation and exposure time
    observations_flagged = observations.copy()
    observations_flagged.sort_values(by=[columnMapping["designation"], columnMapping["exp_mjd"]], inplace=True)
    
    # Initialize the keep column
    observations_flagged["keep"] = np.zeros(len(observations_flagged), dtype=int)
    
    for designation in observations_flagged[columnMapping["designation"]].unique():
        object_observations_flagged = observations_flagged[observations_flagged[columnMapping["designation"]] == designation]
        num_obs = len(object_observations_flagged)
        
        # Grab magnitudes
        mag_cols = columnMapping["mag"]
        mag = object_observations_flagged[mag_cols].values
        
        # Calculate the median magnitude in each filter, find the difference between
        # magnitudes and median and use only observations that have all bands within magCut
        # of respective medians to fit linear magnitude model
        med_mag = np.median(mag, axis=0)
        dev_mag = np.abs(mag - med_mag)
        good = np.all(dev_mag < magCut, axis=1) 
        num_obs_cut = num_obs - len(good[good == True])
        
        # No observations passed the initial cut
        if np.all(good == False):
            continue
        # More observations than maxObsCut were cut
        elif num_obs_cut > (num_obs * maxObsCut):
            continue
        #  or fewer than minObs are left
        elif (num_obs - num_obs_cut) < minObs:
            continue
        else:
            mag_fit = np.zeros_like(mag)
            # Fit linear magnitude model per band
            for i, f in enumerate(mag_cols):
                mag_slope_intercept = np.polyfit(
                    object_observations_flagged[columnMapping["exp_mjd"]].values[good], 
                    object_observations_flagged[f].values[good], 
                    1)
                mag_fit[:, i] = np.polyval(
                    mag_slope_intercept,
                    object_observations_flagged[columnMapping["exp_mjd"]].values)
            
            # Keep only observations that are within magCut (per band) from 
            # linear model
            dev_mag = np.abs(mag_fit - mag)
            final_good = np.all(dev_mag < magCut, axis=1)

            # Update keep column in observations
            observations_flagged.loc[observations_flagged[columnMapping["designation"]] == designation, "keep"] = final_good.astype(int)
    
    num_obs_total = len(observations_flagged)
    num_obs_cut_total = len(observations_flagged[observations_flagged["keep"] == 0])
    num_obj_total = observations_flagged[columnMapping["designation"]].nunique()
    num_obj_cut_total = num_obj_total - observations_flagged[observations_flagged["keep"] == 1][columnMapping["designation"]].nunique()
    print("Number of observations that were cut: {} ({:.2f}%)".format(num_obs_cut_total, num_obs_cut_total/num_obs_total * 100.))
    print("Number of observations that survived cut: {} ({:.2f}%)".format(num_obs_total - num_obs_cut_total, (num_obs_total - num_obs_cut_total)/num_obs_total * 100.))
    print("Number of unique objects that were cut: {} ({:.2f}%)".format(num_obj_cut_total, num_obj_cut_total/num_obj_total * 100.))
    print("Number of unique objects that survived cut: {} ({:.2f}%)".format(num_obj_total - num_obj_cut_total, (num_obj_total - num_obj_cut_total)/num_obj_total * 100.))
    print("The observations DataFrame has been updated: see observations['keep'] for clipping flag.")
    return observations_flagged

def modifyErrors(observations, obs, sigma=0.15, columnMapping=Config.columnMapping):
    """
    Adds a constant magnitude offset to the errors in an observations DataFrame.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids.
    obs : `~atm.obs.Observatory`
        Observatory object containing filter zeropoint information.
    sigma : float, optional
        Constant magnitude error offset to add to observations. This error in magnitudes is also used
        to add an error to the fluxes in the DataFrame.
        [Default = 0.15]
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]
        
    Returns
    -------
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations of asteroids now with increased
        errors.
    """
    observations = observations.copy()
        
    for magErr in columnMapping["magErr"]:
        print("Added {} magnitude errors to {}.".format(sigma, magErr))
        observations[magErr] = observations[magErr] + sigma

    fluxErr = obs.convertMagErrToFluxLambdaErr(observations[columnMapping["mag"]].values, observations[columnMapping["magErr"]].values)
    print("Converted magnitude errors to flux errors.")
    for i, flambdaErr in enumerate(columnMapping["fluxErr_si"]):
        print("Updating {} with new error.".format(flambdaErr))
        observations[flambdaErr] = fluxErr[:, i]

    print("Done.")
    print("")
    return observations