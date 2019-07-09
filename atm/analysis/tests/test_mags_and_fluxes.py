#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

from ...config import Config
from ...obs import WISE
from ..mags_and_fluxes import calcMagResiduals
from ..mags_and_fluxes import calcMagChi2
from ..mags_and_fluxes import calcMagReducedChi2


def test_calcMagResiduals():
    # Initialize observatory
    obs = WISE()

    # Create a DataFrame of magnitudes: W1: 1, W2: 2, W3: 3, W4: 4
    columns=["mag_W1", "mag_W2", "mag_W3", "mag_W4"]
    observations = pd.DataFrame(np.ones((10000, 4)), columns=columns)
    observations["obs_id"] = np.arange(1, len(observations) + 1)
    observations = observations[["obs_id"] + columns]
    for i, c in enumerate(columns[1:]):
        observations[c] += (i+1)

    # Create a DataFrame of model magnitudes: 
    # run0 (W1: 1, W2: 0, W3: -1, W4: -2)
    # run1 (W1: 3, W2: 0, W3: -3, W4: -6)
    columns=["model_mag_W1", "model_mag_W2", "model_mag_W3", "model_mag_W4"]
    model_observations = pd.DataFrame(np.ones((20000, 4)), columns=columns)
    model_observations["obs_id"] = np.concatenate([np.arange(1, len(model_observations[:10000]) + 1), 
                                                   np.arange(1, len(model_observations[:10000]) + 1)])
    model_observations["code"] = (["run0" for i in range(len(model_observations[:10000]))] 
                                  + ["run1" for i in range(len(model_observations[10000:]))])
    model_observations = model_observations[["obs_id", "code"] + columns]
    for i, c in enumerate(columns[1:]):
            model_observations[c] += (-1 * (i + 1))
    model_observations.loc[model_observations["code"] == "run1", "model_mag_W1"] += 2 
    model_observations.loc[model_observations["code"] == "run1", "model_mag_W3"] -= 2 
    model_observations.loc[model_observations["code"] == "run1", "model_mag_W4"] -= 4 

    residual_columns=["residual_W1", "residual_W2", "residual_W3", "residual_W4"]
    desired_residuals = pd.DataFrame(np.ones((20000, 4)) * 2.0, columns=residual_columns)
    desired_residuals["obs_id"] = model_observations["obs_id"].values
    desired_residuals["code"] = model_observations["code"].values
    desired_residuals = desired_residuals[["obs_id", "code"] + residual_columns]
    desired_residuals.loc[desired_residuals["code"] == "run0", "residual_W1"] = 0
    desired_residuals.loc[desired_residuals["code"] == "run0", "residual_W3"] = 4
    desired_residuals.loc[desired_residuals["code"] == "run0", "residual_W4"] = 6
    desired_residuals.loc[desired_residuals["code"] == "run1", "residual_W1"] = -2
    desired_residuals.loc[desired_residuals["code"] == "run1", "residual_W3"] = 6
    desired_residuals.loc[desired_residuals["code"] == "run1", "residual_W4"] = 10

    mag_residuals = calcMagResiduals(obs, observations, model_observations)
    pd.testing.assert_frame_equal(desired_residuals, mag_residuals)
    
def test_calcMagChi2():
    # Initialize observatory
    obs = WISE()

    # Create a DataFrame of magnitudes: W1: 1, W2: 2, W3: 3, W4: 4
    magErr_columns=["magErr_W1", "magErr_W2", "magErr_W3", "magErr_W4"]
    observations = pd.DataFrame(np.ones((10000, 4)) * 0.5, columns=magErr_columns)
    observations["obs_id"] = np.arange(1, len(observations) + 1)
    observations = observations[["obs_id"] + magErr_columns]

    # Create a skeleton of model observations
    columns=["model_mag_W1", "model_mag_W2", "model_mag_W3", "model_mag_W4"]
    model_observations = pd.DataFrame(np.ones((20000, 4)), columns=columns)
    model_observations["obs_id"] = np.concatenate([np.arange(1, len(model_observations[:10000]) + 1), 
                                                   np.arange(1, len(model_observations[:10000]) + 1)])
    model_observations["code"] = (["run0" for i in range(len(model_observations[:10000]))] 
                                  + ["run1" for i in range(len(model_observations[10000:]))])
    model_observations = model_observations[["obs_id", "code"] + columns]
        
    # Create a DataFrame of residuals: 
    # run0 (W1: 1, W2: 0, W3: -1, W4: -2)
    # run1 (W1: 3, W2: 0, W3: -3, W4: -6)
    residual_columns=["residual_W1", "residual_W2", "residual_W3", "residual_W4"]
    residuals = pd.DataFrame(np.ones((20000, 4)) * 2.0, columns=residual_columns)
    residuals["obs_id"] = model_observations["obs_id"].values
    residuals["code"] = model_observations["code"].values
    residuals = residuals[["obs_id", "code"] + residual_columns]
    residuals.loc[residuals["code"] == "run0", "residual_W1"] = 0
    residuals.loc[residuals["code"] == "run0", "residual_W3"] = 4
    residuals.loc[residuals["code"] == "run0", "residual_W4"] = 6
    residuals.loc[residuals["code"] == "run1", "residual_W1"] = -2
    residuals.loc[residuals["code"] == "run1", "residual_W3"] = 6
    residuals.loc[residuals["code"] == "run1", "residual_W4"] = 10
    
    # Create a DataFrame of chi2 values: 
    chi2_desired_columns=["chi2_W1", "chi2_W2", "chi2_W3", "chi2_W4", "chi2"]
    chi2_desired = pd.DataFrame(np.ones((20000, 5)) * 2.0**2 / 0.5**2, columns=chi2_desired_columns)
    chi2_desired["obs_id"] = model_observations["obs_id"].values
    chi2_desired["code"] = model_observations["code"].values
    chi2_desired = chi2_desired[["obs_id", "code"] + chi2_desired_columns]
    chi2_desired.loc[chi2_desired["code"] == "run0", "chi2_W1"] = 0**2 / 0.5**2
    chi2_desired.loc[chi2_desired["code"] == "run0", "chi2_W3"] = 4**2 / 0.5**2
    chi2_desired.loc[chi2_desired["code"] == "run0", "chi2_W4"] = 6**2 / 0.5**2
    chi2_desired.loc[chi2_desired["code"] == "run1", "chi2_W1"] = (-2)**2 / 0.5**2
    chi2_desired.loc[chi2_desired["code"] == "run1", "chi2_W3"] = 6**2 / 0.5**2
    chi2_desired.loc[chi2_desired["code"] == "run1", "chi2_W4"] = 10**2 / 0.5**2
    chi2_desired.loc[chi2_desired["code"] == "run0", "chi2"] = (0**2 + 2**2 + 4**2 + 6**2) / 0.5**2
    chi2_desired.loc[chi2_desired["code"] == "run1", "chi2"] = (2**2 + (-2)**2 + 6**2 + 10**2) / 0.5**2

    chi2 = calcMagChi2(obs, observations, residuals)
    
    pd.testing.assert_frame_equal(chi2_desired, chi2)
    
def test_calcMagReducedChi2():
    # Initialize observatory
    obs = WISE()

    # Create a DataFrame of magnitudes: W1: 1, W2: 2, W3: 3, W4: 4
    columns=["mag_W1", "mag_W2", "mag_W3", "mag_W4"]
    observations = pd.DataFrame(np.ones((100, 4)), columns=columns)
    observations["obs_id"] = np.arange(1, len(observations) + 1)
    observations["designation"] = (["test_1" for i in range(40)] 
        + ["test_2" for i in range(60)])
    observations = observations[["obs_id", "designation"] + columns]
    for i, c in enumerate(columns[1:]):
        observations[c] += (i+1)

    chi2_W1 = np.random.rand(200) * 5 
    chi2_W2 = np.random.rand(200) * 3
    chi2_W3 = np.random.rand(200) * 10
    chi2_W4 = np.random.rand(200) * 2
    chi2 = chi2_W1 + chi2_W2 + chi2_W3 + chi2_W4

    chi2_columns =["chi2_W1", "chi2_W2", "chi2_W3", "chi2_W4", "chi2"]
    chi2_df = pd.DataFrame(np.ones((200, 5)), columns=chi2_columns)
    chi2_df["code"] = (["run0" for i in range(len(observations))] 
        + ["run1" for i in range(len(observations))])
    chi2_df["obs_id"] = np.concatenate([observations["obs_id"], observations["obs_id"]])
    chi2_df["chi2_W1"] = chi2_W1
    chi2_df["chi2_W2"] = chi2_W2
    chi2_df["chi2_W3"] = chi2_W3
    chi2_df["chi2_W4"] = chi2_W4
    chi2_df["chi2"] = chi2

    reduced_chi2 = np.array([
        np.sum(chi2[0:40]) / (40*4 - 2),
        np.sum(chi2[100:140]) / (40*4 - 4),
        np.sum(chi2[40:100]) / (60*4 - 2),
        np.sum(chi2[140:]) / (60*4 - 4),
    ])
    desired_reduced_chi2 = pd.DataFrame({
        "designation" : ["test_1", "test_1", "test_2", "test_2"],
        "code" : ["run0", "run1", "run0", "run1"],
        "chi2" : [
            np.sum(chi2[0:40]), 
            np.sum(chi2[100:140]),  
            np.sum(chi2[40:100]),
            np.sum(chi2[140:])],
        "num_W1" : [40, 40, 60, 60],
        "num_W2" : [40, 40, 60, 60],
        "num_W3" : [40, 40, 60, 60],
        "num_W4" : [40, 40, 60, 60],
        "num_obs" : [40*4, 40*4, 60*4, 60*4],
        "reduced_chi2" : reduced_chi2,
    })

    summary = pd.DataFrame({
        "parameter" : ["logD", "logT1", "logD", "logT1", "eps_W1W2", "eps_W3"],
        "code" : ["run0", "run0", "run1", "run1", "run1", "run1"],
        })

    reduced_chi2 = calcMagReducedChi2(obs, chi2_df, observations, summary)

    pd.testing.assert_frame_equal(reduced_chi2, desired_reduced_chi2)


    # Create a DataFrame of magnitudes: W1: 1, W2: 2, W3: 3, W4: 4
    columns=["mag_W1", "mag_W2", "mag_W3", "mag_W4"]
    observations = pd.DataFrame(np.ones((100, 4)), columns=columns)
    observations["obs_id"] = np.arange(1, len(observations) + 1)
    observations["designation"] = (["test_1" for i in range(40)] 
        + ["test_2" for i in range(60)])
    observations = observations[["obs_id", "designation"] + columns]
    for i, c in enumerate(columns[1:]):
        observations[c] += (i+1)

    no_obs_w1 = np.random.choice(np.arange(1, 40 + 1), 10, replace=False)
    no_obs_w3 = np.random.choice(np.arange(1, 40 + 1), 7, replace=False)
    no_obs_w2 = np.random.choice(np.arange(41, 100 + 1), 23, replace=False)
    no_obs_w4 = np.random.choice(np.arange(41, 100 + 1), 12, replace=False)
    observations.loc[observations["obs_id"].isin(no_obs_w1), "mag_W1"] = np.NaN
    observations.loc[observations["obs_id"].isin(no_obs_w3), "mag_W3"] = np.NaN
    observations.loc[observations["obs_id"].isin(no_obs_w2), "mag_W2"] = np.NaN
    observations.loc[observations["obs_id"].isin(no_obs_w4), "mag_W4"] = np.NaN

    chi2_W1 = np.random.rand(200) * 5 
    chi2_W1[no_obs_w1] = np.NaN
    chi2_W2 = np.random.rand(200) * 3
    chi2_W2[no_obs_w2] = np.NaN
    chi2_W3 = np.random.rand(200) * 10
    chi2_W3[no_obs_w3] = np.NaN
    chi2_W4 = np.random.rand(200) * 2
    chi2_W4[no_obs_w4] = np.NaN
    chi2 = chi2_W1 + chi2_W2 + chi2_W3 + chi2_W4

    chi2_columns =["chi2_W1", "chi2_W2", "chi2_W3", "chi2_W4", "chi2"]
    chi2_df = pd.DataFrame(np.ones((200, 5)), columns=chi2_columns)
    chi2_df["code"] = (["run0" for i in range(len(observations))] 
        + ["run1" for i in range(len(observations))])
    chi2_df["obs_id"] = np.concatenate([observations["obs_id"], observations["obs_id"]])
    chi2_df["chi2_W1"] = chi2_W1
    chi2_df["chi2_W2"] = chi2_W2
    chi2_df["chi2_W3"] = chi2_W3
    chi2_df["chi2_W4"] = chi2_W4
    chi2_df["chi2"] = chi2

    reduced_chi2 = np.array([
        np.nansum(chi2[0:40]) / (40*4 - 10 - 7 - 2),
        np.nansum(chi2[100:140]) / (40*4 - 10 - 7 - 4),
        np.nansum(chi2[40:100]) / (60*4 - 12 - 23 - 2),
        np.nansum(chi2[140:]) / (60*4 - 12 - 23 - 4),
    ])
    desired_reduced_chi2 = pd.DataFrame({
        "designation" : ["test_1", "test_1", "test_2", "test_2"],
        "code" : ["run0", "run1", "run0", "run1"],
        "chi2" : [
            np.nansum(chi2[0:40]), 
            np.nansum(chi2[100:140]),  
            np.nansum(chi2[40:100]),
            np.nansum(chi2[140:])],
        "num_W1" : [40-10, 40-10, 60, 60],
        "num_W2" : [40, 40, 60-23, 60-23],
        "num_W3" : [40-7, 40-7, 60, 60],
        "num_W4" : [40, 40, 60-12, 60-12],
        "num_obs" : [40*4 - 10 - 7, 40*4 - 10 - 7, 60*4 - 12 - 23, 60*4 - 12 - 23],
        "reduced_chi2" : reduced_chi2,
    })

    summary = pd.DataFrame({
        "parameter" : ["logD", "logT1", "logD", "logT1", "eps_W1W2", "eps_W3"],
        "code" : ["run0", "run0", "run1", "run1", "run1", "run1"],
        })

    reduced_chi2 = calcMagReducedChi2(obs, chi2_df, observations, summary)

    pd.testing.assert_frame_equal(reduced_chi2, desired_reduced_chi2)