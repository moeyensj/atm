#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

from ...config import Config
from ...obs import WISE
from ..colors import calcColors
from ..colors import calcModelColors
from ..colors import calcColorResiduals

def test_calcColors():
    # Initialize observatory
    obs = WISE()
    
    # Create a DataFrame of magnitudes: W1: 1, W2: 2, W3: 3, W4: 4
    # Colors should all be -1
    columns=["mag_W1", "mag_W2", "mag_W3", "mag_W4"]
    observations = pd.DataFrame(np.ones((10000, 4)), columns=columns)
    observations["obs_id"] = np.arange(1, len(observations) + 1)
    observations = observations[["obs_id"] + columns]
    for i, c in enumerate(columns[1:]):
        observations[c] += (i+1)
    
    colors = calcColors(obs, observations)
    desired_colors = pd.DataFrame(np.ones((10000, 3)) * -1, columns=["W1-W2", "W2-W3", "W3-W4"])
    desired_colors["obs_id"] = observations["obs_id"].values
    desired_colors = desired_colors[["obs_id", "W1-W2", "W2-W3", "W3-W4"]]
    
    pd.testing.assert_frame_equal(desired_colors, colors)
    
    # Initialize observatory
    obs = WISE()
    
    # Create a DataFrame of magnitudes: W1: 1, W2: 0, W3: -1, W4: -2
    # Colors should all be 1
    columns=["mag_W1", "mag_W2", "mag_W3", "mag_W4"]
    observations = pd.DataFrame(np.ones((10000, 4)), columns=columns)
    observations["obs_id"] = np.arange(1, len(observations) + 1)
    observations = observations[["obs_id"] + columns]
    for i, c in enumerate(columns[1:]):
        observations[c] += (-1 * (i + 1))
        
    colors = calcColors(obs, observations)
    desired_colors = pd.DataFrame(np.ones((10000, 3)) * 1, columns=["W1-W2", "W2-W3", "W3-W4"])
    desired_colors["obs_id"] = observations["obs_id"].values
    desired_colors = desired_colors[["obs_id", "W1-W2", "W2-W3", "W3-W4"]]
    
    pd.testing.assert_frame_equal(desired_colors, colors)
    
def test_calcModelColors():
    # Initialize observatory
    obs = WISE()
    
    # Create a DataFrame of magnitudes: W1: 1, W2: 2, W3: 3, W4: 4
    # Colors should all be -1
    columns=["model_mag_W1", "model_mag_W2", "model_mag_W3", "model_mag_W4"]
    model_observations = pd.DataFrame(np.ones((10000, 4)), columns=columns)
    model_observations["obs_id"] = np.arange(1, len(model_observations) + 1)
    model_observations["code"] = ["run0" for i in range(len(model_observations))]
    model_observations = model_observations[["obs_id",  "code"] + columns]
    for i, c in enumerate(columns[1:]):
        model_observations[c] += (i+1)
    
    model_colors = calcModelColors(obs, model_observations)
    desired_colors = pd.DataFrame(np.ones((10000, 3)) * -1, columns=["model_W1-W2", "model_W2-W3", "model_W3-W4"])
    desired_colors["obs_id"] = model_observations["obs_id"].values
    desired_colors["code"] = ["run0" for i in range(len(desired_colors))]

    desired_colors = desired_colors[["obs_id", "code", "model_W1-W2", "model_W2-W3", "model_W3-W4"]]
    
    pd.testing.assert_frame_equal(desired_colors, model_colors)
    
    # Initialize observatory
    obs = WISE()
    
    # Create a DataFrame of magnitudes: W1: 1, W2: 0, W3: -1, W4: -2
    # Colors should all be 1
    columns=["model_mag_W1", "model_mag_W2", "model_mag_W3", "model_mag_W4"]
    model_observations = pd.DataFrame(np.ones((10000, 4)), columns=columns)
    model_observations["obs_id"] = np.arange(1, len(model_observations) + 1)
    model_observations["code"] = ["run0" for i in range(len(model_observations))]
    model_observations = model_observations[["obs_id", "code"] + columns]
    for i, c in enumerate(columns[1:]):
        model_observations[c] += (-1 * (i + 1))
        
    model_colors = calcModelColors(obs, model_observations)
    desired_colors = pd.DataFrame(np.ones((10000, 3)) * 1, columns=["model_W1-W2", "model_W2-W3", "model_W3-W4"])
    desired_colors["obs_id"] = model_observations["obs_id"].values
    desired_colors["code"] = ["run0" for i in range(len(desired_colors))]
    desired_colors = desired_colors[["obs_id", "code", "model_W1-W2", "model_W2-W3", "model_W3-W4"]]
    
    pd.testing.assert_frame_equal(desired_colors, model_colors)
    
def test_calcColorResiduals():
    # Initialize observatory
    obs = WISE()

    # Create a DataFrame of magnitudes: W1: 1, W2: 2, W3: 3, W4: 4
    # Colors should all be -1
    columns=["mag_W1", "mag_W2", "mag_W3", "mag_W4"]
    observations = pd.DataFrame(np.ones((10000, 4)), columns=columns)
    observations["obs_id"] = np.arange(1, len(observations) + 1)
    observations = observations[["obs_id"] + columns]
    for i, c in enumerate(columns[1:]):
        observations[c] += (i+1)

    colors = calcColors(obs, observations)

    # Create a DataFrame of model magnitudes: 
    # Colors should all be 1 for run0 (W1: 1, W2: 0, W3: -1, W4: -2)
    # Colors should all be 3 for run1 (W1: 3, W2: 0, W3: -3, W4: -6)
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

    model_colors = calcModelColors(obs, model_observations)

    residual_columns=["residual_W1-W2", "residual_W2-W3", "residual_W3-W4"]
    desired_residuals = pd.DataFrame(np.ones((20000, 3)) * -2.0, columns=residual_columns)
    desired_residuals["obs_id"] = model_observations["obs_id"].values
    desired_residuals["code"] = model_observations["code"].values
    desired_residuals = desired_residuals[["obs_id", "code"] + residual_columns]
    for r in residual_columns:
        desired_residuals.loc[desired_residuals["code"] == "run1", r] = -4.0

    color_residuals = calcColorResiduals(obs, colors, model_colors)
    pd.testing.assert_frame_equal(desired_residuals, color_residuals)
    