#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import platform
if platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from ..analysis import calcStdDev

__all__ = ["plotHist"]

def plotHist(ax, xValues, xRange, 
             bins=10, 
             numGauss=1, 
             robust=True, 
             swapAxes=False,
             useMedian=True,
             statRange=None,
             histKwargs={
                "histtype" : "stepfilled",
                "alpha" : 0.5, 
                "normed" : True}, 
             plotKwargs={
                "ls" : "-",
                "c" : "red"}, 
             plotKwargsComponents={
                "ls" : ":",
                "c" : "red",
                "alpha" : 0.5}):
    
    xValues_cut = xValues[(xValues > xRange[0]) & (xValues < xRange[1])]
    if statRange is not None:
        xValues_stat = xValues_cut[(xValues_cut > statRange[0]) & (xValues_cut < statRange[1])]
    else:
        xValues_stat = xValues_cut
    print("{} values are outside the defined minimum and maximum.".format(len(xValues) - len(xValues_cut)))
    
    if statRange is not None:
        print("Statistics will be calculated between {} and {}".format(statRange[0], statRange[1]))
    
    ax.hist(xValues, bins=np.linspace(xRange[0], xRange[1], bins), **histKwargs)
    
    stats = []
    if numGauss > 0:
        if numGauss == 1:
            stats = np.zeros(3)
            w = 1
            if useMedian is True:
                mu = np.median(xValues_stat)
            else:
                mu = np.mean(xValues_stat)
                
            sigma = calcStdDev(xValues_stat, robust=robust)
            grid = np.linspace(*xRange, 1000)
            gauss = norm(mu, sigma).pdf(grid)
            if swapAxes is True:
                ax.plot(gauss, grid, **plotKwargs)
            else:
                ax.plot(grid, gauss, **plotKwargs)

            stats[0] = mu
            stats[1] = sigma
            stats[2] = w

        else:
            # Fit GMM
            gmm = GaussianMixture(n_components=numGauss, covariance_type="full", tol=0.00000001)
            gmm = gmm.fit(X=np.expand_dims(xValues_stat, 1))

            # Evaluate GMM
            gmm_x = np.linspace(*xRange, 1000)
            gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1))) 
            
            if swapAxes is True:
                ax.plot(gmm_y, gmm_x, **plotKwargs)
            else:
                ax.plot(gmm_x, gmm_y, **plotKwargs)

            mu = gmm.means_.ravel()
            sigma = np.sqrt(gmm.covariances_.ravel())
            w = gmm.weights_.ravel()

            for i in range(numGauss):
                grid = np.linspace(*xRange, 1000)
                gauss = norm(mu[i], sigma[i]).pdf(grid)
                ax.plot(grid, w[i] * gauss, **plotKwargsComponents)

            stats = np.zeros((numGauss, 3))
            stats[:, 0] = mu
            stats[:, 1] = sigma
            stats[:, 2] = w
        
    return ax, stats
    