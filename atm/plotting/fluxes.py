#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import platform
if platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from ..config import Config

__all__ = ["plotObservations",
           "plotSED"]

def plotObservations(obs, data, 
                     plotMedian=False,
                     ax=None,
                     figKwargs={"dpi": 200}, 
                     columnMapping=Config.columnMapping):
    """
    Plot observations for a single asteroid. Can also plot the median flux and median 
    errors on flux if desired. 
    
    Parameters
    ----------
    obs : `~atm.obs.Observatory`
        Observatory object containing filter bandpass information.
    data : `~pandas.DataFrame`
        DataFrame containing the relevant data to plot. The user should define 
        the columnMapping dictionary which maps internally used variables to the 
        variables used in the user's DataFrame.
    plotMedian : bool, optional
        Instead of plotting all fluxes and flux uncertainties plot the median per band.
        [Default = False]
    ax : {None, `~matplotlib.Axes.ax`}, optional
        Plot on the passed matplotlib axis. If ax is None then create a new figure.
        [Default = None]
    figKwargs : dict, optional
        Keyword arguments to pass to figure API.
        [Default = {"dpi" : 200}]
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]

    Returns
    -------
    `~matplotlib.Figure.figure, `~matplotlib.Axes.ax` 
        Returns matplotlib figure and axes object if the user passed no 
        pre-existing axes. 
    """
    m_to_mum = 1e6 # simple conversion from m to micron
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, **figKwargs)
        
    for i, f in enumerate(obs.filterNames):
        if plotMedian is True:
            ax.errorbar(obs.filterEffectiveLambdas[i] * m_to_mum, 
                        1/m_to_mum*np.median(data[columnMapping["flux_si"][i]].values), 
                        yerr=1/m_to_mum*np.median(data[columnMapping["fluxErr_si"][i]].values), 
                        fmt='o',
                        c="k",
                        ms=2,
                        capsize=1,
                        elinewidth=1)
        else:
            ax.errorbar(obs.filterEffectiveLambdas[i] * m_to_mum * np.ones(len(data)), 
                        1/m_to_mum*data[columnMapping["flux_si"][i]].values, 
                        yerr=1/m_to_mum*data[columnMapping["fluxErr_si"][i]].values, 
                        fmt='o',
                        c="k",
                        ms=2,
                        capsize=1,
                        elinewidth=1)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"Flux ($F_\lambda$) [$W m^{-2} \mu m^{-1}$]")
    ax.set_xlabel(r"Wavelength ($\lambda$) [$\mu m$]")
    
    if fig is not None:
        return fig, ax
    else:
        return
    
def plotSED(sed, 
            ax=None, 
            figKwargs={"dpi" : 200}, 
            plotKwargs={"label" : "SED", 
                        "ls" : ":", 
                        "c" : "k",
                        "lw" : 1}):
    """
    Plot an asteroid's thermal spectral energy distribution. This function 
    takes the output DataFrame from `~atm.function.calcFluxLambdaSED` and 
    plots the flux as a function of wavelength.
    
    Parameters
    ----------
    sed : `~pandas.DataFrame`
        DataFrame containing a columns of fluxes and wavelengths. 
    ax : {None, `~matplotlib.Axes.ax`}, optional
        Plot on the passed matplotlib axis. If ax is None then create a new figure.
        [Default = None]
    figKwargs : dict, optional
        Keyword arguments to pass to figure API.
        [Default = {"dpi" : 200}]
    plotKwargs : dict, optional
        Keyword arguments to pass to ax.plot API.
        [Default = {"label" : "SED", 
                    "ls" : ":", 
                    "c" : "k",
                    "lw" : 1}]
    columnMapping : dict, optional
        This dictionary should define the column names of the user's data relative to the
        internally used names.
        [Default = `~atm.Config.columnMapping`]

    Returns
    -------
    `~matplotlib.Figure.figure, `~matplotlib.Axes.ax` 
        Returns matplotlib figure and axes object if the user passed no 
        pre-existing axes. 

    See Also
    --------
    `~atm.function.calcFluxLambdaSED` : Calculates a thermal model SED for a given wavelength 
        range and step.
    """
    
    m_to_mum = 1e6 # simple conversion from m to micron
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, **figKwargs)
    
    ax.plot(sed["lambda"] * m_to_mum, 1/m_to_mum*sed["flux"], **plotKwargs)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"Flux ($F_\lambda$) [$W m^{-2} \mu m^{-1}$]")
    ax.set_xlabel(r"Wavelength ($\lambda$) [$\mu m$]")
    
    if fig is not None:
        return fig, ax
    else:
        return
    
    