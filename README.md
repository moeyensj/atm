# ATM
Asteroid Thermal Modeling  
[![Build Status](https://dev.azure.com/moeyensj/atm/_apis/build/status/moeyensj.atm?branchName=master)](https://dev.azure.com/moeyensj/atm/_build/latest?definitionId=3&branchName=master)
[![Build Status](https://www.travis-ci.com/moeyensj/atm.svg?token=sWjpnqPgpHyuq3j7qPuj&branch=master)](https://www.travis-ci.com/moeyensj/atm)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/atm/badge.svg?branch=master&t=wABWWi)](https://coveralls.io/github/moeyensj/atm?branch=master)
[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/atm)](https://hub.docker.com/r/moeyensj/atm)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Table of Contents
- [Overview](#Overview)
- [Results/Output](#Results/Output)
- [Installation](#installation)
- [Contributions](#contributions)


## Overview

Motivated by a desire to enable transparency and reproducibility of results, and to foster collaborative software development, we release  ATM, a general tool for interpreting infrared flux measurements of asteroids.

We emphasize that our analysis presented here, and corresponding catalogs with best-fit sizes and other parameters, are far from definitive and can be improved in various ways. 

This is the main code repository that contains the ATM code. While this code can be used in a standalone fashion, please consider also downloading the atm_notebooks and atm_data repositories. 

The reference paper "ATM: An open-source tool for asteroid thermal modeling and its application to NEOWISE data" can be found at: https://doi.org/10.1016/j.icarus.2019.113575. 

The corresponding notebook repository with tutorial notebooks and notebooks that reproduce all the results in the reference paper can be found at: https://github.com/moeyensj/atm_notebooks. 

The data repository used for the reference paper can be found at: https://github.com/moeyensj/atm_data

## Details (Based on Abstract)

This Python package is created to model asteroid flux measurements to estimate an asteroid’s size, surface temperature distribution, and emissivity. 

We implement the most popular static asteroid thermal models with A number of the most popular static asteroid the reflected solar light contribution and Kirchhoff’s

law accounted for.

NEATM (Near-Earth Asteroid Thermal Model): Is a thermal model used to interpret infrared observations of near-Earth asteroids (NEAs). Developed to address limitations present in the Standard Thermal Model (STM).

STM (Standard Thermal Model): The foundational thermal model used for interpreting infrared observations of asteroids.

FRM (Fast Rotating Model): Thermel model used for interpreting the infrared observations of celestial bodies, particularly asteroids. 

Included: Data files with ~10 million WISE flux measurements for ~150,000 unique asteroids and additional Minor Planet Center data are also included with the package, as well as Python Jupyter Notebooks with examples of

how to select subsamples of objects,filter and process data, and use ATM to fit for the desired model parameters. 

## Results/Output

Achieve a sub-percent bias and a scatter of only 6% for best-fit size estimates for well-observed asteroids published in 2016 by the NEOWISE team (Mainzer et al., 2016).

The majority of over 100,000 objects with WISE-based size estimates random uncertainties (precision) are about 10%; systematic uncertainties  within the adopted model framework, such as NEATM, and with assumed emissivity

for WISE W3 and W4 bands, are likely in the range of 10–20%. Hence, the accuracy of WISE-based asteroid size estimates is approximately in the range of 15–20% for most objects.

Our analysis gives support to the claim by Harris & Drube (2014) that candidate metallic asteroids can be selected using the best-fit temperature parameter and infrared albedo.

We investigate a correlation between SDSS colors and optical albedo derived using WISE-based size estimates and show that this correlation can be used to estimate asteroid sizes with optical data alone, with a precision 

of about 17% relative to WISE-based size estimates. 

## Installation

We recommend installing the code along one of two installation paths: either a source code installation, or an installation via docker.

### Source

Clone this repository using either `ssh` or `https`:

```GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:moeyensj/atm.git``` 

Once cloned and downloaded, `cd` into the repository. To install ATM in its own `conda` enviroment please do the following:  

```conda create -n atm_py36 -c defaults -c conda-forge --file requirements.txt python=3.6```  

Or, to install ATM in a pre-existing `conda` environment called `env`:  

```conda activate env```  
```conda install -c defaults -c conda-forge -c astropy --file requirements.txt```  

Or, to install pre-requisite software using `pip`:  

```pip install -r requirements.txt```

At this stage, you can use the code to create model spectral energy distributions for the different thermal models. However, if you want to fit observations of asteroids you will need to download the lookup tables for the relevant models. Activate the environment in which the ATM pre-requisite software is installed and then proceed to download the model tables:

To download the NEATM lookup tables:

```git lfs pull --include="NEATM*.npz"``` 

To download all model tables:

```git lfs pull --include="*.npz"```  

To download everything:  

```git lfs pull```

Once pre-requisites have been installed and any additional data has been downloaded then:  

```python setup.py install```

### Docker

A Docker container with the latest version of the code can be pulled using:  

```docker pull moeyensj/atm:latest```

To run the container:  

```docker run -it moeyensj/atm:latest```

The ATM code is installed the /projects directory, and is by default also installed in the container's Python installation. 

**If you would like to run Jupyter Notebook or Juptyter Lab with ATM please see the installation instructions in the ATM notebooks repository.**

## Contributions

Contributions and feature requests are welcome for more details of potential contributions please check out the issues page.
