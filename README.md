# ATM
Asteroid Thermal Modeling  
[![Build Status](https://dev.azure.com/moeyensj/atm/_apis/build/status/moeyensj.atm?branchName=master)](https://dev.azure.com/moeyensj/atm/_build/latest?definitionId=3&branchName=master)
[![Build Status](https://www.travis-ci.com/moeyensj/atm.svg?token=sWjpnqPgpHyuq3j7qPuj&branch=master)](https://www.travis-ci.com/moeyensj/atm)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/atm/badge.svg?branch=master&t=wABWWi)](https://coveralls.io/github/moeyensj/atm?branch=master)
[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/atm)](https://hub.docker.com/r/moeyensj/atm)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

This is the main code repository that contains the ATM code. While this code can be used in a standalone fashion, please consider also downloading the atm_notebooks and atm_data repositories. 

The reference paper "ATM: An open-source tool for asteroid thermal modeling and its application to NEOWISE data" can be found at: https://doi.org/10.1016/j.icarus.2019.113575. 

The corresponding notebook repository with tutorial notebooks and notebooks that reproduce all the results in the reference paper can be found at: https://github.com/moeyensj/atm_notebooks. 

The data repository used for the reference paper can be found at: https://github.com/moeyensj/atm_data

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
