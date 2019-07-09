# Asteroid Thermal Modeling 

[![Build Status](https://dev.azure.com/moeyensj/atm/_apis/build/status/moeyensj.atm?branchName=master)](https://dev.azure.com/moeyensj/atm/_build/latest?definitionId=3&branchName=master)
[![Build Status](https://www.travis-ci.com/moeyensj/atm.svg?token=sWjpnqPgpHyuq3j7qPuj&branch=master)](https://www.travis-ci.com/moeyensj/atm)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/atm/badge.svg?branch=master&t=wABWWi)](https://coveralls.io/github/moeyensj/atm?branch=master)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

### Installation:

To install pre-requisite software (with your choice of favorite package manager):

With `conda`:  
```
conda install -c defaults -c conda-forge --file requirements.txt
```

With `pip`:  
```
pip install -r requirements.txt
```

If using `conda`, we recommend creating a fresh environment as follows:  
```
conda create -n atm_py36 -c defaults -c conda-forge --file requirements.txt python=3.6
```

Once pre-requisite software has been installed, this repository can be cloned (or forked if you wish to contribute code):

Clone the repository using:

```
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:moeyensj/atm.git  
```  

At this stage, you can use the code to create model spectral energy distributions for the different thermal models. However, if you want to fit observations of asteroids you will need to download the lookup tables for the relevant models. 

To download the NEATM lookup tables:

```
git lfs pull --include="NEATM*.npz"
``` 

To download all model tables:

```
git lfs pull --include="*.npz"
```  

If you would like to access the two observations databases:

```
git lfs pull --include="data/sample*.db"
```

To download thermal modeling results databases:

```
git lfs pull --include="data/runs/atm_results*.db"
```

To download everything:  
```
git lfs pull
```
