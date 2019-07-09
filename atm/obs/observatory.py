#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

from ..constants import Constants

__all__ = ["Observatory"]

# Initialize constants
c = Constants.LIGHTSPEED

class Observatory(object):

    def __init__(self, name=None,
                       acronym=None,
                       filterNames=None,
                       filterEffectiveLambdas=None,
                       filterQuadratureLambdas=None,
                       filterEffectiveNus=None,
                       filterQuadratureNus=None,
                       fluxLambdaNorm=None,
                       fluxNuNorm=None,
                       deltaMagnitudes=None):
        self._name = name
        self._acronym = acronym
        self._filterNames = filterNames
        self._filterEffectiveLambdas = filterEffectiveLambdas
        self._filterQuadratureLambdas = filterQuadratureLambdas
        self._filterEffectiveNus = filterEffectiveNus
        self._filterQuadratureNus = filterQuadratureNus
        self._fluxLambdaNorm = fluxLambdaNorm
        self._fluxNuNorm = fluxNuNorm
        self._deltaMagnitudes = deltaMagnitudes
        
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @name.deleter
    def name(self):
        del self._name

    @property
    def acronym(self):
        return self._acronym

    @acronym.setter
    def acronym(self, value):
        self._acronym = value

    @acronym.deleter
    def acronym(self):
        del self._acronym

    @property
    def filterNames(self):
        return self._filterNames

    @filterNames.setter
    def filterNames(self, value):
        self._filterNames = value

    @filterNames.deleter
    def filterNames(self):
        del self._filterNames

    @property
    def filterEffectiveLambdas(self):
        return self._filterEffectiveLambdas

    @filterEffectiveLambdas.setter
    def filterEffectiveLambdas(self, value):
        self._filterEffectiveLambdas = value

    @filterEffectiveLambdas.deleter
    def filterEffectiveLambdas(self):
        del self._filterEffectiveLambdas

    @property
    def filterQuadratureLambdas(self):
        return self._filterQuadratureLambdas

    @filterQuadratureLambdas.setter
    def filterQuadratureLambdas(self, value):
        self._filterQuadratureLambdas = value

    @filterQuadratureLambdas.deleter
    def filterQuadratureLambdas(self):
        del self._filterQuadratureLambdas

    @property
    def filterEffectiveNus(self):
        return self._filterEffectiveNus

    @filterEffectiveNus.setter
    def filterEffectiveNus(self, value):
        self._filterEffectiveNus = value

    @filterEffectiveNus.deleter
    def filterEffectiveNus(self):
        del self._filterEffectiveNus

    @property
    def filterQuadratureNus(self):
        return self._filterQuadratureNus

    @filterQuadratureNus.setter
    def filterQuadratureNus(self, value):
        self._filterQuadratureNus = value

    @filterQuadratureNus.deleter
    def filterQuadratureNus(self):
        del self._filterQuadratureNus

    @property
    def fluxLambdaNorm(self):
        return self._fluxLambdaNorm

    @fluxLambdaNorm.setter
    def fluxLambdaNorm(self, value):
        print("Cannot set fluxLambdaNorm, please re-initialize class.")

    @fluxLambdaNorm.deleter
    def fluxLambdaNorm(self):
        print("Cannot delete fluxLambdaNorm, please re-initialize class.")

    @property
    def fluxNuNorm(self):
        return self._fluxNuNorm

    @fluxNuNorm.setter
    def fluxNuNorm(self, value):
        print("Cannot set fluxNuNorm, please re-initialize class.")

    @fluxNuNorm.deleter
    def fluxNuNorm(self):
        print("Cannot delete fluxNuNorm, please re-initialize class.")
        
    @property
    def deltaMagnitudes(self):
        return self._deltaMagnitudes

    @deltaMagnitudes.setter
    def deltaMagnitudes(self, value):
        self._deltaMagnitudes = value

    @deltaMagnitudes.deleter
    def deltaMagnitudes(self):
        del self._deltaMagnitudes

    def bandpassLambda(self):
        raise NotImplementedError("Bandpass lambda is not defined.")

    def bandpassNu(self):
        raise NotImplementedError("Bandpass nu is not defined.")

    def _testshape(self, values):
        if len(values.shape) == 1:
            num_filters = values.shape[0]
            num_obs = 1
        if len(values.shape) == 2:
            num_filters = values.shape[1]
            num_obs = values.shape[0]

        if num_filters != len(self.filterNames):
            raise ValueError("""
                The array of values does 
                not have the expected shape.

                This observatory ({0}) expects data for {1} filter{4}. 
                The array of values has {2} data points
                for {3} filter{5}. 

                If data for this filter is not available or if any observations
                are missing data in a specific filter replace these values
                with np.NaN.
            """.format(
                self.name, 
                len(self.filterNames), 
                num_obs, 
                num_filters,
                "s" if len(self.filterNames) > 1 else "",
                "s" if num_filters > 1 else ""))
        return

    
    def convertFluxNuToFluxLambda(self, fluxNu):
        """
        Convert frequency fluxes to wavelength fluxes. Wavelengths fluxes
        are used for thermal models. 

        Use np.NaN values for any missing data points. 
        
        Parameters
        ----------
        fluxNu : `~np.ndarray` (N, N)
            Array of flux values in units of Jy (N data points by M filters).
        
        Returns
        -------
        fluxLambda : `~np.ndarray` (N, N)
            Array of flux values in units of W m^-3 (N data points by M filters).
        
        """
        return (10**-26 * c / self.filterEffectiveLambdas**2 * fluxNu).astype(float)
        
    def convertFluxLambdaToFluxNu(self, fluxLambda):
        """
        Convert wavelength fluxes to frequency fluxes. Wavelengths fluxes
        are used for thermal models. 

        Use np.NaN values for any missing data points. 
        
        Parameters
        ----------
        fluxLambda : `~np.ndarray` (N, M)
            Array of flux values in units of W m^-3 (N data points by M filters).
            
        Returns
        -------
        fluxNu : `~np.ndarray` (N, M)
            Array of flux values in units of Jy (N data points by M filters).

        """
        return (10**26 * self.filterEffectiveLambdas**2 / c * fluxLambda).astype(float)

    def convertFluxLambdaToMag(self, fluxLambda):
        """
        Convert flux lambda to magnitudes. The magnitude system to which to convert to
        can be changed by appropriately setting the observatory class's deltaMagnitudes
        attribute.

        Use np.NaN values for any missing data points. 
        
        Parameters
        ----------
        fluxLambda : `~np.ndarray` (N, M)
            Array of flux values in units of W m^-3 (N data points by M filters).

        Returns
        -------
        mag : `~np.ndarray` (N, M)
            Array of magnitudes (N data points by M filters).
        """
        self._testshape(fluxLambda)

        if self._fluxNuNorm is not None:
            fluxNu = self.convertFluxLambdaToFluxNu(fluxLambda)
            return -2.5 * np.log10(fluxNu / self.fluxNuNorm) 
        elif self._fluxLambdaNorm is not None:
            return -2.5 * np.log10(fluxLambda / self.fluxLambdaNorm)
        else:
            raise valueError("Neither fluxLambdaNorm nor fluxNuNorm are defined.")
            
    def convertMagToFluxLambda(self, mag):
        """
        Convert flux lambda to magnitudes. The magnitude system to which to convert to
        can be changed by appropriately setting the observatory class's deltaMagnitudes
        attribute.

        Use np.NaN values for any missing data points. 
        
        Parameters
        ----------
        fluxLambda : `~np.ndarray` (N, M)
            Array of flux values in units of W m^-3 (N data points by M filters).

        Returns
        -------
        mag : `~np.ndarray` (N, M)
            Array of magnitudes (N data points by M filters).
        """
        self._testshape(mag)

        if self._fluxNuNorm is not None:
            fluxNu = 10**(- mag / 2.5) * self._fluxNuNorm
            return self.convertFluxNuToFluxLambda(fluxNu)
        elif self._fluxLambdaNorm is not None:
            return 10**(- mag / 2.5) * self._fluxLambdaNorm
        else:
            raise valueError("Neither fluxLambdaNorm nor fluxNuNorm are defined.")
            
    def convertFluxLambdaErrToMagErr(self, fluxLambda, fluxLambdaErr):
        """
        Convert flux lambda errors to magnitude errors. The magnitude system to which to convert to
        can be changed by appropriately setting the observatory class's deltaMagnitudes
        attribute.

        Use np.NaN values for any missing data points. 
        
        Parameters
        ----------
        fluxLambda : 
            Array of flux values in units of W m^-3 (N data points by M filters).
        fluxLambdaErr : `~np.ndarray` (N, M)
            Array of flux error values in units of W m^-3 (N data points by M filters).

        Returns
        -------
        magErr : `~np.ndarray` (N, M)
            Array of magnitude errors (N data points by M filters).
        """
        self._testshape(fluxLambda)
        self._testshape(fluxLambdaErr)
        return 2.5 / np.log(10) * (fluxLambdaErr / fluxLambda)
    
    def convertMagErrToFluxLambdaErr(self, mag, magErr):
        """
        Convert magnitude errors to flux lambda errors. The magnitude system to which to convert to
        can be changed by appropriately setting the observatory class's deltaMagnitudes
        attribute.

        Use np.NaN values for any missing data points. 
        
        Parameters
        ----------
        mag : `~np.ndarray` (N, M)
            Array of magnitudes (N data points by M filters).
        magErr : `~np.ndarray` (N, M)
            Array of magnitude errors (N data points by M filters).

        Returns
        -------
        fluxLambdaErr : `~np.ndarray` (N, M)
            Array of flux error values in units of W m^-3 (N data points by M filters).
        """
        self._testshape(mag)
        self._testshape(magErr)
        return magErr * np.log(10) / 2.5 * self.convertMagToFluxLambda(mag)
            