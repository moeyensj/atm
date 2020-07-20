#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import glob
import warnings
import numpy as np
from scipy import interpolate
from multiprocessing import Pool

from ..config import Config

__all__ = ["Model"]


class Model(object):
    """
    Model: Class that defines the temperature distribution across the surface
        of an asteroid, the resulting emitted flux and how to account for the
        phase angle of the observation.

    Parameters
    ----------
    name : str
        Model name.
    acronym : str
        Model acronym.
    tableDir : str
        Look-up table directory.
    tableLambdas : list
        List of wavelengths at which look-up tables exist.
    tableLambdaFiles : dict
        Dictionary keyed on the wavelengths of look-up tables
        and valued with the path of their location.
    tableNus : list
        List of frequencies at which look-up tables exist.
    tableNuFiles : dict
        Dictionary keyed on the frequencies of look-up tables
        and valued with the path of their location.
    verbose : bool
        Print progress statements?
        [Default = `~atm.Config.verbose`]
    """
    def __init__(self,
                 name="Model",
                 acronym=None,
                 tableDir=None,
                 tableLambdas=None,
                 tableLambdaFiles=None,
                 verbose=Config.verbose):
        self._name = name
        self._acronym = acronym
        self._tableDir = tableDir
        self._tableLambdas = tableLambdas
        self._tableLambdaFiles = tableLambdaFiles
        self._verbose = verbose

        if self._tableDir is None:
            warnings.warn("""No table directory has been set! If tables for this model
            have been generated previously please set tableDir to the
            location of these tables. If no tables have been generated
            and the intention is to run fitting routines please see
            model.buildTables to get started.""", UserWarning)
        else:
            self._readTables(verbose=verbose)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @name.deleter
    def name(self):
        self._name = None

    @property
    def acronym(self):
        return self._acronym

    @acronym.setter
    def acronym(self, value):
        self._acronym = value

    @acronym.deleter
    def acronym(self):
        self._acronym = None

    @property
    def tableDir(self):
        return self._tableDir

    @tableDir.setter
    def tableDir(self, value):
        self._tableDir = value

    @property
    def tableLambdas(self):
        return self._tableLambdas

    @tableLambdas.setter
    def tableLambdas(self, value):
        self._tableLambdas = value

    @property
    def tableLambdaFiles(self):
        return self._tableLambdaFiles

    @tableLambdaFiles.setter
    def tableLambdaFiles(self, value):
        self._tableLambdaFiles = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def _readTables(self, verbose=Config.verbose):
        tableFiles = glob.glob(self._tableDir + "{}_*.npz".format(self._acronym))
        if verbose is True:
            print("Found {} tables for this model.".format(len(tableFiles)))

        if tableFiles is None:
            print("No tables found.")

        else:

            if self._tableLambdas is None:
                self._tableLambdas = []
                self._tableLambdaFiles = {}

            if verbose is True:
                print("")

            for tableFile in tableFiles:

                # Grab only the table file name and remove extenstion
                tableFile = os.path.basename(tableFile)
                tableFileName, tableFileExt = os.path.splitext(tableFile)
                if verbose is True:
                    print("Found table file: {}".format(tableFileName))
                # "acronym_lambda/nu_lambd/nu_t_tMin_tMax_tStep_a_alphaMin_alphaMax_alphaStep"
                # Split the table file name elements and print them out
                elements = tableFileName.split("_")

                if verbose is True:
                    print("Wavelength: {}".format(elements[2]))
                self._tableLambdas.append(float(elements[2]))
                self._tableLambdaFiles[float(elements[2])] = tableFile
                
                if verbose is True:
                    print("Subsolar Temperature:")
                    print("\tMin: {}".format(elements[4]))
                    print("\tMax: {}".format(elements[5]))
                    print("\tStep: {}".format(elements[6]))
                    print("Alpha:")
                    print("\tMin: {}".format(elements[8]))
                    print("\tMax: {}".format(elements[9]))
                    print("\tStep: {}".format(elements[10]))
                    print("")

            self._tableLambdas = np.array(self._tableLambdas)
            if verbose is True:
                print("Done.")
                print("")
        return

    def buildLambdaTables(self, lambds,
                          tRange=Config.tableParameterLimits["T_ss"][0],
                          tStep=Config.tableParameterLimits["T_ss"][1],
                          alphaRange=Config.tableParameterLimits["alpha"][0],
                          alphaStep=Config.tableParameterLimits["alpha"][1],
                          threads=Config.threads,
                          verbose=Config.verbose):
        """
        Build lookup tables for use later when fitting models. Depending
        on the complexity of the model and grid size, this may take a while!

        Parameters
        ----------
        lambds : `~numpy.ndarray`
            Wavelengths in m.
        tRange : list
            Minimum and maximum temperature (range) in K.
            [Default = `~atm.Config.tableParameterLimits["T_ss"][0]`]
        tStep : float, optional
            Temperature step size in K.
            [Default = `~atm.Config.tableParameterLimits["T_ss"][1]`]
        alphaRange : `~numpy.ndarray`, optional
            Minimum and maximum phase angle in radians.
            [Default = `~atm.Config.tableParameterLimits["alpha"][0]`]
        alphaStep : float, optional
            Phase angle step size in radians.
            [Default = `~atm.Config.tableParameterLimits["alpha"][1]`]
        threads : int, optional
            Number of processors to use.
            [Default = `atm.Config.threads`]
        verbose : bool, optional
            Print output statements?
            [Default = `~atm.Config.verbose`]

        Returns
        -------
        None
        """
        if verbose is True:
            print("Building tables for {} wavelength(s)...".format(len(lambds)))
            print("")

        for i, lambd in enumerate(lambds):
            lambd_str = "{:0=15.10f}e-06".format(lambd / 1e-6)
            lambd_float = float(lambd_str)
            if verbose is True:
                print("Building table {} out of {}".format(i + 1, len(lambds)))

            temps = np.arange(tRange[0], tRange[1] + tStep, tStep)
            alpha = np.arange(alphaRange[0], alphaRange[1] + alphaStep, alphaStep)
            t, a = np.meshgrid(temps, alpha)
            t = t.flatten(order="F")
            a = a.flatten(order="F")
            l = np.ones(len(t))*lambd_float

            outFileName = ("{}_lambda_{}_" 
                           + "t_{:.4f}_{:.4f}_{:.4f}_"
                           + "a_{:.4f}_{:.4f}_{:.4f}").format(self._acronym,
                                                              lambd_str,
                                                              tRange[0],
                                                              tRange[1],
                                                              tStep,
                                                              alphaRange[0],
                                                              alphaRange[1],
                                                              alphaStep)
            outFile = os.path.join(self._tableDir, outFileName)
            
            # If file does not exist build it 
            if not os.path.exists(outFile + ".npz"):
                if verbose is True:
                    print("Building table file: {}".format(outFileName))
                    print("Wavelength: {}".format(lambd_str))
                    print("Subsolar Temperature:")
                    print("\tMin: {:.4f}".format(tRange[0]))
                    print("\tMax: {:.4f}".format(tRange[1]))
                    print("\tStep: {:.4f}".format(tStep))
                    print("\tPoints: {}".format(len(temps)))
                    print("Alpha:")
                    print("\tMin: {:.4f}".format(alphaRange[0]))
                    print("\tMax: {:.4f}".format(alphaRange[1]))
                    print("\tStep: {:.4f}".format(alphaStep))
                    print("\tPoints: {}".format(len(alpha)))
                    print("Total number of integrations: {}".format(len(t)))
                    print("Starting integrations (this may take a while)...")

                flux = self.calcTotalFluxLambdaEmittedToObsMany(l, t, a, threads=threads)
                flux = flux.reshape([len(temps), len(alpha)])

                np.savez(outFile, T_ss=temps, alpha=alpha, fluxLambda=flux)
            
            # If the file does exist, move to the next wavelength and do nothing
            else:
                if verbose is True:
                    print("Table file {} exists. Moving on...".format(outFileName))

            if verbose is True:
                print("Done.")
                print("")
        return

    def loadLambdaTables(self, lambds, verbose=Config.verbose):
        """
        Loads previously built lookup tables and initializes
        `~scipy.interpolate.RectBivariateSpline` callable functions.

        This is required for interpTotalFluxLambdaEmittedToObs to work.

        Parameters
        ----------
        lambds : list or `~numpy.ndarray`
            Wavelengths in m to load. Tables should have been
            built beforehand.
        verbose : bool, optional
            Print output statements?
            [Default = `~atm.Config.verbose`]

        Returns
        -------
        None
        """
        self.loadedLambdaTables = {}
        if verbose is True:
            print("Loading tables for {} wavelength(s).".format(len(lambds)))
        for lambd in lambds:
            tableFile = os.path.join(self._tableDir, self.tableLambdaFiles[lambd])
            tableFileName, tableFileExt = os.path.splitext(tableFile)
            table = np.load(tableFile)
            T_ss = table["T_ss"]
            alpha = table["alpha"]
            flux = table["fluxLambda"]
            self.loadedLambdaTables[lambd] = interpolate.RectBivariateSpline(T_ss, alpha, flux)
            if verbose is True:
                print("Loaded table file: {}".format(os.path.basename(tableFileName)))
                print("Wavelength: {}".format(lambd))
        if verbose is True:
            print("Done.")
            print("")
        return

    def calcT(self):
        """
        This method should return the characteristic Planck temperature
        as a function of the geometry of an asteroid and the subolar
        temperature (model dependent).

        Throws NotImplementedError when not defined.
        """
        raise NotImplementedError("calcT is not defined.")

    def calcFluxLambdaEmitted(self):
        """
        This method should calculate the flux emitted at a specific point
        on the surface of an asteroid.

        Throws NotImplementedError when not defined.
        """
        raise NotImplementedError("calcFluxLambdaEmitted is not defined.")

    def calcTotalFluxLambdaEmitted(self):
        """
        This method should integrate the calcFluxLambdaEmitted method over
        the entire surface of an asteroid.

        Throws NotImplementedError when not defined.
        """
        raise NotImplementedError("calcTotalFluxLambdaEmitted is not defined.")

    def calcTotalFluxLambdaEmittedMany(self, lambd, T_ss,
                                       threads=Config.threads):
        """
        Multiprocessed version of calcTotalFluxLambdaEmitted.

        calcTotalFluxLambdaEmitted cannot handle arrays unless looped through
        element by element (which is painfully slow), it can also not
        be easily vectorized due to integration.

        Calculates the total flux emitted at the surface of an asteroid
        of size unity as a function of wavelength and the subsolar temperature.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        lambd : float or `~numpy.ndarray`
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.
        verbose : bool, optional
            Print output statements?
            [Default = `~atm.Config.verbose`]

        Returns
        -------
        `~numpy.ndarray`
            Total flux emitted in units of W m^-3.
        """
        args = np.array([lambd, T_ss]).T
        pool = Pool(threads)
        flux = pool.starmap(self.calcTotalFluxLambdaEmitted, args)
        pool.close()
        return np.array(flux)

    def calcFluxLambdaEmittedToObs(self):
        """
        This method should calculated the emitted flux at a specific point
        on the surface of an asteroid directed towards
        an observer at phase angle alpha.

        Throws NotImplementedError when not defined.
        """
        raise NotImplementedError("calcFluxLambdaEmittedToObs is not defined.")

    def calcTotalFluxLambdaEmittedToObs(self):
        """
        This method should integrate calcFluxLambdaEmittedObs over the hemisphere
        of an asteroid facing the observer at phase angle alpha.

        Throws NotImplementedError when not defined.
        """
        raise NotImplementedError("calcTotalFluxLambdaEmittedToObs is not defined.")

    def calcTotalFluxLambdaEmittedToObsMany(self, lambd, T_ss, alpha,
                                            threads=Config.threads):
        """
        Multiprocessed version of calcTotalFluxLambdaEmittedToObs.

        calcTotalFluxLambdaEmittedObs cannot handle arrays unless looped through
        element by element (which is painfully slow), it can also not be
        easily vectorized due to integration.

        Calculates the total flux emitted at the surface of an asteroid
        of size unity towards an observer at phase angle alpha
        as a function of wavelength, subsolar temperature and phase angle.

        Returns flux in units of W m^-3.

        Parameters
        ----------
        lambd : float or `~numpy.ndarray`
            Wavelength in meters.
        T_ss : `~numpy.ndarray`
            Subsolar temperature in K.
        alpha : `~numpy.ndarray`
            Phase angle in radians.
        threads : int, optional
            Number of processors to use.
            [Default = `atm.Config.threads`]

        Returns
        -------
        `~numpy.ndarray`
            Total flux emitted in units of W m^-3 to an observer
            located in the direction of alpha.
        """
        # If lambda is a float, convert it to an array of floats
        # of equal length to T_ss
        # This is to make obs.bandpassLambda use the multi-threaded calcFlux
        if type(lambd) == float:
            lambd = np.ones(len(T_ss)) * lambd
        args = np.array([lambd, T_ss, alpha]).T
        pool = Pool(threads)
        flux = pool.starmap(self.calcTotalFluxLambdaEmittedToObs, args)
        pool.close()
        return np.array(flux)

    def interpTotalFluxLambdaEmittedToObs(self, lambd, T_ss, alpha):
        """
        Interpolated version of calcTotalFluxLambdaEmittedToObs /
        calcTotalFluxLambdaEmittedToObsMany.

        Uses loaded tables to interpolate the total flux emitted at the surface
        of an asteroid of size unity towards an observer at phase angle alpha
        as a function of wavelength, subsolar temperature and phase angle.

        Requires the following:
        - buildLambdaTables should have been run at the desired wavelength
        - loadLambdaTables should have been run prior to calling this function

        Parameters
        ----------
        lambd : float
            Wavelength in m.
        T_ss : float or `~numpy.ndarray`
            Subsolar temperature in K.
        alpha : float or `~numpy.ndarray`
            Phase angle in radians.

        Returns
        -------
        float or `~numpy.ndarray`
            Total flux emitted in units of W m^-3 to an observer
            located in the direction of alpha.
        """
        return self.loadedLambdaTables[lambd](T_ss, alpha, grid=False)
