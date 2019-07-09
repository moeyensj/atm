import warnings
import numpy as np
import pandas as pd

from .config import Config

__all__ = ["__handleParameters"]

def __handleParameters(obs,
                       fitParameters, 
                       dataParameters, 
                       emissivitySpecification="perBand", 
                       albedoSpecification="auto", 
                       columnMapping=Config.columnMapping):
    """
    Helper function that builds lists of albedo and emissivity parameters and warns the
    user if they are potentially violating Kirchoff's law.
    """
    # Required parameter set (keys in columnMapping)
    requiredParametersSet = set([
            "r_au", 
            "delta_au", 
            "alpha_rad",
            "G", 
            "logT1", 
            "logD"])

    # Add emissivity and albedo parameters to required key set
    emissivityParameters = []
    albedoParameters = []

    if emissivitySpecification != "auto":
        if albedoSpecification != "auto":
            warnings.warn("""
                Potentially violating Kirchoff's law!
                Insure your albedo specification is compatible with your emissivity specification.
                You can also define one and set the other to 'auto'.""", RuntimeWarning)

        if emissivitySpecification == None:
            emissivityParameters = "eps"
            requiredParametersSet.add("eps")
            if albedoSpecification == "auto":
                requiredParametersSet.add("p")
                albedoParameters = "p"
        elif emissivitySpecification == "perBand":
            for f in obs.filterNames:
                eps_f = "eps_{}".format(f)
                requiredParametersSet.add(eps_f)
                emissivityParameters.append(eps_f)
                if albedoSpecification == "auto":
                    p_f = "p_{}".format(f)
                    requiredParametersSet.add(p_f)
                    albedoParameters.append(p_f)
        elif type(emissivitySpecification) == dict:
            filterSet = set(obs.filterNames)
            if albedoSpecification != "auto":
                warnings.warn("Potentially violating Kirchoff's law!", RuntimeWarning)
            for k, v in emissivitySpecification.items():
                for f in v:
                    try:
                        filterSet.remove(f)
                        requiredParametersSet.add(k)
                        emissivityParameters.append(k)

                        if albedoSpecification == "auto":
                            p_f = "p_{}".format("".join(k.split("_")[1:]))
                            requiredParametersSet.add(p_f)
                            albedoParameters.append(p_f)
                    except:
                        raise ValueError("Cannot bind one filter to more than 1 epsilon.")


            if len(filterSet) != 0:
                raise ValueError("Not all filters are bound to an emissivity parameter. " +
                                 "Please define emissivity specification correctly.")
        else:
            raise ValueError("emissivitySpecification should be one of {None, 'perBand', dict}")

    if albedoSpecification != "auto":
        if emissivitySpecification != "auto":
            warnings.warn("""
                Potentially violating Kirchoff's law!
                Insure your albedo specification is compatible with your emissivity specification.
                You can also define one and set the other to 'auto'.""", RuntimeWarning)

        if albedoSpecification == None:
            albedoParameters = "p"
            requiredParametersSet.add("p")
            if emissivitySpecification == "auto":
                requiredParametersSet.add("eps")
                emissivityParameters = "eps"
        elif albedoSpecification == "perBand":
            for f in obs.filterNames:
                p_f = "p_{}".format(f)
                requiredParametersSet.add(p_f)
                albedoParameters.append(p_f)
                if emissivitySpecification == "auto":
                    eps_f = "eps_{}".format(f)
                    requiredParametersSet.add(eps_f)
                    emissivityParameters.append(eps_f)
        elif type(albedoSpecification) == dict:
            if emissivitySpecification != "auto":
                warnings.warn("Potentially violating Kirchoff's law!", RuntimeWarning)
            filterSet = set(obs.filterNames)
            for k, v in albedoSpecification.items():
                for f in v:
                    try:
                        filterSet.remove(f)
                        requiredParametersSet.add(k)
                        albedoParameters.append(k)

                        if emissivitySpecification == "auto":
                            eps_f = "eps_{}".format("".join(k.split("_")[1:]))
                            requiredParametersSet.add(eps_f)
                            emissivityParameters.append(eps_f)
                    except:
                        raise ValueError("Cannot bind one filter to more than 1 albedo.")


            if len(filterSet) != 0:
                raise ValueError("Not all filters are bound to an albedo parameter. " +
                                 "Please define albedo specification correctly.")
        else:
            raise ValueError("albedoSpecification should be one of {'auto', None, 'perBand', dict}")

    if emissivitySpecification == "auto" and albedoSpecification == "auto":
        raise ValueError("Both emissivitySpecification and albedoSpecification cannot be 'auto'.")

    # Set of fit parameters
    fitParametersSet = set(fitParameters)

    # Set of parameters that can not be fit for:
    # fluxes, flux errors and the observation ids
    unfittableParametersSet = set([
        "flux_si",
        "fluxErr_si",
        "obs_id",
        "exp_mjd",
        "designation"
    ])
    
    
    requiredDataParametersSet = requiredParametersSet - fitParametersSet 
    # If albedo will be automatically calculated, remove it from the 
    # required set of parameters that should be in data
    if albedoSpecification == "auto":
        if albedoParameters == "p":
            requiredDataParametersSet.remove(albedoParameters)
        if type(albedoParameters) == list:
            for albedo in albedoParameters:
                if albedo in requiredDataParametersSet:
                    requiredDataParametersSet.remove(albedo)

    # If emissivity will be automatically calculated, remove it from the 
    # required set of parameters that should be in data
    if emissivitySpecification == "auto":
        if emissivityParameters == "p":
            requiredDataParametersSet.remove(emissivityParameters)
        if type(emissivityParameters) == list:
            for emissivity in albedoParameters:
                if emissivity in requiredDataParametersSet:
                    requiredDataParametersSet.remove(emissivity)

    # Set of all parameters in data
    dataParametersSet = set(dataParameters)
    
    # Lets run through the required parameters and make sure
    # we have what we need 
    dataParametersToIgnoreSet = set()
    for parameter in requiredParametersSet:
        if ((parameter == "p") and (albedoParameters == "p")):
            # Is it a single albedo parameter? If so, lets make sure
            # that it exists in data, or is being fitted for, or can be calculated
            # using an appropriate emissivity
            # If it is a fit parameter and it exists in data, add it to the set of 
            # parameters to ignore
            if ((parameter in fitParametersSet) 
                and (albedoSpecification != "auto")
                and (columnMapping[parameter] in dataParametersSet)):
                dataParametersToIgnoreSet.add(parameter)

            if ((parameter not in fitParametersSet) 
                and (albedoSpecification != "auto")
                and (columnMapping[parameter] not in dataParametersSet)
                and (parameter not in dataParametersSet)):
                raise ValueError("""
                Albedo ({}) is not a fit parameter and is not contained within
                data. Please add an assumed value for albedo to data or set
                albedoSpecification to 'auto' if emissivities have been specified.
                """.format(parameter))

        elif ((type(albedoParameters) == list) and (parameter in set(albedoParameters))):
            if ((parameter in fitParametersSet) and (parameter in dataParametersSet)):
                    dataParametersToIgnoreSet.add(parameter)

            if ((parameter in set(albedoParameters)) 
                and (parameter not in fitParametersSet) 
                and (albedoSpecification != "auto")
                and (parameter not in dataParametersSet)):
                raise ValueError("""
                    Albedo ({}) is not a fit parameter and is not contained within
                    data. Please add an assumed value for emissivity to data or set
                    albedoSpecification to 'auto' if emissivities have been specified.
                    """.format(parameter))

        elif ((parameter == "eps") and (emissivityParameters == "eps")):
            # Is it a single emissivity parameter? If so, lets make sure
            # that it exists in data, or is being fitted for, or can be calculated
            # using an appropriate albedo
            # If it is a fit parameter and it exists in data, add it to the set of 
            # parameters to ignore
            if ((parameter in fitParametersSet) 
                and (emissivitySpecification != "auto")
                and (columnMapping[parameter] in dataParametersSet)):
                dataParametersToIgnoreSet.add(parameter)

            if ((parameter not in fitParametersSet) 
                and (emissivitySpecification != "auto")
                and (columnMapping[parameter] not in dataParametersSet)
                and (parameter not in dataParametersSet)):
                raise ValueError("""
                Emissivity ({}) is not a fit parameter and is not contained within
                data. Please add an assumed value for emissivity to data or set
                emissivtySpecification to 'auto' if albedos have been specified.
                """.format(parameter))          
        elif ((type(emissivityParameters) == list) and (parameter in set(emissivityParameters))):
            if ((parameter in fitParametersSet) and (parameter in dataParametersSet)):
                    dataParametersToIgnoreSet.add(parameter)

            if ((parameter in set(emissivityParameters)) 
                and (parameter not in fitParametersSet) 
                and (emissivitySpecification != "auto")
                and (parameter not in dataParametersSet)):
                raise ValueError("""
                    Emissivity ({}) is not a fit parameter and is not contained within
                    data. Please add an assumed value for emissivity to data or set
                    emissivitySpecification to 'auto' if albedos have been specified.
                    """.format(parameter))
        else:
            if ((parameter in fitParametersSet) and (columnMapping[parameter] in dataParametersSet)):
                dataParametersToIgnoreSet.add(parameter)

            if (columnMapping[parameter] not in dataParametersSet) and (parameter not in fitParametersSet):
                raise ValueError("""
                        Parameter ({}) is not a fit parameter and is not contained within
                        data.
                        """.format(parameter))
                
    return fitParametersSet, requiredDataParametersSet, emissivityParameters, albedoParameters, dataParametersToIgnoreSet