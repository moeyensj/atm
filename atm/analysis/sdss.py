import io
import os
import gzip
import urllib
import pandas as pd

__all__ = [
    "getSDSSMOC",
    "readSDSSMOC"
]


def getSDSSMOC():
    """
    Download the SDSS MOC v4 if it is not already
    included with this repository.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    moc_file = os.path.join(os.path.dirname(__file__), "data/ADR4.dat")
    if os.path.isfile(moc_file) is not True:
        print("Could not find SDSS MOC. Attempting download...")
        url = 'http://faculty.washington.edu/ivezic/sdssmoc/ADR4.dat.gz'  
        response = urllib.request.urlopen(url)  
        print("File succesfully downloaded.")
        print("Decompressing...")
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)

        with open(moc_file, 'wb') as outfile:
            outfile.write(decompressed_file.read())
        print("Done.")
    else:
        print("SDSS MOC v4 has already been acquired.")
    
    print("")
    return

def readSDSSMOC(mocFile=os.path.join(os.path.dirname(__file__), "data/ADR4.dat"), download=True):
    """
    Read the SDSS MOC from a file. 

    Parameters
    ----------
    mocFile : str, optional
        Path to SDSS MOC.
        [Default = "data/ADR4.dat"] 
    download : bool, optional
        If the SDSS MOC file is not found, download it. 
        [Default = True]

    Returns
    -------
    sdssmoc : `~pandas.DataFrame`
        The MOC in a DataFrame.
    """
    if os.path.isfile(mocFile) is not True:
        if download is True:
            getSDSSMOC()
        
    columns = [
        "mo_id",
        "run",
        "col",
        "field",
        "object",
        "rowc",
        "colc",
        "mjd",
        "RA_deg",
        "Dec_deg",
        "Lambda_deg",
        "Beta_deg",
        "Phi_deg",
        "vMu_deg_p_day",
        "vMuErr_deg_p_day",
        "vNu_deg_p_day",
        "vNuErr_deg_p_day",
        "vLambda_deg_p_day",
        "vBeta_deg_p_day",
        "u_mag",
        "u_magErr",
        "g_mag",
        "g_magErr",
        "r_mag",
        "r_magErr",
        "i_mag",
        "i_magErr",
        "z_mag",
        "z_magErr",
        "a",
        "aErr",
        "V_mag",
        "B_mag",
        "identification_flag",
        "numeration",
        "designation",
        "detection_counter",
        "total_detection_count",
        "flags",
        "computed_RA_deg",
        "computed_Dec_deg",
        "computed_mag",
        "r_au",
        "delta_au",
        "alpha_deg",
        "catalog_id",
        "H_mag",
        "G",
        "arc",
        "epoch",
        "a_au",
        "e",
        "i_deg",
        "ascNode_deg",
        "argPeri_deg",
        "meanAnom_deg",
        "proper_elements_catalog",
        "a'",
        "e'",
        "sin(i')",
        "binary_processing_flags",

    ]
        
    colspec = [
        (0, 6),
        (7, 12),
        (13, 14),
        (15, 19),
        (20, 25),
        (26, 34),
        (34, 43),
        (46, 58),
        (59, 69),
        (70, 80),
        (81, 91),
        (92, 102),
        (103, 114),
        (115, 123),
        (124, 130),
        (131, 138),
        (139, 145),
        (146, 153),
        (154, 161),
        (163, 168),
        (169, 173),
        (174, 179),
        (180, 184),
        (185, 190),
        (191, 195),
        (196, 201),
        (202, 206),
        (207, 212),
        (213, 217),
        (218, 223),
        (224, 228),
        (230, 235),
        (236, 241),
        (242, 243),
        (244, 251),
        (252, 272),
        (273, 275),
        (276, 278),
        (279, 287),
        (289, 299),
        (300, 310),
        (311, 316),
        (318, 325),
        (326, 333),
        (334, 339),
        (341, 351),
        (362, 367),
        (368, 372),
        (373, 378),
        (379, 392),
        (393, 405),
        (406, 416),
        (417, 427),
        (428, 438),
        (439, 449),
        (483, 395),
        (496, 506),
        (507, 517),
        (518, 645)
    ]
    
    sdssmoc = pd.read_fwf(mocFile, names=columns, colspec=colspec, header=None, index_col=False)
    return sdssmoc
    