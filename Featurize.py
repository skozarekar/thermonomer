'''
    this python file contains functions for obtaining feature data for monomers

    Available features:
        1. pre-estimation property (PEP) - dH only
        2. Tanimoto M/S similarity
        3. Dipole moment
        4. Solvent parameters
        5. Solute parameters
        6. RDKit descriptors
        STERICS
        7. Steric descriptors
        8. Wiener Index 
        9. Find chiral centers
        10. Radius of Gyration
        11. Sphereocity
        12. Stereogenic (=stereocenter) atoms and bonds
'''

import math
import os
import requests
import warnings
import csv
from urllib import parse

import numpy as np
import pandas as pd

from openbabel import pybel
from openbabel._openbabel import OBChargeModel_FindType

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles, GetDistanceMatrix, FindMolChiralCenters
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdmolops import GetShortestPath, FindPotentialStereo
from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration, CalcSpherocityIndex, DoubleCubicLatticeVolume, CalcNumBridgeheadAtoms
from rdkit.Chem import AllChem


import periodictable

# imports from github/other
import pgthermo.properties as prop
from Polymerize import Polymerization

# ---------- VARIABLE DEFINITIONS ---------- #

# Define repeat_rxn_dict which is used in steric feature 
repeat_vinyl = rdChemReactions.ReactionFromSmarts(
    "[C:1]=[C:2].[*:3]-[*:4]>>[*:3]-[C:1]-[C:2]-[*:4]"
)

repeat_romp = rdChemReactions.ReactionFromSmarts(
    "[C:1]=[C:2].([*:3]=[*:4])>>([C:1]=[*:3].[*:4]=[C:2])"
)

repeat_rop = rdChemReactions.ReactionFromSmarts(
    "([C,S:1](=[O,S:2])@[O,S,N:3]@[C:4]).[*:5]-[*:6]>>([C,S:1](=[O,S:2])[*:5].[*:6]-[O,S,N:3][C:4])"
)

repeat_ionic = rdChemReactions.ReactionFromSmarts(
    "[C:1]=[O,S:2].[*:3]-[*:4]>>[*:3]-[C:1]-[O,S:2]-[*:4]"
)

repeat_cyclic = rdChemReactions.ReactionFromSmarts(
    "([CH2:1]-[CH2:2]).[*:3]-[*:4]>>([C:2]-[*:3].[C:1]-[*:4])"
)

repeat_cationic = rdChemReactions.ReactionFromSmarts(
    "([*:1]@[*:2]).[*:3]-[*:4]>>([*:2][*:3].[*:1][*:4])"
)

repeat_rxn_dict = {
    "ROP": repeat_rop,
    "ROMP": repeat_romp,
    "cationic": repeat_cationic,
    "vinyl": repeat_vinyl,
    "ionic": repeat_ionic,
    "cyclic": repeat_cyclic,
}

def getRepeatUnit(monomer_smiles, polymerization_type, relative_path= "data archive/featurized_archive.csv"):
    '''
        A function that gets the monomer repeat unit 

        Parameters:
            monomer_smiles (str): smiles string of monomer
            polymerization_type (str): the type of polymerization the monomer undergoes

        Output:
            (str): the repeat unit smiles string
    '''

    # First, try to match in archive
    monomer = MolFromSmiles(monomer_smiles)

    try:
        # Turn the csv into a pandas array
        archive_df = pd.read_csv(relative_path)

        idx_match = archive_df.loc[
            archive_df["Canonical SMILES"] == monomer_smiles
        ].index[0]
        repeat_unit = archive_df.iloc[idx_match]["REPEAT_UNIT"]
        return repeat_unit
    except:
        print(f"{monomer_smiles} does not have a match in archive. Using rxn to calculate repeat unit.")
        # Then use operator
        try:
            rxn = repeat_rxn_dict[polymerization_type]

            if polymerization_type == "ROMP":
                temp = MolFromSmiles("*=*")
            else:
                temp = MolFromSmiles("**")

            prod = rxn.RunReactants((monomer, temp))[0][0]
            repeat_unit = MolToSmiles(prod)
            return repeat_unit
        except:
            print("No repeat unit for", monomer_smiles)
            return np.nan

# ---------- HELPER FUNCTIONS ---------- #

def dipoleHelper(method, smiles):
    '''
        a function that calculates the dipole moment of the monomer 

        Parameters:
            method (str): the method to use for calculating dipole
            smiles (str): the canonical smiles string of the molecule 

        Output:
            (float): dipole moment
    '''
    if "Si" in smiles and method != "mmff94":
        # molecules with Si do not work with methods other than mmff94
        return np.nan
    else:
        os.system('obabel -:"' + str(smiles) + '" -i smiles -O temp.sdf --gen3d')

        for mol in pybel.readfile("sdf", "temp.sdf"):
            # Calculate dipole with given method
            cm = OBChargeModel_FindType(method)
            cm.ComputeCharges(mol.OBMol)
            dipole = cm.GetDipoleMoment(mol.OBMol)

            # Return scalar
            calc= math.sqrt(dipole.GetX() ** 2 + dipole.GetY() ** 2 + dipole.GetZ() ** 2)

            if calc == 0:
                return np.nan
            
            else:
                return calc
        
def chempropSolvationHelper(monomer_smiles, monomer_base_state, solvent_smiles):
    '''
        a function that retrieves solvent data from a website

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
            monomer_base_state (str): the state (liquid, solution or gas) that the monomer is in 
            solvent_smiles (str): the canonical smiles string of the molecule 

        Output:
            (dictionary): collection of solvent features obtained from rmg.mit. keys have prefix MIX_
    '''
    empty_dict = {
        "MIX_dGsolv298(kJ/mol)": 0,
        "MIX_dHsolv298(kJ/mol)": 0,
        "MIX_dSsolv298(kJ/mol/K)": 0,
        "MIX_logK": 0,
        "MIX_logP": 0,
        "MIX_dGsolv298(kJ/mol) epi.unc.": 0,
        "MIX_dHsolv298(kJ/mol) epi.unc.": 0,
    }
    # set variables
    base_url = "https://rmg.mit.edu/database/solvation/searchML/solv_solu="
    url_flags = "__dG=True__dH=True__dS=True__logK=True__logP=True__unit=kJ/mol"

    # if we have solution or liquid as the monomer state, set the URL
    if monomer_base_state == "s":
        if pd.isnull(solvent_smiles):
            return empty_dict

        url = base_url + str(monomer_smiles) + "_" + str(solvent_smiles) + url_flags
    elif monomer_base_state == "l":
        url = base_url + str(monomer_smiles) + "_" + str(monomer_smiles) + url_flags
    # otherwise, return empty dataframe
    else:
        return empty_dict

    # make request
    url = url.replace(" ", "")
    encoded_url = parse.quote(url, safe=":/")
    response = requests.get(encoded_url)
    if response.status_code != 200:  # catch errors
        return empty_dict
    table = pd.read_html(response.content)
    df = table[0]

    # store the values from chemprop
    solvent_data = {
        "MIX_dGsolv298(kJ/mol)": df["dGsolv298(kJ/mol)"],
        "MIX_dHsolv298(kJ/mol)": df["dHsolv298(kJ/mol)"],
        "MIX_dSsolv298(kJ/mol/K)": df["dSsolv298(kJ/mol/K)"],
        "MIX_logK": df["logK"],
        "MIX_logP": df["logP"],
        "MIX_dGsolv298(kJ/mol) epi.unc.": df["dGsolv298 epi.unc.(kJ/mol)"],
        "MIX_dHsolv298(kJ/mol) epi.unc.": df["dHsolv298 epi.unc.(kJ/mol)"],
    }

    return solvent_data

def soluteHelper(monomer_smiles):
    '''
        a function that retrieves solute features from rmg.mit

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer

        Output:
            (dict): collection of data on the solute. keys have the prefix SOLUTE_
    '''

    # base url to website that does the calculations
    base_url = "https://rmg.mit.edu/database/solvation/soluteSearch/solu="
    url_flags_GC = "__estimator=SoluteGC__solv=None__unit=kJ/mol"
    url_flags_ML = "__estimator=SoluteML__solv=None__unit=kJ/mol"

    if "[Si]" in monomer_smiles or "[Se]" in monomer_smiles:
        # website software can not handle molecules with [Si] or [Se]
        return pd.Series([np.nan] * 12)

    # make request to GC
    url = base_url + monomer_smiles + url_flags_GC
    url = url.replace(" ", "")
    encoded_url = parse.quote(url, safe=":/")
    response = requests.get(encoded_url)
    if response.status_code != 200:  # catch errors
        return pd.Series([np.nan] * 7)

    table = pd.read_html(response.content)
    df_gc = table[0]

    # make request to ML
    url = base_url + str(monomer_smiles) + url_flags_ML
    url = url.replace(" ", "")
    encoded_url = parse.quote(url, safe=":/")
    response = requests.get(encoded_url)
    table = pd.read_html(response.content)
    df_ml = table[0]

    solute_data = {
        "SOLUTE_PARAM_E_GC": df_gc["E"],
        "SOLUTE_PARAM_S_GC": df_gc["S"],
        "SOLUTE_PARAM_A_GC": df_gc["A"],
        "SOLUTE_PARAM_B_GC": df_gc["B"],
        "SOLUTE_PARAM_L_GC": df_gc["L"],
        "SOLUTE_PARAM_V_GC": df_gc["V"],
        "SOLUTE_PARAM_E_ML": df_ml["E"],
        "SOLUTE_PARAM_S_ML": df_ml["S"],
        "SOLUTE_PARAM_A_ML": df_ml["A"],
        "SOLUTE_PARAM_B_ML": df_ml["B"],
        "SOLUTE_PARAM_L_ML": df_ml["L"],
        "SOLUTE_PARAM_V_ML": df_ml["V"],
    }

    # return the values from chemprop
    return solute_data

def stericsHelper(repeat_unit_smiles):
    """
        Takes in a repeat unit SMILES and determines steric quantities 

        Parameters:
            repeat_unit_smiles (str): string containing SMILES of RU. Must have two '*' atoms representing connection points

        Returns:
            backbone length, ratio of side chain-containing atoms
    """

    mol = MolFromSmiles(repeat_unit_smiles)

    # list that holds wildcard atoms (atoms with atomic number of 0). A wildcard can represent any type of atom
    wildcard_idxs = []

    # get all the atoms that are in the monomer
    atom_list = mol.GetAtoms()
    
    for atom in atom_list:
        if atom.GetAtomicNum() == 0:
            wildcard_idxs.append(atom.GetIdx())

    try:
        assert len(wildcard_idxs) == 2

        backbone = GetShortestPath(mol, wildcard_idxs[0], wildcard_idxs[1])

        backbone_len = len(backbone)
        ratio = round((len(atom_list) - len(backbone)) / len(atom_list), 2)

        return backbone_len, ratio
    except:
        print("Error: too many *s in RU", repeat_unit_smiles)
        return 0

def getConformers(mol):
    """
        A function to generate the conformeres for a molecule to work in 3d space

        Parameters:
            mol (mol): from rdkit, a molecule

        Returns:
            m3d
    """

    # generate conformers for molecule to work in 3d space
    m3d=Chem.AddHs(mol)
    AllChem.EmbedMolecule(m3d)
    AllChem.MMFFOptimizeMolecule(m3d)
    return m3d

# ---------- FEATURE FUNCTIONS ---------- #
def getAllFeatures(monomer_smiles, monomer_base_state, polymerization_type, dp, solvent_name, target):
    '''
        A function that gets all the features for a monomer

        Parameters:
            monomer_smiles (str): canonical smiles string of monomer
            monomer_base_state (str): the state (liquid, solution or gas) that the monomer is in 
            polymerization_type (str): the type of polymerization that the monomer undergoes
            dp (int): degree of polymerization
            solvent_name (str): the name of the solvent
            target(str): either "dH" for enthalpy or "dS" for entropy. only difference in output is inclusion of PGTHERMO_PEP feature

        Output:
            (dict): a dictionary containing all of the features and input parameters
    '''

    output = {}
    # store original data into the dictionary
    output["Canonical_Monomer_SMILES"] = monomer_smiles
    output["Solvent"] = solvent_name
    output["BASE_Monomer_State"] = monomer_base_state
    output["BASE_polymerization_type"] = polymerization_type

    solvent_smiles = np.nan

    if not pd.isna(solvent_name):
        output.update(getExperimentalSolvent(solvent_name))
        solvent_smiles = output["Solvent_SMILES"]

    # include degrees of polymerization
    output.update(Polymerization(monomer_smiles, polymerization_type,dp).main())

    if target == "dH":
        # this is a parameter that is special to enthalpy
        output["PGTHERMO_PEP_ (kcal/mol)"] = getPEP(monomer_smiles, polymerization_type, dp)

    output["Tanimoto_Similarity"] = getTanimotoSimilarity(monomer_smiles,monomer_base_state,solvent_smiles)
    
    # for functions that return a dictionary, rather than making a new (key, value) pair, update the current dictionary instead
    output.update(getDipoles(monomer_smiles))   
    output.update(getSolventFeatures(monomer_smiles, monomer_base_state, solvent_smiles))
    output.update(getSoluteFeatures(monomer_smiles))
    output.update(getRdkitDescriptors(monomer_smiles))
    output.update(getSterics(monomer_smiles, polymerization_type))

    output["STERIC_wienerIndex"] = getWienerIndex(monomer_smiles, polymerization_type)
    output["STERIC_chiralCenters"] = getNumChiralCenters(monomer_smiles)
    output["STERIC_radiusGyration"] = getRadiusofGyration(monomer_smiles)
    output["STERIC_spherocity"] = getSpherocity(monomer_smiles)
    
    output.update(getVolume(monomer_smiles))

    output["STERIC_numBridgeheadAtoms"] = getNumBridgeheadAtoms(monomer_smiles)
    output["STERIC_numStereocenters"] = getNumStereocenters(monomer_smiles, polymerization_type)

    return output

def getPEP(monomer_smiles, polymerization_type,dp):
    '''
        A function that solves for and returns the pre-estimation property feature.
        The pre-estimation property is found using pgthermo and is only used for enthalpy features.

        Parameters:
            monomer_smiles (str): canonical smiles string of monomer
            polymerization_type (str): the type of polymerization the monomer undergoes
            dp (str): the degree of polymerization want to calculate it

        Output:
            (float): the PEP
    '''
    if dp < 3:
        raise ValueError("Degree of polymerization input must be greater than 2")
    # Calculate Hf
    hf_missing = []
    hf_list = []

    # add the polymers and solvents to data
    polymerize_data = Polymerization(monomer_smiles, polymerization_type,dp)
    polymer_smiles = polymerize_data.main()

    if polymer_smiles == np.nan or polymer_smiles == None:
        return np.nan
    
    else:

        # iterate through all the degrees of polymerization and get the corresponding heat of formation
        for molecule in polymer_smiles.values():
            enthalpy = prop.Hf(molecule, return_missing_groups=True)

            if type(enthalpy) is list: 
                # there are missing groups present so pgthermo can't compelte enthalpy calculation
                # enthalpy variable is a list of the groups that are missing
                hf_missing.append(list(set(enthalpy))) 
                # eliminate duplicates in the list
                hf_list.append(np.nan)

            else:  # type is float, hf successfully calculated
                hf_list.append(enthalpy)

        #calculate the estimation property PGTHERMO_dH_ (our PEP)
        pep = round(
            hf_list[2] - hf_list[1] - hf_list[0], 2
        )
        
        return pep

def getTanimotoSimilarity(monomer_smiles,monomer_base_state,solvent_smiles):
    '''
        A function that uses tanimoto to quantify similarity between monomer and solvent.

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
            monomer_base_state (str): the state (liquid, solution or gas) that the monomer is in 
            solvent_smiles (str): the canonical smiles string of the solvent

        Output:
            (float): similarity value
    '''

    # If liquid monomer, max similarity
    if monomer_base_state == "l":
        return 1

    # If solution-phase monomer, calculate similarity based on fingerprints
    elif monomer_base_state == "s":
        if pd.isna(solvent_smiles):
            # Error if solvent data missing
            print("\tss without solvent:", monomer_smiles)
            return -2

        # Calculate
        mol1 = MolFromSmiles(monomer_smiles)
        mol2 = MolFromSmiles(solvent_smiles)

        fp1 = GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
        fp2 = GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)

        score = round(TanimotoSimilarity(fp1, fp2), 3)

        return score

    # For gas monomer, similarity should be none
    else:
        return -1

def getDipoles(monomer_smiles):
    '''
        A function that solves for and saves the dipole moment of the monomer

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer

        Output:
            (dictionary): the keys are the methods to calculate the dipole and the values are the dipole values
    '''
    # methods used to calculate dipole: gasteiger, mmff94, eem2015bm
    dipoles = {}

    dipoles["DIPOLE_gasteiger"] = (dipoleHelper("gasteiger", monomer_smiles))
    dipoles["DIPOLE_mmff94"] = (dipoleHelper("mmff94", monomer_smiles))
    dipoles["DIPOLE_eem2015bm"] = (dipoleHelper("eem2015bm", monomer_smiles))

    return dipoles

def getSolventFeatures(monomer_smiles, monomer_base_state, solvent_smiles = "NA", archive_path = "data archive/featurized_archive.csv"):
    '''
        A function that returns the solvent features for a monomer

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
            monomer_base_state (str): the state (liquid, solution or gas) that the monomer is in 
            archive_path (str): the path to the featurized_archive.csv file
            solvent_smiles (str)*: the canonical smiles string of the solvent

            *solvent_smiles is optional, default value being NA. Only pass in solvent smiles if the monomer base state is in solution

        Output:
            (dictionary): a dictionary containg the solvent data with the keys having prefix MIX_
    '''
    solvent_data = {
        "MIX_dGsolv298(kJ/mol)": 0,
        "MIX_dHsolv298(kJ/mol)": 0,
        "MIX_dSsolv298(kJ/mol/K)": 0,
        "MIX_logK": 0,
        "MIX_logP": 0,
        "MIX_dGsolv298(kJ/mol) epi.unc.": 0,
        "MIX_dHsolv298(kJ/mol) epi.unc.": 0,
    }

    # Turn the csv into a pandas array
    archive_df = pd.read_csv(archive_path)

    if solvent_smiles == np.nan:
        # there is no solvent
        solvent_smiles = "NA"

    solvent_pair_lookup = np.nan

    #solvent pair lookup is dependent on the base state of the monomer
    if monomer_base_state == 'l':
        solvent_pair_lookup = monomer_smiles+ '_' + monomer_smiles
    elif monomer_base_state == 's' and solvent_smiles != "NA":
        solvent_pair_lookup = monomer_smiles + '_' + solvent_smiles
    elif monomer_base_state == 's' and solvent_smiles != "NA":
        raise ValueError("Solvent smiles must be provided for monomers with base state 's'")
    else:
        print(f"No solvent data for monomers with base state '{monomer_base_state}'")
        return solvent_data
    
    match_columns = ["MIX_dGsolv298(kJ/mol)", "MIX_dHsolv298(kJ/mol)", "MIX_dSsolv298(kJ/mol/K)", "MIX_logK", "MIX_logP", "MIX_dGsolv298(kJ/mol) epi.unc.", "MIX_dHsolv298(kJ/mol) epi.unc."]

    if solvent_pair_lookup in archive_df["Solvent_pair_lookup"].values:
        # Loop through all columns and see if there is a matching value 
        for col_name in match_columns:
            corresponding_i = archive_df.loc[
                archive_df["Solvent_pair_lookup"] == solvent_pair_lookup, col_name
            ].values

            solvent_data[col_name] = corresponding_i[0]

    all_nans = all(np.isnan(key) if isinstance(key, float) else False for key in solvent_data.keys())

    if all_nans:
        print("all nans")
        # If no matches in archive, query website
        return chempropSolvationHelper(monomer_smiles, monomer_base_state, solvent_smiles)
    
    return solvent_data

def getSoluteFeatures(monomer_smiles, archive_path = "data archive/featurized_archive.csv"):
    '''
        A function that returns solute features for a specific monomer

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
            archive_path (str): the path to the featurized_archive.csv file

        Output:
            (dictionary): dictionary of solute data with keys having prefix SOLUTE_
    '''

    # Start with GC
    
    # Turn the csv into a pandas array
    archive_df = pd.read_csv(archive_path)

    solute_data = {
        "SOLUTE_PARAM_E_GC": np.nan,
        "SOLUTE_PARAM_S_GC": np.nan,
        "SOLUTE_PARAM_A_GC": np.nan,
        "SOLUTE_PARAM_B_GC": np.nan,
        "SOLUTE_PARAM_L_GC": np.nan,
        "SOLUTE_PARAM_V_GC": np.nan,
        "SOLUTE_PARAM_E_ML": np.nan,
        "SOLUTE_PARAM_S_ML": np.nan,
        "SOLUTE_PARAM_A_ML": np.nan,
        "SOLUTE_PARAM_B_ML": np.nan,
        "SOLUTE_PARAM_L_ML": np.nan,
        "SOLUTE_PARAM_V_ML": np.nan,
    }

    # Look for matches in archive
    if monomer_smiles in archive_df['Canonical SMILES'].values:
        # Loop through all columns with missing values
        for key in solute_data:
            corresponding_i = archive_df.loc[
                archive_df["Canonical SMILES"] == monomer_smiles, key
            ].values

            solute_data[key] = corresponding_i[0]

    all_nans = all(np.isnan(key) if isinstance(key, float) else False for key in solute_data.keys())

    # if not in archive data, query website
    if all_nans: 
            return soluteHelper()

    return solute_data

def getRdkitDescriptors(monomer_smiles):
    '''
        A function that returns rdkit descriptor features for a specific monomer

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer

        Output:
            (dictionary): a dictionary of all the rdkit values with keys with prefix RDKIT_
    '''
    # Disable RDKit warnings
    RDLogger.DisableLog('rdApp.warning')

    # get column names, add prefix, and append
    names = [("RDKIT_" + x[0]) for x in Descriptors._descList]
    # Convert list to dictionary with default value None
    rdkit_dict = {col: np.nan for col in names}

    mol = MolFromSmiles(monomer_smiles)

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList]
    )
    dict_vals = calc.CalcDescriptors(mol)
    keys = list(rdkit_dict.keys())
    rdkit_dict.update(dict(zip(keys, dict_vals)))


    return rdkit_dict

# STERIC feature functions (relevant to entropy)
def getSterics(monomer_smiles, polymerization_type):
    '''
        A function that returns steric features for a specific monomer

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
            polymerization_type (str): the type of polymerization the monomer undergoes

        Output:
            (dict): a dictionary of steric features. key values have the prefix STERIC_
    '''
    repeat_unit = getRepeatUnit(monomer_smiles, polymerization_type)

    bb_lens = np.nan
    ratios = np.nan

    # retrieve backbone length and ratio
    try:
        bb_lens, ratios = stericsHelper(repeat_unit)

    except:
        bb_lens = np.nan
        ratios = np.nan
    
    output = {"STERIC_backbone_length": bb_lens, "STERIC_ratioSideChainContainingAtoms": ratios}
    
    return output

def getWienerIndex(monomer_smiles, polymerization_type):
    '''
        A function that calculates the Wiener Index using an altered version of Greg Landrum's code: 
        https://sourceforge.net/p/rdkit/mailman/message/36802142/

        Parameters: 
            monomer_smiles(str): the canonical smiles string of the monomer
            polymerization_type (str): the type of polymerization the monomer undergoes

        Output:
            (float): the sum of distances between all pairs of atoms in the molecule

    '''    
    try:
        repeat_unit = getRepeatUnit(monomer_smiles, polymerization_type)
        mol = MolFromSmiles(repeat_unit)
        sum = 0
        # amat is a distance matrix
        amat = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()

        for i in range(num_atoms):
            for j in range(i+1,num_atoms):
                # adds the distance between atom i and j
                sum += amat[i][j]
        return sum
    except: 
        print(f"Could not get Wiener Index for {monomer_smiles}")
        return np.nan

def getNumChiralCenters(monomer_smiles):
    '''
        A function that provides which atoms in the molecule are chiral centers and the orientation of the center

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
        
        Output:
            (int): the count of chiral centers
    '''

    mol = MolFromSmiles(monomer_smiles)
    chiral_list = FindMolChiralCenters(mol)

    fin_list = []
    for center in chiral_list:
        atomic_num = center[0]
        atom_name = periodictable.elements[atomic_num]
        # [atom, orientation]
        fin_list.append([atom_name,center[1]])

    ## can adjust the function to output fin_list if you would rather have atom_name and center orientation information 
    return len(fin_list)

def getRadiusofGyration(monomer_smiles):
    '''
        A function that calcualtes the radius of gyration with code altered from the following github code: 
        https://github.com/rdkit/rdkit/issues/2924

        Parameters: 
            monomer_smiles(str): the canonical smiles string of the monomer
        
        Output:
            (float): the radius
    '''    

    mol = MolFromSmiles(monomer_smiles)

    # generate conformers for molecule to work in 3d space
    m3d=getConformers(mol)

    if m3d.GetNumConformers()>=1:
        radius = CalcRadiusOfGyration(m3d)
        return radius
    else: 
        return np.nan

def getSpherocity(monomer_smiles):
    '''
        A function that measures the spherocity of a monomer using RDKIT

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer

        Output:
            (float): spheroity id
    '''

    try:
        mol = MolFromSmiles(monomer_smiles)

        # generate conformers for molecule to work in 3d space
        m3d=getConformers(mol)

        sphere_id = CalcSpherocityIndex(m3d)
        return sphere_id
    except:
        print(f"Could not get spherocity of {monomer_smiles}")
        return np.nan

def getVolume(monomer_smiles):
    '''
        A function that uses rdkit to find the Van der Waals volume and total volume of a molecule 

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer

        Output:
            (dictionary) contains the volumes with keys having prefix STERIC_
    '''

    try: 
        output = {}
        mol= MolFromSmiles(monomer_smiles)
        
        # generate conformers for molecule to work in 3d space
        m3d=getConformers(mol)

        dcl_id = DoubleCubicLatticeVolume(m3d)

        # Get the van der Waals Volume of the molecule 
        output["STERIC_vdwVolume"] = dcl_id.GetVDWVolume()
        output["STERIC_totalVolume"] = dcl_id.GetVolume()

        return output

    except:
        print(f"unable to get volume/vdw volume for {monomer_smiles}")
        return np.nan

def getNumBridgeheadAtoms(monomer_smiles):
    '''
        A function that uses rdkit to find the number of bridgehead atoms

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer

        Output:
            (int): a count of the number of bridgehead atoms
    '''

    mol = MolFromSmiles(monomer_smiles)
    bh = CalcNumBridgeheadAtoms(mol)
    return bh

def getNumStereocenters(monomer_smiles, polymerization_type):
    '''
        A function that uses rdkit to find potential stereo elements in a molecule 

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
            polymerization_type (str): the type of polymerization the monomer undergoes

        Output:
            (int) a count of the number of stereocenters
    '''

    repeat_unit_smiles = getRepeatUnit(monomer_smiles, polymerization_type)
    mol = MolFromSmiles(repeat_unit_smiles)
    # p_stereos is a list of StereoInfo objects
    p_stereos = FindPotentialStereo(mol)
    return len(p_stereos)

def getExperimentalSolvent(solvent_name, solvent_path="data archive/solvents_archive.csv"):
    '''
        A function that retrieves the experimental solvent data in solvent_archive

        Parameters: 
            solvent_name (str): the name of the solvent that the monomer is in
            solvent_path (str): the path to the solvent_archive.csv
        
        Output:
            (dict): a dictionary of experimental solvent data
    '''

    try:
        # Dataframes
        solvent_df = pd.read_csv(
            solvent_path,
            encoding="utf-8",
            usecols=[
                "Solvent",
                "Solvent_SMILES",
                "Solvent_SMILES_2",
                "SOLV_PARAM_s_g",
                "SOLV_PARAM_b_g",
                "SOLV_PARAM_e_g",
                "SOLV_PARAM_l_g",
                "SOLV_PARAM_a_g",
                "SOLV_PARAM_c_g",
                "SOLV_PARAM_visc at 298 K (cP)",
                "SOLV_PARAM_dielectric constant",
            ],
        )

        idx_match = solvent_df.loc[
            solvent_df["Solvent"] == solvent_name
        ].index[0]
        
        matching_row = solvent_df.iloc[idx_match]
        solvent_data = {
            "Solvent": matching_row["Solvent"],
            "Solvent_SMILES": matching_row["Solvent_SMILES"],
            "Solvent_SMILES_2": matching_row["Solvent_SMILES_2"],
            "SOLV_PARAM_s_g": matching_row["SOLV_PARAM_s_g"],
            "SOLV_PARAM_b_g": matching_row["SOLV_PARAM_b_g"],
            "SOLV_PARAM_e_g": matching_row["SOLV_PARAM_e_g"],
            "SOLV_PARAM_l_g": matching_row["SOLV_PARAM_l_g"],
            "SOLV_PARAM_a_g": matching_row["SOLV_PARAM_a_g"],
            "SOLV_PARAM_c_g": matching_row["SOLV_PARAM_c_g"],
            "SOLV_PARAM_visc at 298 K (cP)": matching_row["SOLV_PARAM_visc at 298 K (cP)"],
            "SOLV_PARAM_dielectric constant": matching_row["SOLV_PARAM_dielectric constant"],
        }

        return solvent_data
    
    except:
        print(f"There is no information on {solvent_name} in the solvent data file")
        # if no experimental solvent data, return null
        solvent_data = {
            "Solvent": np.nan,
            "Solvent_SMILES": np.nan,
            "Solvent_SMILES_2": np.nan,
            "SOLV_PARAM_s_g": np.nan,
            "SOLV_PARAM_b_g": np.nan,
            "SOLV_PARAM_e_g": np.nan,
            "SOLV_PARAM_l_g": np.nan,
            "SOLV_PARAM_a_g": np.nan,
            "SOLV_PARAM_c_g": np.nan,
            "SOLV_PARAM_visc at 298 K (cP)": np.nan,
            "SOLV_PARAM_dielectric constant": np.nan,
        }
        return solvent_data

def main(infile_path, target, dp):
    # This function takes around 5 minutes to run
    # target is dS (J/mol/K) or dH (KJ/mol)
    filename = os.path.splitext(os.path.basename(infile_path))[0]
    directory_path = os.path.dirname(infile_path)

    new_file_name = "featurized_" +  filename + ".csv"
    featurized_path = directory_path + "/" + new_file_name

    # turn .csv into a pandas dataframe
    unfeaturized_df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")

    for index,row in unfeaturized_df.iterrows():
        #  get the parameter data
        canonical_monomer_smiles = row["Canonical SMILES"]
        print(f"Featurizing {canonical_monomer_smiles}")

        #  get the monomer and polymer base states
        base_state = row["BASE_State"]
        monomer_base_state_col = base_state[0]
        poly_base_state_col = base_state[1:]

        # get the parameters that will go into getAllFeatures
        solvent_name = row["Solvent"]
        # print(solvent_name)
        monomer_base_state = monomer_base_state_col
        polymerization_type = row["BASE_Category"]

        update_dict = getAllFeatures(canonical_monomer_smiles, monomer_base_state, polymerization_type, dp, solvent_name, target)
        if target == "dH":
            update_dict["dH (KJ/mol)"] = row["dH (KJ/mol)"]
        else:
            update_dict["dS (J/mol/K)"] = row["dS (J/mol/K)"]

        update_dict["BASE_Polymer_State"] = poly_base_state_col

        if index ==0:
            featurized_df = pd.DataFrame([update_dict])
            featurized_df.to_csv(featurized_path, encoding='utf-8-sig')

        # Update the DataFrame containing featurized data
        featurized_df.loc[len(featurized_df)] = update_dict

        # Update the csv by overwriting latest version
        featurized_df.to_csv(featurized_path, encoding='utf-8-sig')

    print("OPERATION COMPLETE :)")

