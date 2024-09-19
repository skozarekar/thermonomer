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

from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdmolops import GetShortestPath
from rdkit.Chem import rdChemReactions

# my imports
import pgthermo.properties as prop
from thermonomer.polymerize import polymerize

# Private variables
_repeat_rxn_dict = {
    "ROP": rdChemReactions.ReactionFromSmarts(
    "([C,S:1](=[O,S:2])@[O,S,N:3]@[C:4]).[*:5]-[*:6]>>([C,S:1](=[O,S:2])[*:5].[*:6]-[O,S,N:3][C:4])"),
    "ROMP": rdChemReactions.ReactionFromSmarts(
    "[C:1]=[C:2].([*:3]=[*:4])>>([C:1]=[*:3].[*:4]=[C:2])"),
    "cationic": rdChemReactions.ReactionFromSmarts(
    "([*:1]@[*:2]).[*:3]-[*:4]>>([*:2][*:3].[*:1][*:4])"),
    "vinyl/acrylic": rdChemReactions.ReactionFromSmarts(
    "[C:1]=[C:2].[*:3]-[*:4]>>[*:3]-[C:1]-[C:2]-[*:4]"),
    "aldehyde": rdChemReactions.ReactionFromSmarts(
    "[C:1]=[O,S:2].[*:3]-[*:4]>>[*:3]-[C:1]-[O,S:2]-[*:4]"),
    "cyclic": rdChemReactions.ReactionFromSmarts(
    "([CH2:1]-[CH2:2]).[*:3]-[*:4]>>([C:2]-[*:3].[C:1]-[*:4])"),
}

def PEP(monomer_SMILES, max_DP, category):
    if category == 'misc':
        return [np.nan for i in range(max_DP + 1)]

    # Get polymers
    try:
        poly_list = polymerize(max_DP, monomer_SMILES, category)
    except:
        raise Exception(f"Unable to get polymers up to DP {max_DP}.")
    
    # Calculate Hfs for monomer through max_DP
    Hfs = []
    for polymer_smiles in poly_list:
        heat_formation = prop.Hf(polymer_smiles)
        Hfs.append(heat_formation)

    try:
        PEP = Hfs[max_DP-1] - Hfs[max_DP-2] - Hfs[0]
        Hfs.append(PEP)
    except:
        raise Exception("Unable to calculate PEP.")

    return Hfs

def tanimoto_monomer_solvent_similarity(monomer_state, monomer_SMILES, solvent_SMILES):
    # If liquid monomer, max similarity
    if monomer_state == "l":
        return 1

    # If solution-phase monomer, calculate similarity based on fingerprints
    elif monomer_state == "s":
        if pd.isna(solvent_SMILES):
            raise Exception("No solvent exists for ss state.")

        # Calculate fingerprints
        fp1 = GetMorganFingerprintAsBitVect(MolFromSmiles(monomer_SMILES), 3, nBits=2048)
        fp2 = GetMorganFingerprintAsBitVect(MolFromSmiles(solvent_SMILES), 3, nBits=2048)

        score = round(TanimotoSimilarity(fp1, fp2), 3)

        return score

    # For gas monomer, similarity should be none
    else:
        return -1

def dipole_moments(monomer_SMILES, dipole_calc_method):
    # Create molecule file
    os.system('obabel -:"' + str(monomer_SMILES) + '" -i smiles -O temp.sdf --gen3d')

    for mol in pybel.readfile("sdf", "temp.sdf"):
        # Calculate dipole with given method
        cm = OBChargeModel_FindType(dipole_calc_method)
        cm.ComputeCharges(mol.OBMol)
        dipole = cm.GetDipoleMoment(mol.OBMol)

        # Delete file
        os.system('rm temp.sdf')

        # Return scalar
        return round(math.sqrt(dipole.GetX() ** 2 + dipole.GetY() ** 2 + dipole.GetZ() ** 2), 2)

def rdkit_2D_features(monomer_SMILES):
    '''
        calculates the RDKIT 2D descriptors

        Args: 
            monomer_SMILES

        Returns: 
            list: the retrieved descriptor data
    '''
    
    mol = MolFromSmiles(monomer_SMILES)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList]
    )

    return list(calc.CalcDescriptors(mol))

def steric_features(monomer_SMILES, category):
    # Calculate repeat unit
    monomer = MolFromSmiles(monomer_SMILES)

    # If misc, leave blank
    if category == "misc":
        return np.nan, np.nan

    # Then use operator
    try:
        rxn = _repeat_rxn_dict[category]

        if category == "ROMP":
            temp = MolFromSmiles("*=*")
        else:
            temp = MolFromSmiles("**")

        mol = rxn.RunReactants((monomer, temp))[0][0]
    
    except:
        print("No RU created:", monomer_SMILES, category)
        return list(np.nan, np.nan)

    # Calculate steric features
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

        return list(backbone_len, ratio)
    except:
        raise Exception("Issue with backbone length and side chain ratio calculation.")

def get_RMG_mix(monomer_state, monomer_SMILES, solvent_SMILES):
    '''
        a function that retrieves solvent data

        Parameters:
            row: a row from the pandas dataframe
        Output:
            (pd.Series): collection of data on the solvent
    '''
    # set variables
    base_url = "https://rmg.mit.edu/database/solvation/searchML/solv_solu="
    url_flags = "__dG=True__dH=True__dS=True__logK=True__logP=True__unit=kJ/mol"

    state = monomer_state
    solute = monomer_SMILES
    solvent = solvent_SMILES

    # if we have solution or liquid as the monomer state, set the URL
    if state == "s":
        if pd.isnull(solvent):
            return [np.nan] * 7

        url = base_url + str(solute) + "_" + str(solvent) + url_flags
    elif state == "l":
        url = base_url + str(solute) + "_" + str(solute) + url_flags
    # otherwise, return empty dataframe
    else:
        return [np.nan] * 7

    # make request
    url = url.replace(" ", "")
    encoded_url = parse.quote(url, safe=":/")
    response = requests.get(encoded_url)
    if response.status_code != 200:  # catch errors
        return [np.nan] * 7

    table = pd.read_html(response.content)
    df = table[0]

    # return the values from chemprop
    return [float(df["dGsolv298(kJ/mol)"].iloc[0]),
            float(df["dHsolv298(kJ/mol)"].iloc[0]),
            float(df["dSsolv298(kJ/mol/K)"].iloc[0]),
            float(df["logK"].iloc[0]),
            float(df["logP"].iloc[0]),
            float(df["dGsolv298 epi.unc.(kJ/mol)"].iloc[0]),
            float(df["dHsolv298 epi.unc.(kJ/mol)"].iloc[0])]

    
def get_RMG_solute_params(monomer_SMILES):
        '''
            a function that retrieves solute data

            Parameters:
                row: a row from the pandas dataframe
            Output:
                (pd.Series): collection of data on the solute
        '''

        # base url to website that does the calculations
        base_url = "https://rmg.mit.edu/database/solvation/soluteSearch/solu="
        url_flags_GC = "__estimator=SoluteGC__solv=None__unit=kJ/mol"
        url_flags_ML = "__estimator=SoluteML__solv=None__unit=kJ/mol"

        if "[Si]" in monomer_SMILES or "[Se]" in monomer_SMILES:
            # website software can not handle molecules with [Si] or [Se]
            return [np.nan] * 12

        # make request to GC
        url = base_url + monomer_SMILES + url_flags_GC
        url = url.replace(" ", "")
        encoded_url = parse.quote(url, safe=":/")
        response = requests.get(encoded_url)
        if response.status_code != 200:  # catch errors
            return [np.nan] * 12

        table = pd.read_html(response.content)
        df_gc = table[0]

        # Abraham solute parameters (E, S, A, B, L, V)
        list_GC = [
            float(df_gc["E"].iloc[0]),
            float(df_gc["S"].iloc[0]),
            float(df_gc["A"].iloc[0]),
            float(df_gc["B"].iloc[0]),
            float(df_gc["L"].iloc[0]),
            float(df_gc["V"].iloc[0]),
        ]

        # make request to ML
        url = base_url + str(monomer_SMILES) + url_flags_ML
        url = url.replace(" ", "")
        encoded_url = parse.quote(url, safe=":/")
        response = requests.get(encoded_url)
        table = pd.read_html(response.content)
        df_ml = table[0]

        list_ML = [
            float(df_ml["E"].iloc[0]),
            float(df_ml["S"].iloc[0]),
            float(df_ml["A"].iloc[0]),
            float(df_ml["B"].iloc[0]),
            float(df_ml["L"].iloc[0]),
            float(df_ml["V"].iloc[0]),
        ]

        # return the values from chemprop
        return list_GC + list_ML