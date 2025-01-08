import math
import os
import requests
from urllib import parse

import numpy as np
import pandas as pd

from openbabel import pybel
from openbabel._openbabel import OBChargeModel_FindType

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem import Descriptors, MolFromSmiles

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)

from rdkit.DataStructs import TanimotoSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdmolops import GetShortestPath, FindPotentialStereo
from rdkit.Chem import rdChemReactions
from rdkit.Chem import Descriptors, MolFromSmiles, GetDistanceMatrix, FindMolChiralCenters
from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration, CalcSpherocityIndex, CalcNumBridgeheadAtoms


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

# Function to calculate Tanimoto similarity between two SMILES
def calculate_tanimoto_similarity(smiles1, smiles2):
    mol1 = MolFromSmiles(smiles1)
    mol2 = MolFromSmiles(smiles2)

    if mol1 is not None and mol2 is not None:
        fp1 = mfpgen.GetFingerprint(mol1)
        fp2 = mfpgen.GetFingerprint(mol2)

        return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        return 0.0

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
        PEP = Hfs[max_DP] - Hfs[max_DP-1] - Hfs[0]
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
        fp1 = mfpgen.GetFingerprint(MolFromSmiles(monomer_SMILES))
        fp2 = mfpgen.GetFingerprint(MolFromSmiles(solvent_SMILES))

        score = round(TanimotoSimilarity(fp1, fp2), 3)

        return score

    # For gas monomer, similarity should be none
    else:
        return -1

def dipole_moments(monomer_SMILES, dipole_calc_method):
    dipole_calcs = []

    # Get an average since 3D generation is stochastic
    for i in range(100):
        # Generate deterministic 3D structure with RDKit
        mol = Chem.MolFromSmiles(monomer_SMILES)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        # Save as SDF for compatibility with Pybel
        sdf_filename = "temp.sdf"
        writer = Chem.SDWriter(sdf_filename)
        writer.write(mol)
        writer.close()

        for mol in pybel.readfile("sdf", sdf_filename):
            cm = OBChargeModel_FindType(dipole_calc_method)
            cm.ComputeCharges(mol.OBMol)
            dipole = cm.GetDipoleMoment(mol.OBMol)
            os.remove(sdf_filename)
            dipole_calcs.append(math.sqrt(dipole.GetX() ** 2 + dipole.GetY() ** 2 + dipole.GetZ() ** 2))

    return round(np.average(dipole_calcs), 2)

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
        return [np.nan, np.nan]

    # Then use operator to get the repeat unit
    try:
        rxn = _repeat_rxn_dict[category]

        if category == "ROMP":
            temp = MolFromSmiles("*=*")
        else:
            temp = MolFromSmiles("**")

        mol = rxn.RunReactants((monomer, temp))[0][0]
    
    except:
        print("No RU created:", monomer_SMILES, category)
        return [np.nan, np.nan]
    
    RU_smiles = Chem.MolToSmiles(mol)

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

        # backbone length and side chain ratio (Shivani)
        backbone_len = len(backbone)
        ratio = round((len(atom_list) - len(backbone)) / len(atom_list), 2)

        # Hunter
        wiener_idx = _get_wiener_idx(monomer_SMILES, category, RU_smiles)
        chiral_centers = _get_num_chiral_centers(monomer_SMILES)
        rad_gyration = _get_radius_of_gyration(monomer_SMILES)
        spherocity = _get_spherocity(monomer_SMILES)
        bridgehead = _get_num_bridgehead_atoms(monomer_SMILES)
        stereocenters = _get_num_stereocenters(monomer_SMILES, category, RU_smiles)

        return [backbone_len, ratio, wiener_idx, chiral_centers, rad_gyration, spherocity, bridgehead, stereocenters]
    except:
        raise Exception("Issue with backbone length and side chain ratio calculation (or Hunter's features but too many to list).")
    

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



def _get_wiener_idx(monomer_smiles, polymerization_type, ru_smiles):
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
        repeat_unit = ru_smiles
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

def _get_num_chiral_centers(monomer_smiles):
    '''
        A function that provides which atoms in the molecule are chiral centers and the orientation of the center

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
        
        Output:
            (int): the count of chiral centers
    '''

    mol = MolFromSmiles(monomer_smiles)
    chiral_list = FindMolChiralCenters(mol)

    ptable = Chem.GetPeriodicTable()

    fin_list = []
    for center in chiral_list:
        atomic_num = center[0]
        atom_name = ptable.GetElementSymbol(atomic_num)
        # [atom, orientation]
        fin_list.append([atom_name,center[1]])

    ## can adjust the function to output fin_list if you would rather have atom_name and center orientation information 
    return len(fin_list)

def _get_radius_of_gyration(monomer_smiles):
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
    m3d=_get_conformers(mol)

    if m3d.GetNumConformers()>=1:
        radius = CalcRadiusOfGyration(m3d)
        return radius
    else: 
        return np.nan

def _get_spherocity(monomer_smiles):
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
        m3d=_get_conformers(mol)

        sphere_id = CalcSpherocityIndex(m3d)
        return sphere_id
    except:
        print(f"Could not get spherocity of {monomer_smiles}")
        return np.nan

def _get_num_bridgehead_atoms(monomer_smiles):
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

def _get_num_stereocenters(monomer_smiles, polymerization_type, ru_smiles):
    '''
        A function that uses rdkit to find potential stereo elements in a molecule 

        Parameters:
            monomer_smiles(str): the canonical smiles string of the monomer
            polymerization_type (str): the type of polymerization the monomer undergoes

        Output:
            (int) a count of the number of stereocenters
    '''

    repeat_unit_smiles = ru_smiles
    mol = MolFromSmiles(repeat_unit_smiles)
    # p_stereos is a list of StereoInfo objects
    p_stereos = FindPotentialStereo(mol)
    return len(p_stereos)


def _get_conformers(mol):
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