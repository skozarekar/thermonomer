'''
This python file adds features to cleaned data 
.csv files that this program uses:
    2c_archive.csv
    2a_polymerized_with_solvents.csv: has polymer SMILES and merged solvents
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

from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdmolops import GetShortestPath

# imports from github/other
import pgthermo.properties as prop
from repeat_unit_dict import repeat_rxn_dict

class AddFeatures:
    '''
        class AddFeatures: 
            A class that featurizes a dataset of monomers and saves the features in a new .csv
            Applys features in the following order:
                1. pre-estimation property (PEP)
                2. Tanimoto M/S similarity
                3. Dipole moment
                4. Solvent parameters
                5. Solute parameters
                6. RDKit descriptors
                7. Steric descriptors

        Parameters for initiation: 
            path (str): the directory path to the directory that contains all of the user's datasets (ie. ends in /0_data_sets/)
            dp (int): degree of polymerization

        Output:
            creates a new .csv file containing monomer and feature data (ie. titled "2c_featurized.csv")
    '''
    
    def __init__(self, path, degree_of_polymerization):
        # in __main__, user must predefine folder path and the max chain length they wish to investigate
        self.path = path
        self.degree_of_polymerization = degree_of_polymerization

        # filter warning
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # Set paths
        #infile path is path to .csv that has polymer SMILES and merged solvents created in part 2A
        self.infile_path = self.path + "2a_polymerized_with_solvents.csv"
        self.archive_path = self.path + "2c_archive.csv"
        # self.feature_path is the path to the .csv file that is being updated everytime a feature is added
        self.feature_path = self.path + "2c_featurized.csv"

        # Read dataframe
        self.df = pd.read_csv(self.infile_path, index_col=0, encoding="utf-8")

        try:
            self.archive_df = pd.read_csv(self.archive_path, index_col=0, encoding="utf-8")
        except:
            print("No archive found.")

    def main(self):
        '''
            Applies all the features and saves after each function call updating 2c_featurized.csv. User can customize
            what features they wish to apply by (un)commenting function calls

            Parameters: 
                None

            Output: 
                None
        '''
        self.apply_pep() 
        self.apply_tanimoto_similarity() 
        self.apply_dipole()
        self.apply_solvent() 
        self.apply_solute()
        self.apply_rdkit_descriptors() 
        self.apply_steric() 

    # ---------- HELPER FUNCTIONS ---------- #
    def save_df(self):
        '''
            saves the input dataframe to 2c_featurized.csv output file

            Parameters:
                None

            Output:
                None
        '''
        self.df.to_csv(self.feature_path, encoding='utf-8-sig')

    def calc_Hf(self,row):
        '''
            a function that calculates the heats of formation for each degree of polymerization in a row of the dataframe

            Parameters:
                row: a row from the pandas dataframe

            Output:
                (pandas series) a one-dimensional labeled array holding hf values
        '''
        # a list that holds the heats of formation for each degree of polymerization
        hf_list = []

        # iterate through all the degrees of polymerization and get the corresponding heat of formation
        for i in range(self.degree_of_polymerization + 1):
            #extract smile string from row data
            smiles = row["DP_" + str(i)] 

            if pd.isna(smiles):  
                # if there is no smile string, append nothing
                hf_list.append(np.nan)

            else:
                enthalpy = prop.Hf(smiles, return_missing_groups=True)

                if type(enthalpy) is list: 
                    # there are missing groups present so pgthermo can't compelte enthalpy calculation
                    # enthalpy variable is a list of the groups that are missing
                    unique_missing = list(set(enthalpy)) # eliminate duplicates in the list

                    # save these groups to be manually dealth with later
                    hf_list.append()
                    # for group in unique_missing:
                    #     print(f"Groups that are missing: {group}")

                else:  # type is float, hf successfully calculated
                    hf_list.append(enthalpy)

        return pd.Series(hf_list)
    
    def tanimoto(self,row):
        '''
            a function that uses tanimoto to quantify similarity between monomer and solvent

            Parameters:
                row: a row from the pandas dataframe

            Output:
                (int) similarity value
        '''

        # Get monomer SMILES and solvent SMILES to compare
        smi1 = row["Canonical SMILES"]
        smi2 = row["Solvent_SMILES"]

        # If liquid monomer, max similarity
        if row["BASE_Monomer_State"] == "l":
            return 1

        # If solution-phase monomer, calculate similarity based on fingerprints
        elif row["BASE_Monomer_State"] == "s":
            if pd.isna(smi2):
                # Error if solvent data missing
                print("\tss without solvent:", smi1)
                return -2

            # Calculate
            mol1 = MolFromSmiles(smi1)
            mol2 = MolFromSmiles(smi2)
            fp1 = GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
            fp2 = GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)

            score = round(TanimotoSimilarity(fp1, fp2), 3)

            return score

        # For gas monomer, similarity should be none
        else:
            return -1
    
    def calc_dipole(self,method, smiles):
        '''
            a function that calculates the dipole moment of the monomer 

            Parameters:
                method (str): the method to use for calculating dipole
                smiles (str): the smiles string of the molecule 

            Output:
                (float) 
        '''
        # Create molecule file
        os.system('obabel -:"' + str(smiles) + '" -i smiles -O temp.sdf --gen3d')

        for mol in pybel.readfile("sdf", "temp.sdf"):
            # Calculate dipole with given method
            cm = OBChargeModel_FindType(method)
            cm.ComputeCharges(mol.OBMol)
            dipole = cm.GetDipoleMoment(mol.OBMol)

            # Return scalar
            return math.sqrt(dipole.GetX() ** 2 + dipole.GetY() ** 2 + dipole.GetZ() ** 2)
        
    def get_chemprop_solvation(self,row):
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

        state = row["BASE_Monomer_State"]
        solute = row["Canonical SMILES"]
        solvent = row["Solvent_SMILES"]

        # if we have solution or liquid as the monomer state, set the URL
        if state == "s":
            if pd.isnull(solvent):
                return pd.Series([np.nan] * 7)

            url = base_url + str(solute) + "_" + str(solvent) + url_flags
        elif state == "l":
            url = base_url + str(solute) + "_" + str(solute) + url_flags
        # otherwise, return empty dataframe
        else:
            return pd.Series([np.nan] * 7)

        # make request
        url = url.replace(" ", "")
        encoded_url = parse.quote(url, safe=":/")
        response = requests.get(encoded_url)
        if response.status_code != 200:  # catch errors
            return pd.Series([np.nan] * 7)

        table = pd.read_html(response.content)
        df = table[0]
        # return the values from chemprop
        return pd.Series(
            [
                float(df["dGsolv298(kJ/mol)"]),
                float(df["dHsolv298(kJ/mol)"]),
                float(df["dSsolv298(kJ/mol/K)"]),
                float(df["logK"]),
                float(df["logP"]),
                float(df["dGsolv298 epi.unc.(kJ/mol)"]),
                float(df["dHsolv298 epi.unc.(kJ/mol)"]),
            ]
        )

    def get_solute_params(self,row):
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

        monomer_smiles = row["Canonical SMILES"]

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

        # Abraham solute parameters (E, S, A, B, L, V)
        list_GC = [
            float(df_gc["E"]),
            float(df_gc["S"]),
            float(df_gc["A"]),
            float(df_gc["B"]),
            float(df_gc["L"]),
            float(df_gc["V"]),
        ]

        # make request to ML
        url = base_url + str(monomer_smiles) + url_flags_ML
        url = url.replace(" ", "")
        encoded_url = parse.quote(url, safe=":/")
        response = requests.get(encoded_url)
        table = pd.read_html(response.content)
        df_ml = table[0]

        list_ML = [
            float(df_ml["E"]),
            float(df_ml["S"]),
            float(df_ml["A"]),
            float(df_ml["B"]),
            float(df_ml["L"]),
            float(df_ml["V"]),
        ]

        # return the values from chemprop
        return pd.Series(list_GC + list_ML) 
    
    def rdkit_descriptors(self):
        '''
            Calculates for all monomers their RDKit 2D descriptors, which include fragment counts.
            Removes any fragment columns where fragments do not show up in any monomers.

            Parameters:
                None

            Returns:
                None
        '''

        def rdkit_2D_features(row):
            '''
                A helper function within rdkit_descriptors() that calculates the RDKIT 2D descriptors

                Parameters: 
                    row: row: a row from the pandas dataframe

                Returns: 
                    (pd.Series): the retrieved descriptor data
            '''
            mol = MolFromSmiles(row["Canonical SMILES"])
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(
                [x[0] for x in Descriptors._descList]
            )
            ds = calc.CalcDescriptors(mol)

            return pd.Series(ds)

        # get column names, add prefix, and append
        names = [("RDKIT_" + x[0]) for x in Descriptors._descList]
        self.df = pd.concat([self.df, pd.DataFrame(columns=names)])

        # apply features
        num_cols = -1 * len(names)
        self.df[self.df.columns[num_cols:]] = self.df.apply(rdkit_2D_features, axis=1)

        # drop columns where fragments columns are mostly 0
        self.df = self.df.loc[:, ~(self.df == 0).all()]

    def get_backbone_length_and_ratio(self,repeat_unit_smiles):
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
    
    # ---------- FEATURE FUNCTIONS ---------- #

    def apply_pep(self):
        '''
            A function that solves for and saves the pre-estimation property feature data in the dataframe. 
            The pre-estimation property is found using pgthermo.

            Parameters:
                None

            Output:
                None
        '''

        # create a column for each degree of polymerization Hf value
        col_names = ["PGTHERMO_Hf_" + str(i) for i in range(self.degree_of_polymerization + 1)]

        # get the heats of formation for each polymer chain in each row
        self.df[col_names] = self.df.apply(func=self.calc_Hf, axis=1)

        #calculate the estimation property PGTHERMO_dH_ (our PEP)
        self.df["PGTHERMO_dH_ (kcal/mol)"] = round(
            self.df["PGTHERMO_Hf_2"] - self.df["PGTHERMO_Hf_1"] - self.df["PGTHERMO_Hf_0"], 2
        )

        # save the current dataframe that now has pre-estimation property (PEP) feature
        self.save_df()

    def apply_tanimoto_similarity(self):
        '''
            A function that solves for and saves the similarity between the monomer and solvent.

            Parameters:
                None

            Output:
                None
        '''
        self.df["TANIMOTO_similiarity"] = self.df.apply(self.tanimoto, axis=1)
        self.save_df()

    def apply_dipole(self):
        '''
            A function that solves for and saves the dipole moment of the monomer

            Parameters:
                None

            Output:
                None
        '''
        #methods used to calculate dipole: gasteiger, mmff94, eem2015bm

        self.df["DIPOLE_gasteiger"] = [
            self.calc_dipole("gasteiger", smi) for smi in self.df["Canonical SMILES"]
        ]

        self.df["DIPOLE_mmff94"] = [self.calc_dipole("mmff94", smi) for smi in self.df["Canonical SMILES"]]

        self.df["DIPOLE_eem2015bm"] = [
            self.calc_dipole("eem2015bm", smi) for smi in self.df["Canonical SMILES"]
        ]

        os.system("rm temp.sdf")

        self.save_df()
    
    def apply_solvent(self):
        '''
            A function that saves solvent features to the dataframe

            Parameters:
                None

            Output:
                None
        '''
        code_list = []

        #solvent pair lookup is dependent on the base state of the monomer
        for index, row in self.df.iterrows(): 
            if row['BASE_Monomer_State'] == 'l':
                code_list.append(row['Canonical SMILES'] + '_' + row['Canonical SMILES'])
            elif row['BASE_Monomer_State'] == 's' and row['Solvent_SMILES']:
                code_list.append(row['Canonical SMILES'] + '_' + row['Solvent_SMILES'])
            else:
                code_list.append("")

        self.df['Solvent_pair_lookup'] = code_list

        # the columns to add to 2c_featurized.csv (=self.df)
        columns = [
            "MIX_dGsolv298(kJ/mol)",
            "MIX_dHsolv298(kJ/mol)",
            "MIX_dSsolv298(kJ/mol/K)",
            "MIX_logK",
            "MIX_logP",
            "MIX_dGsolv298(kJ/mol) epi.unc.",
            "MIX_dHsolv298(kJ/mol) epi.unc.",
        ]

        for col in columns:
            self.df[col] = 0

        indices = [i for i in range(len(columns))]

        match_columns = ["MIX_dGsolv298(kJ/mol)", "MIX_dHsolv298(kJ/mol)", "MIX_dSsolv298(kJ/mol/K)", "MIX_logK", "MIX_logP", "MIX_dGsolv298(kJ/mol) epi.unc.", "MIX_dHsolv298(kJ/mol) epi.unc."]

        for index, row in self.df.iterrows():
            # Check if row has all zeros
            all_zeros = row[columns].eq(0).all().all()
            all_nans = row[columns].eq(np.nan).all().all()

            # Make sure data is not already there
            if all_zeros or all_nans:
                # Look for matches in archive
                lookup_val = row['Solvent_pair_lookup']
                if lookup_val in self.archive_df['Solvent_pair_lookup'].values:
                    print(lookup_val)
                    # Loop through all columns with missing values
                    for col_name in match_columns:
                        corresponding_i = self.archive_df.loc[
                            self.archive_df["Solvent_pair_lookup"] == lookup_val, col_name
                        ].values

                        self.df.loc[index, col_name] = corresponding_i[0]

                # If no matches in archive, query website
                web_vals = self.get_chemprop_solvation(row)

                for pair in zip(indices, columns):
                    dp_column_index, dp_column_name = pair

                    # Update columns starting from 'DP_0'
                    self.df.at[index, dp_column_name] = web_vals[dp_column_index]

        self.save_df()

    def apply_solute(self):
        '''
            A function that saves solute features to the dataframe

            Parameters:
                None

            Output:
                None
        '''
        
        match_columns = [
            "SOLUTE_PARAM_E_GC",
            "SOLUTE_PARAM_S_GC",
            "SOLUTE_PARAM_A_GC",
            "SOLUTE_PARAM_B_GC",
            "SOLUTE_PARAM_L_GC",
            "SOLUTE_PARAM_V_GC",
            "SOLUTE_PARAM_E_ML",
            "SOLUTE_PARAM_S_ML",
            "SOLUTE_PARAM_A_ML",
            "SOLUTE_PARAM_B_ML",
            "SOLUTE_PARAM_L_ML",
            "SOLUTE_PARAM_V_ML",
        ]

        for col in match_columns:
            self.df[col] = 0
        
        for index, row in self.df.iterrows():
            # Check if row has all zeros
            all_zeros = row[match_columns].eq(0).all().all()
            all_nans = row[match_columns].eq(np.nan).all().all()

            # Make sure data is not already there
            if all_zeros or all_nans:
                # Look for matches in archive
                lookup_val = row['Canonical SMILES']
                if lookup_val in self.archive_df['Canonical SMILES'].values:
                    # Loop through all columns with missing values
                    for col_name in match_columns:
                        corresponding_i = self.archive_df.loc[
                            self.archive_df["Canonical SMILES"] == lookup_val, col_name
                        ].values

                        self.df.loc[index, col_name] = corresponding_i[0]

                else:
                    print(f"no match for {lookup_val}. strange")
    
        # make new storage dataframe for unique solutes (monomers)
        unique_smiles = list(set(self.df["Canonical SMILES"].tolist()))
        solute_db = pd.DataFrame({"Canonical SMILES": unique_smiles})

        # add the RMG columns
        columns_to_add = [
            "SOLUTE_PARAM_E_GC",
            "SOLUTE_PARAM_S_GC",
            "SOLUTE_PARAM_A_GC",
            "SOLUTE_PARAM_B_GC",
            "SOLUTE_PARAM_L_GC",
            "SOLUTE_PARAM_V_GC",
            "SOLUTE_PARAM_E_ML",
            "SOLUTE_PARAM_S_ML",
            "SOLUTE_PARAM_A_ML",
            "SOLUTE_PARAM_B_ML",
            "SOLUTE_PARAM_L_ML",
            "SOLUTE_PARAM_V_ML",
        ]
        solute_db = pd.concat(
            [solute_db, pd.DataFrame(0, columns=columns_to_add, index=solute_db.index)], axis=1
        )

        indices = [i for i in range(len(columns_to_add))]

        # loop through solute DB
        for index, row in solute_db.iterrows():
            # check if column has all zeros
            all_zeros = row[columns_to_add].eq(0).all().all()
            all_nans = row[columns_to_add].eq(np.nan).all().all()

            # check to see if data already downloaded
            if all_zeros or all_nans:
                web_vals = self.get_solute_params(row)
                zip_index_cols = zip(indices, columns_to_add)

                for pair in zip_index_cols:
                    dp_column_index, dp_column_name = pair

                    # Update columns starting from 'DP_0'
                    solute_db.at[index, dp_column_name] = web_vals[dp_column_index]

        solute_db.to_csv(path + "2c_solute_parameters.csv")
        # Note: [Si]-containing atoms will have a nan

        columns = [
            "SOLUTE_PARAM_E_GC",
            "SOLUTE_PARAM_S_GC",
            "SOLUTE_PARAM_A_GC",
            "SOLUTE_PARAM_B_GC",
            "SOLUTE_PARAM_L_GC",
            "SOLUTE_PARAM_V_GC",
            "SOLUTE_PARAM_E_ML",
            "SOLUTE_PARAM_S_ML",
            "SOLUTE_PARAM_A_ML",
            "SOLUTE_PARAM_B_ML",
            "SOLUTE_PARAM_L_ML",
            "SOLUTE_PARAM_V_ML",
        ]

        for col in columns:
            self.df[col] = 0

        # now loop through df, find match between df and solute_db, and copy values for the columns
        for index, row in self.df.iterrows():
            # Save canonical SMILES value
            canonical_smiles_value = row["Canonical SMILES"]

            # Check if the value appears in the 'Canonical SMILES' column of archive_df
            if canonical_smiles_value in solute_db["Canonical SMILES"].values:
                # Loop through all columns with missing values
                for col_name in columns:
                    corresponding_i = solute_db.loc[
                        solute_db["Canonical SMILES"] == canonical_smiles_value, col_name
                    ].values

                    # Check if there is a corresponding DP_i value
                    self.df.loc[index, col_name] = corresponding_i[0]
            else:
                print(f"{canonical_smiles_value} does not appear in solute_db")

        self.save_df()

    def apply_rdkit_descriptors(self):
        '''
            A function that saves rdkit descriptor features to the dataframe

            Parameters:
                None

            Output:
                None
        '''
        self.rdkit_descriptors()
        self.save_df()

    def apply_steric(self):
        '''
            A function that saves steric features to the dataframe

            Parameters:
                None

            Output:
                None
        '''

        self.df["REPEAT_UNIT"] = ""

        # loop through df for any missing values, then pull from archive_df
        for idx, row in self.df.iterrows():
            monomer = row["DP_0"]

            # First, try to match in archive
            try:
                idx_match = self.archive_df.index[self.archive_df["DP_0"] == monomer].tolist()[0]
                self.df.at[idx, "REPEAT_UNIT"] = self.archive_df.iloc[idx_match]["REPEAT_UNIT"]
            except:
                print(f"{monomer} does not have a match in archive. Using rxn to calculate repeat unit.")

                category = row["BASE_Category"]
                monomer = MolFromSmiles(row["DP_0"])

                # Then use operator
                try:
                    rxn = repeat_rxn_dict[category]

                    if category == "ROMP":
                        temp = MolFromSmiles("*=*")
                    else:
                        temp = MolFromSmiles("**")

                    prod = rxn.RunReactants((monomer, temp))[0][0]
                    self.df.at[idx, "REPEAT_UNIT"] = MolToSmiles(prod)
                except:
                    print('No RU for', row["DP_0"])

        self.save_df()

        bb_lens = []
        ratios = []

        # retrieve backbone length and ratio
        for _, row in self.df.iterrows():
            try:
                
                bb_len, ratio = self.get_backbone_length_and_ratio(row["REPEAT_UNIT"])

                bb_lens.append(bb_len)
                ratios.append(ratio)
            except:
                bb_lens.append(np.nan)
                ratios.append(np.nan)

        self.df["STERICS_Backbone_Length"] = bb_lens
        self.df["STERICS_Side_Chain_Ratio"] = ratios

        #final save
        self.save_df()

# if __name__ == "__main__":
#     #filepath to the folder that holds all the dataset .csv files. Should likely end in ".../0_data_sets/"
#     path = "/Users/hunter/Downloads/BROADBELT LAB/TC_ML_forked/dH_prediction/0_data_sets/"
#     #maximum chain length you wish to investigate
#     degree_of_polymerization = 5

#     # add the features
#     featurize = AddFeatures(path,degree_of_polymerization)
#     featurize.main()





