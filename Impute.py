'''
This python file imputes necessary data. Usually the following needs to be imputed:
   
    1. Solute Parameters (missing parameterization)
    2. Dipole Moment (missing parameterization)
    3. pep (pgthermo dH)
    NOTE: Solvent and SOLV_ columns will have np.nan values however do NOT impute for these columns
    as not every monomer has a solvent and not all solvents have experimental data 

Other helpful functions available for use:
    printColumnsWithMissing(pd dataframe)

'''
import pandas as pd

import numpy as np

import os
 
from rdkit import Chem
from rdkit.Chem import AllChem

class ImputeData:
    def __init__(self,infile_path):
        self.infile_path = infile_path
        filename = os.path.splitext(os.path.basename(self.infile_path))[0]
        new_file_name = "imputed_" +  filename + ".csv"
        new_file_name = "cleaned_" +  new_file_name 

        self.final_path = "data/" + new_file_name
        self.cleaned_path = "data/" + new_file_name

        # Read df
        self.df = pd.read_csv(self.infile_path, index_col=0, encoding="utf-8")
        self.missing = self.printColumnsWithMissing(self.df)

    def main(self):
        # find the columns with cells that need to be imputed
        self.imputeMissingParam()
        self.saveDf(self.final_path)
        print("Imputing Complete")

        self.oneHotEncode()
        print("One-hot Encoding Complete")
        self.cleanRDKIT()
        print("RDKIT Parameter Cleaning Complete")
        self.saveDf(self.cleaned_path)

    def imputeMissingParam(self):
        # Columns that need to be imputed that are missing parameterizations
        impute_columns_nan = [col for col in self.df.columns if "SOLUTE_PARAM_" in col and col in self.missing] + [col for col in self.df.columns if "DIPOLE_" in col and col in self.missing] + ["PGTHERMO_PEP_ (kcal/mol)"]

        # imputes data that is missing due to missing paramaterizations
        # Loop over each row in the dataframe
        for idx, row in self.df.iterrows():
            # Loop through all columns you want to impute that have NANs
            for col in impute_columns_nan:
                # If nan
                if pd.isna(row[col]):
                    # Find the most similar value and replace   
                    best_val = 0
                    best_score = 0
                    for idx2, row2 in self.df.iterrows():  # Loop through df
                        if not pd.isna(row2[col]):  # if not empty, then calculate score
                            score = self.calculateTanimotoSimilarity(row["DP_0"], row2["DP_0"])
                            if score > best_score:
                                best_score = score
                                best_val = row2[col]

                    # Set the best value if the score ever goes higher than 0
                    if best_score > 0:
                        self.df.loc[idx, col] = best_val
        
    def printColumnsWithMissing(self, pdframe):
        # Find columns with missing values and exclude those for solvent name, smiles and solvent experimental parameters
        missing = [
            col for col in pdframe.columns[self.df.isnull().any()]
            if "SOLV" not in col and "Solvent" not in col and "solvent" not in col
        ]
        if len(missing) == 0:
            print("No imputing needs to be done")
        else:
            for col_name in missing:
                print(col_name)
        return missing

    # What is left to do:
    def oneHotEncode(self):
        # one-hot encode state and category columns
        encode_columns = [col for col in self.df.columns if "BASE_" in col and col != "BASE_State"]
        self.df = pd.get_dummies(self.df, columns=encode_columns)

        # self.df.drop(columns=['BASE_State'], inplace=True)

    def cleanRDKIT(self):
        # Extract the filename without the directory path and without the extension
        threshold = 0.90 * len(self.df)
        columns_to_remove = []
        for column in self.df.columns:
            if (
                column.startswith("RDKIT_fr_")
                and (self.df[column] == 0).sum() >= threshold
            ):
                columns_to_remove.append(column)
        self.df = self.df.drop(columns=columns_to_remove)
        print(f"Columns removed: {columns_to_remove}")


    # ---------- HELPER FUNCTIONS ---------- #
    def saveDf(self, path):
        self.df.to_csv(self.final_path, encoding="utf-8")

    # Function to calculate Tanimoto similarity between two SMILES
    def calculateTanimotoSimilarity(self, smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is not None and mol2 is not None:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
            return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
        else:
            return 0.0