'''
This python file makes changes to data to prepare it for use in ML models. The three main
preparations are IMPUTING values that are np.nan, ONE-HOT ENCODING categorical values, and CLEAN
RDKIT columns by removing the ones where more than 90% of the values are the same 

Usually the following needs to be imputed:
   
    1. Solute Parameters (missing parameterization)
    2. Dipole Moment (missing parameterization)
    3. pep 
    NOTE: Solvent and SOLV_ columns will have np.nan values however do NOT impute for these columns
    as not every monomer has a solvent and not all solvents have experimental data 

Call main(infile_path, target) to run all of the prepatory steps with one call. 

'''
import pandas as pd

import numpy as np

import os
 
from rdkit import Chem
from rdkit.Chem import AllChem

# -------------- HELPER FUNCTIONS -------------- #

def calculateTanimotoSimilarity(smiles1, smiles2):
    '''
        a function that calculates the tanimoto similarity between two molecules

        Parameters:
            smiles1 (str): the canonical smiles string of the molecule 
            smiles2 (str): the canonical smiles string of the molecule 

        Output:
            (float): tanimoto similarity value
    '''

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is not None and mol2 is not None:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        return 0.0
        
def getColumnsWithMissing(infile_path):
    '''
        a function that finds and returns what columns have np.nan values

        Parameters:
            infile_path (str): the path to the dataset that you want to find missing columns of

        Output:
            (list): a list of the column names of columns that have np.nan values
    '''

    df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")

    # Find columns with missing values and exclude those for solvent name, smiles and solvent experimental parameters
    missing = [
        col for col in df.columns[df.isnull().any()]
        if "SOLV" not in col and "Solvent" not in col and "solvent" not in col and "Temp (ºC)" not in col
    ]
    if len(missing) == 0:
        print("No imputing needs to be done")
    # else:
    #     for col_name in missing:
    #         print(col_name)
    return missing

# -------------- MAIN FUNCTIONS -------------- #

def impute(infile_path):
    '''
        a function that finds and returns what columns have np.nan values

        Parameters:
            infile_path (str): the path to the dataset that you want to impute

        Output:
            saves a .csv file starting with imputed_
    '''
    df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")
    missing = getColumnsWithMissing(infile_path)

    filename = os.path.splitext(os.path.basename(infile_path))[0]
    directory_path = os.path.dirname(infile_path)
    imputed_file_name = "imputed_" +  filename + ".csv"
    final_path = directory_path + "/" + imputed_file_name


    # Columns that need to be imputed that are missing parameterizations
    impute_columns_nan= np.nan

    impute_columns_nan = [col for col in df.columns if "SOLUTE_PARAM_" in col and col in missing] + [col for col in df.columns if "DIPOLE_" in col and col in missing] + ["PGTHERMO_PEP_ (kcal/mol)"]

    # imputes data that is missing due to missing paramaterizations
    # Loop over each row in the dataframe
    for idx, row in df.iterrows():
        # Loop through all columns you want to impute that have NANs
        for col in impute_columns_nan:
            # If nan
            if pd.isna(row[col]):
                # Find the most similar value and replace   
                best_val = 0
                best_score = 0
                for idx2, row2 in df.iterrows():  # Loop through df
                    if not pd.isna(row2[col]):  # if not empty, then calculate score
                        score = calculateTanimotoSimilarity(row["DP_0"], row2["DP_0"])
                        if score > best_score:
                            best_score = score
                            best_val = row2[col]

                # Set the best value if the score ever goes higher than 0
                if best_score > 0:
                    df.loc[idx, col] = best_val
    df.to_csv(final_path, encoding="utf-8")
    print("Imputing Complete")

    return final_path

# What is left to do:
def oneHotEncode(infile_path):
    '''
        a function that replaces values that are categorical with binary values that work in regression models

        Parameters:
            infile_path (str): the path to the dataset that you want to one-hot encode

        Output:
            saves a .csv file starting with encoded_
    '''

    df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")
    filename = os.path.splitext(os.path.basename(infile_path))[0]
    directory_path = os.path.dirname(infile_path)
    encoded_file_name = "encoded_" +  filename + ".csv"
    encoded_path = directory_path + "/" + encoded_file_name

    # one-hot encode state and category columns
    encode_columns = [col for col in df.columns if "BASE_" in col and col != "BASE_State"]
    df = pd.get_dummies(df, columns=encode_columns)
    df.to_csv(encoded_path, encoding="utf-8")

    # self.df.drop(columns=['BASE_State'], inplace=True)
    print("One-hot Encoding Complete")
    return encoded_path

def cleanRDKIT(infile_path):
    '''
        a function that removes RDKIT columns where 90% or more of the values are the same

        Parameters:
            infile_path (str): the path to the dataset that you want to clean 

        Output:
            saves a .csv file starting with cleaned_
    '''
    df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")
    filename = os.path.splitext(os.path.basename(infile_path))[0]
    directory_path = os.path.dirname(infile_path)
    cleaned_file_name = "cleaned_" +  filename + ".csv"
    cleaned_path = directory_path + "/" + cleaned_file_name

    # Extract the filename without the directory path and without the extension
    threshold = 0.90 * len(df)
    columns_to_remove = []
    for column in df.columns:
        if (
            column.startswith("RDKIT_fr_")
            and (df[column] == 0).sum() >= threshold
        ):
            columns_to_remove.append(column)
    df = df.drop(columns=columns_to_remove)
    print(f"Columns removed: {columns_to_remove}")
    df.to_csv(cleaned_path, encoding="utf-8")
    print("RDKIT Feature Cleaning Complete")
    return cleaned_path

def main(infile_path):
    '''
        a function that calls all the functions (impute, oneHotEncode, conductAll) completing all of the data
        preprocessing that must be done to prepare for use in ML regression models. 

        Parameters:
            infile_path (str): the path to the dataset that you want to impute, one-hot encode and clean RDKIT columns

        Output:
            saves a .csv file starting with cleaned_encoded_imputed
    '''

    filename = os.path.splitext(os.path.basename(infile_path))[0]
    directory_path = os.path.dirname(infile_path)

    imputed_file_name = "imputed_" +  filename + ".csv"
    encoded_file_name = "encoded_" +  imputed_file_name

    imputed_path = directory_path + "/" + imputed_file_name
    encoded_path = directory_path + "/" + encoded_file_name

    # find the columns with cells that need to be imputed
    impute(infile_path)

    oneHotEncode(imputed_path)
    
    cleanRDKIT(encoded_path)

    print("ALL OPERATIONS COMPLETE :)")
