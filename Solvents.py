import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from rdkit import Chem
import os

# https://depts.washington.edu/eooptic/linkfiles/dielectric_chart%5B1%5D.pdf

class SolventData():
    def __init__(self,infile_path):
        self.infile_path = infile_path

        filename = os.path.splitext(os.path.basename(infile_path))[0]
        directory_path = os.path.dirname(infile_path)

        updated_filename = "updated_" +  filename + ".csv"
        dielectric_filepath = directory_path + "/dielectric.csv"

        self.updated_path = directory_path + "/" + updated_filename

        # Read df
        self.df = pd.read_csv(self.infile_path, index_col=0, encoding="utf-8")
        self.dielectric_df = pd.read_csv(dielectric_filepath, index_col=0, encoding="utf-8")
        self.update_df = self.df

    def main(self):
        '''
            a function that executes retrieving data from the website for rows that have experimental data missing. 

            Parameters:
                None
            Output:
                None        
        '''
        # find the columns with cells that need to be imputed
        for _, row in self.df.iterrows():
            assert pd.notna(row["Solvent"]) 
            if pd.isna(row["SOLV_PARAM_s_g"]): # check if expeirmental data is missing
                # If empty, experimental solvent data has not been filled in yet
                to_match = self.solvationHelper(row["Solvent"])
                if pd.notna(to_match):
                    # if experimental data was sraped, remove the current row of the solvent that has np.nan values
                    self.update_df = self.update_df[self.update_df["Solvent"] != row["Solvent"]]
                    
                    # add the row of experimental data for the solvent
                    self.update_df = pd.concat([self.update_df, pd.DataFrame([to_match])], ignore_index=True)
                    
                    # Save the DataFrame
                    self.saveDf()

    def solvationHelper(self, solvent_name):
        '''
            a function that retrieves solvent experimental data from a website

            Parameters:
                solvent_name (str): the name of the solvent

            Output:
                (dictionary): collection of experimental data on the solvent obtained from rmg.mit
        '''

        # set variables
        base_url = "https://rmg.mit.edu/database/solvation/libraries/solvent/"

        # make request
        response = requests.get(base_url)
        if response.status_code != 200:  # catch errors
            print(f"Could not access page when searching for solvent {solvent_name}")
            return np.nan
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the link where the text contains "Result #1"
        link = soup.find('a', text=re.compile(re.escape(solvent_name), re.IGNORECASE))

        if link:
            # Get the URL of the link
            link_url = link['href']
            full = "https://rmg.mit.edu" + link_url
            
            # fetch the linked page
            linked_response = requests.get(full)
            
            # Parse the HTML content
            soup = BeautifulSoup(linked_response.text, 'html.parser')
            
            canonical_smiles = np.nan

            # Find the <h2> tag with the specific text
            target_header = soup.find('h2', text='Molecule SMILES')
            if target_header:
                next_p = target_header.find_next_sibling('p')
                if next_p:
                    smiles = next_p.get_text(strip=True)
                    # Convert the SMILES string to a molecule object
                    molecule = Chem.MolFromSmiles(smiles)

                    # Convert the molecule object back to canonical SMILES
                    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)


            table = pd.read_html(linked_response.content)
            df = table[0]

            columns = ["s_g", "b_g", "e_g","l_g","a_g","c_g","Associated Solvation Free Energy Mean Absolute Error", "s_h", "b_h", "e_h", "l_h","a_h","c_h","Associated Solvation Enthalpy Mean Absolute Error","Solvent Viscosity μ at 298 K","ε"]
            val_list = [np.nan] * len(columns)

            for i in range(len(columns)):
                col_name = columns[i]
                raw_val = df.loc[df[0] == col_name][2]
                if raw_val.empty or pd.isna(raw_val.iloc[0]):
                    val_list[i] = np.nan
                else:
                    val_list[i] = float(raw_val.iloc[0].split()[0])

            if pd.isna(canonical_smiles):
                print(f"Smiles string for {solvent_name} could not be found. You must find manually.")

            if pd.isna(val_list[15]):
                # check to see if the dielectric constant is in dielectric.csv if it was not scraped from the website
                self.dielectric_df
                
            solvent_data = {
                "Solvent": solvent_name,
                "Solvent_SMILES": canonical_smiles,
                "Solvent_SMILES_2": np.nan,
                "SOLV_PARAM_s_g": val_list[0],
                "SOLV_PARAM_b_g": val_list[1],
                "SOLV_PARAM_e_g": val_list[2],
                "SOLV_PARAM_l_g": val_list[3],
                "SOLV_PARAM_a_g": val_list[4],
                "SOLV_PARAM_c_g": val_list[5],
                "SOLV_PARAM_abraham err": val_list[6], 
                "SOLV_PARAM_s_h": val_list[7],
                "SOLV_PARAM_b_h": val_list[8],
            	"SOLV_PARAM_e_h": val_list[9],
                "SOLV_PARAM_l_h": val_list[10],
                "SOLV_PARAM_a_h": val_list[11],
            	"SOLV_PARAM_c_h": val_list[12],
                "SOLV_PARAM_mintz err": val_list[13],
                "SOLV_PARAM_visc at 298 K (cP)": val_list[14],
                "SOLV_PARAM_dielectric constant": val_list[15],
            }
            # print(solvent_data)
        else:
            print(f"No experimental solvent data found for {solvent_name}")
            return np.nan


        return solvent_data
    
    def saveDf(self):
        '''
            a function that saves a pandas dataframe to a csv

            Parameters:
                None

            Output:
                saves a .csv file to the same folder that the original solvent file was in. The new files starts with updated_
        '''

        self.update_df.to_csv(self.updated_path, index =False)