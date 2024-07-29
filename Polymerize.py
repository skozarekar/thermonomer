import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import re

class Polymerization:
    '''
        class Polymerization: 
            A class that polymerizes monomers in a dataset up to degree of polymerization chosen by user. Data is
            saved in a new .csv file to which solvent data is added as well. Types of polymerizations available here: ROP (covers O, N, S), 
            ROMP, vinyl, ionic, cationic, cyclic

        Parameters for initiation: 
            path (str): the directory path to the directory that contains all of the user's datasets (ie. ends in /0_data_sets/)
            dp (int): degree of polymerization. The amount of times the user wants the molecule to be polymerized

        Output:
            creates a new csv file containing polymer and solvent data (ie. titled "2a_polymerized_with_solvents.csv")
    '''

    def __init__(self, path, degree_of_polymerization):
        '''
            NOTE: Make sure the datasets are named corectly before you run. 

        '''
        # Define dataset paths
        self.infile_path = path + "1e_unfeaturized_data_clean.csv"
        self.solvent_path = path + "2_solvents.csv"
        self.final_path = path + "2a_polymerized_with_solvents.csv"

        # This path will be used to fill in values that were done manually such as monomers that fall under base_state 'misc'
        self.archive_path = path + "2_archive_static.csv"

        self.degree_of_polymerization = degree_of_polymerization
        self.df = pd.read_csv(self.infile_path, index_col=0, encoding="utf-8")

        # Initiate the variables that will be used for polymerization
        self.dp = degree_of_polymerization
        self.monomer_smiles = ""
        self.initiator_smiles = ""
        self.initiation_rxn_str = ""
        self.propagation_rxn_str = ""
        self.rxn_mechanism = ""


        # Dictionary covers various configurations of Ring-Opening Polymerization (ROP)
        # [substructure, initiation rxn, propagation rxn]
        self.ROP_dict = {
            "ROP_N": [
                "[O;X1]=[C;X3]-[N]",
                    "([C:1](=[O:2])[N:3]@[C:4]).[Ge:5]>>([C:1](=[O:2])[Ge:5].[N:3][C:4])"
                ,
                    "[Ge]-[CX3:3]=[OX1:4].[#6X4:1]-[N:2]>>[#6X4:1]-[N:2]-[CX3:3]=[OX1:4]"
                    ,
            ],
            "ROP_O": [
                "[O;X1]=[C;X3]-[O]",
                    "([C:1](=[O:2])@[O:3]@[C:4]).[Ge:5]>>([C:1](=[O:2])[Ge:5].[O:3][C:4])"
                ,
                    "[#6X4:1]-[OX2H:2].[Ge]-[CX3:3]=[OX1:4]>>[#6X4:1]-[O:2]-[CX3:3]=[OX1:4]"
                    ,
            ],
            "ROP_S_A": [
                "[O;X1]=[C;X3]-[S]",
                    "([C:1](=[O:2])[S:3]@[C:4]).[Ge:5]>>([C:1](=[O:2])[Ge:5].[S:3][C:4])"
                ,
                    "[Ge]-[C:3]=[OX1:4].[#6X4:1]-[S:2]>>[#6X4:1]-[S:2]-[C:3]=[OX1:4]"
                    ,
            ],
            "ROP_S_B": [
                "[O;X1]=[S;X3]-[S]",
                    "([S:1](=[O:2])[S:3]@[C:4]).[Ge:5]>>([S:1](=[O:2])[Ge:5].[S:3][C:4])"
                ,
                    "[Ge]-[S:3]=[OX1:4].[#6X4:1]-[S:2]>>[#6X4:1]-[S:2]-[S:3]=[OX1:4]"
                    ,
            ],
            "ROP_S_C": [
                "[S;X1]=[C;X3]-[S]",
                    "([C:1](=[S:2])[S:3]@[C:4]).[Ge:5]>>([C:1](=[S:2])[Ge:5].[S:3][C:4])"
                ,
                    "[Ge]-[C:3]=[SX1:4].[#6X4:1]-[S:2]>>[#6X4:1]-[S:2]-[C:3]=[SX1:4]"
                    ,
            ],
            "ROP_S_D": [
                "[O;X1]=[S]-[O]",
                    "([S:1](=[O:2])[O:3]@[C:4]).[Ge:5]>>([S:1](=[O:2])[Ge:5].[O:3][C:4])"
                ,
                    "[Ge]-[S:3]=[OX1:4].[#6X4:1]-[O:2]>>[#6X4:1]-[O:2]-[S:3]=[OX1:4]"
                    ,
            ],
            "ROP_S_E": [
                "[S;X1]=[C;X3]-[O]",
                    "([C:1](=[S:2])[O:3]@[C:4]).[Ge:5]>>([C:1](=[S:2])[Ge:5].[O:3][C:4])"
                ,
                    "[Ge]-[C:3]=[SX1:4].[#6X4:1]-[O:2]>>[#6X4:1]-[O:2]-[C:3]=[SX1:4]"
                    ,
            ],
        }
        
        # Dictionary that covers various types of polymerization 
        # [initiation rxn, propagation rxn]
        self.type = {
            "ROMP": [
                "([C:3]=[C:4]).[CH2:1]=[CH2:2]>>([C:1]=[C:3].[C:4]=[C:2])",
                "[C:1]=[C:2].([C:3]=[C:4])>>([C:1]=[C:3].[C:4]=[C:2])"
            ],
            # ROP is empty because it is handled by self.ROP_dict
            "ROP": ["", ""],
            # misc is empty because it is filled in later with archive data
            "misc":["",""],
            "vinyl": [
                "[C:1]=[C:2].[Ge:3]>>[C:1]-[C:2]-[Ge:3]",
                "[C:1]-[Ge:2].[C:3]=[C:4]>>[C:1]-[C:3]-[C:4]-[Ge:2]"
            ],
            "ionic": [
                #initiation
                [
                    #if "S" or "s" is present in smiles:
                    "[CX3:1](=[S:2]).[S:3]>>[S:3]-[C:1]-[S:2]", 
                    # else:
                    "[CX3:1](=[O:2]).[Ge:3]>>[Ge:3]-[C:1]-[O:2]"
                ],
                # propagation
                [
                    #if "S" or "s" is present in smiles:
                    "[SH:3].[CX3:1](=[S:2])>>[S:3]-[C:1]-[S:2]",
                    # else:
                    "[Ge:3].[CX3:1](=[O:2])>>[O:3]-[C:1]-[Ge:2]"
                ]
            ],
            "cyclic": [
                    "([CH2:1][CH2:2]).[Ge:3]>>([C:2][Ge:3].[C:1][OH])",
                   "[Ge][C:1].[OH]-[C:2]>>[C:1][C:2]"
            ],
            "cationic": [ 
                # initiation
                [
                    # if c in smiles:
                    "([CH2:1][*:2]).[Ge:3]>>([*:2][Ge:3].[C:1][OH])", 
                    # else:
                    "([*:1]@[*:2]).[Ge:3]>>([*:2][Ge:3].[*:1][OH])"
                ],
                #propagation
                [
                    # if c in smiles:
                    "[Ge][*:1].[OH]-[C:2]>>[*:1][C:2]",
                    # else
                    "[Ge][*:1].[OH]-[*:2]>>[*:1][*:2]"
                ]
            ],
        }

    def main(self):
        '''
            When polymerization.main() is called, polymerizes every monomer in the dataset to the user's requested
            degree of polymerization (dp). Additionally, adds solvent data to the dataset. 

            Parameters: 
                None
            
            Output:
                None
        '''
                
        self.addSolventCol()
        
        # Drop old columns in case this has been run before
        self.df = self.df[self.df.columns.drop(list(self.df.filter(regex="DP_")))]

        col_names = ["DP_" + str(i) for i in range(self.degree_of_polymerization + 1)]

        new_columns_df = pd.DataFrame("", columns=col_names, index=self.df.index)
        self.df = pd.concat([self.df, new_columns_df], axis=1)

        num_cols = -1 * (self.degree_of_polymerization + 1)

        # Apply polymerize data
        self.df[self.df.columns[num_cols:]] = self.df.apply(func=self.makePolymers, axis=1)

        # Fill in missing polymerizations with old data
        self.unavailablePolymers()

        # save final dataframe
        self.df.to_csv(self.final_path, encoding='utf-8-sig')

        print("\nPolymerization Complete :)")

    def addSolventCol(self):
        '''
            A function that adds solvent data to the new data csv that will have the polymerized smiles

            Parameters: 
                None
            
            Output:
                None
        '''

        # Dataframes
        solvent_df = pd.read_csv(
            self.solvent_path,
            index_col=0,
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

        if "SOLV_PARAM_s_g" in self.df.columns: 
            print("Solvent Columns have already been added")
        else:
            # Merge on solvents (make sure to run just once!)
            self.df = pd.merge(self.df, solvent_df, on="Solvent", how="left")

    def makePolymers(self, row):
        '''
            A function that defines the initiation and propagation reaction depending on the type of polymerization
            the monomer undergoes. In most cases, Ge is used as the initiator for easy replacement with carbon later on. Calls
            the functino that polymerizes the monomer and returns its output

            Parameters: 
                row: row from the datatable
            
            Output:
                (pd.Series): the array of polymerized smile strings. Returns an empty array if the monomer could not be polymerized
        '''

        # Create molecules
        self.rxn_mechanism = row["BASE_Category"] # type of polymerization
        self.monomer_smiles = row["Canonical SMILES"]

        try:
            # create molecules
            self.initiator_smiles = "[Ge]"

            if self.rxn_mechanism == "ROMP":
                self.initiator_smiles = "C=C"

            # create reaction steps
            self.initiation_rxn_str = self.type[self.rxn_mechanism][0]
            self.propagation_rxn_str = self.type[self.rxn_mechanism][1]

            # Adjust initiation and propagation for reactions that depend on certain atoms in smile string
            if self.rxn_mechanism == "ionic":
                if "S" in self.monomer_smiles or "s" in self.monomer_smiles:
                    self.initiator_smiles = "S"
                    self.initiation_rxn_str = self.initiation_rxn_str[0]
                    self.propagation_rxn_str = self.propagation_rxn_str[0]
                else:
                    self.initiation_rxn_str = self.initiation_rxn_str[1]
                    self.propagation_rxn_str = self.propagation_rxn_str[1]
 
            if self.rxn_mechanism == "cationic":
                if ("c" in self.monomer_smiles):  # account for any with benzene rings... deal with robustness later
                    self.initiation_rxn_str = self.initiation_rxn_str[0]
                    self.propagation_rxn_str = self.propagation_rxn_str[0]
                else:
                    self.initiation_rxn_str = self.initiation_rxn_str[1]
                    self.propagation_rxn_str = self.propagation_rxn_str[1]

            # Adjust initiation and propagation for all_ROP polymerizations
            if self.rxn_mechanism == "ROP":
                self.getROPSteps()
                if self.initiation_rxn_str == "" or self.propagation_rxn_str == "": 
                    print(
                f"No matching substructure found. {self.monomer_smiles} cannot be polymerized using any kind of ROP."
            )
                    return pd.Series([np.nan] * (self.dp + 1))

            return self.polymerize()
        
        except:
            print(f"Polymerization mechanism {self.rxn_mechanism} not found for {self.monomer_smiles}.")
            return pd.Series([np.nan] * (self.dp + 1))

    def getROPSteps(self):
        '''
            Defines the initiation and propagation steps for ROP monomers. 

            Parameters:
                None
            
            Output:
                initiation (str): the initiation reaction string
                propagation (str): the propagation reaction string
                reaction_mechanism (str): a more descriptive title for the ROP that the monomer undergoes
        '''

        monomer = Chem.MolFromSmiles(self.monomer_smiles)
        self.initiation_rxn_str = ""
        self.propagation_rxn_str = ""
        
        # Search for substructure match
        for key, val in self.ROP_dict.items():
            temp_match = monomer.GetSubstructMatch(Chem.MolFromSmarts(val[0]))

            # If match found, set variables and stop searching
            if len(temp_match) > 0:
                self.rxn_mechanism = key
                self.initiation_rxn_str = val[1]
                self.propagation_rxn_str = val[2]

                break
        
    def replaceElement(self,molecule):
        '''
            Replaces the endgroup with a carbon atom

            Parameters: 
                molecule (str): the smiles string for the molecule to be edited
            
            Output:
                (str): the new molecule with carbon endgroup
        '''

        smile_to_change = Chem.MolToSmiles(molecule)
        # initiator ie. "[Ge]" or "C=C"
        element_to_replace = self.initiator_smiles.replace('[', '').replace(']', '')

        pattern = fr'\[{element_to_replace}[^\]]*\]'

        # change the end group to carbon
        return (re.sub(pattern, "C", smile_to_change))

    def polymerize(self):
        '''
            Executes the initiation and propagation reactions. Polymerizes the molecule up to dp

            Parameters: 
                None            

            Output:
                (pd.Series): an array of the polymerized molecules
        '''
                
        monomer = Chem.MolFromSmiles(self.monomer_smiles)
        initiator = Chem.MolFromSmiles(self.initiator_smiles)

        # dp must be at least 0
        poly_list = [self.monomer_smiles]

        initiation = rdChemReactions.ReactionFromSmarts(self.initiation_rxn_str)
        propagation = rdChemReactions.ReactionFromSmarts(self.propagation_rxn_str)

        # If DP is 0, return the monomer as a single-element Series
        if self.dp == 0: 
            return poly_list
        
        activated_monomer = np.nan
        
        # INITIATION
        # Open the ring to start polymerization (DP = 1)
        try:
            activated_monomer = initiation.RunReactants((monomer, initiator))[0][0]
        except IndexError:
            print(f"{self.monomer_smiles} cannot be polymerized using {self.rxn_mechanism}.")
            return pd.Series([np.nan] * (self.dp + 1))
                
        Chem.SanitizeMol(activated_monomer)
        if self.initiator_smiles != "C=C":
            to_add = self.replaceElement(activated_monomer)
            # change the end group to carbon
            poly_list.append(to_add)

        else:
            # end group is already carbon
            poly_list.append(Chem.MolToSmiles(activated_monomer))

        if self.dp == 1:
            return pd.Series(poly_list)
        
        # dp > 1, iteratively polymerize the monomer
        polymer = activated_monomer
        for degree in range(self.dp - 1):
            if self.rxn_mechanism == "vinyl" or self.rxn_mechanism == "ionic":
                polymer = propagation.RunReactants((polymer, monomer))[0][0]
            else:
                polymer = propagation.RunReactants((polymer, activated_monomer))[0][0]
            Chem.SanitizeMol(polymer)  # save this one for the next iteration
            
            if self.initiator_smiles != "C=C":
                to_add = self.replaceElement(polymer)
                # change the end group to carbon
                poly_list.append(to_add)

            else:
                poly_list.append(Chem.MolToSmiles(polymer))

        return pd.Series(poly_list)
    
    def unavailablePolymers(self):
        '''
            Fills in the data for polymers that are missing (ie. the ones labeled misc or that could not be polymerized) with
            data from archive. 

            Parameters: 
                None
            
            Output:
                None
        '''
        archive_df = pd.read_csv(self.archive_path, index_col = 0, encoding="utf-8")
        canonical = []
        for smi in archive_df["Monomer_SMILES"].tolist():
            mol = Chem.MolFromSmiles(smi)
            canonical.append(Chem.MolToSmiles(mol))
        archive_df["Canonical SMILES"] = canonical
        num_unknown = 0

        # Loop through the dataframe, look for matches in the archive dataframe if DP_x pieces are missing
        for index, row in self.df.iterrows():
            # Check if the monomer (DP_0) is blank

            if pd.isnull(row["DP_0"]):
                num_unknown += 1
                # Save canonical SMILES value
                canonical_smiles_value = row["Canonical SMILES"]

                # Check if the value appears in the 'Canonical SMILES' column of archive_df
                if canonical_smiles_value in archive_df["Canonical SMILES"].values:
                    # Loop through all columns with missing values
                    for i in range(self.degree_of_polymerization + 1):
                        dp_column_name = f"DP_{i}"

                        # Get the corresponding DP_i value from archive_df
                        corresponding_dp_i = archive_df.loc[
                            archive_df["Canonical SMILES"] == canonical_smiles_value,
                            dp_column_name,
                        ].values

                        # self.df[dp_column_name] = self.df[dp_column_name].astype(object)

                        # Check if there is a corresponding DP_i value
                        self.df.loc[index, dp_column_name] = corresponding_dp_i[0]
                else:
                    print(f"{canonical_smiles_value} does not appear in archive_df")



# if __name__ == "__main__":
#     # Path to folder with data sets (should end in something simlar to ''/0_data_sets/')
#     path = "/Users/hunter/Downloads/BROADBELT LAB/TC_ML_forked/dH_prediction/0_data_sets/"

#     # Set max degree of polymerization
#     degree_of_polymerization = 5

#     # add the features
#     polymerize = polymerization(path, degree_of_polymerization)
#     polymerize.main()


