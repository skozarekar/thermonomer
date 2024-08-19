import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import re

class Polymerization:
    '''
        class Polymerization: 
            A class that polymerizes a monomer up to degree of polymerization chosen by user. Types of polymerizations available here: ROP (covers O, N, S), 
            ROMP, vinyl, ionic, cationic, cyclic

        Parameters for initiation: 
            monomer_smiles (str): smiles string of the monomer
            poly_type (str): the type of polymerization that the monomer undergoes
            dp (int): degree of polymerization. The amount of times the user wants the molecule to be polymerized

        Output:
            (dictionary): a dictionary with keys being dp and values being the smile string of the polymer
    '''

    def __init__(self, monomer_smiles, poly_type,dp):

        # This path will be used to fill in values that were done manually such as monomers that fall under base_state 'misc'
        relative_path = "data/archive_data.csv"
        # Turn the csv into a pandas array
        self.archive_df = pd.read_csv(relative_path)

        # Initiate the variables that will be used for polymerization
        self.dp = dp
        self.monomer_smiles = monomer_smiles
        self.rxn_mechanism = poly_type
        self.initiator_smiles = ""
        self.initiation_rxn_str = ""
        self.propagation_rxn_str = ""

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
            When polymerization.main() is called, polymerizes the monomer to the user's requested
            degree of polymerization (dp). 

            Parameters: 
                None
            
            Output:
                None
        '''

        keys = ["DP_" + str(i) for i in range(self.dp + 1)]

        # Apply polymerize data
        try: 
            polymerized_data = self.makePolymers()
            if polymerized_data == np.nan:
                # if fail to polymerize data, check archive and return from archive
                return self.unavailablePolymers(keys)
            else:
                return dict(zip(keys, polymerized_data))
        

        except:
            # Fill in missing polymerizations with old data
            return self.unavailablePolymers(keys)

    def makePolymers(self):
        '''
            A function that defines the initiation and propagation reaction depending on the type of polymerization
            the monomer undergoes. In most cases, Ge is used as the initiator for easy replacement with carbon later on. Calls
            the function that polymerizes the monomer and returns its output

            Parameters: 
                None
            
            Output:
                (list): the array of polymerized smile strings. Returns an empty array if the monomer could not be polymerized
        '''

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
                    return np.nan
            return self.polymerize()
        
        except:
            print(f"Polymerization mechanism {self.rxn_mechanism} not found for {self.monomer_smiles}.")
            return np.nan

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
                (list): a list of the polymerized molecules as smiles strings
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

        return poly_list
    
    def unavailablePolymers(self,keys):
        '''
            Fills in the data for polymers that are missing (ie. misc or could not be polymerized) with
            data from archive. 

            Parameters: 
                keys(list): the keys to the dictionary that will be returned. the keys for the dict are the degrees of polymerization 
            
            Output:
                None
        '''

        # make dictionary to return
        return_dict = {key: np.nan for key in keys}

        # Check if the value appears in the 'Canonical SMILES' column of archive_df
        if self.monomer_smiles in self.archive_df["Canonical SMILES"].values:
            # Loop through all columns with missing values
            for i in range(self.dp + 1):
                dp_column_name = f"DP_{i}"

                # Get the corresponding DP_i value from archive_df
                corresponding_dp_i = self.archive_df.loc[
                    self.archive_df["Canonical SMILES"] == self.monomer_smiles,
                    dp_column_name,
                ].values

                # Check if there is a corresponding DP_i value
                return_dict[dp_column_name] = corresponding_dp_i[0]
        else:
            print(f"{self.monomer_smiles} polymers could not be found")
            return np.nan
        
        return return_dict

