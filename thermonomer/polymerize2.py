import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

def get_polymer_list(degree_of_polymerization, monomer_SMILES, polymerization_type, end_group="C"):
    # Set I, P, T reactions
    initiation = _initiation_dictionary[polymerization_type]
    propagation = rdChemReactions.ReactionFromSmarts("[*:4]-[Po:1].[At:2]-[*:3]>>[*:4]-[*:3]")
    terminationAt = rdChemReactions.ReactionFromSmarts(f"[At:1]>>[{end_group}:1]")
    terminationPo = rdChemReactions.ReactionFromSmarts(f"[Po:1]>>[{end_group}:1]")

    # Create molecules
    helper = Chem.MolFromSmiles("[Po]-[At]")
    dp0 = Chem.MolFromSmiles(monomer_SMILES)

    # Initialize return list
    polymer_list = [dp0]

    # Run polymerization reactions
    for i in range(degree_of_polymerization):
        if i == 0: # initation - create repeat unit
            repeat_unit = initiation.RunReactants((dp0, helper))[1][0]
            polymer = repeat_unit

            polymer_list.append(repeat_unit) # DP = 1
        else: # propagation - add repeat unit onto polymer
            products = [item[0] for item in propagation.RunReactants((polymer, repeat_unit))]
            assert(len(products) ==  1) # Check that all propagation reactions create only one product

            # Set the polymer to be added to as the first product
            polymer = products[0]
            polymer_list.append(polymer)

    # "Termination": edit all end groups
    for mol, i in zip(polymer_list, list(range(degree_of_polymerization+1))):
        if i > 0:
            terminationAt.RunReactantInPlace(mol)
            terminationPo.RunReactantInPlace(mol)
    
    # Return SMILES of polymer strings from monomer - DP_n
    return polymer_list, [Chem.MolToSmiles(mol) for mol in polymer_list]

_initiation_dictionary = {
    # "ROP": _ROP_all,
    # "ROMP": _ROMP,
    "vinyl/acrylic": rdChemReactions.ReactionFromSmarts("[C:1]=[C:2].[Po]-[At]>>[Po]-[C:1]-[C:2]-[At]"),
    "aldehyde": rdChemReactions.ReactionFromSmarts("[O:1]=[C:2].[Po]-[At]>>[Po]-[O:1]-[C:2]-[At]"),
    # these two are the same
    "cyclic": rdChemReactions.ReactionFromSmarts("([*:1]@[*:2]).[Po:3]-[At:4]>>([*:1][Po:3].[*:2][At:4])"),
    "cationic": rdChemReactions.ReactionFromSmarts("([*:1]@[*:2]).[Po:3]-[At:4]>>([*:1][Po:3].[*:2][At:4])")
}


