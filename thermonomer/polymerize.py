import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

### Public polymerization function ###
def polymerize(DP, ring_SMILES, polymerization_category):
    return _polymerization_dict[polymerization_category](DP, ring_SMILES)

### Private functions and variables ###
def _ROMP(DP, ring_SMILES):
    """
    Perform Ring-Opening Metathesis Polymerization (ROMP) to generate a polymer chain.

    Args:
        DP (int): The degree of polymerization (number of repeating units).
        smiles (str): SMILES representation of the monomer molecule.

    Returns:
        pd.Series: A pandas Series containing SMILES strings representing the polymer chain
                   at different degrees of polymerization.
    """
    # Define the initiator and monomer molecules
    alkene = Chem.MolFromSmiles("C=C") 
    ring = Chem.MolFromSmiles(
        ring_SMILES
    ) 

    # Define the ROMP reactions
    opening_rxn = rdChemReactions.ReactionFromSmarts(
        "[CH2:1]=[CH2:2].([C:3]=[C:4])>>([C:1]=[C:3].[C:4]=[C:2])"
    )
    romp = rdChemReactions.ReactionFromSmarts(
        "[C:1]=[C:2].([C:3]=[C:4])>>([C:1]=[C:3].[C:4]=[C:2])"
    )

    # Initialize the polymer list with the monomer SMILES
    poly_list = [ring_SMILES]

    # If DP is 0, return the monomer as a single-element Series
    if DP == 0:
        return pd.Series(poly_list)

    # Open the ring to start polymerization (DP = 1)
    try:
        polymer = opening_rxn.RunReactants((alkene, ring))[0][0]
    except IndexError:
        # print(f"{ring_SMILES} cannot be polymerized using ROMP.")

        # Return a list that is the appropriate length, filled with N/A
        return pd.Series([np.nan] * (DP + 1))

    # Sanitize the polymer molecule and add its SMILES to the list
    Chem.SanitizeMol(polymer)
    poly_list.append(Chem.MolToSmiles(polymer))

    # If DP is 1, return a Series with the monomer and the first polymer unit
    if DP == 1:
        return pd.Series(poly_list)

    # If DP > 1, iteratively polymerize the monomer
    for degree in range(DP - 1):
        polymer = romp.RunReactants((polymer, ring))[0][0]
        Chem.SanitizeMol(polymer)
        poly_list.append(Chem.MolToSmiles(polymer))

    # Return a Series containing the polymer chain at different degrees of polymerization
    return pd.Series(poly_list)


def _vinyl(DP, ring_SMILES):
    """
    Perform vinyl polymerization to generate a polymer chain.

    Args:
        DP (int): The degree of polymerization (number of repeating units).
        monomer_smiles (str): SMILES representation of the monomer molecule.

    Returns:
        pd.Series: A pandas Series containing SMILES strings representing the polymer chain
                   at different degrees of polymerization.
    """
    # Define necessary molecules
    monomer = Chem.MolFromSmiles(ring_SMILES)
    water = Chem.MolFromSmiles("O")

    # Define vinyl polymerization reactions
    initiation = rdChemReactions.ReactionFromSmarts(
        "[C:1]=[C:2].[O:3]>>[C:1]-[C:2]-[O:3]"
    )
    vinyl_propagation = rdChemReactions.ReactionFromSmarts(
        "[C:1]-[OH:2].[C:3]=[C:4]>>[C:1]-[C:3]-[C:4]-[O:2]"
    )

    # Initialize the polymer list with the monomer SMILES
    poly_list = [ring_SMILES]

    # If DP is 0, return the monomer as a single-element Series
    if DP == 0:
        return pd.Series(poly_list)

    # Initiation
    try:
        polymer = initiation.RunReactants((monomer, water))[0][0]
    except IndexError:
        # print(f"{ring_SMILES} cannot be polymerized using vinyl polymerization.")
        return pd.Series([np.nan] * (DP + 1))

    # Sanitize the polymer molecule and add its SMILES to the list
    Chem.SanitizeMol(polymer)
    poly_list.append(Chem.MolToSmiles(polymer))

    # If DP is 1, return a Series with the monomer and the first polymer unit
    if DP == 1:
        return pd.Series(poly_list)

    # If DP > 1, iteratively polymerize the monomer
    for degree in range(DP - 1):
        polymer = vinyl_propagation.RunReactants((polymer, monomer))[0][0]
        Chem.SanitizeMol(polymer)
        poly_list.append(Chem.MolToSmiles(polymer))

    # Return a Series containing the polymer chain at different degrees of polymerization
    return pd.Series(poly_list)


def _aldehyde(DP, ring_SMILES):
    """
    Perform aldehyde polymerization to generate a polymer chain.

    Args:
        DP (int): The degree of polymerization (number of repeating units).
        aldehyde_smiles (str): SMILES representation of the aldehyde monomer.

    Returns:
        pd.Series: A pandas Series containing SMILES strings representing the polymer chain
                   at different degrees of polymerization.
    """
    # Create molecules
    monomer = Chem.MolFromSmiles(ring_SMILES, sanitize=True)
    water = Chem.MolFromSmiles("O")

    # Define aldehyde polymerization reactions based on the presence of sulfur
    if "S" in ring_SMILES or "s" in ring_SMILES:
        initiation = rdChemReactions.ReactionFromSmarts(
            "[S:3].[CX3:1](=[S:2])>>[S:3]-[C:1]-[S:2]"
        )
        aldehyde_reaction = rdChemReactions.ReactionFromSmarts(
            "[SH:3].[CX3:1](=[S:2])>>[S:3]-[C:1]-[S:2]"
        )
        water = Chem.MolFromSmiles("S")
    else:
        initiation = rdChemReactions.ReactionFromSmarts(
            "[OX2H2:3].[CX3:1](=[O:2])>>[O:3]-[C:1]-[O:2]"
        )
        aldehyde_reaction = rdChemReactions.ReactionFromSmarts(
            "[OH:3].[CX3:1](=[O:2])>>[O:3]-[C:1]-[O:2]"
        )

    # dp must be at least 0
    poly_list = [
        ring_SMILES,
    ]
    if DP == 0:
        return pd.Series(poly_list)

    # Initiation
    try:
        polymer = initiation.RunReactants((water, monomer))[0][0]
    except IndexError:
        # print(f"{ring_SMILES} cannot be polymerized using aldehyde polymerization.")
        return pd.Series([np.nan] * (DP + 1))

    # Sanitize the polymer molecule and add its SMILES to the list
    Chem.SanitizeMol(polymer)
    poly_list.append(Chem.MolToSmiles(polymer))

    # If DP is 1, return a Series with the monomer and the first polymer unit
    if DP == 1:
        return pd.Series(poly_list)

    # If DP > 1, iteratively polymerize the monomer
    for degree in range(DP - 1):
        polymer = aldehyde_reaction.RunReactants((polymer, monomer))[0][0]
        Chem.SanitizeMol(polymer)
        poly_list.append(Chem.MolToSmiles(polymer))

    # Return a Series containing the polymer chain at different degrees of polymerization
    return pd.Series(poly_list)


def _cyclic(DP, ring_SMILES):
    """
    Perform cyclic polymerization of a ring to generate a polymer chain.

    Args:
        DP (int): The degree of polymerization (number of repeating units).
        ring_smiles (str): SMILES representation of the ring monomer.

    Returns:
        pd.Series: A pandas Series containing SMILES strings representing the polymer chain
                   at different degrees of polymerization.
    """
    # Define polymerization reactions for cyclic polymerization
    open_ring = rdChemReactions.ReactionFromSmarts(
        "([CH2:1][CH2:2]).[O:3]>>([C:2][O:3].[C:1][OH])"
    )
    cycloalkane = rdChemReactions.ReactionFromSmarts("[OH][C:1].[OH]-[C:2]>>[C:1][C:2]")

    # Create molecules
    water = Chem.MolFromSmiles("O")
    ring = Chem.MolFromSmiles(ring_SMILES)

    # Initialize the polymer list with the monomer SMILES
    poly_list = [ring_SMILES]

    # If DP is 0, return the monomer as a single-element Series
    if DP == 0:
        return pd.Series(poly_list)

    # Initiation
    try:
        opened_ring = open_ring.RunReactants((ring, water))[0][0]
    except IndexError:
        # print(f"{ring_SMILES} cannot be polymerized using cyclic polymerization.")
        return pd.Series([np.nan] * (DP + 1))

    # Sanitize the opened ring molecule and add its SMILES to the list
    Chem.SanitizeMol(opened_ring)
    poly_list.append((Chem.MolToSmiles(opened_ring)).replace("O", "C"))

    # If DP is 1, return a Series with the monomer and the first polymer unit
    if DP == 1:
        return pd.Series(poly_list)

    # If DP > 1, iteratively polymerize the monomer
    polymer = opened_ring
    for i in range(DP - 1):
        polymer = cycloalkane.RunReactants((polymer, opened_ring))[0][0]
        Chem.SanitizeMol(polymer)
        poly_list.append((Chem.MolToSmiles(polymer)).replace("O", "C"))

    # Return a Series containing the polymer chain at different degrees of polymerization
    return pd.Series(poly_list)


def _cationic(DP, ring_SMILES):
    """
    Perform cationic polymerization of a ring to generate a polymer chain.

    Args:
        DP (int): The degree of polymerization (number of repeating units).
        ring_smiles (str): SMILES representation of the ring monomer.

    Returns:
        pd.Series: A pandas Series containing SMILES strings representing the polymer chain
                   at different degrees of polymerization.
    """

    # reactions
    if (
        "c" in ring_SMILES
    ):  # account for any with benzene rings... deal with robustness later
        open_ring = rdChemReactions.ReactionFromSmarts(
            "([CH2:1][*:2]).[O:3]>>([*:2][O:3].[C:1][OH])"
        )
        cycloalkane = rdChemReactions.ReactionFromSmarts(
            "[OH][*:1].[OH]-[C:2]>>[*:1][C:2]"
        )
    else:  # searches for any ring bond
        open_ring = rdChemReactions.ReactionFromSmarts(
            "([*:1]@[*:2]).[O:3]>>([*:2][O:3].[*:1][OH])"
        )
        cycloalkane = rdChemReactions.ReactionFromSmarts(
            "[OH][*:1].[OH]-[*:2]>>[*:1][*:2]"
        )

    # molecules
    water = Chem.MolFromSmiles("O")
    ring = Chem.MolFromSmiles(ring_SMILES)

    # dp must be at least 0
    poly_list = [
        ring_SMILES,
    ]
    if DP == 0:
        return poly_list

    # initiation
    try:
        opened_ring = open_ring.RunReactants((ring, water))[0][0]
    except IndexError:
        # print(f"{ring_SMILES} cannot be polymerized using cationic polymerization.")
        return pd.Series([np.nan] * (DP + 1))

    Chem.SanitizeMol(opened_ring)
    # poly_list.append((Chem.MolToSmiles(opened_ring)).replace('O','C'))
    poly_list.append(Chem.MolToSmiles(opened_ring))
    if DP == 1:
        return pd.Series(poly_list)

    # dp > 1
    polymer = opened_ring
    for i in range(DP - 1):
        polymer = cycloalkane.RunReactants((polymer, opened_ring))[0][0]
        Chem.SanitizeMol(polymer)  # save this one for the next iteration
        poly_list.append(Chem.MolToSmiles(polymer))
        # poly_list.append((Chem.MolToSmiles(polymer)).replace('O','C'))
    return pd.Series(poly_list)


def _ROP_all(DP, ring_SMILES):
    """
    Perform Ring-Opening Polymerization (ROP) to generate a polymer chain. Applicable to N-, S-, and O-containing rings

    Args:
        DP (int): The degree of polymerization (number of repeating units).
        smiles (str): SMILES representation of the monomer molecule.

    Returns:
        pd.Series: A pandas Series containing SMILES strings representing the polymer chain
                   at different degrees of polymerization.
    """

    # Create molecules
    water = Chem.MolFromSmiles("O")
    ring = Chem.MolFromSmiles(ring_SMILES)

    # Search for substructure match
    match_name = ""
    for key, val in _ROP_dict.items():
        temp_match = ring.GetSubstructMatch(Chem.MolFromSmarts(val[0]))

        # If match found, set variables and stop searching
        if len(temp_match) > 0:
            match_name = key
            opening_rxn = val[1]
            rop_rxn = val[2]

            break

    # If no substructure was matched, then return error and empty series
    if match_name:
        # Initialize the polymer list with the monomer SMILES
        poly_list = [ring_SMILES]

        # If DP is 0, return the monomer as a single-element Series
        if DP == 0:
            return pd.Series(poly_list)

        # Initiation
        try:
            opened_ring = opening_rxn.RunReactants((ring, water))[0][0]
        except IndexError:
            return pd.Series([np.nan] * (DP + 1))
            print(
                f"{ring_SMILES} cannot be polymerized using {match_name}. Index error."
            )

        # Sanitize the opened ring molecule and add its SMILES to the list
        Chem.SanitizeMol(opened_ring)
        poly_list.append(Chem.MolToSmiles(opened_ring))

        # If DP is 1, return a Series with the monomer and the first polymer unit
        if DP == 1:
            return pd.Series(poly_list)

        # If DP > 1, iteratively polymerize the monomer
        polymer = opened_ring
        for _ in range(DP - 1):
            polymer = rop_rxn.RunReactants((polymer, opened_ring))[0][0]
            Chem.SanitizeMol(polymer)
            poly_list.append(Chem.MolToSmiles(polymer))

        # Return a Series containing the polymer chain at different degrees of polymerization
        return pd.Series(poly_list)
    # Case where no matching substructure found
    else:
        # print(
        #     f"No matching substructure found. {ring_SMILES} cannot be polymerized using any kind of ROP."
        # )
        return pd.Series([np.nan] * (DP + 1))

# Dictionary covers various configurations of Ring-Opening Polymerization
_ROP_dict = {
    "ROP_N": [
        "[O;X1]=[C;X3]-[N]",
        rdChemReactions.ReactionFromSmarts(
            "([C:1](=[O:2])[N:3]@[C:4]).[O:5]>>([C:1](=[O:2])[O:5].[N:3][C:4])"
        ),
        rdChemReactions.ReactionFromSmarts(
            "[OX2H]-[CX3:3]=[OX1:4].[#6X4:1]-[N:2]>>[#6X4:1]-[N:2]-[CX3:3]=[OX1:4]"
        ),
    ],
    "ROP_O": [
        "[O;X1]=[C;X3]-[O]",
        rdChemReactions.ReactionFromSmarts(
            "([C:1](=[O:2])@[O:3]@[C:4]).[O:5]>>([C:1](=[O:2])[O:5].[O:3][C:4])"
        ),
        rdChemReactions.ReactionFromSmarts(
            "[#6X4:1]-[OX2H:2].[OX2H]-[CX3:3]=[OX1:4]>>[#6X4:1]-[O:2]-[CX3:3]=[OX1:4]"
        ),
    ],
    "ROP_S_A": [
        "[O;X1]=[C;X3]-[S]",
        rdChemReactions.ReactionFromSmarts(
            "([C:1](=[O:2])[S:3]@[C:4]).[O:5]>>([C:1](=[O:2])[O:5].[S:3][C:4])"
        ),
        rdChemReactions.ReactionFromSmarts(
            "[OX2H]-[C:3]=[OX1:4].[#6X4:1]-[S:2]>>[#6X4:1]-[S:2]-[C:3]=[OX1:4]"
        ),
    ],
    "ROP_S_B": [
        "[O;X1]=[S;X3]-[S]",
        rdChemReactions.ReactionFromSmarts(
            "([S:1](=[O:2])[S:3]@[C:4]).[O:5]>>([S:1](=[O:2])[O:5].[S:3][C:4])"
        ),
        rdChemReactions.ReactionFromSmarts(
            "[OX2H]-[S:3]=[OX1:4].[#6X4:1]-[S:2]>>[#6X4:1]-[S:2]-[S:3]=[OX1:4]"
        ),
    ],
    "ROP_S_C": [
        "[S;X1]=[C;X3]-[S]",
        rdChemReactions.ReactionFromSmarts(
            "([C:1](=[S:2])[S:3]@[C:4]).[O:5]>>([C:1](=[S:2])[O:5].[S:3][C:4])"
        ),
        rdChemReactions.ReactionFromSmarts(
            "[OX2H]-[C:3]=[SX1:4].[#6X4:1]-[S:2]>>[#6X4:1]-[S:2]-[C:3]=[SX1:4]"
        ),
    ],
    "ROP_S_D": [
        "[O;X1]=[S]-[O]",
        rdChemReactions.ReactionFromSmarts(
            "([S:1](=[O:2])[O:3]@[C:4]).[O:5]>>([S:1](=[O:2])[O:5].[O:3][C:4])"
        ),
        rdChemReactions.ReactionFromSmarts(
            "[OX2H]-[S:3]=[OX1:4].[#6X4:1]-[O:2]>>[#6X4:1]-[O:2]-[S:3]=[OX1:4]"
        ),
    ],
    "ROP_S_E": [
        "[S;X1]=[C;X3]-[O]",
        rdChemReactions.ReactionFromSmarts(
            "([C:1](=[S:2])[O:3]@[C:4]).[O:5]>>([C:1](=[S:2])[O:5].[O:3][C:4])"
        ),
        rdChemReactions.ReactionFromSmarts(
            "[OX2H]-[C:3]=[SX1:4].[#6X4:1]-[O:2]>>[#6X4:1]-[O:2]-[C:3]=[SX1:4]"
        ),
    ],
}

_polymerization_dict = {
    "ROP": _ROP_all,
    "ROMP": _ROMP,
    "vinyl/acrylic": _vinyl,
    "aldehyde": _aldehyde,
    "cyclic": _cyclic,
    "cationic": _cationic
}