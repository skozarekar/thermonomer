![image](https://github.com/user-attachments/assets/10d0170d-8f08-4130-af70-f84eecda168b)

# thermonomer
This code is used to generate a phase-specific fingerprint for a given polymer system.


## Prerequisites 

rdkit >= 2023.03.2 (Older version will cause error)

pandas



## Installation 
1. Navigate to the installation folder
   ```
   cd ~/GitHub-Repos
   ```
2. Clone the repo, navigate to the folder
   ```
   git clone https://github.com/skozarekar/thermonomer.git
   cd ~/GitHub-Repos/thermonomer
   ```
3. Install using pip
   ```
   pip install -e .
   ```

## Update Program
In case the code is updated, you may use git pull to update yours to the latest version.

1. Navigate to the pgthermo folder
   ```   
   cd ~/GitHub-Repos/thermonomer
   ```
2. Pull. If you used the -e flag as recommended when installing, it will update automatically after pull.
   ```
   git pull
   ```



## Usage
The polymerize module takes in a degree of polymerization, monomer SMILES string, and polymerization type and will return a list of polymers. The first item in the list is the monomer SMILES and the second is the opened ring (sometimes the same molecule).
```
   >> from thermonomer.polymerize import polymerize
   >> polymerize(2, "C=C", 'vinyl/acrylic')
   0             C1CCOC(=O)C1
   1              O=C(O)CCCCO
   2    O=C(O)CCCCOC(=O)CCCCO
```

The featurize module allows you to generate the components of a phase-specific fingerprint, which includes:
1. pgthermo $\Delta H_p$
2. Tanimoto M/S similarity
3. Dipole moment
4. Mix parameters
5. Solute parameters
6. RDKit descriptors
7. Steric descriptors

```
   >> from thermonomer.featurize import *
   >> featurize.tanimoto_monomer_solvent_similarity("s", "C=C(C)C(N)=O", "CC(=O)C")
   0.385
```



## Contact
Shivani Kozarekar - shivanikozarekar2026@u.northwestern.edu


## Acknowledgments
Prof. Linda Broadbelt
Hunter Lee
