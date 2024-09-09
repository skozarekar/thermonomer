import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as cl
import numpy as np
import pandas as pd
import scipy
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns

from rdkit import Chem

# def saveBaseStates(infile_path):
#     # turn .csv into a pandas dataframe
#     # create donuts for category and monomer phase
#     df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")
#     monomer_bs_list = []
#     polymer_bs_list = []
#     new_dict = df.to_dict()
#     for index,row in df.iterrows():
#         base_state = row["BASE_State"]
#         monomer_base_state = base_state[0]
#         poly_base_state = base_state[1:]
#         monomer_bs_list.append(monomer_base_state)
#         polymer_bs_list.append(poly_base_state)
#     new_dict["BASE_Monomer_State"] = monomer_bs_list
#     new_dict["BASE_Polymer_State"] = polymer_bs_list
#     return pd.DataFrame(new_dict)

def addSources(no_source_path, sourced_path):
    # no_source_path = entropy_data.csv
    # sourced_path =  raw_entropy_v3
    no_source_df = pd.read_csv(no_source_path)
    sourced_df = pd.read_csv(sourced_path)
    unique_sources = []
    # print(no_source_df.columns)
    no_source_dict = no_source_df.to_dict()
    # print(no_source_dict)
    count = 0

    final_path = os.path.dirname(no_source_path) + "/sourced_entropy.csv"
    # Loop through all columns and see if there is a matching row
    # conditions 
    for idx, row in no_source_df.iterrows():
        try:

            value_df = sourced_df[sourced_df["dS (J/mol/K)"] == row["dS (J/mol/K)"]]
            # source = value_df[0, "Source"]
            value_df_narrowed = value_df[value_df["Monomer SMILES"] == row["Monomer SMILES"]]

            print(f"length: {len(value_df_narrowed)}")
            # if len(value_df_narrowed) != 1: 
            #     value_df_narrowed = value_df_narrowed[value_df_narrowed["Solvent"] == row["Solvent"]]

            if len(value_df_narrowed) != 1: 
                print("ERROR: NOT UNIQUE ENOUGH")

            # print(f"new length: {len(value_df_narrowed)}")

            # return_this = value_more_narrow.iloc[0]
            # print(f"investigate: {return_this}")
            source = value_df_narrowed.iloc[0]["Source"]
            print(f"Source: {source}")
            # corresponding_i = sourced_df.loc[
            #     sourced_df["Monomer SMILES"] == row["Monomer SMILES"] and sourced_df["dS (J/mol/K)"] == row["ds (J/mol/K)"]
            #     ].values
            # print(f"look here: {corresponding_i}")
            # source = corresponding_i["Source"]

            # print(f"Source: {corresponding_i}")
            # no_source_df.update(overwrite = )[0]["Source"] = source
            no_source_dict["Source"][idx] = source
            if source not in unique_sources:
                unique_sources.append(source)
        
            print(f"Unique Sources: {unique_sources}")
            pd.DataFrame(no_source_dict).to_csv(final_path)
        
        except:
            count= count + 1
            print("failed :(")
    print(f"number missing: {count}")


def split_state(path):
    df = pd.read_csv(path)
    # split into unqiue states
    df["BASE_Monomer_State"] = ""
    df["BASE_Polymer_State"] = ""
    unique_states = ["l", "c", "s", "g", "c'", "a"]
    for state in unique_states:
        df.loc[df.BASE_State.str.startswith(state), "BASE_Monomer_State"] = state
        df.loc[df.BASE_State.str.endswith(state), "BASE_Polymer_State"] = state

    return df

def checkUnique(infile_path):
    df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")
    tags = []

    for _, row in df.iterrows():
        tag = row['BASE_State'] + row['BASE_Category'] + row['Canonical SMILES'] + str(row['Solvent'])
        tags.append(tag)

    return (len(tags) == len(set(tags)))

def donutPlots(infile_path):
    # infile_path = path to featurized entropy.csv
    # /Users/hunter/Downloads/BROADBELT LAB/thermonomer/monomer data build/featurized_entropy.csv
    directory_path = os.path.dirname(infile_path) + "/"
    if not os.path.exists(directory_path + "data_stats"):
        # Check if an output folder exist if not make one
        os.makedirs(directory_path + "data_stats")

    final_folder_path = directory_path + "data_stats/"
    df = pd.read_csv(infile_path)
    file_path = "A_report.txt"

    with open(file_path, "w") as file:
        # Count different polymerization types
        mechanisms = df["BASE_Category"].unique().tolist()
        mechanism_counts = []

        file.write("Category      | Occurences\n")
        file.write("-" * 14 + "|------------\n")

        for mechanism in mechanisms:
            count = df["BASE_Category"].value_counts()[mechanism]
            mechanism_counts.append(count)
            formatted_line = f"{mechanism.ljust(13)} | {str(count)}\n"
            file.write(formatted_line)

        # Count unique monomer states
        monomer_states = df["BASE_Monomer_State"].unique().tolist()
        monomer_counts = []

        file.write("\n\nMonomer State | Occurences\n")
        file.write("-" * 14 + "|------------\n")

        for state in monomer_states:
            count = df["BASE_Monomer_State"].value_counts()[state]
            monomer_counts.append(count)
            formatted_line = f"{state.ljust(13)} | {str(count)}\n"
            file.write(formatted_line)

        # Count unique overall states
        states = df["BASE_State"].unique().tolist()
        counts = []

        file.write("\n\nOverall State | Occurences\n")
        file.write("-" * 14 + "|------------\n")

        for state in states:
            count = df["BASE_State"].value_counts()[state]
            counts.append(count)
            formatted_line = f"{state.ljust(13)} | {str(count)}\n"
            file.write(formatted_line)

        # plotting the different rings
        colors = ["#5c3c8b", "#92c36d", "#ee9432", "#496391", "#85a5cd"]

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        wedges, texts = axs[0].pie(
            mechanism_counts,
            labels=mechanisms,
            wedgeprops={"width": 0.5},
            startangle=200,
            labeldistance=1.1,
            colors=colors,
        )

        # Add hatching to specific wedges
        for i, wedge in enumerate(wedges):
            mech = mechanisms[i]
            if mech in ['cyclic', 'ROP', 'ROMP', 'cationic']:
                # wedge.set_hatch(hatch_patterns.pop(0))
                wedge.set_hatch('//////')
                wedge.set_edgecolor('white')  # Set edge color to white for better contrast
                wedge.set_linewidth(0.5) 

        wedges, texts = axs[1].pie(
            monomer_counts,
            labels=["bulk", "solution", "gas"],
            wedgeprops={"width": 0.5},
            startangle=90,
            colors=colors,
        )

        fig.suptitle(f"Total Points: {len(df['Monomer SMILES'])}", x=0.5, y=0.1)
        axs[0].set_title("Category")
        axs[1].set_title("Monomer Phase")

        plt.savefig(final_folder_path + "mechanism_state_donut.png")
        print("Donuts saved :)")

def generateHistograms(infile_path, target):
    # Split histogram
    df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")

    directory_path = os.path.dirname(infile_path) + "/"
    if not os.path.exists(directory_path + "data_stats"):
        # Check if an output folder exist if not make one
        os.makedirs(directory_path + "data_stats")
    final_folder_path = directory_path + "data_stats/"

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 4.5)
    ax1.grid(False)
    ax2.grid(False)

    bins = np.linspace(-250, 0, 40)

    hist_colors = {"dH (kJ/mol)": "#4EACC5", "dS (J/mol/K)": "#FF9C34"}

    ax1.hist(df[target], bins, color=hist_colors[target])

    if target == "dH (kJ/mol)":
        ax1.set(xlabel="$\Delta H$ (kJ/mol)", ylabel="Frequency")
        plt.savefig(final_folder_path + "dH_freq_hist.png")
    else:
        ax1.set(xlabel="$\Delta S$ (J/mol/K)", ylabel="Frequency")
        plt.savefig(final_folder_path + "dS_freq_hist.png")

    plt.close()

