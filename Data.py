import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def summarizeTrainTestResults(infile_path, target):
    # Get a list of all files in the directory  
    file_list = os.listdir(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"
    units = "(kcal/mol)"
    if target == "dS (J/mol/K)":
        units = "cal/mol/K"

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    output_df = pd.DataFrame(columns=["FILE NAME", f"AVG TRAIN ERROR ({units})", f"AVG TEST ERROR ({units})"])
    final_path = parent_directory + "final_results/" + "trainTest_summary.csv"

    csv_files = [file for file in file_list]

    # Read each CSV file into a pandas DataFrame and store it in the dictionary
    for file in csv_files:
        df = pd.read_csv(infile_path + "/" + file)
        # Create an empty DataFrame with the specified columns
        new_row = {"FILE NAME": file,
                       f"AVG TRAIN ERROR ({units})": np.average(df[f'Train Error per Iteration ({units})']),
                       f"AVG TEST ERROR ({units})": np.average(df[f'Test Error per Iteration ({units})'])
                       }
        # Add the new row
        output_df.loc[len(output_df)] = new_row

    output_df.to_csv(final_path, index = False)

def summarizeLOOCVResults(infile_path, target):
    units = "kcal/mol"
    if target == "dS (J/mol/K)":
        units = "cal/mol/K"

    # Get a list of all files in the directory  
    file_list = os.listdir(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"

    output_df = pd.DataFrame(columns=["FILE NAME", f"MEAN ERROR ({units})", f"RMSE ({units})"])
    final_path = parent_directory + "final_results/LOOCV_summary.csv"

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    csv_files = [file for file in file_list]

    # Read each CSV file into a pandas DataFrame and store it in the dictionary
    for file in csv_files:
        df = pd.read_csv(infile_path + "/" + file)

        # Calc RMSE
        rmse = np.sqrt(mean_squared_error(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})']))
        new_row = {"FILE NAME": file,
                       f"MEAN ERROR ({units})": np.average(df[f'Mean Error ({units})']),
                       f"RMSE ({units})": rmse
                       }
        # Add the new row
        output_df.loc[len(output_df)] = new_row

    output_df.to_csv(final_path, index = False)

def getColorYear(year, key = {"1942-1990": "#496391", 
              "1990-2000": "#5c3c8b" , 
              "2000-2010": "#ce2879",
              "2010-2020": "#92c36d", 
              "2020-2025": "#ee9432"}):
    # colors = {"purple": "#5c3c8b", 
    #           "green": "#92c36d" , 
    #           "orange": "#ee9432",
    #           "dark blue": "#496391", 
    #           "pink": "#ce2879"}

    # 1942 - 1990
    # 1991-2000
    # 2001-2010
    # 2011-2020
    # 2021-2025
    year = int(year)
    if year >= 1940 and year <= 1990:
        return key["1942-1990"]
    elif year > 1990 and year <= 2000:
        return key["1990-2000"]
    elif year > 2000 and year <= 2010:
        return key["2000-2010"]
    elif year > 2010 and year <= 2020:
        return key["2010-2020"]
    else:
        return key["2020-2025"]

def getBASE(bs, key = {"gg": "#92c36d",
           "ls": "#85A5CD",
           "ss": "#5c3c8b"}):
    colors = {"purple": "#5c3c8b", 
              "green": "#92c36d" , 
              "orange": "#ee9432",
              "dark blue": "#496391", 
              "pink": "#ce2879",
              "light blue": "#85A5CD",
              "yellow": "#ffe24c"}

    # 1942 - 1990
    # 1991-2000
    # 2001-2010
    # 2011-2020
    # 2021-2025

    if bs == "gg":
        return key["gg"]
    elif bs == "ls":
        return key["ls"]
    else:
        return key["ss"]

def getPolymerization(poly_type, my_key ={"ROMP": "#5c3c8b", 
              "vinyl": "#144a7c" , 
              "ionic": "#92c36d",
              "cationic": "#85A5CD", 
              "cyclic": "#0d680a",
              "ROP": "#ee9432",
              "misc": "#ce2879"}):
    for key, val in my_key.items():
        if poly_type == key:
            return val

    # ROMP, vinyl, ionic, cationic, cyclic, ROP, misc
# ------ Graph ------- #

def graphExpPred(infile_path, target, source = False, base_state = False, polymerization = False):
    plt.close()

    init_dir = os.path.dirname(infile_path)
    parent_directory = os.path.dirname(init_dir) + "/"
    file_name = os.path.basename(os.path.normpath(infile_path))
    id = file_name.split('.')[0].split('_')[2]
    model = file_name.split('.')[0].split('_')[0]

    titles = {
        "1": "Full + Solvent Params",
        "2": "Full - Solvent Params",
        "3": "Solvents Only",
        "4": "Bulk"

    }

    units = "kcal/mol"
    if target == "dS (J/mol/K)":
        units = "cal/mol/K"

    df = pd.read_csv(infile_path)

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    # Calculate R-squared
    r_squared = r2_score(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'])

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'])
    plt.plot([-50,20],[-50,20], '--', zorder=1)

    if source and not base_state and not polymerization: 
        plt.scatter(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'],color=df["Year"].apply(getColorYear),zorder=2, alpha=0.75)
        output_path = parent_directory + "final_results/" + id + model + "_year_scatterPlot.png"
        key = {"1942-1990": "#496391", 
              "1990-2000": "#5c3c8b" , 
              "2000-2010": "#ce2879",
              "2010-2020": "#92c36d", 
              "2020-2025": "#ee9432"}

        for year, color in key.items():
            plt.scatter([], [], color=color, label=f"{year}")  # Descriptive labels

        plt.legend(title="Year of Publication", loc='lower right', title_fontsize='13', fontsize='11')
        # plt.legend(title="Year of Source Publication", bbox_to_anchor=(1.01, 1), loc='upper left', title_fontsize='13', fontsize='11')

    elif base_state and not source and not polymerization: 
        plt.scatter(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'],color=df["BASE_State"].apply(getBASE)
,zorder=2, alpha=0.75)
        # Add legend
        key = {"gg": "#92c36d",
           "ls": "#85A5CD",
           "ss": "#5c3c8b"}
        for state, color in key.items():
            plt.scatter([], [], color=color, label=f"{state}")  # Descriptive labels
        # plt.legend(title="State", bbox_to_anchor=(1.01, 1), loc='upper left', title_fontsize='13', fontsize='11')
        plt.legend(title="State", loc='lower right', title_fontsize='13', fontsize='11')

        output_path = parent_directory + "final_results/" + id + model + "_state_scatterPlot.png"
    elif polymerization and not base_state and not source:
        key ={"ROMP": "#5c3c8b", 
              "vinyl": "#144a7c" , 
              "ionic": "#92c36d",
              "cationic": "#85A5CD", 
              "cyclic": "#0d680a",
              "ROP": "#ee9432",
              "misc": "#ce2879"}

        plt.scatter(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'],color=df["Polymerization"].apply(getPolymerization)
,zorder=2, alpha=0.75)
        # Add legend
        for poly_type, color in key.items():
            plt.scatter([], [], color=color, label=f"{poly_type}")  # Descriptive labels
        plt.legend(title="Polymerization Mechanism", bbox_to_anchor=(1.01, 1), loc='upper left', title_fontsize='13', fontsize='11')
        # plt.legend(title="Polymerization Mechanism", loc='lower right', title_fontsize='13', fontsize='11')

        output_path = parent_directory + "final_results/" + id + model + "_polyType_scatterPlot.png"

    else:
        plt.scatter(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'],color='#5c3c8b',zorder=2, alpha=0.75)
        output_path = parent_directory + "final_results/" + id +model + "_scatterPlot.png"


    plt.xlim([-50,20])
    plt.ylim([-50,20])

    plt.text(-45, 7.5, f'$R^2$ = {r_squared:.2f}', fontsize=12)
    plt.text(-45, 11.5, f'MAE = {mae:.2f} {units}', fontsize=12)

    plt.xlabel(f'Experimental Value ({units})')
    plt.ylabel(f'Predicted Value ({units})')

    # Extract the part before the underscore
    split_list = file_name.split('_')
    model = split_list[0]
    split_list = file_name.split(".")[0].split('_LOOCV_')
    case = split_list[1]
    if model == "XGB":
        model_fix = "XGBoost"
    plot_title = titles[case] + ": " + model_fix

    plt.title(plot_title)
    plt.savefig(output_path, bbox_inches='tight')  # Adjust bounding box

    # plt.savefig(output_path)  # Saves as a PNG file


    # colors = ["#5c3c8b" purple, "#92c36d" green, "#ee9432" orange, "#496391" dark blue, "#85a5cd" light blue]
    # cases 1/2, 3, 4, 5

def graphFeatureRanking(infile_path, num_feat):
    plt.close()

    titles = {
        "1": "Full + Solvent Params",
        "2": "Full - Solvent Params",
        "3": "Solvents Only",
        "4": "Bulk"

    }
    feature_df = pd.read_csv(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"
    id = os.path.basename(infile_path).split(".")[0].split("_")[2]
    model = os.path.basename(infile_path).split(".")[0].split("_")[0]
    final_path = parent_directory + id + model + "_featureRanks.png"
    # Sort the DataFrame by 'IMPORTANCE' in ascending order
    feature_df = feature_df.sort_values(by="IMPORTANCE", ascending=False)

    top_features, importances = [], []
    plt.figure(figsize=(18, 6))

    for index, row in feature_df.iterrows():
        top_features.append(row["FEATURE NAME"])
        importances.append(row["IMPORTANCE"])
    fontname = "Arial"
    size=20
    plt.barh(top_features[:num_feat], importances[:num_feat], color='#5c3c8b')
    plt.xlabel('Feature Importance', fontdict={'fontname': fontname}, fontsize = size)
    plt.ylabel('Feature', fontdict={'fontname': fontname}, fontsize = size)
    plt.title(f'Top {num_feat} {model} Features, {titles[id]}', fontdict={'fontname': fontname}, fontsize = size)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top

    plt.xticks(fontname=fontname, fontsize = size)
    plt.yticks(fontname=fontname)
    plt.savefig(final_path)  # Saves as a PNG file

    plt.show()


# plotting the different rings
# colors = ["#5c3c8b", "#92c36d", "#ee9432", "#496391", "#85a5cd"]
# hist_colors = {"dH (kJ/mol)": "#4EACC5", "dS (J/mol/K)": "#FF9C34"}

# ax1.hist(df[target], bins, color=hist_colors[target])

# def addSourcesandBase(core_path, edit_path):
#     # no_source_path = entropy_data.csv
#     # sourced_path =  raw_entropy_v3
#     to_update = pd.read_csv(edit_path)
#     source_path = pd.read_csv(core_path)

#     # Concatenating the specific columns from df1 to df2
#     df_combined = pd.concat([to_update, source_path[['Year', 'BASE_State', 'BASE_Category']]], axis=1)
#     df_combined.to_csv(edit_path)
