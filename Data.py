import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def summarizeTrainTestResults(infile_path, target):
    # Get a list of all files in the directory  
    file_list = os.listdir(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    output_df = pd.DataFrame(columns=["FILE NAME", "AVG TRAIN ERROR (kcal/mol)", "AVG TEST ERROR (kcal/mol)"])
    final_path = parent_directory + "final_results/" + "trainTest_summary.csv"

    csv_files = [file for file in file_list]

    units = "(kcal/mol)"
    if target == "dS (J/mol/K)":
        units = "(kcal/mol/K)"

    # Read each CSV file into a pandas DataFrame and store it in the dictionary
    for file in csv_files:
        df = pd.read_csv(infile_path + "/" + file)
        # Create an empty DataFrame with the specified columns
        new_row = {"FILE NAME": file,
                       f"AVG TRAIN ERROR ({units})": np.average(df['Train Error per Iteration (kcal/mol)']),
                       f"AVG TEST ERROR ({units})": np.average(df['Test Error per Iteration (kcal/mol)'])
                       }
        # Add the new row
        output_df.loc[len(output_df)] = new_row

    output_df.to_csv(final_path, index = False)

def summarizeLOOCVResults(infile_path, target):
    # Get a list of all files in the directory  
    file_list = os.listdir(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"

    output_df = pd.DataFrame(columns=["FILE NAME", "MEAN ERROR", "RMSE"])
    final_path = parent_directory + "final_results/LOOCV_summary.csv"

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    csv_files = [file for file in file_list]

    units = "kcal/mol"
    if target == "dS (J/mol/K)":
        units = "kcal/mol/K"

    # Read each CSV file into a pandas DataFrame and store it in the dictionary
    for file in csv_files:
        df = pd.read_csv(infile_path + "/" + file)

        # Calc RMSE
        rmse = np.sqrt(mean_squared_error(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})']))
        new_row = {"FILE NAME": file,
                       f"MEAN ERROR ({units})": np.average(df['Mean Error (kcal/mol)']),
                       f"RMSE ({units})": rmse
                       }
        # Add the new row
        output_df.loc[len(output_df)] = new_row

    output_df.to_csv(final_path, index = False)

def graphExpPred(infile_path, target):
    init_dir = os.path.dirname(infile_path)
    parent_directory = os.path.dirname(init_dir) + "/"
    file_name = os.path.basename(os.path.normpath(infile_path))
    id = file_name.split('.')[0]
    output_path = parent_directory + "final_results/graph_" + id + ".png"

    units = "kcal/mol"
    if target == "dS (J/mol/K)":
        units = "kcal/mol/K"

    df = pd.read_csv(infile_path)

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    # Calculate R-squared
    r_squared = r2_score(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'])

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'])

    plt.plot([5,-30],[5,-30], '--', zorder=1)
    plt.scatter(df[f'Experimental Val ({units})'], df[f'Predicted Val ({units})'],color='#5c3c8b',zorder=2, alpha=0.75)

    plt.xlim([-30,5])
    plt.ylim([-30,5])

    plt.text(-29, 0, f'$R^2$ = {r_squared:.2f}', fontsize=12)
    plt.text(-29, 2.5, f'MAE = {mae:.2f} {units}', fontsize=12)

    plt.xlabel(f'Experimental Value ({units})')
    plt.ylabel(f'Predicted Value ({units})')

    # Extract the part before the underscore
    split_list = file_name.split('_')
    model = split_list[0]
    split_list = file_name.split(".")[0].split('_LOOCV_')
    case = split_list[1]
    plot_title = "Case " + case + ": " + model

    plt.title(plot_title)

    plt.savefig(output_path)  # Saves as a PNG file


    # colors = ["#5c3c8b", "#92c36d", "#ee9432", "#496391", "#85a5cd"]
    # cases 1/2, 3, 4, 5

def graphFeatureRanking(infile_path, num_feat):
    feature_df = pd.read_csv(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"
    id = os.path.basename(infile_path).split(".")[0].split("_")[2]
    final_path = parent_directory + "featureGraph_" + id + ".png"

    top_features, importances = [], []
    plt.figure(figsize=(18, 6))

    for index, row in feature_df.iterrows():
        top_features.append(row["FEATURE NAME"])
        importances.append(row["IMPORTANCE"])

    plt.barh(top_features[:num_feat], importances[:num_feat], color='#85a5cd',)
    plt.xlabel('Feature Importance', fontdict={'fontname': 'Times New Roman'})
    plt.ylabel('Feature', fontdict={'fontname': 'Times New Roman'})
    plt.title(f'Top {num_feat} Features, Case {id}', fontdict={'fontname': 'Times New Roman'})
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top

    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.savefig(final_path)  # Saves as a PNG file

    plt.show()


