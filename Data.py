import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def summarizeTrainTestResults(infile_path):
    # Get a list of all files in the directory  
    file_list = os.listdir(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    output_df = pd.DataFrame(columns=["FILE NAME", "AVG TRAIN ERROR (kcal/mol)", "AVG TEST ERROR (kcal/mol)"])
    final_path = parent_directory + "final_results/" + "trainTest_summary.csv"

    csv_files = [file for file in file_list]
    # Read each CSV file into a pandas DataFrame and store it in the dictionary
    for file in csv_files:
        df = pd.read_csv(infile_path + "/" + file)
        # Create an empty DataFrame with the specified columns
        new_row = {"FILE NAME": file,
                       "AVG TRAIN ERROR (kcal/mol)": np.average(df['Train Error per Iteration (kcal/mol)']),
                       "AVG TEST ERROR (kcal/mol)": np.average(df['Test Error per Iteration (kcal/mol)'])
                       }
        # Add the new row
        output_df.loc[len(output_df)] = new_row

    output_df.to_csv(final_path, index = False)

def summarizeLOOCVResults(infile_path):
    # Get a list of all files in the directory  
    file_list = os.listdir(infile_path)
    parent_directory = os.path.dirname(infile_path) + "/"

    output_df = pd.DataFrame(columns=["FILE NAME", "MEAN ERROR", "RMSE"])
    final_path = parent_directory + "final_results/LOOCV_summary.csv"

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    csv_files = [file for file in file_list]

    # Read each CSV file into a pandas DataFrame and store it in the dictionary
    for file in csv_files:
        df = pd.read_csv(infile_path + "/" + file)

        # Calc RMSE
        rmse = np.sqrt(mean_squared_error(df['Experimental Val (kcal/mol)'], df['Predicted Val (kcal/mol)']))
        new_row = {"FILE NAME": file,
                       "MEAN ERROR": np.average(df['Mean Error (kcal/mol)']),
                       "RMSE": rmse
                       }
        # Add the new row
        output_df.loc[len(output_df)] = new_row

    output_df.to_csv(final_path, index = False)

def graphExpPred(infile_path):
    init_dir = os.path.dirname(infile_path)
    parent_directory = os.path.dirname(init_dir) + "/"
    file_name = os.path.basename(os.path.normpath(infile_path))
    id = file_name.split('.')[0]
    output_path = parent_directory + "final_results/graph_" + file_name + ".png"

    df = pd.read_csv(infile_path)

    if not os.path.exists(parent_directory + "final_results"):
        # Check if an output folder exist if not make one
        os.makedirs(parent_directory + "final_results")

    # Calculate R-squared
    r_squared = r2_score(df['Experimental Val (kcal/mol)'], df['Predicted Val (kcal/mol)'])

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(df['Experimental Val (kcal/mol)'], df['Predicted Val (kcal/mol)'])

    plt.plot([5,-30],[5,-30], '--', zorder=1)
    plt.scatter(df['Experimental Val (kcal/mol)'], df['Predicted Val (kcal/mol)'],color='#5c3c8b',zorder=2, alpha=0.75)

    plt.xlim([-30,5])
    plt.ylim([-30,5])

    plt.text(-29, 0, f'$R^2$ = {r_squared:.2f}', fontsize=12)
    plt.text(-29, 2.5, f'MAE = {mae:.2f} kcal/mol', fontsize=12)

    plt.xlabel('Experimental Value (kcal/mol)')
    plt.ylabel('Predicted Value (kcal/mol)')

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
