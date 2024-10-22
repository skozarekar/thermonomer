import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

# Function to calculate Tanimoto similarity between two SMILES
def calculate_tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is not None and mol2 is not None:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        return 0.0
    
# Function to run a model with full LOO cross-validation
def train_model_LOOCV(model_name, regressor, feature_subset, X, y, results_folder):
    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Perform Leave One Out
    loo = LeaveOneOut()
    error_scores = []
    expt_vals = []
    predicted_vals = []

    for train_index, test_index in loo.split(X):
        try: # ndarray
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index], X[test_index] 
        # Get datasets
        except:  # dataframe
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]      


        # Fit your regressor on the training data
        if model_name == "NN":
            regressor.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)
        else:
            regressor.fit(X_train, y_train.values.ravel())

        # Make predictions on the test data
        y_pred = regressor.predict(X_test)

        # Calculate MAE for this specific test point
        err = mean_absolute_error(y_test, y_pred) / 4.184

        # Append the values to the list
        error_scores.append(err)
        expt_vals.append(y_test.iloc[0]/4.184)
        predicted_vals.append(y_pred[0]/4.184)

    # Output useful information
    print(
        f"params {results_folder}/{model_name}, mae: {np.mean(error_scores)}, std: {np.std(error_scores)}"
    )

    mae_score_df = pd.DataFrame(
        {
            "Mean Error (kcal/mol)": error_scores,
            "Experimental Val (kcal/mol)": expt_vals,
            "Predicted Val (kcal/mol)": predicted_vals,
        }
    )
    mae_score_df.to_csv(f"{results_folder}/{model_name}_{feature_subset}.csv")


from sklearn.model_selection import train_test_split

def train_test_model(model_name, regressor, feature_subset, X, y, results_folder, repetitions=200):
    # Initialize lists to store MAE values for each repetition
    train_error_scores = []
    test_error_scores = []

    for _ in range(repetitions):
        # Split the data into training and testing sets (90/10 stratified split based on phase)
        g_index = X.columns.get_loc('BASE_Monomer_State_g')
        l_index = X.columns.get_loc('BASE_Monomer_State_l')
        s_index = X.columns.get_loc('BASE_Monomer_State_s')
        stratify_labels = X.values[:, [g_index, l_index, s_index]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=stratify_labels)

        if model_name != "RF" and model_name != "XGB":
            # Scale Data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        # Fit your regressor on the training data
        if model_name == "NN":
            regressor.fit(X_train_scaled, y_train, epochs=10, batch_size=10, verbose=0)
        elif model_name == "RF" or model_name == "XGB":
            regressor.fit(X_train, y_train.values.ravel())
        else:
            regressor.fit(X_train_scaled, y_train.values.ravel())

        # Make predictions on the test data
        try:
            y_pred_test = regressor.predict(X_test_scaled)
        except:
            y_pred_test = regressor.predict(X_test)

        # Make predictions on the training data for comparison
        try:
            y_pred_train = regressor.predict(X_train_scaled)
        except:
            y_pred_train = regressor.predict(X_train)

        # Calculate MAE for this specific test point and append to list
        test_err = mean_absolute_error(y_test, y_pred_test) / 4.184
        test_error_scores.append(test_err)

        # Calculate MAE for this specific training point and append to list
        train_err = mean_absolute_error(y_train, y_pred_train) / 4.184
        train_error_scores.append(train_err)

    # Calculate the average MAE over all repetitions for both training and testing
    mean_train_mae = np.mean(train_error_scores)
    std_train_mae = np.std(train_error_scores)
    mean_test_mae = np.mean(test_error_scores)
    std_test_mae = np.std(test_error_scores)

    # Output useful information
    print(
        f"params {results_folder}/{model_name}, train_avg_mae: {mean_train_mae}, train_std_mae: {std_train_mae}, test_avg_mae: {mean_test_mae}, test_std_mae: {std_test_mae}"
    )

    # Save MAE values to a CSV file
    mae_score_df = pd.DataFrame({
        "Train Error per Iteration (kcal/mol)": train_error_scores,
        "Test Error per Iteration (kcal/mol)": test_error_scores
    })
    mae_score_df.to_csv(f"{results_folder}/{model_name}_{feature_subset}.csv")

def train_test_model_top_feats(model_name, regressor, feature_subset, X, y, results_folder, num_features, top_features_list, repetitions=200):
    # Initialize lists to store MAE values for each repetition
    train_error_scores = []
    test_error_scores = []

    for _ in range(repetitions):
        # Split the data into training and testing sets (90/10 stratified split based on phase)
        g_index = X.columns.get_loc('BASE_Monomer_State_g')
        l_index = X.columns.get_loc('BASE_Monomer_State_l')
        s_index = X.columns.get_loc('BASE_Monomer_State_s')
        stratify_labels = X.values[:, [g_index, l_index, s_index]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=stratify_labels)

        # Down select to the number of features
        X_train = X_train[top_features_list[:num_features]]
        X_test = X_test[top_features_list[:num_features]]

        if model_name != "RF" and model_name != "XGB":
            # Scale Data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        # Fit your regressor on the training data
        if model_name == "NN":
            regressor.fit(X_train_scaled, y_train, epochs=10, batch_size=10, verbose=0)
        elif model_name == "RF" or model_name == "XGB":
            regressor.fit(X_train, y_train.values.ravel())
        else:
            regressor.fit(X_train_scaled, y_train.values.ravel())

        # Make predictions on the test data
        try:
            y_pred_test = regressor.predict(X_test_scaled)
        except:
            y_pred_test = regressor.predict(X_test)

        # Make predictions on the training data for comparison
        try:
            y_pred_train = regressor.predict(X_train_scaled)
        except:
            y_pred_train = regressor.predict(X_train)

        # Calculate MAE for this specific test point 
        test_err = mean_absolute_error(y_test, y_pred_test) / 4.184
        test_error_scores.append(test_err)

        # Calculate MAE for this specific training point
        train_err = mean_absolute_error(y_train, y_pred_train) / 4.184
        train_error_scores.append(train_err)

    # Calculate the average MAE over all repetitions for both training and testing
    mean_train_mae = np.mean(train_error_scores)
    std_train_mae = np.std(train_error_scores)
    mean_test_mae = np.mean(test_error_scores)
    std_test_mae = np.std(test_error_scores)

    # Output useful information
    print(
        f"params {results_folder}/{model_name}, train_avg_mae: {mean_train_mae}, train_std_mae: {std_train_mae}, test_avg_mae: {mean_test_mae}, test_std_mae: {std_test_mae}"
    )

    # Save MAE values to a CSV file
    mae_score_df = pd.DataFrame({
        "Train Error per Iteration (kcal/mol)": train_error_scores,
        "Test Error per Iteration (kcal/mol)": test_error_scores
    })
    mae_score_df.to_csv(f"{results_folder}/{model_name}_{feature_subset}_{num_features}_features.csv")