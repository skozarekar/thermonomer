import xgboost as xgb
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import pandas as pd
import pickle

from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense
from keras.models import Sequential

import warnings
warnings.filterwarnings("ignore")

import os
import re
import pandas as pd
import numpy as np

# -------------- HELPER FUNCTIONS -------------- #
def searchSpaceInit():
    '''
        Called by the helper function optimizerInit(n_iters). Defines the parameter space for each model. 

        Parameters: 
            None
            
        Output:
            xgb_param_space, rf_param_space, svr_param_space,kr_param_space, gp_param_space (dict)
    '''

    # Define parameter space for XGB
    xgb_param_space = {'learning_rate': (0.01, 0.3),
                'n_estimators': (100, 300),
                'max_depth': (3, 5),
                'min_child_weight': (1, 10),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'gamma': (0, 0.3),
                'reg_alpha': (0, 0.3),
                'reg_lambda': (0, 0.3)}

    # Define parameter space for Random Forest
    rf_param_space = {'n_estimators': Integer(10, 300),
                    'max_depth': Integer(1, 20),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 20)}

    # Define parameter space for SVR
    svr_param_space = {'C': Real(1e-6, 1e+6, prior='log-uniform'),
                    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                    'epsilon': Real(1e-3, 1e-1, prior='log-uniform')}

    # Define parameter space for Kernel Ridge Regressor
    kr_param_space = {'alpha': Real(1e-6, 1e+6, prior='log-uniform'),
                    'kernel': ['linear', 'poly', 'rbf', 'laplacian', 'sigmoid'],
                    'gamma': Real(1e-6, 1e+1, prior='log-uniform')}

    # Define parameter space for Gaussian Process Regressor
    gp_param_space = {'alpha': Real(1e-6, 1e+10, prior='log-uniform')}

    return xgb_param_space, rf_param_space, svr_param_space,kr_param_space, gp_param_space

def optimizerInit(n_iters, bayes_search = True):
    '''
        This function uses the search spaces obtained from searchSpaceInit() to get the optimizers for each model

        Parameters: 
            n_iters (int): the number of iterations for the optimizer
            
        Output:
            xgb_optimizer, rf_optimizer, svr_optimizer, kr_optimizer, gp_optimizer: the optimizers for each model
    '''

    xgb_param_space, rf_param_space, svr_param_space,kr_param_space, gp_param_space = searchSpaceInit()

    if bayes_search:

        # Initialize BayesSearchCV for each model: optimize paramters for 12hr / test Dataset
        ### baysian search replace with something else. grid search cv maybe. bayes search doesn't need to be this specifically look into other options
        # grid search?
        xgb_optimizer = BayesSearchCV(estimator=xgb.XGBRegressor(),
                        search_spaces=xgb_param_space,
                        n_iter=n_iters,
                        cv=5,
                        random_state=42)

        rf_optimizer = BayesSearchCV(estimator=RandomForestRegressor(),
                                    search_spaces=rf_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        svr_optimizer = BayesSearchCV(estimator=SVR(),
                                    search_spaces=svr_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        kr_optimizer = BayesSearchCV(estimator=KernelRidge(),
                                    search_spaces=kr_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        gp_optimizer = BayesSearchCV(estimator=GaussianProcessRegressor(kernel=RBF()),
                                    search_spaces=gp_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)
    
    else:
        xgb_optimizer = GridSearchCV(estimator=xgb.XGBRegressor(),
                        search_spaces=xgb_param_space,
                        n_iter=n_iters,
                        cv=5,
                        random_state=42)

        rf_optimizer = GridSearchCV(estimator=RandomForestRegressor(),
                                    search_spaces=rf_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        svr_optimizer = GridSearchCV(estimator=SVR(),
                                    search_spaces=svr_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        kr_optimizer = GridSearchCV(estimator=KernelRidge(),
                                    search_spaces=kr_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        gp_optimizer = GridSearchCV(estimator=GaussianProcessRegressor(kernel=RBF()),
                                    search_spaces=gp_param_space,
                                    n_iter=n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)
    
    return xgb_optimizer, rf_optimizer, svr_optimizer, kr_optimizer, gp_optimizer

def splitDatasetHyperparams(n_iters, x_df_dict, y_df_dict, xgb_only = False, bayes = True):
    '''
        Obtains and saves the hyperparameters for each model and dataset pair. Calls getHyperparams.

        Parameters: 
            n_iters (int): the number of iterations for the optimizer
            x_df_dict (dict): a dictionary where the values are the input pandas dataframes for each split
            y_df_dict (dict): a dictionary where the values are the output pandas dataframes for each split
            
        Output:
            (dict of dicts): a dict containing dictionaries of hyperparams for each model for each dataset.
            saves a .csv file that contains the hyperparameter data
    '''
    hyperparams = {
        "1": np.nan,
        "2": np.nan,
        "3": np.nan,
        "4": np.nan,
    }

    # save params for set 1 / XGB model
    xgb_best_params_1 = getHyperparams(n_iters, x_df_dict["1"], y_df_dict["1"], xgb_only = True, bayes_search=bayes)
    print("Dataset 1 Hyperparameters Complete")

    hyperparams = {
        "1": xgb_best_params_1,
        "2": np.nan,
        "3": np.nan,
        "4": np.nan,
    }
    with open('hyperparams.pkl', 'wb') as file:
        pickle.dump(hyperparams, file)

    # save params for set 2 / all models
    best_params_2 = getHyperparams(n_iters, x_df_dict["2"], y_df_dict["2"], xgb_only = xgb_only, bayes_search=bayes)
    hyperparams = {
        "1": xgb_best_params_1,
        "2": best_params_2,
        "3": np.nan,
        "4": np.nan,
    }
    with open('hyperparams.pkl', 'wb') as file:
        pickle.dump(hyperparams, file)
    print("Dataset 2 Hyperparameters complete")

    # save params for set 3 (solution phase only) / all models
    best_params_3 = getHyperparams(n_iters, x_df_dict["3"], y_df_dict["3"], xgb_only = xgb_only, bayes_search=bayes)
    hyperparams = {
        "1": xgb_best_params_1,
        "2": best_params_2,
        "3": best_params_3,
        "4": np.nan,
    }
    with open('hyperparams.pkl', 'wb') as file:
        pickle.dump(hyperparams, file)
    print("Dataset 3 Hyperparameters complete")

    # save params for set 4 (bulk phase only) / all models
    best_params_4 = getHyperparams(n_iters, x_df_dict["4"], y_df_dict["4"], xgb_only = xgb_only, bayes_search=bayes)
    print("Dataset 4 Hyperparameters complete")

    hyperparams = {
        "1": xgb_best_params_1,
        "2": best_params_2,
        "3": best_params_3,
        "4": best_params_4,
    }

    with open('hyperparams.pkl', 'wb') as file:
        pickle.dump(hyperparams, file)

    return hyperparams

def create_nn_model():
    '''
        Neural network model. Untested and not used in this .py file. 

        Parameters: 
            None
            
        Output:
            model
    '''

    model = Sequential()
    model.add(Dense(10, input_dim=173, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_model_LOOCV(model_name, regressor, X, y, target, categories = np.nan):
    '''
        Function to run a model with full LOO cross-validation

        Parameters: 
            model_name (str): the name of the model being trained
            regressor
            X (pd dataframe): input dataset
            y (pd dataframe): output dataset
            
        Output:
            output_dict (dict): a dictionary containing the important output values from the model
    '''

    if model_name != "RF":
        # Scale Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    # Perform Leave One Out
    loo = LeaveOneOut()
    error_scores = []
    expt_vals = []
    predicted_vals = []
        # if not pd.isna(categories):
        #     X_train, X_test, y_train, y_test, year_range_train, year_range_test, source_range_train, source_range_test, base_range_train, base_range_test = train_test_split(
        #         X, y, categories_df["Year", categories_df["Source", categories_df["BASE_State"]]], test_size=0.2, random_state=42)
        # else:
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=stratify_labels)

    for train_index, test_index in loo.split(X):
        try: # ndarray
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index], X[test_index] 
        # Get datasets
        except:  # dataframe
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]      

        # categories_df = pd.DataFrame(categories)
        # if not pd.isna(categories):
        #     year_range_train, year_range_test = categories_df.loc[train_index, 'Year'], categories_df.loc[test_index, 'Year']
        #     source_range_train, source_range_test = categories_df.loc[train_index, 'Source'], categories_df.loc[test_index, 'Source']
        #     base_range_train, base_range_test =categories_df.loc[train_index, 'BASE_State'], categories_df.loc[test_index, 'BASE_State']
        # Fit your regressor on the training data
        # if model_name == "NN":
        #     regressor.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)
        # else:
        regressor.fit(X_train, y_train.values.ravel())

        # Make predictions on the test data
        y_pred = regressor.predict(X_test)

        # Calculate MAE for this specific test point
        err = mean_absolute_error(y_test, y_pred) / 4.184
        expt_val = y_test.iloc[0]/4.184
        pred_val = y_pred[0]/4.184
        # converts to kcal / mol for dH
        # converts to cal / mol / K for dS
        error_scores.append(err)
        # Append the values to the list
        expt_vals.append(expt_val)
        predicted_vals.append(pred_val)

        units = "kcal/mol"

        if target == "dS (J/mol/K)":
            units = "cal/mol/K"

    if not pd.isna(categories):
        output_dict = {
                f"Mean Error ({units})": error_scores,
                f"Experimental Val ({units})": expt_vals,
                f"Predicted Val ({units})": predicted_vals,
                "MAE": np.mean(error_scores), 
                "STD": np.std(error_scores),
                "Year": categories["Year"], 
                "Source": categories["Source"], 
                "BASE_State": categories["BASE_State"]
            }
        
    else:
        output_dict = {
            f"Mean Error ({units})": error_scores,
            f"Experimental Val ({units})": expt_vals,
            f"Predicted Val ({units})": predicted_vals,
            "MAE": np.mean(error_scores), 
            "STD": np.std(error_scores)
        }

    
    print(f"TRAIN {model_name} COMPLETE")

    return output_dict

def train_test_model(model_name, regressor, X, y, target, repetitions=200):
    '''
        Function to run a model where 90/10 stratified split based on phase

        Parameters: 
            model_name (str): the name of the model being trained
            regressor
            X (pd dataframe): input dataset
            y (pd dataframe): output dataset
            repetitions (int)
            
        Output:
            output_dict (dict): a dictionary containing the important output values from the model
    '''

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

        if model_name != "RF":
            # Scale Data
            # standard scalar is a normalization method. this model bad with big numbers so scale the data so the 
            # mean is 0 and stdev is 1 making the space one that the model can work with better
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        # Fit your regressor on the training data
        # if model_name == "NN":
        #     regressor.fit(X_train_scaled, y_train, epochs=10, batch_size=10, verbose=0)
        if model_name == "RF":
            regressor.fit(X_train, y_train.values.ravel())
        else:
            regressor.fit(X_train_scaled, y_train.values.ravel())

        # Make predictions on the test data
        # how does it do on data it hasn't seen. best measure of how good it is
        try:
            y_pred_test = regressor.predict(X_test_scaled)
        except:
            y_pred_test = regressor.predict(X_test)

        # Make predictions on the training data for comparison
        # for comparison, see how well it did on training data to see over/under fit
        try:
            y_pred_train = regressor.predict(X_train_scaled)
        except:
            y_pred_train = regressor.predict(X_train)

        # Calculate MAE for this specific test point and append to list
        test_err = mean_absolute_error(y_test, y_pred_test) / 4.184

        # Calculate MAE for this specific training point and append to list
        train_err = mean_absolute_error(y_train, y_pred_train) / 4.184
        # converts to kcal / mol for dH
        # converts to cal / mol / K for dS
        train_error_scores.append(train_err)
        test_error_scores.append(test_err)

        units = "kcal/mol"
        if target == "dS (J/mol/K)":
            units = "cal/mol/K"

    # Calculate the average MAE over all repetitions for both training and testing
    mean_train_mae = np.mean(train_error_scores)
    std_train_mae = np.std(train_error_scores)
    mean_test_mae = np.mean(test_error_scores)
    std_test_mae = np.std(test_error_scores)

    # Output useful information
    # print(
    #     f"params {results_folder}/{model_name}, train_avg_mae: {mean_train_mae}, train_std_mae: {std_train_mae}, test_avg_mae: {mean_test_mae}, test_std_mae: {std_test_mae}"
    # )

    # Save MAE values to a CSV file
    output_dict = {
        f"Train Error per Iteration ({units})": train_error_scores,
        f"Test Error per Iteration ({units})": test_error_scores,
        "Train Avg MAE": mean_train_mae,
        "Train STD MAE": std_train_mae,
        "Test Avg MAE": mean_test_mae,
        "Test STD MAE": std_test_mae
    }

    print(f"TRAIN {model_name} LOOCV COMPLETE")

    return output_dict
                
def getFeatureRanking(X, model, model_name, y, num_iterations = 100):
    # a list of the feature names
    feature_names = pd.DataFrame(X).columns.tolist()

    # if model_name != "XGB" or model_name != "RF":
    #     # Initialize SelectKBest with mutual information scoring
    #     selector = SelectKBest(score_func=mutual_info_regression, k=50)
    #     # Fit selector and transform X
    #     selector.fit(X, y)
    #     # Get scores for each feature
    #     feature_scores = selector.scores_
    #     feature_score_dict = dict(zip(feature_names, feature_scores))
    #     feature_score_df = pd.DataFrame(feature_score_dict)
    #     return feature_score_df

    if model_name == "XGB":
        for _ in range(num_iterations):
            model.fit(X,y)
            # Get feature importances
            importances = model.feature_importances_

        # Create a DataFrame to hold feature names and their importances
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        # Sort the DataFrame by importance
        sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        # Print top ten features:
        # sorted_df.head(10)

        return sorted_df

    else: #return empty df
        return pd.DataFrame()

def getCats(df):
    source = df['Source']
    year = df['Year']
    basestate = df['BASE_State']

    df = df.drop(
            columns=[
                "Source",
                "Year",
                "BASE_State"])
    
    my_dict = {"Source": source,
               "Year": year,
               "BASE_State": basestate}
    return my_dict, df

# -------------- MAIN FUNCTIONS -------------- #
def getXy(infile_path, target):
    '''
        A function to get the X and y datasets that go into the model

        Parameters: 
            infile_path (str): the path to the monomer featurized dataset that has been cleaned, imputed and one-hot encoded
            target (str): enthalpy ("dH (kJ/mol)") or entropy ("dS (J/mol/K)")
            
        Output:
            x_df, y_df (pandas df)
    '''

    df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")

    x_df = df.drop(columns=[target])
    y_df = df[target]
    return x_df, y_df

def createDataSets(infile_path, target):
    '''
        A function that takes in a monomer featurized dataset and splits it into four different datasets. Models will be run on all data sets and results can be compared.

        4 total data sets:
            1. Full + solvent parameters
            2. Full without solvent parameters
            3. Solvents only (with solvent parameters)
            4. Bulk only

        Parameters:
            infile_path (str): path to the .csv containing the monomer featurized dataset that has been cleaned, one-hot encoded and imputed
            target (str): enthalpy ("dH (kJ/mol)") or entropy ("dS (J/mol/K)")

        Output:
            saves each split dataset as a .csv in a new folder called 'splits'
            x_df_dict (dict): a dictionary where the values are the input pandas dataframes for each split
            y_df_dict (dict): a dictionary where the values are the output pandas dataframes for each split
    '''

    imputed_df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")

    base_path = os.path.dirname(infile_path)

    # drop cols that shouldnt be used in any Dataset
    imputed_df = imputed_df.drop(
        columns=[
            "Solvent",
            "Canonical_Monomer_SMILES",
            "Solvent_SMILES",
            "Solvent_SMILES_2",
            "DP_0",
            "DP_1",
            "DP_2",
            "DP_3",
            "DP_4",
            "DP_5",
            "MIX_dGsolv298(kJ/mol) epi.unc.",
            "MIX_dHsolv298(kJ/mol) epi.unc.",
            # "Source",
            # "Year",
            # "BASE_State"
            ]
    )

    # SET 1: Full + solvent parameters + temp
    X_full_with_solvent_params_init = imputed_df

    s1_categories_dict, X_full_with_solvent_params = getCats(X_full_with_solvent_params_init)

    # SET 2: Drop solvent parameters and temperature, both of which have np.nan values
    if target == "dS (J/mol/K)":
        imputed_df = imputed_df.drop(
            columns=[
                "SOLV_PARAM_s_g",
                "SOLV_PARAM_b_g",
                "SOLV_PARAM_e_g",
                "SOLV_PARAM_l_g",
                "SOLV_PARAM_a_g",
                "SOLV_PARAM_c_g",
                "SOLV_PARAM_visc at 298 K (cP)",
                "SOLV_PARAM_dielectric constant",
                "Temp (C)"
                ]
            )
    else:
        imputed_df = imputed_df.drop(
            columns=[
                "SOLV_PARAM_s_g",
                "SOLV_PARAM_b_g",
                "SOLV_PARAM_e_g",
                "SOLV_PARAM_l_g",
                "SOLV_PARAM_a_g",
                "SOLV_PARAM_c_g",
                "SOLV_PARAM_visc at 298 K (cP)",
                "SOLV_PARAM_dielectric constant",
                ]
            )

    # Set 2: full + no solvent params no temp
    X_full_init = imputed_df
    s2_categories_dict, X_full = getCats(X_full_init)

    # Set 3: solvents only no temp
    if target == "dS (J/mol/K)":
        solvent_no_temp = X_full_with_solvent_params_init.drop(
                columns=[
                    "Temp (C)"])
        X_solvent_only = solvent_no_temp.loc[solvent_no_temp.BASE_Monomer_State_s]
    else: 
        X_solvent_only = X_full_with_solvent_params_init.loc[X_full_with_solvent_params_init.BASE_Monomer_State_s]
    s3_categories_dict, X_solvent_only = getCats(X_solvent_only)

    # Set 4: bulk only
    X_bulk_only = X_full_init.loc[X_full_init.BASE_Monomer_State_l]
    s4_categories_dict, X_bulk_only = getCats(X_bulk_only)

    # Create an empty dictionary to store data frames
    # _x does not include df
    x_df_dict = {
        "1": X_full_with_solvent_params.drop(columns=[target]),
        "2": X_full.drop(columns=[target]),
        "3": X_solvent_only.drop(columns=[target]),
        "4": X_bulk_only.drop(columns=[target]),
    }

    # _y is just dS values
    y_df_dict = {
        "1": X_full_with_solvent_params[target],
        "2": X_full[target],
        "3": X_solvent_only[target],
        "4": X_bulk_only[target],
    }

    categories_dict = {
        "1": s1_categories_dict,
        "2": s2_categories_dict,
        "3": s3_categories_dict,
        "4": s4_categories_dict
    }

    # Save as csv just to have as a record
    # Create the folder
    split_path = base_path + "/splits/"
    os.makedirs(split_path, exist_ok=True)
    X_full_with_solvent_params.to_csv(split_path + "split_1.csv")
    X_full.to_csv(split_path + "split_2.csv")
    X_solvent_only.to_csv(split_path + "split_3.csv")
    X_bulk_only.to_csv(split_path + "split_4.csv")

    with open('categories.pkl', 'wb') as file:
        pickle.dump(categories_dict, file)

    return x_df_dict, y_df_dict

def getHyperparams(n_iters, x_df, y_df, xgb_only = False, bayes_search = True):
    '''
        This function finds hyperparameters using the output optimizer functions retrieved from calling optimizerInit for each model

        Parameters: 
            n_iters (int): the number of iterations for the optimizer
            x_df (pandas df): a pandas dataframe contianing input values
            y_df(pandas df): a pandas dataframe contianing output values
            xgb_only (bool)*: True if you want just the xgb hyperparams

            *xgb_only is an optional input, default value is False
            
        Output:
            (dict): a dicitonary of the hyperparams for each model
    '''

    xgb_optimizer, rf_optimizer, svr_optimizer, kr_optimizer, gp_optimizer = optimizerInit(n_iters, bayes_search=bayes_search)

    xgb_optimizer.fit(x_df, y_df)
    best_params_xgb = xgb_optimizer.best_params_
    output = {"XGB": best_params_xgb, 
            "SVR": np.nan,     
            "GP": np.nan,
            "KR": np.nan,
            "RF": np.nan,
            }
    
    if not xgb_only:
        rf_optimizer.fit(x_df, y_df)
        best_params_rf = rf_optimizer.best_params_

        svr_optimizer.fit(x_df, y_df)
        best_params_svr = svr_optimizer.best_params_

        kr_optimizer.fit(x_df, y_df)
        best_params_kr = kr_optimizer.best_params_

        gp_optimizer.fit(x_df, y_df)
        best_params_gp = gp_optimizer.best_params_

        output = {"XGB": best_params_xgb, 
                  "SVR": best_params_svr,     
                  "GP": best_params_gp,
                  "KR": best_params_kr,
                  "RF": best_params_rf,
                  }

    return output

def initModelRegressors(params):
    '''
        A function that initializes the regressors given a dictionary of hyperparameters for XGB, SVR, GP, KR and RF models

        Parameters: 
            params (dict): a dictionary of parameters for a set of models. Example below. 
                        = {"XGB": best_params_xgb, 
                        "SVR": best_params_svr,     
                        "GP": best_params_gp,
                        "KR": best_params_kr,
                        "RF": best_params_rf,
                        }
                
        Output:
            regressor_dict (dict): a dictionary of regressors for 5 differentm odels
    '''
 
    # create model dicts setting it up with the params
    regressor_dict = {
        "XGB": xgb.XGBRegressor(**params["XGB"]),
        "SVR": SVR(**params["SVR"]),
        "GP": GaussianProcessRegressor(**params["GP"]),
        "KR": KernelRidge(**params["KR"]),
        "RF": RandomForestRegressor(**params["RF"]),
    }

    return regressor_dict

def runModels(regressor_dict, X, y, target, uniq_id = ""):
    '''
        A function that trains models with their given regressor and retrieves and saves the output in a .csv

        Parameters: 
            regressor_dict (dict): {model name: regressor}
            X (pd dataframe): model input
            y (pd dataframe): model target output
            uniq_id (str): a string that is included in naming the model file output
                
        Output:
            NONE
            saves model output in .csv
    '''
    with open('categories.pkl', 'rb') as file:
        categories_dict = pickle.load(file)
        # print(categories_dict)

    for model_name, regressor in regressor_dict.items():
        print(f"RUNNING {model_name}")

        categories = categories_dict[uniq_id]
        ttest_output = train_test_model(model_name, regressor, X, y, target, repetitions=200)

        # save model data output 
        ttest_df = pd.DataFrame(ttest_output)
        if not os.path.exists("model_results/"):
            # Check if an output folder exist if not make one
            os.makedirs("model_results/")
        ttest_df.to_csv(f"model_results/{model_name}_{uniq_id}.csv")
        # if not ttest_features_df.empty:
        #     ttest_features_df.to_csv(f"model_results/{model_name}_{uniq_id}_featureRanks.csv")

        t_LOOCV_output = train_model_LOOCV(model_name, regressor, X, y, target, categories=categories)
        
        t_LOOCV_df = pd.DataFrame(t_LOOCV_output)
        if not os.path.exists("model_results_LOOCV/"):
            os.makedirs("model_results_LOOCV/")
        t_LOOCV_df.to_csv(f"model_results_LOOCV/{model_name}_LOOCV_{uniq_id}.csv")
        # if not ttest_features_df.empty:
        #     t_LOOCV_features_df.to_csv(f"model_results_LOOCV/{model_name}_LOOCV_{uniq_id}_featureRanks.csv")

def main(infile_path, n_iters, target, get_hyperparams = True, get_models = True, XGB_only = False, bayes = True):
    '''
        A function that takes in a dataset and runs models on four unique datasets that are subsets of the original set. 
        Results are saved in .csv files. 

        Parameters: 
            infile_path (str): the path to the monomer featurized dataset that has been cleaned, imputed and one-hot encoded
            n_iters (int): the number of iterations for the optimizer
            target (str): enthalpy ("dH (kJ/mol)") or entropy ("dS (J/mol/K)")

        Output:
            NONE
    '''

    x_df_dict, y_df_dict = createDataSets(infile_path, target)

    print("UNIQUE DATASETS CREATED:")
    print("    1: Full + solvent parameters")
    print("    2: Full without solvent parameters")
    print("    3: Solvents only (with solvent parameters)")
    print("    4: Bulk only")

    if get_hyperparams:
        hyperparam_dicts = splitDatasetHyperparams(n_iters, x_df_dict, y_df_dict, xgb_only = XGB_only, bayes=bayes)
    else:
        with open('hyperparams.pkl', 'rb') as file:
            hyperparam_dicts = pickle.load(file)

    h_1 = hyperparam_dicts["1"]
    h_2 = hyperparam_dicts["2"]
    h_3 = hyperparam_dicts["3"]
    h_4 = hyperparam_dicts["4"]

    if get_models:
        if XGB_only:
            # create model dicts setting it up with the params
            models_1 = {
                "XGB": xgb.XGBRegressor(**h_1["XGB"]),
            }
            print("MODELS FOR DATASET 1 CREATED")

            models_2 = {
                "XGB": xgb.XGBRegressor(**h_2["XGB"]),
            }

            print("MODELS FOR DATASET 2 CREATED")

            models_3 = {
                "XGB": xgb.XGBRegressor(**h_3["XGB"]),
            }

            print("MODELS FOR DATASET 3 CREATED")

            models_4 = {
                "XGB": xgb.XGBRegressor(**h_4["XGB"]),
            }

            print("MODELS FOR DATASET 4 CREATED")
        else: 
            models_1 = {
                "XGB": xgb.XGBRegressor(**h_1["XGB"]),
            }
            print("MODELS FOR DATASET 1 CREATED")

            models_2 = initModelRegressors(h_2)
            print("MODELS FOR DATASET 2 CREATED")

            models_3 = initModelRegressors(h_3)
            print("MODELS FOR DATASET 3 CREATED")

            models_4 = initModelRegressors(h_4)
            print("MODELS FOR DATASET 4 CREATED")

        model_mega_dict = {
            "1": models_1,
            "2": models_2,
            "3": models_3,
            "4": models_4,
        }

        # Store in pickle file
        with open('models.pkl', 'wb') as file:
            pickle.dump(model_mega_dict, file)
    else:
        # Open the store dictionary of models 
        with open('models.pkl', 'rb') as file:
            model_mega_dict = pickle.load(file)

    # run models
    for key, value in model_mega_dict.items():
        print(f"RUNNING MODELS FOR DATASET {key}")

        # Get dataframes
        X, y = x_df_dict[key], y_df_dict[key]

        # dictionary with keys for each model and values the corresponding regressor function
        regressor_dict = value

        xgb_feats_df = topFeatures(X, y, regressor_dict["XGB"])

        parent_directory = os.path.dirname(os.path.dirname(infile_path)) + "/"

        if not os.path.exists(parent_directory + "final_results"):
            # Check if an output folder exist if not make one
            os.makedirs(parent_directory + "final_results")

        final_feat_path = parent_directory + "final_results/XGB_featRank_" + key + ".csv"
        final_feat_path2 = parent_directory + "final_results/RF_featRank_" + key + ".csv"

        xgb_feats_df.to_csv(final_feat_path, index = False)

        if key != "1":
            RF_feats_df = topFeatures(X, y, regressor_dict["RF"])
            RF_feats_df.to_csv(final_feat_path2, index = False)

        runModels(regressor_dict, X, y, target, uniq_id = key)

    print("OPERATION COMPLETE :)")

def topFeatures(X, y, model, num_iterations = 100, num_top_features = 50):
    # Initialize dictionary to store feature counts
    feature_counts = {feature: 0 for feature in X.columns}
    # Initialize array to store feature importances
    feature_importances_sum = np.zeros(len(X.columns))

    for _ in range(num_iterations):
        model.fit(X, y)

        # Get feature importances
        feature_importances = model.feature_importances_

        # Sum up feature importances
        feature_importances_sum += feature_importances

        # Sort feature importances and select the top features
        top_indices = np.argsort(feature_importances)[::-1][:num_top_features]
        top_features = X.columns[top_indices]

        # Update feature counts
        for feature in top_features:
            feature_counts[feature] += 1

    # Sort features by their counts
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    # Store top features and their importances for the current case
    top_features_importances = [(feature, feature_importances_sum[idx] / num_iterations) for feature, idx in zip(sorted_features[:num_top_features], top_indices)]

    output_df = pd.DataFrame(columns=["FEATURE NAME", "IMPORTANCE"])

    for sorted_feature, importance in top_features_importances:
        new_row = {"FEATURE NAME": sorted_feature[0],
                            "IMPORTANCE": importance
                            }
        # Add the new row
        output_df.loc[len(output_df)] = new_row

    return output_df


# if __name__ == "__MAIN__":
#     # only works if u specifically run this file HECK YEAH
