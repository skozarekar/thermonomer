import xgboost as xgb
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import pandas as pd

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

import pickle
import os
import re
import pandas as pd
import numpy as np

class runModels:
    def __init__(self, infile_path, n_iter, target):
        self.n_iters = n_iter
        # infile path is path to imputed data file
        self.infile_path = infile_path
        # path to folder that holds all of the datasets
        self.base_path = os.path.dirname(self.infile_path)

        self.imputed_df = pd.read_csv(infile_path, index_col=0, encoding="utf-8")

        # Initialize values that will be filled in later on
        # _x does not include dH values
        self.data_frame_X = np.nan
        # _y is just dH values
        self.data_frame_y = np.nan

        self.target = np.nan

        if target == "dH":
            self.target = "dH (kJ/mol)"
        else:
            self.target = "dS (J/mol/K)"

    def main(self):
        self.createDataSets()
        # print("------------- DATA SET SPLIT INTO 4 -------------")
        self.saveHyperParams()
        print("------------- HYPER PARAMETERS SAVED -------------")
        self.runModels()
        print("------------- MODEL RUNNING COMPLETE -------------")

    def createDataSets(self):
        '''
        4 total data sets:
            1. Full + solvent parameters
            2. Full without solvent parameters
            3. Solvents only (with solvent parameters)
            4. Bulk only
        '''

        # drop cols that shouldnt be used in any case
        self.imputed_df = self.imputed_df.drop(
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
                ]
        )

        # SET 1: Full + solvent parameters
        X_full_with_solvent_params = self.imputed_df

        # SET 2: Drop solvent parameters 
        self.imputed_df = self.imputed_df.drop(
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

        # full + no solvent params
        X_full = self.imputed_df

        # Set 3: solvents only
        X_solvent_only = X_full_with_solvent_params.loc[X_full_with_solvent_params.BASE_Monomer_State_s]

        # Set 4: bulk only
        X_bulk_only = X_full.loc[X_full.BASE_Monomer_State_l]

        # Create an empty dictionary to store data frames
        # _x does not include df
        self.data_frame_X = {
            "1": X_full_with_solvent_params.drop(columns=[self.target]),
            "2": X_full.drop(columns=[self.target]),
            "3": X_solvent_only.drop(columns=[self.target]),
            "4": X_bulk_only.drop(columns=[self.target]),
        }

        # _y is just dS values
        self.data_frame_y = {
            "1": X_full_with_solvent_params[self.target],
            "2": X_full[self.target],
            "3": X_solvent_only[self.target],
            "4": X_bulk_only[self.target],
        }

        # Save as csv just to have as a record
        # Create the folder
        split_path = self.base_path + "/splits/"
        os.makedirs(split_path, exist_ok=True)
        X_full_with_solvent_params.to_csv(split_path + "split_1.csv")
        X_full.to_csv(split_path + "split_2.csv")
        X_solvent_only.to_csv(split_path + "split_3.csv")
        X_bulk_only.to_csv(split_path + "split_4.csv")

    def saveHyperParams(self):        
        # save params for set 1 / XGB model
        xgb_case_1 = self.saveHyperParamsHelper("1")

        print("------- Case 1 Hyper Params Complete -------")

        # save params for set 2 / all models
        best_params_2 = self.saveHyperParamsHelper("2")

        print("------- Case 2 Hyper Params complete -------")

        # save params for set 3 (solution phase only) / all models
        best_params_3 = self.saveHyperParamsHelper("3")

        print("------- Case 3 Hyper Params complete -------")

        # save params for set 4 (bulk phase only) / all models
        best_params_4 = self.saveHyperParamsHelper("4")

        print("------- Case 4 Hyper Params complete -------")

        final_param_list = [xgb_case_1] + best_params_2 + best_params_3 + best_params_4

        print(f"------- Final params list: {final_param_list} -------")
                                                    
        # pickle it
        with open('hyperparameters.pkl', 'wb') as f:
            pickle.dump(final_param_list, f)

    def runModels(self):
        # bring in parameters
        print("------------- INSIDE RUNMODELS -------------")

        f = open("hyperparameters.pkl", "rb")

        (   best_params_xgb_case_1, best_params_xgb_case_2, best_params_xgb_case_3, best_params_xgb_case_4, 
            best_params_rf_case_2, best_params_rf_case_3, best_params_rf_case_4, 
            best_params_svr_case_2, best_params_svr_case_3, best_params_svr_case_4,
            best_params_gp_case_2, best_params_gp_case_3, best_params_gp_case_4,
            best_params_kr_case_2, best_params_kr_case_3, best_params_kr_case_4,
        )  = pickle.load(f)

        f.close()
        print("------------- PICKLE FILE OPENED -------------")

        # create model dicts setting it up with the params
        models_1 = {
            "XGB": xgb.XGBRegressor(**best_params_xgb_case_1),
        }

        print("------------- MODEL 1 CREATED -------------")


        models_2 = {
            "XGB": xgb.XGBRegressor(**best_params_xgb_case_2),
            "SVR": SVR(**best_params_svr_case_2),
            "GP": GaussianProcessRegressor(**best_params_gp_case_2),
            "KR": KernelRidge(**best_params_kr_case_2),
            "RF": RandomForestRegressor(**best_params_rf_case_2),
            #  "NN": create_nn_model(),
        }

        print("------------- MODEL 2 CREATED -------------")

        models_3 = {
            "XGB": xgb.XGBRegressor(**best_params_xgb_case_3),
            "SVR": SVR(**best_params_svr_case_3),
            "GP": GaussianProcessRegressor(**best_params_gp_case_3),
            "KR": KernelRidge(**best_params_kr_case_3),
            "RF": RandomForestRegressor(**best_params_rf_case_3),
            #  "NN": create_nn_model(),
        }

        print("------------- MODEL 3 CREATED -------------")

        models_4 = {
            "XGB": xgb.XGBRegressor(**best_params_xgb_case_4),
            "SVR": SVR(**best_params_svr_case_4),
            "GP": GaussianProcessRegressor(**best_params_gp_case_4),
            "KR": KernelRidge(**best_params_kr_case_4),
            "RF": RandomForestRegressor(**best_params_rf_case_4),
            #  "NN": create_nn_model(),
        }

        print("------------- MODEL 4 CREATED -------------")

        model_mega_dict = {
            "1": models_1,
            "2": models_2,
            "3": models_3,
            "4": models_4,
        }

        # run models
        for key, value in model_mega_dict.items():
            print(f"------------- RUNNING MODEL: {key}, {value} -------------")

            # Get dataframes
            X, y = self.data_frame_X[key], self.data_frame_y[key]

            # Loop through the
            model_dict = value
            for model_name, regressor in model_dict.items():
                print(f"----- model name: {model_name},regressor: {regressor} -----")

                self.train_test_model(model_name, regressor, key, X, y, "model_results", repetitions=200)
                print("TRAIN TEST MODEL COMPLETE")
                self.train_model_LOOCV(model_name, regressor, key, X, y, "model_results_LOOCV")
                print("TRAIN MODEL LOOCV COMPLETE")

# --------- Helper Functions --------- #
    def searchSpaceInit(self):
        # Define parameter space for XGB
        xgb_param_space = {'learning_rate': (0.01, 0.3),
                    'n_estimators': (100, 300),
                    'max_depth': (3, 9),
                    'min_child_weight': (1, 5),
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

    def optimizerInit(self):
        xgb_param_space, rf_param_space, svr_param_space,kr_param_space, gp_param_space = self.searchSpaceInit()

        # Initialize BayesSearchCV for each model: optimize paramters for 12hr / test case
        ### baysian search replace with something else. grid search cv maybe. bayes search doesn't need to be this specifically look into other options
        # grid search?
        xgb_optimizer = BayesSearchCV(estimator=xgb.XGBRegressor(),
                        search_spaces=xgb_param_space,
                        n_iter=self.n_iters,
                        cv=5,
                        random_state=42)

        rf_optimizer = BayesSearchCV(estimator=RandomForestRegressor(),
                                    search_spaces=rf_param_space,
                                    n_iter=self.n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        svr_optimizer = BayesSearchCV(estimator=SVR(),
                                    search_spaces=svr_param_space,
                                    n_iter=self.n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        kr_optimizer = BayesSearchCV(estimator=KernelRidge(),
                                    search_spaces=kr_param_space,
                                    n_iter=self.n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)

        gp_optimizer = BayesSearchCV(estimator=GaussianProcessRegressor(kernel=RBF()),
                                    search_spaces=gp_param_space,
                                    n_iter=self.n_iters,
                                    cv=5,
                                    random_state=42,
                                    n_jobs=-1)
        
        return xgb_optimizer, rf_optimizer, svr_optimizer, kr_optimizer, gp_optimizer

    def saveHyperParamsHelper(self, num_str):
        print(f"------- inside params helper for {num_str} -------")
        xgb_optimizer, rf_optimizer, svr_optimizer, kr_optimizer, gp_optimizer = self.optimizerInit()

        xgb_optimizer.fit(self.data_frame_X[num_str], self.data_frame_y[num_str])
        best_params_xgb = xgb_optimizer.best_params_

        if num_str == "1":
            return best_params_xgb
        else:
            rf_optimizer.fit(self.data_frame_X[num_str], self.data_frame_y[num_str])
            best_params_rf = rf_optimizer.best_params_

            svr_optimizer.fit(self.data_frame_X[num_str], self.data_frame_y[num_str])
            best_params_svr = svr_optimizer.best_params_

            kr_optimizer.fit(self.data_frame_X[num_str], self.data_frame_y[num_str])
            best_params_kr = kr_optimizer.best_params_

            gp_optimizer.fit(self.data_frame_X[num_str], self.data_frame_y[num_str])
            best_params_gp = gp_optimizer.best_params_

            # output = {"xgb": best_params_xgb, 
            #           "rf": best_params_rf,
            #           "svr": best_params_svr,
            #           "kr": best_params_kr,
            #           "gp": best_params_gp
            #           }

            return [best_params_xgb,best_params_rf,best_params_svr,best_params_kr,best_params_gp]

    def create_nn_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=173, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model
        
    # Function to run a model with full LOO cross-validation
    def train_model_LOOCV(self,model_name, regressor, feature_subset, X, y, results_folder):
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

    def train_test_model(self, model_name, regressor, feature_subset, X, y, results_folder, repetitions=200):
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
            if model_name == "NN":
                regressor.fit(X_train_scaled, y_train, epochs=10, batch_size=10, verbose=0)
            elif model_name == "RF":
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

    def train_test_model_top_feats(self,model_name, regressor, feature_subset, X, y, results_folder, num_features, top_features_list, repetitions=200):
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

            if model_name != "RF":
                # Scale Data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

            # Fit your regressor on the training data
            if model_name == "NN":
                regressor.fit(X_train_scaled, y_train, epochs=10, batch_size=10, verbose=0)
            elif model_name == "RF":
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
        mae_score_df.to_csv(f"{results_folder}/{model_name}_{feature_subset}_{num_features}_features.csv")