import os

import joblib
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from utils import LOGGING_PATH, MODEL_PATH, create_logger, process_data


def train_linear_regression(X, y, logger):
    """
    Trains a linear regression model on the inputs and saves the model
    as a pickle file

    Parameters:
    X (pd.DataFrame): training data
    y (pd.Series):  value to be predicted
    """
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    housing_predictions = lin_reg.predict(X)
    lin_mse = mean_squared_error(y, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    logger.info("linear regression rmse: %s" % (lin_rmse))
    lin_mae = mean_absolute_error(y, housing_predictions)
    logger.info("linear regression mae: %s" % (lin_mae))
    joblib.dump(lin_reg, os.path.join(MODEL_PATH, "linear_regression.pkl"))


def train_decision_trees(X, y, logger):
    """
    Trains a decision tree on the inputs and saves the model as a pickle file

    Parameters:
    X (pd.DataFrame): training data
    y (pd.Series):  value to be predicted
    """
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X, y)

    housing_predictions = tree_reg.predict(X)
    tree_mse = mean_squared_error(y, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    logger.info("Decision trees rmse: %s" % (tree_rmse))
    joblib.dump(tree_reg, os.path.join(MODEL_PATH, "decision_tree.pkl"))


def train_RFR_random_search(X, y, logger):
    """
    Trains a random forest model on the inputs using random search and saves
    the model as a pickle file

    Parameters:
    X (pd.DataFrame): training data
    y (pd.Series):  value to be predicted
    """
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X, y)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logger.info(
            "random forest with random search gave score \n %s for parameters %s"
            % (str(np.sqrt(-mean_score)), str(params))
        )
    joblib.dump(
        rnd_search, os.path.join(MODEL_PATH, "random_forest_random_search.pkl")
    )


def train_RFR_grid_search(X, y, logger):
    """
    Trains a random forest model on the inputs using grid search and saves
    the model as a pickle file

    Parameters:
    X (pd.DataFrame): training data
    y (pd.Series):  value to be predicted
    """
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X, y)

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logger.info(
            "random forest with grid search gave score \n %s for parameters %s"
            % (str(np.sqrt(-mean_score)), str(params))
        )

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, X.columns), reverse=True)

    final_model = grid_search.best_estimator_
    joblib.dump(
        final_model, os.path.join(MODEL_PATH, "random_forest_grid_search.pkl")
    )


if __name__ == "__main__":

    housing_prepared, housing_labels = process_data(is_train=True)

    logger = create_logger(LOGGING_PATH, "train.log")

    os.makedirs(MODEL_PATH, exist_ok=True)

    train_linear_regression(housing_prepared, housing_labels, logger)
    train_decision_trees(housing_prepared, housing_labels, logger)
    train_RFR_random_search(housing_prepared, housing_labels, logger)
    train_RFR_grid_search(housing_prepared, housing_labels, logger)
