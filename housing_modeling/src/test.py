import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from utils import create_logger

path = Path(os.getcwd())

LOGGING_PATH = os.path.join(path.parent.parent, "housing_modeling/logs")
HOUSING_PATH = os.path.join(path.parent.parent, "datasets/housing")
MODEL_PATH = os.path.join(path.parent.parent, "datasets/modeling")


def get_test_score(name, X, y):
    """
    logs the test score for a given model name and test dataset

    Parameters:
    X (pd.DataFrame): testing data
    y (pd.Series):  value to be predicted
    """
    model_path = os.path.join(MODEL_PATH, name + ".pkl")
    model = joblib.load(model_path)
    final_predictions = model.predict(X)
    mse = mean_squared_error(y, final_predictions)
    rmse = np.sqrt(mse)
    logger.info("Test RMSE score %s: %s" % (name, rmse))


if __name__ == "__main__":
    strat_test_set = pd.read_parquet(
        os.path.join(HOUSING_PATH, "test.parquet")
    )
    imputer = joblib.load(os.path.join(MODEL_PATH, "imputer.pkl"))
    logger = create_logger(LOGGING_PATH, "test.log")

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    get_test_score("linear_regression", X_test_prepared, y_test)
    get_test_score("decision_tree", X_test_prepared, y_test)
    get_test_score("random_forest_grid_search", X_test_prepared, y_test)
