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


strat_test_set = pd.read_parquet(os.path.join(HOUSING_PATH, "test.parquet"))
lin_reg = joblib.load(os.path.join(MODEL_PATH, "linear_regression.pkl"))
tree_reg = joblib.load(os.path.join(MODEL_PATH, "decision_tree.pkl"))
final_model = joblib.load(
    os.path.join(MODEL_PATH, "random_forest_grid_search.pkl")
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

final_predictions = lin_reg.predict(X_test_prepared)
lin_mse = mean_squared_error(y_test, final_predictions)
lin_rmse = np.sqrt(lin_mse)
logger.info("Test RMSE score linear regression: %s" % (lin_rmse))

final_predictions = tree_reg.predict(X_test_prepared)
tree_mse = mean_squared_error(y_test, final_predictions)
tree_rmse = np.sqrt(tree_mse)
logger.info("Test RMSE score decision trees: %s" % (tree_rmse))

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
logger.info("Test RMSE score Grid search: %s" % (final_rmse))
