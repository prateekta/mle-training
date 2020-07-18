import os

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

from utils import LOGGING_PATH, MODEL_PATH, create_logger, process_data


def get_test_score(name, X, y, logger):
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

    logger = create_logger(LOGGING_PATH, "test.log")
    X_test_prepared, y_test = process_data(is_train=False)

    get_test_score("linear_regression", X_test_prepared, y_test, logger)
    get_test_score("decision_tree", X_test_prepared, y_test)
    get_test_score("random_forest_grid_search", X_test_prepared, y_test)
