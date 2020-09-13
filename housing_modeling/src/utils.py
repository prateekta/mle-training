import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

path = Path(os.path.dirname(os.path.abspath(__file__)))
LOGGING_PATH = os.path.join(path.parent.parent, "housing_modeling/logs")
HOUSING_PATH = os.path.join(path.parent.parent, "datasets/housing")
MODEL_PATH = os.path.join(path.parent.parent, "datasets/modeling")


def create_logger(logging_path, name):
    os.makedirs(logging_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "'%(asctime)s %(name)-12s %(levelname)-8s %(message)s'"
    )
    file_handler = logging.FileHandler(os.path.join(logging_path, name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def process_data(is_train):
    if is_train:
        file = "train.parquet"
    else:
        file = "test.parquet"
    strat_set = pd.read_parquet(os.path.join(HOUSING_PATH, file))
    housing = strat_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)

    if is_train:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(housing_num)
        joblib.dump(imputer, os.path.join(MODEL_PATH, "imputer.pkl"))
    else:
        imputer = joblib.load(os.path.join(MODEL_PATH, "imputer.pkl"))

    X = imputer.transform(housing_num)

    housing_new = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_new["rooms_per_household"] = (
        housing_new["total_rooms"] / housing_new["households"]
    )
    housing_new["bedrooms_per_room"] = (
        housing_new["total_bedrooms"] / housing_new["total_rooms"]
    )
    housing_new["population_per_household"] = (
        housing_new["population"] / housing_new["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_new.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    return housing_prepared, housing_labels


def predict(data):
    housing_num = data.drop("ocean_proximity", axis=1)

    imputer = joblib.load(os.path.join(MODEL_PATH, "imputer.pkl"))

    X = imputer.transform(housing_num)

    housing_new = pd.DataFrame(
        X, columns=housing_num.columns, index=data.index
    )
    housing_new["rooms_per_household"] = (
        housing_new["total_rooms"] / housing_new["households"]
    )
    housing_new["bedrooms_per_room"] = (
        housing_new["total_bedrooms"] / housing_new["total_rooms"]
    )
    housing_new["population_per_household"] = (
        housing_new["population"] / housing_new["households"]
    )

    housing_new["ocean_proximity_INLAND"] = int(
        data[["ocean_proximity"]].values[0][0] == "INLAND"
    )
    housing_new["ocean_proximity_NEAR BAY"] = int(
        data[["ocean_proximity"]].values[0][0] == "NEAR BAY"
    )
    housing_new["ocean_proximity_ISLAND"] = int(
        data[["ocean_proximity"]].values[0][0] == "ISLAND"
    )
    housing_new["ocean_proximity_NEAR OCEAN"] = int(
        data[["ocean_proximity"]].values[0][0] == "NEAR OCEAN"
    )

    model = joblib.load(
        os.path.join(MODEL_PATH, "random_forest_grid_search.pkl")
    )

    final_predictions = model.predict(housing_new)
    return final_predictions[0]
