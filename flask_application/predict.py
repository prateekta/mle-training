import os
from pathlib import Path

import joblib
import pandas as pd

path = Path(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(path.parent, "datasets/modeling/")


def process_data(data):
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

    return housing_new


def make_prediction(form):
    dictionary = {}
    dictionary["longitude"] = float(form.latitude.data)
    dictionary["latitude"] = float(form.latitude.data)
    dictionary["housing_median_age"] = float(form.housing_median_age.data)
    dictionary["total_rooms"] = float(form.total_rooms.data)
    dictionary["total_bedrooms"] = float(form.total_bedrooms.data)
    dictionary["population"] = float(form.population.data)
    dictionary["households"] = float(form.households.data)
    dictionary["median_income"] = float(form.median_income.data)
    dictionary["ocean_proximity"] = str(form.latitude.data).upper()

    data = pd.DataFrame(dictionary, index=[0])
    X = process_data(data)
    model = joblib.load(
        os.path.join(MODEL_PATH, "random_forest_grid_search.pkl")
    )

    final_predictions = model.predict(X)

    return final_predictions[0]
