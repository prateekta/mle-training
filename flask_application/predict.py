import os
import sys
from pathlib import Path

import pandas as pd

path = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
MODEL_PATH = os.path.join(path.parent, "datasets/modeling/")

from housing_modeling.src.utils import predict  # isort:skip # noqa


def make_prediction(dictionary={}, form=None):
    if form is not None:
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
    return predict(data)


# requests.post(url,json={'longitude':2, 'latitude':2, 'housing_median_age':1000, 'total_rooms':2, 'total_bedrooms':3, 'population':2, 'households':3, 'median_income':10000, 'ocean_proximity':"ISLAND"})
# curl -i -H "Content-Type: application/json" -X POST -d '{"longitude":2, "latitude":2, "housing_median_age":1000, "total_rooms":2, "total_bedrooms":3, "population":2, "households":3, "median_income":10000, "ocean_proximity":"ISLAND"}' http://localhost:5000/predict
