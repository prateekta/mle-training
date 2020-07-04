import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .utils import create_logger

path = Path(os.getcwd())

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(path.parent.parent, "datasets/housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
LOGGING_PATH = os.path.join(path.parent.parent, "housing_modeling/logs")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    fetches the data from the given url and extracts its contents

    Parameters:
    housing_url (str): url of the dataset
    housing_path (str): location to store the data
    """
    if os.path.isfile(HOUSING_PATH + "/housing.tgz"):
        return
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    loads the data from the given path and returns the dataframe

    Parameters:
    housing_path (str): location to store the data

    Returns:
    pd.DataFrame: housing data
    """
    try:
        csv_path = os.path.join(housing_path, "housing.csv")
    except FileNotFoundError:
        raise "file does not exist"

    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """
    returns the value counts of income categories

    Parameters:
    data (pd.DataFrame): the dataset
    
    Returns:
    pd.DataFrame
    """
    return data["income_cat"].value_counts() / len(data)


if __name__ == "__main__":
    fetch_housing_data()

    housing = load_housing_data()
    logger = create_logger(LOGGING_PATH, "dataset.log")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )
    logger.info("The probability values are\n %s" % (compare_props))

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    strat_train_set.to_parquet(os.path.join(HOUSING_PATH, "train.parquet"))
    strat_test_set.to_parquet(os.path.join(HOUSING_PATH, "test.parquet"))
