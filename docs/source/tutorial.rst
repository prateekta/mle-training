.. _tutorial:

Running the code
============

loading the data
----------------

The first thing we need to do is load the data, and split it into train and test sets. For this we create another folder ``mle_prediction/datasets/housing`` using::

    >>python housing_modeling/src/create_dataset.py

After loading and splitting the data, it'll generate the logs in ``mle_prediction/housing_modeling/logs`` and store the splitted data as parquet files in the dataset folder.

training
--------

After loading the data, we can run the training using::

    >>python housing_modeling/src/train.py

This will train the data on linear regression, decision trees and random forest, and store the model files in the folder ``mle_prediction/datasets/modeling``, along with the imputer for the training data.  The logs will also be generated and stored in the same folder as for loading the data.

training
--------
We'll first use the imputer from the training data to impute the values in test data, and after loading the model files, the logs will be generated for the mean absolute error for all the three models. The script can be run using::

    >>python housing_modeling/src/test.py


Unittesting
-----------

Loading the data is a prerequisite for unittesting. To run the tests, you can use::

    >>pytest housing_modeling/tests/test_inputs_pytest.py.py