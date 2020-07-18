.. _intro:

Introduction
============

What is housing_prediction?
---------------------------
The library is used for making median housing value predictions. The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

* Linear regression
* Decision Tree
* Random Forest

Steps performed
---------------
* We prepare and clean the data. We check and impute for missing values.
* Features are generated and the variables are checked for correlation.
* Multiple sampling techinuqies are evaluated. The data set is split into train and test.
* All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

Folder structure
----------------
All the files are stored in the following manner:

| mle_prediction
| ├── housing_modeling
| │   ├── src
| │   │    ├── __init__.py
| │   │    ├── _version.py
| │   │    ├── create_dataset.py
| │   │    ├── train.py
| │   │    ├── utils.py
| │   │    └── test.py
| │   ├── __init__.py
| │   └── tests
| │        └── test_inputs_pytest.py
| ├── docs
| │   ├── build
| │   ├── make.bat
| │   ├── Makefile
| │   └── source
| ├── env.yml
| ├── env_full.yml
| ├── README.md
| ├── nonstandardcode.py
| └── setup.py
| 
| 

In the folder structure above:

- ``mle_prediction`` is the folder we get when we issue a ``git pull/clone`` command
- ``mle_prediction/docs`` is the directory where our Sphinx documentation will reside
- ``mle_prediction/housing_modeling/src`` is the directory where our main code reside
- ``mle_prediction/housing_modeling/tests`` is the directory where our unittests reside
- ``mle_prediction/env.yml`` contains the list of dependencies
- ``mle_prediction/setup.py`` makes the code packagable


Installation
------------
Ideally, one should create a separate virtual environment (using anaconda), and the dependencies can be installed using::

    >>conda env create -f env.yml -n <env_name>

housing_prediction can be converted to a python package using::

    >>python setup.py sdist bdist_wheel


Documentation
-------------
Referenced document can be generated directly from the repository using Sphinx_.