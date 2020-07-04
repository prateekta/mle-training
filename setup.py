from setuptools import find_packages, setup

exec(open("housing_modeling/src/_version.py").read())
setup(
    name="housing_predicion",
    author="Prateek",
    author_email="prateek.agarwal@tigeranalytics.com",
    version=__version__,  # noqa
    packages=find_packages(),
    description="Train and predict multiple models on housing prediction",
)
