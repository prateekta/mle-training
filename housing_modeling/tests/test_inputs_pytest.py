import pandas as pd
import pytest

from housing_modeling.src.create_dataset import (
    income_cat_proportions,
    load_housing_data,
)


def test_income_proportion():  # (data, expected):
    data = pd.DataFrame({"income_cat": ["a", "a", "b", "a", "c", "c", "a"]})
    expected = pd.Series(
        [0.57, 0.29, 0.14], name="income_cat", index=["a", "c", "b"]
    )
    output = income_cat_proportions(data)
    out = all(round(output, 2).eq(expected))
    assert out


def test_income_proportion2():  # (data, expected):
    data = pd.DataFrame({"income_cat": []})
    expected = pd.Series(name="income_cat", dtype="object")
    output = income_cat_proportions(data)
    out = all(round(output, 2).eq(expected))
    assert out


first_df = pd.DataFrame({"income_cat": ["a", "a", "b", "a", "c", "c", "a"]})
first_out = pd.Series(
    [0.57, 0.29, 0.14], name="income_cat", index=["a", "c", "b"]
)
second_df = pd.DataFrame({"income_cat": []})
second_out = pd.Series(name="income_cat", dtype="object")


# @pytest.mark.parametrize(
#     "data", "expected", (first_df, first_out), (second_df, second_out)
# )


# @pytest.mark.parametrize(
#     "data",
#     "expected",
#     (
#         pd.DataFrame({"income_cat": ["a", "a", "b", "a", "c", "c", "a"]}),
#         pd.Series(
#             [0.57, 0.29, 0.14], name="income_cat", index=["a", "c", "b"]
#         ),
#     ),
#     (
#         pd.DataFrame({"income_cat": []}),
#         pd.Series(name="income_cat", dtype="object"),
#     ),
# )
# def test_income_proportion3(data, expected):
#     output = income_cat_proportions(data)
#     out = all(round(output, 2).eq(expected))
#     assert out
