import pandas as pd

def detect_problem_type(y):
    """
    Automatically detect whether a target is for classification
    or regression.

    Categorical targets are classification.
    Numeric targets with few distinct values are classification.
    Other numeric targets are regression.
    """

    y = y.dropna()

    if y.empty:
        raise ValueError("Target column contains no valid values.")

    unique_count = y.nunique()
    total_count = len(y)
    unique_ratio = unique_count / total_count

    # Categorical target
    if not pd.api.types.is_numeric_dtype(y):
        return "Classification"

    # Numeric target with few distinct values
    if unique_count <= 10 or unique_ratio <= 0.05:
        return "Classification"

    return "Regression"