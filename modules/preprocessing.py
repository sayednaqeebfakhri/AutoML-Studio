
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def separate_features_target(df, target_column):
    """
    Separate input features (X) from the target (y).
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if df[target_column].isna().all():
        raise ValueError(
            "The selected target column contains only missing values."
        )

    if df[target_column].isna().any():
        raise ValueError(
            "The target column contains missing values. "
            "Please remove those rows before training."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.shape[1] == 0:
        raise ValueError(
            "The dataset must contain at least one feature column."
        )

    return X, y


def identify_column_types(X):
    """
    Identify numeric and categorical columns.
    """

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


def build_preprocessor(numeric_columns, categorical_columns):
    """
    Build preprocessing pipelines for numeric and categorical data.
    """

    transformers = []

    # Numeric pipeline
    if numeric_columns:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            ),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    # Categorical pipeline
    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True
                )
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError(
            "No numeric or categorical features were provided."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return preprocessor
