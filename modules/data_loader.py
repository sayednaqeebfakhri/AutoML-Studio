import pandas as pd


def load_csv(file):
    """
    Load a CSV file into a Pandas DataFrame.
    Automatically detects the delimiter.
    """

    try:
        df = pd.read_csv(file, sep=None, engine="python")

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        return df

    except Exception as e:
        raise ValueError(f"Could not load CSV file: {e}")


def validate_dataset(df):
    """
    Perform basic dataset validation.
    """

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    if df.shape[1] < 2:
        raise ValueError(
            "Dataset must contain at least two columns."
        )

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    return True