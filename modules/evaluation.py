
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# CLASSIFICATION EVALUATION
# ==================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    return metrics


def compare_models(results):

    if not results:
        raise ValueError("No classification results available.")

    comparison = pd.DataFrame(results).T

    if "F1 Score" not in comparison.columns:
        raise ValueError("F1 Score is missing from evaluation results.")

    return comparison.sort_values(
        by="F1 Score",
        ascending=False
    )


def get_classification_report(model, X_test, y_test):

    predictions = model.predict(X_test)

    return classification_report(
        y_test,
        predictions,
        zero_division=0
    )


# ==================================================
# REGRESSION EVALUATION
# ==================================================

def evaluate_regression_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    # R² is undefined with fewer than two test samples.
    r2 = (
        r2_score(y_test, predictions)
        if len(y_test) >= 2
        else float("nan")
    )

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R² Score": r2
    }

    return metrics


def compare_regression_models(results):

    if not results:
        raise ValueError("No regression results available.")

    comparison = pd.DataFrame(results).T

    if "RMSE" not in comparison.columns:
        raise ValueError("RMSE is missing from evaluation results.")

    return comparison.sort_values(
        by="RMSE",
        ascending=True
    )
