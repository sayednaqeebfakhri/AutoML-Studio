from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def get_classification_models():
    """
    Return a dictionary of classification models.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=42
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        )
    }
    return models

def get_regression_models():
    """
    Return a collection of regression models.
    """
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42
        ),

        "Support Vector Regressor": SVR(),

        "K-Nearest Neighbors Regressor": KNeighborsRegressor(
            n_neighbors=5
        )
    }
    return models


def train_model(model, X_train, y_train):
    """
    Train a machine learning model.
    """

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    if len(X_train) != len(y_train):
        raise ValueError("Features and target have different numbers of samples.")

    # Adjust KNN neighbors for small training datasets.
    if hasattr(model, "n_neighbors"):
        model.n_neighbors = min(
            model.n_neighbors,
            len(X_train)
        )

    model.fit(X_train, y_train)

    return model


def predict_model(model, X):
    """
    Generate predictions using a trained model.
    """

    predictions = model.predict(X)

    return predictions


def predict_probabilities(model, X):
    """
    Generate prediction probabilities.
    """

    predict_proba = getattr(model, "predict_proba", None)

    if not callable(predict_proba):
        raise ValueError(
            "This model does not support probability predictions."
        )

    probabilities = predict_proba(X)

    return probabilities