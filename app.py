import streamlit as st
import pandas as pd
import io
import hashlib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modules.preprocessing import (
    identify_column_types,
    build_preprocessor
)

from modules.model_training import (
    get_classification_models,
    get_regression_models
)

from modules.evaluation import (
    evaluate_model,
    get_classification_report,
    compare_models,
    evaluate_regression_model,
    compare_regression_models
)

from modules.task_detection import detect_problem_type

from modules.data_loader import load_csv, validate_dataset
from modules.pdf_report import generate_pdf_report


from modules.eda import (
    get_dataset_overview,
    get_column_summary,
    get_numeric_statistics,
    get_categorical_statistics,
    get_missing_values_summary,
    get_outlier_summary,
    plot_numeric_distribution,
    plot_numeric_boxplot,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_scatter_relationship
)


def save_widget_value(widget_key, saved_key):
    """Copy a widget value to a non-widget session key before navigation cleanup."""
    if widget_key in st.session_state:
        st.session_state[saved_key] = st.session_state[widget_key]



# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="",
    layout="wide"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None


# ==========================================================
# APPLICATION TITLE
# ==========================================================

st.title("🤖 AutoML Studio")

st.write(
    "Automated Machine Learning Platform "
    "for Tabular Data"
)

st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset Analysis",
        "Model Training",
        "Predictions"
    ]
)


# ==========================================================
# GLOBAL DATASET UPLOADER
# ==========================================================

st.sidebar.divider()

st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ==========================================================
# LOAD DATASET
# ==========================================================

if uploaded_file is not None:

    try:

        file_bytes = uploaded_file.getvalue()
        new_fingerprint = hashlib.sha256(file_bytes).hexdigest()

        df = load_csv(
            io.BytesIO(file_bytes)
        )

        validate_dataset(df)

        old_fingerprint = st.session_state.get("dataset_fingerprint")

        if (
            old_fingerprint is not None
            and old_fingerprint != new_fingerprint
        ):
            keys_to_clear = [
                "trained_models",
                "best_model",
                "best_model_name",
                "comparison",
                "X_test",
                "y_test",
                "training_results",
                "classification_report",
                "training_signature",
                "target_column",
                "problem_type_choice",
                "last_prediction",
                "prediction_probabilities",
                "prediction_inputs",
                "shap_results",
                "shap_error"
            ]

            for key in keys_to_clear:
                st.session_state.pop(key, None)

            for key in list(st.session_state.keys()):
                if key.startswith("prediction_input_") or key.startswith("saved_prediction_input_"):
                    st.session_state.pop(key, None)

        st.session_state["dataset"] = df
        st.session_state["dataset_name"] = uploaded_file.name
        st.session_state["dataset_fingerprint"] = new_fingerprint

    except Exception as e:

        st.session_state["dataset"] = None
        st.session_state["dataset_name"] = None

        st.sidebar.error(
            f"Could not load dataset: {e}"
        )


# Get shared dataset
dataset = st.session_state["dataset"]


# ==========================================================
# DATASET STATUS
# ==========================================================

if dataset is not None:

    st.sidebar.success("Dataset loaded!")

    st.sidebar.write(
        f"**File:** {st.session_state['dataset_name']}"
    )

    st.sidebar.write(
        f"**Rows:** {dataset.shape[0]}"
    )

    st.sidebar.write(
        f"**Columns:** {dataset.shape[1]}"
    )

else:

    st.sidebar.info(
        "No dataset uploaded yet."
    )


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.header("Welcome to AutoML Studio")

    st.write(
        "Upload your tabular dataset, explore your data, "
        "train machine learning models, and generate predictions."
    )

    st.info(
        "Start by uploading your CSV dataset from the sidebar."
    )

    if dataset is not None:

        st.success(
            f"Your dataset '{st.session_state['dataset_name']}' "
            "is ready to use."
        )


# ==========================================================
# DATASET ANALYSIS PAGE
# ==========================================================

elif page == "Dataset Analysis":

    st.header("Dataset Analysis")

    if dataset is None:

        st.warning(
            "Please upload a CSV dataset from the sidebar."
        )

    else:

        st.success("Dataset uploaded successfully!")

        # ==================================================
        # 1. DATASET OVERVIEW
        # ==================================================

        st.subheader("1. Dataset Overview")

        overview = get_dataset_overview(dataset)

        numeric_columns = dataset.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = dataset.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        constant_columns = [
            column for column in dataset.columns
            if dataset[column].nunique(dropna=False) <= 1
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Rows", overview["Total Rows"])
        col2.metric("Total Columns", overview["Total Columns"])
        col3.metric("Numerical Columns", len(numeric_columns))
        col4.metric("Categorical Columns", len(categorical_columns))

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Missing Values",
            overview["Missing Values"]
        )

        col2.metric(
            "Duplicate Rows",
            overview["Duplicate Rows"]
        )

        col3.metric(
            "Memory Usage (KB)",
            overview["Memory Usage (KB)"]
        )

        st.write("**Constant Columns:**", constant_columns or "None")

        # ==================================================
        # 2. DATASET PREVIEW
        # ==================================================

        st.divider()
        st.subheader("2. Dataset Preview")

        st.dataframe(
            dataset.head(10),
            use_container_width=True
        )

        # ==================================================
        # 3. COLUMN INFORMATION
        # ==================================================

        st.divider()
        st.subheader("3. Column Information")

        st.dataframe(
            get_column_summary(dataset),
            use_container_width=True
        )

        # ==================================================
        # 4. DATA QUALITY ANALYSIS
        # ==================================================

        st.divider()
        st.subheader("4. Data Quality Analysis")

        st.write("Missing values by column:")

        missing_summary = get_missing_values_summary(dataset)

        missing_summary = missing_summary[
            missing_summary["Missing Values"] > 0
        ]

        if missing_summary.empty:

            st.success("No missing values found!")

        else:

            st.dataframe(
                missing_summary,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            missing_summary["Missing Values"].plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Missing Values by Column")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Missing Count")
            ax.tick_params(axis="x", rotation=45)

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        st.write("Potential numerical outliers:")

        outlier_summary = get_outlier_summary(dataset)

        if outlier_summary.empty:

            st.info("No numerical columns available.")

        else:

            st.dataframe(
                outlier_summary,
                use_container_width=True
            )

        # ==================================================
        # 5. STATISTICAL ANALYSIS
        # ==================================================

        st.divider()
        st.subheader("5. Statistical Analysis")

        with st.expander("Numerical Statistics", expanded=True):

            if numeric_columns:

                st.dataframe(
                    get_numeric_statistics(dataset),
                    use_container_width=True
                )

            else:

                st.info("No numerical columns available.")

        with st.expander("Categorical Statistics"):

            if categorical_columns:

                categorical_stats = (
                    get_categorical_statistics(dataset)
                )

                for column, statistics in categorical_stats.items():

                    st.write(f"**{column}**")

                    st.dataframe(
                        statistics.head(20).to_frame("Count"),
                        use_container_width=True
                    )

            else:

                st.info("No categorical columns available.")

        # ==================================================
        # 6. NUMERICAL VISUALIZATIONS
        # ==================================================

        st.divider()
        st.subheader("6. Numerical Data Visualization")

        if numeric_columns:

            selected_numeric = st.selectbox(
                "Select a numerical column",
                options=numeric_columns,
                key="eda_numeric_column"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write("**Distribution Histogram**")

                fig = plot_numeric_distribution(
                    dataset,
                    selected_numeric
                )

                st.pyplot(fig)
                plt.close(fig)

            with col2:

                st.write("**Box Plot**")

                fig = plot_numeric_boxplot(
                    dataset,
                    selected_numeric
                )

                st.pyplot(fig)
                plt.close(fig)

        else:

            st.info("No numerical columns available.")

        # ==================================================
        # 7. CATEGORICAL VISUALIZATION
        # ==================================================

        st.divider()
        st.subheader("7. Categorical Data Visualization")

        if categorical_columns:

            selected_categorical = st.selectbox(
                "Select a categorical column",
                options=categorical_columns,
                key="eda_categorical_column"
            )

            fig = plot_categorical_distribution(
                dataset,
                selected_categorical
            )

            st.pyplot(fig)
            plt.close(fig)

        else:

            st.info("No categorical columns available.")

        # ==================================================
        # 8. CORRELATION ANALYSIS
        # ==================================================

        st.divider()
        st.subheader("8. Correlation Analysis")

        if len(numeric_columns) >= 2:

            st.write(
                "Explore correlations between numerical features."
            )

            fig = plot_correlation_heatmap(dataset)

            if fig is not None:

                st.pyplot(fig)
                plt.close(fig)

        else:

            st.info(
                "At least two numerical columns are required "
                "for correlation analysis."
            )

        # ==================================================
        # 9. FEATURE RELATIONSHIP ANALYSIS
        # ==================================================

        st.divider()
        st.subheader("9. Feature Relationship Analysis")

        if len(numeric_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:

                x_column = st.selectbox(
                    "Select X-axis feature",
                    options=numeric_columns,
                    index=0,
                    key="eda_x_column"
                )

            with col2:

                y_options = [
                    column for column in numeric_columns
                    if column != x_column
                ]

                y_column = st.selectbox(
                    "Select Y-axis feature",
                    options=y_options,
                    key="eda_y_column"
                )

            fig = plot_scatter_relationship(
                dataset,
                x_column,
                y_column
            )

            st.pyplot(fig)
            plt.close(fig)

        else:

            st.info(
                "At least two numerical columns are required "
                "for scatter plot analysis."
            )


# ==========================================================
# MODEL TRAINING PAGE
# ==========================================================

elif page == "Model Training":

    st.header("Model Training")

    if dataset is None:

        st.warning(
            "Please upload a CSV dataset from the sidebar."
        )

    else:

        st.success(
            "Dataset is available for model training!"
        )

        st.write(
            f"Dataset: **{st.session_state['dataset_name']}**"
        )

        # ==================================================
        # TARGET SELECTION
        # ==================================================

        st.subheader("Select Target Column")

        # Optional helper for our sample student dataset
        create_pass_target = False

        if "G3" in dataset.columns:

            st.info(
                "Sample dataset detected. You may create a "
                "Pass/Fail target from G3 for testing."
            )

            create_pass_target = st.checkbox(
                "Create a Pass/Fail target using G3 "
                "(10 or above = Yes, below 10 = No)"
            )

        if create_pass_target:

            # Create working copy
            working_df = dataset.copy()

            # Create target
            working_df["Passed"] = (
                working_df["G3"] >= 10
            ).map({
                True: "Yes",
                False: "No"
            })

            target_column = "Passed"

            # Prevent target leakage
            excluded_columns = [
                "G1",
                "G2",
                "G3",
                "Passed"
            ]

            X = working_df.drop(
                columns=excluded_columns
            )

            y = working_df[target_column]

            st.info(
                "G1, G2, and G3 are excluded from the "
                "features because the target is derived from G3."
            )

        else:

            saved_target = st.session_state.get("target_column")
            target_index = (
                dataset.columns.get_loc(saved_target)
                if saved_target in dataset.columns
                else 0
            )

            target_column = st.selectbox(
                "What would you like your model to predict?",
                options=dataset.columns.tolist(),
                index=target_index,
                key="target_column_selector"
            )

            X = dataset.drop(
                columns=[target_column]
            )

            y = dataset[target_column]

        # ==================================================
        # STORE TARGET INFORMATION
        # ==================================================

        st.session_state["target_column"] = target_column
        st.session_state["X"] = X
        st.session_state["y"] = y

        # ==================================================
        # TARGET INFORMATION
        # ==================================================

        st.divider()

        st.subheader("Target Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Target Column",
            target_column
        )

        col2.metric(
            "Total Features",
            X.shape[1]
        )

        col3.metric(
            "Total Samples",
            X.shape[0]
        )

        # ==================================================
        # TARGET DISTRIBUTION
        # ==================================================

        st.subheader("Target Distribution")

        target_distribution = (
            y.value_counts(dropna=False)
        )

        st.dataframe(
            target_distribution
            .rename("Count")
            .to_frame(),
            use_container_width=True
        )

        # ==================================================
        # PROBLEM TYPE DETECTION
        # ==================================================

        st.divider()

        st.subheader("Problem Type Detection")

        try:

            detected_type = detect_problem_type(y)

            st.write(
                "Automatically detected problem type:",
                f"**{detected_type}**"
            )

            problem_type_options = [
                "Automatic",
                "Classification",
                "Regression"
            ]
            saved_problem_choice = st.session_state.get(
                "problem_type_choice", "Automatic"
            )
            if saved_problem_choice not in problem_type_options:
                saved_problem_choice = "Automatic"

            problem_type_choice = st.selectbox(
                "Choose problem type",
                options=problem_type_options,
                index=problem_type_options.index(saved_problem_choice),
                key="problem_type_selector"
            )
            st.session_state["problem_type_choice"] = problem_type_choice

            if problem_type_choice == "Automatic":
                problem_type = detected_type
            else:
                problem_type = problem_type_choice

            st.session_state["problem_type"] = problem_type

            st.info(
                f"Selected problem type: {problem_type}"
            )

        except Exception as e:

            st.error(
                f"Could not detect problem type: {e}"
            )

            problem_type = None

        # --------------------------------------------------
        # Invalidate old results if dataset/target/task changes
        # --------------------------------------------------

        current_training_signature = (
            st.session_state.get("dataset_fingerprint"),
            target_column,
            problem_type,
            tuple(X.columns)
        )

        if (
            "training_signature" in st.session_state
            and st.session_state["training_signature"]
            != current_training_signature
        ):
            keys_to_clear = [
                "trained_models",
                "best_model",
                "best_model_name",
                "comparison",
                "X_test",
                "y_test",
                "training_results",
                "classification_report",
                "training_signature",
                "last_prediction",
                "prediction_probabilities",
                "prediction_inputs",
                "shap_results",
                "shap_error"
            ]

            for key in keys_to_clear:
                st.session_state.pop(key, None)

            # Clear old prediction widget values as well.
            for key in list(st.session_state.keys()):
                if key.startswith("prediction_input_") or key.startswith("saved_prediction_input_"):
                    st.session_state.pop(key, None)

            st.info(
                "Dataset or training settings changed. "
                "Please train the models again."
            )

        # ==================================================
        # FEATURE PREVIEW
        # ==================================================

        st.subheader("Feature Preview")

        st.dataframe(
            X.head(10),
            use_container_width=True
        )

        st.success(
            "Target column selected and features "
            "prepared successfully!"
        )

        
        # ==========================================================
        # TRAINING SECTION
        # ==========================================================

        st.divider()

        st.subheader("Train and Compare Models")

        train_button_disabled = (
            problem_type not in ["Classification", "Regression"]
        )

        train_clicked = st.button(
            "🚀 Train All Models",
            type="primary",
            disabled=train_button_disabled
        )

        # ==========================================================
        # MODEL TRAINING WORKFLOW
        # ==========================================================

        if train_clicked:

            try:

                # ------------------------------------------
                # Remove missing target values
                # ------------------------------------------

                valid_rows = y.notna()

                X_clean = X.loc[valid_rows].copy()
                y_clean = y.loc[valid_rows].copy()

                # ------------------------------------------
                # Validate target
                # ------------------------------------------

                if len(y_clean) < 5:

                    st.error(
                        "The dataset must contain at least "
                        "5 valid target values."
                    )

                elif y_clean.nunique() < 2:

                    st.error(
                        "The target must contain at least "
                        "two distinct values."
                    )

                elif problem_type == "Classification" and (
                    y_clean.value_counts().min() < 2
                ):

                    st.error(
                        "Each target class must contain at least "
                        "two samples for stratified training."
                    )

                else:

                    # --------------------------------------
                    # Train/test split
                    # --------------------------------------

                    if problem_type == "Classification":

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X_clean,
                                y_clean,
                                test_size=0.2,
                                random_state=42,
                                stratify=y_clean
                            )
                        )

                    else:

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X_clean,
                                y_clean,
                                test_size=0.2,
                                random_state=42
                            )
                        )

                    st.write(
                        f"Training samples: {len(X_train)}"
                    )

                    st.write(
                        f"Testing samples: {len(X_test)}"
                    )

                    # --------------------------------------
                    # Identify feature types
                    # --------------------------------------

                    (
                        numeric_columns,
                        categorical_columns
                    ) = identify_column_types(X_train)

                    # --------------------------------------
                    # Build preprocessor
                    # --------------------------------------

                    preprocessor = build_preprocessor(
                        numeric_columns,
                        categorical_columns
                    )

                    # --------------------------------------
                    # Select models
                    # --------------------------------------

                    if problem_type == "Classification":

                        models = get_classification_models()
                        evaluation_function = evaluate_model
                        comparison_function = compare_models
                        sorting_metric = "F1 Score"

                    else:

                        models = get_regression_models()
                        evaluation_function = evaluate_regression_model
                        comparison_function = compare_regression_models
                        sorting_metric = "RMSE"

                    # --------------------------------------
                    # Initialize results
                    # --------------------------------------

                    results = {}
                    trained_models = {}

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # --------------------------------------
                    # Train all models
                    # --------------------------------------

                    for i, (name, model) in enumerate(
                        models.items()
                    ):

                        status_text.write(
                            f"Training: {name}..."
                        )

                        try:

                            pipeline = Pipeline([
                                ("preprocessor", preprocessor),
                                ("model", model)
                            ])

                            pipeline.fit(
                                X_train,
                                y_train
                            )

                            metrics = evaluation_function(
                                pipeline,
                                X_test,
                                y_test
                            )

                            results[name] = metrics
                            trained_models[name] = pipeline

                        except Exception as model_error:

                            st.warning(
                                f"{name} could not be trained: "
                                f"{model_error}"
                            )

                        progress_bar.progress(
                            (i + 1) / len(models)
                        )

                    status_text.success(
                        "Model training completed!"
                    )

                    # --------------------------------------
                    # Compare models
                    # --------------------------------------

                    if results:

                        comparison = comparison_function(
                            results
                        )

                        # ----------------------------------
                        # Select best model and save results
                        # ----------------------------------

                        best_model_name = comparison.index[0]
                        best_model = trained_models[best_model_name]

                        if problem_type == "Classification":
                            report = get_classification_report(
                                best_model,
                                X_test,
                                y_test
                            )
                        else:
                            report = None

                        st.session_state["X"] = X
                        st.session_state["target_column"] = target_column
                        st.session_state["trained_models"] = trained_models
                        st.session_state["best_model"] = best_model
                        st.session_state["best_model_name"] = best_model_name
                        st.session_state["comparison"] = comparison
                        st.session_state["X_test"] = X_test
                        st.session_state["y_test"] = y_test
                        st.session_state["problem_type"] = problem_type
                        st.session_state["training_results"] = results
                        st.session_state["classification_report"] = report
                        st.session_state["training_signature"] = (
                            st.session_state.get("dataset_fingerprint"),
                            target_column,
                            problem_type,
                            tuple(X.columns)
                        )

                        # A newly trained model starts a fresh prediction history.
                        for key in [
                            "last_prediction",
                            "prediction_probabilities",
                            "shap_results",
                            "shap_error"
                        ]:
                            st.session_state.pop(key, None)

                    else:

                        st.error(
                            "None of the models could be trained."
                        )

            except Exception as e:

                st.error(
                    f"Training failed: {e}"
                )

        # ==================================================
        # DISPLAY SAVED TRAINING RESULTS
        # ==================================================

        if "best_model" in st.session_state:

            st.divider()
            st.subheader("Saved Model Training Results")

            comparison = st.session_state["comparison"]
            best_model_name = st.session_state["best_model_name"]
            problem_type = st.session_state["problem_type"]
            best_metrics = st.session_state["training_results"][best_model_name]

            st.subheader("Model Comparison")
            st.dataframe(
                comparison.round(3),
                use_container_width=True
            )

            st.subheader("Best Model")

            if problem_type == "Classification":
                st.success(
                    f"Best model based on weighted F1 Score: {best_model_name}"
                )
                st.metric(
                    "Weighted F1 Score",
                    f"{best_metrics['F1 Score']:.3f}"
                )
                st.subheader("Classification Report")
                st.code(st.session_state["classification_report"])

            else:
                st.success(f"Best model based on RMSE: {best_model_name}")
                st.write("Lower RMSE indicates better prediction performance.")

                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{best_metrics['MAE']:.3f}")
                col2.metric("MSE", f"{best_metrics['MSE']:.3f}")

                col3, col4 = st.columns(2)
                col3.metric("RMSE", f"{best_metrics['RMSE']:.3f}")
                col4.metric("R² Score", f"{best_metrics['R² Score']:.3f}")


# ==========================================================
# PREDICTIONS PAGE
# ==========================================================

elif page == "Predictions":

    st.header("Predictions")

    # ------------------------------------------------------
    # Check whether a model has been trained
    # ------------------------------------------------------

    if "best_model" not in st.session_state:

        st.warning(
            "Please train the models first from the "
            "'Model Training' page."
        )

    else:

        best_model = st.session_state["best_model"]
        best_model_name = st.session_state["best_model_name"]
        problem_type = st.session_state["problem_type"]

        st.success(
            f"Using best model: **{best_model_name}**"
        )

        st.write(
            f"Problem type: **{problem_type}**"
        )

        st.divider()

        st.subheader("Enter Input Data")

        # --------------------------------------------------
        # Get original feature columns
        # --------------------------------------------------

        target_column = st.session_state["target_column"]

        X_original = st.session_state["X"]

        feature_columns = X_original.columns.tolist()

        input_data = {}

        # --------------------------------------------------
        # Create input fields
        # --------------------------------------------------

        for column in feature_columns:

            if pd.api.types.is_numeric_dtype(
                X_original[column]
            ):

                default_value = float(
                    X_original[column].median()
                )

                widget_key = f"prediction_input_{column}"
                saved_key = f"saved_prediction_input_{column}"

                if widget_key not in st.session_state and saved_key in st.session_state:
                    st.session_state[widget_key] = st.session_state[saved_key]

                input_data[column] = st.number_input(
                    column,
                    value=st.session_state.get(saved_key, default_value),
                    key=widget_key,
                    on_change=save_widget_value,
                    args=(widget_key, saved_key)
                )
                st.session_state[saved_key] = input_data[column]

            else:

                values = (
                    X_original[column]
                    .dropna()
                    .unique()
                    .tolist()
                )

                widget_key = f"prediction_input_{column}"
                saved_key = f"saved_prediction_input_{column}"

                if widget_key not in st.session_state and saved_key in st.session_state:
                    saved_value = st.session_state[saved_key]
                    if saved_value in values:
                        st.session_state[widget_key] = saved_value

                saved_value = st.session_state.get(saved_key)
                default_index = values.index(saved_value) if saved_value in values else 0

                input_data[column] = st.selectbox(
                    column,
                    options=values,
                    index=default_index,
                    key=widget_key,
                    on_change=save_widget_value,
                    args=(widget_key, saved_key)
                )
                st.session_state[saved_key] = input_data[column]

        # --------------------------------------------------
        # Prediction button
        # --------------------------------------------------

        predict_clicked = st.button(
            " Make Prediction",
            type="primary"
        )

        if predict_clicked:

            try:

                input_df = pd.DataFrame(
                    [input_data]
                )

                prediction = best_model.predict(
                    input_df
                )

                # Save prediction for PDF report
                st.session_state["last_prediction"] = str(prediction[0])
                # Reset previous SHAP results
                st.session_state["shap_results"] = None

                st.session_state["prediction_inputs"] = input_data.copy()
                st.session_state["prediction_probabilities"] = None
                st.session_state["shap_error"] = None

                # ------------------------------------------
                # Classification probabilities
                # ------------------------------------------

                if (
                    problem_type == "Classification"
                    and hasattr(
                        best_model,
                        "predict_proba"
                    )
                ):

                    probabilities = (
                        best_model.predict_proba(
                            input_df
                        )[0]
                    )

                    classes = (
                        best_model.classes_
                    )

                    probability_df = pd.DataFrame({
                        "Class": classes,
                        "Probability": probabilities
                    })

                    probability_df[
                        "Probability"
                    ] = (
                        probability_df[
                            "Probability"
                        ] * 100
                    ).round(2)

                    st.dataframe(
                        probability_df,
                        use_container_width=True
                    )

                
                # ------------------------------------------
                # SHAP EXPLANATION
                # ------------------------------------------

                st.divider()

                try:

                    # Get preprocessor and model
                    preprocessor = best_model.named_steps[
                        "preprocessor"
                    ]

                    model = best_model.named_steps[
                        "model"
                    ]

                    # Transform input
                    transformed_input = preprocessor.transform(
                        input_df
                    )

                    # Convert sparse matrix if necessary
                    if hasattr(transformed_input, "toarray"):
                        transformed_input = transformed_input.toarray()

                    # Get feature names
                    feature_names = (
                        preprocessor.get_feature_names_out()
                    )

                    # Use a background dataset from training data
                    X_background = st.session_state["X_test"].head(50)

                    transformed_background = preprocessor.transform(
                        X_background
                    )

                    if hasattr(transformed_background, "toarray"):
                        transformed_background = transformed_background.toarray()

                    # Create model-agnostic explainer
                    explainer = shap.Explainer(
                        model.predict_proba if problem_type == "Classification"
                        else model.predict,
                        transformed_background
                    )

                    shap_values = explainer(
                        transformed_input
                    )

                    shap_array = shap_values.values

                    # Select the predicted class for classification
                    if problem_type == "Classification":

                        if len(shap_array.shape) == 3:

                            class_index = list(
                                model.classes_
                            ).index(prediction[0])

                            shap_array = shap_array[
                                0, :, class_index
                            ]

                        elif len(shap_array.shape) == 2:

                            shap_array = shap_array[0]

                    else:

                        if len(shap_array.shape) == 2:
                            shap_array = shap_array[0]

                    # Build SHAP results table
                    shap_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP Value": shap_array
                    })

                    shap_df["Absolute Impact"] = (
                        shap_df["SHAP Value"].abs()
                    )

                    shap_df = shap_df.sort_values(
                        "Absolute Impact",
                        ascending=False
                    ).head(10)

                    # Save SHAP results for PDF report
                    st.session_state["shap_results"] = (
                        shap_df[["Feature", "SHAP Value"]].copy()
                        )

                except Exception as shap_error:
                    st.session_state["shap_results"] = None
                    st.session_state["shap_error"] = str(shap_error)


            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

        # ==================================================
        # DISPLAY SAVED PREDICTION RESULTS
        # ==================================================

        if "last_prediction" in st.session_state:

            st.divider()
            st.subheader("Saved Prediction Result")

            st.success(
                f"Prediction: **{st.session_state['last_prediction']}**"
            )

            saved_probabilities = st.session_state.get(
                "prediction_probabilities"
            )

            if saved_probabilities is not None:
                st.subheader("Prediction Probabilities")
                st.dataframe(
                    saved_probabilities,
                    use_container_width=True
                )

            st.subheader("Why did the model make this prediction?")

            saved_shap = st.session_state.get("shap_results")

            if saved_shap is not None:
                st.write("Top features influencing this prediction:")
                st.dataframe(
                    saved_shap.round(4),
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(
                    saved_shap["Feature"].iloc[::-1],
                    saved_shap["SHAP Value"].iloc[::-1]
                )
                ax.axvline(x=0, linewidth=1)
                ax.set_title("Feature Impact on Prediction")
                ax.set_xlabel("SHAP Value")
                ax.set_ylabel("Features")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            elif st.session_state.get("shap_error"):
                st.warning(
                    "SHAP explanation could not be generated: "
                    f"{st.session_state['shap_error']}"
                )

        # ==================================================
        # PDF REPORT GENERATION
        # ==================================================

        st.divider()

        st.subheader("Download ML Analysis Report")

        if "last_prediction" in st.session_state:

            try:

                # Get dataset overview
                dataset_overview = get_dataset_overview(
                    st.session_state["dataset"]
                )

                # Get saved training information
                target_column = st.session_state["target_column"]

                problem_type = st.session_state["problem_type"]

                comparison = st.session_state["comparison"]

                best_model_name = st.session_state["best_model_name"]

                best_metrics = st.session_state["training_results"][
                    best_model_name
                ]

                # Get prediction and SHAP results
                prediction = st.session_state["last_prediction"]

                shap_results = st.session_state.get(
                    "shap_results"
                )

                # Generate PDF
                pdf_buffer = generate_pdf_report(
                    dataset_overview=dataset_overview,
                    target_column=target_column,
                    problem_type=problem_type,
                    comparison=comparison,
                    best_model_name=best_model_name,
                    best_metrics=best_metrics,
                    prediction=prediction,
                    shap_results=shap_results
                )

                # Download button
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name="automl_studio_report.pdf",
                    mime="application/pdf",
                    type="primary"
                )

            except Exception as pdf_error:

                st.error(
                    f"Could not generate PDF report: {pdf_error}"
                )

        else:

            st.info(
                "Make a prediction first to generate your PDF report."
            )