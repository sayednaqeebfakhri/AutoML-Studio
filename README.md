# AutoML Studio

AutoML Studio is a web-based machine learning platform built with Python and Streamlit. It allows users to upload tabular datasets, explore their data, train and compare machine learning models, make predictions, and generate reports.

The project aims to simplify the machine learning workflow by bringing common data analysis and model development tasks into one interface.

## Features

### 1. Dataset Upload and Validation

- Upload CSV datasets.
- Automatically detect CSV delimiters.
- Validate datasets before analysis.
- Preview uploaded data.

### 2. Exploratory Data Analysis (EDA)

- Dataset overview and column information.
- Numerical and categorical statistics.
- Missing-value analysis.
- Outlier detection.
- Histograms and box plots.
- Categorical distribution plots.
- Correlation heatmaps.
- Scatter plots for feature relationships.

### 3. Data Preprocessing

- Separate features and target variables.
- Automatically identify numerical and categorical columns.
- Handle missing numerical values using median imputation.
- Handle missing categorical values using most-frequent imputation.
- Scale numerical features.
- Encode categorical features using One-Hot Encoding.

### 4. Automated Model Training

Supports two machine learning tasks:

**Classification**

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

**Regression**

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

### 5. Model Evaluation and Comparison

**Classification metrics**

- Accuracy
- Precision
- Recall
- F1-score

**Regression metrics**

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Models are compared using appropriate evaluation metrics.

### 6. Predictions

- Make predictions using trained models.
- Enter feature values through dynamically generated input fields.
- View predicted classes and class probabilities for classification tasks.

### 7. Model Explainability

- SHAP-based model explanations.
- Feature contribution analysis.
- SHAP summary visualizations.

### 8. PDF Reports

Generate downloadable PDF reports containing:

- Dataset overview.
- Selected target and problem type.
- Model comparison results.
- Best model performance.
- Prediction information, when available.
- SHAP explanation results, when available.

## Technologies Used

| Category             | Technologies              |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Web Framework        | Streamlit                 |
| Data Analysis        | Pandas, NumPy             |
| Visualization        | Matplotlib, Seaborn       |
| Machine Learning     | Scikit-learn              |
| Explainable AI       | SHAP                      |
| PDF Generation       | ReportLab                 |
| Data Storage         | SQLite, SQLAlchemy        |
| Development          | Jupyter Notebook, VS Code |

## Project Structure

```text
AutoML-Studio/
│
├── app.py
│
├── modules/
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── model_training.py
│   ├── pdf_report.py
│   ├── preprocessing.py
│   └── task_detection.py
│
├── notebooks/
│   ├── data_loading.ipynb
│   ├── eda.ipynb
│   ├── model_comparison.ipynb
│   ├── model_training.ipynb
│   └── preprocessing.ipynb
│
├── sample_data/
│   └── student-mat.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AutoML-Studio.git
```

### 2. Navigate to the project directory

```bash
cd AutoML-Studio
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

## Sample Dataset

The project includes `student-mat.csv` as a sample dataset for development and testing.

The application is designed for general tabular datasets rather than being limited to student data.

The student dataset is used to demonstrate data analysis, classification, regression workflows, model evaluation, and predictions.

## Notebooks

The `notebooks/` directory contains separate notebooks for learning and experimenting with the machine learning workflow:

- `data_loading.ipynb` — Loading and inspecting datasets.
- `eda.ipynb` — Exploratory data analysis and visualization.
- `preprocessing.ipynb` — Data cleaning and preprocessing.
- `model_training.ipynb` — Model training and evaluation.
- `model_comparison.ipynb` — Comparing machine learning models.

## Project Objectives

The main objectives of AutoML Studio are to:

- Simplify common tabular machine learning workflows.
- Reduce repetitive data preparation and model training tasks.
- Help users understand their datasets through visual analysis.
- Compare multiple machine learning algorithms.
- Provide interpretable model predictions.
- Generate structured reports of machine learning experiments.

## Future Improvements

Potential future improvements include:

- Support for additional machine learning algorithms.
- Advanced hyperparameter optimization.
- More interactive visualizations.
- Improved feature selection.
- Support for larger datasets.
- Additional explainability techniques.
- Experiment tracking and model management.

## Disclaimer

AutoML Studio is an educational and development project. Model performance depends on dataset quality, feature selection, preprocessing, and evaluation methodology. Predictions and explanations should be interpreted in the context of the dataset and should not be treated as guarantees of real-world outcomes.
