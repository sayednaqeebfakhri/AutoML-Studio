import pandas as pd
import matplotlib.pyplot as plt

def get_dataset_overview(df):
    """
    Generate basic information about the dataset.
    """

    overview = {
        "Total Rows": df.shape[0],
        "Total Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage (KB)": round(
            df.memory_usage(deep=True).sum() / 1024, 2
        )
    }

    return overview


def get_column_summary(df):
    """
    Generate summary information for each column.
    """

    summary = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })

    return summary


def get_numeric_statistics(df):
    """
    Generate statistical information for numeric columns.
    """

    return df.describe().T


def get_categorical_statistics(df):
    """
    Generate frequency information for categorical columns.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    results = {}

    for column in categorical_columns:
        results[column] = df[column].value_counts()

    return results


def get_missing_values_summary(df):
    """Return missing-value counts and percentages by column."""
    summary = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().sum() / len(df) * 100
        ).round(2)
    })

    return summary.sort_values(
        "Missing Values",
        ascending=False
    )


def get_outlier_summary(df):
    """Detect potential numerical outliers using the IQR method."""
    numeric_df = df.select_dtypes(include=["number"])
    results = {}

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (series < lower_bound) |
            (series > upper_bound)
        )

        results[column] = {
            "Outlier Count": int(outliers.sum()),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2)
        }

    return pd.DataFrame(results).T


def plot_numeric_distribution(df, column):
    """Plot the distribution of a numerical column."""
    fig, ax = plt.subplots(figsize=(9, 5))

    values = df[column].dropna()

    ax.hist(values, bins=20, edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def plot_numeric_boxplot(df, column):
    """Create a box plot for a numerical column."""
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.boxplot(df[column].dropna(), vert=False)
    ax.set_title(f"Box Plot of {column}")
    ax.set_xlabel(column)

    fig.tight_layout()
    return fig


def plot_categorical_distribution(df, column):
    """Plot the most frequent categories."""
    fig, ax = plt.subplots(figsize=(9, 5))

    counts = df[column].value_counts().head(15)

    counts.plot(kind="bar", ax=ax)

    ax.set_title(f"Category Distribution: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Visualize correlations between numerical columns."""
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        return None

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(max(8, len(correlation.columns) * 0.7),
                 max(6, len(correlation.columns) * 0.6))
    )

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=90
    )

    ax.set_yticklabels(correlation.columns)

    ax.set_title("Numerical Feature Correlation")

    fig.colorbar(image, ax=ax, label="Correlation")

    fig.tight_layout()
    return fig


def plot_scatter_relationship(df, x_column, y_column):
    """Plot the relationship between two numerical columns."""
    fig, ax = plt.subplots(figsize=(8, 5))

    clean_data = df[[x_column, y_column]].dropna()

    ax.scatter(
        clean_data[x_column],
        clean_data[y_column],
        alpha=0.6
    )

    ax.set_title(f"{x_column} vs {y_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    fig.tight_layout()
    return fig
