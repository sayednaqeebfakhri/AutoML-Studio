
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
import io
import matplotlib.pyplot as plt


def generate_pdf_report(
    dataset_overview,
    target_column,
    problem_type,
    comparison,
    best_model_name,
    best_metrics,
    prediction=None,
    shap_results=None
):
    """
    Generate a PDF report for AutoML Studio.
    """

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1F4E79"),
        fontSize=22,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=15,
        spaceAfter=10
    )

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "AutoML Studio - ML Analysis Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    # --------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------

    story.append(
        Paragraph("1. Dataset Overview", heading_style)
    )

    overview_data = [["Metric", "Value"]]

    for key, value in dataset_overview.items():
        overview_data.append([str(key), str(value)])

    overview_table = Table(
        overview_data,
        colWidths=[250, 220]
    )

    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(overview_table)

    # --------------------------------------------------
    # PROBLEM INFORMATION
    # --------------------------------------------------

    story.append(
        Paragraph("2. Machine Learning Task", heading_style)
    )

    story.append(
        Paragraph(f"<b>Target Column:</b> {target_column}", styles["Normal"])
    )

    story.append(
        Paragraph(f"<b>Problem Type:</b> {problem_type}", styles["Normal"])
    )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    story.append(
        Paragraph("3. Model Comparison", heading_style)
    )

    comparison_data = [
        ["Model"] + list(comparison.columns)
    ]

    for model_name, row in comparison.iterrows():
        comparison_data.append(
            [str(model_name)] + [
                f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                for value in row
            ]
        )

    comparison_table = Table(
        comparison_data,
        repeatRows=1
    )

    comparison_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(comparison_table)

    # --------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------

    story.append(
        Paragraph("4. Best Model", heading_style)
    )

    story.append(
        Paragraph(
            f"<b>Selected Model:</b> {best_model_name}",
            styles["Normal"]
        )
    )

    metrics_data = [["Metric", "Value"]]

    for key, value in best_metrics.items():
        metrics_data.append([
            str(key),
            f"{value:.4f}"
        ])

    metrics_table = Table(metrics_data, colWidths=[250, 220])

    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(metrics_table)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if prediction is not None:

        story.append(
            Paragraph("5. Prediction Result", heading_style)
        )

        story.append(
            Paragraph(
                f"<b>Prediction:</b> {prediction}",
                styles["Normal"]
            )
        )

    # --------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------

    if shap_results is not None:

        story.append(
            Paragraph(
                "6. SHAP Feature Explanation",
                heading_style
            )
        )

        story.append(
            Paragraph(
                "The following features had the greatest "
                "influence on the prediction.",
                styles["Normal"]
            )
        )

        # SHAP results table
        shap_data = [["Feature", "SHAP Value"]]

        for _, row in shap_results.iterrows():

            shap_data.append([
                str(row["Feature"]),
                f"{row['SHAP Value']:.4f}"
            ])

        shap_table = Table(shap_data)

        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#1F4E79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(shap_table)
        story.append(Spacer(1, 20))

        # Generate SHAP bar chart
        fig, ax = plt.subplots(figsize=(9, 5))

        chart_data = shap_results.sort_values(
            "SHAP Value",
            ascending=True
        )

        ax.barh(
            chart_data["Feature"],
            chart_data["SHAP Value"]
        )

        ax.axvline(x=0, linewidth=1)

        ax.set_title("SHAP Feature Impact")
        ax.set_xlabel("SHAP Value")
        ax.set_ylabel("Features")

        fig.tight_layout()

        # Save chart in memory
        chart_buffer = io.BytesIO()

        fig.savefig(
            chart_buffer,
            format="png",
            dpi=150,
            bbox_inches="tight"
        )

        chart_buffer.seek(0)

        plt.close(fig)

        # Add chart to PDF
        from reportlab.platypus import Image

        story.append(
            Image(
                chart_buffer,
                width=450,
                height=250
            )
        )
    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(story)

    buffer.seek(0)

    return buffer
