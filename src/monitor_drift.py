import pandas as pd
from evidently.metric_preset import DataDriftPreset 
from evidently.report import Report
import mlflow


# Load datasets
reference_data = pd.read_csv("../data/spam_cleaned_original.csv")
current_data = pd.read_csv("../data/spam_cleaned_new.csv")

# Create and run the report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save as HTML (view in browser)
report.save_html("../reports/data_drift_report.html")

# Start MLflow run
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("sqlite:///mlflow.db")
with mlflow.start_run() as run:
    # Log the report as HTML artifact
    report.save_html("../reports/data_drift_report.html")
    mlflow.log_artifact("../reports/data_drift_report.html")
    
    # Log key drift metrics
    drift_metrics = report.as_dict()["metrics"][1]["result"]["drift_by_columns"]["X"]
    mlflow.log_metrics({
        "drift_score": float(drift_metrics["drift_score"]),
        "stattest_threshold": drift_metrics["stattest_threshold"]
    })
    
    # Log dataset stats
    mlflow.log_params({
        "reference_rows": len(reference_data),
        "current_rows": len(current_data)
    })