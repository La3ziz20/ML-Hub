from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

import mlflow
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.metrics import DatasetDriftMetric

router = APIRouter(prefix="/drift", tags=["Drift"])

class DriftResponse(BaseModel):
    drift_share: float
    drifted_columns: int
    total_columns: int
    dataset_drifted: bool
    status: str
    message: str

def run_drift_simulation_task():
    try:
        # 1. Load data
        df = pd.read_csv('data/cleaned_dataset.csv')
        
        target_col = 'Price'
        if target_col not in df.columns:
            target_col = df.columns[-1]

        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Simulate drift on Production data (based on test set)
        X_prod = X_test.copy()
        num_cols = X_prod.select_dtypes(include=np.number).columns
        
        # Drift 2 features
        for col in num_cols[:2]:
            X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

        # 3. Setup MLflow
        mlflow.set_tracking_uri("sqlite:///./data/mlruns.db")
        mlflow.set_experiment('monitoring_drift')

        with mlflow.start_run(run_name='drift_check_api'):
            # 4. Generate Evidently Report
            report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
            report.run(reference_data=X_train, current_data=X_prod)
            
            # Ensure static directory exists
            os.makedirs('static', exist_ok=True)
            report_path = 'static/drift_report.html'
            report.save_html(report_path)
            mlflow.log_artifact(report_path)

            # 5. Extract numerical scores
            score_report = Report(metrics=[DatasetDriftMetric()])
            score_report.run(reference_data=X_train, current_data=X_prod)
            result = score_report.as_dict()
            
            drift_share = result['metrics'][0]['result']['drift_share']
            dataset_drift = result['metrics'][0]['result']['dataset_drift']
            n_drifted = result['metrics'][0]['result']['number_of_drifted_columns']
            n_total = result['metrics'][0]['result']['number_of_columns']
            
            mlflow.log_metric('drift_share', drift_share)
            mlflow.log_metric('drifted_columns', n_drifted)
            mlflow.log_metric('total_columns', n_total)
            mlflow.log_metric('dataset_drifted', int(dataset_drift))

            # Store the latest summary in a JSON file to easily return it
            import json
            summary = {
                "drift_share": drift_share,
                "drifted_columns": n_drifted,
                "total_columns": n_total,
                "dataset_drifted": bool(dataset_drift),
                "status": "success",
                "message": "Drift simulation completed."
            }
            with open('static/drift_summary.json', 'w') as f:
                json.dump(summary, f)

    except Exception as e:
        import traceback
        traceback.print_exc()

@router.post("/simulate")
def simulate_drift(background_tasks: BackgroundTasks):
    # We can run it in background or synchronously. 
    # Since the user wants to see results, and it's fast on a small dataset, we'll run it synchronously for the API.
    # Actually, calculating Evidently report on 15,000 rows might take 10-15 seconds.
    # Let's run it synchronously so the frontend can await it and get the summary directly.
    try:
        # Load data
        df = pd.read_csv('data/cleaned_dataset.csv')
        
        target_col = 'Price'
        if target_col not in df.columns:
            target_col = df.columns[-1]

        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Simulate drift
        X_prod = X_test.copy()
        num_cols = X_prod.select_dtypes(include=np.number).columns
        
        for col in num_cols[:2]:
            X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

        # Setup MLflow
        mlflow.set_tracking_uri("sqlite:///./data/mlruns.db")
        mlflow.set_experiment('monitoring_drift')

        with mlflow.start_run(run_name='drift_check_api'):
            report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
            report.run(reference_data=X_train, current_data=X_prod)
            
            os.makedirs('static', exist_ok=True)
            report_path = 'static/drift_report.html'
            report.save_html(report_path)
            mlflow.log_artifact(report_path)

            score_report = Report(metrics=[DatasetDriftMetric()])
            score_report.run(reference_data=X_train, current_data=X_prod)
            result = score_report.as_dict()
            
            drift_share = result['metrics'][0]['result']['drift_share']
            dataset_drift = result['metrics'][0]['result']['dataset_drift']
            n_drifted = result['metrics'][0]['result']['number_of_drifted_columns']
            n_total = result['metrics'][0]['result']['number_of_columns']
            
            mlflow.log_metric('drift_share', drift_share)
            mlflow.log_metric('drifted_columns', n_drifted)
            mlflow.log_metric('total_columns', n_total)
            mlflow.log_metric('dataset_drifted', int(dataset_drift))

            summary = {
                "drift_share": float(drift_share),
                "drifted_columns": int(n_drifted),
                "total_columns": int(n_total),
                "dataset_drifted": bool(dataset_drift),
                "status": "success",
                "message": "Drift simulation completed."
            }
            
            # AUTOMATED RETRAINING PIPELINE (CI/CD Closed Loop)
            if float(drift_share) > 0.30:
                mlflow.log_metric('retrain_triggered', 1)
                
                from app.models.database import SessionLocal
                from app.models.schema import Experiment
                from app.services.ml_service import ml_service
                
                db = SessionLocal()
                # Retrain using a robust default model (RandomForest) on the new data
                experiment = Experiment(
                    model_name="RandomForest",
                    target_column=target_col,
                    hyperparameters={"n_estimators": 100},
                    status="pending"
                )
                db.add(experiment)
                db.commit()
                db.refresh(experiment)
                
                background_tasks.add_task(ml_service.train_model_background, experiment.id)
                db.close()
                
                summary["message"] = "Drift > 30% detected! Automated retraining of RandomForest triggered in the background."

            import json
            with open('static/drift_summary.json', 'w') as f:
                json.dump(summary, f)

            return DriftResponse(**summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
def get_drift_summary():
    import json
    if not os.path.exists('static/drift_summary.json'):
        return {"status": "none", "message": "No drift simulation has been run yet."}
    with open('static/drift_summary.json', 'r') as f:
        return json.load(f)

@router.get("/report")
def get_drift_report():
    if not os.path.exists('static/drift_report.html'):
        raise HTTPException(status_code=404, detail="Drift report not found. Run simulation first.")
    return FileResponse('static/drift_report.html', media_type='text/html')
