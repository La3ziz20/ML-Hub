import os
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

import mlflow
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.metrics import DatasetDriftMetric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simulate_and_detect_drift():
    # 1. Load data
    try:
        df = pd.read_csv('data/cleaned_dataset.csv')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # In Tache 4, the target was typically 'Category'
    target_col = 'Category'
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found. Using the last column.")
        target_col = df.columns[-1]

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Simulate drift on Production data (based on test set)
    print("Simulating data drift...")
    X_prod = X_test.copy()
    num_cols = X_prod.select_dtypes(include=np.number).columns
    
    # Drift 2 features
    for col in num_cols[:2]:
        # Multiply by 1.6 and add noise
        X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))
        print(f"Moyenne feature '{col}' - Ref: {X_train[col].mean():.3f} | Prod: {X_prod[col].mean():.3f}")

    # 3. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///./data/mlruns.db")
    mlflow.set_experiment('monitoring_drift')

    with mlflow.start_run(run_name='drift_check_v1'):
        # 4. Generate Evidently Report
        print("Running Evidently Data Drift analysis...")
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X_train, current_data=X_prod)
        report.save_html('drift_report.html')
        mlflow.log_artifact('drift_report.html')

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
        
        print(f"Drift share : {drift_share:.2%} | Colonnes driftées : {n_drifted}/{n_total}")

        # 6. Statistical Test (KS-test)
        print("Running KS-tests...")
        ks_results = []
        for col in X_train.select_dtypes(include='number').columns:
            stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
            ks_results.append({
                'feature': col,
                'ks_stat': round(stat, 4),
                'p_value': round(pvalue, 4),
                'drifted': pvalue < 0.05
            })
            mlflow.log_metric(f'ks_pvalue_{col}', pvalue)
            
        df_drift = pd.DataFrame(ks_results)
        df_drift.to_csv('ks_drift_results.csv', index=False)
        mlflow.log_artifact('ks_drift_results.csv')
        print(df_drift.to_string(index=False))

        # 7. Automatic Retraining Trigger
        SEUIL_DRIFT = 0.30  # 30% threshold
        SEUIL_WARN = 0.15
        
        print("\nEvaluating drift thresholds...")
        if drift_share > SEUIL_DRIFT:
            print(f"CRITIQUE : drift {drift_share:.2%} > seuil {SEUIL_DRIFT:.0%}")
            print("Triggering automatic retraining...")
            mlflow.log_metric('retrain_triggered', 1)
            try:
                subprocess.run([sys.executable, 'src/train_batch.py'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Retraining failed: {e}")
        elif drift_share > SEUIL_WARN:
            print(f"AVERTISSEMENT : drift {drift_share:.2%} — surveillance renforcée")
            mlflow.log_metric('retrain_triggered', 0)
        else:
            print(f"OK : drift {drift_share:.2%} — modèle stable")
            mlflow.log_metric('retrain_triggered', 0)

if __name__ == "__main__":
    simulate_and_detect_drift()
