import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Add the root backend directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_service import data_service

def run_experiment(cfg, X_train, X_test, y_train, y_test):
    run_name = f"{cfg['model']}_depth_{cfg.get('max_depth', 'none')}_est_{cfg.get('n_estimators', 'none')}"
    
    with mlflow.start_run(run_name=run_name):
        # 1. Log parameters
        mlflow.log_params(cfg)
        
        # 2. Select Model
        if cfg['model'] == 'rf':
            model = RandomForestClassifier(
                n_estimators=cfg.get('n_estimators', 100),
                max_depth=cfg.get('max_depth', None),
                random_state=42
            )
        elif cfg['model'] == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=cfg.get('n_estimators', 100),
                learning_rate=cfg.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            model = LogisticRegression(max_iter=500, random_state=42)

        # 3. Train Model
        print(f"Training {run_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except Exception:
            roc_auc = 0.0
            
        # 4. Log Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc
        }
        mlflow.log_metrics(metrics)
        print(f"  --> Accuracy: {metrics['accuracy']:.4f}")

        # 5. Log Artifacts (Confusion Matrix & Report)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close(fig)

        report = classification_report(y_test, y_pred, zero_division=0)
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report.txt')

        # 6. Log Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        print(f"  --> Model saved to MLflow")

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///./data/mlruns.db")
    mlflow.set_experiment('mon_projet_ml')

    # Prepare Data using existing data service
    # Assuming 'Category' is the target column based on Tâche 4
    try:
        X_train, X_test, y_train, y_test = data_service.prepare_data(target_column="Category")
    except Exception as e:
        print(f"Error preparing data (ensure cleaned_dataset.csv is in backend/data/): {e}")
        sys.exit(1)

    configs = [
        {'model': 'rf', 'n_estimators': 50, 'max_depth': 3},       # Run 1 - Baseline
        {'model': 'rf', 'n_estimators': 200, 'max_depth': 10},     # Run 2 - Profond
        {'model': 'gb', 'n_estimators': 100, 'learning_rate': 0.1}, # Run 3 - Alt (GradientBoosting)
        {'model': 'lr', 'C': 1.0}                                  # Run 4 - Alt (LogisticRegression)
    ]

    for cfg in configs:
        run_experiment(cfg, X_train, X_test, y_train, y_test)
        
    print("Batch training complete. View results with 'mlflow ui --port 5000'")
