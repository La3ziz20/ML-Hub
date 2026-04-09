import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from app.services.data_service import data_service
from app.models.database import SessionLocal
from app.models.schema import Experiment

class MLService:
    def __init__(self):
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/mlruns", exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///./data/mlruns.db")
        mlflow.set_experiment("ml_platform_experiments")
        self.classifiers = {
            "RandomForest": RandomForestClassifier,
            "SVM": SVC,
            "KNN": KNeighborsClassifier,
            "Linear/Logistic": LogisticRegression
        }
        self.regressors = {
            "RandomForest": RandomForestRegressor,
            "SVM": SVR,
            "KNN": KNeighborsRegressor,
            "Linear/Logistic": LinearRegression
        }

    def determine_task_type(self, y):
        if y.dtype == object or y.dtype == bool or y.nunique() <= 20:
            return "classification"
        return "regression"

    def train_model_background(self, experiment_id: int):
        db = SessionLocal()
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            db.close()
            return

        try:
            experiment.status = "running"
            db.commit()

            # Prepare data
            X_train, X_test, y_train, y_test = data_service.prepare_data(
                target_column=experiment.target_column
            )
            
            task_type = self.determine_task_type(y_train)

            if task_type == "classification":
                model_class = self.classifiers.get(experiment.model_name)
            else:
                model_class = self.regressors.get(experiment.model_name)

            if not model_class:
                raise ValueError(f"Unknown model type: {experiment.model_name} for task {task_type}")

            # Initialize model with hyperparameters if provided
            hyperparams = experiment.hyperparameters or {}
            
            # Adjust params specific to models
            if experiment.model_name == "SVM" and task_type == "classification":
                hyperparams["probability"] = True
                
            model = model_class(**hyperparams)
            
            # Save feature columns
            experiment.feature_columns = X_train.columns.tolist()

            # MLFlow Start Run
            mlflow.start_run(run_name=f"{experiment.model_name}_exp_{experiment.id}")
            mlflow.log_params(hyperparams)
            mlflow.log_param("target_column", experiment.target_column)
            mlflow.log_param("algo", experiment.model_name)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            
            if task_type == "classification":
                # Predict Probabilities for ROC
                try:
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        y_score = model.decision_function(X_test)
                    else:
                        y_score = None
                except Exception:
                    y_score = None

                # Metrics calculation
                metrics = {
                    "task_type": "classification",
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
                }
                
                if hasattr(model, "feature_importances_"):
                    # Pair feature importance with columns
                    importances = model.feature_importances_.tolist()
                    metrics["feature_importances"] = dict(zip(X_train.columns.tolist(), importances))

                experiment.metrics = metrics

                cm = confusion_matrix(y_test, y_pred)
                experiment.confusion_matrix = cm.tolist()

                mlflow.log_metrics({
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"]
                })

                if y_score is not None and len(set(y_test)) == 2: # binary classification for simple ROC
                    try:
                        y_test_bin = (y_test == list(set(y_test))[1]).astype(int)
                        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                        auc_score = float(auc(fpr, tpr))
                        experiment.roc_curve = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "auc": auc_score
                        }
                        mlflow.log_metric("auc", auc_score)
                    except Exception as e:
                        pass
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    "task_type": "regression",
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2)
                }
                experiment.metrics = metrics
                experiment.confusion_matrix = None
                experiment.roc_curve = None
                
                mlflow.log_metrics({
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2)
                })

            # Save model
            model_path = f"data/models/model_{experiment_id}.joblib"
            joblib.dump(model, model_path)
            
            mlflow.sklearn.log_model(model, "model")
            mlflow.end_run()
            
            experiment.status = "completed"
            
        except Exception as e:
            if mlflow.active_run():
                mlflow.end_run()
            experiment.status = "failed"
            experiment.error_message = str(e)
            
        finally:
            db.commit()
            db.close()

ml_service = MLService()
