import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor  # type: ignore
from sklearn.metrics import (
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
        self.regressors = {
            "RandomForest": RandomForestRegressor,
            "SVM": SVR,
            "KNR": KNeighborsRegressor,
            "LinearReg": LinearRegression,
            "AdaBoost": AdaBoostRegressor,
            "XGBoost": XGBRegressor
        }

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
            
            # Subsample for SVM to prevent hanging on large datasets
            if experiment.model_name == "SVM" and len(X_train) > 5000:
                sample_indices = X_train.sample(n=5000, random_state=42).index
                X_train = X_train.loc[sample_indices]
                y_train = y_train.loc[sample_indices]
            
            model_class = self.regressors.get(experiment.model_name)

            if not model_class:
                raise ValueError(f"Unknown regression model type: {experiment.model_name}")

            # Initialize model with hyperparameters if provided
            hyperparams = experiment.hyperparameters or {}
            
            # Adjust params specific to models
            if experiment.model_name == "SVM":
                hyperparams["max_iter"] = 500 # Prevent infinite hanging
                hyperparams["cache_size"] = 1000 # Increase cache for faster computation
                
                # Note: We specifically do NOT set probability=True here anymore,
                # because it forces a 5-fold cross-validation internally which takes forever.
                # The ROC curve will fallback to using decision_function instead.
                
            model = model_class(**hyperparams)
            
            # Save feature columns
            experiment.feature_columns = X_train.columns.tolist()

            # MLFlow Start Run
            mlflow.set_experiment("ml_platform_experiments")
            mlflow.start_run(run_name=f"{experiment.model_name}_exp_{experiment.id}")
            mlflow.log_params(hyperparams)
            mlflow.log_param("target_column", experiment.target_column)
            mlflow.log_param("algo", experiment.model_name)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            
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
            
            mlflow.sklearn.log_model(model, "model", registered_model_name=experiment.model_name)
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
