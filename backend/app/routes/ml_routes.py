from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import os

from app.models.database import get_db
from app.models.schema import Experiment
from app.models.domain import TrainRequest, ModelResponse
from app.services.ml_service import ml_service

router = APIRouter(prefix="/models", tags=["Models"])

@router.post("/train", response_model=ModelResponse)
def train_model(
    request: TrainRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Validate model name
    valid_models = ["RandomForest", "SVM", "KNN", "Linear/Logistic"]
    if request.model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Choose from {valid_models}")
    
    experiment = Experiment(
        model_name=request.model_name,
        target_column=request.target_column,
        hyperparameters=request.hyperparameters,
        status="pending"
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    # Launch background training
    background_tasks.add_task(ml_service.train_model_background, experiment.id)

    return ModelResponse(
        id=experiment.id,
        model_name=experiment.model_name,
        target_column=experiment.target_column,
        status="pending"
    )

@router.get("/models", response_model=List[ModelResponse])
def get_models(db: Session = Depends(get_db)):
    experiments = db.query(Experiment).order_by(Experiment.id.desc()).all()
    return [
        ModelResponse(
            id=e.id,
            model_name=e.model_name,
            target_column=e.target_column,
            status=e.status,
            metrics=e.metrics
        ) for e in experiments
    ]

@router.get("/models/{model_id}")
def get_model_details(model_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == model_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Model not found")
        
    return {
        "id": experiment.id,
        "model_name": experiment.model_name,
        "target_column": experiment.target_column,
        "status": experiment.status,
        "metrics": experiment.metrics,
        "confusion_matrix": experiment.confusion_matrix,
        "roc_curve": experiment.roc_curve,
        "feature_columns": experiment.feature_columns,
        "hyperparameters": experiment.hyperparameters,
        "error_message": experiment.error_message
    }

from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@router.post("/models/{model_id}/predict")
def predict_with_model(model_id: int, request: PredictRequest, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == model_id).first()
    if not experiment or experiment.status != "completed":
        raise HTTPException(status_code=404, detail="Model not ready or missing")
        
    file_path = f"data/models/model_{model_id}.joblib"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file is missing")
        
    try:
        model = joblib.load(file_path)
        
        # Build dataframe in same order as training
        if not experiment.feature_columns:
            raise ValueError("Feature columns not recorded for this model.")
            
        df = pd.DataFrame([request.features])
        # Ensure all required columns are present (with 0 if missing)
        for col in experiment.feature_columns:
            if col not in df.columns:
                df[col] = 0
                
        # Reorder to match training precisely
        df = df[experiment.feature_columns]
        
        pred = model.predict(df)
        
        # Try to get probabilities
        prob = None
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(df)[0]
            prob = prob_arr.tolist()
            
        return {
            "prediction": pred[0].item() if hasattr(pred[0], "item") else pred[0],
            "probabilities": prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/download")
def download_model(model_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == model_id).first()
    if not experiment or experiment.status != "completed":
        raise HTTPException(status_code=404, detail="Model not found or not ready yet")
        
    file_path = f"data/models/model_{model_id}.joblib"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file missing from disk")
        
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=f"{experiment.model_name.replace('/', '_').lower()}_{model_id}.joblib"
    )
