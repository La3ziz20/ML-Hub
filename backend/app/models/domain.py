from pydantic import BaseModel
from typing import Dict, Any, Optional

class TrainRequest(BaseModel):
    model_name: str
    target_column: str
    hyperparameters: Optional[Dict[str, Any]] = None
    test_size: float = 0.2
    random_state: int = 42

class ModelResponse(BaseModel):
    id: int
    model_name: str
    target_column: str
    status: str
    metrics: Optional[Dict[str, Any]] = None
