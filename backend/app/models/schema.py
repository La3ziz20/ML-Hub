from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.sql import func
from .database import Base

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    target_column = Column(String)
    status = Column(String) # pending, running, completed, failed
    hyperparameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    roc_curve = Column(JSON, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
