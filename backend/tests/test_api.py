import pytest
import os
import time
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.models.database import Base, get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sql_app.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Patch the default SessionLocal so background tasks use test DB too
import app.models.database as db_module
db_module.SessionLocal = TestingSessionLocal

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML Platform API"}

def test_dataset_preview():
    # Make sure mock data or real data exists for test
    if not os.path.exists("cleaned_dataset.csv"):
        # Create a dummy dataset for testing if none exists
        with open("cleaned_dataset.csv", "w") as f:
            f.write("feature1,feature2,target\n1,2,0\n3,4,1\n1,2,0\n3,4,1\n1,2,0\n3,4,1\n1,2,0\n3,4,1\n1,2,0\n3,4,1\n")
            
    response = client.get("/dataset/")
    assert response.status_code == 200
    data = response.json()
    assert "columns" in data
    assert "preview" in data
    assert len(data["columns"]) > 0

def test_train_model():
    # Get columns
    response = client.get("/dataset/")
    target = response.json()["columns"][-1]
    
    # Trigger train
    train_resp = client.post(
        "/models/train",
        json={"model_name": "LogisticRegression", "target_column": target, "hyperparameters": {}}
    )
    assert train_resp.status_code == 200
    model_data = train_resp.json()
    assert "id" in model_data
    assert model_data["status"] == "pending"
    
    model_id = model_data["id"]
    
    # Wait for background task to complete
    max_retries = 10
    for _ in range(max_retries):
        resp = client.get(f"/models/models/{model_id}")
        assert resp.status_code == 200
        status = resp.json()["status"]
        if status in ["completed", "failed"]:
            break
        time.sleep(1)
        
    final_resp = client.get(f"/models/models/{model_id}")
    final_data = final_resp.json()
    if final_data["status"] == "failed":
        print("Training Error:", final_data["error_message"])
    assert final_data["status"] == "completed"
    assert "metrics" in final_data
    assert final_data["metrics"] is not None
    assert "accuracy" in final_data["metrics"]

@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Cleanup logic if needed
    pass
