import os
import shutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.database import Base, engine
from app.routes import dataset_routes, ml_routes, mlflow_routes, drift_routes

# Create DB Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ML Platform API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset_routes.router)
app.include_router(ml_routes.router)
app.include_router(mlflow_routes.router)
app.include_router(drift_routes.router)

# Ensure the dataset exists where we expect it
@app.on_event("startup")
def startup_event():
    dest_path = "cleaned_dataset.csv"
    src_path = "../cleaned_dataset.csv"  # relative to backend dir
    if not os.path.exists(dest_path) and os.path.exists(src_path):
        import shutil
        shutil.copy(src_path, dest_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Platform API"}
