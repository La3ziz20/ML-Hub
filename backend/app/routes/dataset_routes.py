from fastapi import APIRouter, HTTPException, UploadFile, File
import os
from typing import Dict, Any
from app.services.data_service import data_service

router = APIRouter(prefix="/dataset", tags=["Dataset"])

@router.get("/")
def get_dataset_info():
    result = data_service.get_dataset_preview()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    os.makedirs("data", exist_ok=True)
    filepath = "data/uploaded_dataset.csv"
    
    try:
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
            
        data_service.set_uploaded_path()
        return {"message": "Dataset uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@router.get("/pca")
def get_pca(target_column: str = None):
    result = data_service.run_pca(target_column)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
