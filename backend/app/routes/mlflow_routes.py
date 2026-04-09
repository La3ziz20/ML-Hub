from fastapi import APIRouter
import mlflow
import json

router = APIRouter(prefix="/mlflow", tags=["MLFlow"])

@router.get("/runs")
def get_mlflow_runs():
    try:
        experiment = mlflow.get_experiment_by_name("ml_platform_experiments")
        if not experiment:
            return {"runs": []}
            
        df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if df.empty:
            return {"runs": []}
            
        # Clean up Pandas nan to None for JSON compliance
        df = df.replace({float('nan'): None})
        
        runs = []
        for index, row in df.iterrows():
            metrics = {k.replace('metrics.', ''): v for k, v in row.items() if k.startswith('metrics.') and v is not None}
            params = {k.replace('params.', ''): v for k, v in row.items() if k.startswith('params.') and v is not None}
            
            run_data = {
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "start_time": row.get("start_time").isoformat() if hasattr(row.get("start_time"), 'isoformat') else None,
                "end_time": row.get("end_time").isoformat() if hasattr(row.get("end_time"), 'isoformat') else None,
                "metrics": metrics,
                "parameters": params,
                "artifact_uri": row.get("artifact_uri")
            }
            runs.append(run_data)
            
        return {"runs": runs}
    except Exception as e:
        return {"error": str(e), "runs": []}
