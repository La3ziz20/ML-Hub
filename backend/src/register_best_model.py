import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    mlflow.set_tracking_uri("sqlite:///./data/mlruns.db")
    client = MlflowClient()
    
    experiment_name = 'mon_projet_ml'
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment {experiment_name} not found. Have you run the training script yet?")
        return
        
    print("Searching for the best run based on metrics.accuracy...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.accuracy DESC'],
        max_results=5
    )
    
    if not runs:
        print("No runs found in the experiment.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics.get('accuracy', 0)
    
    print(f"Best run : {best_run_id}")
    print(f"Accuracy : {best_accuracy:.4f}")
    print(f"Parameters : {best_run.data.params}")
    
    # Register the model
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "mon_modele_production"
    
    print(f"\nRegistering model: {model_name}")
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"Version registered : {registered.version}")
    
    # Update description and tags
    client.update_registered_model(
        name=model_name,
        description="Modèle de classification automatique"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=registered.version,
        key='validated_by',
        value='equipe_data'
    )
    
    # Manage Lifecycle
    SEUIL_PRODUCTION = 0.85
    
    print("\nManaging lifecycle stages...")
    # First promote to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage='Staging',
        archive_existing_versions=False
    )
    print(f"Model v{registered.version} promoted to Staging.")
    
    # Then promote to Production if threshold met
    if best_accuracy >= SEUIL_PRODUCTION:
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage='Production'
        )
        print(f"Model v{registered.version} promoted to Production! (Accuracy {best_accuracy:.3f} >= {SEUIL_PRODUCTION})")
    else:
        print(f"Model NOT promoted to Production : accuracy {best_accuracy:.3f} < {SEUIL_PRODUCTION}")

if __name__ == "__main__":
    register_best_model()
