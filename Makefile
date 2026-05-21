.PHONY: setup train register serve drift_check test

setup:
	npm run install:all

train:
	cd backend && .\venv\Scripts\python src/train_batch.py

register:
	cd backend && .\venv\Scripts\python src/register_best_model.py

serve:
	cd backend && .\venv\Scripts\mlflow models serve -m "models:/mon_modele_production/Production" --port 1234 --no-conda

drift_check:
	cd backend && .\venv\Scripts\python src/simulate_drift.py

test:
	cd backend && .\venv\Scripts\python src/test_api.py

pipeline: train register serve test
	@echo "Pipeline complet execute avec succes"
