# ML Hub: Car Price Prediction & MLOps Platform

A specialized, high-performance Machine Learning platform built exclusively for **Regression-based Car Price Prediction**. Combining a robust **Python FastAPI backend** with a sleek **React (Vite) frontend**, ML Hub acts as an interactive command center for training, logging, and benchmarking powerful regression models natively while actively monitoring data drift.

## 🚀 Key Features

* **Regression Focus**: Exclusively optimized for predicting continuous variables (Car Prices) using top-tier algorithms like **XGBoost, Random Forest, AdaBoost, SVR, KNN, and Linear Regression**.
* **Robust Data Pipeline**: Includes automated 3x IQR (Interquartile Range) outlier filtering during preprocessing to ensure extreme anomalies don't ruin model weights and R² scores.
* **Native MLflow Integration**: Telemetry is automatically routed into a local SQLite tracking database via the `mlflow` python APIs. This ensures every single execution is traceable.
* **MLOps & Data Drift Detection**: Integrated with **Evidently AI** to automatically calculate statistical data drift (using Wasserstein Distance and Kolmogorov-Smirnov tests) between uploaded datasets and baseline reference data.
* **Model Benchmark Leaderboards**: Track, sort, and isolate the top-performing regression models with dynamic metrics (R², RMSE, MSE, MAE).
* **Principal Component Analysis (PCA)**: Seamlessly perform real-time dimensionality reduction over the dataset to understand feature variance.

## 🛠️ Stack Architecture

**Frontend**
- **Vite 4 / React 19**
- **Tailwind CSS v4** (Global variables, Glassmorphism, Premium palettes)
- **Recharts** (Dense, native SVG metric graphs)
- **Lucide React** (Modern iconography)

**Backend**
- **FastAPI** (High-performance Async API)
- **Scikit-Learn & XGBoost** (Regression Modeling)
- **MLFlow** (Telemetry tracking & Experiment management)
- **Evidently AI** (Data drift detection reports)
- **SQLAlchemy & SQLite** (Model Metadata ORM)

## 📦 Booting Up Locally

The project has been industrialized to support concurrent startup from the root directory.

### Single Command Startup
Make sure you have Node.js and Python installed. From the **root folder**, simply run:

```bash
# Install root concurrently tool (one time)
npm install

# Start both the Backend (Uvicorn) and Frontend (Vite) simultaneously
npm run dev
```

### Accessing the Dashboards
* **Main Application**: `http://localhost:5173/`
* **Backend API Docs**: `http://localhost:8000/docs`
* **MLflow UI Server**: Run `mlflow ui --backend-store-uri sqlite:///data/mlruns.db --port 5000` inside the backend directory, then visit `http://localhost:5000/`.

## 🔐 Configuration (CORS)

By default, the Python application actively guards the internal APIs. If your Vite development server opens on an unpredictable port (e.g. `5175`), you will encounter a `Network Error` protecting your data. 
To whitelist an unpredictable testing port, simply add the active localhost address to your `allow_origins` array located in `backend/app/main.py`.

---
*Industrialized and optimized for high-accuracy regression modeling.*
