# ML Hub: Advanced Experimentation Platform

A modern, highly-dense, full-stack Machine Learning experimentation platform combining a robust **Python FastAPI backend** with a sleek **React (Vite) frontend**. Styled to echo professional data analytics tools like *Tableau* and *PowerBI*, ML Hub acts as an interactive command center for training, logging, and benchmarking Scikit-Learn models natively.

![ML Hub Snapshot](https://via.placeholder.com/800x400.png?text=ML+Hub+Dashboard) *(Replace with actual screenshot!)*

## 🚀 Key Features

* **Dense Tableau-Inspired Grid Layout**: No wasted space. Analytics components (like ROC curves, Confusion Matrices, and Feature Importance graphs) are mapped dynamically side-by-side using high-scale glassmorphic CSS grids.
* **Native MLflow Integration**: Telemetry is automatically routed into a local SQLite tracking database via the `mlflow` python APIs. This ensures every single execution is traceable.
* **Interactive Sandbox**: The dashboard provides a dynamic feature prediction sandbox to test fully trained `.joblib` models in real-time within the browser.
* **Principal Component Analysis (PCA)**: Seamlessly perform real-time 2D dimensionality reduction over imported `.csv` datasets and render dynamic Recharts scatter plots instantly.
* **Model Benchmark Leaderboards**: Track, sort, and isolate the top-performing regression or classification algorithms.

## 🛠️ Stack Architecture

**Frontend**
- **Vite 4 / React 19** 
- **Tailwind CSS v4** (Global variables, Glassmorphism, Premium Indigo palettes)
- **Recharts** (Dense, native SVG metric graphs)
- **Framer Motion** (Smooth routing transitions)

**Backend**
- **FastAPI** (High-performance API)
- **Scikit-Learn** (Random Forest, SVM, KNN, Regression)
- **MLFlow** (Telemetry tracking database engine)
- **SQLAlchemy** (Model Metadata ORM)

## 📦 Booting Up Locally

To get the full stack actively running on your local machine, open two separate terminals.

### 1. The FastAPI Backend
Navigate to the `backend` folder, initiate the Python environment, install data dependencies, and run Uvicorn.
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
# source venv/bin/activate

pip install -r requirements.txt

# Start the Python Server with Hot-Reloading
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. The React Frontend
Navigate to the `frontend` folder, pull down the NPM dependencies, and initiate the Vite compiler.
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173/` or `5174` (depending on the Vite output link) to begin training models!

## 🔐 Configuration (CORS)

By default, the Python application actively guards the internal APIs. If your Vite development server opens on an unpredictable port (e.g. `5175`), you will encounter a `Network Error` protecting your data. 
To whitelist an unpredictable testing port, simply add the active localhost address to your `origins` array located in `backend/app/main.py`.

---
*Created dynamically for powerful, iterative ML modeling.*
