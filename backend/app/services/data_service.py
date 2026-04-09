import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataService:
    def __init__(self):
        self._refresh_path()

    def _refresh_path(self):
        os.makedirs("data", exist_ok=True)
        if os.path.exists("data/uploaded_dataset.csv"):
            self.data_path = "data/uploaded_dataset.csv"
        elif os.path.exists("data/cleaned_dataset.csv"):
            self.data_path = "data/cleaned_dataset.csv"
        elif os.path.exists("../cleaned_dataset.csv"):
            self.data_path = "../cleaned_dataset.csv"
        else:
            self.data_path = "cleaned_dataset.csv"

    def set_uploaded_path(self):
        self.data_path = "data/uploaded_dataset.csv"

    def get_dataset_preview(self, num_rows: int = 10):
        self._refresh_path()
        try:
            df = pd.read_csv(self.data_path)
            preview = df.head(num_rows).to_dict(orient="records")
            columns = df.columns.tolist()
            return {"columns": columns, "preview": preview, "total_rows": len(df)}
        except Exception as e:
            return {"error": str(e)}

    def get_dataset(self):
        self._refresh_path()
        return pd.read_csv(self.data_path)

    def prepare_data(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self._refresh_path()
        df = pd.read_csv(self.data_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
            
        # Basic preprocessing
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Simple encoding for categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

    def run_pca(self, target_column: str = None):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self._refresh_path()
        df = pd.read_csv(self.data_path)
        
        df = df.dropna()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {"error": "Not enough numeric columns for PCA."}
            
        features = df[numeric_cols]
        target_data = []
        
        if target_column and target_column in df.columns:
            target_data = df[target_column].tolist()
            if target_column in features.columns:
                features = features.drop(columns=[target_column])
                
        scaled_features = StandardScaler().fit_transform(features)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_features)
        
        return {
            "pca1": components[:, 0].tolist(),
            "pca2": components[:, 1].tolist(),
            "target": target_data,
            "variance_ratio": pca.explained_variance_ratio_.tolist()
        }

data_service = DataService()
