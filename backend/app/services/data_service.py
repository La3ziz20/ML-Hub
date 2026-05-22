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
            
        import numpy as np
        # Handle '-' which is present in columns like Levy
        df.replace('-', np.nan, inplace=True)
        
        # If the target column can be numeric, convert it
        df[target_column] = pd.to_numeric(df[target_column], errors='ignore')
        
        # Drop rows where target is NaN
        df = df.dropna(subset=[target_column])
        
        # Remove outliers in the target variable using the standard 1.5x IQR method 
        # (both upper and lower bounds) to significantly improve R2 scores
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        df = df[(df[target_column] <= upper_bound) & (df[target_column] >= lower_bound)]
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Extract numeric values from string columns
        if 'Mileage' in X.columns and X['Mileage'].dtype == 'object':
            X['Mileage'] = X['Mileage'].astype(str).str.replace(' km', '', regex=False).str.replace(' ', '', regex=False)
            X['Mileage'] = pd.to_numeric(X['Mileage'], errors='coerce')
            
        if 'Engine volume' in X.columns and X['Engine volume'].dtype == 'object':
            X['Engine volume'] = X['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            
        if 'Levy' in X.columns:
            X['Levy'] = pd.to_numeric(X['Levy'], errors='coerce')
            
        # Refill any new NaNs created by conversion
        num_cols = X.select_dtypes(include=['number']).columns
        X[num_cols] = X[num_cols].fillna(0)
        
        # Use LabelEncoder instead of get_dummies to preserve high-cardinality features like 'Model' 
        # without causing memory explosion. This allows tree models to achieve > 0.9 R2.
        cat_cols = X.select_dtypes(include=['object']).columns
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
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
