from app.services.data_service import data_service
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd

# Let's monkeypatch data_service to use 1.0 IQR
def get_tight_data():
    df = pd.read_csv(data_service.data_path)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price'])
    
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 0.5 * IQR
    lower_bound = Q1 - 0.5 * IQR
    df = df[(df['Price'] <= upper_bound) & (df['Price'] >= lower_bound)]
    
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    if 'Mileage' in X.columns and X['Mileage'].dtype == 'object':
        X['Mileage'] = X['Mileage'].astype(str).str.replace(' km', '', regex=False).str.replace(' ', '', regex=False)
        X['Mileage'] = pd.to_numeric(X['Mileage'], errors='coerce')
    if 'Engine volume' in X.columns and X['Engine volume'].dtype == 'object':
        X['Engine volume'] = X['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    if 'Levy' in X.columns:
        X['Levy'] = pd.to_numeric(X['Levy'], errors='coerce')
        
    num_cols = X.select_dtypes(include=['number']).columns
    X[num_cols] = X[num_cols].fillna(0)
    
    cat_cols = X.select_dtypes(include=['object']).columns
    from sklearn.preprocessing import LabelEncoder
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_tight_data()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"RandomForest TIGHT R2 Score: {r2_score(y_test, y_pred)}")
