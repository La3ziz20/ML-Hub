from app.services.data_service import data_service
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd

X_train, X_test, y_train, y_test = data_service.prepare_data(target_column="Price")

# Drop ID
X_train = X_train.drop(columns=['ID'], errors='ignore')
X_test = X_test.drop(columns=['ID'], errors='ignore')

model = RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"RandomForest R2 Score: {r2_score(y_test, y_pred)}")
