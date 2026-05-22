from app.services.data_service import data_service
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = data_service.prepare_data(target_column="Price")

# Drop ID
X_train = X_train.drop(columns=['ID'], errors='ignore')
X_test = X_test.drop(columns=['ID'], errors='ignore')

# Log transform
y_train_log = np.log1p(y_train)

model = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8)
model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

print(f"Log-Transformed XGBoost R2 Score: {r2_score(y_test, y_pred)}")
print(f"Log-Transformed Log-R2 Score (fake): {r2_score(np.log1p(y_test), y_pred_log)}")
