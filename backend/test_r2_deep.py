from app.services.data_service import data_service
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pandas as pd

X_train, X_test, y_train, y_test = data_service.prepare_data(target_column="Price")

model = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Deep XGBoost R2 Score: {r2_score(y_test, y_pred)}")

importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)
