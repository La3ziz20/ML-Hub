from app.services.data_service import data_service
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = data_service.prepare_data(target_column="Price")

model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"NEW XGBoost R2 Score: {r2_score(y_test, y_pred)}")
