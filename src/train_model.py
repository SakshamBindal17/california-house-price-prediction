import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

gb_best = GradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=4,
    n_estimators=200,
    subsample=1.0,
    random_state=42
)
gb_best.fit(X_train_scaled, y_train)

joblib.dump(gb_best, '../models/tuned_gradient_boosting_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
