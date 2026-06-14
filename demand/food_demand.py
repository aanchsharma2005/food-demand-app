"""
FOOD DEMAND PREDICTION MODEL
File: demand/food_demand.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

print("=" * 60)
print("FOOD DEMAND PREDICTION MODEL")
print("=" * 60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

# Load datasets (CSV format)
train = pd.read_csv('train.csv')
meal = pd.read_csv('meal_info.csv')
center = pd.read_csv('fulfilment_center_info.csv')
print(f"Train: {train.shape}, Meal: {meal.shape}, Center: {center.shape}")

# Merge datasets
df = train.merge(meal, on='meal_id').merge(center, on='center_id')
df.drop(columns=['id', 'city_code', 'region_code'], inplace=True)
print(f"After merge: {df.shape}")

# Check missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Separate columns
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

# Fill missing values
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

# Cap outliers for num_orders and op_area
cols_to_cap = ['num_orders', 'op_area']
for col in cols_to_cap:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
    print(f"Capped {col} to [{lower:.2f}, {upper:.2f}]")

# Label encoding
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("Label encoding completed")

# Prepare features and target
X = df.drop(columns=['num_orders'])
y = df['num_orders']
print(f"Features shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Mutual Information Feature Selection
mi_scores = mutual_info_regression(X, y, random_state=42)
scores_mi = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("\nMutual Information scores:")
print(scores_mi.round(4))

# Select features with MI > 0.03
selected_features = scores_mi[scores_mi > 0.03].index.tolist()
print(f"\nSelected features: {selected_features}")
print(f"Dropped: {[f for f in X.columns if f not in selected_features]}")

X_selected = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, shuffle=True, random_state=42
)
print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")

# Train XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
print("Model training completed")

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")

# Save model and data
joblib.dump(model, 'models/xgboost_model.pkl')
pd.Series(selected_features).to_csv('models/selected_features.csv', index=False, header=False)
df.to_csv('processed_data/cleaned_DEMAND_dataset.csv', index=False)
X_train.to_csv('processed_data/X_train.csv', index=False)
X_test.to_csv('processed_data/X_test.csv', index=False)
y_train.to_csv('processed_data/y_train.csv', index=False)
y_test.to_csv('processed_data/y_test.csv', index=False)

print("\n✅ Model saved to: models/xgboost_model.pkl")
print("✅ Data saved to: processed_data/")