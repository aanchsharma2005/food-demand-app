"""
FOOD WASTAGE DATA CLEANING
File: wastage/food_wastage.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

print("=" * 60)
print("FOOD WASTAGE DATA CLEANING")
print("=" * 60)

# Create directories
os.makedirs('processed_data', exist_ok=True)

# Load dataset (CSV format)
df = pd.read_csv('food_wastage_data.csv')
print(f"Original shape: {df.shape}")

# Drop irrelevant columns
cols_to_drop = ['Event Type', 'Purchase History', 'Preparation Method', 'Pricing']
existing_cols = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=existing_cols, inplace=True)
print(f"After dropping columns: {df.shape}")
print(f"Remaining columns: {df.columns.tolist()}")

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

# Fill missing values
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# Outlier treatment (capping)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

print("Outlier treatment completed")

# Calculate waste rate
df['waste_rate'] = df['Wastage Food Amount'] / df['Quantity of Food']
print("\nWaste rates calculated")

# Calculate average waste rate by food type
waste_by_food = df.groupby('Type of Food')['waste_rate'].mean()
print("\nAverage Waste Rates by Food Type:")
print(waste_by_food.round(4))

# Label encode categorical columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print(f"\n✅ Final shape: {df.shape}")

# Save cleaned dataset
df.to_csv('processed_data/cleaned_food_wastage_data.csv', index=False)
waste_by_food.to_csv('processed_data/waste_rate_by_food_type.csv')
print("✅ Saved to: processed_data/cleaned_food_wastage_data.csv")