"""
HUNGER INDICATOR PROCESSING
File: hunger/hunger-indicator1.py
"""

import pandas as pd
import numpy as np
import os

print("=" * 60)
print("HUNGER DATA PROCESSING")
print("=" * 60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

# Load dataset (CSV format - since your files are .csv)
df = pd.read_csv('suite-of-food-security-indicators_ind.csv')
print(f"Original shape: {df.shape}")

# Keep only required columns
required_cols = ['Item', 'Value', 'Year', 'Area']
existing_cols = [col for col in required_cols if col in df.columns]
df = df[existing_cols]
print(f"After column selection: {df.shape}")

# Handle missing values
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# Filter for undernourishment data
hunger_df = df[df['Item'] == 'Prevalence of undernourishment (percent) (3-year average)']
print(f"Filtered for undernourishment: {hunger_df.shape}")

# Sort by year and get latest
hunger_df = hunger_df.sort_values(by='Year')
latest_hunger = hunger_df.iloc[-1]
national_hunger_rate = latest_hunger['Value']
latest_year = latest_hunger['Year']

print(f"\n✅ National Hunger Rate: {national_hunger_rate}% (Year: {latest_year})")

# Classify hunger intensity
def classify_hunger(rate):
    if rate >= 10:
        return 'High'
    elif rate >= 5:
        return 'Medium'
    else:
        return 'Low'

hunger_df['Hunger_Intensity'] = hunger_df['Value'].apply(classify_hunger)

# Save cleaned dataset
hunger_df.to_csv('processed_data/cleaned_hunger_dataset.csv', index=False)
print(f"✅ Saved to: processed_data/cleaned_hunger_dataset.csv")
print(f"✅ Final shape: {hunger_df.shape}")