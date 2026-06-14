"""
COMBINED ANALYSIS - DEMAND + WASTAGE + HUNGER
File: combined_analysis.py
Run this AFTER the three individual scripts
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

print("=" * 60)
print("COMBINED ANALYSIS - DEMAND + WASTAGE + HUNGER")
print("=" * 60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)
os.makedirs('streamlit_app', exist_ok=True)

# ============================================
# STEP 1: Load preprocessed data
# ============================================

print("\nSTEP 1: Loading preprocessed data")

X_train = pd.read_csv('processed_data/X_train.csv')
X_test = pd.read_csv('processed_data/X_test.csv')
y_train = pd.read_csv('processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('processed_data/y_test.csv').values.ravel()

selected_features_df = pd.read_csv('models/selected_features.csv', header=None)
selected_features = selected_features_df[0].tolist()

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Selected features: {selected_features}")

# ============================================
# STEP 2: Train optimized model
# ============================================

print("\nSTEP 2: Training optimized model")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model R² Score: {r2:.4f} ({r2*100:.2f}%)")

# Save final model
joblib.dump(model, 'models/final_xgboost_model.pkl')
print("✅ Model saved to models/final_xgboost_model.pkl")

# ============================================
# STEP 3: Load full dataset for predictions
# ============================================

print("\nSTEP 3: Loading full dataset")

train = pd.read_csv('train.csv')
meal = pd.read_csv('meal_info.csv')
center = pd.read_csv('fulfilment_center_info.csv')

df = train.merge(meal, on='meal_id').merge(center, on='center_id')
df.drop(columns=['id', 'city_code', 'region_code'], inplace=True)
print(f"Full dataset shape: {df.shape}")

# ============================================
# STEP 4: Preprocess full dataset
# ============================================

print("\nSTEP 4: Preprocessing full dataset")

categorical_cols = ["category", "cuisine", "center_type"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X_full = df[selected_features]
df["predicted_orders"] = model.predict(X_full)

print(f"Predictions complete!")
print(f"Predicted orders range: {df['predicted_orders'].min():.0f} - {df['predicted_orders'].max():.0f}")

# ============================================
# STEP 5: Calculate waste rates
# ============================================

print("\nSTEP 5: Calculating waste rates")

wastage_df = pd.read_csv('processed_data/cleaned_food_wastage_data.csv')
wastage_df["waste_rate"] = wastage_df["Wastage Food Amount"] / wastage_df["Quantity of Food"]
waste_by_food = wastage_df.groupby("Type of Food")["waste_rate"].mean()

print("Waste rates by food type:")
print(waste_by_food.round(4))

# Map categories
meal_info = pd.read_csv('meal_info.csv')
meal_to_category = dict(zip(meal_info['meal_id'], meal_info['category']))
df['category_name'] = df['meal_id'].map(meal_to_category)

def get_food_type(category):
    mapping = {
        'Beverages': 'Beverages',
        'Salad': 'Vegetables', 'Soup': 'Vegetables',
        'Pizza': 'Grains', 'Pasta': 'Grains', 'Rice Bowl': 'Grains',
        'Biryani': 'Grains', 'Sandwich': 'Grains',
        'Meat': 'Meat', 'Seafood': 'Meat', 'Fish': 'Meat',
        'Desert': 'Desserts'
    }
    return mapping.get(category, 'General')

df['food_type'] = df['category_name'].apply(get_food_type)

def get_waste_rate(food_type):
    if food_type == 'Vegetables':
        return waste_by_food.get('Vegetables', 0.068)
    elif food_type == 'Meat':
        return waste_by_food.get('Meat', 0.069)
    elif food_type == 'Beverages':
        return 0.035
    elif food_type == 'Grains':
        return waste_by_food.get('Baked Goods', 0.069)
    else:
        return waste_by_food.mean()

df["waste_rate"] = df["food_type"].apply(get_waste_rate)
df["wastage_kg"] = df["predicted_orders"] * df["waste_rate"]

print(f"Total predicted wastage: {df['wastage_kg'].sum():.0f} kg")

# ============================================
# STEP 6: Apply hunger logic
# ============================================

print("\nSTEP 6: Applying hunger logic")

hunger_df = pd.read_csv('processed_data/cleaned_hunger_dataset.csv')
national_hunger_rate = hunger_df['Value'].iloc[-1]
print(f"National Hunger Rate: {national_hunger_rate}%")

def classify_hunger(rate):
    if rate >= 10:
        return "High"
    elif rate >= 5:
        return "Medium"
    else:
        return "Low"

def week_to_season(week):
    week_in_year = ((week - 1) % 52) + 1
    if week_in_year <= 13:
        return "Winter"
    elif week_in_year <= 26:
        return "Spring"
    elif week_in_year <= 39:
        return "Summer"
    else:
        return "Fall"

seasonal_factors = {"Winter": 1.15, "Spring": 1.05, "Summer": 0.90, "Fall": 0.85}
df["season"] = df["week"].apply(week_to_season)
df["seasonal_factor"] = df["season"].apply(lambda s: seasonal_factors.get(s, 1.0))
df["adjusted_hunger_rate"] = national_hunger_rate * df["seasonal_factor"]
df["hunger_intensity"] = df["adjusted_hunger_rate"].apply(classify_hunger)

print(f"Hunger distribution:\n{df['hunger_intensity'].value_counts()}")

# ============================================
# STEP 7: Redistribution decision
# ============================================

print("\nSTEP 7: Redistribution decision")

df["redistribute"] = (df["wastage_kg"] > 5) & (df["hunger_intensity"].isin(["High", "Medium"]))
print(f"Meals to redistribute: {df['redistribute'].sum():,}")

# ============================================
# STEP 8: Save results
# ============================================

print("\nSTEP 8: Saving results")

redistribution_df = df[df["redistribute"] == True][
    ["center_id", "meal_id", "category_name", "food_type", 
     "predicted_orders", "wastage_kg", "hunger_intensity", "season", "week"]
].copy()

redistribution_df.columns = ["Center ID", "Meal ID", "Category", "Food Type", 
                              "Predicted Orders", "Wastage (kg)", "Hunger Intensity", "Season", "Week"]

# Save to various locations
redistribution_df.to_csv('processed_data/redistribution_recommendations.csv', index=False)
redistribution_df.to_csv('streamlit_app/redistribution_recommendations.csv', index=False)
df.to_csv('processed_data/full_analysis_results.csv', index=False)
df.to_csv('streamlit_app/full_analysis_results.csv', index=False)

print(f"\n✅ Saved {len(redistribution_df)} redistribution recommendations")
print("✅ Files saved to: processed_data/ and streamlit_app/")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
📈 Model Performance:
   - R² Score: {r2:.4f} ({r2*100:.2f}%)
   - Features used: {len(selected_features)}

🍽️ Food Waste Statistics:
   - Total predicted orders: {df['predicted_orders'].sum():,.0f}
   - Total predicted wastage: {df['wastage_kg'].sum():,.0f} kg

🔄 Redistribution Impact:
   - Meals to redistribute: {len(redistribution_df):,}
   - Wastage to redistribute: {redistribution_df['Wastage (kg)'].sum():,.0f} kg

✅ Ready for Streamlit! Run: cd streamlit_app && streamlit run app.py
""")