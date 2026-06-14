"""
AI-Driven Food Demand & Wastage Prediction System
Run with: cd streamlit_app && python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import time

# Page configuration
st.set_page_config(
    page_title="Food Demand & Redistribution System",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .donate-btn {
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        width: 100%;
    }
    .donate-btn:hover {
        background-color: #45a049;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .hunger-high {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .hunger-medium {
        background-color: #ffa500;
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .hunger-low {
        background-color: #4caf50;
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'donation_made' not in st.session_state:
    st.session_state.donation_made = False
if 'phone_number' not in st.session_state:
    st.session_state.phone_number = ''
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Load data functions
@st.cache_data
def load_meal_data():
    try:
        meal_df = pd.read_csv('../meal_info.csv')
        return meal_df
    except:
        # Fallback data if file not found
        return pd.DataFrame({
            'meal_id': [1885, 1993, 2539, 2139, 2631, 1248, 1778, 1062, 2707, 1207],
            'category': ['Beverages', 'Beverages', 'Beverages', 'Beverages', 'Beverages', 
                        'Rice Bowl', 'Pizza', 'Pasta', 'Biryani', 'Sandwich'],
            'cuisine': ['Thai', 'Thai', 'Thai', 'Indian', 'Indian', 
                       'Indian', 'Italian', 'Italian', 'Indian', 'Continental']
        })

@st.cache_data
def load_center_data():
    try:
        center_df = pd.read_csv('../fulfilment_center_info.csv')
        return center_df
    except:
        return pd.DataFrame({
            'center_id': [11, 13, 55, 66, 94, 124],
            'center_type': ['TYPE_A', 'TYPE_B', 'TYPE_C', 'TYPE_A', 'TYPE_C', 'TYPE_B'],
            'op_area': [3.7, 6.7, 2.0, 4.1, 3.6, 4.0]
        })

@st.cache_data
def load_hunger_data():
    try:
        hunger_df = pd.read_csv('../processed_data/cleaned_hunger_dataset.csv')
        return hunger_df
    except:
        return pd.DataFrame({
            'Year': [2024],
            'Value': [12.0],
            'Hunger_Intensity': ['High']
        })

# Load data
meal_df = load_meal_data()
center_df = load_center_data()
hunger_df = load_hunger_data()

# Get national hunger rate
national_hunger_rate = hunger_df['Value'].iloc[-1] if len(hunger_df) > 0 else 12.0

# Function to get hunger intensity based on location type
def get_hunger_intensity(center_id):
    # Get center type
    center_info = center_df[center_df['center_id'] == center_id]
    if len(center_info) > 0:
        center_type = center_info['center_type'].iloc[0]
        # Map center type to location
        location_map = {'TYPE_A': 'Urban', 'TYPE_B': 'Suburban', 'TYPE_C': 'Rural'}
        location = location_map.get(center_type, 'Urban')
        
        # Hunger rates by location (simulated based on typical patterns)
        location_hunger = {
            'Urban': national_hunger_rate * 0.8,      # Urban: lower hunger
            'Suburban': national_hunger_rate * 1.0,    # Suburban: average
            'Rural': national_hunger_rate * 1.3        # Rural: higher hunger
        }
        hunger_rate = location_hunger.get(location, national_hunger_rate)
        
        # Classify
        if hunger_rate >= 10:
            return "High", hunger_rate, location
        elif hunger_rate >= 5:
            return "Medium", hunger_rate, location
        else:
            return "Low", hunger_rate, location
    return "Medium", national_hunger_rate, "Unknown"

# Function to predict demand (simplified - replace with your actual model)
def predict_demand(center_id, meal_id, season):
    """
    Simplified prediction logic.
    Replace this with your actual XGBoost model prediction.
    """
    # Base demand by meal
    meal_demand = {
        1885: 309, 1993: 205, 2539: 189, 2139: 94, 2631: 40,
        1248: 32, 1778: 141, 1062: 141, 2707: 377, 1207: 501
    }
    
    # Seasonal factors
    seasonal_factors = {
        "Winter": 1.15,
        "Spring": 1.05,
        "Summer": 0.90,
        "Fall": 0.85
    }
    
    base = meal_demand.get(meal_id, 150)
    factor = seasonal_factors.get(season, 1.0)
    
    # Center size factor
    center_info = center_df[center_df['center_id'] == center_id]
    if len(center_info) > 0:
        op_area = center_info['op_area'].iloc[0]
        area_factor = 0.8 + (op_area / 20)  # Larger area = more orders
    else:
        area_factor = 1.0
    
    predicted = int(base * factor * area_factor)
    return max(predicted, 10)  # Minimum 10 orders

# Function to calculate wastage
def calculate_wastage(predicted_orders, category):
    waste_rates = {
        'Beverages': 0.035,
        'Vegetables': 0.068,
        'Meat': 0.069,
        'Grains': 0.069,
        'Desserts': 0.070,
        'Snacks': 0.065
    }
    
    # Map category to food type
    category_to_food = {
        'Beverages': 'Beverages',
        'Rice Bowl': 'Grains',
        'Starters': 'Snacks',
        'Pasta': 'Grains',
        'Sandwich': 'Grains',
        'Biryani': 'Grains',
        'Pizza': 'Grains',
        'Seafood': 'Meat',
        'Desert': 'Desserts',
        'Soup': 'Vegetables',
        'Salad': 'Vegetables'
    }
    
    food_type = category_to_food.get(category, 'Grains')
    waste_rate = waste_rates.get(food_type, 0.068)
    wastage = predicted_orders * waste_rate
    return wastage, waste_rate

# Function to send proof (simulated)
def send_donation_proof(phone_number, center_id, meal_id, wastage_kg):
    # In a real app, this would send an SMS or email
    # For demo, we'll just simulate
    proof_message = f"""
    🍽️ DONATION PROOF
    
    Center ID: {center_id}
    Meal ID: {meal_id}
    Quantity Donated: {wastage_kg:.1f} kg of food
    Recipient: Local Community Kitchen
    Date: {time.strftime('%Y-%m-%d')}
    Time: {time.strftime('%H:%M:%S')}
    
    ✅ Donation successfully delivered to those in need!
    """
    return proof_message

# ============================================
# HOME PAGE (Donation Page)
# ============================================

st.markdown('<p class="main-header">🍽️ AI-Driven Food Demand & Wastage Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Supporting SDG 2: Zero Hunger</p>', unsafe_allow_html=True)

# Check if we're showing donation success page
if st.session_state.get('show_donation_success', False):
    st.markdown(f"""
    <div class="success-box">
        <h3>🎉 Thank You for Your Donation!</h3>
        <p>Your generosity will help feed someone in need. A confirmation message has been sent.</p>
        <p><strong>Center ID:</strong> {st.session_state.get('last_center', 'N/A')}</p>
        <p><strong>Meal ID:</strong> {st.session_state.get('last_meal', 'N/A')}</p>
        <p><strong>Quantity Donated:</strong> {st.session_state.get('last_wastage', 0):.1f} kg</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ask if they want proof
    st.markdown("### 📱 Would you like to receive proof of donation?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        phone = st.text_input("Enter your phone number (with country code):", 
                              placeholder="+91 XXXXXXXXXX",
                              key="proof_phone")
    with col2:
        if st.button("📩 Send Proof", use_container_width=True):
            if phone:
                proof = send_donation_proof(phone, 
                                           st.session_state.get('last_center', 'N/A'),
                                           st.session_state.get('last_meal', 'N/A'),
                                           st.session_state.get('last_wastage', 0))
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Proof Sent!</h4>
                    <p>A donation receipt has been sent to <strong>{phone}</strong></p>
                    <p>The local NGO will also contact you with photos of the distribution.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add home button after proof
                if st.button("🏠 Return to Home", use_container_width=True):
                    st.session_state.show_donation_success = False
                    st.session_state.donation_made = False
                    st.rerun()
            else:
                st.error("Please enter a valid phone number")
    
    # Skip proof option
    if st.button("⏭️ Skip & Return to Home", use_container_width=True):
        st.session_state.show_donation_success = False
        st.session_state.donation_made = False
        st.rerun()
    
    st.stop()

# ============================================
# MAIN PREDICTION FORM
# ============================================

st.markdown("## 📊 Predict Demand & Check Wastage")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    # Center ID selection
    center_ids = center_df['center_id'].tolist()
    selected_center = st.selectbox("🏪 Select Center ID", center_ids)
    
    # Get center info
    center_info = center_df[center_df['center_id'] == selected_center]
    if len(center_info) > 0:
        center_type = center_info['center_type'].iloc[0]
        op_area = center_info['op_area'].iloc[0]
        st.caption(f"📍 Type: {center_type} | Area: {op_area} sq units")

with col2:
    # Category selection (unique categories)
    categories = meal_df['category'].unique().tolist()
    selected_category = st.selectbox("🍽️ Select Food Category", categories)
    
    # Filter meals by selected category
    filtered_meals = meal_df[meal_df['category'] == selected_category]
    selected_meal = st.selectbox("🍲 Select Meal", filtered_meals['meal_id'].tolist(),
                                  format_func=lambda x: f"{x} - {meal_df[meal_df['meal_id']==x]['cuisine'].iloc[0]} Cuisine")

# Season selection
season = st.selectbox("🌤️ Select Season", ["Winter", "Spring", "Summer", "Fall"])

# Get hunger intensity for this center
hunger_intensity, hunger_rate, location = get_hunger_intensity(selected_center)

# ============================================
# PREDICT BUTTON
# ============================================

if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
    with st.spinner("Analyzing data..."):
        # Simulate processing
        time.sleep(1)
        
        # Make prediction
        predicted_orders = predict_demand(selected_center, selected_meal, season)
        
        # Calculate wastage
        wastage_kg, waste_rate = calculate_wastage(predicted_orders, selected_category)
        
        # Store in session state
        st.session_state.prediction_result = {
            'center_id': selected_center,
            'meal_id': selected_meal,
            'category': selected_category,
            'predicted_orders': predicted_orders,
            'wastage_kg': wastage_kg,
            'waste_rate': waste_rate,
            'hunger_intensity': hunger_intensity,
            'hunger_rate': hunger_rate,
            'location': location,
            'season': season
        }

# ============================================
# DISPLAY RESULTS (if prediction exists)
# ============================================

if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    # Results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Predicted Orders", f"{result['predicted_orders']:,}")
        st.caption(f"Based on {result['season']} season demand patterns")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🗑️ Expected Wastage", f"{result['wastage_kg']:.1f} kg")
        st.caption(f"Waste rate: {result['waste_rate']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌍 Hunger Rate", f"{result['hunger_rate']:.1f}%")
        st.caption(f"Location: {result['location']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Hunger Intensity Display
    st.markdown("### 🎯 Hunger Intensity in This Area")
    if result['hunger_intensity'] == 'High':
        st.markdown(f"""
        <div class="hunger-high">
            🔴 HIGH HUNGER INTENSITY - {result['hunger_rate']:.1f}% undernourishment<br>
            Your donation can make a significant impact here!
        </div>
        """, unsafe_allow_html=True)
    elif result['hunger_intensity'] == 'Medium':
        st.markdown(f"""
        <div class="hunger-medium">
            🟠 MEDIUM HUNGER INTENSITY - {result['hunger_rate']:.1f}% undernourishment<br>
            Your donation would be greatly appreciated.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="hunger-low">
            🟢 LOW HUNGER INTENSITY - {result['hunger_rate']:.1f}% undernourishment<br>
            Still, every donation helps build food security.
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Info
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **📋 Summary for Center {result['center_id']} | Meal {result['meal_id']} ({result['category']})**
    
    - This meal typically generates **{result['wastage_kg']:.1f} kg** of potential wastage
    - In {result['season']}, demand is {'higher' if result['season'] in ['Winter', 'Spring'] else 'lower'} than average
    - The area has **{result['hunger_intensity'].lower()} food insecurity**
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # DONATE BUTTON (Always visible regardless of intensity)
    st.markdown("---")
    st.markdown("### 🤝 Make a Difference")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🍽️ DONATE THIS MEAL", type="primary", use_container_width=True):
            # Store donation info
            st.session_state.last_center = result['center_id']
            st.session_state.last_meal = result['meal_id']
            st.session_state.last_wastage = result['wastage_kg']
            st.session_state.show_donation_success = True
            st.rerun()

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    <p>🤝 Every meal donated helps someone in need. Together we can achieve Zero Hunger.</p>
    <p>AI-Driven Food Demand System | SDG 2: Zero Hunger</p>
</div>
""", unsafe_allow_html=True)

