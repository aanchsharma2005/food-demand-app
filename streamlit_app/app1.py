"""
AI-Driven Food Demand & Wastage Prediction System
Run with: cd streamlit_app && streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)

# Load data functions
@st.cache_data
def load_redistribution_data():
    try:
        df = pd.read_csv('redistribution_recommendations.csv')
        return df
    except:
        return None

@st.cache_data
def load_full_analysis():
    try:
        df = pd.read_csv('full_analysis_results.csv')
        return df
    except:
        return None

# Load data
redistribution_df = load_redistribution_data()
full_df = load_full_analysis()

# Sidebar
st.sidebar.markdown("# 🍽️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔄 Redistribution", "📊 Dashboard", "ℹ️ About"])

st.sidebar.markdown("---")
if redistribution_df is not None:
    st.sidebar.metric("🔄 Meals to Redistribute", len(redistribution_df))
    st.sidebar.metric("📦 Total Wastage", f"{redistribution_df['Wastage (kg)'].sum():.0f} kg")
else:
    st.sidebar.info("Run combined_analysis.py first")

# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🍽️ AI-Driven Food Demand & Wastage Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Supporting SDG 2: Zero Hunger</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌟 Welcome
        
        This AI-powered system helps:
        - **Predict** food demand with 86.5% accuracy
        - **Estimate** potential food wastage
        - **Identify** surplus meals for redistribution
        - **Support** communities facing food insecurity
        
        ### 🎯 How It Works
        
        1. XGBoost model analyzes historical order data
        2. Predicts demand for each meal at each center
        3. Calculates waste rates based on food type
        4. Assesses hunger levels using national data
        5. Generates redistribution recommendations
        """)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Model Performance")
        st.markdown("""
        - **R² Score:** 86.5%
        - **MAE:** 55.5 orders
        - **RMSE:** 79.4 orders
        - **Algorithm:** XGBoost
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': ['meal_id', 'checkout_price', 'base_price', 'category', 
                   'cuisine', 'center_id', 'op_area', 'homepage_featured'],
        'Importance': [0.39, 0.36, 0.32, 0.28, 0.07, 0.06, 0.04, 0.03]
    })
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance (Higher = More Important)",
                 color='Importance', color_continuous_scale='Greens')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# REDISTRIBUTION PAGE
# ============================================
elif page == "🔄 Redistribution":
    st.markdown("## 🔄 Redistribution Recommendations")
    
    if redistribution_df is not None and len(redistribution_df) > 0:
        st.success(f"### 🎯 Found {len(redistribution_df)} meals recommended for redistribution!")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Meals", len(redistribution_df))
        with col2:
            st.metric("Total Wastage", f"{redistribution_df['Wastage (kg)'].sum():.0f} kg")
        with col3:
            st.metric("Avg per Meal", f"{redistribution_df['Wastage (kg)'].mean():.1f} kg")
        with col4:
            st.metric("Centers Involved", redistribution_df['Center ID'].nunique())
        
        # Filters
        st.markdown("### 🔍 Filter Recommendations")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            seasons = ['All'] + list(redistribution_df['Season'].unique())
            selected_season = st.selectbox("Season", seasons)
        
        with filter_col2:
            hunger_levels = ['All'] + list(redistribution_df['Hunger Intensity'].unique())
            selected_hunger = st.selectbox("Hunger Intensity", hunger_levels)
        
        # Apply filters
        filtered_df = redistribution_df.copy()
        if selected_season != 'All':
            filtered_df = filtered_df[filtered_df['Season'] == selected_season]
        if selected_hunger != 'All':
            filtered_df = filtered_df[filtered_df['Hunger Intensity'] == selected_hunger]
        
        st.markdown(f"**Showing {len(filtered_df)} recommendations**")
        
        # Display table
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Recommendations (CSV)",
            data=csv,
            file_name="redistribution_recommendations.csv",
            mime="text/csv"
        )
        
        # Visualization
        st.markdown("### 📊 Wastage by Center")
        center_summary = filtered_df.groupby('Center ID')['Wastage (kg)'].sum().reset_index()
        fig = px.bar(center_summary, x='Center ID', y='Wastage (kg)',
                     title="Wastage Available for Redistribution by Center",
                     color='Wastage (kg)', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ No redistribution recommendations found.")
        st.info("Please run combined_analysis.py first")

# ============================================
# DASHBOARD PAGE
# ============================================
elif page == "📊 Dashboard":
    st.markdown("## 📊 Analytics Dashboard")
    
    if full_df is not None:
        # KPI Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Orders", f"{full_df['predicted_orders'].sum():,.0f}")
        with col2:
            st.metric("Total Wastage", f"{full_df['wastage_kg'].sum():,.0f} kg")
        with col3:
            st.metric("Avg Waste Rate", f"{full_df['waste_rate'].mean():.1%}")
        
        # Weekly trends
        st.markdown("### 📅 Weekly Trends")
        weekly_data = full_df.groupby('week').agg({
            'predicted_orders': 'sum',
            'wastage_kg': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_data['week'], y=weekly_data['predicted_orders'],
                                 name='Predicted Orders', line=dict(color='#2E7D32', width=2)))
        fig.add_trace(go.Scatter(x=weekly_data['week'], y=weekly_data['wastage_kg'],
                                 name='Wastage (kg)', line=dict(color='#FF5722', width=2)))
        fig.update_layout(title='Weekly Demand and Wastage Trends',
                         xaxis_title='Week', yaxis_title='Quantity')
        st.plotly_chart(fig, use_container_width=True)
        
        # Hunger and Season distribution
        col1, col2 = st.columns(2)
        
        with col1:
            hunger_dist = full_df['hunger_intensity'].value_counts()
            fig2 = px.pie(values=hunger_dist.values, names=hunger_dist.index,
                         title="Hunger Intensity Distribution",
                         color_discrete_sequence=['#FF4444', '#FFA500', '#4CAF50'])
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            season_dist = full_df['season'].value_counts()
            fig3 = px.bar(x=season_dist.index, y=season_dist.values,
                         title="Distribution by Season",
                         labels={'x': 'Season', 'y': 'Count'},
                         color=season_dist.values, color_continuous_scale='Blues')
            st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.warning("⚠️ Full analysis data not found.")
        st.info("Please run combined_analysis.py first")

# ============================================
# ABOUT PAGE
# ============================================
else:
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This AI-driven system predicts food demand, estimates wastage, and provides redistribution 
    recommendations to support **UN Sustainable Development Goal 2: Zero Hunger**.
    
    ### 📊 Technical Details
    
    | Component | Technology |
    |-----------|------------|
    | Machine Learning | XGBoost Regressor |
    | Feature Selection | Mutual Information |
    | Data Processing | Pandas, NumPy |
    | Visualization | Plotly |
    | UI Framework | Streamlit |
    
    ### 📈 Model Performance
    - **R² Score:** 86.5%
    - **MAE:** 55.5 orders
    - **RMSE:** 79.4 orders
    
    ### 👤 Team Member
    - **Name:** Aanch Sharma
    - **SRN:** PES1PG25CA001
    - **Course:** Artificial Intelligence and Machine Learning
    - **Domain:** SDG 2 - Zero Hunger
    """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>AI-Driven Food Demand System | SDG 2: Zero Hunger</p>", unsafe_allow_html=True)