"""
Employee Earnings Predictor - Streamlit Glassmorphism Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Employee Earnings ML Predictor",
    page_icon="💼",
    layout="wide"
)

# Custom Glassmorphism Styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .stApp {
        background-color: #090d16;
    }
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Employee Earnings Predictor (XGBoost ML)")
st.caption("Predict employee annual salaries based on machine learning models trained on multi-variate income datasets.")

# Check for model files
if os.path.exists("salary_model.joblib"):
    model = joblib.load("salary_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoders = joblib.load("encoders.joblib")
    st.sidebar.success("✅ XGBoost Model Loaded")
else:
    st.sidebar.warning("⚠️ Run `python train_model.py` first to train the model!")
    model = None

# Sidebar Inputs
st.sidebar.header("👤 Employee Profile Inputs")
age = st.sidebar.slider("Age", 18, 75, 34)
experience = st.sidebar.slider("Work Experience (Years)", 0, min(45, age - 16), 8)
hours = st.sidebar.slider("Hours Per Week", 15, 80, 40)

education = st.sidebar.selectbox("Education Level", ['bachelors', 'masters', 'doctorate', 'associate', 'highschool', 'professional'])
occupation = st.sidebar.selectbox("Job Role / Occupation", ['swe', 'ds_ai', 'mgmt', 'finance', 'health', 'design', 'sales', 'edu', 'trades'])
work_mode = st.sidebar.selectbox("Work Arrangement", ['hybrid', 'remote', 'onsite'])
company_size = st.sidebar.selectbox("Company Scale", ['mid', 'startup', 'enterprise'])
location_tier = st.sidebar.selectbox("Regional Market", ['tier1', 'tier2', 'tier3', 'global'])
gender = st.sidebar.selectbox("Demographic", ['female', 'male', 'nonbinary'])

# Perform Prediction
if model is not None:
    # Prepare input dataframe
    input_data = pd.DataFrame([{
        'age': age,
        'experience': experience,
        'hours': hours,
        'education': encoders['education'].transform([education])[0],
        'occupation': encoders['occupation'].transform([occupation])[0],
        'work_mode': encoders['work_mode'].transform([work_mode])[0],
        'company_size': encoders['company_size'].transform([company_size])[0],
        'location_tier': encoders['location_tier'].transform([location_tier])[0],
        'gender': encoders['gender'].transform([gender])[0]
    }])
    
    # Scale numericals
    input_data[['age', 'experience', 'hours']] = scaler.transform(input_data[['age', 'experience', 'hours']])
    
    pred_salary = model.predict(input_data)[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Salary", f"${pred_salary:,.0f} / yr")
    col2.metric("Monthly Estimate", f"${pred_salary/12:,.0f}")
    col3.metric("Hourly Equivalent", f"${pred_salary/(hours*52):,.2f} / hr")
    col4.metric("Confidence Interval", f"${pred_salary*0.82:,.0f} - ${pred_salary*1.22:,.0f}")

    st.markdown("---")
    
    # Feature Importance Plot
    st.subheader("📊 Model Feature Importances")
    importance_df = pd.DataFrame({
        'Feature': ['Age', 'Experience', 'Hours', 'Education', 'Occupation', 'Work Mode', 'Company Size', 'Location Tier', 'Gender'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="XGBoost Feature Importance Breakdown", color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please run `python train_model.py` to generate the `.joblib` model artifact.")
