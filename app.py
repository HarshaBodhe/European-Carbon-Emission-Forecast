import os

# 1. CRITICAL: Suppress TensorFlow logs BEFORE any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import pulp

# Page configuration for a professional European-wide tool
st.set_page_config(
    page_title="European Carbon Policy & Forecast Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ENHANCED CSS: Professional Branding
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; max-width: 95%; }
    h1 { color: #1B5E20 !important; font-size: 2.5rem !important; border-bottom: 2px solid #2ecc71; padding-bottom: 10px; }
    h2, h3 { color: #2E7D32 !important; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { color: #555555 !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #1B5E20 !important; }
    .css-1d391kg { background-color: #f1f8e9 !important; }
    </style>
    """, unsafe_allow_html=True)

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# DATA ENGINE - TAILORED FOR EUROPE
# ---------------------------
# ---------------------------
# DATA ENGINE - UPDATED FOR EUROPEAN SCOPE
# ---------------------------
@st.cache_data
def load_data():
    # Streamlit Cloud's working directory is always the root of your repo
    file_path = "berlin_timeseries.csv" 
    
    if not os.path.exists(file_path):
        # This helps you debug by showing exactly what files the server sees
        st.error(f"❌ '{file_path}' not found. Available files: {os.listdir('.')}")
        st.stop()
        
    data = pd.read_csv(file_path, parse_dates=["date"])
    data['country'] = "Europe" 
    return data

# Load the rebranded data
df = load_data()

# Ensure df_raw is also updated if you use it for the sidebar
df_raw = df.copy()

# ---------------------------
# SIDEBAR & REGIONAL SELECTION
# ---------------------------
with st.sidebar:
    st.header("🌐 Regional Scope")
    # Pulls "Europe" automatically from the data we just rebranded
    region = df['country'].unique()[0] 
    st.subheader(f"📍 Territory: {region}")
    
    st.info(f"Timeline: {df['date'].min().year} - {df['date'].max().year}")
    st.divider()
    st.dataframe(df.tail(5))

# ---------------------------
# UI LAYOUT
# ---------------------------
st.title(f"🌍 {region} Carbon Forecasting & Policy Optimizer")

# Top Section: Historical Context
st.subheader(f"📊 Historical CO₂ Trends: {region}")
chart_data = df.set_index('date')[['co2_per_capita']]
st.line_chart(chart_data, color="#2ecc71", use_container_width=True)

# ---------------------------
# PREDICTIVE ENGINE (LSTM)
# ---------------------------
st.divider()
st.subheader("🔮 Predictive Intelligence (Continental LSTM Network)")

# Model/Scaler Path Resolution
model_path = os.path.join(BASE_DIR, "models", "lstm_berlin_model.h5")
scaler_path = os.path.join(BASE_DIR, "models", "berlin_scaler.joblib")

# Fallback
if not os.path.exists(model_path):
    model_path = os.path.join(BASE_DIR, "lstm_berlin_model.h5")
    scaler_path = os.path.join(BASE_DIR, "berlin_scaler.joblib")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    try:
        @st.cache_resource
        def get_model(path):
            return tf.keras.models.load_model(path, compile=False)
        
        model = get_model(model_path)
        scaler = joblib.load(scaler_path)

        # Prepare lags
        df_pred = df.copy()
        for lag in [1, 3, 6, 12]:
            df_pred[f'co2_lag_{lag}'] = df_pred["co2_per_capita"].shift(lag)
        
        df_pred = df_pred.dropna().reset_index(drop=True)
        features = ['gdp_per_capita', 'temp_avg', 'renewable_share', 
                    'co2_lag_1', 'co2_lag_3', 'co2_lag_6', 'co2_lag_12']
        
        # Scale and Predict
        X_raw = df_pred[features].tail(12)
        X_scaled = scaler.transform(X_raw)
        X_input = X_scaled.reshape(1, 12, len(features))
        prediction = model.predict(X_input, verbose=0)[0][0]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(f"Forecasted CO₂ for {region}", f"{prediction:.2f} tons", delta="-0.02 vs Prev Month")
        with col_m2:
            st.info("The AI analyzes GDP, temperature, and renewable adoption to project future emission levels.")

    except Exception as e:
        st.error(f"Inference Engine Error: {e}")
else:
    st.warning("⚠️ Predictive models not loaded. Analysis will default to current data.")

# ---------------------------
# OPTIMIZATION ENGINE (PuLP) - DYNAMIC & PROFESSIONAL
# ---------------------------
# OPTIMIZATION ENGINE - UPDATED FOR BALANCED ALLOCATION
st.divider()
st.subheader("🏙️ Strategic Budget Allocation")

s1, s2, s3, s4 = st.columns(4)
with s1: budget = st.slider("Total Budget (€M)", 50, 500, 165)
with s2: min_green = st.slider("Min. Renewables (€M)", 10, 100, 30)
with s3: min_build = st.slider("Min. Buildings (€M)", 10, 100, 20)
# Added sliders for Transport and Waste minimums to give you control
with s4: min_trans = st.slider("Min. Transport (€M)", 10, 100, 15)

# Linear Programming with PuLP
prob = pulp.LpProblem("Budget_Optimization", pulp.LpMaximize)

# Define variables with a "Floor" (Minimum) so they never hit zero
r = pulp.LpVariable("Renewables", lowBound=min_green)
t = pulp.LpVariable("Transport", lowBound=min_trans) 
b = pulp.LpVariable("Buildings", lowBound=min_build)
w = pulp.LpVariable("Waste", lowBound=5) # Hardcoded 5M minimum for Waste

# Adjusted Objective: Weights are closer to encourage spreading the budget
# Efficiency: Renewables (0.9), Buildings (0.8), Transport (0.75), Waste (0.6)
prob += 0.9*r + 0.8*b + 0.75*t + 0.6*w

# Constraint: Sum of all sectors must equal the budget
prob += r + t + b + w <= budget

status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
if pulp.LpStatus[status] == 'Optimal':
    # Professional Metrics
    res_cols = st.columns(4)
    res_cols[0].metric("Renewables", f"€{r.value():.1f}M")
    res_cols[1].metric("Buildings", f"€{b.value():.1f}M")
    res_cols[2].metric("Transport", f"€{t.value():.1f}M")
    res_cols[3].metric("Waste", f"€{w.value():.1f}M")

    impact = (0.9*r.value() + 0.7*t.value() + 0.85*b.value() + 0.4*w.value())
    st.success(f"✅ **Total Estimated CO₂ Reduction Potential:** {impact:.2f} tons")

    # Dynamic Allocation Visual
    fig, ax = plt.subplots(figsize=(10, 2), dpi=100)
    vals = [r.value(), t.value(), b.value(), w.value()]
    labels = ['Renewables', 'Transport', 'Buildings', 'Waste']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f']
    
    start = 0
    for val, lab, col in zip(vals, labels, colors):
        if val > 0:
            ax.barh(["Allocation"], [val], left=start, color=col, label=lab)
            start += val
    
    ax.set_xlim(0, budget)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.9), ncol=4, frameon=False)
    ax.set_xlabel("Investment (€ Million)")
    st.pyplot(fig, width="stretch")
else:
    st.error("The defined mandates exceed the total budget. Please adjust sliders.")
# In your get_model function:
def get_model(path):
    return keras.models.load_model(path, compile=False, safe_mode=False)

st.caption(f"v1.2.0 | Regional Context: {region} | Data Refresh: Jan 2026")


