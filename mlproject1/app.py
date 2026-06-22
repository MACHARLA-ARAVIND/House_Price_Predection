import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide"
)

# 2. Setup safe paths relative to this file's folder (mlproject1)
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'rf_model.joblib')
data_path = os.path.join(base_dir, 'cleaned_data.csv')

# 3. Load the trained model and dataset
@st.cache_resource
def load_assets():
    # Load your Random Forest model dynamically
    model = joblib.load(model_path)
    
    # Load your raw dataset dynamically
    df = pd.read_csv(data_path)
    
    # Recreate the LabelEncoder exactly how you did in cell [22] of EDA.ipynb
    encoder = LabelEncoder()
    # Fit it on the original location text column
    encoder.fit(df["location"])
    
    return model, df, encoder

try:
    model, df, encoder = load_assets()
    # Sort the locations alphabetically for a clean dropdown selection menu
    location_list = sorted(df['location'].unique())
except Exception as e:
    st.error(f"Error loading project assets: {e}")
    st.stop()

# 4. Custom Styling (CSS Layout)
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #2E4053;
        margin-bottom: 20px;
    }
    .prediction-text {
        font-size: 26px;
        font-weight: bold;
        color: #1F618D;
        background-color: #EBF5FB;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2980B9;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 5. App Layout Split (Columns)
col1, col2 = st.columns([1, 1.2], gap="large")

# Left Column: Brand Identity Visuals
with col1:
    st.markdown('<div class="main-title">House Price Prediction App</div>', unsafe_allow_html=True)
    
    # Vector simulation of house icon
    st.markdown("""
        <div style="text-align: center; margin-top: 30px;">
            <svg width="250" height="200" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <path d="M10 60 L50 20 L90 60 H80 V90 H20 V60 Z" fill="none" stroke="#2E4053" stroke-width="4" stroke-linejoin="round"/>
                <path d="M40 90 V65 H60 V90" fill="none" stroke="#2E4053" stroke-width="4"/>
                <rect x="30" y="45" width="12" height="12" fill="none" stroke="#2E4053" stroke-width="3"/>
                <rect x="58" y="45" width="12" height="12" fill="none" stroke="#2E4053" stroke-width="3"/>
                <path d="M5 62 Q 50 45, 95 62" fill="none" stroke="#2E4053" stroke-width="3"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)

# Right Column: User Manual Inputs & Interactivity
with col2:
    st.write("### Property Specifications")
    
    # Input placement grid split layout
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        selected_location = st.selectbox("🗺️ Location:", options=location_list)
        sqft = st.number_input("📐 Sq.ft:", min_value=100, max_value=50000, value=1000, step=50)
        
    with input_col2:
        bathrooms = st.selectbox("🛁 No of bathrooms:", options=[1, 2, 3, 4, 5, 6, 7, 8], index=1)
        bhk = st.selectbox("🏠 BHK:", options=[1, 2, 3, 4, 5, 6, 7, 8], index=1)
        
    st.write("")
    predict_btn = st.button("💲 Predict", use_container_width=True)

    # 6. Prediction Logic Execution
    if predict_btn:
        try:
            # Match cell [22]: Encode the text choice into the matching number using your LabelEncoder object
            encoded_value = encoder.transform([selected_location])[0]

            # Construct the DataFrame with columns matching your notebook features EXACTLY:
            # total_sqft, bath, bhk, encoded_loc
            input_features = pd.DataFrame([{
                'total_sqft': float(sqft),
                'bath': float(bathrooms),
                'bhk': float(bhk),
                'encoded_loc': int(encoded_value)
            }])
            
            # Reorder columns explicitly to match the position array expected by the model
            input_features = input_features[['total_sqft', 'bath', 'bhk', 'encoded_loc']]
            
            # Run inference
            predicted_value = model.predict(input_features)[0]
            
            # Output display formatting 
            st.markdown(
                f'<div class="prediction-text">Predicted Price: Lakhs/Rs. {predicted_value:,.2f}</div>', 
                unsafe_allow_html=True
            )
            
        except Exception as err:
            st.error(f"Pipeline Execution Error: {err}")
