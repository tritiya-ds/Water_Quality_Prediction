# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .prediction-box {
        background: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load the model and structure
@st.cache_resource
def load_models():
    import os
    import urllib.request
    
    # Replace these URLs with your actual Google Drive direct download links
    MODEL_URL = "https://drive.google.com/uc?id=1kBhg5vuGdPw9NKKLr9Y89PAWf_V8-avA&export=download"
    COLS_URL = "https://drive.google.com/uc?id=1U9ljX7rj7gYFywMqBNuKgHS9oDa0AQk5&export=download"
    
    try:
        # Check if files exist locally, if not download them
        if not os.path.exists("pollution_model.pkl"):
            st.info("📥 Downloading model file... This may take a moment for large files.")
            urllib.request.urlretrieve(MODEL_URL, "pollution_model.pkl")
            st.success("✅ Model file downloaded successfully!")
        
        if not os.path.exists("model_columns.pkl"):
            st.info("📥 Downloading model columns...")
            urllib.request.urlretrieve(COLS_URL, "model_columns.pkl")
            st.success("✅ Model columns downloaded successfully!")
        
        # Load the models
        model = joblib.load("pollution_model.pkl")
        model_cols = joblib.load("model_columns.pkl")
        st.success("🎯 Models loaded successfully!")
        return model, model_cols
        
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.info("Please check if model files are accessible at the Google Drive links")
        return None, None

model, model_cols = load_models()

# Header
st.markdown('<div class="main-header">🌊 Water Quality Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced AI-powered water pollutant level prediction system</div>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("🔧 Input Parameters")
    
    # Year input with better validation
    year_input = st.slider(
        "📅 Select Year",
        min_value=2000,
        max_value=2100,
        value=2022,
        help="Choose the year for prediction"
    )
    
    # Station ID input with suggestions
    st.subheader("🏭 Station Information")
    station_id = st.text_input(
        "Enter Station ID",
        value='1',
        help="Enter the monitoring station identifier"
    )
    
    # Additional info section
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ About This Tool</h4>
        <p>This predictor uses machine learning to estimate water pollutant levels based on historical data and station characteristics.</p>
        <p><strong>Predicted Pollutants:</strong></p>
        <ul>
            <li>O2 - Dissolved Oxygen</li>
            <li>NO3 - Nitrates</li>
            <li>NO2 - Nitrites</li>
            <li>SO4 - Sulfates</li>
            <li>PO4 - Phosphates</li>
            <li>CL - Chlorides</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction section
    st.header("🎯 Prediction Results")
    
    # Predict button
    if st.button('🔮 Generate Prediction', key="predict_btn"):
        if not station_id:
            st.warning('⚠️ Please enter a valid station ID')
        elif model is None or model_cols is None:
            st.error('❌ Model not loaded properly')
        else:
            with st.spinner('🔄 Analyzing water quality data...'):
                try:
                    # Prepare the input
                    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
                    input_encoded = pd.get_dummies(input_df, columns=['id'])
                    
                    # Align with model columns
                    for col in model_cols:
                        if col not in input_encoded.columns:
                            input_encoded[col] = 0
                    input_encoded = input_encoded[model_cols]
                    
                    # Make prediction
                    predicted_pollutants = model.predict(input_encoded)[0]
                    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
                    pollutant_names = ['Dissolved Oxygen', 'Nitrates', 'Nitrites', 'Sulfates', 'Phosphates', 'Chlorides']
                    
                    # Display results
                    st.success(f"✅ Prediction completed for Station {station_id} in {year_input}")
                    
                    # Create metrics display
                    st.subheader("📊 Predicted Pollutant Levels")
                    
                    # Display metrics in columns
                    cols = st.columns(3)
                    for i, (p, full_name, val) in enumerate(zip(pollutants, pollutant_names, predicted_pollutants)):
                        with cols[i % 3]:
                            st.metric(
                                label=f"{p} ({full_name})",
                                value=f"{val:.2f}",
                                help=f"Predicted level of {full_name}"
                            )
                    
                    # Create visualization
                    st.subheader("📈 Visual Analysis")
                    
                    # Bar chart
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=pollutants,
                            y=predicted_pollutants,
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                            text=[f'{val:.2f}' for val in predicted_pollutants],
                            textposition='auto',
                        )
                    ])
                    fig_bar.update_layout(
                        title=f"Predicted Pollutant Levels - Station {station_id} ({year_input})",
                        xaxis_title="Pollutants",
                        yaxis_title="Concentration Level",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Radar chart
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=predicted_pollutants,
                        theta=pollutants,
                        fill='toself',
                        name=f'Station {station_id}',
                        line_color='#1f77b4'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(predicted_pollutants) * 1.1]
                            )),
                        showlegend=True,
                        title=f"Pollutant Profile - Station {station_id} ({year_input})",
                        height=400
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please check your input parameters and try again.")

with col2:
    # Info panel
    st.header("📚 Information Panel")
    
    # Model info
    st.subheader("🤖 Model Information")
    if model is not None:
        st.info(f"Model Type: {type(model).__name__}")
        st.info(f"Features: {len(model_cols) if model_cols else 'N/A'}")
    
    # Quick stats
    st.subheader("📊 Quick Statistics")
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Prediction Accuracy</h4>
        <p>Our model has been trained on extensive historical water quality data to provide reliable predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips section
    st.subheader("💡 Tips for Better Predictions")
    st.markdown("""
    <div class="info-box">
        <ul>
            <li>Ensure the Station ID corresponds to a monitored location</li>
            <li>Recent years may provide more accurate predictions</li>
            <li>Consider seasonal variations in water quality</li>
            <li>Use multiple stations for comprehensive analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌊 Water Quality Predictor | Built with Streamlit and Machine Learning</p>
    <p>For environmental monitoring and water quality assessment</p>
</div>
""", unsafe_allow_html=True)