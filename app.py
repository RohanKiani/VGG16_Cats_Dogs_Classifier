"""
Cat vs Dog Classifier - Main Streamlit Application
A professional deep learning web app using EfficientNetB0 for binary image classification
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom components and utilities
from components.sidebar import render_sidebar
from components.ui_components import (
    render_header, 
    render_file_uploader, 
    render_image_display,
    render_prediction_button,
    render_prediction_results,
    render_loading_spinner
)
from utils.model_utils import load_model, predict_image

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .uploadedFile {
        border: 2px dashed #FF6B6B;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Load model on first run
    if not st.session_state.model_loaded:
        with st.spinner('🤖 Loading AI model... This may take a moment on first run.'):
            st.session_state.model = load_model()
            st.session_state.model_loaded = True
            
            if st.session_state.model is not None:
                st.success('✅ Model loaded successfully!')
            else:
                st.error('❌ Failed to load model. Please check if the model file exists.')
                st.stop()
    
    # Render sidebar and get settings
    settings = render_sidebar(st.session_state.model)
    
    # Main content area
    render_header()
    
    # Check if model is loaded before proceeding
    if st.session_state.model is None:
        st.error("❌ Model not available. Please refresh the page.")
        st.stop()
    
    # File upload section
    uploaded_file = render_file_uploader()
    
    if uploaded_file is not None:
        # Display uploaded image
        processed_image, original_image = render_image_display(uploaded_file, settings)
        
        if processed_image is not None:
            # Prediction button
            predict_clicked = render_prediction_button()
            
            if predict_clicked:
                # Show loading spinner and make prediction
                with st.spinner('🔮 Analyzing image... Please wait!'):
                    results = predict_image(st.session_state.model, processed_image)
                
                # Display results
                if results:
                    render_prediction_results(results, settings)
                    
                    # Add some fun interactions
                if results['confidence'] >= settings['confidence_threshold']:
                    st.balloons()

                else:
                    st.error("❌ Failed to make prediction. Please try again.")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>👆 Upload an image to get started!</h3>
            <p>Select a clear photo of a cat or dog using the file uploader above.</p>
            <p><strong>Supported formats:</strong> JPG, JPEG, PNG, BMP</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show some example instructions
        st.markdown("---")
        st.markdown("### 🎯 How it works:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Step 1: Upload** 📁  
            Choose a clear image of a cat or dog from your device.
            """)
        
        with col2:
            st.markdown("""
            **Step 2: Predict** 🔮  
            Click the predict button to analyze the image with AI.
            """)
        
        with col3:
            st.markdown("""
            **Step 3: Results** 🎉  
            View the prediction and confidence score instantly.
            """)

# Footer
def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 Powered by VGG16 Deep Learning Model | Built with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
        render_footer()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please refresh the page or contact support if the problem persists.")
        
        # Debug information (only show in development)
        with st.expander("🔧 Debug Information"):
            st.exception(e)