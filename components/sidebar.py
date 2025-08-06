"""
Enhanced sidebar component with modern design for the Cat-Dog Classifier app
"""
import streamlit as st
import plotly.graph_objects as go
from utils.model_utils import get_model_info

def render_sidebar(model):
    """
    Render an enhanced, modern sidebar with interactive elements
    
    Args:
        model: Loaded Keras model
    """
    with st.sidebar:
        # Modern Header with gradient background
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        ">
            <h1 style="color: white; margin: 0; font-size: 1.8em;">🐱🐶</h1>
            <h2 style="color: white; margin: 5px 0; font-size: 1.2em;">AI Classifier</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9em;">
                Powered by Deep Learning
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme Toggle
        st.markdown("### 🎨 Appearance")
        theme_choice = st.selectbox(
            "Choose Theme",
            ["🌙 Dark", "☀️ Light", "🌈 Colorful"],
            index=0
        )
        
        st.markdown("---")
        
        # Interactive App Guide
        st.markdown("### 📋 Quick Guide")
        
        steps = [
            {"icon": "📁", "title": "Upload", "desc": "Select a clear cat/dog image"},
            {"icon": "🔮", "title": "Predict", "desc": "Click to analyze with AI"},
            {"icon": "📊", "title": "Results", "desc": "View predictions & confidence"}
        ]
        
        for i, step in enumerate(steps, 1):
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
                padding: 12px;
                border-radius: 10px;
                margin: 8px 0;
                color: white;
            ">
                <strong>{step['icon']} Step {i}: {step['title']}</strong><br>
                <small style="opacity: 0.9;">{step['desc']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Performance Metrics (Interactive)
        st.markdown("### 🤖 Model Dashboard")
        
        if model is not None:
            model_info = get_model_info(model)
            if model_info:
                # Create mini performance dashboard
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Parameters", 
                        f"{model_info['total_params']/1e6:.1f}M",
                        help="Total model parameters"
                    )
                
                with col2:
                    st.metric(
                        "Layers", 
                        model_info['num_layers'],
                        help="Number of neural network layers"
                    )
                
                # Model architecture visualization
                trainable_pct = (model_info['trainable_params'] / model_info['total_params']) * 100
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Trainable', 'Frozen'],
                        values=[model_info['trainable_params'], model_info['non_trainable_params']],
                        hole=0.4,
                        marker_colors=['#ff6b6b', '#4ecdc4']
                    )
                ])
                
                fig.update_layout(
                    title="Parameter Distribution",
                    height=200,
                    showlegend=True,
                    margin=dict(t=40, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model specs in expandable section
                with st.expander("🔧 Technical Specs"):
                    st.markdown(f"""
                    **Architecture:** VGG16 + Custom Head  
                    **Input Shape:** {model_info['input_shape']}  
                    **Output Shape:** {model_info['output_shape']}  
                    **Classification:** Binary (Sigmoid)  
                    **Preprocessing:** ImageNet Normalization
                    """)
        else:
            st.error("❌ Model not loaded")
            st.markdown("Please check your model file!")
        
        st.markdown("---")
        
        # Enhanced Settings with better UX
        st.markdown("### ⚙️ Prediction Settings")
        
        # Confidence threshold with visual indicator
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Minimum confidence for reliable predictions"
        )
        
        # Visual confidence indicator
        if confidence_threshold >= 0.8:
            conf_color = "#4CAF50"
            conf_text = "High Precision"
        elif confidence_threshold >= 0.7:
            conf_color = "#FF9800"
            conf_text = "Balanced"
        else:
            conf_color = "#F44336"
            conf_text = "High Recall"
            
        st.markdown(f"""
        <div style="
            background: {conf_color}20;
            border: 1px solid {conf_color};
            border-radius: 8px;
            padding: 8px;
            text-align: center;
            margin: 10px 0;
        ">
            <small style="color: {conf_color}; font-weight: bold;">
                {conf_text} Mode
            </small>
        </div>
        """, unsafe_allow_html=True)
        
        # Display options with icons
        st.markdown("#### 📊 Display Options")
        
        show_probabilities = st.toggle(
            "📈 Show Probability Charts",
            value=True,
            help="Display interactive probability visualizations"
        )
        
        show_image_info = st.toggle(
            "📷 Show Image Details",
            value=False,
            help="Display technical image information"
        )
        
        show_processing_steps = st.toggle(
            "🔄 Show Processing Steps",
            value=True,
            help="Display preprocessing pipeline steps"
        )
        
        st.markdown("---")
        
        # Performance Tips with expandable sections
        st.markdown("### 💡 Pro Tips")
        
        with st.expander("📸 Best Photo Practices"):
            st.markdown("""
            • **Clear subject focus** - Cat/dog should be main subject
            • **Good lighting** - Avoid shadows and darkness  
            • **Minimal background** - Less distracting elements
            • **High resolution** - At least 300x300 pixels
            • **Natural poses** - Avoid heavy filters/effects
            """)
        
        with st.expander("🎯 Accuracy Tips"):
            st.markdown("""
            • **Single animal** - One cat or dog per image
            • **Full body or clear face** - More features = better accuracy
            • **Standard breeds** - Common breeds work best
            • **Adult animals** - Puppies/kittens can be trickier
            """)
        
        with st.expander("📊 Understanding Results"):
            st.markdown(f"""
            • **Confidence > {confidence_threshold:.0%}** - Reliable prediction
            • **50-{confidence_threshold:.0%}** - Uncertain, review image
            • **< 50%** - Poor quality or unusual image
            • **Chart colors** - 🐱 Red=Cat, 🐶 Teal=Dog
            """)
        
        st.markdown("---")
        
        # Fun stats section
        st.markdown("### 📈 Session Stats")
        
        # Initialize session state for stats
        if 'predictions_made' not in st.session_state:
            st.session_state.predictions_made = 0
        if 'cats_detected' not in st.session_state:
            st.session_state.cats_detected = 0
        if 'dogs_detected' not in st.session_state:
            st.session_state.dogs_detected = 0
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.metric("🔮 Predictions", st.session_state.predictions_made)
            st.metric("🐱 Cats Found", st.session_state.cats_detected)
            
        with stats_col2:
            st.metric("🐶 Dogs Found", st.session_state.dogs_detected)
            if st.session_state.predictions_made > 0:
                accuracy_display = "Analyzing..."
            else:
                accuracy_display = "No data"
            st.metric("📊 Session", accuracy_display)
        
        # Footer with version
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8em;">
            <p>🚀 v2.0 Enhanced UI<br>
            Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Return enhanced settings
        return {
            'confidence_threshold': confidence_threshold,
            'show_probabilities': show_probabilities,
            'show_image_info': show_image_info,
            'show_processing_steps': show_processing_steps,
            'theme': theme_choice
        }