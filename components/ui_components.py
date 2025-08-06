"""
Enhanced UI components for the Cat-Dog Classifier app with modern design
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time
from utils.image_utils import validate_image, preprocess_image, get_image_info
from utils.model_utils import predict_image

def render_header():
    """Render the main app header with modern design and sharp emojis"""
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 40px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(31, 38, 135, 0.2);
    ">
        <div style="
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h1 style="
                font-size: 3.5em; 
                margin-bottom: 15px;
                font-weight: 800;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            ">
                <span style="font-size: 1.2em; color: #FF6B6B;">🐱</span> 
                <span style="
                    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">AI Pet Classifier</span> 
                <span style="font-size: 1.2em; color: #4ECDC4;">🐶</span>
            </h1>
            <p style="
                font-size: 1.3em; 
                color: rgba(255, 255, 255, 0.9); 
                margin-bottom: 10px;
                font-weight: 300;
            ">
                Advanced Deep Learning for Pet Recognition
            </p>
            <p style="
                font-size: 1em; 
                color: rgba(255, 255, 255, 0.7); 
                margin: 0;
                font-style: italic;
            ">
                Upload an image and let our VGG16 AI model identify cats vs dogs with precision
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_file_uploader():
    """
    Render enhanced file uploader component with modern styling
    
    Returns:
        uploaded_file: Streamlit uploaded file object or None
    """
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2px;
        border-radius: 15px;
        margin: 20px 0;
    ">
        <div style="
            background: white;
            padding: 25px;
            border-radius: 13px;
            text-align: center;
        ">
            <h3 style="
                color: #333;
                margin-bottom: 15px;
                font-size: 1.4em;
            ">📁 Upload Your Pet Image</h3>
            <p style="
                color: #666;
                margin-bottom: 20px;
                font-size: 1em;
            ">Drag & drop or click to browse • Supports JPG, PNG, BMP formats</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="📸 For best results: Use clear, well-lit images with the pet as the main subject",
        label_visibility="collapsed"
    )
    
    # Upload tips in an expandable section
    if not uploaded_file:
        with st.expander("💡 Photo Upload Tips", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **✅ Good Photos:**
                - Single cat or dog clearly visible
                - Good lighting and focus
                - Minimal background distractions
                - Standard camera angles
                """)
            
            with col2:
                st.markdown("""
                **❌ Avoid:**
                - Multiple animals in one photo
                - Very dark or blurry images
                - Heavy filters or effects
                - Extreme close-ups of just paws/tails
                """)
    
    return uploaded_file

def render_image_display(uploaded_file, settings):
    """
    Display the uploaded image with enhanced styling and information
    
    Args:
        uploaded_file: Streamlit uploaded file object
        settings: Settings dictionary from sidebar
    
    Returns:
        tuple: (processed_image, original_image) or (None, None)
    """
    if not validate_image(uploaded_file):
        st.error("❌ Invalid image file. Please upload a valid JPG, PNG, or BMP image.")
        return None, None
    
    # Show processing steps if enabled
    if settings['show_processing_steps']:
        render_processing_steps()
    
    # Process the image with progress indication
    with st.spinner('🔄 Processing image...'):
        processed_image, original_image = preprocess_image(uploaded_file)
    
    if processed_image is None:
        st.error("❌ Failed to process the image. Please try a different file.")
        return None, None
    
    # Success message
    st.success("✅ Image processed successfully!")
    
    # Display the image in a modern container
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3px;
        border-radius: 20px;
        margin: 20px 0;
    ">
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        # Image container with glassmorphism effect
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
            margin: 10px 0;
        ">
        """, unsafe_allow_html=True)
        
        st.image(
            original_image, 
            caption=f"📷 {uploaded_file.name}",
            use_container_width=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show image information if enabled
    if settings['show_image_info']:
        render_image_info(original_image, uploaded_file.name)
    
    return processed_image, original_image

def render_processing_steps():
    """Display the image processing pipeline steps"""
    st.markdown("#### 🔄 Processing Pipeline")
    
    steps = [
        {"step": "1", "name": "Validation", "desc": "Checking file format and integrity", "icon": "🔍"},
        {"step": "2", "name": "Loading", "desc": "Reading image data into memory", "icon": "📁"},
        {"step": "3", "name": "Resizing", "desc": "Scaling to model input size (224x224)", "icon": "📐"},
        {"step": "4", "name": "Normalization", "desc": "Applying ImageNet preprocessing", "icon": "⚡"},
        {"step": "5", "name": "Ready", "desc": "Prepared for model inference", "icon": "✅"}
    ]
    
    cols = st.columns(5)
    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 15px 10px;
                border-radius: 12px;
                text-align: center;
                margin: 5px 0;
                font-size: 0.85em;
            ">
                <div style="font-size: 1.5em; margin-bottom: 5px;">{step['icon']}</div>
                <div style="font-weight: bold; margin-bottom: 3px;">{step['name']}</div>
                <div style="opacity: 0.9; font-size: 0.8em;">{step['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

def render_image_info(image, filename):
    """Display detailed image information in a modern card"""
    image_info = get_image_info(image)
    
    st.markdown("#### 📊 Image Analysis")
    
    # Create modern info cards
    col1, col2, col3, col4 = st.columns(4)
    
    cards = [
        {"title": "Format", "value": image_info['format'], "icon": "🖼️", "color": "#FF6B6B"},
        {"title": "Dimensions", "value": f"{image_info['size'][0]}×{image_info['size'][1]}", "icon": "📏", "color": "#4ECDC4"},
        {"title": "Megapixels", "value": f"{image_info['megapixels']} MP", "icon": "🔍", "color": "#45B7D1"},
        {"title": "Color Mode", "value": image_info['mode'], "icon": "🎨", "color": "#96CEB4"}
    ]
    
    for i, card in enumerate(cards):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {card['color']}20, {card['color']}10);
                border: 1px solid {card['color']};
                border-radius: 12px;
                padding: 20px 15px;
                text-align: center;
                margin: 5px 0;
            ">
                <div style="font-size: 1.8em; margin-bottom: 8px;">{card['icon']}</div>
                <div style="font-weight: bold; color: {card['color']}; margin-bottom: 5px;">
                    {card['title']}
                </div>
                <div style="color: #333; font-size: 1.1em; font-weight: 600;">
                    {card['value']}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_prediction_button():
    """
    Render the enhanced prediction button with modern styling
    
    Returns:
        bool: True if button was clicked
    """
    st.markdown("#### 🔮 AI Analysis")
    
    # Create an eye-catching button container
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3px;
        border-radius: 15px;
        margin: 20px 0;
    ">
        <div style="
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h4 style="color: white; margin-bottom: 15px; font-size: 1.3em;">
                🚀 Ready to Analyze Your Pet Photo?
            </h4>
            <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 20px;">
                Our advanced VGG16 neural network will process your image and provide detailed predictions
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button(
            "🎯 Analyze Image with AI",
            type="primary",
            use_container_width=True,
            help="Click to classify the uploaded image using our trained deep learning model"
        )
    
    return predict_clicked

def render_prediction_results(results, settings):
    """
    Display prediction results with enhanced visualizations and modern styling
    
    Args:
        results: Dictionary containing prediction results
        settings: Settings dictionary from sidebar
    """
    if results is None:
        st.error("❌ Could not generate prediction. Please try again with a different image.")
        return
    
    # Update session statistics
    st.session_state.predictions_made += 1
    if results['predicted_class'] == 'Cat':
        st.session_state.cats_detected += 1
    else:
        st.session_state.dogs_detected += 1
    
    # Main prediction result with enhanced styling
    st.markdown("### 🎯 AI Prediction Results")
    
    predicted_class = results['predicted_class']
    confidence = results['confidence']
    
    # Choose styling based on prediction
    if predicted_class == 'Cat':
        emoji = "🐱"
        color = "#FF6B6B"
        gradient = "linear-gradient(135deg, #FF6B6B, #FF8E8E)"
        animal_fact = "Cats have excellent night vision and can rotate their ears 180 degrees!"
    else:
        emoji = "🐶" 
        color = "#4ECDC4"
        gradient = "linear-gradient(135deg, #4ECDC4, #44B3D9)"
        animal_fact = "Dogs have an incredible sense of smell - 40x better than humans!"
    
    # Enhanced main result display
    if confidence >= settings['confidence_threshold']:
        st.markdown(f"""
        <div style="
            background: {gradient};
            padding: 40px 30px;
            border-radius: 20px;
            text-align: center;
            margin: 25px 0;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            border: 3px solid white;
        ">
            <div style="
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            ">
                <div style="font-size: 4em; margin-bottom: 15px;">{emoji}</div>
                <h2 style="color: white; margin: 15px 0; font-size: 2.2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    It's a {predicted_class}!
                </h2>
                <h3 style="color: rgba(255,255,255,0.9); margin: 15px 0; font-size: 1.5em;">
                    Confidence: {confidence:.1%}
                </h3>
                <p style="color: rgba(255,255,255,0.8); font-size: 1em; margin-top: 20px; font-style: italic;">
                    💡 Fun Fact: {animal_fact}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence level indicator
        st.success(f"🎯 High confidence prediction! The model is {confidence:.1%} certain.")
        
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FFA500, #FF8C00);
            padding: 40px 30px;
            border-radius: 20px;
            text-align: center;
            margin: 25px 0;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            border: 3px solid white;
        ">
            <div style="
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            ">
                <div style="font-size: 4em; margin-bottom: 15px;">🤔</div>
                <h2 style="color: white; margin: 15px 0; font-size: 2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    Uncertain Prediction
                </h2>
                <h3 style="color: rgba(255,255,255,0.9); margin: 15px 0; font-size: 1.3em;">
                    Best Guess: {predicted_class} ({confidence:.1%})
                </h3>
                <p style="color: rgba(255,255,255,0.8); font-size: 1em; margin-top: 20px;">
                    ⚠️ Confidence below threshold ({settings['confidence_threshold']:.0%})
                </p>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin-top: 10px;">
                    Try uploading a clearer image for better accuracy
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning(f"⚠️ Low confidence prediction. Consider uploading a clearer image.")
    
    # Enhanced probability visualization
    if settings['show_probabilities']:
        render_probability_section(results)

def render_probability_section(results):
    """Render enhanced probability analysis section"""
    st.markdown("#### 📊 Detailed Probability Analysis")
    
    # Create enhanced probability metrics
    col1, col2 = st.columns(2)
    
    cat_prob = results['cat_probability']
    dog_prob = results['dog_probability']
    
    with col1:
        # Cat probability card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF6B6B20, #FF6B6B10);
            border: 2px solid #FF6B6B;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            margin: 10px 0;
        ">
            <div style="font-size: 2.5em; margin-bottom: 10px;">🐱</div>
            <h4 style="color: #FF6B6B; margin-bottom: 10px;">Cat Probability</h4>
            <div style="font-size: 2em; font-weight: bold; color: #333; margin-bottom: 10px;">
                {cat_prob:.1%}
            </div>
            <div style="
                background: #FF6B6B;
                height: 8px;
                border-radius: 4px;
                width: {cat_prob*100}%;
                margin: 0 auto;
            "></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Dog probability card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4ECDC420, #4ECDC410);
            border: 2px solid #4ECDC4;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            margin: 10px 0;
        ">
            <div style="font-size: 2.5em; margin-bottom: 10px;">🐶</div>
            <h4 style="color: #4ECDC4; margin-bottom: 10px;">Dog Probability</h4>
            <div style="font-size: 2em; font-weight: bold; color: #333; margin-bottom: 10px;">
                {dog_prob:.1%}
            </div>
            <div style="
                background: #4ECDC4;
                height: 8px;
                border-radius: 4px;
                width: {dog_prob*100}%;
                margin: 0 auto;
            "></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced probability chart
    create_enhanced_probability_chart(results['all_probabilities'])
    
    # Confidence interpretation
    render_confidence_interpretation(results, cat_prob, dog_prob)

def create_enhanced_probability_chart(probabilities):
    """
    Create an enhanced visual chart showing class probabilities
    
    Args:
        probabilities: Dictionary with class probabilities
    """
    # Create modern bar chart with custom styling
    fig = go.Figure()
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig.add_trace(go.Bar(
        x=classes,
        y=probs,
        text=[f"{prob:.1%}" for prob in probs],
        textposition='auto',
        textfont=dict(size=16, color='white', family='Arial Black'),
        marker=dict(
            color=colors,
            line=dict(color='white', width=2),
            pattern_shape=['', '']
        ),
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>',
        name='Probability'
    ))
    
    fig.update_layout(
        title=dict(
            text="🎯 Class Probability Distribution",
            font=dict(size=20, family='Arial', color='#333'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text="Animal Classes",
                font=dict(size=14, family='Arial', color='#666')
            ),
            tickfont=dict(size=12, family='Arial', color='#333')
        ),
        yaxis=dict(
            title=dict(
                text="Probability", 
                font=dict(size=14, family='Arial', color='#666')
            ),
            tickfont=dict(size=12, family='Arial', color='#333'),
            range=[0, 1],
            tickformat='.0%'
        ),
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)

def render_confidence_interpretation(results, cat_prob, dog_prob):
    """Render confidence level interpretation"""
    st.markdown("#### 🧠 AI Confidence Analysis")
    
    confidence = results['confidence']
    predicted_class = results['predicted_class']
    
    # Determine confidence level
    if confidence >= 0.9:
        level = "Extremely High"
        color = "#4CAF50"
        icon = "🎯"
        description = "The model is very certain about this prediction."
    elif confidence >= 0.8:
        level = "High" 
        color = "#8BC34A"
        icon = "✅"
        description = "Strong confidence in the prediction result."
    elif confidence >= 0.7:
        level = "Moderate"
        color = "#FFC107"
        icon = "⚠️"
        description = "Reasonable confidence, but some uncertainty remains."
    elif confidence >= 0.6:
        level = "Low"
        color = "#FF9800"
        icon = "🤔"
        description = "Limited confidence - consider image quality."
    else:
        level = "Very Low"
        color = "#F44336"
        icon = "❓"
        description = "High uncertainty - image may be unclear or ambiguous."
    
    st.markdown(f"""
    <div style="
        background: {color}20;
        border: 2px solid {color};
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.5em; margin-right: 10px;">{icon}</span>
            <span style="font-size: 1.2em; font-weight: bold; color: {color};">
                {level} Confidence ({confidence:.1%})
            </span>
        </div>
        <p style="color: #333; margin: 0; font-size: 1em;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_loading_spinner():
    """Render enhanced loading spinner during prediction"""
    loading_messages = [
        "🔮 Initializing neural network...",
        "📸 Preprocessing image data...", 
        "🧠 Running AI analysis...",
        "📊 Calculating confidence scores...",
        "✨ Finalizing results..."
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, message in enumerate(loading_messages):
        status_text.text(message)
        progress_bar.progress((i + 1) / len(loading_messages))
        time.sleep(0.3)
    
    status_text.text("🎉 Analysis complete!")
    progress_bar.empty()
    status_text.empty()