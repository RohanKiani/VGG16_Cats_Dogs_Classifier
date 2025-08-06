"""
Image utilities for preprocessing and validation
"""
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Use EfficientNetB0  preprocessing from tf.keras
# Use VGG16 preprocessing from tf.keras
preprocess_input = tf.keras.applications.vgg16.preprocess_input


# Configuration constants
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp']

# ... (rest of the code remains unchanged)

def validate_image(uploaded_file):
    """
    Validate if the uploaded file is a valid image
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        bool: True if valid, False otherwise
    """
    if uploaded_file is None:
        return False
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        st.error(f"❌ Unsupported file format. Please upload: {', '.join(ALLOWED_EXTENSIONS)}")
        return False
    
    # Check file size (limit to 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ File too large. Please upload an image smaller than 10MB.")
        return False
    
    return True

def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image for model prediction using ImageNet preprocessing
    (matches the training preprocessing with VGG16 preprocess_input)
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (processed_image_array, original_pil_image)
    """
    try:
        # Open and convert image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (handles PNG with transparency)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Store original for display
        original_image = image.copy()
        
        # Resize image to model input size (224x224)
        image = image.resize(IMAGE_SIZE)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Add batch dimension first
        image_array = np.expand_dims(image_array, axis=0)
        
        # Apply VGG16 ImageNet preprocessing (same as training)
        # This applies: RGB->BGR conversion + mean subtraction 
        # [103.939, 116.779, 123.68] for BGR channels
        image_array = preprocess_input(image_array)
        
        return image_array, original_image
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def get_image_info(image):
    """
    Get basic information about the image
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: Image information
    """
    return {
        'format': image.format,
        'mode': image.mode,
        'size': image.size,
        'megapixels': round((image.size[0] * image.size[1]) / 1000000, 2)
    }