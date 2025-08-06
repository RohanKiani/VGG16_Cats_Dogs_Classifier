"""
Model utilities for loading and prediction
"""
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np
import os

# Class labels (update these to match your training labels)
CLASS_NAMES = ['Cat', 'Dog']  # Adjust order based on your training

import gdown

@st.cache_resource
def load_model():
    """
    Load the trained Vgg16 model from Google Drive if not already present.
    
    Returns:
        tensorflow.keras.Model: Loaded model
    """
    try:
        model_path = "vgg16_cat_dog_classifier.h5"
        model_url = "https://drive.google.com/uc?export=download&id=1YucE0YxT0iDXPGXK83NoL7c5ht4g9q1Q"
        
        # Download the model if it's not already present
        if not os.path.exists(model_path):
            with st.spinner("📥 Downloading model from Google Drive... Please wait."):
                gdown.download(model_url, model_path, quiet=False)
                st.success("✅ Model downloaded successfully!")
        
        # Load the model
        model = keras.models.load_model(model_path)

        # Display model info in console
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_image(model, processed_image):
    """
    Make prediction on preprocessed image
    
    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array
        
    Returns:
        dict: Prediction results with confidence scores
    """
    try:
        if model is None:
            return None
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get prediction probabilities
        if len(predictions[0]) == 1:
            # Binary classification with sigmoid
            prob = predictions[0][0]
            cat_prob = 1 - prob
            dog_prob = prob
        else:
            # Multi-class with softmax
            cat_prob = predictions[0][0]
            dog_prob = predictions[0][1]
        
        # Determine predicted class
        predicted_class = "Dog" if dog_prob > cat_prob else "Cat"
        confidence = max(cat_prob, dog_prob)
        
        # Format results
        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'cat_probability': float(cat_prob),
            'dog_probability': float(dog_prob),
            'all_probabilities': {
                'Cat': float(cat_prob),
                'Dog': float(dog_prob)
            }
        }
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None

def get_model_info(model):
    """
    Get information about the loaded model

    Args:
        model: Loaded Keras model

    Returns:
        dict: Model information
    """
    if model is None:
        return None

    try:
        # Total number of parameters
        total_params = model.count_params()

        # Trainable parameters calculated from variable shapes
        trainable_params = np.sum([np.prod(w.shape) for w in model.trainable_weights])

        return {
            'total_params': total_params,
            'trainable_params': int(trainable_params),
            'non_trainable_params': int(total_params - trainable_params),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'num_layers': len(model.layers)
        }

    except Exception as e:
        st.error(f"❌ Error getting model info: {str(e)}")
        return None
