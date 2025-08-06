# 🐶🐱 VGG16 Cats vs Dogs Classifier

A deep learning-based image classification app that distinguishes between cats and dogs using the **VGG16** pre-trained convolutional neural network. This web app is built with **Streamlit** and deployed for easy interaction through your browser.

## 📌 Project Overview

This project leverages **transfer learning** with the VGG16 architecture to classify input images as either a **Cat** or a **Dog**. Users can upload an image, and the model provides a prediction along with a confidence score.

## 🚀 Features

- ✅ Upload and classify your own cat or dog images
- ✅ Modern UI built with Streamlit
- ✅ Image preview and prediction result display
- ✅ Confidence score of prediction
- ✅ Lightweight, fast, and easy to use

## 🧠 Model Details

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Architecture**: VGG16 with custom top layers for binary classification
- **Output Layer**: 1 neuron with **sigmoid** activation
- **Classes**: 
  - `Cat` → 0  
  - `Dog` → 1

## 📁 Folder Structure

VGG16_Cats_Dogs_Classifier/
├── app.py # Main Streamlit app
├── vgg16_cat_dog_classifier.h5 # Trained Keras model
├── requirements.txt # Required Python packages
├── components/ # UI components
│ ├── init.py
│ ├── ui_components.py
│ └── sidebar.py
├── utils/ # Utility scripts
│ ├── init.py
│ ├── model_utils.py
│ └── image_utils.py


## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RohanKiani/VGG16_Cats_Dogs_Classifier.git
   cd VGG16_Cats_Dogs_Classifier
Install dependencies:

  pip install -r requirements.txt

Run the Streamlit app:

  streamlit run app.py
  
🧪 Testing the App
Upload .jpg, .jpeg, or .png images of cats or dogs and the app will:

Display the image

Predict the label (Cat or Dog)

Show the confidence score

🌐 Deployment
This app can be deployed to Streamlit Cloud by uploading this GitHub repository.

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

🙌 Acknowledgements
VGG16 - Keras Applications
ImageNet
Streamlit
