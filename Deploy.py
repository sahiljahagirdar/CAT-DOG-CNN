import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

IMAGE_SIZE = (256, 256)  
CLASS_NAMES = ['Cat', 'Dog']
# -----------------


def load_keras_model(model_path):
    """
    Loads the Keras model from the specified path.
    Uses st.cache_resource to load the model only once.
    """
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_file):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    """
    try:
        
        image = Image.open(image_file)
        
        
        image = image.convert('RGB')
        
        
        image = image.resize(IMAGE_SIZE, resample=Image.LANCZOS)
        
       
        img_array = np.asarray(image)
        
        
        img_array = img_array / 255.0
        
        
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None



st.title("🐱🐶 Cat vs. Dog Image Classifier")
st.write("Upload an image, and the model will predict if it's a cat or a dog.")


MODEL_FILE = "Cat_Dog_Model.keras"
MODEL_ID = "1KoZ8i3zIlMDiFme_TgHFYmPuM7blVE7e"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# Download model if not present
if not os.path.exists(MODEL_FILE):
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Load model
model = load_model(MODEL_FILE)

if model is not None:

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        st.write("")
        st.write("Classifying...")

        
        processed_image = preprocess_image(uploaded_file)

        if processed_image is not None:
            
            try:
                prediction = model.predict(processed_image)
                
            
                score = prediction[0][0]
                
            
                if score > 0.5:
                    label = CLASS_NAMES[1]  
                    confidence = score
                else:
                    label = CLASS_NAMES[0]  
                    confidence = 1 - score

                
                st.success(f"**Prediction: {label}**")
            

            except Exception as e:
                st.error(f"Error during prediction: {e}")
else:
    st.error("Model file 'Cat_Dog_Model.keras' not found or failed to load.")