import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os

# Page Configuration
st.set_page_config(
    page_title="Fish Classifier",
    page_icon="🐟",
    layout="centered"
)
st.markdown("""
<style>
    .stApp {
        background-color: #eef2f7;   /* Soft gray-blue */
    }

    h1 {
        color: #1a237e;
        text-align: center;
    }

    h2, h3 {
        color: #283593;
    }

    div.stButton > button {
        background-color: #3949ab;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1a237e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


st.title("🐟 Fish Image Classification")
st.write("Upload a fish image and the model will predict the category.")

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    model_path = r"D:\Projects\MulticlassFishImageClassification\Notebooks\models\MobileNet_final.h5"
    
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None
    
    model = keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# ------------------------------
# Load Class Names
# ------------------------------
def load_class_names():
    metadata_path = "models/model_metadata.json"
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata["class_names"]
    
    return [
        'animal fish',
        'animal fish bass',
        'fish sea_food black_sea_sprat',
        'fish sea_food gilt_head_bream',
        'fish sea_food hourse_mackerel',
        'fish sea_food red_mullet',
        'fish sea_food red_sea_bream',
        'fish sea_food sea_bass',
        'fish sea_food shrimp',
        'fish sea_food striped_red_mullet',
        'fish sea_food trout'
    ]

class_names = load_class_names()

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------------------
# Prediction Function
# ------------------------------
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[0][predicted_index])
    
    return predicted_index, confidence, predictions[0]

# ------------------------------
# Upload Section
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:    
    image = Image.open(uploaded_file).convert("RGB")    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    with col2:
        if st.button("Predict"):
            
            with st.spinner("Predicting..."):
                index, confidence, all_probs = predict(image)

            st.success("Prediction Complete!")

            st.subheader("Prediction Result")
            st.write("**Predicted Class:**", class_names[index])
            st.write("**Confidence:**", f"{confidence:.2%}")

            st.markdown("---")
            st.subheader("Top 5 Probabilities")

            top_indices = np.argsort(all_probs)[-5:][::-1]

            for i in top_indices:
                st.write(f"{class_names[i]} : {all_probs[i]:.2%}")
                st.progress(float(all_probs[i]))

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.write("Built with TensorFlow, Keras and Streamlit")
