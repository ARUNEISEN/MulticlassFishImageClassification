import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


# Load Model and Labels

MODEL_PATH = r"D:\Projects\MulticlassFishImageClassification\Models\MobileNetV2_model.h5"

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image and the model will predict its species.")

@st.cache_resource
def load_fish_model():
    model = load_model(MODEL_PATH)
    return model

model = load_fish_model()

label_dict = {
 'animal fish': 0,
 'animal fish bass': 1,
 'fish sea_food black_sea_sprat': 2,
 'fish sea_food gilt_head_bream': 3,
 'fish sea_food hourse_mackerel': 4,
 'fish sea_food red_mullet': 5,
 'fish sea_food red_sea_bream': 6,
 'fish sea_food sea_bass': 7,
 'fish sea_food shrimp': 8,
 'fish sea_food striped_red_mullet': 9,
 'fish sea_food trout': 10
}

# Reverse mapping for predictions
labels = {v: k for k, v in label_dict.items()}


# Process & Predict Function

def predict_fish(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return labels[class_idx], confidence, prediction[0]



# Streamlit UI

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            class_name, confidence, all_scores = predict_fish(img)

        st.success(f"### 🐠 Predicted Category: **{class_name}**")
        st.write(f"### 🔥 Confidence: **{confidence * 100:.2f}%**")

        # Show confidence scores for all classes
        st.subheader("Confidence Scores for All Classes")
        for idx, score in enumerate(all_scores):
            st.write(f"**{labels[idx]}:** {score * 100:.2f}%")
