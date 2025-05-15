
import os
import gdown
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# --- Step 1: Download model if not already present ---
file_id = "1AbCdEfGhIjKlMnOpQrS"  # 🔁 Replace with your actual file ID
url = f"https://drive.google.com/uc?id={file_id}"

model_path = "forgery_detection_model.h5"
if not os.path.exists(model_path):
    st.info("Downloading model...")
    gdown.download(url, model_path, quiet=False)

# --- Step 2: Load the model ---
model = load_model(model_path)
st.success("Model loaded successfully.")

# --- Step 3: Define prediction function ---
def preprocess_image(img):
    img = img.resize((128, 128))  # Match your model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)[0][0]
    label = "Tampered" if prediction >= 0.5 else "Authentic"
    return label, prediction

# --- Step 4: Streamlit UI ---
st.title("Image Forgery Detection")
st.write("Upload an image to check if it is tampered or authentic.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Check Image"):
        label, confidence = predict(image)
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence Score: {confidence:.2f}")
