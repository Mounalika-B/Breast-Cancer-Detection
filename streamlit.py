import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🔬 Breast Cancer Detection from Histopathology Image")
st.write("Upload a 50x50 histopathology image to detect breast cancer.")

model = load_model("breast_cancer_cnn.h5")

uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((50, 50))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 50, 50, 3)

    # Predict
    prediction = model.predict(img_array)[0]
    label = "Benign (No Cancer)" if np.argmax(prediction) == 0 else "Malignant (Cancer)"
    confidence = prediction[np.argmax(prediction)] * 100

    st.subheader("🔍 Prediction")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
