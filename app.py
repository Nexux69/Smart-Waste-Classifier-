import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import base64


# --- Background Setup ---
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Set your background image
set_background('assets/background.png')

# --- Custom CSS ---
st.markdown("""
<style>
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
.st-emotion-cache-1kyxreq {
    animation: fadeIn 0.5s ease-in;
}
.prediction-box {
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    background: linear-gradient(145deg, rgba(240, 242, 246, 0.9), rgba(255, 255, 255, 0.95));
    box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
}
.prediction-box h3 {
    color: #2e8b57 !important;
}
.prediction-box p {
    color: #333 !important;
}
.confidence-bar {
    height: 20px;
    border-radius: 10px;
    margin: 10px 0;
    background: #e0e0e0;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    transition: width 0.5s ease-out;
}
.stButton>button {
    background-color: #2e8b57 !important;
    color: white !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_my_model():
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        status_text.text(f"Loading model... {i + 1}%")

    model = load_model('model.h5')
    progress_bar.empty()
    status_text.empty()
    return model


model = load_my_model()


# --- Prediction Function ---
def predict(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))
    confidence = float(np.max(pred))
    classes = ['Biodegradable ♻️', 'Non-Biodegradable 🚯']

    # No object detection logic
    if confidence < 0.7:  # threshold
        return "No Object Found 🚫", confidence
    else:
        return classes[np.argmax(pred)], confidence


# --- App Interface ---
st.title("🌱 Smart Waste Classifier")
st.markdown("""
<div style='background-color:rgba(240, 248, 255, 0.7); padding:10px; border-radius:10px; margin-bottom:20px;'>
    <h3 style='color:#2e8b57;'>Upload an image or use your webcam to classify waste</h3>
</div>
""", unsafe_allow_html=True)

# Input Selection
option = st.radio("", ("📁 Upload Image", "📷 Use Webcam"), horizontal=True)

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        with st.spinner('Processing image...'):
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption='Uploaded Image', use_container_width=True)

            if st.button('🔍 Analyze Waste', use_container_width=True):
                with st.spinner('Classifying...'):
                    time.sleep(0.5)
                    label, confidence = predict(image)

                    st.markdown(f"""
                    <div class='prediction-box'>
                        <h3 style='color:#2e8b57;'> {label}</h3>
                        <p style='color:#333;'>Confidence:</p>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width:{confidence * 100}%'></div>
                        </div>
                        <p style='color:#333;'>{confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

else:  # Webcam option
    st.markdown("### Smile for the planet! 🌍")
    picture = st.camera_input("", label_visibility="collapsed")

    if picture:
        with st.spinner('Processing capture...'):
            image = np.array(Image.open(picture))

            if st.button('🔍 Analyze Waste', use_container_width=True):
                with st.spinner('Classifying...'):
                    time.sleep(0.5)
                    label, confidence = predict(image)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption='Captured Image', use_container_width=True)
                    with col2:
                        st.markdown(f"""
                        <div class='prediction-box'>
                            <h3 style='color:#2e8b57;'>Prediction: {label}</h3>
                            <p style='color:#333;'>Confidence:</p>
                            <div class='confidence-bar'>
                                <div class='confidence-fill' style='width:{confidence * 100}%'></div>
                            </div>
                            <p style='color:#333;'>{confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: small;'>
    Made with ♻️ using Streamlit | Model Accuracy: 92.4%
</div>
""", unsafe_allow_html=True)
