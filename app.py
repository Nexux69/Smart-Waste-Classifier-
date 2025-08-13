import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import base64
import os

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

# --- Load Classifier Model ---
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

# --- Load Object Detector (MobileNet SSD) ---
@st.cache_resource
def load_detector():
    prototxt = "ssd/deploy.prototxt"
    model_path = "ssd/mobilenet_iter_73000.caffemodel"
    if not os.path.exists(prototxt) or not os.path.exists(model_path):
        st.error("SSD model files not found. Please add them in 'ssd/' folder.")
    net = cv2.dnn.readNetFromCaffe(prototxt, model_path)
    return net

detector = load_detector()

# --- Prediction Function (SSD + Classifier) ---
def detect_and_classify(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    detector.setInput(blob)
    detections = detector.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence_det = detections[0, 0, i, 2]
        if confidence_det > 0.5:  # object detected
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            obj_crop = image[startY:endY, startX:endX]
            if obj_crop.size == 0:
                continue

            img_resized = cv2.resize(obj_crop, (224, 224)) / 255.0
            pred = model.predict(np.expand_dims(img_resized, axis=0))
            classes = ['Biodegradable ♻️', 'Non-Biodegradable 🚯']
            label = classes[np.argmax(pred)]
            conf_cls = float(np.max(pred))

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf_cls*100:.1f}%", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            results.append((label, conf_cls))

    if not results:
        return image, [("No Object Found 🚫", 0.0)]
    return image, results

# --- App Interface ---
st.title("🌱 Smart Waste Classifier")
st.markdown("""
<div style='background-color:rgba(240, 248, 255, 0.7); padding:10px; border-radius:10px; margin-bottom:20px;'>
    <h3 style='color:#2e8b57;'>Upload an image or use your webcam to classify waste</h3>
</div>
""", unsafe_allow_html=True)

option = st.radio("", ("📁 Upload Image", "📷 Use Webcam"), horizontal=True)

# --- Upload Image Mode (Direct Classification) ---
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        if st.button('🔍 Analyze Waste', use_container_width=True):
            with st.spinner('Classifying...'):
                img_resized = cv2.resize(image, (224, 224)) / 255.0
                pred = model.predict(np.expand_dims(img_resized, axis=0))
                classes = ['Biodegradable ♻️', 'Non-Biodegradable 🚯']
                label = classes[np.argmax(pred)]
                conf_cls = float(np.max(pred))

                st.image(image, caption='Uploaded Image', use_container_width=True)
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3>{label}</h3>
                    <p>Confidence:</p>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width:{conf_cls * 100}%'></div>
                    </div>
                    <p>{conf_cls:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

# --- Webcam Mode (SSD + Classifier) ---
else:
    picture = st.camera_input("", label_visibility="collapsed")
    if picture:
        image = np.array(Image.open(picture))
        if st.button('🔍 Analyze Waste', use_container_width=True):
            with st.spinner('Detecting and Classifying...'):
                output_img, preds = detect_and_classify(image.copy())
                st.image(output_img, caption='Detection Result', use_container_width=True)
                for label, conf in preds:
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <h3>{label}</h3>
                        <p>Confidence:</p>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width:{conf * 100}%'></div>
                        </div>
                        <p>{conf:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: small;'>
    Made with ♻️ using Streamlit | Model Accuracy: 92.4%
</div>
""", unsafe_allow_html=True)
