import streamlit as st
import torch
import cv2
import numpy as np
from model import UNet
from utils import preprocess_image, postprocess_mask

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")

# ===== Load Model =====
@st.cache_resource
def load_model():
    model = UNet()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ===== Title =====
st.title("🧠 Brain MRI Tumor Segmentation")
st.markdown("Upload an MRI image to detect tumor regions.")

# ===== Upload =====
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # preprocess
    input_tensor = preprocess_image(image)

    # predict
    with torch.no_grad():
        pred = model(input_tensor)

    mask = postprocess_mask(pred)

    # resize mask to original image size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # ===== Create 3 columns =====
    col1, col2, col3 = st.columns(3)

    # Original Image
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=300)

    # Ground Truth (optional placeholder)
    with col2:
        st.subheader("Mask (Predicted)")
        st.image(mask, width=300, clamp=True)

    # Prediction overlay
    with col3:
        st.subheader("Overlay")

        # create overlay
        overlay = image.copy()
        overlay[mask == 1] = [0, 255, 0]  # green tumor

        overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width=300)