import os
from PIL import Image

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Depth Estimation Tool", layout="wide")
st.title("Depth Estimation Tool")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "test-imgs")
CSV_PATH = os.path.join(IMAGE_FOLDER, "results_detections.csv")

# Load data
predictions = pd.read_csv(CSV_PATH)
images = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

# Layout
left, right = st.columns([1, 1])

with left:
    st.subheader("Calibration image (representative)")
    selected_image = st.selectbox("Choose an image", images)

    img_path = os.path.join(IMAGE_FOLDER, selected_image)
    img = Image.open(img_path)
    st.image(img, caption=selected_image, use_container_width=True)

with right:
    st.subheader("Detections for selected image")

    if "relative_path" in predictions.columns:
        df = predictions[predictions["relative_path"] == selected_image].copy()
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(df)} detections for {selected_image}")
    else:
        st.warning("CSV missing required column: relative_path. Showing full table.")
        st.dataframe(predictions, use_container_width=True)
