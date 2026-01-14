import os
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(page_title="Depth Estimation Tool", layout="wide")
st.title("Depth Estimation Tool")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "test-imgs")
CSV_PATH = os.path.join(IMAGE_FOLDER, "results_detections.csv")


def _safe_int(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def draw_bboxes(img: Image.Image, df: pd.DataFrame) -> Image.Image:
    """
    Draw bounding boxes from df on a copy of img.
    Expects bbox_left, bbox_top, bbox_right, bbox_bottom columns.
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    required = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
    if not all(c in df.columns for c in required):
        return annotated

    for _, row in df.iterrows():
        left = _safe_int(row.get("bbox_left"))
        top = _safe_int(row.get("bbox_top"))
        right = _safe_int(row.get("bbox_right"))
        bottom = _safe_int(row.get("bbox_bottom"))
        if None in (left, top, right, bottom):
            continue

        draw.rectangle([(left, top), (right, bottom)], width=3)

        label = str(row.get("label", "")).strip()
        conf = row.get("confidence", None)
        if conf is not None and conf != "":
            try:
                label = f"{label} ({float(conf):.2f})"
            except Exception:
                pass

        if label:
            draw.text((left, max(0, top - 14)), label, font=font)

    return annotated


# Load data
predictions = pd.read_csv(CSV_PATH)

# List images
images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

# UI
left, right = st.columns([1, 1])

with left:
    st.subheader("Select image")
    selected_image = st.selectbox("Image", images)
    show_boxes = st.toggle("Show bounding boxes", value=True)

    img_path = os.path.join(IMAGE_FOLDER, selected_image)
    img = Image.open(img_path).convert("RGB")

with right:
    st.subheader("Detections for selected image")

    if "relative_path" not in predictions.columns:
        st.error("CSV missing required column: relative_path")
        st.stop()

    df_img = predictions[predictions["relative_path"] == selected_image].copy()

    # Optional label filter
    label_filter = "All"
    if "label" in df_img.columns and len(df_img) > 0:
        labels = sorted(df_img["label"].astype(str).unique().tolist())
        label_filter = st.selectbox("Filter by label", ["All"] + labels)

    if label_filter != "All" and "label" in df_img.columns:
        df_show = df_img[df_img["label"].astype(str) == label_filter].copy()
    else:
        df_show = df_img

    st.dataframe(df_show, use_container_width=True)
    st.caption(f"{len(df_show)} detections shown (of {len(df_img)} total)")

st.divider()
st.subheader("Preview")

preview = img
if show_boxes and len(df_show) > 0:
    preview = draw_bboxes(img, df_show)

st.image(preview, caption=selected_image, use_container_width=True)