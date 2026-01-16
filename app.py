import os
from typing import Optional, Tuple

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


def compute_sample_point(left: int, top: int, right: int, bottom: int, mode: str) -> Tuple[int, int]:
    """Return a representative pixel location for depth sampling within a bbox."""
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    if mode == "bottom-center":
        return (cx, bottom)
    return (cx, cy)


def draw_overlays(
    img: Image.Image,
    df: pd.DataFrame,
    point_mode: str,
    show_boxes: bool = True,
    show_points: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes and sampling points from df on a copy of img.
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

        # Bounding box
        if show_boxes:
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

        # Sampling point marker
        if show_points:
            x, y = compute_sample_point(left, top, right, bottom, point_mode)
            r = 5
            draw.ellipse([(x - r, y - r), (x + r, y + r)], width=3)

    return annotated


# ----- Load data -----
if not os.path.exists(CSV_PATH):
    st.error(f"Detections CSV not found: {CSV_PATH}")
    st.stop()

predictions = pd.read_csv(CSV_PATH)

images = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)
if not images:
    st.error(f"No images found in {IMAGE_FOLDER}")
    st.stop()


# ----- UI -----
left, right = st.columns([1, 1])

with left:
    st.subheader("Select image")
    selected_image = st.selectbox("Image", images)

    show_boxes = st.toggle("Show bounding boxes", value=True)
    show_points = st.toggle("Show sampling points", value=True)

    point_mode = st.radio(
        "Sampling point definition",
        ["bottom-center", "center"],
        horizontal=True,
        help="Bottom-center is often closer to ground contact (useful for distance).",
    )

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
        df_show = df_img.copy()

    # Add sample point columns to table (if bbox columns exist)
    required_bbox_cols = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
    if len(df_show) > 0 and all(c in df_show.columns for c in required_bbox_cols):
        xs = []
        ys = []
        for _, row in df_show.iterrows():
            left_i = _safe_int(row.get("bbox_left"))
            top_i = _safe_int(row.get("bbox_top"))
            right_i = _safe_int(row.get("bbox_right"))
            bottom_i = _safe_int(row.get("bbox_bottom"))
            if None in (left_i, top_i, right_i, bottom_i):
                xs.append(None)
                ys.append(None)
                continue
            x, y = compute_sample_point(left_i, top_i, right_i, bottom_i, point_mode)
            xs.append(x)
            ys.append(y)
        df_show["sample_x"] = xs
        df_show["sample_y"] = ys

    st.dataframe(df_show, use_container_width=True)
    st.caption(f"{len(df_show)} detections shown (of {len(df_img)} total)")

st.divider()
st.subheader("Preview")

preview = img
if len(df_show) > 0 and (show_boxes or show_points):
    preview = draw_overlays(
        img,
        df_show,
        point_mode=point_mode,
        show_boxes=show_boxes,
        show_points=show_points,
    )

st.image(preview, caption=f"{selected_image} | points: {point_mode}", use_container_width=True)