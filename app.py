import os
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(page_title="Depth Estimation Tool", layout="wide")
st.title("Depth Estimation Tool")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "test-imgs")
CSV_PATH = os.path.join(IMAGE_FOLDER, "results_detections.csv")
CALIBRATION_CSV_PATH = os.path.join(IMAGE_FOLDER, "calibration_points.csv")


# ---------- helpers ----------
def _safe_int(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
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

        if show_points:
            x, y = compute_sample_point(left, top, right, bottom, point_mode)
            r = 6
            draw.ellipse([(x - r, y - r), (x + r, y + r)], width=3)

    return annotated


def load_calibration_points() -> pd.DataFrame:
    """Load calibration points from CSV if it exists; otherwise return empty DF."""
    cols = ["x", "y", "distance_m"]
    if os.path.exists(CALIBRATION_CSV_PATH):
        try:
            df = pd.read_csv(CALIBRATION_CSV_PATH)
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            return df[cols].copy()
        except Exception:
            return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)


def save_calibration_points(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CALIBRATION_CSV_PATH), exist_ok=True)
    df.to_csv(CALIBRATION_CSV_PATH, index=False)


# ---------- load data ----------
if not os.path.exists(CSV_PATH):
    st.error(f"Detections CSV not found: {CSV_PATH}")
    st.stop()

predictions = pd.read_csv(CSV_PATH)

images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
if not images:
    st.error(f"No images found in {IMAGE_FOLDER}")
    st.stop()

if "relative_path" not in predictions.columns:
    st.error("CSV missing required column: relative_path")
    st.stop()

# session state
if "calib_df" not in st.session_state:
    st.session_state.calib_df = load_calibration_points()

if "last_click" not in st.session_state:
    st.session_state.last_click = None  # (x, y)


# ---------- UI layout ----------
controls_col, preview_col = st.columns([0.30, 0.70], gap="large")

with controls_col:
    st.subheader("Controls")

    selected_image = st.selectbox("Image", images)

    show_boxes = st.toggle("Show bounding boxes", value=True)
    show_points = st.toggle("Show sampling points", value=True)

    point_mode = st.radio(
        "Sampling point definition",
        ["bottom-center", "center"],
        horizontal=True,
        help="Bottom-center is often closer to ground contact (useful for distance).",
    )

    # Filter detections for image
    df_img = predictions[predictions["relative_path"] == selected_image].copy()

    # Sort by confidence if available
    if "confidence" in df_img.columns:
        df_img["confidence"] = pd.to_numeric(df_img["confidence"], errors="coerce")
        df_img = df_img.sort_values("confidence", ascending=False)

    # Optional label filter
    label_filter = "All"
    if "label" in df_img.columns and len(df_img) > 0:
        labels = sorted(df_img["label"].astype(str).unique().tolist())
        label_filter = st.selectbox("Filter by label", ["All"] + labels)

    if label_filter != "All":
        df_show = df_img[df_img["label"].astype(str) == label_filter].copy()
    else:
        df_show = df_img.copy()

    # Add sample point columns to table
    required_bbox_cols = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
    if len(df_show) > 0 and all(c in df_show.columns for c in required_bbox_cols):
        xs, ys = [], []
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

    st.caption(f"{len(df_show)} detections shown (of {len(df_img)} total)")

    st.divider()
    st.subheader("Calibration points (click image + enter distance)")

    st.caption("Click on the preview image to capture (x, y), then enter the known distance (meters) and add the point.")

    # Distance only
    dist_in = st.number_input("known distance (m)", min_value=0.0, value=5.0, step=0.5)

    last = st.session_state.last_click
    if last is None:
        st.info("Click on the preview image (right) to select a calibration point.")
    else:
        st.success(f"Selected point: x={last[0]}, y={last[1]}")

    add_clicked = st.button("Add calibration point", use_container_width=True)

    if add_clicked:
        d_val = _safe_float(dist_in)
        if last is None:
            st.error("Please click on the preview image first.")
        elif d_val is None or d_val <= 0:
            st.error("Please enter a distance > 0 meters.")
        else:
            new_row = pd.DataFrame([{"x": int(last[0]), "y": int(last[1]), "distance_m": float(d_val)}])
            st.session_state.calib_df = pd.concat([st.session_state.calib_df, new_row], ignore_index=True)
            st.success(f"Added point: ({int(last[0])}, {int(last[1])}) → {float(d_val)} m")

    st.write("Current calibration points:")
    st.dataframe(st.session_state.calib_df, use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save calibration points", use_container_width=True):
            save_calibration_points(st.session_state.calib_df)
            st.success(f"Saved to {CALIBRATION_CSV_PATH}")
    with b2:
        if st.button("Reload from disk", use_container_width=True):
            st.session_state.calib_df = load_calibration_points()
            st.success("Reloaded calibration points from disk")

    if st.button("Clear all calibration points", use_container_width=True):
        st.session_state.calib_df = pd.DataFrame(columns=["x", "y", "distance_m"])
        st.session_state.last_click = None
        st.warning("Cleared calibration points (not saved until you click Save).")


with preview_col:
    st.subheader("Preview")
    st.markdown("---")

    img_path = os.path.join(IMAGE_FOLDER, selected_image)
    img = Image.open(img_path).convert("RGB")

    preview = img
    if len(df_show) > 0 and (show_boxes or show_points):
        preview = draw_overlays(
            img,
            df_show,
            point_mode=point_mode,
            show_boxes=show_boxes,
            show_points=show_points,
        )

    st.caption("Click anywhere on the image to select a calibration point.")
    click = streamlit_image_coordinates(preview, key="preview_click")

    if click is not None:
        st.session_state.last_click = (int(click["x"]), int(click["y"]))

    # Optional caption (since streamlit_image_coordinates already renders the image)
    st.caption(f"{selected_image} | points: {point_mode}")

st.subheader("Detections table")
st.dataframe(df_show, use_container_width=True)