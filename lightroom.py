import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import rawpy
import numpy as np
import io

st.set_page_config(page_title="RAW + JPG Image Editor", layout="centered")
st.title("🌌 Milky Way Photo Editor (RAW + JPG Support)")

# File upload
uploaded_file = st.file_uploader("📁 Upload a Milky Way image", type=["jpg", "jpeg", "png", "cr2", "nef", "arw", "dng", "rw2"])

def apply_exposure(image_pil, ev):
    img_np = np.asarray(image_pil).astype(np.float32) / 255.0
    factor = 2 ** ev
    img_np = np.clip(img_np * factor, 0, 1)
    return Image.fromarray((img_np * 255).astype(np.uint8))

def apply_tone_curve(image_pil, highlights, shadows, whites, blacks):
    img_np = np.asarray(image_pil).astype(np.float32) / 255.0

    # Highlights (target upper range)
    if highlights != 0:
        img_np = img_np + (highlights / 100.0) * (img_np - img_np**2)

    # Shadows (target lower range)
    if shadows != 0:
        img_np = img_np + (shadows / 100.0) * ((1 - img_np) * img_np)

    # Whites
    img_np = np.clip(img_np + (whites / 100.0), 0, 1)

    # Blacks
    img_np = np.clip(img_np + (blacks / 100.0), 0, 1)

    return Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))

if uploaded_file:
    file_ext = uploaded_file.name.lower().split('.')[-1]

    if file_ext in ['cr2', 'nef', 'arw', 'dng', 'rw2']:
        try:
            with rawpy.imread(uploaded_file) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
                input_image = Image.fromarray(rgb)
                st.success("✅ RAW image successfully processed!")
        except Exception as e:
            st.error(f"❌ Error reading RAW file: {e}")
            st.stop()
    else:
        input_image = Image.open(uploaded_file).convert("RGB")

    st.image(input_image, caption="Original Image", use_container_width=True)

    # Sidebar sliders
    st.sidebar.header("🎚️ Editing Controls")
    ev_value = st.sidebar.slider("Exposure (EV Stops)", 0.0, 2.0, 0.5, step=0.1)
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
    sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0)
    blur = st.sidebar.slider("Blur", 0.0, 5.0, 0.0)

    st.sidebar.header("🌌 Milky Way Tone Adjustments")
    highlights = st.sidebar.slider("Highlights", -100, 100, -100)
    shadows = st.sidebar.slider("Shadows", -100, 100, 60)
    whites = st.sidebar.slider("Whites", -100, 100, 30)
    blacks = st.sidebar.slider("Blacks", -100, 100, -50)

    # Apply adjustments
    edited_image = input_image.copy()
    edited_image = apply_exposure(edited_image, ev_value)
    edited_image = apply_tone_curve(edited_image, highlights, shadows, whites, blacks)
    edited_image = ImageEnhance.Brightness(edited_image).enhance(brightness)
    edited_image = ImageEnhance.Contrast(edited_image).enhance(contrast)
    edited_image = ImageEnhance.Sharpness(edited_image).enhance(sharpness)
    if blur > 0:
        edited_image = edited_image.filter(ImageFilter.GaussianBlur(blur))

    st.subheader("🖼️ Edited Image")
    st.image(edited_image, use_container_width=True)

    # Download
    buf = io.BytesIO()
    edited_image.save(buf, format="JPEG")
    st.download_button(
        label="📥 Download Edited Image",
        data=buf.getvalue(),
        file_name="milkyway_edited.jpg",
        mime="image/jpeg"
    )
