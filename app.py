import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

# =====================================
# LOAD SECRETS (Cloud + Local fallback)
# =====================================

if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]
    API_KEY = st.secrets["API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    API_URL = os.getenv("API_URL")
    API_KEY = os.getenv("API_KEY")

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🍃",
    layout="wide"
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🍃 About The Project")
st.sidebar.image("img.png", width=250)

st.sidebar.markdown("""
### 🌱 Plant Disease Detection System

This system uses:

- YOLOv8x4 for Leaf Detection  
- CNN for Disease Classification  
- 102 Plant Disease Classes  

The model can detect:
- Different crops
- Healthy leaves
- Multiple plant diseases

---

### 🧪 Want Test Images?

👉 https://www.kaggle.com/datasets/armia123/plant-leaf-disease-classification?select=test

---

### 👩‍💻 Developers
- Armia Gamal
- Sara Essam
""")

st.sidebar.markdown("---")
st.sidebar.info("Upload a plant image to start detection.")

# =====================================
# MAIN TITLE
# =====================================

st.title("🍃 Plant Disease Detection System")
st.markdown("AI-powered detection & classification with **102 classes**")

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    if not API_URL or not API_KEY:
        st.error("API configuration missing. Check Secrets settings.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image)

    with st.spinner("Detecting and Classifying..."):

        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            headers = {
                "Authorization": f"Bearer {API_KEY}"
            }

            response = requests.post(
                API_URL,
                headers=headers,
                files=files,
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            st.error("API connection failed.")
            st.text(str(e))
            st.stop()

    # =====================================
    # Annotated Image
    # =====================================

    with col2:
        st.subheader("Annotated Image")

        annotated_b64 = data.get("annotated_image")

        if annotated_b64:
            annotated_bytes = base64.b64decode(annotated_b64)
            annotated_img = Image.open(io.BytesIO(annotated_bytes))
            st.image(annotated_img)
        else:
            st.warning("No annotated image returned.")

    # =====================================
    # RESULTS
    # =====================================

    st.divider()
    st.subheader("Classification Results")

    results = data.get("results", [])

    if results:
        formatted_results = []

        for r in results:

            confidence_value = float(r["confidence"].replace('%','')) / 100

            crop_name = r["crop"]
            disease = r["disease"]
            confidence = r["confidence"]

            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            crop_img = original_image.crop((x1, y1, x2, y2))

            color = "green" if disease.lower() == "healthy" else "red"

            st.markdown(f"""
            <h3 style='color:{color};'>
            🌿 {crop_name}
            </h3>
            <b>Disease:</b> {disease}<br>
            <b>Confidence:</b> {confidence}
            """, unsafe_allow_html=True)

            st.image(crop_img, width=300)
            st.progress(confidence_value)
            st.divider()

            formatted_results.append({
                "Box": r["Box"],
                "Crop": crop_name,
                "Disease": disease,
                "Confidence": confidence
            })

        st.subheader("📊 Summary Table")
        st.dataframe(formatted_results, use_container_width=True)

    else:
        st.warning("No objects detected.")

# =====================================
# FOOTER
# =====================================

st.markdown("""
---
<div style='text-align: center;'>
🌿 Developed by <b>Armia Gamal</b> & <b>Sara Essam</b>  
AI Plant Disease Detection System — 102 Classes
</div>
""", unsafe_allow_html=True)
