import streamlit as st
import requests
import base64
from PIL import Image
import io
import numpy as np

# =====================================
# API CONFIG (HARDCODED)
# =====================================

API_URL = "https://armia-gamal-plant-disease-api.hf.space/detect_and_classify"
API_KEY = "secret123"

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

st.sidebar.markdown("""
### 🌱 Plant Disease Detection System

This system uses:

- YOLOv8x4 for Leaf Detection  
- CNN for Disease Classification  
- 102 Plant Disease Classes  

The model can detect and classify:
- Different crops
- Healthy leaves
- Multiple plant diseases

---

### 🧪 Want Test Images?

👉 https://www.kaggle.com/datasets/armia123/plant-leaf-disease-classification?select=test

After downloading:
1. Open the **test** folder  
2. Choose any image  
3. Upload it here  

---

### 👩‍💻 Developers
- **Armia Gamal**
- **Sara Essam**
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

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # =====================================
    # SHOW ORIGINAL IMAGE (SAFE)
    # =====================================

    with col1:
        st.subheader("Original Image")

        uploaded_file.seek(0)  # مهم جدًا
        original_image = Image.open(uploaded_file).convert("RGB")
        original_np = np.array(original_image)

        st.image(original_np, width=500)

    # =====================================
    # CALL API
    # =====================================

    with st.spinner("🔍 Detecting and Classifying..."):

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
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout:
            st.error("⏳ Request timed out. Model might be loading.")
            st.stop()

        except requests.exceptions.RequestException as e:
            st.error("❌ API connection failed.")
            st.text(str(e))
            st.stop()

        except Exception as e:
            st.error("Unexpected Error")
            st.text(str(e))
            st.stop()

    # =====================================
    # Annotated Image
    # =====================================

    with col2:
        st.subheader("Annotated Image")

        annotated_b64 = data.get("annotated_image")

        if annotated_b64:
            try:
                annotated_bytes = base64.b64decode(annotated_b64)
                annotated_img = Image.open(io.BytesIO(annotated_bytes)).convert("RGB")
                annotated_np = np.array(annotated_img)

                st.image(annotated_np, width=500)

            except Exception:
                st.warning("Annotated image decoding failed.")
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

            confidence_raw = str(r.get("confidence", "0")).replace('%','')

            try:
                confidence_value = float(confidence_raw) / 100
            except:
                confidence_value = 0.0

            crop_name = r.get("crop", "Unknown")
            disease = r.get("disease", "Unknown")
            confidence = r.get("confidence", "0%")

            x1 = int(r.get("x1", 0))
            y1 = int(r.get("y1", 0))
            x2 = int(r.get("x2", 0))
            y2 = int(r.get("y2", 0))

            if x2 > x1 and y2 > y1:
                crop_img = original_image.crop((x1, y1, x2, y2))
                crop_np = np.array(crop_img)
            else:
                crop_np = original_np

            color = "green" if disease.lower() == "healthy" else "red"

            st.markdown(f"""
            <h3 style='color:{color};'>
            🌿 {crop_name}
            </h3>
            <b>Disease:</b> {disease}<br>
            <b>Confidence:</b> {confidence}
            """, unsafe_allow_html=True)

            st.image(crop_np, width=300)
            st.progress(confidence_value)
            st.divider()

            formatted_results.append({
                "Box": r.get("Box", ""),
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
