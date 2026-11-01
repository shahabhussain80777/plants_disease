import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="centered")

st.title("🌱 Leaf Health & Disease Diagnostic System")
st.write("Upload an image of a bean plant leaf to diagnose fungal diseases or confirm healthy status.")

# Updated with your actual Hugging Face repository
MODEL_REPO = "shahab1/vit-beans-plant-disease"

@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model=MODEL_REPO)

try:
    classifier = load_classifier()
except Exception as e:
    st.error(f"Error loading model from Hugging Face: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    
    with st.spinner("Analyzing leaf patterns..."):
        predictions = classifier(image)
        top_pred = predictions[0]
        label = top_pred["label"].replace("_", " ").title()
        confidence = top_pred["score"] * 100

    st.markdown("---")
    st.subheader("Diagnostic Results")
    
    if "Healthy" in label:
        st.success(f"**Diagnosis:** {label}")
    else:
        st.warning(f"**Diagnosis:** {label}")
        
    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
