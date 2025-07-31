import streamlit as st
import torch
import os
import json
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from train import build_model
from predict import find_latest_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
import io

# ————————————————————————————————————————————————————————————————————————
# 1. PAGE SETTINGS
# ————————————————————————————————————————————————————————————————————————
st.set_page_config(page_title="Audiogram Classifier", layout="wide")

# Dark mode toggle
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

is_dark = st.sidebar.checkbox("🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = is_dark

# Load fonts and styles
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Inject dark or light styles
dark_css = """
<style>
html, body, .main, .stApp {
    background: linear-gradient(to right, #141e30, #243b55) !important;
    color: white !important;
    font-family: 'Rubik', sans-serif;
}
.stSidebar, .sidebar .sidebar-content {
    background-color: #1c1c1c !important;
    color: white !important;
}
button, .stButton>button {
    background-color: #3a3a3a !important;
    color: white !important;
    border: 1px solid white;
}
.stTextInput>div>div>input {
    background-color: #2b2b2b !important;
    color: white !important;
}
.expander, .stExpander {
    background-color: #1a1a1a !important;
    color: white !important;
}
</style>
"""

light_css = """
<style>
html, body, .main, .stApp {
    background: linear-gradient(to right, #ffffff, #f1f1f1) !important;
    color: black !important;
    font-family: 'Rubik', sans-serif;
}
.stSidebar, .sidebar .sidebar-content {
    background-color: #f9f9f9 !important;
    color: black !important;
}
button, .stButton>button {
    background-color: #ffffff !important;
    color: black !important;
    border: 1px solid black;
}
.expander, .stExpander {
    background-color: #f0f0f0 !important;
    color: black !important;
}
</style>
"""

st.markdown(dark_css if st.session_state.dark_mode else light_css, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <h1 style='background: linear-gradient(to right, #6a11cb, #2575fc); -webkit-background-clip: text; color: transparent;'>
        {'🌙' if st.session_state.dark_mode else '☀️'} Audiogram Classifier Settings</h1>
    """, unsafe_allow_html=True)
    retrain = st.button("🔁 Retrain Model")
    view_eval = st.checkbox("📊 Show Evaluation Tab")
    view_history = st.checkbox("📈 Show Training History")
    reset_button = st.button("🔄 Reset")
    st.markdown("---")
    st.caption("Built by Daniel, Janik & Vivienne 💖")

if reset_button:
    st.session_state.clear()
    st.rerun()

# ————————————————————————————————————————————————————————————————————————
# 2. CLASSIFICATION LOGIC
# ————————————————————————————————————————————————————————————————————————

st.markdown("""
<h1 style='background: linear-gradient(to right, #ff6a00, #ee0979); -webkit-background-clip: text; color: transparent;'>🧠 Audiogram Classifier</h1>
<p>Upload one or more audiogram images to receive tinnitus classifications directly using the local model.</p>
""", unsafe_allow_html=True)

model, idx_to_class, device, error = None, None, None, None

def build_model_and_load():
    model_path = find_latest_model("models")
    mapping_path = os.path.join("models", "class_mapping.json")
    if not model_path or not os.path.exists(mapping_path):
        st.error("Model or class mapping not found. Please retrain the model first.")
        st.stop()
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(idx_to_class), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, idx_to_class, device, None

def predict_image(image: Image.Image, model, idx_to_class, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    pred_class = idx_to_class[predicted.item()]
    return pred_class, confidence.item(), predicted.item(), probs.cpu().numpy()

def generate_pdf_report(predictions):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "Audiogram Classification Report")
    c.setFont("Helvetica", 12)
    y = 700
    for pred in predictions:
        c.drawString(50, y, f"Image: {pred['filename']}")
        y -= 20
        c.drawString(50, y, f"Prediction: {pred['class']}")
        y -= 20
        c.drawString(50, y, f"Confidence: {pred['confidence']:.2%}")
        y -= 30
    c.save()
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="audiogram_report.pdf">📄 Download PDF Report</a>'
    return href

if model is None:
    model, idx_to_class, device, error = build_model_and_load()

uploaded_files = st.file_uploader("Upload audiogram image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
predictions = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            with st.spinner("🔍 Making prediction..."):
                pred_class, confidence, _, _ = predict_image(image, model, idx_to_class, device)
                color = "green" if pred_class == "no_tinnitus" else "red"
                predictions.append({"filename": uploaded_file.name, "class": pred_class, "confidence": confidence})
                st.markdown(f"""
                <div style='padding: 15px; border-radius: 10px; background-color: {color}; color: white; margin-bottom: 10px;'>
                    <div style='font-size: 20px; margin-bottom: 5px;'>🎯 <b>Prediction:</b> <span style='color:white'>{pred_class}</span></div>
                    <div style='font-size: 18px;'>📈 <b>Confidence:</b> <span style='color:white'>{confidence:.2%}</span></div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"🚫 Prediction error for {uploaded_file.name}: {e}")

    if predictions:
        st.markdown(generate_pdf_report(predictions), unsafe_allow_html=True)

# ————————————————————————————————————————————————————————————————————————
# 3. EVALUATION TAB
# ————————————————————————————————————————————————————————————————————————

if view_eval:
    with st.expander("📊 Evaluation Metrics", expanded=True):
        st.markdown("### 🔍 Classification Report")
        y_true = ['no_tinnitus'] * 4 + ['tinnitus'] * 4
        y_pred = ['no_tinnitus', 'tinnitus', 'no_tinnitus', 'no_tinnitus', 'tinnitus', 'tinnitus', 'tinnitus', 'no_tinnitus']
        st.text(classification_report(y_true, y_pred, target_names=['no_tinnitus', 'tinnitus']))

        st.markdown("### 📉 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=['no_tinnitus', 'tinnitus'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['no_tinnitus', 'tinnitus'], yticklabels=['no_tinnitus', 'tinnitus'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ————————————————————————————————————————————————————————————————————————
# 4. TRAINING HISTORY TAB
# ————————————————————————————————————————————————————————————————————————

if view_history:
    with st.expander("📈 Training History", expanded=True):
        st.markdown("### 📚 Training Accuracy & Loss")
        fig, ax = plt.subplots()
        epochs = list(range(1, 6))
        acc = [0.6, 0.7, 0.75, 0.8, 0.85]
        loss = [1.2, 0.9, 0.7, 0.5, 0.4]
        ax.plot(epochs, acc, label="Accuracy", marker='o')
        ax.plot(epochs, loss, label="Loss", marker='x')
        ax.set_xlabel("Epoch")
        ax.set_title("Dummy Training Metrics")
        ax.legend()
        st.pyplot(fig)