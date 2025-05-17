import streamlit as st
import os
import torch
import joblib
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import gdown
    
# 🔁 Your Google Drive file ID here:
GOOGLE_DRIVE_FILE_ID = "1B4OsZCpWMXSsCW_8QWY6HcYDftYOfCLA"

MODEL_PATH = "models/distilbert_model/model.safetensors"

# Check if model.safetensors exists, else download
def ensure_model_weights():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model weights from Google Drive...")
        os.makedirs("models/distilbert_model", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")

# Load all components
@st.cache_resource
def load_model():
    ensure_model_weights()

    tokenizer = DistilBertTokenizer.from_pretrained("models/tokenizer")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_model")
    label_encoder = joblib.load("models/label_encoder.pkl")

    # Optional category mapping
    try:
        category_mapping = joblib.load("models/category_mapping.pkl")
    except:
        category_mapping = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer, label_encoder, category_mapping, device

# Clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Predict category
def predict_category(text, model, tokenizer, label_encoder, category_mapping, device):
    clean = clean_text(text)

    encoding = tokenizer(
        clean,
        add_special_tokens=True,
        max_length=64,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    category = label_encoder.inverse_transform([predicted_idx.item()])[0]

    if category_mapping:
        category = category_mapping.get(category, category)

    return category, confidence.item()

# Streamlit UI
st.set_page_config(page_title="🧾 Product Categorizer", page_icon="📦")
st.title("🧾 Product Description Categorizer")
st.write("Enter a product description to get the predicted category.")

# Load model and tokenizer
model, tokenizer, label_encoder, category_mapping, device = load_model()

user_input = st.text_area("✍️ Enter product description:", height=100)

if st.button("Predict Category"):
    if not user_input.strip():
        st.warning("Please enter a product description.")
    else:
        with st.spinner("Predicting..."):
            category, confidence = predict_category(user_input, model, tokenizer, label_encoder, category_mapping, device)
        st.success(f"📦 Predicted Category: **{category}**")
        st.info(f"🔍 Confidence Score: `{confidence:.4f}`")
