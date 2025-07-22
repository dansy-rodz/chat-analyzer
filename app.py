import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import platform
import re
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# ✅ Cross-platform compatibility for Tesseract (OCR)
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🔧 Streamlit page setup
st.set_page_config(page_title="Chat Analyzer", layout="wide")
st.title("💬 Universal Chat Analyzer")

# 📥 Input Options
input_mode = st.radio("Choose Input Type", ["📷 Upload Screenshot", "📋 Paste Chat Text", "📁 Upload File (CSV/TXT)"])

raw_text = ""

# 🖼️ Screenshot Upload + OCR
if input_mode == "📷 Upload Screenshot":
    image_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Screenshot", use_column_width=True)
        with st.spinner("Extracting text from image..."):
            raw_text = pytesseract.image_to_string(image)
        st.text_area("🧾 OCR Extracted Text", raw_text, height=200)

# 📋 Paste Chat
elif input_mode == "📋 Paste Chat Text":
    raw_text = st.text_area("Paste chat below", height=250)

# 📁 CSV/TXT Upload
elif input_mode == "📁 Upload File (CSV/TXT)":
    file = st.file_uploader("Upload chat file", type=["txt", "csv"])
    if file:
        if file.name.endswith(".txt"):
            raw_text = file.read().decode("utf-8")
        else:
            df_uploaded = pd.read_csv(file)
            raw_text = "\n".join(df_uploaded[df_uploaded.columns[0]].astype(str))

# 🧠 Process Extracted Text
if raw_text:
    st.divider()
    st.subheader("🔍 Parsed Chat Messages")

    # 🧠 Option 2: AI-style chunking without sender/timestamp requirement
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    messages = []
    current_msg = ""

    for line in lines:
        if re.match(r"^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", line) or ":" in line[:15]:
            if current_msg:
                messages.append(current_msg.strip())
            current_msg = line
        else:
            current_msg += " " + line

    if current_msg:
        messages.append(current_msg.strip())

    if not messages:
        st.error("❌ No valid chat lines found.")
        st.stop()

    df = pd.DataFrame(messages, columns=["Raw"])

    # Split Sender & Message if possible
    df["Sender"] = df["Raw"].apply(lambda x: x.split(":")[0] if ":" in x else "Unknown")
    df["Message"] = df["Raw"].apply
