import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import pytesseract
import platform
from PIL import Image
import re
import os

# ✅ Cross-platform compatibility for Tesseract (OCR)
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Chat Analyzer", layout="wide")
st.title("💬 Universal Chat Analyzer")

# Input options
input_mode = st.radio("Choose Input Type", ["📷 Upload Screenshot", "📋 Paste Chat Text", "📁 Upload File (CSV/TXT)"])

raw_text = ""

if input_mode == "📷 Upload Screenshot":
    image_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Screenshot", use_column_width=True)
        with st.spinner("Extracting text with OCR..."):
            raw_text = pytesseract.image_to_string(image)
        st.text_area("🧾 OCR Extracted Text", raw_text, height=200)

elif input_mode == "📋 Paste Chat Text":
    raw_text = st.text_area("Paste Chat Below", height=250)

elif input_mode == "📁 Upload File (CSV/TXT)":
    file = st.file_uploader("Upload Chat File", type=["txt", "csv"])
    if file:
        if file.name.endswith(".txt"):
            raw_text = file.read().decode("utf-8")
        else:
            df = pd.read_csv(file)
            raw_text = "\n".join(df[df.columns[0]].astype(str))

if raw_text:
    st.divider()
    st.subheader("🔍 Parsed Chat Messages")

    # AI-based message chunking (Option 2)
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
        st.error("❌ No valid messages found.")
        st.stop()

    df = pd.DataFrame(messages, columns=["Message"])

    # Basic sender separation
    df["Sender"] = df["Message"].apply(lambda x: x.split(":")[0] if ":" in x else "Unknown")
    df["Message"] = df["Message"].apply(lambda x: x.split(":", 1)[1].strip() if ":" in x else x)

    # Sentiment
    df["Polarity"] = df["Message"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(lambda p: "Positive" if p > 0 else "Negative" if p < 0 else "Neutral")

    # Show table
    st.dataframe(df)

    # Charts
    st.subheader("📊 Sentiment Analysis")
    st.bar_chart(df["Sentiment"].value_counts())

    st.line_chart(df["Polarity"])

    # Key Issues
    st.subheader("🧩 Key Issues")
    keywords = ["refund", "delay", "problem", "broken", "cancel", "error", "missing", "bad"]
    df["Issues"] = df["Message"].apply(lambda x: ", ".join([k for k in keywords if k in x.lower()]))
    st.write(df[df["Issues"] != ""])

    # Word Cloud
    st.subheader("☁️ Word Cloud")
    all_text = " ".join(df["Message"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Summary
    st.subheader("🧠 Chat Summary")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_input = all_text[:3000]
    summary = summarizer(summary_input, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
    st.success(summary)

    # Download
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "chat_analysis.csv", "text/csv")
