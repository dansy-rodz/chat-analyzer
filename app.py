import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import pytesseract

# ---- Setup Tesseract Path (Windows only) ----
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\cinetra rodrigues\AppData\Roaming\Python\Python313\Scripts\tesseract.exe'

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Chat Screenshot Analyzer", layout="wide")
st.title("🖼️ Chat Screenshot Analyzer")

# ---- Upload Image ----
image_file = st.file_uploader("📷 Upload a WhatsApp chat screenshot", type=["jpg", "jpeg", "png"])

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(image_file)

    with st.spinner("🔍 Extracting text from image using OCR..."):
        raw_text = pytesseract.image_to_string(image)
        st.success("✅ Text extracted successfully!")
        st.text_area("📄 Extracted Text", raw_text, height=200)

    # ---- Parse Text into Chat Messages ----
    st.subheader("🧾 Parsed Chat Messages")
    lines = raw_text.split("\n")
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s+-\s+(.*?):\s+(.*)$'
    records = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            date, time, sender, message = match.groups()
            records.append({
                "Timestamp": f"{date} {time}",
                "Sender": sender,
                "Message": message
            })

    if not records:
        st.error("⚠️ Could not parse valid chat lines from image.")
        st.stop()

    df = pd.DataFrame(records)
    st.dataframe(df)

    # ---- Sentiment Analysis ----
    df['Polarity'] = df['Message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral')
    df['Sentiment_Score'] = df['Polarity']

    st.subheader("📈 Sentiment Trend")
    st.line_chart(df['Sentiment_Score'])

    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(df['Sentiment'].value_counts())

    # ---- Key Issues Detection ----
    st.subheader("🧩 Detected Issues")
    keywords = ["refund", "delay", "cancel", "problem", "broken", "fail", "missing", "bad", "error", "late", "poor"]
