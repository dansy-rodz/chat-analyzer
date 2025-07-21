# app.py
import streamlit as st
import pandas as pd
import pytesseract
import re
from PIL import Image
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import io

# Optional: If running locally with tesseract installed at custom path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Chat Analyzer", layout="wide")

# ---- LOGIN SYSTEM ----
users = {"admin": "admin123", "user": "pass"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login to Chat Analyzer")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if users.get(username) == password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

st.title("💬 Chat Analyzer")
st.markdown("Upload a chat file, paste a chat, or upload an image.")
st.markdown("---")

# --- INPUT OPTION ---
input_type = st.radio("Choose input type", ["📄 Upload File (CSV/TXT)", "📝 Paste Chat Text", "📷 Upload Screenshot"])
chat_text = ""

if input_type == "📄 Upload File (CSV/TXT)":
    uploaded_file = st.file_uploader("Upload chat CSV or TXT", type=["csv", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if {"Sender", "Message"}.issubset(df.columns):
                chat_text = "\n".join(df["Sender"] + ": " + df["Message"])
            else:
                st.error("CSV must have 'Sender' and 'Message' columns.")
        else:
            chat_text = uploaded_file.read().decode("utf-8")

elif input_type == "📝 Paste Chat Text":
    chat_text = st.text_area("Paste chat below", height=300)

elif input_type == "📷 Upload Screenshot":
    image_file = st.file_uploader("Upload chat screenshot", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Screenshot", use_column_width=True)
        chat_text = pytesseract.image_to_string(image)
        st.text_area("🧾 OCR Extracted Text", chat_text, height=200)

# --- PROCESSING ---
if chat_text:
    st.markdown("---")
    st.subheader("🔍 Parsed Chat Messages")

    # Parse common formats
    pattern = r'^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s+-\s+(.*?):\s+(.*)$'
    lines = chat_text.strip().split("\n")
    records = []

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            date, time, sender, message = match.groups()
            records.append({"Timestamp": f"{date} {time}", "Sender": sender, "Message": message})

    if not records:
        st.error("❌ Could not parse any chat lines.")
        st.stop()

    df = pd.DataFrame(records)
    st.dataframe(df)

    # --- Sentiment Analysis ---
    df["Polarity"] = df["Message"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(lambda p: "Positive" if p > 0 else "Negative" if p < 0 else "Neutral")

    st.subheader("📊 Sentiment Analysis")
    st.line_chart(df["Polarity"])
    st.bar_chart(df["Sentiment"].value_counts())

    # --- Satisfaction Score ---
    st.subheader("⭐ Estimated Customer Satisfaction")
    customer_msgs = df[df["Sender"].str.lower() == "customer"]
    pos = customer_msgs[customer_msgs["Sentiment"] == "Positive"].shape[0]
    neg = customer_msgs[customer_msgs["Sentiment"] == "Negative"].shape[0]
    total = customer_msgs.shape[0]
    if total > 0:
        score = round(((pos - neg) / total) * 5 + 2.5, 2)
        score = max(0, min(5, score))
        st.markdown(f"### 🟡 Score: `{score} / 5`")
    else:
        st.info("No customer messages detected.")

    # --- Key Issues ---
    st.subheader("🧩 Key Issues Detected")
    keywords = ["refund", "delay", "broken", "cancel", "not working", "problem", "issue", "missing", "bad"]

    def detect_issues(msg):
        return ", ".join([k for k in keywords if k in msg.lower()])

    df["Issues"] = df["Message"].apply(detect_issues)
    issues = df["Issues"][df["Issues"] != ""]
    st.write(issues.unique().tolist() or "No issues found")

    # --- Word Cloud ---
    st.subheader("☁️ Word Cloud")
    all_text = " ".join(df["Message"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # --- Summary ---
    st.subheader("🧠 Chat Summary")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(all_text[:3000], max_length=130, min_length=30, do_sample=False)
    st.success(summary[0]['summary_text'])

    # --- Download ---
    st.subheader("⬇️ Download")
    st.download_button("Download Analyzed CSV", df.to_csv(index=False), "chat_analysis.csv", "text/csv")
