import streamlit as st
import pandas as pd
import re
import pytesseract
from PIL import Image
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# Optional local tesseract path (ONLY for local dev)
# Comment out if using Streamlit Cloud
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Config ---
st.set_page_config(page_title="Chat Analyzer", layout="wide")

# --- Login ---
users = {"admin": "admin123", "guest": "guest123"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login to Chat Analyzer")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    st.stop()

# --- Header ---
st.title("💬 Chat Transcript & Screenshot Analyzer")

upload_option = st.radio("Choose input method:", (
    "📷 Image (Screenshot)",
    "📄 CSV/TXT File",
    "📝 Paste Chat Text"
))

# ========== IMAGE OCR ==========
if upload_option == "📷 Image (Screenshot)":
    image_file = st.file_uploader("Upload chat screenshot", type=["jpg", "jpeg", "png"])
    if image_file:
        st.image(image_file, use_column_width=True)
        image = Image.open(image_file)
        with st.spinner("Extracting text..."):
            raw_text = pytesseract.image_to_string(image)

        st.subheader("📝 OCR Extracted Text")
        st.text_area("Text from Image", raw_text, height=200, key="ocr_text")
        st.markdown("""
            <button onclick="navigator.clipboard.writeText(document.querySelector('textarea[data-testid=stTextArea]').value)">📋 Copy to Clipboard</button>
        """, unsafe_allow_html=True)

        # Parse chat
        lines = raw_text.split("\n")
        pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s+-\s+(.*?):\s+(.*)$'
        records = []
        for line in lines:
            match = re.match(pattern, line)
            if match:
                date, time, sender, message = match.groups()
                records.append({"Timestamp": f"{date} {time}", "Sender": sender, "Message": message})
        if not records:
            st.warning("⚠️ Could not parse valid chat lines.")
            st.stop()
        df = pd.DataFrame(records)

# ========== FILE UPLOAD ==========
elif upload_option == "📄 CSV/TXT File":
    file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            raw_text = file.read().decode("utf-8")
            lines = raw_text.split("\n")
            pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s+-\s+(.*?):\s+(.*)$'
            records = []
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    date, time, sender, message = match.groups()
                    records.append({"Timestamp": f"{date} {time}", "Sender": sender, "Message": message})
            if not records:
                st.warning("⚠️ Could not parse valid chat lines.")
                st.stop()
            df = pd.DataFrame(records)

# ========== PASTE CHAT ==========
elif upload_option == "📝 Paste Chat Text":
    raw_text = st.text_area("Paste your chat below:", height=300)
    if raw_text.strip():
        lines = raw_text.split("\n")
        pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s+-\s+(.*?):\s+(.*)$'
        records = []
        for line in lines:
            match = re.match(pattern, line)
            if match:
                date, time, sender, message = match.groups()
                records.append({"Timestamp": f"{date} {time}", "Sender": sender, "Message": message})
        if not records:
            st.warning("⚠️ Could not parse valid chat lines.")
            st.stop()
        df = pd.DataFrame(records)

# ========== ANALYSIS ==========
if 'df' in locals():
    st.subheader("📋 Parsed Chat Log")
    st.dataframe(df)

    # Sentiment
    df['Polarity'] = df['Message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral')
    df['Sentiment_Score'] = df['Polarity']

    st.subheader("📈 Sentiment Trend")
    st.line_chart(df['Sentiment_Score'])

    st.subheader("📊 Sentiment Breakdown")
    st.bar_chart(df['Sentiment'].value_counts())

    # ⭐ Satisfaction Score
    st.subheader("⭐ Estimated Satisfaction Score")
    customer_msgs = df[df['Sender'].str.lower().str.contains("customer", na=False)]
    pos = customer_msgs[customer_msgs['Sentiment'] == 'Positive'].shape[0]
    neg = customer_msgs[customer_msgs['Sentiment'] == 'Negative'].shape[0]
    total = customer_msgs.shape[0]
    if total > 0:
        score = round(((pos - neg) / total) * 5, 2)
        score = max(0, min(5, score + 2.5))
        st.markdown(f"<h2 style='color:#FFD700'>{score} / 5</h2>", unsafe_allow_html=True)
    else:
        st.info("Not enough messages from 'customer' to compute satisfaction.")

    # 🧩 Issues
    st.subheader("🧩 Key Issues Detected")
    issue_keywords = ["refund", "delay", "broken", "cancel", "late", "not working", "missing", "bad", "poor", "problem", "error", "fail"]
    df['Issues'] = df['Message'].apply(lambda msg: ', '.join([word for word in issue_keywords if word in msg.lower()]))
    issue_list = df["Issues"][df["Issues"] != ""]
    if not issue_list.empty:
        st.write(issue_list.unique())
    else:
        st.info("No major issues detected.")

    # ☁️ Word Cloud
    st.subheader("☁️ Word Cloud")
    all_text = " ".join(df["Message"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # 🧠 Summary
    st.subheader("🧠 AI Summary")
    with st.spinner("Generating summary..."):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        limited_text = all_text[:3000]
        summary = summarizer(limited_text, max_length=130, min_length=30, do_sample=False)
        summary_text = summary[0]["summary_text"]
        st.success(summary_text)
        st.text_area("AI Summary", summary_text, height=150, key="summary_text")
        st.markdown("""
            <button onclick="navigator.clipboard.writeText(document.querySelectorAll('textarea[data-testid=stTextArea]')[1].value)">📋 Copy Summary</button>
        """, unsafe_allow_html=True)

    # ⬇️ Download
    st.download_button("⬇️ Download Analyzed CSV", df.to_csv(index=False), "analyzed_chat.csv", "text/csv")
