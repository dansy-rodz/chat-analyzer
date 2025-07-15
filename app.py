import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# ------------------- Safe OCR Import -------------------
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ------------------- App Settings -------------------
st.set_page_config(page_title="Chat Analyzer", layout="wide", page_icon="💬")
st.title("💬 Chat Transcript Analyzer for WhatsApp / Raw Text")

# ------------------- Login -------------------
users = {"admin": "admin123"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")
        if login_btn:
            if users.get(username) == password:
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    st.stop()

# ------------------- Chat Upload or Paste -------------------
st.subheader("📥 Upload or Paste Chat Transcript")

options = ["Paste chat text", "Upload .txt file"]
if OCR_AVAILABLE:
    options.append("Upload image (.jpg, .png)")
else:
    st.warning("⚠️ OCR not available in this environment (e.g., Streamlit Cloud).")

option = st.radio("Choose input method:", options)
raw_chat = ""

if option == "Paste chat text":
    raw_chat = st.text_area("Paste chat here (WhatsApp-style format)", height=300)

elif option == "Upload .txt file":
    file = st.file_uploader("Upload chat text file", type=["txt"])
    if file:
        raw_chat = file.read().decode("utf-8")

elif option == "Upload image (.jpg, .png)" and OCR_AVAILABLE:
    image_file = st.file_uploader("Upload a chat screenshot", type=["jpg", "jpeg", "png"])
    if image_file:
        st.image(image_file, caption="Uploaded Chat Image", use_column_width=True)
        image = Image.open(image_file)
        with st.spinner("🧠 Extracting text from image..."):
            raw_chat = pytesseract.image_to_string(image)
            st.success("✅ Text extracted from image!")
            st.text_area("Extracted Chat Text", raw_chat, height=200)

# ------------------- Continue If Chat Is Present -------------------
if raw_chat:
    st.subheader("🔍 Parsed Chat Preview")
    lines = raw_chat.split("\n")

    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s+-\s+(.*?):\s+(.*)$'
    records = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            date, time, sender, message = match.groups()
            records.append({"Sender": sender, "Message": message, "Timestamp": f"{date} {time}"})
    if not records:
        st.error("❌ Could not parse any chat lines. Please check your format.")
        st.stop()

    df = pd.DataFrame(records)
    st.dataframe(df.head())

    # ------------------- Sentiment Analysis -------------------
    df['Polarity'] = df['Message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral')
    df['Sentiment_Score'] = df['Polarity']

    st.subheader("📈 Sentiment Trend")
    st.line_chart(df['Sentiment_Score'], use_container_width=True)

    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts()
    fig1, ax1 = plt.subplots()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax1)
    ax1.set_title("Sentiment Distribution")
    st.pyplot(fig1)

    # ------------------- Issue Detection -------------------
    st.subheader("🧩 Key Issues Detected")
    issue_keywords = ["refund", "delay", "broken", "cancel", "late", "not working", "missing", "bad", "poor", "problem", "error", "fail", "issue"]

    def extract_issues(text):
        found = [word for word in issue_keywords if re.search(r'\\b' + re.escape(word) + r'\\b', text.lower())]
        return ', '.join(found)

    df['Key_Issues'] = df['Message'].apply(extract_issues)
    issues = df['Key_Issues'][df['Key_Issues'] != '']
    with st.expander("🔍 View Detected Issues"):
        st.write(issues.unique().tolist())

    # ------------------- Word Cloud -------------------
    st.subheader("☁️ Word Cloud")
    all_text = " ".join(df['Message'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)

    # ------------------- Summary -------------------
    st.subheader("🧠 AI Summary")
    with st.spinner("Summarizing..."):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        limited_text = all_text[:3000]
        summary = summarizer(limited_text, max_length=130, min_length=30, do_sample=False)
        st.success("Summary:")
        st.info(summary[0]['summary_text'])

    # ------------------- Download -------------------
    st.subheader("⬇️ Download Chat with Sentiment")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="analyzed_chat.csv",
        mime="text/csv"
    )

# ------------------- Footer -------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Chat Analyzer | WhatsApp/Text Analysis</p>", unsafe_allow_html=True)
