import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import re

# Page settings
st.set_page_config(page_title="Chat Transcript Analyzer", layout="wide", page_icon="💬")
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>💬 Chat Transcript Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your customer support chat log and gain insights such as sentiment trends, satisfaction score, and detected key issues.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload section
st.subheader("📤 Step 1: Upload Chat Log (.csv)")
uploaded_file = st.file_uploader("Upload CSV with columns: Sender, Message", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, quotechar='"', skipinitialspace=True)
    
    if {"Sender", "Message"}.issubset(df.columns):
        st.success("✅ Chat log uploaded successfully!")
        with st.expander("📝 Preview Chat Log"):
            st.dataframe(df.head())

        # Sentiment analysis
        df['Polarity'] = df['Message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral')
        df['Sentiment_Score'] = df['Polarity']

        # Sentiment trend line chart
        st.subheader("📈 Step 2: Sentiment Trend Over Time")
        st.line_chart(df['Sentiment_Score'], use_container_width=True)

        # Sentiment distribution bar chart
        st.subheader("📊 Sentiment Breakdown")
        sentiment_counts = df['Sentiment'].value_counts()
        fig1, ax1 = plt.subplots()
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax1)
        ax1.set_title("Sentiment Distribution")
        ax1.set_ylabel("Message Count")
        st.pyplot(fig1)

        # Satisfaction score
        st.subheader("⭐ Estimated Customer Satisfaction Score")
        customer_msgs = df[df['Sender'].str.lower() == 'customer']
        pos = customer_msgs[customer_msgs['Sentiment'] == 'Positive'].shape[0]
        neg = customer_msgs[customer_msgs['Sentiment'] == 'Negative'].shape[0]
        total = customer_msgs.shape[0]
        if total > 0:
            satisfaction_score = round(((pos - neg) / total) * 5, 2)
            satisfaction_score = max(0, min(5, satisfaction_score + 2.5))
        else:
            satisfaction_score = "N/A"
        st.markdown(f"<h2 style='color:#FFD700'>{satisfaction_score} / 5</h2>", unsafe_allow_html=True)

        # Key issues extraction
        st.subheader("🧩 Key Issues Detected")
        issue_keywords = ["refund", "delay", "broken", "cancel", "late", "not working", "missing", "bad", "poor", "problem", "error", "fail", "issue"]

        def extract_issues(text):
            found = [word for word in issue_keywords if re.search(r'\\b' + re.escape(word) + r'\\b', text.lower())]
            return ', '.join(found)

        df['Key_Issues'] = df['Message'].apply(extract_issues)
        issues = df['Key_Issues'][df['Key_Issues'] != '']
        with st.expander("🔍 View All Detected Issues"):
            st.write(issues.unique().tolist())

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        all_text = " ".join(df['Message'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)

        # Chat Summary
        st.subheader("🧠 Summary of Conversation")
        with st.spinner("Generating AI summary..."):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            limited_text = all_text[:3000]
            summary = summarizer(limited_text, max_length=130, min_length=30, do_sample=False)
            st.success("Summary generated:")
            st.info(summary[0]['summary_text'])

        # Download
        st.subheader("⬇️ Download Analyzed Chat Log")
        st.download_button(
            label="Download CSV with Sentiment & Issues",
            data=df.to_csv(index=False),
            file_name="analyzed_chat.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ CSV must contain at least 'Sender' and 'Message' columns.")
else:
    st.info("👈 Upload a CSV file to start analyzing.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size: 14px;'>© 2025 Chat Analyzer | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
