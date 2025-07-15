import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import re

# ---- Basic Login System ----
st.set_page_config(page_title="Chat Transcript Analyzer", layout="wide", page_icon="💬")

# Hardcoded credentials (you can change or load from a file/db)
users = {"admin": "admin123", "user": "password"}

# Track login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login form
if not st.session_state.logged_in:
    st.title("🔐 Login to Chat Transcript Analyzer")
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")
        if login_btn:
            if users.get(username) == password:
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")
    st.stop()  # Stop rest of app if not logged in
