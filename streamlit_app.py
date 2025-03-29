import os
import logging
import requests
from bs4 import BeautifulSoup
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from urllib.parse import urlparse
from newspaper import Article, ArticleException
import time
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import pipeline
import torch

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'article_content' not in st.session_state:
    st.session_state.article_content = None
if 'article_title' not in st.session_state:
    st.session_state.article_title = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Reuse your existing classes (ContentExtractor, ExtractiveTextRankSummarizer, HuggingFaceAISummarizer)
# ... [Keep all your existing class definitions] ...

# Initialize summarizers
@st.cache_resource
def get_summarizers():
    return {
        'extractive': ExtractiveTextRankSummarizer(),
        'ai': HuggingFaceAISummarizer()
    }

# Initialize content extractor
@st.cache_resource
def get_content_extractor():
    return ContentExtractor()

def main():
    st.set_page_config(
        page_title="Saransh AI - Intelligent Article Reader & Summarizer",
        page_icon="📚",
        layout="wide"
    )

    # Header
    st.title("Saransh AI")
    st.markdown("### Intelligent Article Reader & Summarizer")

    # URL Input
    url = st.text_input("Enter article URL:", placeholder="https://example.com/article")
    
    if url:
        with st.spinner("Extracting article content..."):
            try:
                extractor = get_content_extractor()
                content, title = extractor.extract_content(url)
                
                if content:
                    st.session_state.article_content = content
                    st.session_state.article_title = title
                    
                    # Display article content
                    st.subheader("Article Content")
                    st.text_area("", content, height=300)
                    
                    # Summarization options
                    st.subheader("Generate Summary")
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        method = st.selectbox(
                            "Choose summarization method:",
                            ["AI (BART) - More concise, may take longer", "Extractive (TextRank) - Faster, 100% accurate"],
                            format_func=lambda x: x.split(" - ")[0]
                        )
                    
                    with col2:
                        if st.button("Generate Summary", type="primary"):
                            with st.spinner("Generating summary..."):
                                summarizers = get_summarizers()
                                if "AI" in method:
                                    summary = summarizers['ai'].summarize(content)
                                else:
                                    summary = summarizers['extractive'].process_article(content)
                                
                                st.session_state.summary = summary
                    
                    if st.session_state.summary:
                        st.subheader("Summary")
                        st.text_area("", st.session_state.summary, height=200)
                        
                        # Download button
                        st.download_button(
                            label="Download Summary",
                            data=st.session_state.summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("Could not extract content from the provided URL")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Sidebar with information
    with st.sidebar:
        st.header("About Saransh AI")
        st.markdown("""
        Saransh AI is an intelligent article reader and summarizer that helps you:
        
        - 📖 Read articles more efficiently
        - 🤖 Get AI-powered summaries
        - ⚡ Quick extractive summaries
        - 🔄 Compare different summarization methods
        
        Choose between:
        - **AI Summarization**: More concise, human-like summaries
        - **Extractive Summarization**: Faster, 100% accurate summaries
        """)
        
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. Enter an article URL
        2. Read the extracted content
        3. Choose your preferred summarization method
        4. Get your summary instantly
        """)

if __name__ == "__main__":
    main() 