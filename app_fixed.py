import os
import logging
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
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

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

class ContentExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.user_agents = [
            'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
            'Twitterbot/1.0'
        ]

    def _extract_with_beautifulsoup(self, url):
        """Extract content using BeautifulSoup (fallback method)."""
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
            
            # Return None if no content was found
            if not content:
                return None
            
            # Process and return the content
            return self.format_article_text(content)
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed: {str(e)}")
            return None 