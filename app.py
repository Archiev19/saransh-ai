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

    def format_article_text(self, text):
        """Format article text to preserve paragraph structure and improve readability."""
        # Step 1: Identify paragraphs by looking for natural breaks 
        # First, normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        
        # Replace multiple newlines with a special token
        text = re.sub(r'\n\s*\n', '[PARAGRAPH_BREAK]', text)
        # Replace single newlines with spaces
        text = re.sub(r'\n', ' ', text)
        # Restore paragraph breaks
        text = re.sub(r'\[PARAGRAPH_BREAK\]', '\n\n', text)
        
        # Step 2: Break up very long paragraphs (more than 5 sentences)
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            sentences = sent_tokenize(paragraph)
            
            # If paragraph is very long, break it up
            if len(sentences) > 5:
                curr_para = []
                for i, sentence in enumerate(sentences):
                    curr_para.append(sentence)
                    if (i + 1) % 4 == 0 and i < len(sentences) - 1:  # Group roughly 4 sentences per paragraph
                        formatted_paragraphs.append(' '.join(curr_para))
                        curr_para = []
                
                if curr_para:  # Add any remaining sentences
                    formatted_paragraphs.append(' '.join(curr_para))
            else:
                formatted_paragraphs.append(paragraph)
        
        # Join paragraphs with double newlines
        formatted_text = '\n\n'.join(formatted_paragraphs)
        
        # Clean up whitespace
        formatted_text = re.sub(r' +', ' ', formatted_text)
        formatted_text = re.sub(r'\n +', '\n', formatted_text)
        
        return formatted_text

app = Flask(__name__)
content_extractor = ContentExtractor()
summarizer = ExtractiveTextRankSummarizer()
ai_summarizer = HuggingFaceAISummarizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/article', methods=['POST'])
def fetch_article():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        try:
            # Extract content from URL with title
            content, title = content_extractor.extract_content(url)
            if not content:
                return jsonify({'error': 'Could not extract content from the provided URL'}), 400
                
            logger.info(f"Extracted {len(content)} characters of content from URL")
            
            return jsonify({
                'article': {
                    'title': title or "Article",
                    'content': content,
                    'url': url
                }
            })
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error extracting article: {str(e)}")
            return jsonify({'error': 'An error occurred while fetching the article'}), 500
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Invalid request format'}), 400

@app.route('/api/summarize', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()
        logger.info(f"Received summary request with data keys: {list(data.keys() if data else [])}")
        
        url = data.get('url')
        content = data.get('content')
        method = data.get('method', 'extractive')  # Default to extractive
        
        if not (url or content):
            logger.error("Missing required parameters: neither URL nor content provided")
            return jsonify({'error': 'Either URL or content is required'}), 400
        
        try:
            if url and not content:
                # Extract content from URL
                logger.info(f"Extracting content from URL: {url}")
                content, title = content_extractor.extract_content(url)
                if not content:
                    logger.error(f"Failed to extract content from URL: {url}")
                    return jsonify({'error': 'Could not extract content from the provided URL'}), 400
                
                logger.info(f"Extracted {len(content)} characters of content from URL")
            else:
                logger.info(f"Using provided content of length: {len(content) if content else 0}")
            
            # Validate content
            if not content or len(content.strip()) < 100:
                logger.error(f"Content too short for summarization: {len(content) if content else 0} chars")
                return jsonify({'error': 'Content too short for summarization'}), 400
            
            # Generate summary based on method
            logger.info(f"Starting {method} summarization process")
            if method.lower() == 'ai':
                summary = ai_summarizer.summarize(content)
                if not summary:
                    logger.warning("AI summarization failed, falling back to extractive")
                    summary = summarizer.process_article(content)
            else:
                summary = summarizer.process_article(content)
            
            logger.info(f"Generated summary with {len(summary)} characters")
            
            # Validate summary
            if not summary or len(summary.strip()) < 50:
                logger.error("Summary generation failed")
                return jsonify({'error': 'Failed to generate a summary. Please try again.'}), 500
            
            # Add debug info in development mode
            response_data = {
                'summary': summary,
                'method_used': method.lower()
            }
            
            if app.debug:
                response_data['debug'] = {
                    'summary_length': len(summary),
                    'content_length': len(content),
                    'paragraphs': len(summary.split('\n\n')),
                    'sentences': len(sent_tokenize(summary))
                }
            
            logger.info(f"Returning summary response: {len(summary)} chars using {method} method")
            return jsonify(response_data)
            
        except ValueError as e:
            logger.error(f"Value error in summary generation: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return jsonify({'error': 'An error occurred while generating the summary'}), 500
        
    except Exception as e:
        logger.error(f"Error processing summary request: {str(e)}", exc_info=True)
        return jsonify({'error': 'Invalid request format'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False) 