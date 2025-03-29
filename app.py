import os
import re
import logging
import numpy as np
import networkx as nx
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from dotenv import load_dotenv
import time
import nltk
from newspaper import Article, ArticleException
from sklearn.metrics.pairwise import cosine_similarity
import torch
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Configure app
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # 5 minutes cache
app.config['TEMPLATES_AUTO_RELOAD'] = False
app.jinja_env.cache = {}

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

    def extract_content(self, url):
        """Extract content from URL using newspaper3k with fallback to BeautifulSoup."""
        if not is_valid_url(url):
            raise ValueError('Invalid URL format')
            
        try:
            # Try newspaper3k first
            article = Article(url)
            article.download()
            article.parse()
            
            content = article.text
            title = article.title
            
            if not content:
                logger.warning("newspaper3k extraction failed, trying BeautifulSoup")
                content = self._extract_with_beautifulsoup(url)
                
            if content:
                return self.format_article_text(content), title
                
            raise ValueError('Could not extract content from URL')
            
        except ArticleException as e:
            logger.warning(f"newspaper3k extraction failed: {str(e)}, trying BeautifulSoup")
            content = self._extract_with_beautifulsoup(url)
            if content:
                return content, "Article"
            raise ValueError('Could not extract content from URL')
            
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            raise ValueError('Could not extract content from URL')

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

class ExtractiveTextRankSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        logger.info("Initializing ExtractiveTextRankSummarizer")

    def preprocess_text(self, text):
        """Clean and tokenize the text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Clean and tokenize each sentence
        clean_sentences = []
        sentence_words = []
        
        for sentence in sentences:
            # Clean punctuation and convert to lowercase
            clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            words = word_tokenize(clean_sentence)
            
            # Remove stopwords and short words
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            if words:  # Only keep sentences with meaningful words
                clean_sentences.append(sentence)
                sentence_words.append(words)
        
        return clean_sentences, sentence_words

    def create_similarity_matrix(self, sentence_words):
        """Create similarity matrix between sentences."""
        similarity_matrix = np.zeros((len(sentence_words), len(sentence_words)))
        
        for i in range(len(sentence_words)):
            for j in range(len(sentence_words)):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(sentence_words[i], sentence_words[j])
                    
        return similarity_matrix

    def sentence_similarity(self, words1, words2):
        """Calculate similarity between two sentences using word overlap."""
        words1_set = set(words1)
        words2_set = set(words2)
        
        overlap = words1_set.intersection(words2_set)
        union = words1_set.union(words2_set)
        
        return len(overlap) / (len(union) + 1e-9)  # Add small epsilon to avoid division by zero

    def rank_sentences(self, similarity_matrix):
        """Rank sentences using PageRank algorithm."""
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        return scores

    def select_top_sentences(self, sentences, scores, num_sentences=None):
        """Select top sentences based on their scores."""
        if num_sentences is None:
            # Dynamically determine number of sentences based on input length
            num_sentences = max(3, len(sentences) // 5)
        
        ranked_sentences = [(score, i, sentence) for i, (sentence, score) in enumerate(zip(sentences, scores.values()))]
        ranked_sentences.sort(reverse=True)
        
        # Select top sentences and sort them by original position
        selected = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
        return [sentence for _, _, sentence in selected]

    def process_article(self, text):
        """Generate extractive summary using TextRank algorithm."""
        try:
            if not text or len(text.strip()) < 100:
                logger.warning("Text too short for summarization")
                return text

            logger.info(f"Starting extractive summarization for text of length: {len(text)}")
            
            # Preprocess text
            sentences, sentence_words = self.preprocess_text(text)
            if len(sentences) < 3:
                logger.warning("Too few sentences for meaningful summarization")
                return text
            
            # Create similarity matrix
            similarity_matrix = self.create_similarity_matrix(sentence_words)
            
            # Rank sentences
            sentence_scores = self.rank_sentences(similarity_matrix)
            
            # Select top sentences
            summary_sentences = self.select_top_sentences(sentences, sentence_scores)
            
            # Join sentences into paragraphs
            summary = '\n\n'.join([' '.join(summary_sentences[i:i+3]) 
                                 for i in range(0, len(summary_sentences), 3)])
            
            logger.info(f"Generated extractive summary of length: {len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            return None

# Initialize components
content_extractor = ContentExtractor()
summarizer = ExtractiveTextRankSummarizer()

class HuggingFaceAISummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        logger.info(f"Initializing HuggingFace summarizer with model: {self.model_name}")
        try:
            logger.info("Loading summarization pipeline...")
            self.summarizer = pipeline("summarization", model=self.model_name)
            logger.info("HuggingFace summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace summarizer: {str(e)}")
            raise

    def post_process_summary(self, text):
        """Clean and format the generated summary."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper sentence spacing
        text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', '\n\n', text)
        
        # Fix common issues
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)  # Add space after punctuation
        
        # Capitalize first letter of sentences
        sentences = text.split('\n\n')
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        text = '\n\n'.join(sentences)
        
        return text

    def chunk_text(self, text, max_chunk_size=512):
        """Split text into chunks that fit within model's max token limit."""
        logger.info(f"Chunking text of length {len(text)} with max_chunk_size {max_chunk_size}")
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if sentence_length > max_chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0
                
                words = sentence.split()
                current_part = []
                
                for word in words:
                    current_part.append(word)
                    if len(current_part) >= max_chunk_size:
                        chunk_text = " ".join(current_part)
                        chunks.append(chunk_text)
                        current_part = []
                
                if current_part:
                    current_chunk = current_part
                    current_length = len(current_part)
            
            elif current_length + sentence_length <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks

    def summarize(self, text, max_length=150, min_length=50):
        """Generate AI-powered summary using BART model."""
        try:
            if not text or len(text.strip()) < 100:
                logger.warning("Text too short for AI summarization")
                return text

            logger.info(f"Starting AI summarization for text of length: {len(text)}")
            
            if not hasattr(self, 'summarizer'):
                logger.error("Summarizer not initialized")
                return None
                
            chunks = self.chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            summaries = []
            
            for i, chunk in enumerate(chunks):
                try:
                    if len(chunk.split()) < 10:
                        logger.warning(f"Chunk {i+1} too short, skipping")
                        continue
                        
                    summary = self.summarizer(chunk, 
                                           max_length=max_length,
                                           min_length=min_length,
                                           do_sample=False)
                                           
                    if not summary or not isinstance(summary, list) or len(summary) == 0:
                        logger.error(f"Invalid summary format for chunk {i+1}")
                        continue
                        
                    summary_text = summary[0].get('summary_text')
                    if not summary_text:
                        logger.error(f"No summary text generated for chunk {i+1}")
                        continue
                        
                    summaries.append(summary_text)
                    
                except Exception as e:
                    logger.error(f"Error summarizing chunk {i+1}: {str(e)}")
                    continue
            
            if not summaries:
                logger.error("No summaries were generated for any chunks")
                return None
                
            final_summary = " ".join(summaries)
            final_summary = self.post_process_summary(final_summary)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Error in AI summarization: {str(e)}")
            return None

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
                summary = HuggingFaceAISummarizer().summarize(content)
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