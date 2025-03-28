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

    def extract_content(self, url):
        """Extract article content using Newspaper3k with BeautifulSoup as fallback."""
        try:
            # Validate URL
            if not is_valid_url(url):
                raise ValueError("Invalid URL format")

            # Add scheme if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Try to bypass any paywall and get content
            article_content, article_title = self.bypass_paywall(url)
            
            if not article_content:
                # Fallback to standard extraction methods
                try:
                    content = self._extract_with_newspaper(url)
                    if content:
                        return content, None
                except Exception as e:
                    logger.warning(f"Newspaper3k extraction failed: {str(e)}. Falling back to BeautifulSoup.")

                # Fallback to BeautifulSoup
                content = self._extract_with_beautifulsoup(url)
                if not content:
                    raise ValueError("No content could be extracted from the URL")

                return content, None
            
            return article_content, article_title

        except Exception as e:
            logger.error(f"Error extracting content from URL: {str(e)}")
            raise

    def bypass_paywall(self, url):
        """Attempt to bypass paywalls using various methods."""
        logger.info(f"Attempting to bypass paywall for URL: {url}")
        
        # Method 1: Direct request with modified headers (Googlebot)
        logger.info("Trying Method 1: Direct request with modified headers")
        for user_agent in self.user_agents:
            try:
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                }
                response = requests.get(url, headers=headers, timeout=10)
                logger.info(f"Direct request status code: {response.status_code}")
                
                if response.status_code == 200:
                    logger.info("Got 200 response, attempting to extract content")
                    article = Article(url)
                    article.set_html(response.text)
                    article.parse()
                    
                    content = article.text
                    title = article.title
                    
                    if content and len(content) > 500:  # Consider it successful if content length is substantial
                        logger.info(f"Successfully extracted content with length: {len(content)}")
                        return self.format_article_text(content), title
                    else:
                        logger.info(f"Extracted content length: {len(content) if content else 0}")
                        logger.info("Extracted content too short")
            except Exception as e:
                logger.warning(f"Error with user agent {user_agent}: {str(e)}")
                continue
        
        # Method 2: Try archive.is
        logger.info("Trying Method 2: archive.is")
        try:
            archive_url = f"https://archive.is/{url}"
            logger.info(f"Requesting archive.is URL: {archive_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = requests.get(archive_url, headers=headers, timeout=15)
            logger.info(f"Archive.is status code: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("Got 200 response from archive.is, attempting to extract content")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Archive.is usually puts content in #job-main-source
                article_div = soup.find(id='job-main-source')
                if not article_div:  # Try other common containers
                    article_div = soup.find('article') or soup.find(class_='article-body')
                
                if article_div:
                    content = article_div.get_text(separator=' ', strip=True)
                    title_elem = soup.find('h1')
                    title = title_elem.text.strip() if title_elem else None
                    
                    if content and len(content) > 500:
                        logger.info(f"Successfully extracted content from archive.is with length: {len(content)}")
                        return self.format_article_text(content), title
                    else:
                        logger.info(f"Extracted content length from archive.is: {len(content) if content else 0}")
                        logger.info("Extracted content from archive.is too short")
        except Exception as e:
            logger.warning(f"Error with archive.is: {str(e)}")
        
        # Method 3: Try Google Cache
        logger.info("Trying Method 3: Google cache")
        try:
            cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
            logger.info(f"Requesting Google cache URL: {cache_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = requests.get(cache_url, headers=headers, timeout=10)
            logger.info(f"Google cache status code: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("Got 200 response from Google cache, attempting to extract content")
                article = Article(url)
                article.set_html(response.text)
                article.parse()
                
                content = article.text
                title = article.title
                
                if content and len(content) > 500:
                    logger.info(f"Successfully extracted content from Google cache with length: {len(content)}")
                    return self.format_article_text(content), title
                else:
                    logger.info(f"Extracted content length from Google cache: {len(content) if content else 0}")
                    logger.info("Extracted content from Google cache too short")
        except Exception as e:
            logger.warning(f"Error with Google cache: {str(e)}")
        
        logger.info("No content found through any method")
        return None, None

    def _extract_with_newspaper(self, url):
        """Extract content using Newspaper3k."""
        article = Article(url)
        article.download()
        # Add a small delay to ensure download is complete
        time.sleep(1)
        article.parse()
        
        content = article.text
        if not content:
            return None
            
        # Format and process the content
        return self.format_article_text(content)

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
    
    def clean_text(self, text):
        """Clean and normalize extracted text."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short sentences and normalize
        sentences = sent_tokenize(text)
        cleaned_sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        return ' '.join(cleaned_sentences)
        
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
        logger.info("ExtractiveTextRankSummarizer initialized")

    def preprocess(self, text):
        """Preprocess text into sentences and remove stop words."""
        logger.info(f"Preprocessing text of length: {len(text)}")
        sentences = sent_tokenize(text)
        logger.info(f"Tokenized into {len(sentences)} sentences")
        
        # Clean sentences
        clean_sentences = []
        original_sentences = []
        
        for sentence in sentences:
            # Original sentence for later use
            original_sentences.append(sentence)
            
            # Clean and tokenize
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            clean_sentences.append(words)
            
        logger.info(f"Preprocessing complete. Clean sentences: {len(clean_sentences)}")
        return clean_sentences, original_sentences

    def create_similarity_matrix(self, sentences):
        """Create similarity matrix among all sentences."""
        # Initialize similarity matrix
        logger.info(f"Creating similarity matrix for {len(sentences)} sentences")
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:  # No need to compare a sentence with itself
                    # Count common words as a measure of similarity
                    common_words = len(set(sentences[i]).intersection(set(sentences[j])))
                    
                    # Normalize by sentence lengths to avoid bias towards longer sentences
                    if len(sentences[i]) > 0 and len(sentences[j]) > 0:
                        similarity_matrix[i][j] = common_words / (np.log10(len(sentences[i]) + 1) + np.log10(len(sentences[j]) + 1))
        
        return similarity_matrix

    def summarize(self, text, num_sentences=5):
        """Generate extractive summary using TextRank algorithm."""
        logger.info(f"Generating extractive summary with {num_sentences} sentences")

        # Check if there's enough text to summarize
        if not text or len(text.strip()) < 100:
            logger.warning("Text too short for summarization")
            return text
            
        try:
            # Preprocess the text
            clean_sentences, original_sentences = self.preprocess(text)
            
            # Check if we have enough sentences
            if len(clean_sentences) <= num_sentences:
                logger.warning(f"Only {len(clean_sentences)} sentences found, returning all of them")
                return text
                
            # Create similarity matrix
            similarity_matrix = self.create_similarity_matrix(clean_sentences)
            
            # Apply PageRank algorithm
            logger.info("Applying PageRank algorithm")
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100)
            
            # Get top sentences based on scores
            logger.info("Ranking sentences")
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(original_sentences)), reverse=True)
            
            # Select top N sentences
            top_sentences = [ranked_sentences[i][2] for i in range(min(num_sentences, len(ranked_sentences)))]
            
            # Sort sentences by their original order in the text to maintain flow
            logger.info("Ordering sentences by original position")
            ordered_sentences = [(original_sentences.index(s), s) for s in top_sentences]
            ordered_sentences.sort()
            
            # Join the selected sentences
            summary = " ".join([s[1] for s in ordered_sentences])
            logger.info(f"Summary generated with {len(summary)} characters")
            
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            # Return a portion of the original text as fallback
            sentences = sent_tokenize(text)
            if len(sentences) > num_sentences:
                return " ".join(sentences[:num_sentences])
            return text

    def process_article(self, text, max_length=1500):
        """Process a full article for summarization."""
        logger.info(f"Processing article of length: {len(text)}")
        
        try:
            # Calculate the number of sentences to extract based on the length of the text
            total_sentences = len(sent_tokenize(text))
            
            # Determine number of sentences for the summary (10-20% of original)
            num_sentences = max(5, min(int(total_sentences * 0.15), 15))
            
            logger.info(f"Article has {total_sentences} sentences, extracting {num_sentences} key sentences")
            
            # Generate the summary
            summary = self.summarize(text, num_sentences=num_sentences)
            
            # Check if summary generation was successful
            if not summary or len(summary) < 100:
                logger.warning("Summary too short, using fallback method")
                sentences = sent_tokenize(text)
                selected = []
                # Take every 5th sentence for a simple extractive summary
                for i in range(0, len(sentences), max(1, len(sentences) // num_sentences)):
                    if len(selected) < num_sentences and i < len(sentences):
                        selected.append(sentences[i])
                summary = " ".join(selected)
            
            # Further process the summary for readability
            processed_summary = self.post_process_summary(summary)
            
            logger.info(f"Final summary length: {len(processed_summary)} characters")
            return processed_summary
        except Exception as e:
            logger.error(f"Error in process_article: {str(e)}")
            # Fallback to a simple extraction of the first few sentences
            try:
                sentences = sent_tokenize(text)
                simple_summary = " ".join(sentences[:min(10, len(sentences))])
                return simple_summary
            except:
                # Last resort
                return text[:1000] + "..."

    def post_process_summary(self, summary):
        """Improve the readability of the summary."""
        try:
            # Fix spacing issues
            summary = re.sub(r'\s+', ' ', summary).strip()
            summary = re.sub(r' \.', '.', summary)
            summary = re.sub(r' ,', ',', summary)
            
            # Break very long summaries into paragraphs for readability
            sentences = sent_tokenize(summary)
            if len(sentences) > 5:
                # Create paragraphs with roughly 3-5 sentences each
                paragraphs = []
                para_size = min(5, max(3, len(sentences) // 3))
                
                for i in range(0, len(sentences), para_size):
                    paragraph = " ".join(sentences[i:i+para_size])
                    paragraphs.append(paragraph)
                    
                return "\n\n".join(paragraphs)
            
            return summary
        except Exception as e:
            logger.error(f"Error in post_process_summary: {str(e)}")
            return summary

app = Flask(__name__)
content_extractor = ContentExtractor()
summarizer = ExtractiveTextRankSummarizer()

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
            
            # Generate extractive summary
            logger.info(f"Starting summarization process for content of length: {len(content)}")
            summary = summarizer.process_article(content)
            logger.info(f"Generated summary with {len(summary)} characters")
            
            # Validate summary
            if not summary or len(summary.strip()) < 50:
                logger.warning("Summary generation produced insufficient results, using fallback")
                # Fallback to simple extraction of first 3 sentences
                sentences = sent_tokenize(content)
                summary = " ".join(sentences[:min(3, len(sentences))])
                logger.info(f"Generated fallback summary with {len(summary)} characters")
                
            if not summary:
                logger.error("All summary generation methods failed")
                return jsonify({'error': 'Failed to generate a summary. Please try again.'}), 500
            
            # Add debug info in development mode
            response_data = {
                'summary': summary,
            }
            
            if app.debug:
                response_data['debug'] = {
                    'summary_length': len(summary),
                    'content_length': len(content),
                    'paragraphs': len(summary.split('\n\n')),
                    'sentences': len(sent_tokenize(summary))
                }
                
            logger.info(f"Returning summary response: {len(summary)} chars, {response_data.get('debug', {}).get('paragraphs', 0)} paragraphs")
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
    app.run(host='0.0.0.0', port=5003, debug=True) 