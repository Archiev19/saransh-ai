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
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.probability import FreqDist
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.question_starters = {
            'what': ['what', 'which', 'whose'],
            'how': ['how'],
            'why': ['why'],
            'when': ['when'],
            'where': ['where'],
            'who': ['who', 'whom']
        }
    
    def analyze_text(self, text):
        """Analyze text and return various metrics."""
        try:
            # Basic metrics
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum()]
            
            # Calculate metrics
            word_count = len(words)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Calculate readability (Flesch Reading Ease)
            blob = TextBlob(text)
            syllable_count = sum([self._count_syllables(word) for word in words])
            if sentence_count == 0 or word_count == 0:
                readability_score = 0
            else:
                readability_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
                readability_score = max(0, min(100, readability_score))  # Clamp between 0 and 100
            
            # Extract keywords
            keywords = self._extract_keywords(text)
            keyword_frequencies = Counter(word.lower() for word in words if word.lower() in keywords)
            
            # Estimate reading time (assuming 200 words per minute)
            reading_time = math.ceil(word_count / 200)
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': round(avg_sentence_length, 1),
                'readability_score': round(readability_score, 1),
                'estimated_read_time': reading_time,
                'keywords': list(keywords),
                'keyword_frequencies': dict(keyword_frequencies)
            }
        except Exception as e:
            logger.error(f"Error in analyze_text: {str(e)}")
            return None
    
    def _count_syllables(self, word):
        """Count the number of syllables in a word."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count
    
    def _extract_keywords(self, text, top_n=10):
        """Extract keywords using TF-IDF."""
        try:
            # Tokenize and clean text
            sentences = sent_tokenize(text)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=50
            )
            
            # Fit and transform the text
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort keywords by score
            keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, _ in keywords[:top_n]]
        except Exception as e:
            logger.error(f"Error in _extract_keywords: {str(e)}")
            return []

class ContentExtractor:
    def __init__(self):
        self.analyzer = TextAnalyzer()
    
    def extract_content(self, url):
        """Extract content from a URL."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Get the main content
            content = article.text
            if not content:
                content = self._extract_with_beautifulsoup(url)
            
            if not content:
                return None
            
            # Analyze the content
            analytics = self.analyzer.analyze_text(content)
            
            return {
                'title': article.title,
                'content': content,
                'analytics': analytics
            }
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return None
    
    def _extract_with_beautifulsoup(self, url):
        """Fallback method to extract content using BeautifulSoup."""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()
            
            # Get the main content
            content = ' '.join(p.get_text().strip() for p in soup.find_all('p'))
            return content if content else None
        except Exception as e:
            logger.error(f"Error in BeautifulSoup extraction: {str(e)}")
            return None

class FAQGenerator:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def generate_faqs(self, text, num_questions=5):
        """Generate FAQs from the given text."""
        try:
            # Split text into sentences
            sentences = sent_tokenize(text)
            
            # Score sentences for FAQ potential
            scored_sentences = self._score_sentences(sentences)
            
            # Select top sentences
            top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:num_questions]
            
            # Generate questions from selected sentences
            faqs = []
            for sentence, _ in top_sentences:
                question = self._generate_question(sentence)
                if question:
                    faqs.append({
                        'question': question,
                        'answer': sentence
                    })
            
            return faqs[:num_questions]
        except Exception as e:
            logger.error(f"Error in generate_faqs: {str(e)}")
            return []
    
    def _score_sentences(self, sentences):
        """Score sentences based on their potential for FAQ generation."""
        scored_sentences = []
        
        for sentence in sentences:
            score = 0
            words = word_tokenize(sentence.lower())
            
            # Length score (prefer medium-length sentences)
            length = len(words)
            if 8 <= length <= 20:
                score += 3
            elif 5 <= length <= 25:
                score += 2
            
            # Contains numbers
            if any(char.isdigit() for char in sentence):
                score += 2
            
            # Contains key phrases
            key_phrases = ['important', 'main', 'key', 'significant', 'primary', 'essential']
            if any(phrase in words for phrase in key_phrases):
                score += 2
            
            # Sentence starts with potential question words
            first_word = words[0] if words else ''
            if first_word in ['this', 'these', 'that', 'those', 'the', 'a', 'an']:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        return scored_sentences
    
    def _generate_question(self, sentence):
        """Generate a question from a sentence."""
        try:
            # Tokenize and tag parts of speech
            words = word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            
            # Determine question type based on content
            if any(tag in ['CD'] for word, tag in pos_tags):
                return f"What is the {self._get_subject(pos_tags)} mentioned in this statement?"
            
            if any(tag.startswith('VB') for word, tag in pos_tags):
                return f"How does {self._get_subject(pos_tags)} work?"
            
            if any(word.lower() in ['because', 'since', 'therefore'] for word, _ in pos_tags):
                return f"Why is {self._get_subject(pos_tags)} important?"
            
            # Default question format
            return f"What does this statement tell us about {self._get_subject(pos_tags)}?"
        except Exception as e:
            logger.error(f"Error in _generate_question: {str(e)}")
            return None
    
    def _get_subject(self, pos_tags):
        """Extract the main subject from POS tags."""
        for word, tag in pos_tags:
            if tag.startswith('NN'):  # Noun
                return word.lower()
        return "this topic"

class HuggingFaceAISummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.analyzer = TextAnalyzer()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def summarize(self, text, params=None):
        """Generate a summary using the BART model."""
        try:
            if not text:
                return None
            
            # Set default parameters if none provided
            if not params:
                params = {
                    'compression_ratio': 0.3,
                    'min_sentences': 3,
                    'max_sentences': 10,
                    'include_analytics': True
                }
            
            # Calculate target length based on compression ratio
            target_length = int(len(text.split()) * params['compression_ratio'])
            target_length = max(30, min(target_length, 150))  # Keep between 30 and 150 words
            
            # Tokenize and generate summary
            inputs = self.tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')
            inputs = inputs.to(self.device)
            
            summary_ids = self.model.generate(
                inputs['input_ids'],
                num_beams=4,
                min_length=target_length // 2,
                max_length=target_length,
                length_penalty=2.0,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Get analytics if requested
            analytics = None
            if params.get('include_analytics'):
                analytics = self.analyzer.analyze_text(summary)
                if analytics:
                    analytics['compression_ratio'] = len(summary.split()) / len(text.split())
            
            return {
                'summary': summary,
                'analytics': analytics
            }
        except Exception as e:
            logger.error(f"Error in AI summarization: {str(e)}")
            return None

class ExtractiveTextRankSummarizer:
    def __init__(self):
        self.analyzer = TextAnalyzer()
    
    def process_article(self, text, params=None):
        """Generate a summary using TextRank algorithm."""
        try:
            if not text:
                return None
            
            # Set default parameters if none provided
            if not params:
                params = {
                    'compression_ratio': 0.3,
                    'min_sentences': 3,
                    'max_sentences': 10,
                    'include_analytics': True
                }
            
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) < params['min_sentences']:
                return {
                    'summary': text,
                    'analytics': self.analyzer.analyze_text(text) if params.get('include_analytics') else None
                }
            
            # Calculate number of sentences for summary
            num_sentences = max(
                params['min_sentences'],
                min(
                    params['max_sentences'],
                    int(len(sentences) * params['compression_ratio'])
                )
            )
            
            # Create sentence vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate similarity matrix
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            
            # Calculate sentence scores
            scores = np.sum(similarity_matrix, axis=1)
            ranked_sentences = [(score, idx, sentence) 
                              for idx, (score, sentence) 
                              in enumerate(zip(scores, sentences))]
            
            # Sort and select top sentences
            ranked_sentences.sort(reverse=True)
            selected_sentences = sorted(
                [(idx, sentence) 
                 for _, idx, sentence 
                 in ranked_sentences[:num_sentences]],
                key=lambda x: x[0]
            )
            
            # Combine sentences
            summary = ' '.join(sentence for _, sentence in selected_sentences)
            
            # Get analytics if requested
            analytics = None
            if params.get('include_analytics'):
                analytics = self.analyzer.analyze_text(summary)
                if analytics:
                    analytics['compression_ratio'] = len(summary.split()) / len(text.split())
            
            return {
                'summary': summary,
                'analytics': analytics
            }
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            return None

# Initialize components
content_extractor = ContentExtractor()
ai_summarizer = HuggingFaceAISummarizer()
extractive_summarizer = ExtractiveTextRankSummarizer()
faq_generator = FAQGenerator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/extract', methods=['POST'])
def extract():
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        result = content_extractor.extract_content(url)
        if not result:
            return jsonify({'error': 'Failed to extract content'}), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in extract endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the URL'}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        content = data.get('content')
        method = data.get('method', 'extractive')
        params = data.get('params', {})
        
        if not content:
            return jsonify({'error': 'Content is required'}), 400
        
        if method == 'ai':
            result = ai_summarizer.summarize(content, params)
        else:
            result = extractive_summarizer.process_article(content, params)
        
        if not result:
            return jsonify({'error': 'Failed to generate summary'}), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while generating the summary'}), 500

@app.route('/api/generate-faqs', methods=['POST'])
def generate_faqs():
    try:
        data = request.json
        content = data.get('content')
        num_questions = data.get('num_questions', 5)
        
        if not content:
            return jsonify({'error': 'Content is required'}), 400
        
        faqs = faq_generator.generate_faqs(content, num_questions)
        if not faqs:
            return jsonify({'error': 'Failed to generate FAQs'}), 400
        
        return jsonify({'faqs': faqs})
    except Exception as e:
        logger.error(f"Error in generate-faqs endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while generating FAQs'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False) 