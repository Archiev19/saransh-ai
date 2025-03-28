# Saransh AI - Intelligent Article Reader & Summarizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

Saransh AI is a powerful article reading and summarization platform that combines advanced AI with traditional summarization techniques to provide the best of both worlds. It offers dual summarization methods - AI-powered BART and extractive TextRank - allowing users to choose the approach that best suits their needs.

## ✨ Features

- **Dual Summarization Methods**
  - 🤖 **AI-Powered BART**: Generate concise, human-like summaries using Facebook's BART model
  - ⚡ **TextRank Algorithm**: Extract key sentences for 100% factual accuracy
  - 🔄 **Side-by-Side Comparison**: Compare both methods to choose the best summary

- **Smart Article Reading**
  - 🔓 Intelligent paywall bypass capabilities
  - 📱 Clean reading mode for distraction-free experience
  - 🌐 Universal compatibility with most news and article websites

- **Modern Interface**
  - 🎨 Beautiful, responsive design
  - ⚡ Real-time summary generation
  - 📋 One-click copy functionality
  - 🔄 Compare summaries side by side

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/saransh-ai.git
   cd saransh-ai
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and visit:
   ```
   http://localhost:5003
   ```

## 🛠️ How It Works

Saransh AI offers two powerful summarization methods:

### 1. AI Summarization (BART)
- Uses Facebook's BART (Bidirectional and Auto-Regressive Transformers) model
- Generates concise, human-like summaries
- Understands context and can rephrase content
- Perfect for creative and narrative content

### 2. Extractive Summarization (TextRank)
- Based on Google's PageRank algorithm
- Selects the most important sentences from the text
- Ensures 100% factual accuracy
- Ideal for news articles and technical content

### Content Processing
1. **Article Extraction**: Uses advanced tools to bypass paywalls and extract clean content
2. **Text Processing**: Handles various formats and structures
3. **Summary Generation**: Choose between AI or extractive methods
4. **Comparison**: View both summaries side by side to pick the best one

## 🎯 Use Cases

- **Research**: Quickly understand academic papers and research articles
- **News Reading**: Stay updated with concise summaries of news articles
- **Content Research**: Efficiently process multiple articles for content creation
- **Academic Study**: Summarize study materials and research papers
- **Business Intelligence**: Quick insights from industry articles and reports

## 🔧 Technical Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, TailwindCSS, JavaScript
- **NLP**: NLTK, Transformers (Hugging Face)
- **AI Model**: facebook/bart-large-cnn
- **Content Processing**: Newspaper3k, BeautifulSoup4

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 💡 Credits

This project was inspired by SMRY.ai and builds upon their innovative approach to article summarization. We've enhanced the concept by adding AI-powered summarization and comparison features.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SMRY.ai for the original inspiration
- Hugging Face for the BART model
- The open-source NLP community

---

Made with ❤️ by [Your Name] 