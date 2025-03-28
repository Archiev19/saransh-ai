# Saransh AI 📚

> An intelligent article reader and summarizer that makes reading easier and more efficient.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

## 🌟 Features

- **Smart Article Reading**: Access and read articles from any website, including those behind paywalls
- **Intelligent Summarization**: Get concise, accurate summaries using advanced TextRank algorithm
- **Universal Compatibility**: Works with any news source, blog, or article website
- **Clean Reading Experience**: Beautiful, distraction-free interface with perfect readability
- **No API Keys Required**: Works out of the box without any external API dependencies

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/saransh-ai.git
   cd saransh-ai
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   - Open your browser and go to `http://localhost:5003`
   - Enter any article URL and start reading!

## 💡 How It Works

Saransh AI uses a sophisticated extractive summarization approach based on the TextRank algorithm:

1. **Content Extraction**: Uses Newspaper3k with BeautifulSoup fallback to extract article content
2. **Intelligent Processing**: Analyzes sentence importance using TextRank (similar to Google's PageRank)
3. **Smart Selection**: Picks the most relevant ~15% of sentences while maintaining context
4. **Clean Formatting**: Organizes content into readable paragraphs with proper spacing

## 🎯 Use Cases

- **Research & Analysis**: Quickly understand long research papers or reports
- **News Reading**: Stay updated with news without spending hours reading
- **Content Research**: Efficiently process multiple articles for content creation
- **Academic Study**: Grasp key concepts from academic papers faster
- **Business Intelligence**: Quick analysis of industry articles and reports

## 🛠️ Technical Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, TailwindCSS, JavaScript
- **NLP**: NLTK, NetworkX
- **Content Processing**: Newspaper3k, BeautifulSoup4
- **Text Analysis**: TextRank Algorithm

## 🎨 Features in Detail

### 1. Article Access
- Intelligent paywall bypass techniques
- Multiple content extraction methods
- Fallback mechanisms for complex websites

### 2. Text Processing
- Advanced sentence tokenization
- Sophisticated text cleaning
- Smart paragraph restructuring

### 3. Summarization
- TextRank-based extractive summarization
- Context-aware sentence selection
- Intelligent content organization

### 4. User Interface
- Clean, modern design
- Dark/light mode support
- Responsive layout
- Distraction-free reading

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💝 Credits

This project was inspired by [SMRY.AI](https://smry.ai/), an excellent article summarization tool. While we've implemented our own approach using TextRank algorithm and added features like paywall bypass, the initial inspiration for the user interface and workflow came from SMRY.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TextRank Algorithm by Rada Mihalcea and Paul Tarau
- SMRY.AI for interface inspiration
- Flask framework and its community
- NLTK and NetworkX libraries
- All our contributors and users

---

Made with ❤️ by [Your Name] 