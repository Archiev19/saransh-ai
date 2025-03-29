# Saransh AI 📚

An intelligent article reader and summarizer that helps you consume online content more efficiently.

## ✨ Features

- **Dual Summarization Methods**
  - AI-powered BART summarization for concise, rephrased summaries
  - Extractive TextRank summarization for faster, key-sentence extraction
- **Side-by-Side Comparison**
  - Compare both summarization methods
  - Choose the best summary for your needs
- **Clean Reading Experience**
  - Distraction-free article view
  - Proper paragraph formatting
  - Easy navigation between article and summary

## 🚀 How It Works

### AI Summarization (BART)
- Uses Facebook's BART model for human-like summaries
- Understands context and generates concise summaries
- Best for creative content and complex narratives
- Takes slightly longer but provides more refined results

### Extractive Summarization (TextRank)
- Uses TextRank algorithm to identify key sentences
- Maintains original wording for 100% accuracy
- Faster processing time
- Perfect for news articles and factual content

## 💻 Technical Stack

- **Backend**: Python Flask
- **AI Model**: facebook/bart-large-cnn
- **Libraries**:
  - `transformers` for AI summarization
  - `networkx` for TextRank implementation
  - `newspaper3k` for article extraction
  - `nltk` for text processing
  - `beautifulsoup4` for HTML parsing

## 🛠️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/saransh-ai.git
   cd saransh-ai
   ```

2. Create a virtual environment:
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

5. Open your browser and navigate to:
   ```
   http://localhost:5003
   ```

## 🎯 Use Cases

- Research and academic reading
- News article consumption
- Content curation
- Study material preparation
- Quick information extraction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Credits

- Inspired by SMRY.ai
- Built with love by the open-source community
- Special thanks to Hugging Face for the BART model

---

Made with ❤️ for efficient reading 