# Saransh AI User Guide 📖

Welcome to Saransh AI! This guide will help you make the most of our intelligent article reader and summarizer.

## Table of Contents
- [Getting Started](#getting-started)
- [Basic Features](#basic-features)
- [Summarization Methods](#summarization-methods)
- [Advanced Features](#advanced-features)
- [Tips & Tricks](#tips--tricks)
- [Troubleshooting](#troubleshooting)
- [FAQs](#faqs)

## Getting Started

### What is Saransh AI?
Saransh AI is your intelligent reading companion that helps you:
- Read articles from any website (even paywalled ones!)
- Get smart summaries that capture the main points
- Enjoy a clean, distraction-free reading experience

### Accessing Saransh AI
1. Open your web browser
2. Go to `http://localhost:5003` (or your configured address)
3. You'll see a clean interface with a URL input box

## Basic Features

### 1. Reading Articles
1. **Paste any article URL** in the input box
2. Click "**Read Article**"
3. The article will appear in a clean, readable format
4. Use the back button anytime to read a different article

### 2. Getting Summaries
1. While reading an article, click "**Generate Summary**"
2. The summary will appear in a new tab
3. Switch between article and summary using the tabs
4. Copy the summary with one click using the copy button

### 3. Clean Reading Mode
- Clear, properly spaced paragraphs
- Perfect contrast for easy reading
- No ads, popups, or distractions
- Responsive design that works on all devices

## Summarization Methods

### 1. AI Summarization (BART)
- **What**: Uses Facebook's BART model for human-like summaries
- **Best For**: 
  - Creative content
  - Complex narratives
  - When you want concise, rephrased content
- **Features**:
  - Natural language generation
  - Context understanding
  - Creative rephrasing

### 2. Extractive Summarization (TextRank)
- **What**: Uses TextRank algorithm to extract key sentences
- **Best For**:
  - News articles
  - Technical content
  - When 100% accuracy is crucial
- **Features**:
  - Original sentence preservation
  - Fast processing
  - Perfect factual accuracy

### Comparison Feature
- Click "Compare Methods" to see both summaries side by side
- Easy visual comparison of different approaches
- Choose the summary that best suits your needs

## Advanced Features

### 1. Paywall Bypass
Saransh AI can access content from paywalled sites using:
- Smart user agent switching
- Archive.is integration
- Google cache access
- Multiple fallback methods

### 2. Smart Summarization
Our TextRank-based summarization:
- Identifies key sentences automatically
- Maintains original context and flow
- Preserves factual accuracy
- Creates well-structured summaries

### 3. Content Processing
- Automatic paragraph detection
- Smart sentence grouping
- Proper formatting preservation
- Clean text extraction

## Tips & Tricks

### For Better Summaries
- Longer articles (>1000 words) work best
- News articles and research papers give great results
- Technical content is handled well
- Multiple languages are supported

### For Better Reading
- Use full-screen mode for best experience
- Adjust your browser zoom if needed
- Use browser reading mode for extra comfort
- Save summaries for later reference

## Troubleshooting

### Common Issues

1. **URL Not Working**
   - Check if the URL is complete (including https://)
   - Try opening the URL in a new tab first
   - Check if the website is accessible in your region

2. **Summary Not Generating**
   - Ensure the article has enough content
   - Try refreshing the page
   - Check your internet connection

3. **Content Not Loading**
   - Verify the website is online
   - Try a different browser
   - Check your firewall settings

### When to Contact Support
- If issues persist after trying solutions
- For feature requests or suggestions
- To report bugs or unexpected behavior

## FAQs

### General Questions

**Q: Is Saransh AI free to use?**
A: Yes, it's completely free and open source.

**Q: Do I need an API key?**
A: No, Saransh AI works without any API keys.

**Q: What types of articles work best?**
A: News articles, blog posts, research papers, and long-form content work great.

**Q: Which summarization method should I use?**
A: Use AI (BART) for creative content and when you want concise, rephrased summaries. Use Extractive (TextRank) for news and when factual accuracy is crucial.

**Q: How long does summarization take?**
A: Extractive summarization is almost instant. AI summarization may take a few seconds depending on article length.

**Q: Can I use both methods?**
A: Yes! Use the "Compare Methods" feature to see both summaries side by side.

### Technical Questions

**Q: How does the summarization work?**
A: We use the TextRank algorithm to identify and extract key sentences while maintaining context.

**Q: Is my data private?**
A: Yes, all processing is done locally on your machine. We don't store any articles or summaries.

**Q: Can I use it offline?**
A: You need internet to fetch articles, but the summarization works locally.

**Q: Does it work with all websites?**
A: Yes, Saransh AI works with most websites, including those with paywalls.

**Q: Is the AI summary always different from the extractive one?**
A: Yes, while they may cover similar points, the AI summary rephrases content while extractive uses original sentences.

### Feature Questions

**Q: Can I save summaries?**
A: Yes, use the copy button to save summaries to your clipboard.

**Q: Does it work with PDFs?**
A: Currently, we support web articles only. PDF support is planned for future releases.

**Q: Which method is more accurate?**
A: Extractive is 100% factually accurate as it uses original sentences. AI provides more concise, rephrased summaries while maintaining accuracy.

## Need More Help?

- Check our [GitHub repository](https://github.com/yourusername/saransh-ai)
- Open an issue for bugs or features
- Contact us through GitHub discussions

---

Remember, Saransh AI is continuously improving. Your feedback helps us make it better! 