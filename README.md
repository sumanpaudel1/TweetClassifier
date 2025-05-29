# Tweet Sentiment Classifier for Hate Speech Detection 🚀

Welcome to the **TweetClassifier** project! This is a deep learning-based application designed to classify tweets into three categories: **Hate Speech**, **Offensive Language**, or **Neither**. The goal is to help social media platforms automate the detection of toxic content for early moderation. Built with Python, Keras, and various NLP tools, this project features two models and detailed analysis of their performance.

---

## 📖 Project Overview

Social media platforms face a significant challenge with harmful content. This project aims to tackle that by using NLP and deep learning to classify tweets effectively. I explored two models—**Simple RNN** and **Enhanced LSTM with GloVe**—to analyze tweet sentiment and detect hate speech.

### Key Features
- **Two Models**: Simple RNN and Enhanced LSTM with GloVe embeddings.
- **Data Balancing**: Tackled class imbalance using downsampling and SMOTE with cosine similarity mapping.
- **Comprehensive Evaluation**: Analyzed performance with confusion matrices, accuracy, and F1-scores.

---

## 📊 Models and Results

I trained two models with the following results, evaluated using `confusion_matrix`, `accuracy_score`, and `classification_report`:

### Simple RNN
- **Architecture**: Basic recurrent layer with EarlyStopping to prevent overfitting.
- **Test Accuracy**: 90%
- **F1-Scores**:
  - Hate: 0.91
  - Offensive: 0.89
  - Neither: 0.91

### Enhanced LSTM with GloVe
- **Architecture**: LSTM layers with GloVe 300d embeddings (Common Crawl 840B dataset), tuned with dropout and layer adjustments.
- **Test Accuracy**: 91% (training accuracy peaked at 95%)
- **F1-Scores**:
  - Hate: 0.91
  - Offensive: 0.89
  - Neither: 0.93

#### Visuals
Below are some visuals showcasing the models’ performance:  
- **Training Curves**: [Link to Curves](#)  
- **Accuracy Bar**: [Link to Bar](#)  
- **F1 Comparison**: [Link to F1 Comparison](#)  
- **Summary Table**: [Link to Table](#)  

*Note*: Replace the `[Link to ...]` placeholders with actual Google Drive links to your images (`all_models_comparison_curves_final.png`, `all_models_accuracy_bar_corrected.png`, `all_models_f1_score_comparison.png`, `all_models_summary_table_final.png`).

---

## 📋 Methodology

### 1. Data Preprocessing
- Used `nltk` for stopword removal, lemmatization, and tokenization on the dataset (`hatevsoffensive_language.csv`).
- Generated word clouds to explore patterns, identifying key terms like "hate" in negative contexts.

### 2. Handling Class Imbalance
- **Downsampling**: Reduced the "Neither" class using TF-IDF and cosine similarity to remove redundant samples.
- **SMOTE**: Generated synthetic samples for "Hate" and "Offensive" classes with SMOTE, then mapped them back to real tweets using cosine similarity and `joblib` for efficiency.

### 3. Model Training
- Tokenized and padded sequences using Keras.
- Applied EarlyStopping to prevent overfitting.
- Used GloVe embeddings (300d vectors) for the LSTM model to capture word relationships.

---

## 🌟 Key Learnings
- **Data Preparation is Crucial**: Imbalanced datasets require careful handling to ensure fair model training.
- **SMOTE with Cosine Similarity**: Mapping synthetic data back to real samples preserved semantic meaning.
- **Power of GloVe**: Pre-trained embeddings significantly improved performance by understanding word relationships.

---

## 👥 Contributors
A big thanks to my friends for their encouragement:  
- **Aarav** (replace with actual name)  
- **Priya** (replace with actual name)

---

## 🙏 Acknowledgments
- [GloVe](https://nlp.stanford.edu/projects/glove/) for the pre-trained embeddings.
- The open-source community for tools like `nltk`, Keras, and SMOTE.

---

## 📬 Get in Touch
I’d love to hear your feedback or collaborate on similar projects!  
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)  
- **Email**: your-email@example.com

Feel free to open issues or submit pull requests if you’d like to contribute!

---

**License**: MIT License  
**Created**: May 2025
