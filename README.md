# 🧠 Microblogging Sentiment Analysis

A machine learning project to analyze and classify the sentiment of microblog posts (e.g., tweets) using natural language processing (NLP) techniques and state-of-the-art ML models.

---

## 🚀 Features

- ✅ Cleaned and preprocessed real-time tweet data (stopword removal, lemmatization, etc.)
- ✅ Vectorized text using TF-IDF and Word Embeddings
- ✅ Trained sentiment classifiers: Logistic Regression, SVM, and Random Forest
- ✅ Optional: Transformer-based models (BERT) for deeper context understanding
- ✅ Real-time sentiment tracking of trending hashtags
- ✅ Visualized results using Matplotlib/Seaborn or Streamlit dashboard

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Libraries**: scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn  
- **Optional (Advanced)**: HuggingFace Transformers, BERT  
- **Deployment (Optional)**: Streamlit / Flask  

---

## 📊 Dataset

- Collected live tweets using Tweepy API  
- Also tested on benchmark datasets like Sentiment140, Kaggle Tweet Sentiment Datasets

---

## 📈 Results

- Traditional ML models achieved up to **85% accuracy** on test sets  
- BERT-based model improved sentiment precision, especially in sarcasm or nuanced cases  
- Real-time visualization dashboard provided interactive sentiment tracking

---

## 📦 How to Run

```bash
git clone https://github.com/karthikeyan-2023/Microblogging-sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
python train_model.py
