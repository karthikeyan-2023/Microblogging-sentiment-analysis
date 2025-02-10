import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your trained model (assuming you have saved it using joblib or pickle)
import joblib
model = joblib.load('twitter_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to preprocess input text
def preprocess_text(text):
    # Clean the text
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stem the words
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    
    return ' '.join(words)

# Streamlit app
st.title('Sentiment Analysis App')

# Input text from user
user_input = st.text_area('Enter text for sentiment analysis')

if st.button('Predict'):
    # Preprocess the input text
    processed_text = preprocess_text(user_input)
    
    # Vectorize the input text
    vectorizer = TfidfVectorizer()
    input_vector = vectorizer.transform([processed_text])
    
    # Predict sentiment
    prediction = model.predict(input_vector)
    
    # Display the result
    if prediction == 1:
        st.write('Positive Sentiment')
    else:
        st.write('Negative Sentiment')