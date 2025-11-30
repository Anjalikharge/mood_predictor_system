import re
import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def download_nltk_data(self):
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def preprocess(self, text):
        text = self.clean_text(text)
        words = text.split()
        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 1:
                lemmatized = self.lemmatizer.lemmatize(word)
                if len(lemmatized) > 1:
                    processed_words.append(lemmatized)
        return ' '.join(processed_words) if processed_words else text

class QuoteManager:
    def __init__(self, quotes_file='data/quotes.json'):
        with open(quotes_file, 'r') as f:
            self.quotes = json.load(f)
    
    def get_quote(self, mood):
        if mood in self.quotes:
            return random.choice(self.quotes[mood])
        return "Every moment is a fresh beginning. - T.S. Eliot"
    
    def get_all_moods(self):
        return list(self.quotes.keys())