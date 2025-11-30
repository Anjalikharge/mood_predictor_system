import json
import random
import os

class QuoteSelector:
    def __init__(self, quotes_path='data/quotes.json'):
        self.quotes_path = quotes_path
        self.quotes = self.load_quotes()
    
    def load_quotes(self):
        if os.path.exists(self.quotes_path):
            with open(self.quotes_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_quote(self, mood):
        if mood in self.quotes and self.quotes[mood]:
            return random.choice(self.quotes[mood])
        return "Stay positive and keep moving forward!"