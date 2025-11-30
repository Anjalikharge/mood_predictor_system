import joblib
import os
try:
    from utils import TextPreprocessor
except ImportError:
    import sys
    sys.path.append('.')
    from utils import TextPreprocessor

class ModelLoader:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
    
    def load_model(self):
        """Load trained model and vectorizer"""
        model_path = os.path.join(self.model_dir, 'mood_model.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model files not found. Train the model first.")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print("Model loaded successfully!")
        return True
    
    def predict(self, text):
        """Predict mood from text"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        processed_text = self.preprocessor.preprocess(text)
        if not processed_text:
            return None, 0.0
        
        text_vec = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_vec)[0]
        confidence = max(self.model.predict_proba(text_vec)[0])
        
        return prediction, confidence

if __name__ == "__main__":
    loader = ModelLoader()
    loader.load_model()
    
    # Test prediction
    test_text = "I am feeling great today!"
    mood, confidence = loader.predict(test_text)
    print(f"Text: '{test_text}'")
    print(f"Predicted mood: {mood} (confidence: {confidence:.3f})")