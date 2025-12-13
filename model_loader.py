import joblib
import os
try:
    from utils import TextPreprocessor
except ImportError:
    import sys
    sys.path.append('.')
    from utils import TextPreprocessor

class ModelLoader:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
    
    def load_model(self):
        """Load trained model and vectorizer with absolute paths"""
        # Try base directory first, then models folder
        locations = [self.base_dir, self.model_dir]
        
        for location in locations:
            model_path = os.path.join(location, 'mood_model.pkl')
            vectorizer_path = os.path.join(location, 'vectorizer.pkl')
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                try:
                    self.model = joblib.load(model_path)
                    self.vectorizer = joblib.load(vectorizer_path)
                    return True
                except Exception as e:
                    continue
        
        raise FileNotFoundError("Model files not found")
    

    
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