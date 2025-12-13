import joblib
import os
try:
    from utils import TextPreprocessor
except ImportError:
    import sys
    sys.path.append('.')
    from utils import TextPreprocessor

class ModelLoader:
    def __init__(self, model_dir=None):
        # Use absolute path relative to this file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_dir is None:
            self.model_dir = os.path.join(self.base_dir, 'models')
        else:
            self.model_dir = model_dir
        
        try:
            self.preprocessor = TextPreprocessor()
        except:
            # Fallback if TextPreprocessor fails
            self.preprocessor = None
        self.model = None
        self.vectorizer = None
    
    def load_model(self):
        """Load trained model and vectorizer with absolute paths"""
        # Use absolute paths for deployment compatibility
        model_path = os.path.join(self.model_dir, 'mood_model.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        
        # Debug info for troubleshooting
        print(f"Looking for model at: {model_path}")
        print(f"Looking for vectorizer at: {vectorizer_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Base directory: {self.base_dir}")
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print(f"Model loaded successfully from {self.model_dir}!")
                return True
            except Exception as e:
                print(f"Error loading model files: {e}")
                return False
        else:
            print(f"Model files not found at expected paths:")
            print(f"Model exists: {os.path.exists(model_path)}")
            print(f"Vectorizer exists: {os.path.exists(vectorizer_path)}")
            
            # List available files for debugging
            if os.path.exists(self.model_dir):
                print(f"Files in {self.model_dir}: {os.listdir(self.model_dir)}")
            else:
                print(f"Model directory {self.model_dir} does not exist")
                # Try to find models in current directory
                current_files = [f for f in os.listdir(self.base_dir) if f.endswith('.pkl')]
                if current_files:
                    print(f"Found .pkl files in base directory: {current_files}")
            
            raise FileNotFoundError(f"Model files not found. Please train the model first by running: python train_model.py")
    

    
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