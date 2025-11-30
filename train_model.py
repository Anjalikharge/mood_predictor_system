import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os
try:
    from utils import TextPreprocessor
except ImportError:
    import sys
    sys.path.append('.')
    from utils import TextPreprocessor

class MoodPredictor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=15000, 
            ngram_range=(1, 4), 
            min_df=1, 
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        # Optimized ensemble for 90%+ accuracy
        self.lr = LogisticRegression(random_state=42, max_iter=3000, C=10.0, solver='liblinear')
        self.rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, min_samples_split=2)
        self.svm = SVC(kernel='rbf', probability=True, random_state=42, C=10.0, gamma='scale')
        self.nb = MultinomialNB(alpha=0.1)
        self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
        self.model = VotingClassifier(
            estimators=[('lr', self.lr), ('rf', self.rf), ('svm', self.svm), ('nb', self.nb), ('gb', self.gb)],
            voting='soft'
        )
        
    def load_data(self, file_path):
        """Load and preprocess dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {len(df)} samples")
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)
        df = df[df['processed_text'].str.len() > 0]  # Remove empty texts
        
        print(f"After preprocessing: {len(df)} samples")
        print(f"Moods in dataset: {df['mood'].unique()}")
        
        return df['processed_text'], df['mood']
    
    def train(self, X, y):
        """Train the mood prediction model"""
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Training ensemble model...")
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Test individual models for comparison
        print("\nIndividual model accuracies:")
        models = [('Logistic Regression', self.lr), ('Random Forest', self.rf), ('SVM', self.svm), 
                 ('Naive Bayes', self.nb), ('Gradient Boosting', self.gb)]
        for name, clf in models:
            clf.fit(X_train_vec, y_train)
            pred = clf.predict(X_test_vec)
            acc = accuracy_score(y_test, pred)
            print(f"{name}: {acc:.4f}")
        
        print(f"\nEnsemble Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """Save trained model and vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/mood_model.pkl')
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer.pkl')
        
        print(f"Model saved to {model_dir}/")
    
    def predict(self, text):
        """Predict mood from text"""
        processed_text = self.preprocessor.preprocess(text)
        text_vec = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_vec)[0]
        confidence = max(self.model.predict_proba(text_vec)[0])
        
        return prediction, confidence

def main():
    # Initialize predictor
    predictor = MoodPredictor()
    
    # Load and train
    X, y = predictor.load_data('data/dataset.csv')
    accuracy = predictor.train(X, y)
    
    # Save model
    predictor.save_model()
    
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
    
    # Test predictions
    test_texts = [
        "I am feeling great today!",
        "I'm so worried about tomorrow",
        "This makes me really angry",
        "I feel so alone right now"
    ]
    
    print(f"\nTest predictions:")
    for text in test_texts:
        mood, confidence = predictor.predict(text)
        print(f"'{text}' -> {mood} ({confidence:.3f})")

if __name__ == "__main__":
    main()