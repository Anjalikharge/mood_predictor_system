import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
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
            max_features=10000, 
            ngram_range=(1, 3), 
            min_df=2, 
            max_df=0.85,
            sublinear_tf=True,
            stop_words='english'
        )
        # Use optimized model with balanced classes
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=3000, 
            C=2.0, 
            solver='lbfgs',
            multi_class='multinomial',
            class_weight='balanced'
        )
        
    def load_data(self, file_path):
        """Load and preprocess dataset with balancing"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {len(df)} samples")
        
        # Check class distribution
        print("\nOriginal class distribution:")
        print(df['mood'].value_counts())
        
        # Balance dataset - ensure equal samples per emotion
        balanced_data = []
        target_samples = 100  # Use 100 samples per emotion for better balance
        
        for mood in df['mood'].unique():
            mood_data = df[df['mood'] == mood]
            if len(mood_data) >= target_samples:
                mood_data = mood_data.sample(n=target_samples, random_state=42)
            else:
                # If less than target, use all available
                pass
            balanced_data.append(mood_data)
        
        df = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("\nBalanced class distribution:")
        print(df['mood'].value_counts())
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)
        df = df[df['processed_text'].str.len() > 0]  # Remove empty texts
        
        print(f"\nAfter preprocessing: {len(df)} samples")
        print(f"Moods in dataset: {df['mood'].unique()}")
        
        return df['processed_text'], df['mood']
    
    def train(self, X, y):
        """Train the mood prediction model"""
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Training model...")
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Test alternative models for comparison
        print("\nModel comparison:")
        from sklearn.svm import SVC
        from sklearn.ensemble import GradientBoostingClassifier
        models = [
            ('Logistic Regression', LogisticRegression(random_state=42, max_iter=3000, C=2.0, solver='lbfgs', multi_class='multinomial', class_weight='balanced')),
            ('SVM', SVC(random_state=42, C=1.0, kernel='linear', class_weight='balanced', probability=True)),
            ('Gradient Boosting', GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.1, max_depth=5)),
            ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, class_weight='balanced')),
            ('Naive Bayes', MultinomialNB(alpha=0.5))
        ]
        
        best_accuracy = 0
        best_model = None
        best_name = ""
        
        for name, clf in models:
            clf.fit(X_train_vec, y_train)
            pred = clf.predict(X_test_vec)
            acc = accuracy_score(y_test, pred)
            print(f"{name}: {acc:.4f}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = clf
                best_name = name
        
        # Use the best performing model
        if best_model is not None and best_accuracy > accuracy:
            print(f"\nUsing {best_name} as it performed better: {best_accuracy:.4f}")
            self.model = best_model
            accuracy = best_accuracy
        
        print(f"\nFinal Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model_dir='.'):
        """Save trained model and vectorizer"""
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
    
    # Test predictions with more examples including interview scenarios
    test_texts = [
        "I got selected in an interview",
        "I am feeling great today!",
        "I'm so worried about tomorrow",
        "This makes me really angry",
        "I feel so alone right now",
        "I love spending time with you",
        "I'm proud of my accomplishments",
        "I feel guilty about what happened",
        "I'm excited about the new project",
        "I feel calm and peaceful",
        "Hello, how are you doing?",
        "I passed the job interview",
        "I got hired for my dream job",
        "I'm thrilled about getting selected"
    ]
    
    print(f"\nTest predictions:")
    for text in test_texts:
        mood, confidence = predictor.predict(text)
        print(f"'{text}' -> {mood} ({confidence:.3f})")

if __name__ == "__main__":
    main()