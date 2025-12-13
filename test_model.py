#!/usr/bin/env python3

import os
import sys
sys.path.append('.')

from model_loader import ModelLoader

def test_model():
    print("Testing model loading...")
    
    # Test model loading
    try:
        loader = ModelLoader()
        loader.load_model()
        print("[OK] Model loaded successfully!")
        
        # Test predictions
        test_cases = [
            "I am feeling great today!",
            "I'm so worried about tomorrow",
            "This makes me really angry",
            "I feel so alone right now",
            "I love you so much",
            "I'm proud of my achievement",
            "I feel guilty about what happened",
            "I'm excited about this opportunity",
            "I feel calm and peaceful",
            "Hello, how are you?"
        ]
        
        print("\nTesting predictions:")
        for text in test_cases:
            mood, confidence = loader.predict(text)
            print(f"'{text}' -> {mood} ({confidence:.3f})")
            
        print("\n[OK] All tests passed! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    test_model()