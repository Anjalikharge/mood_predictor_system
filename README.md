# 🎭 Mood Predictor

A machine learning-powered web application that analyzes text to predict mood and provides motivational quotes.

## 📁 Project Structure
```
mood_predictor/
├── data/
│   ├── dataset.csv          # Training dataset
│   └── quotes.json          # Motivational quotes
├── models/                  # Trained model files
├── templates/
│   └── index.html          # Web interface
├── app.py                  # Flask web application
├── data_cleaner.py         # Dataset cleaning
├── model_loader.py         # Model loading and prediction
├── train_model.py          # Model training
├── utils.py                # Text preprocessing
├── quote_selector.py       # Quote selection logic
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Application
```bash
python app.py
```

### 4. Open Browser
Navigate to `http://localhost:5000`

## 📋 Step-by-Step Instructions

### Step 1: Environment Setup
1. Open terminal in project directory
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Data Preparation
1. Run data cleaning (creates sample data if needed):
   ```bash
   python data_cleaner.py
   ```

### Step 3: Model Training
1. Train the mood prediction model:
   ```bash
   python train_model.py
   ```
   This will:
   - Clean the dataset
   - Preprocess text data
   - Train a Naive Bayes classifier
   - Save model and vectorizer to `models/` folder

### Step 4: Test Model (Optional)
1. Test the model directly:
   ```bash
   python model_loader.py
   ```

### Step 5: Launch Web Application
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open browser and go to `http://localhost:5000`
3. Enter text and click "Analyze Mood"

## 🎯 Features

- **Text Preprocessing**: Removes stopwords, applies stemming
- **Mood Detection**: Classifies text into 5 moods (happy, sad, angry, anxious, neutral)
- **Confidence Scoring**: Shows prediction confidence percentage
- **Motivational Quotes**: Displays mood-appropriate inspirational messages
- **Web Interface**: Clean, responsive design with color-coded results

## 🔧 Customization

### Adding New Moods
1. Update `data/quotes.json` with new mood categories
2. Add training data for new moods in `data/dataset.csv`
3. Retrain the model with `python train_model.py`

### Adding More Quotes
Edit `data/quotes.json` and add quotes to existing mood categories.

## 🛠️ Troubleshooting

**Model not found error:**
- Run `python train_model.py` first

**NLTK download errors:**
- The app will automatically download required NLTK data

**Port already in use:**
- Change port in `app.py`: `app.run(debug=True, port=5001)`

## 📊 Model Performance

The model uses TF-IDF vectorization with Naive Bayes classification. Accuracy depends on training data quality and size.
import webbrowser
webbrowser.open("http://127.0.0.1:5000")
from flask import Flask
import webbrowser

app = Flask(__name__)

@app.route("/")
def home():
    return "Mood Predictor Running!"

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
