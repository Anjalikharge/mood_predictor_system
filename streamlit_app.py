import streamlit as st
import os
import sys
import pandas as pd
import time
from datetime import datetime

# Add current directory to path
sys.path.append('.')

try:
    import model_loader
    import quote_selector
    from model_loader import ModelLoader
    from quote_selector import QuoteSelector
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all files are in the same directory and dependencies are installed.")
    st.info("Run: pip install -r requirements.txt")
    st.stop()

# Page config
st.set_page_config(
    page_title="🎭 Mood Predictor",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive background and animations
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.95) !important;
    color: #333 !important;
    border: 2px solid rgba(255,255,255,0.8) !important;
    border-radius: 15px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}
.stTextArea textarea::placeholder {
    color: #666 !important;
}
.stTextArea label {
    color: white !important;
    font-weight: bold !important;
    font-size: 18px !important;
}
.mood-card {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    margin: 10px 0;
    text-align: center;
}
.emoji-animation {
    font-size: 4rem;
    animation: bounce 1s infinite;
}
@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-10px); }
    60% { transform: translateY(-5px); }
}
</style>
""", unsafe_allow_html=True)

# Mood configurations with emojis and colors
MOOD_CONFIG = {
    'happy': {'emoji': '😊', 'color': '#FFD93D', 'bg': 'linear-gradient(135deg, #FFD93D, #FF6B6B)'},
    'sad': {'emoji': '😢', 'color': '#6C5CE7', 'bg': 'linear-gradient(135deg, #74b9ff, #0984e3)'},
    'angry': {'emoji': '😠', 'color': '#E17055', 'bg': 'linear-gradient(135deg, #fd79a8, #e84393)'},
    'fearful': {'emoji': '😰', 'color': '#FDCB6E', 'bg': 'linear-gradient(135deg, #fdcb6e, #e17055)'},
    'excited': {'emoji': '🤩', 'color': '#FD79A8', 'bg': 'linear-gradient(135deg, #fd79a8, #fdcb6e)'},
    'calm': {'emoji': '😌', 'color': '#00B894', 'bg': 'linear-gradient(135deg, #00cec9, #00b894)'},
    'love': {'emoji': '😍', 'color': '#E84393', 'bg': 'linear-gradient(135deg, #fd79a8, #e84393)'},
    'lonely': {'emoji': '😔', 'color': '#636E72', 'bg': 'linear-gradient(135deg, #b2bec3, #636e72)'},
    'proud': {'emoji': '😎', 'color': '#A29BFE', 'bg': 'linear-gradient(135deg, #a29bfe, #6c5ce7)'},
    'guilty': {'emoji': '😳', 'color': '#FAB1A0', 'bg': 'linear-gradient(135deg, #fab1a0, #e17055)'},
    'neutral': {'emoji': '😐', 'color': '#95A5A6', 'bg': 'linear-gradient(135deg, #bdc3c7, #95a5a6)'}
}

# Initialize components
def load_model():
    try:
        model_loader = ModelLoader()
        model_loader.load_model()
        return model_loader
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

@st.cache_resource
def load_quotes():
    return QuoteSelector()

def save_user_data(text, predicted_mood, confidence, actual_mood=None):
    """Save user input and predictions to CSV"""
    user_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'text': text,
        'predicted_mood': predicted_mood,
        'confidence': confidence,
        'actual_mood': actual_mood or predicted_mood
    }
    
    file_path = 'data/user_interactions.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([user_data])], ignore_index=True)
    else:
        df = pd.DataFrame([user_data])
    
    df.to_csv(file_path, index=False)
    return True

# Load components
model_loader = load_model()
quote_selector = load_quotes()

# App header with gradient
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="font-size: 3rem; background: linear-gradient(45deg, #FFD93D, #FF6B6B, #74B9FF);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text; margin-bottom: 10px;">🎭 Mood Predictor</h1>
    <p style="font-size: 1.2rem; opacity: 0.8;">✨ Discover your emotions ✨</p>
</div>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Text input
    text_input = st.text_area(
        "💭 How are you feeling today?",
        height=120,
        placeholder="Share your thoughts and emotions here...",
        help="Express yourself freely - I'll analyze your mood!"
    )

    # Predict button with custom styling
    if st.button("🔮 Analyze My Mood", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("💬 Please share your thoughts first!")
        elif model_loader is None:
            st.error("🚫 Model not loaded. Run `python train_model.py` first.")
        else:
            try:
                # Show loading animation
                with st.spinner('🧠 Analyzing your emotions...'):
                    time.sleep(1)  # Brief pause for effect
                    mood, confidence = model_loader.predict(text_input)
                    quote = quote_selector.get_quote(mood)
                
                # Get mood configuration
                config = MOOD_CONFIG.get(mood, MOOD_CONFIG['happy'])
                
                # Display results with animated emoji and styling
                st.markdown(f"""
                <div class="mood-card" style="background: {config['bg']}; color: white;">
                    <div class="emoji-animation">{config['emoji']}</div>
                    <h2 style="margin: 10px 0; text-transform: uppercase; letter-spacing: 2px;">{mood}</h2>
                    <div style="background: rgba(255,255,255,0.3); border-radius: 10px; padding: 10px; margin: 15px 0;">
                        <div style="background: linear-gradient(90deg, transparent 0%, white {confidence*100}%, transparent {confidence*100}%);
                                    height: 8px; border-radius: 4px; margin-bottom: 5px;"></div>
                        <p style="margin: 0; font-weight: bold;">Confidence: {confidence*100:.1f}%</p>
                    </div>
                    <p style="font-style: italic; font-size: 1.1rem; margin: 15px 0;">"{quote}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Save user data
                save_user_data(text_input, mood, confidence)
                
                # Feedback section
                st.markdown("---")
                st.markdown("### 🎯 Was this prediction accurate?")
                
                feedback_col1, feedback_col2 = st.columns(2)
                with feedback_col1:
                    if st.button("✅ Yes, spot on!"):
                        st.success("Thanks for the feedback! 🎉")
                        save_user_data(text_input, mood, confidence, mood)
                
                with feedback_col2:
                    actual_mood = st.selectbox(
                        "🔄 Select correct mood:",
                        options=list(MOOD_CONFIG.keys()),
                        index=list(MOOD_CONFIG.keys()).index(mood)
                    )
                    if st.button("📝 Update & Learn"):
                        save_user_data(text_input, mood, confidence, actual_mood)
                        # Add to training data
                        new_data = pd.DataFrame([{'text': text_input, 'mood': actual_mood}])
                        if os.path.exists('data/dataset.csv'):
                            existing_data = pd.read_csv('data/dataset.csv')
                            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                            updated_data.to_csv('data/dataset.csv', index=False)
                        st.success(f"✨ Added to training data as '{actual_mood}' - Thanks for helping me learn!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); 
                border-radius: 15px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">🎭 About</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 AI-Powered Mood Detection")
    st.write("This app uses advanced machine learning to analyze your emotions from text.")
    
    st.markdown("### 🎨 Supported Moods:")
    for mood, config in MOOD_CONFIG.items():
        st.markdown(f"**{config['emoji']} {mood.title()}**")
    
    st.markdown("---")
    st.markdown("### 📊 Your Stats")
    
    # Show user interaction stats
    if os.path.exists('data/user_interactions.csv'):
        df = pd.read_csv('data/user_interactions.csv')
        st.metric("Total Predictions", len(df))
        if len(df) > 0:
            most_common = df['predicted_mood'].mode()[0]
            st.metric("Most Common Mood", f"{MOOD_CONFIG[most_common]['emoji']} {most_common.title()}")
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("💭 Be descriptive about your feelings for better accuracy!")
    st.info("🔄 Help improve the AI by providing feedback on predictions!")