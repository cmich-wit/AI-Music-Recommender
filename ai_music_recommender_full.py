import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Download required NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load pre-trained MobileNetV2 model for emotion detection
# This is a placeholder, a real trained model should be used
model = load_model('mobilenetv2_emotion.h5')  # Assume a pre-trained model exists

# Define mood-to-music mapping
MOOD_TO_MUSIC = {
    "happy": ["Pop", "Dance", "Upbeat Indie"],
    "sad": ["Acoustic", "Blues", "Soft Piano"],
    "angry": ["Rock", "Metal", "Hard Rap"],
    "relaxed": ["Jazz", "Lo-Fi", "Classical"],
    "neutral": ["Top 40", "Alternative"]
}

# Function to analyze text sentiment
def analyze_text_sentiment(text):
    """
    Uses VADER Sentiment Analysis to determine the sentiment score.
    """
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.5:
        return "happy"
    elif sentiment['compound'] <= -0.3:
        return "sad"
    elif sentiment['compound'] < 0.5 and sentiment['compound'] > 0.1:
        return "relaxed"
    else:
        return "neutral"

# Function to analyze image sentiment
def analyze_image_sentiment(image):
    """
    Processes an uploaded image to classify facial emotion.
    """
    image = Image.open(image).convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  # Resize for MobileNetV2
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    mood_index = np.argmax(predictions)  # Get the highest probability mood
    mood_labels = ["happy", "sad", "angry", "relaxed", "neutral"]  # Example moods
    return mood_labels[mood_index]

# Streamlit UI setup
st.title("🎶 AI-Powered Mood-Based Music Recommender")
st.write("Enter text or upload an image, and AI will recommend music based on your mood!")

# Text input for mood analysis
text_input = st.text_area("How are you feeling today? Describe in a sentence:")
if st.button("Analyze Text Mood"):
    if text_input:
        mood = analyze_text_sentiment(text_input)
        st.write(f"Detected Mood: {mood.capitalize()}")
        st.write(f"Recommended Playlist: {MOOD_TO_MUSIC.get(mood, ['No suggestion available'])}")
    else:
        st.write("Please enter some text.")

# Image upload for mood analysis
uploaded_image = st.file_uploader("Upload a selfie to detect mood:", type=["jpg", "png"])
if uploaded_image:
    if st.button("Analyze Image Mood"):
        mood = analyze_image_sentiment(uploaded_image)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Detected Mood: {mood.capitalize()}")
        st.write(f"Recommended Playlist: {MOOD_TO_MUSIC.get(mood, ['No suggestion available'])}")

st.write("\n🎵 Enjoy your personalized music recommendations!")
