# 🎶 AI-Powered Mood-Based Music Recommender

**Last Updated:** April 2, 2025  
**Developed for:** Women in Technology (WiT) Presentations (April 2 & April 9, 2025)

## 📌 Overview
This AI-powered music recommender system suggests personalized music playlists based on the user's **emotional state**, which is detected using either **text sentiment analysis** or **facial emotion recognition** from an uploaded selfie.

Users can interact with the app through a friendly **Streamlit interface**, allowing them to enter text or upload an image to receive mood-based music genre suggestions.

## 🎯 Purpose
This project was created to demonstrate the intersection of **AI, sentiment analysis**, and **user-centered design** during the WiT April events. It serves as a beginner-friendly yet impactful example of applying machine learning in creative, real-world scenarios.

## 🧠 Features

### ✅ Text-Based Mood Detection
- Uses **NLTK's VADER SentimentIntensityAnalyzer** to analyze user input.
- Maps detected sentiment to a corresponding mood and playlist genre.

### ✅ Image-Based Mood Detection
- Accepts selfies uploaded by the user.
- Uses a **pre-trained MobileNetV2** model for facial emotion recognition.
- Outputs a predicted mood from the image and suggests matching music.

### ✅ Streamlit Web App
- Intuitive interface for users to input emotions via text or image.
- Real-time mood detection and playlist recommendation.

## 📁 Files

### `ai_music_recommender_full.py`
- **Completed version** of the application with working sentiment analysis and emotion classification logic.
- Integrates both text and image-based mood recognition.
- Suitable for demo purposes and deployment.

### `ai_music_recommender_partial.py`
- **Work-in-progress version** of the same project.
- Includes code structure and placeholder areas marked as `TODO`.
- Used as a teaching example to walk through the development process step-by-step during the April 2 presentation.

## 🔧 Requirements

- Python 3.7+
- Streamlit
- TensorFlow / Keras
- NLTK
- OpenCV (`cv2`)
- PIL (Pillow)

Install dependencies with:
```bash
pip install streamlit tensorflow nltk opencv-python pillow
```

## 🚀 Running the App

To launch the Streamlit app:

```bash
streamlit run ai_music_recommender_full.py
```

## 📌 Notes
- Ensure that the **`mobilenetv2_emotion.h5`** pre-trained model file is present in the same directory. This model is a placeholder and should be replaced with a trained emotion detection model.
- The VADER lexicon will be downloaded at runtime using `nltk.download('vader_lexicon')`.

## 🧑‍💻 Author & Credits

Author: Edom Belayneh

Created and presented by the Women in Technology (WiT) organization at **Central Michigan University**.  
This project is part of our initiative to **make AI accessible**, **encourage experimentation**, and **inspire innovation** among underrepresented groups in tech.
We're committed to supporting women in STEM through mentorship, community, and practical learning experiences.
