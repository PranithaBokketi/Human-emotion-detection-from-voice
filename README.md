Human Emotion Detection from Voice using Streamlit

Overview:
This project allows users to detect human emotions from voice recordings. Users can either upload a pre-recorded `.wav` file or use their microphone to record audio directly in the browser.

Features:
- Extracts MFCC, Chroma, and Mel spectrogram features from voice input
- Predicts emotion using a trained RandomForest model
- Simple and interactive Streamlit interface
- Microphone support with real-time recording

Requirements:
- Python 3.8+
- Streamlit
- streamlit-webrtc
- librosa
- numpy
- soundfile
- scikit-learn

Setup Instructions:
1. Clone this repository.
2. Install the required packages:
   pip install -r requirements.txt
3. Make sure the file `emotion_model.pkl` (trained model) is present in the root directory.
4. Run the app using:
   streamlit run app.py

Usage:
- Upload a `.wav` audio file to analyze emotion.
- Or record using the microphone feature directly in the app.
- The app will display the predicted emotion on screen.

Author:
Bokketi Pranitha
