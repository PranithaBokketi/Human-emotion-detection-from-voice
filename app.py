import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pickle
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import tempfile

# Load the trained model
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to extract 180 features (40 MFCC + 12 Chroma + 128 Mel)
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)

    # Handle short audio clips
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    # MFCCs (40)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Chroma (12) with adjusted n_fft
    n_fft = 2048 if len(y) >= 2048 else 512  # Use smaller window if signal is short
    stft = np.abs(librosa.stft(y, n_fft=n_fft))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    # Mel spectrogram (128)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)

    # Combine all features
    features = np.hstack([mfccs, chroma, mel])
    return features.reshape(1, -1)



# Function to predict emotion
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    prediction = model.predict(features)
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="Voice Emotion Detector", layout="centered")
st.title("Human Emotion Detection from Voice")

st.markdown("### Upload a .wav File")
uploaded_file = st.file_uploader("Drag and drop file here", type=["wav"])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name

    st.audio(tmpfile_path, format='audio/wav')
    result = predict_emotion(tmpfile_path)
    st.success(f"Predicted Emotion: {result.upper()}")

st.markdown("---")
st.markdown("### Or Record Using Microphone")

# Record from mic and save temp audio
ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    audio_receiver_size=2048,
)



if ctx.audio_receiver:
    try:
        audio_frames = ctx.audio_receiver.get_frames(timeout=3)
        if audio_frames:
            audio_data = b''.join([frame.to_ndarray().tobytes() for frame in audio_frames])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, np.frombuffer(audio_data, dtype=np.int16), 48000)
                st.audio(f.name, format='audio/wav')
                result = predict_emotion(f.name)
                st.success(f"Predicted Emotion: {result.upper()}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")

