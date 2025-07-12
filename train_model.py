import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Emotion code mapping based on RAVDESS dataset
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Absolute path to this script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))
audio_data_dir = os.path.join(base_dir, "audio_data")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        return np.hstack([mfcc, chroma, mel])
    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None

features = []
labels = []

# Walk through audio_data folder
for root, dirs, files in os.walk(audio_data_dir):
    for file in files:
        if file.endswith(".wav"):
            print(f"\n➡ Found file: {file}")
            parts = file.split("-")
            print("🧩 parts =", parts)

            if len(parts) < 3:
                print(f"❌ Skipping invalid filename: {file}")
                continue

            emotion_code = parts[2]
            print(f"🧩 Extracted emotion_code: {emotion_code} from file: {file}")

            emotion = emotion_map.get(emotion_code)
            if emotion is None:
                print(f"⛔ Skipped: emotion_code '{emotion_code}' not found in emotion_map.")
                continue

            file_path = os.path.join(root, file)
            print(f"📂 Full path: {file_path}")

            feature = extract_features(file_path)

            if feature is not None:
                features.append(feature)
                labels.append(emotion)
                print(f"✅ Processed: {file} → {emotion}")
            else:
                print(f"⚠️ Feature extraction failed for {file}")

print(f"\n✅ Loaded {len(features)} audio samples.")

if len(features) == 0:
    raise ValueError("❌ No features extracted. Check audio_data folder and file names.")

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\n✅ Model trained. Accuracy: {accuracy:.2f}")

# Save the model
model_path = os.path.join(base_dir, "emotion_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n📦 Model saved to {model_path}")


# Save the model
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)  # 👈 Add protocol=4

