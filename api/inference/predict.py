import numpy as np
import librosa
import pickle
import tensorflow as tf
import os

# -----------------------------
# MODEL YOLLARI
# -----------------------------
# -----------------------------
# MODEL YOLLARI (DÜZELTİLDİ)
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# -----------------------------
# MODEL + ENCODER YÜKLE
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------------
# MFCC AYARLARI
# -----------------------------
MAX_LEN = 259
N_MFCC = 40
SR = 22050

def extract_mfcc(file_path: str):
    y, sr = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    if mfcc.shape[0] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]

    return mfcc

def predict_audio(wav_path: str):
    mfcc = extract_mfcc(wav_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    probs = model.predict(mfcc, verbose=0)
    class_id = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = label_encoder.inverse_transform([class_id])[0]
    return label, confidence

