
import os
import numpy as np
import librosa
import pickle
import tensorflow as tf

# ----------------------------
# PATHLER
# ----------------------------
PROJECT_DIR = r"C:\Users\Başak\Desktop\tbtk\solunum_proje"
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ----------------------------
# PARAMETRELER
# ----------------------------
MAX_LEN = 259
N_MFCC = 40
SR = 22050

# ----------------------------
# MODEL VE ENCODER YÜKLE
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LE_PATH, "rb") as f:
    le = pickle.load(f)

# ----------------------------
# MFCC ÇIKARMA
# ----------------------------
def extract_mfcc_for_dl(file_path, max_len=MAX_LEN, n_mfcc=N_MFCC, sr=SR):
    y, sr = librosa.load(file_path, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T

    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc

# ----------------------------
# TEK SES → TEK TAHMİN
# ----------------------------
def predict_audio(wav_path):
    mfcc = extract_mfcc_for_dl(wav_path)
    mfcc = mfcc.reshape(1, MAX_LEN, N_MFCC)

    probs = model.predict(mfcc)
    idx = np.argmax(probs)

    label = le.inverse_transform([idx])[0]
    confidence = float(np.max(probs))

    return label, confidence
