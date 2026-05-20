import numpy as np
import librosa
import pickle
import os
from tflite_runtime.interpreter import Interpreter

# -----------------------------
# MODEL YOLLARI
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.tflite")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Model ve encoder'ı lazy load et
interpreter = None
input_details = None
output_details = None
label_encoder = None
def _load_model():
    global interpreter, input_details, output_details, label_encoder
    if interpreter is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"TFLite model not found: {MODEL_PATH}. "
                "Run tools/convert_to_tflite.py to generate best_model.tflite and commit it."
            )

        interpreter_local = Interpreter(model_path=MODEL_PATH)
        interpreter_local.allocate_tensors()

        interpreter = interpreter_local
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

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
    _load_model()
    mfcc = extract_mfcc(wav_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    # Ensure dtype matches the model input.
    input_index = int(input_details[0]["index"])
    expected_dtype = input_details[0].get("dtype", np.float32)
    input_tensor = mfcc.astype(expected_dtype, copy=False)

    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    output_index = int(output_details[0]["index"])
    probs = interpreter.get_tensor(output_index)
    class_id = int(np.argmax(probs, axis=-1).item())
    confidence = float(np.max(probs).item())

    label = label_encoder.inverse_transform([class_id])[0]
    return label, confidence


