import json
import numpy as np
import librosa
import os
import onnxruntime as ort

# -----------------------------
# MODEL YOLLARI
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.onnx")
LABELS_PATH = os.path.join(MODELS_DIR, "label_classes.json")

# Model ve labels'ı lazy load et
session = None
input_name = None
output_name = None
label_classes: list[str] | None = None
def _load_model():
    global session, input_name, output_name, label_classes
    if session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"ONNX model not found: {MODEL_PATH}. "
                "Run tools/convert_to_onnx.py to generate best_model.onnx and commit it."
            )

        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                f"Label classes file not found: {LABELS_PATH}. "
                "Run tools/export_label_classes.py to generate label_classes.json and commit it."
            )

        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        session = sess

        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Expected 1 input and 1 output, got {len(inputs)} inputs and {len(outputs)} outputs"
            )

        input_name = inputs[0].name
        output_name = outputs[0].name

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            label_classes = json.load(f)

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

    # ONNX Runtime expects float32 in most Keras-exported graphs.
    input_tensor = mfcc.astype(np.float32, copy=False)

    probs = session.run([output_name], {input_name: input_tensor})[0]
    class_id = int(np.argmax(probs, axis=-1).item())
    confidence = float(np.max(probs).item())

    label = label_classes[class_id]
    return label, confidence


