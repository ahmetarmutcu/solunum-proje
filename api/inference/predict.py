import numpy as np
import librosa
import pickle
import os
import json

# -----------------------------
# MODEL YOLLARI
# -----------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Model ve encoder'ı lazy load et
model = None
label_encoder = None


def _normalize_keras_config(node):
    if isinstance(node, dict):
        class_name = node.get("class_name")
        cfg = node.get("config")

        if class_name == "InputLayer" and isinstance(cfg, dict):
            if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")

        # Keras3 style dtype policy objects can break older tf.keras deserialization.
        if isinstance(cfg, dict) and isinstance(cfg.get("dtype"), dict):
            dtype_obj = cfg["dtype"]
            if dtype_obj.get("class_name") == "DTypePolicy":
                cfg["dtype"] = dtype_obj.get("config", {}).get("name", "float32")

        for value in node.values():
            _normalize_keras_config(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_keras_config(item)


def _load_h5_with_inputlayer_patch(model_path: str):
    """Fallback loader for H5 models saved with newer Keras config keys."""
    from tensorflow import keras
    import h5py

    with h5py.File(model_path, "r") as h5:
        model_config = h5.attrs.get("model_config")
        if model_config is None:
            raise ValueError("H5 model_config missing")

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        config_obj = json.loads(model_config)

        _normalize_keras_config(config_obj)

        loaded_model = keras.models.model_from_json(json.dumps(config_obj))
        loaded_model.load_weights(model_path)
        return loaded_model

def _load_model():
    global model, label_encoder
    if model is None:
        from tensorflow import keras
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
        except Exception:
            # Compatibility path for models saved with different Keras config keys.
            model = _load_h5_with_inputlayer_patch(MODEL_PATH)
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

    probs = model.predict(mfcc, verbose=0)
    class_id = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = label_encoder.inverse_transform([class_id])[0]
    return label, confidence

