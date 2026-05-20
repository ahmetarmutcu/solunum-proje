import os


def main() -> None:
    """Convert models/best_model.h5 to models/best_model.tflite.

    Notes:
    - This script is meant to be run locally (not on Vercel).
    - It requires TensorFlow installed locally.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    h5_path = os.path.join(models_dir, "best_model.h5")
    tflite_path = os.path.join(models_dir, "best_model.tflite")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing input model: {h5_path}")

    import tensorflow as tf

    model = tf.keras.models.load_model(h5_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Keep it conservative; size is already dominated by runtime deps.
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Wrote: {tflite_path} ({os.path.getsize(tflite_path)} bytes)")


if __name__ == "__main__":
    main()
