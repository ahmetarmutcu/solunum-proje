import os


def main() -> None:
    """Convert models/best_model.h5 to models/best_model.onnx.

    Notes:
    - Run locally (not on Vercel).
    - Requires TensorFlow + tf2onnx installed locally.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    h5_path = os.path.join(models_dir, "best_model.h5")
    onnx_path = os.path.join(models_dir, "best_model.onnx")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing input model: {h5_path}")

    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(h5_path, compile=False)

    # Build a concrete function for conversion.
    input_shape = model.inputs[0].shape
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name=model.inputs[0].name.split(":")[0])]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=17)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Wrote: {onnx_path} ({os.path.getsize(onnx_path)} bytes)")


if __name__ == "__main__":
    main()
