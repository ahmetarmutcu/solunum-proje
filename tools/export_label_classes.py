import json
import os
import pickle


def main() -> None:
    """Export sklearn LabelEncoder classes to JSON.

    This runs locally (not on Vercel) to remove the scikit-learn runtime dependency.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")

    encoder_pkl = os.path.join(models_dir, "label_encoder.pkl")
    out_json = os.path.join(models_dir, "label_classes.json")

    if not os.path.exists(encoder_pkl):
        raise FileNotFoundError(f"Missing input encoder: {encoder_pkl}")

    with open(encoder_pkl, "rb") as f:
        encoder = pickle.load(f)

    classes = [str(x) for x in getattr(encoder, "classes_", [])]
    if not classes:
        raise ValueError("LabelEncoder has no classes_")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)

    print(f"Wrote: {out_json} ({len(classes)} classes)")


if __name__ == "__main__":
    main()
