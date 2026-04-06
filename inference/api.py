
from fastapi import FastAPI, File, UploadFile
import shutil
import os

from inference.predict import predict_audio

app = FastAPI(title="Respiratory Sound Analysis API")

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Dosyayı geçici olarak kaydet
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Model ile tahmin yap
    label, confidence = predict_audio(file_path)

    # Dosyayı sil
    os.remove(file_path)

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }
