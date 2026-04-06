from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from pydantic import BaseModel
import shutil
import os
from datetime import datetime
from sqlalchemy.orm import Session

from api.database import SessionLocal, engine
from api.models import Base, User, Prediction
from api.auth import create_user, authenticate_user
from api.inference.predict import predict_audio

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Respiratory Sound Analysis API")

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------- DB SESSION ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------- MODELS ----------------
class UserRequest(BaseModel):
    tc: str
    name: str | None = None
    password: str

# ---------------- REGISTER ----------------
@app.post("/register")
def register(user: UserRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.tc == user.tc).first():
      raise HTTPException(status_code=400, detail="User already exists")

    new_user = create_user(db, user.tc, user.name, user.password)

    return {
        "message": "User created",
        "user_id": new_user.id
    }


# ---------------- LOGIN ----------------
@app.post("/login")
def login(user: UserRequest, db: Session = Depends(get_db)):
    db_user = authenticate_user(db, user.tc, user.password)

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "message": "Login successful",
        "user_id": db_user.id
    }


# ---------------- PREDICT ----------------
@app.post("/predict")
async def predict(
    user_id: int = Form(...),   # 🔥 FORM OLMAK ZORUNDA
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence = predict_audio(file_path)

    record = Prediction(
        user_id=user_id,
        filename=file.filename,
        prediction=label,
        confidence=float(confidence),
        created_at=datetime.now()
    )

    db.add(record)
    db.commit()

    return {
        "prediction": label,
        "confidence": round(float(confidence), 4)
    }

# -------------------------------------------------
# CHAT MODEL
# -------------------------------------------------
class ChatRequest(BaseModel):
    user_id: int
    message: str


# -------------------------------------------------
# CHATBOT ENGINE
# -------------------------------------------------
def generate_medical_response(message: str) -> str:

    msg = message.lower()

    advice_mode = any(word in msg for word in [
        "ne yap", "ne yapmalıyım", "ne yapmam", "tedavi", "öneri"
    ])

    # ---------------- ASTHMA ----------------
    if "astım" in msg or "asthma" in msg:

        if advice_mode:
            return """
Astım için öneriler:

• Alerjenlerden uzak durun (toz, polen, sigara dumanı)
• Doktorun verdiği inhaleri düzenli kullanın
• Soğuk havada dikkatli olun
• Şiddetli nefes darlığında acile başvurun
"""
        else:
            return """
Astım kronik bir hava yolu hastalığıdır.

Belirtiler:
• Nefes darlığı
• Hırıltı
• Gece artan öksürük
"""

    # ---------------- COPD ----------------
    if "koah" in msg or "copd" in msg:

        if advice_mode:
            return """
KOAH için öneriler:

• Sigara bırakılmalıdır
• Düzenli doktor kontrolü yapılmalıdır
• Solunum egzersizleri faydalıdır
• Grip aşısı önerilir
"""
        else:
            return "KOAH kronik obstrüktif akciğer hastalığıdır."

    # ---------------- URTI ----------------
    if "urti" in msg or "üst solunum" in msg:

        if advice_mode:
            return """
Üst solunum yolu enfeksiyonunda:

• Bol sıvı tüketin
• Dinlenin
• Ateşi takip edin
• 1 haftadan uzun sürerse doktora başvurun
"""
        else:
            return "URTI üst solunum yolu enfeksiyonudur."

    # ---------------- LRTI ----------------
    if "lrti" in msg or "alt solunum" in msg:

        if advice_mode:
            return """
Alt solunum yolu enfeksiyonunda:

• Doktor kontrolü önemlidir
• Yüksek ateşte tıbbi destek alın
"""
        else:
            return "LRTI alt solunum yolu enfeksiyonudur."

    # ---------------- PNEUMONIA ----------------
    if "zatürre" in msg or "pneumonia" in msg:

        if advice_mode:
            return """
Zatürrede:

• Acil tıbbi değerlendirme gerekir
• Antibiyotik tedavisi doktor tarafından planlanır
"""
        else:
            return "Zatürre akciğer enfeksiyonudur."

    # ---------------- BRONCHIECTASIS ----------------
    if "bronchiectasis" in msg:

        if advice_mode:
            return """
Bronşektazide:

• Düzenli balgam temizliği yapılmalı
• Solunum fizyoterapisi önerilir
"""
        else:
            return "Bronşektazi hava yollarının kalıcı genişlemesidir."

    # ---------------- BRONCHIOLITIS ----------------
    if "bronchiolitis" in msg:

        if advice_mode:
            return """
Bronşiolitte:

• Özellikle çocuklarda dikkatli takip gerekir
• Hızlı nefes alıp verme varsa doktora başvurun
"""
        else:
            return "Bronşiolit küçük hava yollarının iltihabıdır."

    # ---------------- HEALTHY ----------------
    if "healthy" in msg or "sağlıklı" in msg:
        return "Analiz sonucu sağlıklı görünüyor. Şikayet varsa doktora başvurun."

    return "Sorunuz desteklenen hastalıklarla eşleşmedi."

@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == req.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    response_text = generate_medical_response(req.message)

    return {
        "response": response_text
    }