from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from api.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    tc = Column(String, unique=True, index=True)
    name = Column(String)
    hashed_password = Column(String)

   

    predictions = relationship("Prediction", back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime)

    user = relationship("User", back_populates="predictions")