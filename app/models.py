from sqlalchemy import Column, Integer, Float, DateTime, String, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class SensorData(Base):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    temperature = Column(Float, nullable=False)
    moisture = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AIAnalysis(Base):
    __tablename__ = "ai_analysis"

    id = Column(Integer, primary_key=True, index=True)

    batch_id = Column(String, default="B001")
    sensor_data_id = Column(Integer, nullable=True)

    maturity = Column(Integer, nullable=False)
    health = Column(Integer, nullable=False)
    phase = Column(String, nullable=False)
    ready_days = Column(Integer, nullable=False)
    next_action = Column(String, nullable=False)
    confidence = Column(Integer, nullable=False)
    reason = Column(Text, nullable=False)

    source = Column(String, nullable=False)       # gemini / fallback
    model_name = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)