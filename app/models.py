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

    source = Column(String, nullable=False)
    model_name = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class BatchSetup(Base):
    __tablename__ = "batch_setup"

    id = Column(Integer, primary_key=True, index=True)

    batch_id = Column(String, default="B001")
    container_ml = Column(Float, nullable=False)

    dry_leaves_g = Column(Float, nullable=False)
    grass_clippings_g = Column(Float, nullable=False)
    vegetable_peels_g = Column(Float, nullable=False)
    coffee_grounds_g = Column(Float, nullable=False)
    cardboard_tissue_g = Column(Float, nullable=False)
    twigs_g = Column(Float, nullable=False)

    start_temp = Column(Float, nullable=True)
    start_humidity = Column(Float, nullable=True)
    start_moisture_raw = Column(Float, nullable=True)
    start_moisture_state = Column(String, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow)

class NotificationLog(Base):
    __tablename__ = "notification_log"

    id = Column(Integer, primary_key=True, index=True)

    event_type = Column(String, nullable=False)      # action / critical / offline
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)

    action = Column(String, nullable=True)
    severity = Column(String, nullable=True)

    source_ai_id = Column(Integer, nullable=True)
    source_sensor_id = Column(Integer, nullable=True)

    sent = Column(String, default="no")              # yes / no
    created_at = Column(DateTime, default=datetime.utcnow)