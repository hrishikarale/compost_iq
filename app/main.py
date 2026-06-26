from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc

from app.db import engine
from app.models import SensorData

app = FastAPI()

# Create DB session
SessionLocal = sessionmaker(bind=engine)


# Request body model
class SensorInput(BaseModel):
    temperature: float
    moisture: float
    humidity: float


# -----------------------
# HOME ROUTE
# -----------------------
@app.get("/")
def home():
    return {"status": "CompostIQ running"}


# -----------------------
# SAVE SENSOR DATA (M5GO / Postman)
# -----------------------
@app.post("/sensor-data")
def add_sensor_data(data: SensorInput):
    db = SessionLocal()

    try:
        new_entry = SensorData(
            temperature=data.temperature,
            moisture=data.moisture,
            humidity=data.humidity
        )

        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)

        return {
            "message": "Data stored successfully",
            "data": {
                "temperature": data.temperature,
                "moisture": data.moisture,
                "humidity": data.humidity
            }
        }

    finally:
        db.close()


# -----------------------
# GET LATEST SENSOR DATA
# -----------------------
@app.get("/latest")
def get_latest():
    db = SessionLocal()

    try:
        latest = db.query(SensorData).order_by(desc(SensorData.id)).first()

        if not latest:
            return {"message": "No data found"}

        return {
            "id": latest.id,
            "temperature": latest.temperature,
            "moisture": latest.moisture,
            "humidity": latest.humidity,
            "timestamp": latest.timestamp
        }

    finally:
        db.close()


# -----------------------
# GET ALL SENSOR DATA (HISTORY)
# -----------------------
@app.get("/sensor-data")
def get_all_data():
    db = SessionLocal()

    try:
        data = db.query(SensorData).order_by(desc(SensorData.id)).all()

        return [
            {
                "id": d.id,
                "temperature": d.temperature,
                "moisture": d.moisture,
                "humidity": d.humidity,
                "timestamp": d.timestamp
            }
            for d in data
        ]

    finally:
        db.close()