from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc
import os
import json

from google import genai

from app.db import engine
from app.models import SensorData

app = FastAPI()
SessionLocal = sessionmaker(bind=engine)

BATCH_SETUP = {
    "batch_id": "B001",
    "container_ml": 400,
    "dry_leaves_g": 15,
    "grass_clippings_g": 17,
    "vegetable_peels_g": 12,
    "coffee_grounds_g": 3,
    "cardboard_tissue_g": 5,
    "twigs_g": 2,
    "start_temp": 29.22,
    "start_humidity": 53.01,
    "start_moisture_raw": 378.59,
    "start_moisture_state": "Good"
}


class SensorInput(BaseModel):
    temperature: float
    moisture: float
    humidity: float


def fallback_analysis(latest):
    if latest.moisture < 40:
        action = "Add Water"
        moisture_state = "Dry"
    elif latest.moisture <= 70:
        action = "Keep Monitoring"
        moisture_state = "Good"
    else:
        action = "Add Dry Material"
        moisture_state = "Wet"

    if latest.temperature < 35:
        phase = "Mesophilic"
    elif latest.temperature < 55:
        phase = "Thermophilic"
    else:
        phase = "Cooling"

    health = round((latest.moisture + min(latest.temperature, 55)) / 2)
    maturity = min(100, max(0, health + 10))
    ready_days = max(1, round((100 - maturity) / 5))

    return {
        "source": "fallback",
        "maturity": maturity,
        "health": health,
        "phase": phase,
        "ready_days": ready_days,
        "next_action": action,
        "confidence": 60,
        "reason": "Fallback rule-based analysis used.",
        "moisture_state": moisture_state
    }


def validate_ai_output(data):
    required = ["maturity", "health", "phase", "ready_days", "next_action", "confidence", "reason"]

    for key in required:
        if key not in data:
            raise ValueError(f"Missing key: {key}")

    data["maturity"] = int(max(0, min(100, data["maturity"])))
    data["health"] = int(max(0, min(100, data["health"])))
    data["ready_days"] = int(max(1, data["ready_days"]))
    data["confidence"] = int(max(0, min(100, data["confidence"])))

    data["source"] = "gemini"
    return data


@app.get("/")
def home():
    return {"status": "CompostIQ running"}


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

        return {"message": "Data stored successfully", "data": data.dict()}
    finally:
        db.close()


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
                "timestamp": str(d.timestamp)
            }
            for d in data
        ]
    finally:
        db.close()


@app.get("/analysis")
def get_analysis():
    db = SessionLocal()

    try:
        latest = db.query(SensorData).order_by(desc(SensorData.id)).first()

        if not latest:
            return {"message": "No data available"}

        sensor_rows = db.query(SensorData).order_by(SensorData.id).limit(20).all()

        sensor_history = [
            {
                "id": r.id,
                "temperature": r.temperature,
                "moisture": r.moisture,
                "humidity": r.humidity,
                "timestamp": str(r.timestamp)
            }
            for r in sensor_rows
        ]

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return fallback_analysis(latest)

        prompt = f"""
You are CompostIQ AI, an expert compost monitoring assistant.

Use the batch setup and first 20 sensor readings to estimate compost condition.

Batch setup:
{json.dumps(BATCH_SETUP)}

Sensor history:
{json.dumps(sensor_history)}

Latest reading:
temperature={latest.temperature}, moisture={latest.moisture}, humidity={latest.humidity}

Return ONLY valid JSON in exactly this format:
{{
  "maturity": 0-100,
  "health": 0-100,
  "phase": "Mesophilic" or "Thermophilic" or "Cooling" or "Maturing",
  "ready_days": integer,
  "next_action": "Add Water" or "Add Dry Material" or "Turn Compost" or "Keep Monitoring" or "short suggested action",
  "confidence": 0-100,
  "reason": "short reason under 20 words"
}}
"""

        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            ai_data = json.loads(response.text)
            return validate_ai_output(ai_data)

        except Exception as e:
            result = fallback_analysis(latest)
            result["ai_error"] = str(e)
            return result

    finally:
        db.close()