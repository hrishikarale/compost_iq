from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import os
import json

from google import genai

from app.db import engine
from app.models import Base, SensorData, AIAnalysis, BatchSetup, NotificationLog

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create tables automatically
Base.metadata.create_all(bind=engine)

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

GEMINI_MODEL = "gemini-2.5-flash"

# Development/demo settings
GEMINI_MIN_INTERVAL_MINUTES = 5
GEMINI_MIN_NEW_ROWS = 10


class SensorInput(BaseModel):
    temperature: float
    moisture: float
    humidity: float

class BatchSetupInput(BaseModel):
    batch_id: str = "B001"
    container_ml: float

    dry_leaves_g: float
    grass_clippings_g: float
    vegetable_peels_g: float
    coffee_grounds_g: float
    cardboard_tissue_g: float
    twigs_g: float

    start_temp: float | None = None
    start_humidity: float | None = None
    start_moisture_raw: float | None = None
    start_moisture_state: str | None = None

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, int(value)))


def fallback_analysis(latest):
    if latest.moisture < 40:
        moisture_state = "Dry"
        action = "Add Water"
    elif latest.moisture <= 70:
        moisture_state = "Good"
        action = "Keep Monitoring"
    else:
        moisture_state = "Wet"
        action = "Add Dry Material"

    if latest.temperature < 35:
        phase = "Mesophilic"
    elif latest.temperature < 55:
        phase = "Thermophilic"
    else:
        phase = "Cooling"

    moisture_score = 100 - abs(55 - latest.moisture) * 2
    temp_score = 100 - abs(40 - latest.temperature) * 2
    humidity_score = 100 - abs(50 - latest.humidity)

    health = round(
        max(0, min(100, (moisture_score * 0.45) + (temp_score * 0.35) + (humidity_score * 0.20)))
    )

    maturity = clamp(health // 2, 0, 100)
    ready_days = clamp(round(70 - (maturity * 0.6)), 1, 90)

    return {
        "maturity": maturity,
        "health": health,
        "phase": phase,
        "ready_days": ready_days,
        "next_action": action,
        "confidence": 60,
        "reason": f"Fallback used. Moisture is {moisture_state}.",
        "source": "fallback",
        "model_name": "rule-based"
    }


def validate_ai_output(data, previous_ai=None):
    required = [
        "maturity",
        "health",
        "phase",
        "ready_days",
        "next_action",
        "confidence",
        "reason"
    ]

    for key in required:
        if key not in data:
            raise ValueError(f"Missing key from Gemini response: {key}")

    allowed_phases = ["Mesophilic", "Thermophilic", "Cooling", "Maturing"]
    allowed_actions = ["Add Water", "Add Dry Material", "Turn Compost", "Keep Monitoring"]

    maturity = clamp(data["maturity"], 0, 100)
    health = clamp(data["health"], 0, 100)
    ready_days = clamp(data["ready_days"], 1, 90)
    confidence = clamp(data["confidence"], 0, 100)

    phase = data["phase"]
    if phase not in allowed_phases:
        phase = "Mesophilic"

    next_action = data["next_action"]
    if next_action not in allowed_actions:
        next_action = "Keep Monitoring"

    reason = str(data["reason"])[:120]

    # Stability control using previous AI result
    if previous_ai:
        previous_maturity = previous_ai.maturity
        previous_ready_days = previous_ai.ready_days

        # Maturity should not jump too wildly
        if maturity > previous_maturity + 8:
            maturity = previous_maturity + 8
        if maturity < previous_maturity - 3:
            maturity = previous_maturity - 3

        # Ready days should not jump randomly
        if ready_days > previous_ready_days + 10:
            ready_days = previous_ready_days + 10
        if ready_days < previous_ready_days - 10:
            ready_days = previous_ready_days - 10

        maturity = clamp(maturity, 0, 100)
        ready_days = clamp(ready_days, 1, 90)

    return {
        "maturity": maturity,
        "health": health,
        "phase": phase,
        "ready_days": ready_days,
        "next_action": next_action,
        "confidence": confidence,
        "reason": reason,
        "source": "gemini",
        "model_name": GEMINI_MODEL
    }


def ai_analysis_to_dict(row):
    return {
        "id": row.id,
        "batch_id": row.batch_id,
        "sensor_data_id": row.sensor_data_id,
        "maturity": row.maturity,
        "health": row.health,
        "phase": row.phase,
        "ready_days": row.ready_days,
        "next_action": row.next_action,
        "confidence": row.confidence,
        "reason": row.reason,
        "source": row.source,
        "model_name": row.model_name,
        "created_at": row.created_at
    }


def save_ai_analysis(db, latest, result):
    new_ai = AIAnalysis(
        batch_id=BATCH_SETUP["batch_id"],
        sensor_data_id=latest.id,
        maturity=result["maturity"],
        health=result["health"],
        phase=result["phase"],
        ready_days=result["ready_days"],
        next_action=result["next_action"],
        confidence=result["confidence"],
        reason=result["reason"],
        source=result["source"],
        model_name=result["model_name"]
    )

    db.add(new_ai)
    db.commit()
    db.refresh(new_ai)

    return new_ai


def should_run_gemini(latest, previous_ai, sensor_count_after_previous):
    if not previous_ai:
        return True, "No previous AI analysis exists."

    age = datetime.utcnow() - previous_ai.created_at

    if age >= timedelta(minutes=GEMINI_MIN_INTERVAL_MINUTES):
        return True, "Minimum AI time interval passed."

    if sensor_count_after_previous >= GEMINI_MIN_NEW_ROWS:
        return True, "Enough new sensor rows collected."

    if latest.temperature > 60 or latest.temperature < 15:
        return True, "Critical temperature detected."

    if latest.moisture > 85 or latest.moisture < 25:
        return True, "Critical moisture detected."

    return False, "Gemini skipped to save tokens."


def run_gemini_analysis(db, latest, previous_ai):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return fallback_analysis(latest)

    first_20_rows = db.query(SensorData).order_by(SensorData.id).limit(20).all()
    latest_20_rows = db.query(SensorData).order_by(desc(SensorData.id)).limit(20).all()

    first_20 = [
        {
            "id": r.id,
            "temperature": r.temperature,
            "moisture": r.moisture,
            "humidity": r.humidity,
            "timestamp": str(r.timestamp)
        }
        for r in first_20_rows
    ]

    latest_20 = [
        {
            "id": r.id,
            "temperature": r.temperature,
            "moisture": r.moisture,
            "humidity": r.humidity,
            "timestamp": str(r.timestamp)
        }
        for r in reversed(latest_20_rows)
    ]

    previous_ai_data = None
    if previous_ai:
        previous_ai_data = ai_analysis_to_dict(previous_ai)

    prompt = f"""
You are CompostIQ AI, an expert compost monitoring assistant.

You must analyze a small compost batch using:
1. Initial batch setup
2. First 20 sensor readings as baseline
3. Latest 20 sensor readings as current trend
4. Previous AI analysis for consistency
5. Latest sensor reading

Important stability rules:
- Do not change ready_days drastically unless sensor readings are critical.
- Maturity should increase gradually.
- Health may change based on current conditions.
- Use previous AI analysis as memory.
- This is a 400 ml small compost container, so composting may be slower and less hot than large piles.

Batch setup:
{json.dumps(BATCH_SETUP)}

First 20 sensor readings:
{json.dumps(first_20)}

Latest 20 sensor readings:
{json.dumps(latest_20)}

Previous AI analysis:
{json.dumps(previous_ai_data, default=str)}

Latest reading:
{{
  "temperature": {latest.temperature},
  "moisture": {latest.moisture},
  "humidity": {latest.humidity}
}}

Return ONLY valid JSON in exactly this format:
{{
  "maturity": 0,
  "health": 0,
  "phase": "Mesophilic",
  "ready_days": 0,
  "next_action": "Keep Monitoring",
  "confidence": 0,
  "reason": "short reason under 20 words"
}}

Allowed phase values:
Mesophilic, Thermophilic, Cooling, Maturing

Allowed next_action values:
Add Water, Add Dry Material, Turn Compost, Keep Monitoring
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        ai_data = json.loads(response.text)
        return validate_ai_output(ai_data, previous_ai)

    except Exception as e:
        result = fallback_analysis(latest)
        result["reason"] = f"Gemini failed. {result['reason']}"
        result["ai_error"] = str(e)[:150]
        return result

def batch_setup_to_dict(row):
    return {
        "batch_id": row.batch_id,
        "container_ml": row.container_ml,
        "dry_leaves_g": row.dry_leaves_g,
        "grass_clippings_g": row.grass_clippings_g,
        "vegetable_peels_g": row.vegetable_peels_g,
        "coffee_grounds_g": row.coffee_grounds_g,
        "cardboard_tissue_g": row.cardboard_tissue_g,
        "twigs_g": row.twigs_g,
        "start_temp": row.start_temp,
        "start_humidity": row.start_humidity,
        "start_moisture_raw": row.start_moisture_raw,
        "start_moisture_state": row.start_moisture_state,
        "updated_at": row.updated_at
    }


def get_or_create_batch_setup(db):
    setup = db.query(BatchSetup).first()

    if setup:
        return setup

    setup = BatchSetup(
        batch_id=BATCH_SETUP["batch_id"],
        container_ml=BATCH_SETUP["container_ml"],
        dry_leaves_g=BATCH_SETUP["dry_leaves_g"],
        grass_clippings_g=BATCH_SETUP["grass_clippings_g"],
        vegetable_peels_g=BATCH_SETUP["vegetable_peels_g"],
        coffee_grounds_g=BATCH_SETUP["coffee_grounds_g"],
        cardboard_tissue_g=BATCH_SETUP["cardboard_tissue_g"],
        twigs_g=BATCH_SETUP["twigs_g"],
        start_temp=BATCH_SETUP["start_temp"],
        start_humidity=BATCH_SETUP["start_humidity"],
        start_moisture_raw=BATCH_SETUP["start_moisture_raw"],
        start_moisture_state=BATCH_SETUP["start_moisture_state"]
    )

    db.add(setup)
    db.commit()
    db.refresh(setup)

    return setup

def notification_to_dict(row):
    return {
        "id": row.id,
        "event_type": row.event_type,
        "title": row.title,
        "message": row.message,
        "action": row.action,
        "severity": row.severity,
        "source_ai_id": row.source_ai_id,
        "source_sensor_id": row.source_sensor_id,
        "sent": row.sent,
        "created_at": row.created_at
    }


def create_notification_if_new(db, event_type, title, message, action=None, severity=None, source_ai_id=None, source_sensor_id=None):
    existing = (
        db.query(NotificationLog)
        .filter(NotificationLog.event_type == event_type)
        .filter(NotificationLog.action == action)
        .filter(NotificationLog.source_ai_id == source_ai_id)
        .filter(NotificationLog.source_sensor_id == source_sensor_id)
        .first()
    )

    if existing:
        return existing, False

    notification = NotificationLog(
        event_type=event_type,
        title=title,
        message=message,
        action=action,
        severity=severity,
        source_ai_id=source_ai_id,
        source_sensor_id=source_sensor_id,
        sent="no"
    )

    db.add(notification)
    db.commit()
    db.refresh(notification)

    return notification, True

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

        return {
            "message": "Data stored successfully",
            "data": data.model_dump()
        }

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
                "timestamp": d.timestamp
            }
            for d in data
        ]

    finally:
        db.close()


@app.get("/analysis")
def get_analysis():
    db = SessionLocal()

    try:
        latest_ai = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).first()

        if latest_ai:
            return ai_analysis_to_dict(latest_ai)

        latest_sensor = db.query(SensorData).order_by(desc(SensorData.id)).first()

        if not latest_sensor:
            return {"message": "No sensor data available"}

        # No Gemini call here. Only fallback if no saved AI exists.
        result = fallback_analysis(latest_sensor)
        saved = save_ai_analysis(db, latest_sensor, result)

        return ai_analysis_to_dict(saved)

    finally:
        db.close()


@app.post("/analysis/run")
def run_analysis(force: bool = False):
    db = SessionLocal()

    try:
        latest_sensor = db.query(SensorData).order_by(desc(SensorData.id)).first()

        if not latest_sensor:
            return {"message": "No sensor data available"}

        previous_ai = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).first()

        sensor_count_after_previous = 0
        if previous_ai and previous_ai.sensor_data_id:
            sensor_count_after_previous = (
                db.query(SensorData)
                .filter(SensorData.id > previous_ai.sensor_data_id)
                .count()
            )

        should_run, reason = should_run_gemini(
            latest_sensor,
            previous_ai,
            sensor_count_after_previous
        )

        if not force and not should_run and previous_ai:
            result = ai_analysis_to_dict(previous_ai)
            result["run_status"] = "skipped"
            result["skip_reason"] = reason
            result["new_sensor_rows"] = sensor_count_after_previous
            return result

        result = run_gemini_analysis(db, latest_sensor, previous_ai)
        saved = save_ai_analysis(db, latest_sensor, result)

        response = ai_analysis_to_dict(saved)
        response["run_status"] = "created"
        response["run_reason"] = "Forced run." if force else reason
        response["new_sensor_rows"] = sensor_count_after_previous

        return response

    finally:
        db.close()


@app.get("/analysis/status")
def analysis_status():
    db = SessionLocal()

    try:
        latest_sensor = db.query(SensorData).order_by(desc(SensorData.id)).first()
        latest_ai = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).first()

        if not latest_sensor:
            return {"message": "No sensor data available"}

        if not latest_ai:
            return {
                "latest_sensor_id": latest_sensor.id,
                "has_ai_analysis": False,
                "message": "No AI analysis yet. Run POST /analysis/run."
            }

        sensor_count_after_previous = (
            db.query(SensorData)
            .filter(SensorData.id > latest_ai.sensor_data_id)
            .count()
        )

        should_run, reason = should_run_gemini(
            latest_sensor,
            latest_ai,
            sensor_count_after_previous
        )

        return {
            "latest_sensor_id": latest_sensor.id,
            "latest_ai_id": latest_ai.id,
            "latest_ai_source": latest_ai.source,
            "latest_ai_created_at": latest_ai.created_at,
            "new_sensor_rows_since_ai": sensor_count_after_previous,
            "should_run_gemini_now": should_run,
            "reason": reason,
            "gemini_interval_minutes": GEMINI_MIN_INTERVAL_MINUTES,
            "gemini_min_new_rows": GEMINI_MIN_NEW_ROWS
        }

    finally:
        db.close()

@app.get("/analysis/history")
def analysis_history():
    db = SessionLocal()

    try:
        rows = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).limit(20).all()

        return [
            {
                "id": row.id,
                "batch_id": row.batch_id,
                "sensor_data_id": row.sensor_data_id,
                "maturity": row.maturity,
                "health": row.health,
                "phase": row.phase,
                "ready_days": row.ready_days,
                "next_action": row.next_action,
                "confidence": row.confidence,
                "reason": row.reason,
                "source": row.source,
                "model_name": row.model_name,
                "created_at": row.created_at
            }
            for row in rows
        ]

    finally:
        db.close()


@app.get("/dashboard-summary")
def dashboard_summary():
    db = SessionLocal()

    try:
        latest_sensor = db.query(SensorData).order_by(desc(SensorData.id)).first()
        latest_ai = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).first()

        sensor_rows = db.query(SensorData).order_by(desc(SensorData.id)).limit(30).all()
        sensor_rows = list(reversed(sensor_rows))

        ai_rows = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).limit(10).all()

        if not latest_sensor:
            return {
                "message": "No sensor data available",
                "device": {
                    "status": "offline",
                    "last_seen": None
                }
            }

        latest_sensor_dict = {
            "id": latest_sensor.id,
            "temperature": latest_sensor.temperature,
            "humidity": latest_sensor.humidity,
            "moisture": latest_sensor.moisture,
            "timestamp": latest_sensor.timestamp
        }

        latest_ai_dict = None

        if latest_ai:
            latest_ai_dict = {
                "id": latest_ai.id,
                "batch_id": latest_ai.batch_id,
                "sensor_data_id": latest_ai.sensor_data_id,
                "maturity": latest_ai.maturity,
                "health": latest_ai.health,
                "phase": latest_ai.phase,
                "ready_days": latest_ai.ready_days,
                "next_action": latest_ai.next_action,
                "confidence": latest_ai.confidence,
                "reason": latest_ai.reason,
                "source": latest_ai.source,
                "model_name": latest_ai.model_name,
                "created_at": latest_ai.created_at
            }

        now = datetime.utcnow()
        last_seen = latest_sensor.timestamp

        if last_seen and (now - last_seen).total_seconds() <= 180:
            device_status = "online"
        else:
            device_status = "offline"

        trends = [
            {
                "id": row.id,
                "temperature": row.temperature,
                "humidity": row.humidity,
                "moisture": row.moisture,
                "timestamp": row.timestamp
            }
            for row in sensor_rows
        ]

        recent_ai = [
            {
                "id": row.id,
                "maturity": row.maturity,
                "health": row.health,
                "phase": row.phase,
                "ready_days": row.ready_days,
                "next_action": row.next_action,
                "confidence": row.confidence,
                "reason": row.reason,
                "source": row.source,
                "created_at": row.created_at
            }
            for row in ai_rows
        ]

        return {
            "latest_sensor": latest_sensor_dict,
            "latest_analysis": latest_ai_dict,
            "device": {
                "name": "Compost Pile 1",
                "status": device_status,
                "last_seen": latest_sensor.timestamp
            },
            "batch_setup": batch_setup_to_dict(get_or_create_batch_setup(db)),
            "trends": trends,
            "recent_ai": recent_ai
        }

    finally:
        db.close()

@app.get("/batch-setup")
def get_batch_setup():
    db = SessionLocal()

    try:
        setup = get_or_create_batch_setup(db)
        return batch_setup_to_dict(setup)

    finally:
        db.close()


@app.put("/batch-setup")
def update_batch_setup(data: BatchSetupInput):
    db = SessionLocal()

    try:
        setup = get_or_create_batch_setup(db)

        setup.batch_id = data.batch_id
        setup.container_ml = data.container_ml
        setup.dry_leaves_g = data.dry_leaves_g
        setup.grass_clippings_g = data.grass_clippings_g
        setup.vegetable_peels_g = data.vegetable_peels_g
        setup.coffee_grounds_g = data.coffee_grounds_g
        setup.cardboard_tissue_g = data.cardboard_tissue_g
        setup.twigs_g = data.twigs_g
        setup.start_temp = data.start_temp
        setup.start_humidity = data.start_humidity
        setup.start_moisture_raw = data.start_moisture_raw
        setup.start_moisture_state = data.start_moisture_state
        setup.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(setup)

        return {
            "message": "Batch setup updated successfully",
            "batch_setup": batch_setup_to_dict(setup)
        }

    finally:
        db.close()

@app.get("/alerts")
def get_alerts():
    db = SessionLocal()

    try:
        latest_sensor = db.query(SensorData).order_by(desc(SensorData.id)).first()
        latest_ai = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).first()

        if not latest_sensor:
            return {
                "alerts": [
                    {
                        "type": "Warning",
                        "title": "No Sensor Data",
                        "message": "No sensor readings received yet.",
                        "severity": "warning"
                    }
                ]
            }

        alerts = []

        # Moisture alerts
        if latest_sensor.moisture < 30:
            alerts.append({
                "type": "Critical",
                "title": "Moisture Too Low",
                "message": "Compost is too dry. Add water gradually.",
                "severity": "critical"
            })

        elif latest_sensor.moisture > 80:
            alerts.append({
                "type": "Warning",
                "title": "Moisture Too High",
                "message": "Compost is too wet. Add dry leaves or cardboard.",
                "severity": "warning"
            })

        # Temperature alerts
        if latest_sensor.temperature > 60:
            alerts.append({
                "type": "Critical",
                "title": "Temperature Too High",
                "message": "Turn compost and improve airflow.",
                "severity": "critical"
            })

        elif latest_sensor.temperature < 15:
            alerts.append({
                "type": "Warning",
                "title": "Temperature Too Low",
                "message": "Compost activity may be slow due to low temperature.",
                "severity": "warning"
            })

        # AI action alert
        if latest_ai and latest_ai.next_action != "Keep Monitoring":
            alerts.append({
                "type": "Action",
                "title": latest_ai.next_action,
                "message": latest_ai.reason,
                "severity": "action"
            })

        # Device offline alert
        now = datetime.utcnow()
        if latest_sensor.timestamp and (now - latest_sensor.timestamp).total_seconds() > 600:
            alerts.append({
                "type": "Warning",
                "title": "Device Offline",
                "message": "No M5Stack sensor data received for more than 10 minutes.",
                "severity": "warning"
            })

        if not alerts:
            alerts.append({
                "type": "Stable",
                "title": "No Critical Alerts",
                "message": "Compost conditions are currently stable.",
                "severity": "stable"
            })

        return {
            "latest_sensor_id": latest_sensor.id,
            "latest_ai_id": latest_ai.id if latest_ai else None,
            "alerts": alerts
        }

    finally:
        db.close()

@app.get("/notifications/check")
def check_notifications():
    db = SessionLocal()

    try:
        latest_sensor = db.query(SensorData).order_by(desc(SensorData.id)).first()
        latest_ai = db.query(AIAnalysis).order_by(desc(AIAnalysis.id)).first()

        if not latest_sensor:
            notification, created = create_notification_if_new(
                db=db,
                event_type="offline",
                title="CompostIQ Sensor Missing",
                message="No sensor data has been received yet.",
                severity="warning"
            )

            return {
                "should_notify": created,
                "notification": notification_to_dict(notification)
            }

        # 1. Critical moisture
        if latest_sensor.moisture < 30:
            notification, created = create_notification_if_new(
                db=db,
                event_type="critical",
                title="Compost Too Dry",
                message="Moisture is too low. Add water gradually.",
                action="Add Water",
                severity="critical",
                source_sensor_id=latest_sensor.id
            )

            return {
                "should_notify": created,
                "notification": notification_to_dict(notification)
            }

        if latest_sensor.moisture > 80:
            notification, created = create_notification_if_new(
                db=db,
                event_type="critical",
                title="Compost Too Wet",
                message="Moisture is too high. Add dry leaves or cardboard.",
                action="Add Dry Material",
                severity="critical",
                source_sensor_id=latest_sensor.id
            )

            return {
                "should_notify": created,
                "notification": notification_to_dict(notification)
            }

        # 2. Critical temperature
        if latest_sensor.temperature > 60:
            notification, created = create_notification_if_new(
                db=db,
                event_type="critical",
                title="Temperature Too High",
                message="Temperature is too high. Turn compost and improve airflow.",
                action="Turn Compost",
                severity="critical",
                source_sensor_id=latest_sensor.id
            )

            return {
                "should_notify": created,
                "notification": notification_to_dict(notification)
            }

        # 3. Device offline
        now = datetime.utcnow()

        if latest_sensor.timestamp and (now - latest_sensor.timestamp).total_seconds() > 600:
            notification, created = create_notification_if_new(
                db=db,
                event_type="offline",
                title="M5Stack Device Offline",
                message="No sensor data received for more than 10 minutes.",
                severity="warning",
                source_sensor_id=latest_sensor.id
            )

            return {
                "should_notify": created,
                "notification": notification_to_dict(notification)
            }

        # 4. AI action change / action required
        if latest_ai and latest_ai.next_action != "Keep Monitoring":
            previous_same_action = (
                db.query(NotificationLog)
                .filter(NotificationLog.event_type == "action")
                .filter(NotificationLog.action == latest_ai.next_action)
                .order_by(desc(NotificationLog.id))
                .first()
            )

            if not previous_same_action or previous_same_action.source_ai_id != latest_ai.id:
                notification, created = create_notification_if_new(
                    db=db,
                    event_type="action",
                    title="CompostIQ Action Required",
                    message=f"{latest_ai.next_action}: {latest_ai.reason}",
                    action=latest_ai.next_action,
                    severity="action",
                    source_ai_id=latest_ai.id,
                    source_sensor_id=latest_ai.sensor_data_id
                )

                return {
                    "should_notify": created,
                    "notification": notification_to_dict(notification)
                }

        return {
            "should_notify": False,
            "notification": None,
            "message": "No new notification needed."
        }

    finally:
        db.close()

@app.get("/notifications/history")
def notification_history():
    db = SessionLocal()

    try:
        rows = db.query(NotificationLog).order_by(desc(NotificationLog.id)).limit(30).all()

        return [
            notification_to_dict(row)
            for row in rows
        ]

    finally:
        db.close()