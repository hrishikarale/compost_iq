from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Float, DateTime
from datetime import datetime


class Base(DeclarativeBase):
    pass


class SensorData(Base):
    __tablename__ = "sensor_data"

    id: Mapped[int] = mapped_column(primary_key=True)

    temperature: Mapped[float] = mapped_column(Float)
    moisture: Mapped[float] = mapped_column(Float)
    humidity: Mapped[float] = mapped_column(Float)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )