import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from app.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
