from app.db import engine
from app.models import Base

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()