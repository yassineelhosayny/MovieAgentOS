from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

#ottenere il path completto
DB_DIR = Path(__file__).resolve().parent 
#generare il db nella stessa cartella
DB_PATH = DB_DIR / "movieagentos.sqlite3"

DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)
#creato un event per garantire che i dati sarrano sempre coherente, in caso di eliminazione
@event.listens_for(engine, "connect")
def abilita_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

#inizializzare db con le tabelle di models
def init_db():
    from backend.db import models
    Base.metadata.create_all(bind=engine)


