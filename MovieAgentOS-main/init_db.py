from backend.db.database import init_db, DB_PATH

init_db()

print(f"database creato correttamente in: {DB_PATH}")