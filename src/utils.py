import logging, os, sqlite3
from logging.handlers import RotatingFileHandler

def get_logger(name="app", log_file="logs/app.log"):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

def ensure_sqlite(db_path="logs/predictions.db"):
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS requests(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT, sepal_length REAL, sepal_width REAL,
          petal_length REAL, petal_width REAL,
          pred INTEGER, latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()