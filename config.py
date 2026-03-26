"""
Zentrale Konfiguration – von Studierenden anzupassen.

Schritt 1: GROUP_ID auf Ihre Gruppe setzen (z.B. "gruppe_03") – vom Dozenten vorgegeben.
Schritt 2: GROUP_PASSWORD auf das Gruppen-Passwort setzen – vom Dozenten per E-Mail.
Schritt 3: STATION_ID auf eine Tankstelle Ihrer Wahl setzen (UUID aus gasprices-Schema).
Schritt 4: FUEL_TYPE auf die gewünschte Kraftstoffsorte setzen ("e5", "e10" oder "diesel").
"""

# ── Von Studierenden anzupassen ──────────────────────────────────────────────

GROUP_ID = "gruppe_XX"            # TODO: Ihre Gruppe eintragen (z.B. "gruppe_03")
                                  # Wird als PostgreSQL-Schema + MLflow-Experiment verwendet.
                                  # Nur Kleinbuchstaben, Ziffern und Unterstriche!

GROUP_PASSWORD = "PASSWORT_EINTRAGEN"  # TODO: Gruppen-Passwort eintragen (vom Dozenten per E-Mail)

STATION_ID = "e1cafea0-3da4-4419-9013-e41754332638"  # Beispiel-Station – in A0 eine eigene wählen!

FUEL_TYPE = "e5"                  # TODO: Kraftstoffsorte wählen – "e5", "e10" oder "diesel"
                                  # Bestimmt, welche Preisspalte aus der DB gelesen wird
                                  # und wie das Modell sowie das MLflow-Experiment benannt werden.

# ── Server-Verbindung (vom Dozenten bereitgestellt) ──────────────────────────

SERVER_IP = "141.47.5.55"

MLFLOW_TRACKING_URI = f"http://{SERVER_IP}:5000"

DB_CONFIG = {
    "host": SERVER_IP,
    "port": 5432,
    "dbname": "mlops_db",
    "user": GROUP_ID,             # Jede Gruppe loggt mit ihrer eigenen Rolle ein
    "password": GROUP_PASSWORD,
}

# ── MinIO Artifact Store (vom Dozenten bereitgestellt) ───────────────────────
# MinIO ist ein S3-kompatibler Object Store auf dem Kurs-Server.
# Jede Gruppe hat ihr eigenes Unterverzeichnis – nur dort darf sie lesen/schreiben.
#   Web-UI:    http://<SERVER_IP>:9001  (zum Browsen eigener Dateien)
#   S3-API:    http://<SERVER_IP>:9000  (von DVC und MLflow genutzt)
#   DVC-Pfad:  s3://dvc/<GROUP_ID>/
#   MLflow:    s3://mlflow/<GROUP_ID>/artifacts/

MINIO_ENDPOINT   = f"http://{SERVER_IP}:9000"
MINIO_ACCESS_KEY = GROUP_ID        # Gleicher Name wie PostgreSQL-Nutzer
MINIO_SECRET_KEY = GROUP_PASSWORD  # Gleiches Passwort wie PostgreSQL

# MinIO-Credentials als Umgebungsvariablen setzen (wird von MLflow-Client + DVC genutzt).
# setdefault: bereits gesetzte Werte (z.B. aus .env) werden nicht überschrieben.
import os as _os
_os.environ.setdefault("AWS_ACCESS_KEY_ID",     MINIO_ACCESS_KEY)
_os.environ.setdefault("AWS_SECRET_ACCESS_KEY", MINIO_SECRET_KEY)
_os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MINIO_ENDPOINT)

# ── Dozenten-Schema (nur Lesezugriff für Studierende) ────────────────────────
# Enthält die Kraftstoffpreis-Historien (täglich aktualisiert).
# Tabellen: gas_station, gas_station_information_history
GASPRICES_SCHEMA = "gasprices"    # Vom Dozenten verwaltet – nicht ändern!

# ── ML-Parameter (können angepasst werden) ───────────────────────────────────

DEFAULT_PARAMS = {
    "features": {
        "target_column": f"{FUEL_TYPE}_price_next_hour",
    },
    "train": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "paths": {
        "raw_prices":  "data/raw/prices.csv",
        "processed":   "data/processed/features.csv",
        "drift_report": "reports/drift_report.html",
    },
    "mlflow": {
        "experiment_name": f"{FUEL_TYPE}-preis-{GROUP_ID}",
        "model_name":      f"{FUEL_TYPE}-preis-regressor-{GROUP_ID}",
        "tracking_uri":    MLFLOW_TRACKING_URI,
    },
}
