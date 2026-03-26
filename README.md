# BIS2242 – Benzinpreis-MLOps-Projekt

**Kurs:** Produktionsreife KI-Systeme in der Praxis (SS 2026)
**Aufgabe:** Vollständige MLOps-Pipeline für Kraftstoffpreis-Vorhersage implementieren

---

## 1. Überblick

Du implementierst eine MLOps-Pipeline, die:
- Historische Kraftstoffpreise aus einer PostgreSQL-Datenbank abruft
- Features engineered (Lag-Features, Moving Average, Zeitmerkmale)
- Ein Regressionsmodell mit MLflow-Tracking trainiert und in der Model Registry verwaltet
- Eine REST-API mit FastAPI zur Vorhersage bereitstellt
- Modell-Drift mit Evidently erkennt und den Produktions-RMSE berechnet

**ML-Task:** Vorhersage des Kraftstoffpreises (e5 / e10 / diesel) für die nächste Stunde (Regression).

---

## 2. Architektur

```
Dozenten-Server                          Dein lokaler Rechner
┌──────────────────────────────┐         ┌──────────────────────────────────────┐
│  gasprices.*                 │         │  config.py  ← GROUP_ID, STATION_ID   │
│    (Lesezugriff für alle)    │◀─TCP───▶│             ← FUEL_TYPE              │
│                              │         │                                      │
│  gruppe_XX.predictions       │         │  notebooks/A0_Setup.ipynb            │
│    .actual_price  ←──────────┤         │  notebooks/A1_Datenerfassung.ipynb   │
│                              │         │  notebooks/A2_Feature_Engineering    │
│  MLflow   :5000              │         │  notebooks/A3_Training.ipynb         │
│  MinIO    :9000/:9001        │         │  notebooks/A4_API.ipynb              │
│  pgAdmin  :5050              │         │  notebooks/A5_Monitoring.ipynb       │
└──────────────────────────────┘         └──────────────────────────────────────┘
```

- **gasprices-Schema:** Vom Dozenten täglich aktualisierte Preishistorie – Lesezugriff für alle Gruppen.
- **gruppe_XX-Schema:** Dein persönliches Schema auf dem Server für Vorhersagen.
- **MLflow:** Experiment-Tracking und Model Registry für alle Gruppen.
- **MinIO:** S3-kompatibler Object Store für DVC-Datensätze und MLflow-Artefakte.

---

## 3. Setup

### 3.1 Abhängigkeiten installieren

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

Öffne und führe vollständig aus: **`notebooks/A0_Setup.ipynb`**

Das Notebook installiert alle Pakete und richtet die Umgebung ein.

### 3.2 config.py anpassen

Öffne `config.py` und setze vier Werte:

```python
GROUP_ID       = "gruppe_03"     # Deine Gruppe – vom Dozenten vorgegeben
GROUP_PASSWORD = "..."           # Dein Passwort – vom Dozenten per E-Mail
STATION_ID     = "UUID..."       # UUID der gewählten Tankstelle (Schritt 3 in A0)
FUEL_TYPE      = "e5"            # Kraftstoffsorte: "e5", "e10" oder "diesel"
```

> **Wichtig:** `GROUP_ID` wird als PostgreSQL-Schemaname verwendet.
> Nur Kleinbuchstaben, Ziffern und Unterstriche – exakt wie vom Dozenten angegeben.

### 3.3 A0_Setup.ipynb ausführen

```bash
jupyter lab notebooks/A0_Setup.ipynb
```

Das Notebook:
1. Installiert alle Python-Pakete
2. Testet PostgreSQL- und MLflow-Verbindung
3. Hilft bei der Tankstellen-Auswahl (`STATION_ID`)
4. Legt die `predictions`-Tabelle in deinem Schema an
5. Konfiguriert den DVC Remote (MinIO)

---

## 4. Aufgaben

Alle Aufgaben werden in **Jupyter Notebooks** im Ordner `notebooks/` implementiert.

### A1: Datenerfassung → `notebooks/A1_Datenerfassung.ipynb` (8 Punkte)

Kraftstoffpreise per SQL aus der Datenbank laden, auf Stundenbasis resamplen und als `data/raw/prices.csv` speichern.

| Funktion | Beschreibung |
|---|---|
| `fetch_price_history()` | Preishistorie per SQL aus `gasprices`-Schema laden |
| `resample_to_hourly()` | Auf stündliche Zeitreihe aggregieren |
| `run_ingestion()` | Orchestrierung: laden + resamplen + speichern |

---

### A2: Feature Engineering → `notebooks/A2_Feature_Engineering.ipynb` (17 Punkte)

Lag-Features, Moving Average und weitere Features berechnen, `data/processed/features.csv` speichern und DVC-Pipeline konfigurieren.

| Funktion | Beschreibung |
|---|---|
| `create_features()` | Lag-Features, Moving Average, Zeitmerkmale, Zielvariable |
| `run_preprocessing()` | Orchestrierung + `features.csv` speichern |

> ⚠️ Kein Shuffle! Temporale Reihenfolge muss erhalten bleiben.

**DVC-Pipeline:**

```bash
dvc repro           # Gesamte Pipeline ausführen (ingest → preprocess → train)
dvc repro ingest    # Nur eine Stage
dvc status          # Status prüfen
dvc dag             # Abhängigkeitsgraph anzeigen
```

---

### A3: Modelltraining → `notebooks/A3_Training.ipynb` (17 Punkte)

Regressionsmodell trainieren, mit MLflow tracken und in der Model Registry registrieren.

| Funktion | Beschreibung |
|---|---|
| `temporal_split()` | Train/Test-Split ohne Shuffle |
| `compute_metrics()` | RMSE, MAE, R² berechnen |
| `train_and_log()` | MLflow-Tracking + Model Registry |

**MLflow UI:** `http://141.47.5.55:5000`
Dein Experiment: `{FUEL_TYPE}-preis-{GROUP_ID}` (z.B. `e5-preis-gruppe_03`)

---

### A4: API → `notebooks/A4_API.ipynb` (12 Punkte)

FastAPI-Anwendung mit Vorhersage-Endpoint und Web-Frontend implementieren.

Implementierungsdateien:
- **`src/predict.py`** – Modell laden und Vorhersage berechnen
- **`app/main.py`** – FastAPI-Endpoints und Web-Frontend

```bash
# API starten:
uvicorn app.main:app --reload --port 8000

# Testen:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"station_id": "...", "predict_for": "2025-04-01T14:00:00"}'

# Swagger UI: http://localhost:8000/docs
# Web-Frontend: http://localhost:8000
```

---

### A5: Monitoring → `notebooks/A5_Monitoring.ipynb` (6 Punkte)

Modell-Drift mit Evidently erkennen, Produktions-RMSE berechnen.

Alle Funktionen werden direkt im Notebook implementiert.

Ergebnisse: `reports/drift_report.html` und `reports/drift_summary.json`

---

## 5. Projektstruktur

```
mlops-gasprices/
├── config.py                    ← HIER anpassen: GROUP_ID, GROUP_PASSWORD, STATION_ID, FUEL_TYPE
├── params.yaml                  ← ML-Parameter für DVC
├── dvc.yaml                     ← DVC-Pipeline (TODO: von Studierenden zu befüllen)
├── requirements.txt
├── notebooks/
│   ├── A0_Setup.ipynb           ← Zuerst ausführen!
│   ├── A1_Datenerfassung.ipynb
│   ├── A2_Feature_Engineering.ipynb
│   ├── A3_Training.ipynb
│   ├── A4_API.ipynb
│   └── A5_Monitoring.ipynb
├── src/
│   └── predict.py               ← A4: Inferenz-Modul (von app/main.py genutzt)
├── app/
│   └── main.py                  ← A4: FastAPI-Anwendung
├── db/
│   └── connection.py            ← Datenbankverbindung (vorgegeben)
├── data/
│   ├── raw/                     ← prices.csv (von A1, via DVC)
│   └── processed/               ← features.csv (von A2, via DVC)
└── reports/                     ← drift_report.html, drift_summary.json (von A5)
```

---

## 6. Abgabe

Den gesamten Projektordner `mlops-gasprices` als ZIP-Archiv in Moodle hochladen.
Alle Notebooks müssen vollständig ausgeführt sein (Zellen mit Ausgaben).

**Bewertung:**

| Aufgabe | Thema | Punkte |
|---|---|---|
| A1 | Datenerfassung | 8 |
| A2 | Feature Engineering & DVC | 17 |
| A3 | Training & MLflow | 17 |
| A4 | API & Prediction App | 12 |
| A5 | Monitoring & Governance | 6 |
| **Gesamt** | | **60** |

---

## 7. Ressourcen

| Ressource | Link |
|---|---|
| MLflow UI | `http://141.47.5.55:5000` |
| pgAdmin | `http://141.47.5.55:5050` |
| MinIO Web-UI | `http://141.47.5.55:9001` |
| Swagger UI (lokal) | `http://localhost:8000/docs` |
| MLflow Docs | https://mlflow.org/docs/latest/ |
| DVC Docs | https://dvc.org/doc |
| FastAPI Docs | https://fastapi.tiangolo.com/ |
| Evidently Docs | https://docs.evidentlyai.com/ |
| scikit-learn Docs | https://scikit-learn.org/stable/ |
