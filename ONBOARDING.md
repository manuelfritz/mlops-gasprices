# BIS2242 – Onboarding: Benzinpreis-MLOps-Projekt

**Kurs:** BIS2242 Produktionsreife KI-Systeme in der Praxis (SS 2026)
**Hochschule:** Hochschule Pforzheim
**Dozent:** Prof. Dr. Manuel Fritz

---

## Überblick: Was ist das Ziel?

Du baust eine vollständige **MLOps-Pipeline** für die Vorhersage von Kraftstoffpreisen.
In 5 Assignments lernst du alle Schritte vom Datenimport bis zum Monitoring:

```
A1: Datenerfassung  →  A2: Feature Engineering  →  A3: Training (MLflow)
                                                           │
A5: Monitoring (Evidently) ←─────────────  A4: API (FastAPI + Web-UI)
```

**Technologie-Stack:**
- Python 3.11+ · pandas · scikit-learn · MLflow · FastAPI · DVC · Evidently · psycopg2
- PostgreSQL (gemeinsamer Server: 141.47.5.55)
- MLflow Tracking Server: http://141.47.5.55:5000
- MinIO Object Store: http://141.47.5.55:9001 (Web-UI)

---

## 1. Voraussetzungen (Tools installieren)

### Python 3.11+
Lade Python von [python.org](https://www.python.org/downloads/) herunter.
Bei der Installation **"Add Python to PATH"** anklicken!

Überprüfung:
```bash
python --version   # Muss 3.11+ anzeigen
pip --version
```

### Git
Lade Git von [git-scm.com](https://git-scm.com/downloads) herunter.

Überprüfung:
```bash
git --version
```

### Git mit SSH einrichten (empfohlen)

Damit du ohne Passwort-Eingabe auf GitHub pushen und pullen kannst, richtest du einmalig
einen SSH-Schlüssel ein. Das Prinzip: Du erzeugst ein Schlüsselpaar (privater Schlüssel
bleibt auf deinem Rechner, öffentlicher Schlüssel wird bei GitHub hinterlegt).

**Schritt-für-Schritt-Anleitung:** [GitHub Docs – Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

Kurzfassung:
```bash
# 1. Schlüsselpaar erzeugen (einmalig)
ssh-keygen -t ed25519 -C "deine@email.de"
# → Datei wird standardmäßig in ~/.ssh/id_ed25519 gespeichert

# 2. Öffentlichen Schlüssel anzeigen und kopieren
cat ~/.ssh/id_ed25519.pub

# 3. Schlüssel bei GitHub hinterlegen:
#    github.com → Settings → SSH and GPG keys → New SSH key → Inhalt einfügen

# 4. Verbindung testen
ssh -T git@github.com
# Erwartete Ausgabe: "Hi <username>! You've successfully authenticated..."
```

Danach das Repository mit SSH-URL klonen:
```bash
git clone git@github.com:<username>/mlops-gasprices.git
```

### Virtuelle Umgebung (empfohlen)
```bash
# Virtuelle Umgebung anlegen
python -m venv .venv

# Aktivieren (Windows):
.venv\Scripts\activate

# Aktivieren (macOS/Linux):
source .venv/bin/activate

# Prompt wechselt zu (.venv) ...
```

> **Conda-Alternative:** Falls du Anaconda nutzt:
> ```bash
> conda create -n bis2242 python=3.11
> conda activate bis2242
> ```

---

## 2. Schritt-für-Schritt-Setup

### Schritt 1: Repository klonen
```bash
git clone <repo-url>
cd mlops-gasprices
```

> Die `<repo-url>` erhältst du vom Dozenten oder über das Kurs-Portal.

### Schritt 2: Virtuelle Umgebung anlegen und aktivieren
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
```

### Schritt 3: config.py anpassen
Öffne `config.py` und passe diese Zeilen an:

```python
GROUP_ID       = "gruppe_03"   # → Deine Gruppe, z.B. "gruppe_03" (vom Dozenten)
GROUP_PASSWORD = "..."         # → Dein Passwort (vom Dozenten per E-Mail)
STATION_ID     = "UUID..."     # → UUID einer Tankstelle (aus Schritt 4)
FUEL_TYPE      = "e5"          # → Kraftstoffsorte: "e5", "e10" oder "diesel"
```

**WICHTIG:** `GROUP_ID` wird als PostgreSQL-Schema verwendet. Nur:
- Kleinbuchstaben (a-z)
- Ziffern (0-9)
- Unterstriche (_)
- Kein Leerzeichen, kein Bindestrich!

### Schritt 4: A0_Setup.ipynb ausführen
```bash
jupyter lab notebooks/A0_Setup.ipynb
```

Dieses Notebook:
- Installiert alle Python-Pakete
- Testet die Datenbankverbindung
- Hilft bei der Tankstellen-Auswahl (`STATION_ID`)
- Legt die `predictions`-Tabelle in deinem Schema an (`gruppe_XX`.predictions)
- Konfiguriert den DVC Remote (MinIO)

---

## 3. Assignments – Überblick und Deadlines

| Assignment | Thema | Punkte | Deadline |
|---|---|---|---|
| A1 | Datenerfassung | 8 | 09.04.2026 |
| A2 | Feature Engineering & DVC | 17 | 23.04.2026 |
| A3 | Training & MLflow | 17 | 14.05.2026 |
| A4 | API & Prediction App | 12 | 04.06.2026 |
| A5 | Monitoring & Governance | 6 | 18.06.2026 |
| **Gesamt** | | **60** | |

Starte immer mit dem jeweiligen Notebook (`notebooks/A1_Datenerfassung.ipynb` usw.)
und folge den TODO-Anweisungen.

---

## 4. Git-Workflow für Abgaben

### Grundprinzip
Dein Code lebt in einem **Git-Repository**. Vor jeder Deadline commit und push!

### Empfohlener Workflow
```bash
# 1. Status prüfen – was hat sich geändert?
git status

# 2. Relevante Dateien stagen
git add notebooks/A1_Datenerfassung.ipynb
# NICHT: git add data/  (Daten werden über DVC versioniert!)

# 3. Committen
git commit -m "A1: fetch_price_history + resample_to_hourly implementiert"

# 4. Pushen (vor der Deadline!)
git push
```

### Commit-Konventionen
```
A1: Kurze Beschreibung was implementiert wurde
A2: Feature Engineering vollständig
A3: 3 MLflow Runs + Model Registry
```

### Was committieren? Was NICHT?
| Committieren ✓ | NICHT committieren ✗ |
|---|---|
| `*.py`, `*.ipynb` | `data/` (zu groß, via DVC) |
| `dvc.yaml`, `params.yaml` | `.venv/` |
| `*.dvc` Dateien | `__pycache__/` |
| `reports/drift_summary.json` | `reports/drift_report.html` (groß) |

> ℹ️ `.gitignore` ist bereits konfiguriert – du musst nichts extra ausschließen.

---

## 5. DVC-Workflow

### Was ist DVC?
**Data Version Control** – wie Git, aber für Daten und ML-Pipelines.
DVC trackt große Dateien (CSV, Modelle) und verwaltet die Pipeline-Abhängigkeiten.

### Warum DVC?
```
Ohne DVC: "Welche CSV hat zu welchem Modell geführt?" → Rätselraten
Mit DVC:  dvc repro  → Pipeline neu ausführen, exakt reproduzierbar
```

### Wichtigste Befehle
```bash
# Pipeline (komplett) neu ausführen:
dvc repro

# Nur einzelne Stages:
dvc repro ingest      # A1: Daten laden
dvc repro preprocess  # A2: Features erstellen
dvc repro train       # A3: Modell trainieren

# Trockentest (was würde ausgeführt?):
dvc repro --dry

# Status prüfen (ist alles aktuell?):
dvc status

# Abhängigkeitsgraph anzeigen:
dvc dag
```

### DVC-Pipeline in dvc.yaml
Die `dvc.yaml` enthält eine Skelett-Vorlage – du füllst in A2 die `cmd`, `deps` und `outs`
für jede Stage aus. Beispiel-Struktur:

```yaml
stages:
  ingest:                         # A1
    cmd: ...                      # TODO
    outs: [data/raw/prices.csv]

  preprocess:                     # A2
    cmd: ...                      # TODO
    deps: [data/raw/prices.csv]
    outs: [data/processed/features.csv]

  train:                          # A3
    cmd: ...                      # TODO
    deps: [data/processed/features.csv]
```

---

## 6. Nützliche Links

| Ressource | URL |
|---|---|
| MLflow UI (Experimente) | http://141.47.5.55:5000 |
| MinIO Web-UI (Dateien) | http://141.47.5.55:9001 |
| Swagger UI (nach uvicorn) | http://localhost:8000/docs |
| Web-Frontend (nach uvicorn) | http://localhost:8000 |
| pandas Dokumentation | https://pandas.pydata.org/docs/ |
| scikit-learn Dokumentation | https://scikit-learn.org/stable/ |
| MLflow Dokumentation | https://mlflow.org/docs/latest/ |
| DVC Dokumentation | https://dvc.org/doc |
| FastAPI Dokumentation | https://fastapi.tiangolo.com/ |
| Evidently Dokumentation | https://docs.evidentlyai.com/ |

---

## 7. Häufige Probleme & Lösungen

### PostgreSQL-Verbindung schlägt fehl
```
psycopg2.OperationalError: could not connect to server
```
**Lösung:**
- Bist du im HfWU-WLAN oder VPN? (Server ist nur im Hochschulnetz erreichbar)
- Ist `GROUP_ID` und `GROUP_PASSWORD` in `config.py` korrekt?
- Test: `ping 141.47.5.55`

### MLflow nicht erreichbar
```
MLflowException: Could not connect to MLflow tracking server
```
**Lösung:**
- Hochschulnetz / VPN prüfen
- `MLFLOW_TRACKING_URI` in `config.py`: `http://141.47.5.55:5000`
- Alternativ als Umgebungsvariable:
  ```bash
  set MLFLOW_TRACKING_URI=http://141.47.5.55:5000  # Windows
  export MLFLOW_TRACKING_URI=http://141.47.5.55:5000  # Linux/macOS
  ```

### `dvc repro` schlägt fehl
```
ERROR: failed to reproduce 'preprocess'
```
**Lösung:**
- `dvc.yaml` vollständig ausgefüllt? (A2-Notebook)
- Abhängigkeiten installiert? `pip install -r requirements.txt`
- Fehler-Details: `dvc repro -v` (verbose)

### `NotImplementedError` beim Ausführen
**Ursache:** Ein TODO ist noch nicht implementiert.
**Lösung:** Lies die Aufgabenbeschreibung im Notebook und implementiere die Funktion.

### JupyterLab startet nicht
```bash
# Sicherstellen dass .venv aktiviert ist:
.venv\Scripts\activate

# JupyterLab installieren:
pip install jupyterlab

# Starten:
jupyter lab
```

### `GROUP_ID` ungültig
```
ERROR: invalid schema name "gruppe 03"   ← Leerzeichen!
ERROR: invalid schema name "Gruppe_03"  ← Großbuchstaben!
```
**Lösung:** `GROUP_ID` in `config.py` muss exakt dem vom Dozenten vorgegebenen Namen entsprechen:
- ✓ `"gruppe_03"`
- ✗ `"Gruppe_03"` (Großbuchstaben)
- ✗ `"gruppe 03"` (Leerzeichen)
- ✗ `"gruppe-03"` (Bindestrich)

### FastAPI-App startet nicht (A4)
```
ModuleNotFoundError: No module named 'src.predict'
```
**Lösung:** uvicorn im Projektverzeichnis starten (nicht im `app/`-Ordner):
```bash
# Korrekt (vom Projektverzeichnis):
cd mlops-gasprices
uvicorn app.main:app --reload --port 8000

# FALSCH (aus app/ heraus):
# cd app && uvicorn main:app  ← Pfade stimmen nicht!
```

### Evidently schlägt fehl (A5)
```
ImportError: cannot import name 'DataDriftPreset'
```
**Lösung:** Evidently-Version prüfen:
```bash
pip show evidently   # Muss 0.4.x sein
pip install "evidently==0.4.30"
```

---

## 8. Abgabe-Checkliste (vor jeder Deadline)

Für jedes Assignment vor der Abgabe prüfen:

**Allgemein:**
- [ ] Virtuelle Umgebung aktiviert?
- [ ] Alle TODO-Zellen implementiert (kein `raise NotImplementedError` mehr)?
- [ ] Notebook komplett durchgelaufen (alle Zellen ausgeführt)?
- [ ] `git push` ausgeführt?

**A1 spezifisch:**
- [ ] `data/raw/prices.csv` vorhanden?
- [ ] Keine NaN-Werte in der Preisspalte (z.B. `e5_price`)?
- [ ] Mindestens 1000 stündliche Einträge vorhanden?

**A2 spezifisch:**
- [ ] `data/processed/features.csv` vorhanden?
- [ ] Mindestens 3 kreative Features implementiert?
- [ ] `dvc.yaml` ausgefüllt und `dvc repro` läuft durch?

**A3 spezifisch:**
- [ ] Mindestens 3 MLflow Runs vorhanden (MLflow UI prüfen)?
- [ ] Modell in Model Registry registriert?

**A4 spezifisch:**
- [ ] FastAPI startet ohne Fehler?
- [ ] `POST /predict` gibt vorhergesagten Preis zurück?
- [ ] Vorhersagen werden in DB gespeichert (pgAdmin prüfen)?
- [ ] HTML-Frontend im Browser sichtbar?

**A5 spezifisch:**
- [ ] `reports/drift_report.html` vorhanden?
- [ ] `reports/drift_summary.json` vorhanden (mit `drifted: true`)?
- [ ] Mindestens 10 Vorhersagen mit `actual_price` in der DB?

---

*BIS2242 SS 2026 · Hochschule Pforzheim · Prof. Dr. Manuel Fritz*
