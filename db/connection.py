"""
Datenbankverbindung zu PostgreSQL.

Das Schema wird von Studierenden eigenständig in pgAdmin angelegt (A0_Setup.ipynb).
Diese Datei stellt nur die Verbindungsfunktion bereit.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import psycopg2
from config import DB_CONFIG


def get_connection():
    """Baut eine Verbindung zur zentralen PostgreSQL-Datenbank auf."""
    return psycopg2.connect(**DB_CONFIG)
