# src/news_classifier.py
"""
Klassifiziert News-Artikel automatisch:
- Ordnet Ticker zu (oder kategorisiert als ATX/World)
- Bewertet Wichtigkeit (importance: 0-10)

Verwendet ein lokales LLM via Ollama (empfohlen: llama3.2:3b)
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[CLASSIFIER] WARNUNG: ollama package nicht installiert. Installiere mit: pip install ollama")

from src.config import DB_PATH, TICKERS


# ---------------------------------------------------------------------------
# ATX Mapping & Ticker-Normalisierung
# ---------------------------------------------------------------------------

ATX_MAPPING: Dict[str, str] = {
    "ANDR.VI": "Andritz AG",
    "ATS.VI": "AT&S Austria Technologie & Systemtechnik",
    "BG.VI": "BAWAG Group AG",
    "CAI.VI": "CA Immobilien Anlagen AG",
    "CPI.VI": "CPI Property Group",
    "DOC.VI": "DO & CO AG",
    "EBS.VI": "Erste Group Bank AG",
    "EVN.VI": "EVN AG",
    "LNZ.VI": "Lenzing AG",
    "OMV.VI": "OMV AG",
    "POST.VI": "Österreichische Post AG",
    "POS.VI": "PORR AG",
    "RBI.VI": "Raiffeisen Bank International AG",
    "SBO.VI": "Schoeller-Bleckmann Oilfield Equipment AG",
    "STR.VI": "Strabag SE",
    "UQA.VI": "UNIQA Insurance Group AG",
    "VER.VI": "Verbund AG",
    "VIG.VI": "Vienna Insurance Group AG",
    "VOE.VI": "voestalpine AG",
    "WIE.VI": "Wienerberger AG",
}

VALID_ATX_TICKERS = list(ATX_MAPPING.keys())


def _normalize_token(s: str) -> str:
    """Hilfsfunktion: lowercase + nur Buchstaben/Ziffern."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# MANUELLE ALIASES – bereits normalisiert
MANUAL_ALIASES = {
    _normalize_token("BAWAG"): "BG.VI",
    _normalize_token("BAWAG GROUP"): "BG.VI",
    _normalize_token("BAWAG GROUP AG"): "BG.VI",
    _normalize_token("BAWAG.VI"): "BG.VI",
    _normalize_token("BG.VI"): "BG.VI",

    _normalize_token("OMV"): "OMV.VI",
    _normalize_token("OMV AG"): "OMV.VI",
    _normalize_token("OMV.VI"): "OMV.VI",
    _normalize_token("OMVKY"): "OMV.VI",

    _normalize_token("voestalpine"): "VOE.VI",
    _normalize_token("voestalpine ag"): "VOE.VI",

    _normalize_token("Erste Group"): "EBS.VI",
    _normalize_token("Erste Group Bank"): "EBS.VI",

    _normalize_token("Raiffeisen Bank International"): "RBI.VI",

    _normalize_token("Vienna Insurance Group"): "VIG.VI",
    _normalize_token("UNIQA"): "UQA.VI",

    _normalize_token("Österreichische Post"): "POST.VI",
    _normalize_token("Oesterreichische Post"): "POST.VI",
    _normalize_token("Austrian Post"): "POST.VI",

    _normalize_token("Wienerberger"): "WIE.VI",
    _normalize_token("Lenzing"): "LNZ.VI",
    _normalize_token("DO & CO"): "DOC.VI",
    _normalize_token("DOCO"): "DOC.VI",

    # aus deinen Logs:
    _normalize_token("LENZ.VI"): "LNZ.VI",
    
    # Weitere häufige Aliases
    _normalize_token("VIENNA INSURANCE"): "VIG.VI",
    _normalize_token("VIENNA INSURANCE GROUP"): "VIG.VI",
    _normalize_token("VIG"): "VIG.VI",
    
    _normalize_token("ERSTE"): "EBS.VI",
    _normalize_token("ERSTE BANK"): "EBS.VI",
    _normalize_token("EBS"): "EBS.VI",
    
    _normalize_token("RBI"): "RBI.VI",
    _normalize_token("RAIFFEISEN"): "RBI.VI",
    
    _normalize_token("VERBUND"): "VER.VI",
    _normalize_token("VER"): "VER.VI",
    
    _normalize_token("VOEST"): "VOE.VI",
    _normalize_token("VOESTALPINE"): "VOE.VI",
    
    _normalize_token("ANDRITZ"): "ANDR.VI",
    _normalize_token("ANDR"): "ANDR.VI",
    
    _normalize_token("AT&S"): "ATS.VI",
    _normalize_token("ATS"): "ATS.VI",
    
    _normalize_token("STRABAG"): "STR.VI",
    _normalize_token("STR"): "STR.VI",
    
    _normalize_token("CA IMMO"): "CAI.VI",
    _normalize_token("CA IMMOBILIEN"): "CAI.VI",
    
    _normalize_token("POST"): "POST.VI",
    _normalize_token("ÖSTERREICHISCHE POST"): "POST.VI",
    _normalize_token("AUSTRIAN POST"): "POST.VI",
}


def map_llm_ticker_to_atx(ticker_raw: Optional[str]) -> Optional[str]:
    """
    Nimmt das vom LLM gelieferte 'ticker'-Feld (z.B. 'BAWAG GROUP AG', 'OMVKY')
    und versucht es auf einen gültigen ATX-Ticker wie 'BG.VI' zu mappen.
    """

    if ticker_raw is None:
        return None

    t = str(ticker_raw).strip()
    if not t:
        return None

    # Werte wie "null", "None", "N/A" → kein Ticker
    if t.lower() in {"null", "none", "n/a", "na"}:
        return None

    t_upper = t.upper()

    # 1) Direkter Treffer
    if t_upper in VALID_ATX_TICKERS:
        return t_upper

    # 2) Ticker-Format XYZ.VI → Basiscode matchen
    if t_upper.endswith(".VI"):
        base = t_upper[:-3]
        for atx_ticker in VALID_ATX_TICKERS:
            if atx_ticker.startswith(base + "."):
                return atx_ticker

    # 3) Basiscode als Substring (BAWAG.VI, OMVKY, BBVA.VI etc.)
    basecode_candidates = [x.split(".")[0] for x in VALID_ATX_TICKERS]  # ANDR, BG, OMV, ...
    for base in basecode_candidates:
        if base in t_upper:
            for atx_ticker in VALID_ATX_TICKERS:
                if atx_ticker.startswith(base + "."):
                    return atx_ticker

    # 4) Aliases & Firmennamen über normalisierte Tokens
    t_norm = _normalize_token(t)

    # 4a) direkter Alias
    if t_norm in MANUAL_ALIASES:
        return MANUAL_ALIASES[t_norm]

    # 4b) Alias als Substring
    for key, val in MANUAL_ALIASES.items():
        if key in t_norm:
            return val

    # 4c) Firmennamen aus ATX_MAPPING
    for atx_ticker, company in ATX_MAPPING.items():
        comp_norm = _normalize_token(company)
        if t_norm == comp_norm or t_norm in comp_norm or comp_norm in t_norm:
            return atx_ticker

    # Nichts erkannt
    return None


# ---------------------------------------------------------------------------
# Dataklassen
# ---------------------------------------------------------------------------

@dataclass
class NewsClassification:
    """Ergebnis der News-Klassifizierung"""
    ticker: Optional[str]  # z.B. "VOE.VI" oder None
    category: str          # "ticker", "ATX", "World"
    importance: int        # 0-10
    reasoning: str         # Begründung vom Modell


# ---------------------------------------------------------------------------
# Haupt-Classifier
# ---------------------------------------------------------------------------

class NewsClassifier:
    """
    Klassifiziert News mit einem lokalen LLM (via Ollama).

    Unterstützte Modelle (Beispiele):
    - llama3.2:3b (3B)  – schnell, gut für Klassifizierung
    - llama3.1:8b (8B)  – präziser, etwas langsamer
    - mistral:7b        – sehr gut für Finanz-News
    - phi3:mini         – Microsoft, Business-optimiert
    """

    def __init__(
        self,
        model_name: str | None = None,  # Auto-detect wenn None
        tickers: List[str] | None = None,
        temperature: float = 0.1,
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package nicht verfügbar. Installiere mit: pip install ollama")

        self.tickers = tickers or TICKERS
        self.temperature = temperature

        # Teste Ollama-Verbindung
        try:
            available_models = ollama.list()
        except Exception as e:
            raise RuntimeError(
                f"Ollama Server nicht erreichbar. Stelle sicher, dass Ollama läuft.\n"
                f"Installation: https://ollama.ai\n"
                f"Starte Server: ollama serve\n"
                f"Fehler: {e}"
            )

        # Auto-detect Modell wenn nicht angegeben
        if model_name is None:
            model_name = self._auto_detect_model(available_models)
            print(f"[CLASSIFIER] Auto-detected Modell: {model_name}")

        self.model_name = model_name

        # Validiere, dass Modell verfügbar ist
        self._validate_model(available_models)

    # --------------------- Modell-Handling ---------------------

    def _auto_detect_model(self, available_models: dict) -> str:
        """Findet automatisch das beste verfügbare Modell"""
        models = available_models.get("models", [])

        if not models:
            raise RuntimeError(
                "Kein Ollama-Modell installiert!\n"
                "Installiere z.B.:\n"
                "  ollama pull llama3.2:3b\n"
                "  ollama pull llama3.1:8b\n"
                "  ollama pull mistral:7b\n"
                "Verfügbare Modelle: ollama list"
            )

        model_names: List[str] = []
        for m in models:
            name = None
            if hasattr(m, "model"):
                name = m.model
            elif hasattr(m, "name"):
                name = m.name
            elif isinstance(m, dict):
                name = m.get("model") or m.get("name")

            if name and isinstance(name, str):
                model_names.append(name.strip())

        if not model_names:
            raise RuntimeError(
                "Keine gültigen Modell-Namen gefunden!\n"
                f"Debug: models={models}\n"
                "Prüfe installierte Modelle: ollama list"
            )

        print(f"[CLASSIFIER] Gefundene Modelle: {model_names}")

        preferences = [
            "llama3.2:3b",
            "llama3.2",
            "llama3.1:8b",
            "llama3.1:latest",
            "llama3.1",
            "mistral:7b",
            "mistral",
            "phi3:mini",
            "phi3",
        ]

        for pref in preferences:
            if pref in model_names:
                print(f"[CLASSIFIER] Gefundenes Modell (exakt): {pref}")
                return pref

        for pref in preferences:
            base_name = pref.split(":")[0]
            for name in model_names:
                if base_name in name:
                    print(f"[CLASSIFIER] Gefundenes Modell (partial match): {name}")
                    return name

        print(f"[CLASSIFIER] Verwende erstes verfügbares Modell: {model_names[0]}")
        return model_names[0]

    def _validate_model(self, available_models: dict):
        """Prüft, ob das gewählte Modell verfügbar ist"""
        models = available_models.get("models", [])

        model_names: List[str] = []
        for m in models:
            name = None
            if hasattr(m, "model"):
                name = m.model
            elif hasattr(m, "name"):
                name = m.name
            elif isinstance(m, dict):
                name = m.get("model") or m.get("name")

            if name and isinstance(name, str):
                model_names.append(name.strip())

        if self.model_name not in model_names:
            partial_match = [m for m in model_names if self.model_name.split(":")[0] in m]
            if partial_match:
                print(f"[CLASSIFIER] Hinweis: '{self.model_name}' nicht gefunden, verwende '{partial_match[0]}'")
                self.model_name = partial_match[0]
            else:
                raise ValueError(
                    f"Modell '{self.model_name}' nicht gefunden!\n"
                    f"Verfügbare Modelle: {', '.join(model_names)}\n\n"
                    f"Installiere das Modell:\n"
                    f"  ollama pull {self.model_name}\n"
                    f"Oder verwende ein verfügbares Modell."
                )

    # --------------------- Prompt-Bau --------------------------

    def _build_prompt(self, headline: str, summary: str, raw_text: str) -> str:
        """Erstellt den Prompt für das LLM"""

        text_snippet = raw_text[:2000] if raw_text else ""

        ticker_list = "\n".join(
            [f"  {ticker} - {name}" for ticker, name in ATX_MAPPING.items()]
        )

        prompt = f"""Du bist ein Finanz-News-Analyst. Analysiere den folgenden Artikel und klassifiziere ihn.

VERFÜGBARE ATX-TICKER:
{ticker_list}

ARTIKEL:
Headline: {headline}
Summary: {summary or "N/A"}
Text-Auszug: {text_snippet}

AUFGABE:
1. Bestimme, ob der Artikel einem spezifischen ATX-Ticker zugeordnet werden kann
   - Suche nach Firmennamen oder direkten Ticker-Erwähnungen
   - WICHTIG: Verwende NUR Ticker aus der obigen Liste!

2. Falls kein spezifischer Ticker passt, kategorisiere als:
   - "ATX" wenn es um österreichische Wirtschaft/Börse/mehrere ATX-Firmen geht
   - "World" für internationale Wirtschaftsnachrichten

3. Bewerte die Wichtigkeit von 0-10:
   - 0-2: Unwichtig (Gerüchte, irrelevant, allgemeine Börsenberichte)
   - 3-4: Mäßig wichtig (Marktkommentare, Branchen-News)
   - 5-6: Wichtig (konkrete Unternehmensnachrichten)
   - 7-8: Sehr wichtig (Quartalszahlen, wichtige Ankündigungen)
   - 9-10: Kritisch (Earnings, M&A, regulatorische Änderungen, CEO-Wechsel)

ANTWORTE NUR MIT DIESEM JSON-FORMAT (keine zusätzlichen Texte):
{{
  "ticker": "TICKER.VI oder null",
  "category": "ticker/ATX/World",
  "importance": 5,
  "reasoning": "Kurze Begründung"
}}"""
        return prompt

    # --------------------- Klassifizierung ---------------------

    def classify_article(
        self,
        headline: str,
        summary: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> NewsClassification:
        """
        Klassifiziert einen einzelnen Artikel.

        Returns:
            NewsClassification mit ticker, category, importance
        """

        prompt = self._build_prompt(headline, summary or "", raw_text or "")

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.temperature,
                    "num_predict": 200,  # Output-Länge begrenzen
                },
            )

            response_text = response["message"]["content"].strip()

            # JSON aus der Antwort extrahieren
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                raise ValueError(f"Kein JSON in Response gefunden: {response_text}")

            result = json.loads(json_match.group())

            raw_ticker = result.get("ticker")
            category = result.get("category", "World")
            importance = int(result.get("importance", 5))
            reasoning = result.get("reasoning", "")

            # "null"/None etc. früh abfangen
            if isinstance(raw_ticker, str) and raw_ticker.lower() in {"null", "none", "n/a", "na"}:
                raw_ticker = None

            mapped_ticker = map_llm_ticker_to_atx(raw_ticker)

            # Nur loggen, wenn LLM irgendwas Sinnvolles behauptet hat, wir aber nichts mappen konnten
            if raw_ticker and not mapped_ticker:
                print(f"[CLASSIFIER] WARNUNG: '{raw_ticker}' ist kein (erkennbarer) ATX-Ticker, setze auf None")

            ticker = mapped_ticker

            # importance clampen
            importance = max(0, min(10, importance))

            # Wenn Ticker gesetzt, Kategorie = "ticker"
            if ticker:
                category = "ticker"

            return NewsClassification(
                ticker=ticker,
                category=category,
                importance=importance,
                reasoning=reasoning,
            )

        except Exception as e:
            print(f"[CLASSIFIER] Fehler bei Klassifizierung: {e}")
            return NewsClassification(
                ticker=None,
                category="World",
                importance=5,
                reasoning=f"Fehler bei Klassifizierung: {e}",
            )

    def classify_batch(
        self,
        articles: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[NewsClassification]:
        """
        Klassifiziert mehrere Artikel.

        Args:
            articles: Liste von Dicts mit keys: headline, summary, raw_text
            verbose: Ob Fortschritt geloggt werden soll
        """
        results: List[NewsClassification] = []

        for i, article in enumerate(articles):
            if verbose and i % 10 == 0:
                print(f"[CLASSIFIER] Klassifiziere Artikel {i + 1}/{len(articles)}")

            classification = self.classify_article(
                headline=article.get("headline", ""),
                summary=article.get("summary"),
                raw_text=article.get("raw_text"),
            )
            results.append(classification)

        return results


# ---------------------------------------------------------------------------
# DB-Schema & Batch-Verarbeitung
# ---------------------------------------------------------------------------

def update_db_schema():
    """
    Erweitert die news_articles Tabelle um die neuen Klassifizierungs-Felder.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(news_articles);")
    columns = [row[1] for row in cur.fetchall()]

    if "classified_ticker" not in columns:
        print("[CLASSIFIER] Füge Spalte 'classified_ticker' hinzu")
        cur.execute("ALTER TABLE news_articles ADD COLUMN classified_ticker TEXT;")

    if "classified_category" not in columns:
        print("[CLASSIFIER] Füge Spalte 'classified_category' hinzu")
        cur.execute("ALTER TABLE news_articles ADD COLUMN classified_category TEXT;")

    if "importance" not in columns:
        print("[CLASSIFIER] Füge Spalte 'importance' hinzu")
        cur.execute("ALTER TABLE news_articles ADD COLUMN importance INTEGER;")

    if "classification_reasoning" not in columns:
        print("[CLASSIFIER] Füge Spalte 'classification_reasoning' hinzu")
        cur.execute("ALTER TABLE news_articles ADD COLUMN classification_reasoning TEXT;")

    if "classified_at" not in columns:
        print("[CLASSIFIER] Füge Spalte 'classified_at' hinzu")
        cur.execute("ALTER TABLE news_articles ADD COLUMN classified_at TEXT;")

    conn.commit()
    conn.close()
    print("[CLASSIFIER] DB-Schema aktualisiert")


def classify_unclassified_news(
    batch_size: int = 50,
    max_articles: Optional[int] = None,
    model_name: str | None = None,  # Auto-detect wenn None
    min_importance: int = 0,
) -> int:
    """
    Klassifiziert alle News ohne Klassifizierung in der DB.
    """

    update_db_schema()

    try:
        classifier = NewsClassifier(model_name=model_name)
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ FEHLER BEI CLASSIFIER-INITIALISIERUNG")
        print("=" * 70)
        print(f"{e}\n")
        print("LÖSUNG:")
        print("  1. Stelle sicher, dass Ollama läuft: ollama serve")
        print("  2. Installiere ein Modell:")
        print("     ollama pull llama3.2:3b    (Schnell, empfohlen)")
        print("     ollama pull llama3.1:8b    (Präziser)")
        print("     ollama pull mistral:7b     (Finanz-News)")
        print("=" * 70 + "\n")
        raise

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total_classified = 0

    while True:
        cur.execute(
            """
            SELECT id, headline, summary, raw_text
            FROM news_articles
            WHERE importance IS NULL
              AND raw_text IS NOT NULL
              AND raw_text NOT LIKE '__HTTP%'
              AND raw_text NOT LIKE '__NO_TEXT%'
            ORDER BY id DESC
            LIMIT ?;
            """,
            (batch_size,),
        )
        rows = cur.fetchall()

        if not rows:
            print("[CLASSIFIER] Keine weiteren Artikel zu klassifizieren")
            break

        print(f"[CLASSIFIER] Klassifiziere Batch mit {len(rows)} Artikeln...")

        for article_id, headline, summary, raw_text in rows:
            classification = classifier.classify_article(headline, summary, raw_text)

            now_iso = datetime.utcnow().isoformat()

            cur.execute(
                """
                UPDATE news_articles
                SET classified_ticker = ?,
                    classified_category = ?,
                    importance = ?,
                    classification_reasoning = ?,
                    classified_at = ?
                WHERE id = ?;
                """,
                (
                    classification.ticker,
                    classification.category,
                    classification.importance,
                    classification.reasoning,
                    now_iso,
                    article_id,
                ),
            )

            total_classified += 1

            if total_classified % 10 == 0:
                conn.commit()
                print(f"[CLASSIFIER] {total_classified} Artikel klassifiziert...")

            if max_articles and total_classified >= max_articles:
                print(f"[CLASSIFIER] Max. Anzahl ({max_articles}) erreicht")
                conn.commit()
                conn.close()
                if min_importance > 0:
                    delete_low_importance_articles(min_importance)
                return total_classified

        conn.commit()

    conn.close()
    print(f"[CLASSIFIER] Insgesamt {total_classified} Artikel klassifiziert")

    if min_importance > 0:
        delete_low_importance_articles(min_importance)

    return total_classified


def delete_low_importance_articles(min_importance: int = 4) -> int:
    """
    Löscht Artikel mit importance < min_importance aus der DB.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DELETE FROM news_articles WHERE importance < ?;", (min_importance,))
    deleted = cur.rowcount

    conn.commit()
    conn.close()

    print(f"[CLASSIFIER] {deleted} Artikel mit importance < {min_importance} gelöscht")
    return deleted


def get_classified_news(
    min_importance: int = 5,
    category: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Holt klassifizierte News aus der DB mit Filtern.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT id, source, category, classified_ticker, classified_category,
               headline, url, published_at, importance, classification_reasoning,
               summary, raw_text
        FROM news_articles
        WHERE importance >= ?
    """
    params: List[Any] = [min_importance]

    if category:
        query += " AND classified_category = ?"
        params.append(category)

    if ticker:
        query += " AND classified_ticker = ?"
        params.append(ticker)

    query += " ORDER BY importance DESC, published_at DESC LIMIT ?;"
    params.append(limit)

    cur.execute(query, params)
    rows = cur.fetchall()
    articles = [dict(row) for row in rows]

    conn.close()
    return articles


def print_classification_stats():
    """Gibt Statistiken über die klassifizierten Artikel aus."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM news_articles;")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM news_articles WHERE importance IS NOT NULL;")
    classified = cur.fetchone()[0]

    cur.execute(
        """
        SELECT classified_category, COUNT(*) as cnt
        FROM news_articles
        WHERE importance IS NOT NULL
        GROUP BY classified_category;
        """
    )
    by_category = cur.fetchall()

    cur.execute(
        """
        SELECT 
            CASE 
                WHEN importance >= 8 THEN 'Sehr wichtig (8-10)'
                WHEN importance >= 5 THEN 'Wichtig (5-7)'
                WHEN importance >= 3 THEN 'Mäßig (3-4)'
                ELSE 'Unwichtig (0-2)'
            END as importance_range,
            COUNT(*) as cnt
        FROM news_articles
        WHERE importance IS NOT NULL
        GROUP BY importance_range
        ORDER BY importance_range DESC;
        """
    )
    by_importance = cur.fetchall()

    cur.execute(
        """
        SELECT classified_ticker, COUNT(*) as cnt
        FROM news_articles
        WHERE classified_ticker IS NOT NULL
        GROUP BY classified_ticker
        ORDER BY cnt DESC
        LIMIT 10;
        """
    )
    top_tickers = cur.fetchall()

    conn.close()

    print("\n" + "=" * 60)
    print("NEWS CLASSIFICATION STATISTIKEN")
    print("=" * 60)
    print(f"Gesamt Artikel: {total}")
    print(f"Klassifiziert: {classified} ({100 * classified / total:.1f}%)")
    print()

    print("Nach Kategorie:")
    for cat, cnt in by_category:
        print(f"  {cat or 'N/A'}: {cnt}")
    print()

    print("Nach Wichtigkeit:")
    for imp_range, cnt in by_importance:
        print(f"  {imp_range}: {cnt}")
    print()

    if top_tickers:
        print("Top 10 Ticker:")
        for ticker, cnt in top_tickers:
            print(f"  {ticker}: {cnt}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print_classification_stats()
    else:
        print("[CLASSIFIER] Starte Klassifizierung...")
        print("[CLASSIFIER] Stelle sicher, dass Ollama läuft und llama3.2 installiert ist:")
        print("[CLASSIFIER]   ollama pull llama3.2:3b")
        print()

        classified = classify_unclassified_news(
            batch_size=20,
            max_articles=None,   # Alle klassifizieren
            model_name="llama3.2:3b",
            min_importance=3,    # Artikel mit importance < 3 werden danach gelöscht
        )

        print(f"\n[CLASSIFIER] ✓ {classified} Artikel klassifiziert")
        print_classification_stats()
