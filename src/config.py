TICKERS = [
    "ANDR.VI",    # Andritz AG
    "ATS.VI",     # AT&S Austria Technologie & Systemtechnik
    "BG.VI",   # BAWAG Group AG
    "CAI.VI",     # CA Immobilien Anlagen AG
    "CPI.VI",     # CPI Property Group (CPI Europe)
    "DOC.VI",     # DO & CO AG
    "EBS.VI",     # Erste Group Bank AG
    "EVN.VI",     # EVN AG
    "LNZ.VI",     # Lenzing AG
    "OMV.VI",     # OMV AG
    "POST.VI",    # Österreichische Post AG
    "POS.VI",    # PORR AG
    "RBI.VI",     # Raiffeisen Bank International AG
    "SBO.VI",     # Schoeller-Bleckmann Oilfield Equipment AG
    "STR.VI",     # Strabag SE
    "UQA.VI",     # UNIQA Insurance Group AG
    "VER.VI",     # Verbund AG
    "VIG.VI",     # Vienna Insurance Group AG
    "VOE.VI",     # voestalpine AG
    "WIE.VI"      # Wienerberger AG
]

FEEDS = {
    "AT_Wirtschaft": [
        "http://rss.ots.at/?wirtschaft",  # APA OTS Wirtschaft
        "https://www.derstandard.at/rss/wirtschaft",
        "https://diepresse.com/rss/wirtschaft",
        "https://kurier.at/wirtschaft/xml/rssd",
        "https://kurier.at/wirtschaftspolitik/xml/rssd",
        "https://kurier.at/wirtschaft/marktplatz/xml/rssd",
        "https://kurier.at/wirtschaft/Börse/xml/rssd",
        "https://kurier.at/wirtschaft/finanzen/xml/rssd",
        "https://rss.orf.at/news.xml",
    ],
    "AT_Finanzen": [
        #"https://www.boerse-express.com/rss/news",
        "https://www.cash.at/news/feed/handel",
        "https://www.cash.at/news/feed/industrie",
        # Trend / Brutkasten später per HTML-Scraper separat
    ],
    "Europa_Macro": [
        "https://de.investing.com/rss/news_11.rss",  # Börse Europa
        "https://de.investing.com/rss/news_95.rss",
        "https://feeds.feedburner.com/euronews/en/home/",
        "https://www.ecb.europa.eu/rss/press.html",
        "https://www.europarl.europa.eu/rss/doc/top-stories/en.xml",
    ],
    "Welt_Macro": [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",  # CNBC Finance
        "http://feeds.marketwatch.com/marketwatch/topstories/",
    ],
}
# config.py
from pathlib import Path

# Basis-Verzeichnis deines Projekts (Ordner "AlpineEdge")
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "news.db"
# falls du irgendwo einen String brauchst:
DB_PATH_STR = str(DB_PATH)
TECH_PATH = BASE_DIR / "data" / "raw"
TECH_DIR = str(TECH_PATH)