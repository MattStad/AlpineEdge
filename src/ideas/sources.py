# sources.py
# Wir nutzen RSS Feeds, das ist am stabilsten.

FEEDS = {
    "AT_Wirtschaft": [
        "https://www.ots.at/rss/b0012", # APA OTS Wirtschaft (Goldstandard!)
        "https://www.derstandard.at/rss/wirtschaft",
        "https://diepresse.com/rss/wirtschaft",
        "https://kurier.at/wirtschaft/xml",
        "https://rss.orf.at/wirtschaft.xml"
    ],
    "AT_Finanzen": [
        "https://www.boerse-express.com/rss/news",
        "https://www.cash.at/rss",
        # Trend und Brutkasten haben oft keine sauberen RSS,
        # hier müssten wir später spezieller scrapen.
    ],
    "Europa_Macro": [
        "https://de.investing.com/rss/news_11.rss", # Börse Europa
        "https://www.euractiv.de/section/finanzen-und-wirtschaft/feed/",
        "https://www.ecb.europa.eu/rss/press.html" # EZB Pressemitteilungen
    ],
    "Welt_Macro": [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", # CNBC Finance
        "http://feeds.marketwatch.com/marketwatch/topstories/"
    ]
}

# Deine Watchlist für Keyword-Filterung (damit wir nicht ALLES lesen)
WATCHLIST = [
    "ATX", "OMV", "Erste Group", "Raiffeisen", "RBI", "Voestalpine", "Verbund",
    "Andritz", "Wienerberger", "BAWAG", "CA Immo", "Uniqa", "VIG",
    "Zinsen", "EZB", "Inflation", "Ölpreis", "Gaspreis"
]