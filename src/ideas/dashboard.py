import streamlit as st
import sqlite3
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Konfiguration der Seite
st.set_page_config(page_title="ServusAlpha Dashboard", layout="wide")


import os

THIS_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(THIS_DIR, "trading_bot.db")

def get_connection():
    return sqlite3.connect(DB_PATH)


def load_market_status():
    conn = get_connection()
    # Hole den allerneuesten Eintrag
    df = pd.read_sql("SELECT * FROM memory ORDER BY id DESC LIMIT 1", conn)
    conn.close()
    return df.iloc[0] if not df.empty else None


def load_signals():
    conn = get_connection()
    # Hole für jeden Ticker das NEUESTE Signal
    query = """
    SELECT t.ticker, t.price, t.rsi, t.sma_200, t.signal_score, t.details, t.timestamp
    FROM tech_signals t
    INNER JOIN (
        SELECT ticker, MAX(id) as max_id
        FROM tech_signals
        GROUP BY ticker
    ) tm ON t.ticker = tm.ticker AND t.id = tm.max_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_news_sentiment(ticker):
    """Holt news-Score (simuliert die Logik aus main_brain)"""
    conn = get_connection()
    clean_name = ticker.replace(".VI", "")
    query = f"SELECT sentiment_score FROM news WHERE title LIKE '%{clean_name}%' ORDER BY id DESC LIMIT 5"
    df = pd.read_sql(query, conn)
    conn.close()
    if not df.empty:
        return df['sentiment_score'].mean()
    return 0.0


# --- UI LAYOUT ---

st.title("🏔️ ServusAlpha - ATX Trading Bot")

# 1. MARKTLAGE (HEADER)
status = load_market_status()
col1, col2, col3 = st.columns(3)

if status is not None:
    score = status['market_score']
    color = "green" if score > 0 else "red"

    col1.metric("Marktstimmung (AI)", f"{score:.2f}", delta_color="normal")
    col2.info(f"💡 AI Fazit: {status['summary_text']}")
else:
    col1.warning("Noch keine Marktdaten.")

# 2. TABELLE DER SIGNALE
st.subheader("Aktuelle Signale")

df_signals = load_signals()

if not df_signals.empty:
    # Wir berechnen den 'Combined Score' live für die Anzeige
    results = []
    market_score = status['market_score'] if status is not None else 0.0

    for index, row in df_signals.iterrows():
        news_score = get_news_sentiment(row['ticker'])

        # Logik aus main_brain (kopiert für Visualisierung)
        if news_score == 0.0:
            total_score = (row['signal_score'] * 0.8) + (market_score * 0.2)
        else:
            total_score = (row['signal_score'] * 0.5) + (news_score * 0.3) + (market_score * 0.2)

        action = "HOLD"
        if total_score >= 0.6:
            action = "STRONG BUY 🚀"
        elif total_score >= 0.35:
            action = "BUY (Weak) ✅"
        elif total_score <= -0.2:
            action = "SELL ❌"

        results.append({
            "Ticker": row['ticker'],
            "Preis": f"€ {row['price']:.2f}",
            "RSI": f"{row['rsi']:.1f}",
            "Tech Score": row['signal_score'],
            "news Score": f"{news_score:.2f}",
            "Total Score": round(total_score, 2),
            "Action": action,
            "Grund": row['details']
        })

    df_display = pd.DataFrame(results)

    # Sortieren nach Score
    df_display = df_display.sort_values(by="Total Score", ascending=False)

    # Die Tabelle anzeigen
    st.dataframe(df_display, use_container_width=True)

    # 3. DETAIL ANSICHT (INTERAKTIV)
    st.markdown("---")
    st.subheader("Detail-Analyse")

    selected_ticker = st.selectbox("Wähle eine Aktie:", df_display["Ticker"].unique())

    if selected_ticker:
        st.write(f"Lade Chart für {selected_ticker}...")

        # Live Daten von Yahoo für den Chart holen
        try:
            stock_data = yf.download(selected_ticker, period="6mo", interval="1d", progress=False)

            # Fix für MultiIndex (wie im Tech-Bot)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(1)

            # Candlestick Chart mit Plotly
            fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                 open=stock_data['Open'],
                                                 high=stock_data['High'],
                                                 low=stock_data['Low'],
                                                 close=stock_data['Close'])])

            fig.update_layout(title=f"{selected_ticker} Kursverlauf", height=500)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Konnte Chart nicht laden: {e}")

else:
    st.info("Noch keine Signale in der Datenbank. Lass erst 'tech_brain.py' laufen!")

# Footer mit Refresh Button
if st.button("Daten aktualisieren"):
    st.rerun()