# src/news/news_sentiment.py
"""
News Sentiment Analysis using local LLM.

Berechnet aggregiertes Sentiment für Ticker über einen Zeitraum.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sqlite3

import pandas as pd
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.config import DB_PATH


def get_db_connection():
    """Erstellt read-only DB Connection"""
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def analyze_headline_sentiment(headline: str, model: str = "llama3.2:3b") -> float:
    """
    Analysiert Sentiment eines einzelnen Headlines.
    
    Returns:
        Float zwischen -1 (very negative) und +1 (very positive)
    """
    if not OLLAMA_AVAILABLE:
        return 0.0
    
    prompt = f"""Rate the sentiment of this financial news headline on a scale from -1 (very negative) to +1 (very positive).

Consider:
- Negative: bankruptcy, losses, layoffs, scandals, downgrades, failures
- Positive: profits, growth, acquisitions, upgrades, innovation, partnerships
- Neutral: routine updates, factual reports without clear direction

Headline: "{headline}"

Respond with ONLY a single number between -1.0 and +1.0, nothing else."""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 20}
        )
        
        text = response["message"]["content"].strip()
        
        # Parse number from response
        try:
            sentiment = float(text)
            # Clamp to -1, +1
            return max(-1.0, min(1.0, sentiment))
        except ValueError:
            # Fallback: suche nach Zahlen im Text
            import re
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                sentiment = float(numbers[0])
                return max(-1.0, min(1.0, sentiment))
            return 0.0
    
    except Exception as e:
        print(f"[SENTIMENT] Error analyzing headline: {e}")
        return 0.0


def calculate_news_sentiment(
    ticker: str,
    days: int = 7,
    min_importance: int = 3,
    model: str = "llama3.2:3b"
) -> Dict[str, Any]:
    """
    Berechnet aggregiertes News Sentiment für einen Ticker.
    
    Args:
        ticker: Ticker Symbol (z.B. "OMV.VI")
        days: Zeitfenster in Tagen
        min_importance: Minimum Importance Score (0-10)
        model: Ollama Model für Sentiment Analysis
    
    Returns:
        Dict mit score, count, trend, recent_headlines
    """
    conn = get_db_connection()
    
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    query = """
        SELECT headline, summary, importance, published_at
        FROM news_articles
        WHERE classified_ticker = ?
          AND published_at >= ?
          AND importance >= ?
        ORDER BY published_at DESC
    """
    
    try:
        df = pd.read_sql(query, conn, params=[ticker, cutoff_date, min_importance])
    except Exception as e:
        print(f"[SENTIMENT] DB Error: {e}")
        conn.close()
        return {
            "score": 0.0,
            "count": 0,
            "trend": "neutral",
            "recent_headlines": []
        }
    finally:
        conn.close()
    
    if df.empty:
        return {
            "score": 0.0,
            "count": 0,
            "trend": "neutral",
            "recent_headlines": []
        }
    
    # Sentiment Analysis für jede Headline
    sentiments = []
    weighted_sentiments = []
    
    for _, row in df.iterrows():
        headline = row['headline']
        importance = row.get('importance', 5)
        
        # Analyze
        sentiment = analyze_headline_sentiment(headline, model)
        
        # Weight mit Importance (Importance 10 = 2x weight, Importance 5 = 1x weight)
        weight = importance / 5.0
        weighted = sentiment * weight
        
        sentiments.append(sentiment)
        weighted_sentiments.append(weighted)
    
    # Aggregated Scores
    avg_sentiment = np.mean(sentiments)
    weighted_avg = np.mean(weighted_sentiments)
    
    # Trend classification
    if weighted_avg > 0.3:
        trend = "bullish"
    elif weighted_avg < -0.3:
        trend = "bearish"
    else:
        trend = "neutral"
    
    # Recent headlines
    recent = df.head(5)['headline'].tolist()
    
    return {
        "score": round(weighted_avg, 2),
        "raw_score": round(avg_sentiment, 2),
        "count": len(df),
        "trend": trend,
        "recent_headlines": recent,
        "days_analyzed": days
    }


def calculate_sentiment_for_universe(
    universe: List[str],
    days: int = 7,
    min_importance: int = 3
) -> Dict[str, Dict[str, Any]]:
    """
    Berechnet Sentiment für alle Ticker im Universe.
    
    Returns:
        Dict[ticker, sentiment_dict]
    """
    print(f"[SENTIMENT] Analyzing sentiment for {len(universe)} tickers...")
    
    results = {}
    
    for ticker in universe:
        sentiment = calculate_news_sentiment(ticker, days, min_importance)
        results[ticker] = sentiment
        
        if sentiment['count'] > 0:
            print(f"  {ticker}: {sentiment['score']:+.2f} ({sentiment['trend']}) | {sentiment['count']} articles")
    
    return results


def format_sentiment_for_prompt(sentiment: Dict[str, Any]) -> str:
    """
    Formatiert Sentiment für LLM Prompt Integration.
    """
    if sentiment['count'] == 0:
        return "NEWS SENTIMENT: No recent news available."
    
    headlines_str = "\n  ".join(sentiment['recent_headlines'][:3])
    
    return f"""
NEWS SENTIMENT ({sentiment['days_analyzed']} days):
  Score: {sentiment['score']:+.2f} ({sentiment['trend'].upper()})
  Articles: {sentiment['count']}
  Recent Headlines:
  {headlines_str}
"""


if __name__ == "__main__":
    # Test
    test_ticker = "OMV.VI"
    
    print(f"Testing sentiment analysis for {test_ticker}...\n")
    
    sentiment = calculate_news_sentiment(test_ticker, days=7, min_importance=3)
    
    print(f"\nResults:")
    print(f"  Score: {sentiment['score']:+.2f}")
    print(f"  Trend: {sentiment['trend']}")
    print(f"  Articles: {sentiment['count']}")
    print(f"\nRecent Headlines:")
    for headline in sentiment['recent_headlines']:
        print(f"  - {headline}")
    
    print(f"\n\nFormatted for Prompt:")
    print(format_sentiment_for_prompt(sentiment))
