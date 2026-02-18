# src/trade_brain.py

from __future__ import annotations
import json
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import requests

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"

# Dein lokaler Swarm
OLLAMA_SWARM_MODELS = [
    "llama3.1",
    "qwen2.5",
    "mistral-nemo",
    "phi3.5",
    "gemma2",
]


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------

class HttpClient:
    async def post_json(self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None,
                        timeout_sec: int = 120) -> Tuple[int, Any]:
        def _do_request():
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
                status = resp.status_code
                try:
                    data = resp.json()
                except Exception:
                    data = resp.text
                return status, data
            except requests.exceptions.RequestException as e:
                return 0, str(e)

        return await asyncio.to_thread(_do_request)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    ticker: str
    agent: str
    action: str
    confidence: float
    max_position_pct: float
    horizon_days: int
    reason: str


# ---------------------------------------------------------------------------
# PROMPT ENGINEERING (OPTIMIZED)
# ---------------------------------------------------------------------------

JSON_SCHEMA_BLOCK = """
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "max_position_pct": 0.0 to 0.10,
  "horizon_days": 10 to 30,
  "reason": "concise explanation, max 200 chars"
}
"""


def build_swarm_prompt(ticker: str, processed_data: Dict[str, Any], macro_context: Dict[str, str],
                       company_news: List[str]) -> str:
    """
    Erstellt einen balancierten Prompt, der FOMO verhindert.
    """

    # 1. Performance Table formatieren
    perf = processed_data.get("performance", {})
    perf_str = ", ".join([f"{k}: {v}%" for k, v in perf.items() if v is not None])

    # 2. Risk Metrics formatieren
    metrics = processed_data.get("metrics", {})
    metrics_str = f"Volatility: {metrics.get('vol_annual', 'N/A')}%, Sharpe: {metrics.get('sharpe_annual', 'N/A')}, MaxDD: {metrics.get('max_drawdown', 'N/A')}%"

    # 3. News formatieren
    comp_news_str = "\n".join(company_news[:5]) if company_news else "No specific news available."

    return f"""
You are a disciplined Senior Portfolio Manager at a top-tier Quant Fund. 
Your philosophy is: "Protect Capital First, Capture Alpha Second."
You do NOT chase hype. You look for asymmetric risk/reward.

ANALYSIS TARGET: {ticker}

=== 1. MARKET REGIME (CONTEXT) ===
Global News: {macro_context.get('global', 'N/A')[:400]}...
Local News: {macro_context.get('austria', 'N/A')[:400]}...

=== 2. ASSET DATA ===
Performance: {perf_str}
Risk Stats: {metrics_str}
Current Price: {metrics.get('current_price', 'N/A')}

=== 3. LATEST INTEL ===
{comp_news_str}

=== 4. ANALYSIS FRAMEWORK ===
A) **Valuation & Overextension Check:** - Has the asset rallied too hard, too fast (e.g. >50% YTD)? If yes, consider SELLING/TAKING PROFITS (Mean Reversion risk).
   - Is the 1-week performance negative while the long-term is positive? This might be a "Buy the Dip" opportunity.

B) **Risk Check:**
   - Is the Sharpe Ratio below 0.5? If yes, the return does not justify the volatility -> HOLD or SELL.
   - Are there negative news headlines? If yes, be defensive.

C) **Decision Logic:**
   - **BUY**: Only if Trend is Up AND News is Good AND Asset is NOT overextended. (High conviction).
   - **SELL**: If Trend is broken, News is bad, OR Asset is significantly overbought/euphoric.
   - **HOLD**: If signals are mixed, volatility is too high, or you lack conviction.

OUTPUT:
Return ONLY a single JSON object.
{JSON_SCHEMA_BLOCK}
""".strip()


# ---------------------------------------------------------------------------
# API Logic
# ---------------------------------------------------------------------------

def _clamp_action(action: str) -> str:
    action = (action or "").upper().strip()
    if action not in ("BUY", "SELL", "HOLD"): return "HOLD"
    return action


async def _call_ollama_model(client: HttpClient, model_name: str, prompt: str, ticker: str) -> TradeDecision:
    agent_name = f"ollama:{model_name}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096}  # Temperatur runter für mehr Disziplin
    }

    status, data = await client.post_json(OLLAMA_URL, payload)

    if status != 200 or not data:
        return TradeDecision(ticker, agent_name, "HOLD", 0.0, 0.0, 0, f"Error {status}")

    try:
        content = data["message"]["content"]
        obj = json.loads(content)
        return TradeDecision(
            ticker=ticker,
            agent=agent_name,
            action=_clamp_action(obj.get("action")),
            confidence=float(obj.get("confidence", 0.0)),
            max_position_pct=float(obj.get("max_position_pct", 0.0)),
            horizon_days=int(obj.get("horizon_days", 14)),
            reason=str(obj.get("reason", ""))[:200]
        )
    except Exception as e:
        return TradeDecision(ticker, agent_name, "HOLD", 0.0, 0.0, 0, f"Parse Error: {e}")


async def get_ai_trade_votes(client: HttpClient, ticker: str, processed_data: Dict, macro_context: Dict,
                             company_news: List) -> List[TradeDecision]:
    prompt = build_swarm_prompt(ticker, processed_data, macro_context, company_news)
    print(f"\n[SWARM] {ticker}: Agents analyzing...")

    decisions = []
    for model in OLLAMA_SWARM_MODELS:
        try:
            print(f"  > {model} thinking...")
            d = await _call_ollama_model(client, model, prompt, ticker)
            decisions.append(d)

            # Farb-Ausgabe für Konsole
            color = "\033[92m" if d.action == "BUY" else "\033[91m" if d.action == "SELL" else "\033[93m"
            reset = "\033[0m"
            print(f"    -> {color}{d.action}{reset} ({d.confidence:.2f}) : {d.reason}")

        except Exception as e:
            print(f"  > {model} failed: {e}")

    return decisions


def majority_vote(decisions: List[TradeDecision]) -> Dict[str, Any]:
    if not decisions: return None

    actions = [d.action for d in decisions]
    cnt = Counter(actions)
    if not cnt: return None

    best_action, count = cnt.most_common(1)[0]
    total_votes = len(decisions)

    # STRENGERE LOGIK:
    # Wir brauchen mindestens 3 Stimmen (bei 5 Agenten) für eine Richtung.
    # Sonst ist es "Uneinigkeit" -> HOLD.
    consensus_threshold = 3

    final_action = best_action

    if count < consensus_threshold:
        # Wenn z.B. 2 BUY, 2 HOLD, 1 SELL -> Kein Konsens -> HOLD
        final_action = "HOLD"
        print(f"    [VOTE] No clear majority ({count}/{total_votes} for {best_action}). Defaulting to HOLD.")

    # Filtere Details basierend auf final_action (oder best_action falls wir HOLD erzwingen)
    # Wir nehmen die Votes, die zur finalen Entscheidung passen.
    # Wenn Fallback auf HOLD, nehmen wir die HOLD votes.
    relevant = [d for d in decisions if d.action == final_action]

    # Falls wir auf HOLD gefallen sind, aber niemand HOLD gestimmt hat (z.B. 2 BUY, 2 SELL, 1 ???),
    # dann nehmen wir alle als Info.
    if not relevant and final_action == "HOLD":
        relevant = decisions
        avg_conf = 0.0
    else:
        avg_conf = sum(d.confidence for d in relevant) / len(relevant) if relevant else 0.0

    return {
        "action": final_action,
        "vote_count": f"{count}/{total_votes} (Original: {best_action})",
        "avg_confidence": round(avg_conf, 2),
        "details": relevant
    }


async def decide_for_ticker(client: HttpClient, ticker: str, processed_data: Dict, macro_context: Dict,
                            company_news: List):
    decisions = await get_ai_trade_votes(client, ticker, processed_data, macro_context, company_news)
    final = majority_vote(decisions)
    return final, decisions