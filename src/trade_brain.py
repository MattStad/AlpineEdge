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
# PROMPT ENGINEERING (Reasoning Focused)
# ---------------------------------------------------------------------------

JSON_SCHEMA_BLOCK = """
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "max_position_pct": 0.0 to 0.10,
  "horizon_days": 10 to 30,
  "reason": "short explanation of your synthesis"
}
"""


def build_swarm_prompt(ticker: str, processed_data: Dict[str, Any], macro_context: Dict[str, str],
                       company_news: List[str]) -> str:
    # Daten Formatierung
    perf = processed_data.get("performance", {})
    perf_str = ", ".join([f"{k}: {v}%" for k, v in perf.items() if v is not None])

    metrics = processed_data.get("metrics", {})
    rsi = metrics.get('rsi_14', 'N/A')
    metrics_str = f"Vol: {metrics.get('vol_annual', 'N/A')}%, Sharpe: {metrics.get('sharpe_annual', 'N/A')}, MaxDD: {metrics.get('max_drawdown', 'N/A')}%, RSI: {rsi}"

    comp_news_str = "\n".join(company_news[:5]) if company_news else "No specific news available."

    return f"""
You are an expert financial analyst with a "Contrarian Value" mindset.
You do not follow simple rules. You think deeply about the narrative vs. the data.

TARGET: {ticker}

=== 1. THE DATA (The Truth) ===
Performance History: {perf_str}
Risk Metrics: {metrics_str}

=== 2. THE NARRATIVE (The Noise?) ===
Company News:
{comp_news_str}

Macro Context (Global/Local):
Global: {macro_context.get('global', 'N/A')[:300]}...
Local: {macro_context.get('austria', 'N/A')[:300]}...

=== YOUR TASK ===
Synthesize these inputs to form a conviction.
Ask yourself:
1. **Discrepancy:** Is the price dropping (bad performance) but the news is actually good/neutral? (Opportunity?)
2. **Momentum:** Is the price rising on strong volume/news, confirming a breakout?
3. **Overreaction:** Is the market panicking over global news that doesn't affect this specific local company?

Do not be afraid to take a stance. If the data supports a move, recommend it.
Only recommend HOLD if you genuinely have no edge or signals are perfectly conflicting.

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
        "options": {"temperature": 0.3, "num_ctx": 4096}  # Etwas mehr Temp für Kreativität
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

            color = "\033[92m" if d.action == "BUY" else "\033[91m" if d.action == "SELL" else "\033[93m"
            reset = "\033[0m"
            print(f"    -> {color}{d.action}{reset} ({d.confidence:.2f}) : {d.reason}")

        except Exception as e:
            print(f"  > {model} failed: {e}")

    return decisions


def majority_vote(decisions: List[TradeDecision]) -> Dict[str, Any]:
    """
    Score-basiertes Voting (wie besprochen, das behalten wir bei,
    damit wir nicht wieder in die HOLD-Falle tappen).
    """
    if not decisions: return None

    score = 0
    buy_votes = 0
    sell_votes = 0
    hold_votes = 0

    for d in decisions:
        if d.action == "BUY":
            score += 1
            buy_votes += 1
        elif d.action == "SELL":
            score -= 1
            sell_votes += 1
        else:
            hold_votes += 1

    final_action = "HOLD"
    # Schwelle für BUY/SELL
    threshold = 1.5

    if score >= threshold:
        final_action = "BUY"
    elif score <= -threshold:
        final_action = "SELL"

    if final_action == "BUY":
        relevant = [d for d in decisions if d.action == "BUY"]
    elif final_action == "SELL":
        relevant = [d for d in decisions if d.action == "SELL"]
    else:
        relevant = [d for d in decisions if d.action == "HOLD"]
        if not relevant: relevant = decisions

    avg_conf = sum(d.confidence for d in relevant) / len(relevant) if relevant else 0.0

    return {
        "action": final_action,
        "vote_count": f"Score: {score} (B:{buy_votes}, S:{sell_votes}, H:{hold_votes})",
        "avg_confidence": round(avg_conf, 2),
        "details": relevant
    }


async def decide_for_ticker(client: HttpClient, ticker: str, processed_data: Dict, macro_context: Dict,
                            company_news: List):
    decisions = await get_ai_trade_votes(client, ticker, processed_data, macro_context, company_news)
    final = majority_vote(decisions)
    return final, decisions