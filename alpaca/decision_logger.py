"""
decision_logger.py - Journal detaille des decisions du bot.

Chaque symbole analyse produit une ligne: signal technique, score contexte,
decision finale et raison. C'est indispensable pour comparer les decisions
acceptees/refusees et ameliorer le moteur sans deviner.
"""

import csv
import os
from datetime import datetime, timezone

from config_alpaca import DECISIONS_LOG_FILE


DECISION_FIELDS = [
    "date_utc",
    "symbol",
    "price",
    "trend",
    "technical_signal",
    "decision",
    "reason",
    "context_score",
    "context_confidence",
    "news_score",
    "technical_context_score",
    "tradingview_score",
    "earnings_score",
    "analyst_score",
    "macro_score",
    "article_count",
    "risk_flag",
    "context_reason",
]


def init_decisions_csv(path: str = DECISIONS_LOG_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=DECISION_FIELDS).writeheader()


def log_decision(
    symbol: str,
    price: float,
    trend: str,
    technical_signal: str,
    decision: str,
    reason: str,
    context=None,
    path: str = DECISIONS_LOG_FILE,
) -> None:
    init_decisions_csv(path)
    row = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "price": round(float(price), 4) if price is not None else "",
        "trend": trend,
        "technical_signal": technical_signal,
        "decision": decision,
        "reason": reason,
        "context_score": "",
        "context_confidence": "",
        "news_score": "",
        "technical_context_score": "",
        "tradingview_score": "",
        "earnings_score": "",
        "analyst_score": "",
        "macro_score": "",
        "article_count": "",
        "risk_flag": "",
        "context_reason": "",
    }
    if context is not None:
        row.update({
            "context_score": round(context.score, 4),
            "context_confidence": round(context.confidence, 4),
            "news_score": round(context.news_score, 4),
            "technical_context_score": round(context.technical_context_score, 4),
            "tradingview_score": round(context.tradingview_score, 4),
            "earnings_score": round(context.earnings_score, 4),
            "analyst_score": round(context.analyst_score, 4),
            "macro_score": round(context.macro_score, 4),
            "article_count": context.article_count,
            "risk_flag": context.risk_flag,
            "context_reason": context.reason,
        })

    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=DECISION_FIELDS).writerow(row)
