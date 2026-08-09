"""
portfolio_intelligence.py - Ranking, exposition, correlation et apprentissage leger.

Ce module ne remplace pas la strategie. Il decide quels signaux meritent d'etre
pris en premier, evite les concentrations evidentes, et garde une trace des
setups pour ameliorer les seuils avec les resultats reels/paper.
"""

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from config_alpaca import (
    ADAPTIVE_STATE_FILE,
    ADAPTIVE_THRESHOLD_ENABLED,
    ADAPTIVE_THRESHOLD_MAX,
    ADAPTIVE_THRESHOLD_MIN,
    ADAPTIVE_THRESHOLD_STEP,
    CORRELATION_LOOKBACK_BARS,
    MAX_AVG_CORRELATION_WITH_POSITIONS,
    MAX_POSITIONS_PER_GROUP,
    SETUP_RANKING_LOG_FILE,
    SYMBOL_TO_GROUP,
    TRADE_REVIEW_LOG_FILE,
    WATCHLIST_LOG_FILE,
    WATCHLIST_TOP_N,
)


RANKING_FIELDS = [
    "date_utc", "rank", "symbol", "group", "price", "signal",
    "tech_score", "context_score", "combined_score", "decision",
    "reason", "corr_to_positions", "analysis_reason", "context_reason",
]

WATCHLIST_FIELDS = [
    "date_utc", "rank", "symbol", "group", "price",
    "tech_score", "context_score", "combined_score", "note",
]

TRADE_REVIEW_FIELDS = [
    "date_utc", "symbol", "event", "qty", "price", "entry_price",
    "pnl_usd", "pnl_pct", "technical_score", "context_score",
    "combined_score", "exit_reason", "setup_reason",
]


def _ensure_csv(path: str, fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def symbol_group(symbol: str) -> str:
    return SYMBOL_TO_GROUP.get(symbol, "unknown")


def context_score(context) -> float:
    if context is None:
        return 0.0
    return float(getattr(context, "score", 0.0))


def combined_setup_score(analysis: dict, context=None) -> float:
    tech = float(analysis.get("score", 0.0))
    ctx = (context_score(context) + 1.0) / 2.0
    confidence = float(getattr(context, "confidence", 0.0)) if context else 0.0
    risk_penalty = 0.12 if context and getattr(context, "risk_flag", False) else 0.0
    score = tech * 0.68 + ctx * 0.22 + confidence * 0.10 - risk_penalty
    return round(max(0.0, min(1.0, score)), 4)


def position_group_counts(pos_state) -> Counter:
    counts = Counter()
    for symbol in getattr(pos_state, "positions", {}).keys():
        counts[symbol_group(symbol)] += 1
    return counts


def avg_correlation_to_positions(symbol: str, open_symbols: list[str], bars_hourly: dict) -> float:
    if not open_symbols or symbol not in bars_hourly:
        return 0.0

    candidate = bars_hourly[symbol]["close"].tail(CORRELATION_LOOKBACK_BARS).pct_change()
    cors = []
    for open_symbol in open_symbols:
        if open_symbol not in bars_hourly or open_symbol == symbol:
            continue
        other = bars_hourly[open_symbol]["close"].tail(CORRELATION_LOOKBACK_BARS).pct_change()
        joined = pd.concat([candidate, other], axis=1).dropna()
        if len(joined) < 20:
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if pd.notna(corr):
            cors.append(float(corr))
    return round(sum(cors) / len(cors), 4) if cors else 0.0


def portfolio_block_reason(symbol: str, pos_state, bars_hourly: dict) -> tuple[bool, str, float]:
    group = symbol_group(symbol)
    group_counts = position_group_counts(pos_state)
    if group_counts[group] >= MAX_POSITIONS_PER_GROUP:
        return True, f"group_exposure:{group}", 0.0

    open_symbols = list(getattr(pos_state, "positions", {}).keys())
    corr = avg_correlation_to_positions(symbol, open_symbols, bars_hourly)
    if corr >= MAX_AVG_CORRELATION_WITH_POSITIONS:
        return True, f"correlation_trop_haute:{corr:.2f}", corr
    return False, "ok", corr


def rank_setups(candidates: list[dict]) -> list[dict]:
    return sorted(
        candidates,
        key=lambda item: (
            item.get("tradable", False),
            item.get("combined_score", 0.0),
            item.get("technical_score", 0.0),
        ),
        reverse=True,
    )


def log_setup_ranking(rows: list[dict], path: str = SETUP_RANKING_LOG_FILE) -> None:
    _ensure_csv(path, RANKING_FIELDS)
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RANKING_FIELDS)
        for i, row in enumerate(rows, start=1):
            writer.writerow({
                "date_utc": date_utc,
                "rank": i,
                "symbol": row["symbol"],
                "group": row["group"],
                "price": round(row["price"], 4),
                "signal": row["signal"],
                "tech_score": round(row["technical_score"], 4),
                "context_score": round(row["context_score"], 4),
                "combined_score": round(row["combined_score"], 4),
                "decision": row.get("decision", ""),
                "reason": row.get("block_reason", ""),
                "corr_to_positions": row.get("corr_to_positions", ""),
                "analysis_reason": row.get("analysis_reason", ""),
                "context_reason": row.get("context_reason", ""),
            })


def log_watchlist(rows: list[dict], path: str = WATCHLIST_LOG_FILE, top_n: int = WATCHLIST_TOP_N) -> None:
    _ensure_csv(path, WATCHLIST_FIELDS)
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WATCHLIST_FIELDS)
        for i, row in enumerate(rows[:top_n], start=1):
            writer.writerow({
                "date_utc": date_utc,
                "rank": i,
                "symbol": row["symbol"],
                "group": row["group"],
                "price": round(row["price"], 4),
                "tech_score": round(row["technical_score"], 4),
                "context_score": round(row["context_score"], 4),
                "combined_score": round(row["combined_score"], 4),
                "note": row.get("analysis_reason", ""),
            })


def log_trade_review(
    symbol: str,
    event: str,
    qty: float,
    price: float,
    entry_price: float = 0.0,
    pnl_usd: float = 0.0,
    pnl_pct: float = 0.0,
    setup: dict | None = None,
    exit_reason: str = "",
    path: str = TRADE_REVIEW_LOG_FILE,
) -> None:
    _ensure_csv(path, TRADE_REVIEW_FIELDS)
    setup = setup or {}
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRADE_REVIEW_FIELDS).writerow({
            "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "event": event,
            "qty": round(float(qty), 4),
            "price": round(float(price), 4),
            "entry_price": round(float(entry_price), 4),
            "pnl_usd": round(float(pnl_usd), 4),
            "pnl_pct": round(float(pnl_pct), 4),
            "technical_score": setup.get("technical_score", ""),
            "context_score": setup.get("context_score", ""),
            "combined_score": setup.get("combined_score", ""),
            "exit_reason": exit_reason,
            "setup_reason": setup.get("analysis_reason", ""),
        })


def update_adaptive_threshold(path: str = ADAPTIVE_STATE_FILE) -> dict:
    state = {"offset": 0.0, "reason": "disabled"}
    if not ADAPTIVE_THRESHOLD_ENABLED:
        return state
    if not os.path.exists(TRADE_REVIEW_LOG_FILE):
        state["reason"] = "no_trade_review"
        return state

    rows = []
    with open(TRADE_REVIEW_LOG_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("event") == "close":
                rows.append(row)

    recent = rows[-20:]
    offset = 0.0
    reason = "neutral"
    if len(recent) < 5:
        offset = -ADAPTIVE_THRESHOLD_STEP
        reason = "peu_de_trades"
    else:
        wins = [r for r in recent if float(r.get("pnl_usd") or 0) > 0]
        win_rate = len(wins) / len(recent)
        avg_pnl = sum(float(r.get("pnl_pct") or 0) for r in recent) / len(recent)
        if win_rate < 0.42 or avg_pnl < -0.35:
            offset = ADAPTIVE_THRESHOLD_STEP
            reason = f"defensif winrate={win_rate:.0%} avg={avg_pnl:+.2f}%"
        elif win_rate > 0.58 and avg_pnl > 0.25:
            offset = -ADAPTIVE_THRESHOLD_STEP
            reason = f"opportuniste winrate={win_rate:.0%} avg={avg_pnl:+.2f}%"

    offset = round(max(ADAPTIVE_THRESHOLD_MIN, min(ADAPTIVE_THRESHOLD_MAX, offset)), 4)
    state = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "offset": offset,
        "reason": reason,
        "closed_trades_sample": len(recent),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state
