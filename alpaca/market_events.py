"""
market_events.py - Sources externes optionnelles: earnings, analyst ratings, macro.

Les fonctions retournent toujours un score entre -1 et +1 et une raison lisible.
Sans cle API ou en cas d'erreur reseau, elles restent neutres pour ne pas casser
le bot de trading.
"""

from datetime import datetime, timedelta, timezone
import requests

from config_alpaca import (
    ENABLE_ANALYST_RATINGS,
    ENABLE_EARNINGS_CALENDAR,
    ENABLE_MACRO_EVENTS,
    FINNHUB_API_KEY,
    FMP_API_KEY,
)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_earnings_risk(symbol: str, days_window: int = 5) -> tuple[float, str, bool]:
    if not ENABLE_EARNINGS_CALENDAR:
        return 0.0, "earnings=off", False
    if not FINNHUB_API_KEY:
        return 0.0, "earnings=pas_de_cle", False

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_window)
    end = today + timedelta(days=days_window)
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={
                "from": start.isoformat(),
                "to": end.isoformat(),
                "symbol": symbol,
                "token": FINNHUB_API_KEY,
            },
            timeout=8,
        )
        r.raise_for_status()
        events = r.json().get("earningsCalendar", [])
    except Exception as exc:
        return 0.0, f"earnings=erreur:{exc}", False

    if not events:
        return 0.0, "earnings=aucun", False

    nearest = min(
        events,
        key=lambda e: abs((datetime.fromisoformat(e["date"]).date() - today).days),
    )
    days = (datetime.fromisoformat(nearest["date"]).date() - today).days
    risk = abs(days) <= 2
    score = -0.45 if risk else -0.15
    return score, f"earnings={nearest['date']} j{days:+d}", risk


def score_analyst_rating(symbol: str) -> tuple[float, str]:
    if not ENABLE_ANALYST_RATINGS:
        return 0.0, "analyst=off"
    if not FINNHUB_API_KEY:
        return 0.0, "analyst=pas_de_cle"

    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/recommendation",
            params={"symbol": symbol, "token": FINNHUB_API_KEY},
            timeout=8,
        )
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:
        return 0.0, f"analyst=erreur:{exc}"

    if not rows:
        return 0.0, "analyst=aucun"

    latest = rows[0]
    positive = latest.get("strongBuy", 0) * 1.0 + latest.get("buy", 0) * 0.5
    negative = latest.get("sell", 0) * 0.5 + latest.get("strongSell", 0) * 1.0
    hold = latest.get("hold", 0)
    total = positive + negative + hold
    if total <= 0:
        return 0.0, "analyst=neutre"
    score = _clamp((positive - negative) / total)
    return score, f"analyst={score:+.2f}"


def score_macro_events() -> tuple[float, str, bool]:
    if not ENABLE_MACRO_EVENTS:
        return 0.0, "macro=off", False
    if not FMP_API_KEY:
        return 0.0, "macro=pas_de_cle", False

    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=3)
    try:
        r = requests.get(
            "https://financialmodelingprep.com/api/v3/economic_calendar",
            params={
                "from": today.isoformat(),
                "to": end.isoformat(),
                "apikey": FMP_API_KEY,
            },
            timeout=8,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as exc:
        return 0.0, f"macro=erreur:{exc}", False

    high_impact = [
        e for e in events
        if str(e.get("impact", "")).lower() == "high"
        or any(word in str(e.get("event", "")).lower() for word in ["cpi", "fomc", "payroll", "interest rate"])
    ]
    if not high_impact:
        return 0.0, "macro=calme", False

    return -0.25, f"macro=risque({len(high_impact)})", True
