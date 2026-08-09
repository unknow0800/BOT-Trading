"""
market_intelligence.py - Contexte externe pour les decisions Alpaca.

Cette couche ne predit pas le marche. Elle sert de filtre de qualite:
news recentes, sentiment simple, momentum/volume et signal TradingView optionnel.
"""

from dataclasses import dataclass
import math
import re

from config_alpaca import (
    ENABLE_ALPACA_NEWS,
    ENABLE_TRADINGVIEW_TA,
    MIN_CONTEXT_CONFIDENCE,
    MIN_CONTEXT_SCORE_TO_BUY,
    NEWS_LOOKBACK_HOURS,
    SCORING_WEIGHTS,
)
from data_alpaca import get_news_articles
from market_events import score_analyst_rating, score_earnings_risk, score_macro_events


POSITIVE_WORDS = {
    "beat", "beats", "upgrade", "upgraded", "bullish", "growth", "profit",
    "profits", "record", "strong", "surge", "surges", "rally", "raises",
    "outperform", "buy", "partnership", "approval", "launch", "expands",
    "positive", "optimistic", "demand", "guidance raised",
}

NEGATIVE_WORDS = {
    "miss", "misses", "downgrade", "downgraded", "bearish", "lawsuit",
    "probe", "investigation", "weak", "falls", "fall", "plunge", "plunges",
    "cuts", "cut", "layoffs", "risk", "warning", "recall", "ban", "delay",
    "negative", "slows", "slowing", "guidance cut", "fraud", "antitrust",
}

HIGH_RISK_WORDS = {
    "sec", "doj", "fda", "antitrust", "lawsuit", "fraud", "bankruptcy",
    "probe", "investigation", "recall", "guidance cut", "misses",
}


@dataclass
class MarketContext:
    symbol: str
    score: float
    confidence: float
    news_score: float
    technical_context_score: float
    tradingview_score: float
    earnings_score: float
    analyst_score: float
    macro_score: float
    article_count: int
    risk_flag: bool
    reason: str

    @property
    def allow_long(self) -> bool:
        if self.confidence < MIN_CONTEXT_CONFIDENCE:
            return True
        if self.risk_flag and self.score < 0:
            return False
        return self.score >= MIN_CONTEXT_SCORE_TO_BUY


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def _keyword_score(text: str) -> tuple[float, bool]:
    positive = sum(1 for word in POSITIVE_WORDS if word in text)
    negative = sum(1 for word in NEGATIVE_WORDS if word in text)
    risk_flag = any(word in text for word in HIGH_RISK_WORDS)
    total = positive + negative
    if total == 0:
        return 0.0, risk_flag
    return _clamp((positive - negative) / total), risk_flag


def score_news(articles: list) -> tuple[float, float, bool, str]:
    if not articles:
        return 0.0, 0.0, False, "news=aucune"

    weighted_scores = []
    risk_flag = False
    for i, article in enumerate(articles[:10]):
        headline = getattr(article, "headline", "")
        summary = getattr(article, "summary", "")
        text = _clean_text(f"{headline} {summary}")
        score, article_risk = _keyword_score(text)
        risk_flag = risk_flag or article_risk
        weight = 1.0 / math.sqrt(i + 1)
        weighted_scores.append(score * weight)

    denom = sum(1.0 / math.sqrt(i + 1) for i in range(len(weighted_scores)))
    score = sum(weighted_scores) / denom if denom else 0.0
    confidence = min(1.0, len(articles) / 6)
    reason = f"news={len(articles)} score={score:+.2f}"
    return _clamp(score), confidence, risk_flag, reason


def score_technical_context(df_daily, df_hourly) -> tuple[float, str]:
    if df_daily is None or df_hourly is None or len(df_daily) < 20 or len(df_hourly) < 10:
        return 0.0, "ctx_tech=insuffisant"

    daily_return_5 = (df_daily["close"].iloc[-1] / df_daily["close"].iloc[-6]) - 1
    hourly_return_6 = (df_hourly["close"].iloc[-1] / df_hourly["close"].iloc[-7]) - 1
    avg_volume = df_hourly["volume"].tail(20).mean()
    volume_ratio = df_hourly["volume"].iloc[-1] / avg_volume if avg_volume else 1.0

    score = 0.0
    score += _clamp(daily_return_5 / 0.05) * 0.45
    score += _clamp(hourly_return_6 / 0.03) * 0.35
    score += _clamp((volume_ratio - 1.0) / 1.5) * 0.20
    score = _clamp(score)

    reason = (
        f"ctx_tech={score:+.2f} "
        f"ret5d={daily_return_5:+.1%} ret6h={hourly_return_6:+.1%} volx={volume_ratio:.1f}"
    )
    return score, reason


def score_tradingview(symbol: str) -> tuple[float, str]:
    if not ENABLE_TRADINGVIEW_TA:
        return 0.0, "tv=off"

    try:
        from tradingview_ta import TA_Handler, Interval
    except Exception:
        return 0.0, "tv=lib_absente"

    try:
        handler = TA_Handler(
            symbol=symbol,
            screener="america",
            exchange="NASDAQ",
            interval=Interval.INTERVAL_1_HOUR,
        )
        summary = handler.get_analysis().summary
        recommendation = summary.get("RECOMMENDATION", "NEUTRAL")
    except Exception as exc:
        return 0.0, f"tv=erreur:{exc}"

    mapping = {
        "STRONG_BUY": 0.75,
        "BUY": 0.35,
        "NEUTRAL": 0.0,
        "SELL": -0.35,
        "STRONG_SELL": -0.75,
    }
    return mapping.get(recommendation, 0.0), f"tv={recommendation}"


def combine_weighted_scores(scores: dict, weights: dict | None = None) -> float:
    weights = weights or SCORING_WEIGHTS
    weighted_sum = 0.0
    active_weight = 0.0
    for key, weight in weights.items():
        if key not in scores:
            continue
        weighted_sum += scores[key] * weight
        active_weight += weight
    if active_weight <= 0:
        return 0.0
    return _clamp(weighted_sum / active_weight)


def build_market_context(
    symbol: str,
    df_daily,
    df_hourly,
    articles: list | None = None,
    include_live_external: bool = True,
) -> MarketContext:
    articles = articles or []
    news_score, news_conf, news_risk, news_reason = score_news(articles)
    tech_score, tech_reason = score_technical_context(df_daily, df_hourly)

    if include_live_external:
        tv_score, tv_reason = score_tradingview(symbol)
        earnings_score, earnings_reason, earnings_risk = score_earnings_risk(symbol)
        analyst_score, analyst_reason = score_analyst_rating(symbol)
        macro_score, macro_reason, macro_risk = score_macro_events()
    else:
        tv_score, tv_reason = 0.0, "tv=backtest_off"
        earnings_score, earnings_reason, earnings_risk = 0.0, "earnings=backtest_off", False
        analyst_score, analyst_reason = 0.0, "analyst=backtest_off"
        macro_score, macro_reason, macro_risk = 0.0, "macro=backtest_off", False

    scores = {
        "news": news_score,
        "technical_context": tech_score,
        "tradingview": tv_score,
        "earnings": earnings_score,
        "analyst": analyst_score,
        "macro": macro_score,
    }
    score = combine_weighted_scores(scores)
    risk_flag = news_risk or earnings_risk or macro_risk
    confidence = _clamp(
        max(
            news_conf,
            0.35 if tech_score else 0.0,
            0.30 if include_live_external and any([tv_score, earnings_score, analyst_score, macro_score]) else 0.0,
        ),
        0.0,
        1.0,
    )
    reason = " | ".join([
        news_reason,
        tech_reason,
        tv_reason,
        earnings_reason,
        analyst_reason,
        macro_reason,
        f"score={score:+.2f}",
    ])

    return MarketContext(
        symbol=symbol,
        score=score,
        confidence=confidence,
        news_score=news_score,
        technical_context_score=tech_score,
        tradingview_score=tv_score,
        earnings_score=earnings_score,
        analyst_score=analyst_score,
        macro_score=macro_score,
        article_count=len(articles),
        risk_flag=risk_flag,
        reason=reason,
    )


def build_market_contexts(news_client, symbols: list, bars_daily: dict, bars_hourly: dict, logger=None) -> dict:
    news_by_symbol = {symbol: [] for symbol in symbols}
    if ENABLE_ALPACA_NEWS and news_client is not None:
        try:
            news_by_symbol = get_news_articles(news_client, symbols, NEWS_LOOKBACK_HOURS)
        except Exception as exc:
            if logger:
                logger.warning(f"[CTX] News indisponibles: {exc}")

    contexts = {}
    for symbol in symbols:
        articles = news_by_symbol.get(symbol, [])
        contexts[symbol] = build_market_context(
            symbol=symbol,
            df_daily=bars_daily.get(symbol),
            df_hourly=bars_hourly.get(symbol),
            articles=articles,
            include_live_external=True,
        )
    return contexts
