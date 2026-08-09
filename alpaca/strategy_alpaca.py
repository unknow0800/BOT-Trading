"""
strategy_alpaca.py — Stratégie pour actions US (GAFA)

Spécificités actions vs crypto :
  - Marché fermé le weekend et la nuit → filtre horaire strict
  - Earnings reports = risque majeur → on évite 2 jours avant/après
  - Tendance sur bougie Daily, signal sur bougie Hourly
  - RSI plus doux (35/65 au lieu de 30/70) car moins volatile
"""

import pandas as pd
import ta
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from indicators import add_all_indicators
from config_alpaca import (RSI_OVERSOLD, RSI_OVERBOUGHT, AVOID_HOURS_ET,
                       SMA_FAST, SMA_SLOW, MIN_VOLUME, USE_EXTENDED_HOURS,
                       MIN_TECHNICAL_SCORE_TO_BUY, MIN_TECHNICAL_SCORE_STRONG)
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def is_market_hours() -> bool:
    """Vérifie qu'on est pendant les heures de trading US (9h-20h ET)."""
    now_et = datetime.now(timezone.utc)
    # Approximation UTC-4 (EDT) / UTC-5 (EST)
    hour_et = (now_et.hour - 4) % 24
    return hour_et not in AVOID_HOURS_ET


def get_market_status(now=None) -> tuple[bool, str]:
    """Retourne (ouvert, raison) pour le marche actions US."""
    tz_et = ZoneInfo("America/New_York")
    now_et = (now or datetime.now(timezone.utc)).astimezone(tz_et)

    if now_et.weekday() >= 5:
        days_until_monday = 7 - now_et.weekday()
        next_open = (now_et + timedelta(days=days_until_monday)).date()
        return False, f"weekend US - prochaine ouverture {next_open} 09:30 ET"

    open_time = time(4, 0) if USE_EXTENDED_HOURS else time(9, 30)
    close_time = time(20, 0) if USE_EXTENDED_HOURS else time(16, 0)
    current_time = now_et.time()

    if current_time < open_time:
        return False, f"avant ouverture US ({now_et:%H:%M} ET, ouverture {open_time:%H:%M} ET)"
    if current_time >= close_time:
        next_day = now_et + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return False, f"apres fermeture US ({now_et:%H:%M} ET, prochaine ouverture {next_day.date()} {open_time:%H:%M} ET)"

    return True, f"marche US ouvert ({now_et:%H:%M} ET)"


def is_market_hours() -> bool:
    """Verifie qu'on est pendant les heures de trading US."""
    is_open, _ = get_market_status()
    return is_open


def get_trend_bias_daily(df_daily: pd.DataFrame) -> str:
    """Analyse la tendance macro sur le Daily."""
    if len(df_daily) < 5:
        return 'neutral'
    df = add_all_indicators(df_daily.copy())
    df["sma_fast"] = df["close"].rolling(min(SMA_FAST, len(df))).mean()
    df["sma_slow"] = df["close"].rolling(min(SMA_SLOW, len(df))).mean()
    df = df.dropna(subset=["sma_fast", "sma_slow", "rsi", "obv_signal"])
    if df.empty:
        return 'neutral'
    last = df.iloc[-1]
    close    = last['close']
    sma_fast = last['sma_fast']
    sma_slow = last['sma_slow']
    obv      = last['obv']
    obv_sig  = last['obv_signal']
    rsi      = last['rsi']

    if (close > sma_fast > sma_slow) and (obv > obv_sig) and (rsi > 45):
        return 'bull'
    elif (close < sma_fast < sma_slow) and (obv < obv_sig) and (rsi < 55):
        return 'bear'
    return 'neutral'


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _score_above(value: float, reference: float, scale: float = 0.03) -> float:
    if reference == 0:
        return 0.5
    return _clamp(0.5 + ((value / reference) - 1.0) / scale * 0.5)


def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute un set d'indicateurs plus riche pour le scoring Alpaca."""
    df = add_all_indicators(df.copy())
    df["sma_fast"] = df["close"].rolling(min(SMA_FAST, len(df))).mean()
    df["sma_slow"] = df["close"].rolling(min(SMA_SLOW, len(df))).mean()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high, low, close, window=min(14, len(df) - 1))
    df["adx"] = adx.adx()
    df["di_plus"] = adx.adx_pos()
    df["di_minus"] = adx.adx_neg()

    boll = ta.volatility.BollingerBands(close, window=min(20, len(df)))
    df["bb_high"] = boll.bollinger_hband()
    df["bb_low"] = boll.bollinger_lband()
    df["bb_mid"] = boll.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=min(14, len(df) - 1))
    df["stoch"] = stoch.stoch()
    df["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=min(14, len(df) - 1)).money_flow_index()
    df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=min(20, len(df))).chaikin_money_flow()
    df["roc"] = ta.momentum.ROCIndicator(close, window=min(10, len(df) - 1)).roc()
    df["volume_ratio"] = volume / volume.rolling(min(20, len(df))).mean()
    df["atr_pct"] = df["atr"] / close
    required = [
        "rsi", "atr", "vwap", "obv", "obv_signal", "sma_fast", "sma_slow",
        "macd", "macd_signal", "adx", "di_plus", "di_minus", "bb_mid",
        "stoch", "mfi", "cmf", "roc", "volume_ratio", "atr_pct",
    ]
    return df.dropna(subset=required)


def analyze_hourly_signal(df_hourly: pd.DataFrame,
                          trend: str,
                          in_position: bool,
                          symbol: str = '',
                          min_score_to_buy: float | None = None,
                          require_market_open: bool = True) -> dict:
    """
    Produit un signal et un diagnostic complet.

    Le score long combine:
    - regime de tendance daily
    - tendance intraday SMA/VWAP
    - momentum RSI/MACD/ROC/Stoch
    - flux OBV/CMF/MFI
    - volume et volatilite
    """
    base = {
        "signal": "hold",
        "score": 0.0,
        "grade": "blocked",
        "reason": "",
        "checks": {},
    }

    threshold = MIN_TECHNICAL_SCORE_TO_BUY if min_score_to_buy is None else min_score_to_buy
    if require_market_open:
        market_open, market_reason = get_market_status()
        if not market_open:
            base["reason"] = market_reason
            return base
    if len(df_hourly) < 30:
        base["reason"] = f"historique_insuffisant:{len(df_hourly)}"
        return base

    df = add_advanced_indicators(df_hourly)
    if len(df) < 3:
        base["reason"] = "indicateurs_insuffisants"
        return base

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["close"])
    rsi = float(last["rsi"])
    rsi_prev = float(prev["rsi"])
    volume = float(last["volume"])
    volume_ratio = float(last["volume_ratio"])
    atr_pct = float(last["atr_pct"])

    checks = {
        "trend": trend,
        "trend_score": {"bull": 1.0, "neutral": 0.45, "bear": 0.0}.get(trend, 0.0),
        "above_vwap": close > float(last["vwap"]),
        "above_sma_fast": close > float(last["sma_fast"]),
        "above_sma_slow": close > float(last["sma_slow"]),
        "obv_bullish": float(last["obv"]) > float(last["obv_signal"]),
        "macd_bullish": float(last["macd"]) > float(last["macd_signal"]),
        "di_bullish": float(last["di_plus"]) > float(last["di_minus"]),
        "cmf_positive": float(last["cmf"]) > 0,
        "volume_ok": volume > MIN_VOLUME,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "rsi_rebound": rsi_prev < RSI_OVERSOLD and rsi > RSI_OVERSOLD,
        "atr_pct": atr_pct,
    }

    if in_position:
        rsi_high = rsi > RSI_OVERBOUGHT
        below_sma = close < float(last["sma_fast"])
        obv_div = close > float(prev["close"]) and float(last["obv"]) < float(prev["obv"])
        if rsi_high or below_sma or obv_div:
            base.update({"signal": "exit_long", "score": 1.0, "grade": "exit", "reason": "exit_score"})
        else:
            base.update({"reason": "position_hold"})
        base["checks"] = checks
        return base

    if trend == "bear":
        base.update({"reason": "daily_bear", "checks": checks})
        return base
    if not checks["volume_ok"]:
        base.update({"reason": "volume_insuffisant", "checks": checks})
        return base

    trend_score = checks["trend_score"]
    location_score = (
        (1.0 if checks["above_vwap"] else 0.0) * 0.35
        + (1.0 if checks["above_sma_fast"] else 0.0) * 0.35
        + (1.0 if checks["above_sma_slow"] else 0.0) * 0.30
    )
    momentum_score = (
        _clamp((rsi - 35) / 30) * 0.25
        + (1.0 if checks["rsi_rebound"] else 0.45 if 45 <= rsi <= 68 else 0.15) * 0.20
        + (1.0 if checks["macd_bullish"] else 0.0) * 0.25
        + _clamp((float(last["roc"]) + 5) / 10) * 0.15
        + _clamp((float(last["stoch"]) - 20) / 60) * 0.15
    )
    flow_score = (
        (1.0 if checks["obv_bullish"] else 0.0) * 0.35
        + (1.0 if checks["di_bullish"] else 0.0) * 0.25
        + _clamp((float(last["cmf"]) + 0.2) / 0.4) * 0.20
        + _clamp((float(last["mfi"]) - 30) / 50) * 0.20
    )
    volume_score = _clamp((volume_ratio - 0.75) / 1.25)
    volatility_score = 1.0 if 0.003 <= atr_pct <= 0.08 else 0.35

    score = (
        trend_score * 0.22
        + location_score * 0.20
        + momentum_score * 0.24
        + flow_score * 0.20
        + volume_score * 0.09
        + volatility_score * 0.05
    )
    score = round(_clamp(score), 4)

    grade = "strong" if score >= MIN_TECHNICAL_SCORE_STRONG else "valid" if score >= threshold else "watch"
    signal = "enter_long" if score >= threshold else "hold"
    reason = (
        f"score={score:.2f}/{threshold:.2f} "
        f"trend={trend_score:.2f} loc={location_score:.2f} "
        f"mom={momentum_score:.2f} flow={flow_score:.2f} "
        f"vol={volume_score:.2f}"
    )

    return {
        "signal": signal,
        "score": score,
        "grade": grade,
        "reason": reason,
        "checks": checks,
    }


def get_signal_hourly(df_hourly: pd.DataFrame,
                      trend: str,
                      in_position: bool,
                      symbol: str = '') -> str:
    """
    Signal d'entrée/sortie sur le hourly.

    Entrée long :
      - Tendance Daily haussière
      - RSI sort de la survente (croise > 35)
      - Prix au-dessus du VWAP
      - Volume suffisant
      - Heures de marché valides

    Sortie long :
      - RSI > 65 (suracheté)
      - Prix sous SMA rapide
      - OBV diverge
    """
    return analyze_hourly_signal(df_hourly, trend, in_position, symbol)["signal"]


def calc_position_size(equity: float, price: float, atr: float,
                       risk_pct: float = 0.01, atr_mult: float = 2.0,
                       position_cap: float = 0.20) -> float:
    """
    Calcule la quantité d'actions à acheter.
    Limite à position_cap% du capital total.
    """
    risk_amount   = equity * risk_pct
    per_unit_risk = atr * atr_mult
    if per_unit_risk <= 0 or price <= 0:
        return 0.0

    qty = risk_amount / per_unit_risk
    max_qty = (equity * position_cap) / price
    qty = min(qty, max_qty)
    return max(0.0, round(qty, 4))
