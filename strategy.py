import pandas as pd
from datetime import datetime, timezone
from indicators import add_all_indicators
from config import (RSI_OVERSOLD, RSI_OVERBOUGHT, AVOID_HOURS_UTC)


# ==========================
# FILTRE HORAIRE
# ==========================

def is_trading_hours() -> bool:
    """
    Retourne False si on est dans une plage horaire à éviter.
    Évite le bruit nocturne (22h-06h UTC).
    """
    now_utc = datetime.now(timezone.utc)
    return now_utc.hour not in AVOID_HOURS_UTC


# ==========================
# ANALYSE MACRO (4H)
# ==========================

def get_trend_bias(df_trend: pd.DataFrame) -> str:
    """
    Analyse la tendance macro sur le 4H.
    Retourne : 'bull', 'bear', 'neutral'
    """
    df = add_all_indicators(df_trend.copy())
    last = df.iloc[-1]

    close     = last['close']
    sma_fast  = last['sma_fast']
    sma_slow  = last['sma_slow']
    rsi       = last['rsi']
    obv       = last['obv']
    obv_sig   = last['obv_signal']

    # Uptrend : prix au-dessus des deux SMA + SMA rapide > SMA lente + OBV haussier
    if (close > sma_fast > sma_slow) and (obv > obv_sig) and (rsi > 45):
        return 'bull'
    # Downtrend
    elif (close < sma_fast < sma_slow) and (obv < obv_sig) and (rsi < 55):
        return 'bear'
    else:
        return 'neutral'


# ==========================
# SIGNAL D'ENTRÉE (30M)
# ==========================

def get_entry_signal(df_30m: pd.DataFrame, trend: str, in_position: bool) -> str:
    """
    Signal d'entrée sur le 30M avec confirmation volume.

    Conditions d'achat (enter_long) :
      1. Tendance 4H haussière
      2. RSI sort de la survente (croise au-dessus de 30)
      3. Prix au-dessus du VWAP
      4. OBV > OBV_signal (pression acheteuse)
      5. Heure de trading valide

    Conditions de sortie (exit_long) :
      - RSI survente > 70
      - Prix repasse sous VWAP
      - OBV diverge (prix monte mais OBV baisse)
    """
    if not is_trading_hours():
        return 'hold'

    df = add_all_indicators(df_30m.copy())
    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi       = last['rsi']
    rsi_prev  = prev['rsi']
    close     = last['close']
    vwap      = last['vwap']
    obv       = last['obv']
    obv_sig   = last['obv_signal']
    sma_fast  = last['sma_fast']
    vol_spike = last['volume_spike']

    if not in_position:
        if trend != 'bull':
            return 'hold'

        # Entrée : RSI sort de survente + VWAP + OBV bullish + volume spike confirmé
        rsi_cross_up = (rsi_prev < RSI_OVERSOLD) and (rsi > RSI_OVERSOLD)
        above_vwap   = close > vwap
        obv_bullish  = obv > obv_sig

        if rsi_cross_up and above_vwap and obv_bullish:
            if vol_spike:
                return 'enter_long_strong'   # Signal fort (volume confirmé)
            else:
                return 'enter_long'          # Signal normal

        return 'hold'

    else:
        # Sortie stratégique (en plus du SL/TP géré par risk.py)
        rsi_overbought = rsi > RSI_OVERBOUGHT
        below_vwap     = close < vwap
        below_sma_fast = close < sma_fast

        # Divergence baissière OBV : prix haussier mais OBV baisse
        obv_divergence = (close > prev['close']) and (obv < prev['obv'])

        if rsi_overbought or below_sma_fast or (below_vwap and obv_divergence):
            return 'exit_long'

        return 'hold'


# ==========================
# RÉSUMÉ COMPLET DU CONTEXTE
# ==========================

def get_market_context(df_trend: pd.DataFrame, df_30m: pd.DataFrame) -> dict:
    """
    Retourne un dict complet du contexte de marché pour le logging.
    """
    df4 = add_all_indicators(df_trend.copy())
    df30 = add_all_indicators(df_30m.copy())

    l4 = df4.iloc[-1]
    l30 = df30.iloc[-1]

    return {
        "trend_4h"     : get_trend_bias(df_trend),
        "close_4h"     : round(l4['close'], 2),
        "rsi_4h"       : round(l4['rsi'], 2),
        "sma_fast_4h"  : round(l4['sma_fast'], 2),
        "sma_slow_4h"  : round(l4['sma_slow'], 2),
        "close_30m"    : round(l30['close'], 2),
        "rsi_30m"      : round(l30['rsi'], 2),
        "vwap_30m"     : round(l30['vwap'], 2),
        "obv_bullish"  : bool(l30['obv'] > l30['obv_signal']),
        "vol_spike"    : bool(l30['volume_spike']),
        "atr_30m"      : round(l30['atr'], 2),
        "trading_hours": is_trading_hours(),
    }