import pandas as pd
import ta
from config import (RSI_PERIOD, SMA_FAST, SMA_SLOW, ATR_PERIOD, OBV_SMA_PERIOD)

MIN_CANDLES_REQUIRED = 5   # testnet 4H a peu d'historique


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    window = min(RSI_PERIOD, len(df) - 1)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
    return df


def add_sma(df: pd.DataFrame) -> pd.DataFrame:
    fast = min(SMA_FAST, len(df))
    slow = min(SMA_SLOW, len(df))
    df['sma_fast'] = df['close'].rolling(fast).mean()
    df['sma_slow'] = df['close'].rolling(slow).mean()
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    window = min(ATR_PERIOD, len(df) - 1)
    if window < 2:
        df['atr'] = df['close'] * 0.01
        return df
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=window
    ).average_true_range()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        df['close'], df['volume']
    ).on_balance_volume()
    window = min(OBV_SMA_PERIOD, len(df))
    df['obv_signal'] = df['obv'].rolling(window).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']
    df['date'] = df.index.date
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol']    = df.groupby('date')['volume'].cumsum()
    df['vwap']       = df['cum_tp_vol'] / df['cum_vol']
    df.drop(columns=['date', 'cum_tp_vol', 'cum_vol', 'tp_vol', 'typical_price'],
            inplace=True)
    return df


def add_volume_spike(df: pd.DataFrame, window: int = 20, threshold: float = 1.5) -> pd.DataFrame:
    w = min(window, len(df))
    df['volume_ma'] = df['volume'].rolling(w).mean()
    df['volume_spike'] = df['volume'] > (df['volume_ma'] * threshold)
    return df


def add_swing_levels(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    w = min(window, max(2, len(df) // 2))
    df['swing_high'] = df['high'][(
        df['high'] == df['high'].rolling(w, center=True).max()
    )]
    df['swing_low'] = df['low'][(
        df['low'] == df['low'].rolling(w, center=True).min()
    )]
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < MIN_CANDLES_REQUIRED:
        raise ValueError(f"Pas assez de bougies : {len(df)} < {MIN_CANDLES_REQUIRED}")
    df = add_rsi(df)
    df = add_sma(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_volume_spike(df)
    df = add_swing_levels(df)
    return df.dropna(subset=['rsi', 'atr'])