"""
ml/features.py — Preparation des features pour LSTM + Transformer

Pipeline :
  1. Calcul de tous les indicateurs techniques
  2. Creation des labels buy/sell/hold (forward-looking)
  3. Normalisation MinMaxScaler par feature
  4. Construction des sequences temporelles (window=60 bougies)
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import add_all_indicators

WINDOW      = 30     # nombre de bougies par sequence
HORIZON     = 4      # on regarde 3 bougies dans le futur pour labelliser
BUY_THRESH  = 0.004  # +0.5% → BUY
SELL_THRESH = 0.004  # -0.5% → SELL
# Classes : 0=HOLD, 1=BUY, 2=SELL

FEATURE_COLS = [
    'close', 'open', 'high', 'low', 'volume',
    'rsi', 'sma_fast', 'sma_slow', 'atr',
    'obv', 'obv_signal', 'vwap',
    'volume_ma',
    # Features derivees ajoutees dans add_derived()
    'close_pct',       # rendement bougie
    'high_low_range',  # amplitude bougie
    'close_vs_vwap',   # position vs VWAP
    'rsi_change',      # momentum RSI
    'volume_ratio',    # volume relatif
]

SCALER_PATH = "ml/scaler.pkl"
MODEL_PATH  = "ml/model.pt"


# ==========================
# FEATURES DERIVEES
# ==========================

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features derivees utiles pour le DL."""
    df['close_pct']     = df['close'].pct_change()
    df['high_low_range']= (df['high'] - df['low']) / df['close']
    df['close_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    df['rsi_change']    = df['rsi'].diff()
    df['volume_ratio']  = df['volume'] / df['volume_ma'].replace(0, 1)
    return df


# ==========================
# LABELLISATION
# ==========================

def make_labels(df: pd.DataFrame) -> pd.Series:
    """
    Label forward-looking :
      - BUY  (1) si le close monte de >0.5% dans les 3 prochaines bougies
      - SELL (2) si le close baisse de >0.5% dans les 3 prochaines bougies
      - HOLD (0) sinon
    """
    future_ret = df['close'].shift(-HORIZON) / df['close'] - 1
    labels = pd.Series(0, index=df.index)  # HOLD par defaut
    labels[future_ret >  BUY_THRESH]  = 1  # BUY
    labels[future_ret < -SELL_THRESH] = 2  # SELL
    return labels


# ==========================
# NORMALISATION
# ==========================

class FeatureScaler:
    """MinMax par colonne, sauvegardable en pickle."""

    def __init__(self):
        self.mins = {}
        self.maxs = {}

    def fit(self, df: pd.DataFrame, cols: list):
        for c in cols:
            self.mins[c] = df[c].min()
            self.maxs[c] = df[c].max()

    def transform(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            rng = self.maxs[c] - self.mins[c]
            if rng == 0:
                out[c] = 0.0
            else:
                out[c] = (df[c] - self.mins[c]) / rng
        return out

    def fit_transform(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        self.fit(df, cols)
        return self.transform(df, cols)

    def save(self, path: str = SCALER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = SCALER_PATH):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ==========================
# CONSTRUCTION DES SEQUENCES
# ==========================

def build_sequences(df_norm: pd.DataFrame,
                    labels: pd.Series,
                    cols: list,
                    window: int = WINDOW):
    """
    Retourne X (N, window, features) et y (N,) en numpy.
    """
    X, y = [], []
    arr = df_norm[cols].values
    lbl = labels.values

    for i in range(window, len(arr) - HORIZON):
        X.append(arr[i - window : i])
        y.append(lbl[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ==========================
# PIPELINE COMPLET
# ==========================

def prepare_data(df_raw: pd.DataFrame, fit_scaler: bool = True):
    """
    Prend un DataFrame OHLCV brut, retourne (X, y, scaler, feature_cols).

    fit_scaler=True  : entraine le scaler (mode training)
    fit_scaler=False : charge le scaler existant (mode inference)
    """
    # 1. Indicateurs
    df = add_all_indicators(df_raw.copy())

    # 2. Features derivees
    df = add_derived(df)

    # 3. Supprimer les NaN
    df = df.dropna(subset=FEATURE_COLS)

    # 4. Labels
    labels = make_labels(df)

    # 5. Scaler
    if fit_scaler:
        scaler = FeatureScaler()
        df_norm = scaler.fit_transform(df, FEATURE_COLS)
        scaler.save()
    else:
        scaler = FeatureScaler.load()
        df_norm = scaler.transform(df, FEATURE_COLS)

    # 6. Sequences
    X, y = build_sequences(df_norm, labels, FEATURE_COLS)

    print(f"[features] X: {X.shape} | y: {y.shape}")
    print(f"[features] BUY: {(y==1).sum()} | HOLD: {(y==0).sum()} | SELL: {(y==2).sum()}")

    return X, y, scaler, FEATURE_COLS


def prepare_last_sequence(df_raw: pd.DataFrame):
    """
    Prepare la derniere sequence pour l'inference en temps reel.
    Retourne un tensor (1, window, features).
    """
    import torch
    df = add_all_indicators(df_raw.copy())
    df = add_derived(df)
    df = df.dropna(subset=FEATURE_COLS)

    if len(df) < WINDOW:
        raise ValueError(f"Pas assez de bougies pour l'inference : {len(df)} < {WINDOW}")

    scaler = FeatureScaler.load()
    df_norm = scaler.transform(df, FEATURE_COLS)

    seq = df_norm[FEATURE_COLS].values[-WINDOW:]
    return torch.tensor(seq[np.newaxis], dtype=torch.float32)