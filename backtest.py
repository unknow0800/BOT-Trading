"""
Backtest rapide de la stratégie RSI + SMA + VWAP + OBV.

Usage :
    python backtest.py --symbol BTCUSDT --interval 30m --start 2023-01-01

Nécessite : pip install vectorbt pandas-ta
"""

import argparse
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime
from indicators import add_all_indicators
from config import (API_KEY, API_SECRET, TESTNET_BASE_URL, TESTNET,
                    RSI_OVERSOLD, RSI_OVERBOUGHT,
                    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
                    RISK_PER_TRADE_PCT)

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    print("⚠️  vectorbt non installé. pip install vectorbt")


# ==========================
# TÉLÉCHARGEMENT HISTORIQUE
# ==========================

def download_history(symbol: str, interval: str, start: str) -> pd.DataFrame:
    """Télécharge l'historique complet depuis Binance."""
    if TESTNET:
        client = Client(API_KEY, API_SECRET,
                        testnet=True,
                        tld='vision')
    else:
        client = Client(API_KEY, API_SECRET)

    print(f"📥 Téléchargement {symbol} {interval} depuis {start}...")
    klines = client.get_historical_klines(symbol, interval, start)

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'trades', 'tbbase', 'tbquote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('timestamp', inplace=True)

    print(f"✅ {len(df)} bougies téléchargées.")
    return df


# ==========================
# GÉNÉRATION DES SIGNAUX
# ==========================

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les indicateurs et génère les signaux buy/sell vectorisés.
    """
    df = add_all_indicators(df.copy())

    rsi = df['rsi']
    close = df['close']
    vwap = df['vwap']
    obv = df['obv']
    obv_sig = df['obv_signal']
    sma_fast = df['sma_fast']
    sma_slow = df['sma_slow']

    # Uptrend macro (filtre 4H simulé ici sur le même TF)
    uptrend = (close > sma_fast) & (sma_fast > sma_slow) & (obv > obv_sig)

    # Signal d'entrée : RSI croise au-dessus de 30 + uptrend + prix > VWAP
    rsi_cross_up = (rsi.shift(1) < RSI_OVERSOLD) & (rsi > RSI_OVERSOLD)
    entries = uptrend & rsi_cross_up & (close > vwap)

    # Signal de sortie : RSI > 70 ou prix < SMA rapide
    exits = (rsi > RSI_OVERBOUGHT) | (close < sma_fast)

    df['entry'] = entries
    df['exit']  = exits
    return df


# ==========================
# BACKTEST VECTORBT
# ==========================

def run_backtest(df: pd.DataFrame, initial_cash: float = 10_000.0):
    if not VBT_AVAILABLE:
        print("vectorbt requis pour le backtest. pip install vectorbt")
        return

    df = generate_signals(df)

    # ATR stop loss dynamique
    sl_stop = ATR_SL_MULTIPLIER * df['atr'] / df['close']
    tp_stop = ATR_TP_MULTIPLIER * df['atr'] / df['close']

    portfolio = vbt.Portfolio.from_signals(
        close        = df['close'],
        entries      = df['entry'],
        exits        = df['exit'],
        sl_stop      = sl_stop,
        tp_stop      = tp_stop,
        init_cash    = initial_cash,
        fees         = 0.001,        # 0.1% frais Binance
        slippage     = 0.0005,       # 0.05% slippage estimé
        size         = RISK_PER_TRADE_PCT,
        size_type    = 'valuepercent',
        freq         = '30T',
    )

    # ==========================
    # MÉTRIQUES
    # ==========================
    stats = portfolio.stats()
    print("\n" + "="*50)
    print("📊  RÉSULTATS DU BACKTEST")
    print("="*50)
    print(f"Période              : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Capital initial      : {initial_cash:,.0f} USDT")
    print(f"Capital final        : {portfolio.final_value():.2f} USDT")
    print(f"Return total         : {stats['Total Return [%]']:.1f}%")
    print(f"Sharpe Ratio         : {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown         : {stats['Max Drawdown [%]']:.1f}%")
    print(f"Win Rate             : {stats['Win Rate [%]']:.1f}%")
    print(f"Profit Factor        : {stats.get('Profit Factor', 'N/A')}")
    print(f"Total Trades         : {stats['Total Trades']}")
    print(f"Avg Trade Duration   : {stats.get('Avg Winning Trade Duration', 'N/A')}")
    print("="*50)

    # Sauvegarde CSV des trades
    trades_df = portfolio.trades.records_readable
    trades_df.to_csv("logs/backtest_trades.csv", index=False)
    print("💾 Trades sauvegardés dans logs/backtest_trades.csv")

    # Equity curve
    eq_curve = portfolio.value()
    eq_curve.to_csv("logs/backtest_equity.csv")
    print("💾 Equity curve sauvegardée dans logs/backtest_equity.csv")

    return portfolio, stats


# ==========================
# BACKTEST SIMPLE (sans vectorbt)
# ==========================

def run_simple_backtest(df: pd.DataFrame, initial_cash: float = 10_000.0):
    """
    Backtest simple sans dépendance externe.
    Utile si vectorbt n'est pas installé.
    """
    df = generate_signals(df)
    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    trades = []
    equity = [cash]

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']

        if position == 0 and row['entry']:
            # Entrée
            risk_amount = cash * RISK_PER_TRADE_PCT
            per_unit_risk = ATR_SL_MULTIPLIER * atr
            qty = risk_amount / per_unit_risk if per_unit_risk > 0 else 0
            cost = qty * price * 1.001  # frais 0.1%
            if cost <= cash:
                cash -= cost
                position = qty
                entry_price = price
                sl = price - ATR_SL_MULTIPLIER * atr
                tp = price + ATR_TP_MULTIPLIER * atr

        elif position > 0:
            sl = entry_price - ATR_SL_MULTIPLIER * atr
            tp = entry_price + ATR_TP_MULTIPLIER * atr

            if price <= sl or price >= tp or row['exit']:
                pnl = (price - entry_price) * position
                cash += position * price * 0.999  # frais
                reason = 'SL' if price <= sl else ('TP' if price >= tp else 'signal')
                trades.append({
                    'date': df.index[i],
                    'entry': entry_price,
                    'exit': price,
                    'pnl': round(pnl, 2),
                    'reason': reason
                })
                position = 0.0
                entry_price = 0.0

        equity.append(cash + position * price)

    # Métriques
    wins   = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)

    print("\n" + "="*50)
    print("📊  BACKTEST SIMPLE")
    print("="*50)
    print(f"Capital initial  : {initial_cash:,.0f} USDT")
    print(f"Capital final    : {equity[-1]:.2f} USDT")
    print(f"PnL total        : {total_pnl:+.2f} USDT ({total_pnl/initial_cash*100:+.1f}%)")
    print(f"Trades total     : {len(trades)}")
    print(f"Win Rate         : {len(wins)/len(trades)*100:.1f}%" if trades else "Win Rate : N/A")
    print(f"Avg Win          : {sum(t['pnl'] for t in wins)/len(wins):.2f} USDT" if wins else "Avg Win : N/A")
    print(f"Avg Loss         : {sum(t['pnl'] for t in losses)/len(losses):.2f} USDT" if losses else "Avg Loss : N/A")

    # Max drawdown
    peak, max_dd = equity[0], 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    print(f"Max Drawdown     : {max_dd*100:.1f}%")
    print("="*50)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.to_csv("logs/backtest_trades_simple.csv", index=False)
        print("💾 logs/backtest_trades_simple.csv")

    return trades, equity


# ==========================
# CLI
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',   default='BTCUSDT')
    parser.add_argument('--interval', default='30m')
    parser.add_argument('--start',    default='2023-01-01')
    parser.add_argument('--cash',     type=float, default=10_000.0)
    parser.add_argument('--simple',   action='store_true',
                        help='Backtest simple sans vectorbt')
    args = parser.parse_args()

    import os
    os.makedirs("logs", exist_ok=True)

    df = download_history(args.symbol, args.interval, args.start)

    if args.simple or not VBT_AVAILABLE:
        run_simple_backtest(df, args.cash)
    else:
        run_backtest(df, args.cash)
