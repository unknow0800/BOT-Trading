"""
backtest_context.py - Compare la strategie Alpaca avec/sans Market Intelligence.

Usage:
    python alpaca/backtest_context.py --symbol AAPL --days 180
    python alpaca/backtest_context.py --symbol AAPL --days 180 --news-csv data/news.csv

Le CSV news optionnel doit contenir: date,symbol,headline,summary.
Sans CSV, le backtest teste le moteur de scoring historique sur les donnees
prix/volume uniquement; les sources live (TradingView, earnings, ratings, macro)
sont desactivees pour eviter le look-ahead bias.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config_alpaca import (
    ALL_SYMBOLS,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    BACKTEST_APPROVAL_FILE,
    ENABLE_MARKET_REGIME_FILTER,
    ENABLE_NO_OVERNIGHT,
    LOSS_COOLDOWN_HOURS,
    LOG_DIR,
    MARKET_REGIME_SYMBOLS,
    MIN_CONTEXT_SCORE_TO_BUY,
    NEWS_LOOKBACK_HOURS,
    NO_OVERNIGHT_EXIT_TIME_ET,
    POSITION_SIZE_CAP,
    RISK_PER_TRADE_PCT,
)
from data_alpaca import get_bars, get_clients
from indicators import add_all_indicators
from market_intelligence import build_market_context
from strategy_alpaca import analyze_hourly_signal, get_trend_bias_daily


@dataclass
class BacktestResult:
    name: str
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: list
    decisions: list


class CsvArticle:
    def __init__(self, headline: str, summary: str, symbols: list[str]):
        self.headline = headline
        self.summary = summary
        self.symbols = symbols


def load_news_csv(path: str | None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["date", "symbol", "headline", "summary"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def get_historical_articles(news_df: pd.DataFrame, symbol: str, timestamp) -> list:
    if news_df.empty:
        return []
    start = timestamp - timedelta(hours=NEWS_LOOKBACK_HOURS)
    rows = news_df[
        (news_df["symbol"] == symbol)
        & (news_df["date"] <= timestamp)
        & (news_df["date"] >= start)
    ].tail(20)
    return [
        CsvArticle(
            headline=str(row.get("headline", "")),
            summary=str(row.get("summary", "")),
            symbols=[symbol],
        )
        for _, row in rows.iterrows()
    ]


def max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd * 100


def profit_factor(trades: list) -> float:
    gross_profit = sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0)
    gross_loss = abs(sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0))
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _parse_et_time(value: str):
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def _is_no_overnight_window(ts) -> bool:
    if not ENABLE_NO_OVERNIGHT:
        return False
    ts_et = ts.astimezone(ZoneInfo("America/New_York"))
    if ts_et.weekday() >= 5:
        return False
    hour, minute = _parse_et_time(NO_OVERNIGHT_EXIT_TIME_ET)
    return (ts_et.hour, ts_et.minute) >= (hour, minute)


def _market_regime_allows_long(ts, bars_daily: dict, bars_hourly: dict) -> tuple[bool, str]:
    if not ENABLE_MARKET_REGIME_FILTER:
        return True, "regime_filter_off"
    for symbol in MARKET_REGIME_SYMBOLS:
        df_d = bars_daily.get(symbol)
        df_h = bars_hourly.get(symbol)
        if df_d is None or df_h is None:
            return False, f"regime_data_missing:{symbol}"
        daily_slice = df_d[df_d.index <= ts].tail(100)
        hourly_slice = df_h[df_h.index <= ts].tail(80)
        if len(daily_slice) < 20 or len(hourly_slice) < 50:
            return False, f"regime_history_missing:{symbol}"
        trend = get_trend_bias_daily(daily_slice)
        sma_slow = hourly_slice["close"].rolling(min(50, len(hourly_slice))).mean().iloc[-1]
        above_slow = float(hourly_slice["close"].iloc[-1]) > float(sma_slow)
        if trend != "bull" or not above_slow:
            return False, f"regime_block:{symbol} trend={trend} above_sma50={above_slow}"
    return True, "regime_ok"


def run_one_mode(
    name: str,
    symbol: str,
    df_daily: pd.DataFrame,
    df_hourly: pd.DataFrame,
    initial_cash: float,
    use_context: bool,
    news_df: pd.DataFrame,
    all_daily: dict,
    all_hourly: dict,
) -> BacktestResult:
    df_hourly = add_all_indicators(df_hourly.copy())
    cash = initial_cash
    qty = 0.0
    entry_price = 0.0
    entry_time = None
    sl = 0.0
    tp = 0.0
    trades = []
    decisions = []
    equity_curve = []
    cooldown_until = None

    for i in range(60, len(df_hourly)):
        ts = df_hourly.index[i]
        row = df_hourly.iloc[i]
        price = float(row["close"])
        atr = float(row["atr"])
        daily_slice = df_daily[df_daily.index <= ts].tail(100)
        hourly_slice = df_hourly.iloc[: i + 1].tail(80)
        if len(daily_slice) < 20 or len(hourly_slice) < 20 or atr <= 0:
            continue

        equity = cash + qty * price
        equity_curve.append(equity)

        if qty > 0:
            if _is_no_overnight_window(ts):
                cash += qty * price
                pnl = (price - entry_price) * qty
                trades.append({
                    "mode": name,
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry": round(entry_price, 4),
                    "exit": round(price, 4),
                    "qty": round(qty, 4),
                    "pnl": round(pnl, 2),
                    "reason": "no_overnight",
                })
                qty = 0.0
                entry_price = 0.0
                entry_time = None
                continue

            exit_analysis = analyze_hourly_signal(
                hourly_slice, "bull", in_position=True,
                symbol=symbol, require_market_open=False
            )
            signal_exit = exit_analysis["signal"]
            exit_reason = None
            if price <= sl:
                exit_reason = "sl"
            elif price >= tp:
                exit_reason = "tp"
            elif signal_exit == "exit_long":
                exit_reason = "signal"

            if exit_reason:
                cash += qty * price
                pnl = (price - entry_price) * qty
                trades.append({
                    "mode": name,
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry": round(entry_price, 4),
                    "exit": round(price, 4),
                    "qty": round(qty, 4),
                    "pnl": round(pnl, 2),
                    "reason": exit_reason,
                })
                qty = 0.0
                entry_price = 0.0
                entry_time = None
                if pnl < 0 and LOSS_COOLDOWN_HOURS > 0:
                    cooldown_until = ts + timedelta(hours=LOSS_COOLDOWN_HOURS)
            continue

        if cooldown_until is not None and ts < cooldown_until:
            decisions.append({
                "mode": name,
                "date": ts,
                "symbol": symbol,
                "price": round(price, 4),
                "trend": "",
                "signal": "hold",
                "technical_score": 0.0,
                "decision": "refused",
                "reason": f"cooldown_loss_until:{cooldown_until}",
                "context_score": "",
                "context_threshold": "",
                "context_reason": "",
            })
            continue

        if _is_no_overnight_window(ts):
            continue

        regime_ok, regime_reason = _market_regime_allows_long(ts, all_daily, all_hourly)

        trend = get_trend_bias_daily(daily_slice)
        analysis = analyze_hourly_signal(
            hourly_slice, trend, in_position=False,
            symbol=symbol, require_market_open=False
        )
        signal = analysis["signal"]
        context = None
        allow_context = True
        if use_context:
            articles = get_historical_articles(news_df, symbol, ts)
            context = build_market_context(
                symbol,
                daily_slice,
                hourly_slice,
                articles=articles,
                include_live_external=False,
            )
            allow_context = context.allow_long

        decision = "accepted" if signal == "enter_long" and allow_context and regime_ok else "refused"
        reason = "buy" if decision == "accepted" else f"technical_context_or_regime_filter:{regime_reason}"
        decisions.append({
            "mode": name,
            "date": ts,
            "symbol": symbol,
            "price": round(price, 4),
            "trend": trend,
            "signal": signal,
            "technical_score": analysis["score"],
            "decision": decision,
            "reason": f"{reason} | {analysis['reason']}",
            "context_score": round(context.score, 4) if context else "",
            "context_threshold": MIN_CONTEXT_SCORE_TO_BUY if context else "",
            "context_reason": context.reason if context else "",
        })

        if decision == "accepted":
            risk_amount = cash * RISK_PER_TRADE_PCT
            per_unit_risk = ATR_SL_MULTIPLIER * atr
            qty = risk_amount / per_unit_risk if per_unit_risk > 0 else 0.0
            max_qty = (cash * POSITION_SIZE_CAP) / price if price > 0 else 0.0
            qty = min(qty, max_qty)
            cost = qty * price
            if qty <= 0 or cost > cash:
                qty = 0.0
                decisions[-1]["decision"] = "refused"
                decisions[-1]["reason"] = "position_size_invalid"
                continue
            cash -= cost
            entry_price = price
            entry_time = ts
            sl = price - ATR_SL_MULTIPLIER * atr
            tp = price + ATR_TP_MULTIPLIER * atr

    final_price = float(df_hourly["close"].iloc[-1])
    final_equity = cash + qty * final_price
    total_return_pct = (final_equity / initial_cash - 1) * 100
    dd = max_drawdown(equity_curve or [initial_cash])
    return BacktestResult(name, final_equity, total_return_pct, dd, trades, decisions)


def save_rows(path: str, rows: list) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_result(result: BacktestResult) -> None:
    wins = [t for t in result.trades if t["pnl"] > 0]
    win_rate = len(wins) / len(result.trades) * 100 if result.trades else 0.0
    pf = profit_factor(result.trades)
    accepted = [d for d in result.decisions if d["decision"] == "accepted"]
    print(f"\n{result.name}")
    print("-" * len(result.name))
    print(f"Capital final  : ${result.final_equity:,.2f}")
    print(f"Return total   : {result.total_return_pct:+.2f}%")
    print(f"Max drawdown   : {result.max_drawdown_pct:.2f}%")
    print(f"Trades         : {len(result.trades)}")
    print(f"Win rate       : {win_rate:.1f}%")
    print(f"Profit factor  : {pf:.2f}")
    print(f"Decisions buy  : {len(accepted)} / {len(result.decisions)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--symbols", default="", help="Liste de symboles separes par des virgules")
    parser.add_argument("--all", action="store_true", help="Backteste tous les actifs de config_alpaca.ALL_SYMBOLS")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--news-csv", default="")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    _, data_client = get_clients()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = ALL_SYMBOLS if args.all else [args.symbol]
    bars_daily = get_bars(data_client, symbols, "Day", days_back=args.days + 80)
    bars_hourly = get_bars(data_client, symbols, "Hour", days_back=args.days)

    news_df = load_news_csv(args.news_csv)
    results = []
    skipped = []
    for symbol in symbols:
        if symbol not in bars_daily or symbol not in bars_hourly:
            skipped.append(symbol)
            continue
        baseline = run_one_mode(
            "baseline_sans_context",
            symbol,
            bars_daily[symbol],
            bars_hourly[symbol],
            args.cash,
            use_context=False,
            news_df=news_df,
            all_daily=bars_daily,
            all_hourly=bars_hourly,
        )
        context = run_one_mode(
            "avec_market_intelligence",
            symbol,
            bars_daily[symbol],
            bars_hourly[symbol],
            args.cash,
            use_context=True,
            news_df=news_df,
            all_daily=bars_daily,
            all_hourly=bars_hourly,
        )
        results.extend([baseline, context])

    if not results:
        raise RuntimeError("Pas assez de donnees Alpaca pour backtester les symboles demandes")

    print("\nCOMPARAISON BACKTEST")
    print("====================")
    for result in results:
        print_result(result)

    summary_rows = []
    for result in results:
        wins = [t for t in result.trades if t["pnl"] > 0]
        summary_rows.append({
            "mode": result.name,
            "symbol": result.decisions[0]["symbol"] if result.decisions else "",
            "final_equity": round(result.final_equity, 2),
            "return_pct": round(result.total_return_pct, 3),
            "max_drawdown_pct": round(result.max_drawdown_pct, 3),
            "trades": len(result.trades),
            "win_rate": round(len(wins) / len(result.trades) * 100, 2) if result.trades else 0.0,
            "profit_factor": round(profit_factor(result.trades), 3),
        })

    all_trades = [trade for result in results for trade in result.trades]
    all_decisions = [decision for result in results for decision in result.decisions]
    save_rows(f"{LOG_DIR}/backtest_context_trades.csv", all_trades)
    save_rows(f"{LOG_DIR}/backtest_context_decisions.csv", all_decisions)
    save_rows(f"{LOG_DIR}/backtest_context_summary.csv", summary_rows)
    print(f"\nCSV: {LOG_DIR}/backtest_context_trades.csv")
    print(f"CSV: {LOG_DIR}/backtest_context_decisions.csv")
    print(f"CSV: {LOG_DIR}/backtest_context_summary.csv")
    if skipped:
        print(f"Symboles ignores faute de donnees: {', '.join(skipped[:30])}")

    context_results = [result for result in results if result.name == "avec_market_intelligence"]
    context_trades = [trade for result in context_results for trade in result.trades]
    context_equity = sum(result.final_equity for result in context_results)
    context_initial = args.cash * len(context_results) if context_results else args.cash
    context_return = (context_equity / context_initial - 1) * 100 if context_initial else 0.0
    context_dd = max((result.max_drawdown_pct for result in context_results), default=100.0)
    context_pf = profit_factor(context_trades)
    approval = {
        "approved": (
            len(context_trades) >= 30
            and context_pf >= 1.15
            and context_dd <= 8.0
            and context_return > 0
        ),
        "mode": "avec_market_intelligence",
        "trades": len(context_trades),
        "min_trades": 30,
        "profit_factor": round(context_pf, 4),
        "max_drawdown_pct": round(context_dd, 4),
        "return_pct": round(context_return, 4),
        "generated_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
    }
    with open(BACKTEST_APPROVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(approval, f, indent=2)
    print(f"Approval: {BACKTEST_APPROVAL_FILE} -> approved={approval['approved']}")


if __name__ == "__main__":
    main()
