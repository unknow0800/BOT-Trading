import os
import csv
import json
import logging
import math
import sys
from datetime import datetime
from typing import Optional
from config import LOG_DIR, TRADES_LOG_FILE, PERF_LOG_FILE

os.makedirs(LOG_DIR, exist_ok=True)

# Fix encoding Windows (cp1252 ne supporte pas les emojis)
# On force UTF-8 sur le StreamHandler
_stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    if sys.platform == 'win32' else sys.stdout
)
_stream_handler.setFormatter(
    logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
)

_file_handler = logging.FileHandler(f"{LOG_DIR}/bot.log", encoding='utf-8')
_file_handler.setFormatter(
    logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
)

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])


class BotLogger:
    def __init__(self):
        self.logger = logging.getLogger("BotExpert")
        self._init_trades_csv()

    def log_info(self, msg: str):
        self.logger.info(msg)

    def log_warning(self, msg: str):
        self.logger.warning(f"[WARN] {msg}")

    def log_error(self, msg: str):
        self.logger.error(f"[ERR] {msg}")

    def log_context(self, ctx: dict):
        self.logger.info(
            f"[MKT] Tendance 4H: {ctx['trend_4h'].upper()} | "
            f"Close: {ctx['close_30m']} | RSI: {ctx['rsi_30m']} | "
            f"VWAP: {ctx['vwap_30m']} | OBV+: {ctx['obv_bullish']} | "
            f"VolSpike: {ctx['vol_spike']} | ATR: {ctx['atr_30m']} | "
            f"Heures OK: {ctx['trading_hours']}"
        )

    def _init_trades_csv(self):
        if not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date', 'symbol', 'side', 'quantity',
                    'price', 'pnl_usdt', 'pnl_pct', 'r_multiple',
                    'reason', 'duration_min'
                ])

    def log_trade(self, symbol: str, side: str, quantity: float,
                  price: float, entry_price: float = 0.0,
                  entry_time: Optional[datetime] = None,
                  r_multiple: float = 0.0, reason: str = ''):
        now = datetime.utcnow()
        pnl_usdt = pnl_pct = duration = 0
        if side == 'sell' and entry_price > 0:
            pnl_usdt = (price - entry_price) * quantity
            pnl_pct  = (price - entry_price) / entry_price * 100
        if entry_time:
            duration = int((now - entry_time).total_seconds() / 60)
        with open(TRADES_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime('%Y-%m-%d %H:%M:%S'), symbol, side, quantity,
                round(price, 2), round(pnl_usdt, 2),
                round(pnl_pct, 3), round(r_multiple, 2), reason, duration
            ])
        if side == 'sell':
            sign = '+' if pnl_usdt >= 0 else ''
            self.logger.info(
                f"[TRADE] Ferme | PnL: {sign}{pnl_usdt:.2f} USDT "
                f"({sign}{pnl_pct:.2f}%) | {r_multiple:.1f}R | {duration} min"
            )

    def update_performance(self, equity_history: list):
        if len(equity_history) < 2:
            return
        returns = [
            (equity_history[i] - equity_history[i-1]) / equity_history[i-1]
            for i in range(1, len(equity_history))
        ]
        trades = self._load_trades()
        wins   = [t for t in trades if float(t['pnl_usdt']) > 0]
        losses = [t for t in trades if float(t['pnl_usdt']) <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win  = sum(float(w['pnl_usdt']) for w in wins) / len(wins) if wins else 0
        avg_loss = sum(float(l['pnl_usdt']) for l in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        if len(returns) > 1:
            mean_r = sum(returns) / len(returns)
            std_r  = math.sqrt(sum((r - mean_r)**2 for r in returns) / len(returns))
            sharpe = (mean_r / std_r * math.sqrt(17520)) if std_r > 0 else 0
        else:
            sharpe = 0
        peak = equity_history[0]
        max_dd = 0.0
        for eq in equity_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        metrics = {
            'last_updated'    : datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trades'    : len(trades),
            'win_rate_pct'    : round(win_rate, 1),
            'profit_factor'   : round(profit_factor, 2),
            'sharpe_ratio'    : round(sharpe, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'current_equity'  : round(equity_history[-1], 2),
            'total_pnl_usdt'  : round(equity_history[-1] - equity_history[0], 2),
        }
        with open(PERF_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(
            f"[PERF] Trades: {metrics['total_trades']} | "
            f"WinRate: {metrics['win_rate_pct']}% | "
            f"Sharpe: {metrics['sharpe_ratio']} | "
            f"MaxDD: {metrics['max_drawdown_pct']}% | "
            f"PnL: {metrics['total_pnl_usdt']:+.2f} USDT"
        )
        return metrics

    def _load_trades(self) -> list:
        if not os.path.exists(TRADES_LOG_FILE):
            return []
        with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))