import os

# ==========================
# BINANCE TESTNET CONFIG
# ==========================

API_KEY    = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")
TESTNET          = True
TESTNET_BASE_URL = "https://testnet.binance.vision"

# ==========================
# SYMBOLE & TIMEFRAMES
# ==========================

SYMBOL = 'BTCUSDT'

# Testnet 4H a peu d'historique -> on utilise 1H pour la tendance
TIMEFRAME_TREND  = '1h'    # Timeframe macro tendance (1H sur testnet)
TIMEFRAME_SIGNAL = '30m'   # Timeframe signal entree
TIMEFRAME_EXEC   = '5m'    # Timeframe execution

LIMIT_CANDLES = 300

# ==========================
# GESTION DU RISQUE
# ==========================

RISK_PER_TRADE_PCT  = 0.01
ATR_SL_MULTIPLIER   = 2.0
ATR_TP_MULTIPLIER   = 4.0
BREAKEVEN_TRIGGER_R = 1.0
SCALE_OUT_PCT       = 0.5
MAX_DRAWDOWN_PCT    = 0.10
MAX_OPEN_TRADES     = 1

# ==========================
# FILTRES HORAIRES
# ==========================

AVOID_HOURS_UTC = list(range(22, 24)) + list(range(0, 6))

# ==========================
# INDICATEURS
# ==========================

RSI_PERIOD      = 14
RSI_OVERSOLD    = 30
RSI_OVERBOUGHT  = 70
SMA_FAST        = 50
SMA_SLOW        = 200
ATR_PERIOD      = 14
OBV_SMA_PERIOD  = 20

# ==========================
# LOGS
# ==========================

LOG_DIR         = "logs"
TRADES_LOG_FILE = "logs/trades.csv"
PERF_LOG_FILE   = "logs/performance.json"
SLEEP_SECONDS   = 60 * 30