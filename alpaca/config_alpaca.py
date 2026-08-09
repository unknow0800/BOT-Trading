import json
import os

ALPACA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_local_secrets() -> dict:
    secrets_path = os.path.join(ALPACA_DIR, "secrets_alpaca.json")
    if not os.path.exists(secrets_path):
        return {}
    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


LOCAL_SECRETS = _load_local_secrets()

# ==========================
# ALPACA PAPER TRADING
# ==========================
AALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", LOCAL_SECRETS.get("ALPACA_API_KEY"))
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", LOCAL_SECRETS.get("ALPACA_API_SECRET"))
ALPACA_PAPER      = True

# ==========================
# UNIVERS D'ACTIFS
# ==========================
# Alpaca donne surtout acces aux actions/ETF US. Pour surveiller "d'autres
# marches", on passe donc par des ETF/ADRs liquides qui servent de proxies
# macro: Europe, Chine, Japon, emergents, obligations, dollar, or, petrole...
SYMBOLS = {
    'mega_tech'      : ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NFLX', 'TSLA'],
    'semiconductors' : ['NVDA', 'AMD', 'AVGO', 'INTC', 'QCOM', 'TSM', 'ASML', 'MU'],
    'banks_us'       : ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC'],
    'finance_giants' : ['BRK.B', 'BLK', 'SCHW', 'AXP', 'V', 'MA', 'PYPL', 'COIN'],
    'insurance'      : ['AIG', 'TRV', 'CB', 'MET', 'PRU'],
    'consumer'       : ['WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX'],
    'healthcare'     : ['UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV'],
    'energy'         : ['XOM', 'CVX', 'COP', 'SLB', 'OXY'],
    'industrials'    : ['CAT', 'DE', 'GE', 'BA', 'HON', 'UPS'],
    'market_etf'     : ['SPY', 'QQQ', 'IWM', 'DIA', 'RSP', 'VTI'],
    'sector_etf'     : ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB', 'XLRE'],
    'global_proxies' : ['EFA', 'VGK', 'EWU', 'EWG', 'EWJ', 'FXI', 'ASHR', 'INDA', 'EEM', 'EWZ'],
    'macro_proxies'  : ['TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'UUP', 'GLD', 'SLV', 'USO', 'UNG'],
}
ALL_SYMBOLS = [s for group in SYMBOLS.values() for s in group]
SYMBOL_TO_GROUP = {
    symbol: group
    for group, symbols in SYMBOLS.items()
    for symbol in symbols
}

# ==========================
# TIMEFRAMES
# ==========================
TIMEFRAME_TREND  = 'Day'    # Tendance macro
TIMEFRAME_SIGNAL = 'Hour'   # Signal entree
LIMIT_BARS       = 300      # Nombre de bougies par symbole
MAX_BARS_PER_REQUEST = 10_000
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # iex, sip, delayed_sip

# ==========================
# RISK MANAGEMENT
# ==========================
RISK_PER_TRADE_PCT   = 0.01   # 1% du capital par trade
ATR_SL_MULTIPLIER    = 2.0
ATR_TP_MULTIPLIER    = 4.0
MAX_DRAWDOWN_PCT     = 0.10
MAX_POSITIONS        = 3      # max 3 positions simultanées
POSITION_SIZE_CAP    = 0.20   # max 20% du capital sur un seul actif
MIN_CONFIDENCE_DL    = 0.60   # seuil confiance DL
MAX_POSITIONS_PER_GROUP = 1
MAX_AVG_CORRELATION_WITH_POSITIONS = 0.75
CORRELATION_LOOKBACK_BARS = 80
WATCHLIST_TOP_N = 10

# Mode defensif active apres les premieres pertes paper.
RISK_PER_TRADE_PCT = 0.003
MAX_POSITIONS = 1
POSITION_SIZE_CAP = 0.05

# ==========================
# FILTRES
# ==========================
MIN_VOLUME           = 500_000   # volume minimum (actions)
USE_EXTENDED_HOURS   = False     # False = 9h30-16h ET, True = 4h-20h ET
MANAGE_POSITIONS_OUTSIDE_MARKET_HOURS = False
ENABLE_NO_OVERNIGHT = True
NO_OVERNIGHT_EXIT_TIME_ET = "15:55"
ENABLE_MARKET_REGIME_FILTER = True
MARKET_REGIME_SYMBOLS = ["SPY", "QQQ"]
LOSS_COOLDOWN_HOURS = 24
REQUIRE_BACKTEST_APPROVAL = True
AVOID_HOURS_ET = [0, 1, 2, 3, 4, 5, 6, 7, 8,   # avant 9h ET
                  20, 21, 22, 23]               # après 20h ET

# ==========================
# INDICATEURS
# ==========================
RSI_PERIOD      = 14
RSI_OVERSOLD    = 35
RSI_OVERBOUGHT  = 65
SMA_FAST        = 20
SMA_SLOW        = 50
ATR_PERIOD      = 14

# Score technique avance: le bot n'achete plus seulement sur un croisement RSI
# tres rare. Il combine tendance, momentum, volume, volatilite et breadth macro.
MIN_TECHNICAL_SCORE_TO_BUY = 0.62
MIN_TECHNICAL_SCORE_STRONG = 0.78
ADAPTIVE_THRESHOLD_ENABLED = True
ADAPTIVE_THRESHOLD_STEP = 0.03
ADAPTIVE_THRESHOLD_MIN = -0.06
ADAPTIVE_THRESHOLD_MAX = 0.09

# ==========================
# LOGS
# ==========================
LOG_DIR              = os.path.join(ALPACA_DIR, "logs", "alpaca")
TRADES_LOG_FILE      = os.path.join(LOG_DIR, "trades.csv")
PERF_LOG_FILE        = os.path.join(LOG_DIR, "performance.json")
DECISIONS_LOG_FILE   = os.path.join(LOG_DIR, "decisions.csv")
SETUP_RANKING_LOG_FILE = os.path.join(LOG_DIR, "setup_ranking.csv")
WATCHLIST_LOG_FILE   = os.path.join(LOG_DIR, "watchlist.csv")
TRADE_REVIEW_LOG_FILE = os.path.join(LOG_DIR, "trade_review.csv")
ADAPTIVE_STATE_FILE  = os.path.join(LOG_DIR, "adaptive_state.json")
COOLDOWN_STATE_FILE  = os.path.join(LOG_DIR, "cooldown_state.json")
BACKTEST_APPROVAL_FILE = os.path.join(LOG_DIR, "backtest_approval.json")
SLEEP_SECONDS        = 15 * 60   # 15 min pour surveiller risque/no-overnight

# ==========================
# MARKET INTELLIGENCE
# ==========================
# Couche d'analyse externe: news, sentiment, momentum de contexte,
# et TradingView si la librairie optionnelle tradingview_ta est installee.
ENABLE_MARKET_INTELLIGENCE = True
ENABLE_ALPACA_NEWS         = True
ENABLE_TRADINGVIEW_TA      = False

NEWS_LOOKBACK_HOURS        = 36
NEWS_LIMIT_PER_REQUEST     = 50
MIN_CONTEXT_SCORE_TO_BUY   = -0.15
MIN_CONTEXT_CONFIDENCE     = 0.25

SCORING_WEIGHTS = {
    "news": 0.28,
    "technical_context": 0.30,
    "tradingview": 0.12,
    "earnings": 0.12,
    "analyst": 0.10,
    "macro": 0.08,
}

# Sources optionnelles. Sans cle API, elles retournent un score neutre.
# PowerShell:
#   $env:FINNHUB_API_KEY="ta_cle"
#   $env:FMP_API_KEY="ta_cle"
# Linux/macOS:
#   export FINNHUB_API_KEY="ta_cle"
#   export FMP_API_KEY="ta_cle"
ENABLE_EARNINGS_CALENDAR  = True
ENABLE_ANALYST_RATINGS    = True
ENABLE_MACRO_EVENTS       = True
FINNHUB_API_KEY           = os.getenv("FINNHUB_API_KEY", LOCAL_SECRETS.get("FINNHUB_API_KEY", ""))
FMP_API_KEY               = os.getenv("FMP_API_KEY", LOCAL_SECRETS.get("FMP_API_KEY", ""))
