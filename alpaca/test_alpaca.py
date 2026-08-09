from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import os

API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

start = datetime.now() - timedelta(days=5)

bars = data_client.get_stock_bars(StockBarsRequest(
    symbol_or_symbols=["AAPL", "GOOG", "MSFT"],
    timeframe=TimeFrame.Hour,
    start=start,
))

# Inspecter la structure
print(type(bars))
print(dir(bars))
print(bars)