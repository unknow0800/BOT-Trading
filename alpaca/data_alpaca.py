"""
data_alpaca.py — Données de marché et compte via Alpaca API
"""
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import AssetClass, OrderClass, OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config_alpaca import (ALPACA_API_KEY, ALPACA_API_SECRET,
                       ALPACA_PAPER, LIMIT_BARS, MAX_BARS_PER_REQUEST,
                       ALPACA_DATA_FEED, NEWS_LIMIT_PER_REQUEST)


BAR_REQUEST_CHUNK_SIZE = 12
NEWS_REQUEST_CHUNK_SIZE = 20

DATA_FEEDS = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
    "delayed_sip": DataFeed.DELAYED_SIP,
    "otc": DataFeed.OTC,
}


def get_clients():
    trading = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    data    = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    return trading, data


def get_news_client():
    return NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)


def get_account_info(trading_client) -> dict:
    acc = trading_client.get_account()
    return {
        'cash'         : float(acc.cash),
        'equity'       : float(acc.equity),
        'buying_power' : float(acc.buying_power),
        'pnl'          : float(acc.equity) - 100_000,
    }


def _bars_to_dataframe(bars) -> pd.DataFrame | None:
    if not bars:
        return None
    records = [{
        'timestamp': b.timestamp,
        'open'     : b.open,
        'high'     : b.high,
        'low'      : b.low,
        'close'    : b.close,
        'volume'   : b.volume,
        'vwap'     : b.vwap if b.vwap else b.close,
    } for b in bars]
    df = pd.DataFrame(records)
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _fetch_bar_chunk(data_client, symbols: list, tf, start) -> dict:
    request_limit = min(MAX_BARS_PER_REQUEST, LIMIT_BARS * max(1, len(symbols)))
    feed = DATA_FEEDS.get(str(ALPACA_DATA_FEED).lower(), DataFeed.IEX)
    barset = data_client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start,
        limit=request_limit,
        feed=feed
    ))
    bars_by_symbol = getattr(barset, 'data', barset)

    result = {}
    for symbol, bars in bars_by_symbol.items():
        df = _bars_to_dataframe(bars)
        if df is not None:
            result[symbol] = df
    return result


def get_bars(data_client, symbols: list, timeframe: str = 'Hour',
             days_back: int = 30, logger=None) -> dict:
    """
    Retourne un dict {symbol: pd.DataFrame} avec colonnes OHLCV.
    Alpaca-py retourne normalement un BarSet dont les donnees sont dans .data.
    """
    tf_map = {
        'Hour' : TimeFrame(1, TimeFrameUnit.Hour),
        'Day'  : TimeFrame(1, TimeFrameUnit.Day),
        'Min'  : TimeFrame(15, TimeFrameUnit.Minute),
    }
    tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))
    start = datetime.now(timezone.utc) - timedelta(days=days_back)

    result = {}
    for i in range(0, len(symbols), BAR_REQUEST_CHUNK_SIZE):
        chunk = symbols[i:i + BAR_REQUEST_CHUNK_SIZE]
        try:
            if logger:
                request_limit = min(MAX_BARS_PER_REQUEST, LIMIT_BARS * max(1, len(chunk)))
                logger.info(
                    f"[DATA] Bars {timeframe} chunk {i//BAR_REQUEST_CHUNK_SIZE + 1} "
                    f"({len(chunk)} symboles) feed={ALPACA_DATA_FEED} limit={request_limit}"
                )
            result.update(_fetch_bar_chunk(data_client, chunk, tf, start))
        except Exception as exc:
            if logger:
                logger.warning(f"[DATA] Chunk bars echoue {chunk}: {exc}")
            for symbol in chunk:
                try:
                    result.update(_fetch_bar_chunk(data_client, [symbol], tf, start))
                except Exception as sym_exc:
                    if logger:
                        logger.warning(f"[DATA] {symbol} ignore: {sym_exc}")

    missing = sorted(set(symbols) - set(result))
    if missing and logger:
        logger.warning(f"[DATA] Donnees absentes ({len(missing)}): {', '.join(missing[:20])}")

    return result


def get_latest_price(data_client, symbol: str) -> float:
    quote = data_client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    )
    q = quote[symbol]
    return (q.ask_price + q.bid_price) / 2


def get_news_articles(news_client, symbols: list, hours_back: int = 24) -> dict:
    """Retourne {symbol: [articles]} depuis Alpaca News."""
    if not symbols:
        return {}

    result = {symbol: [] for symbol in symbols}
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)

    for i in range(0, len(symbols), NEWS_REQUEST_CHUNK_SIZE):
        chunk = symbols[i:i + NEWS_REQUEST_CHUNK_SIZE]
        response = news_client.get_news(NewsRequest(
            symbols=",".join(chunk),
            start=start,
            end=end,
            sort="desc",
            limit=NEWS_LIMIT_PER_REQUEST,
            include_content=False,
            exclude_contentless=True,
        ))

        articles = getattr(response, 'data', {}).get('news', [])
        for article in articles:
            for symbol in getattr(article, 'symbols', []) or []:
                if symbol in result:
                    result[symbol].append(article)
    return result


def get_open_positions(trading_client) -> dict:
    """Retourne {symbol: {'qty': float, 'avg_price': float, 'market_value': float}}"""
    positions = trading_client.get_all_positions()
    return {
        p.symbol: {
            'qty'         : float(p.qty),
            'avg_price'   : float(p.avg_entry_price),
            'market_value': float(p.market_value),
            'unrealized_pl': float(p.unrealized_pl),
        }
        for p in positions
    }


def place_market_order(trading_client, symbol: str, side: str,
                       qty: float, logger=None) -> dict | None:
    """
    Passe un ordre market BUY ou SELL.
    side : 'buy' | 'sell'
    """
    try:
        order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6),
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(req)
        msg = f"[ALPACA] {side.upper()} {qty:.4f} {symbol} | OrderID: {order.id}"
        if logger:
            logger.log_info(msg)
        else:
            print(msg)
        return {'id': str(order.id), 'symbol': symbol, 'side': side, 'qty': qty}
    except Exception as e:
        msg = f"[ALPACA] Erreur ordre {side} {symbol}: {e}"
        if logger:
            logger.log_error(msg)
        else:
            print(msg)
        return None


def place_bracket_order(trading_client, symbol: str, qty: float,
                        take_profit_price: float, stop_price: float,
                        logger=None) -> dict | None:
    """Passe un ordre bracket: entree market + TP/SL chez Alpaca."""
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
        )
        order = trading_client.submit_order(req)
        msg = (
            f"[ALPACA] BRACKET BUY {qty:.4f} {symbol} | "
            f"TP={take_profit_price:.2f} SL={stop_price:.2f} | OrderID: {order.id}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return {'id': str(order.id), 'symbol': symbol, 'side': 'buy', 'qty': qty}
    except Exception as e:
        msg = f"[ALPACA] Erreur bracket {symbol}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None


def close_position(trading_client, symbol: str, logger=None) -> bool:
    """Ferme entièrement une position existante."""
    try:
        trading_client.close_position(symbol)
        msg = f"[ALPACA] Position {symbol} fermée"
        if logger:
            logger.log_info(msg)
        else:
            print(msg)
        return True
    except Exception as e:
        msg = f"[ALPACA] Erreur fermeture {symbol}: {e}"
        if logger:
            logger.log_error(msg)
        else:
            print(msg)
        return False
