import time
import hmac
import hashlib
import requests
import pandas as pd
from binance.client import Client
from config import (SYMBOL, TIMEFRAME_TREND, TIMEFRAME_SIGNAL,
                    TIMEFRAME_EXEC, LIMIT_CANDLES,
                    API_KEY, API_SECRET)

BASE_URL = "https://testnet.binance.vision/api"

# Clés stockées au niveau module (évite pb accès client.API_KEY)
_API_KEY    = API_KEY
_API_SECRET = API_SECRET


def set_credentials(key: str, secret: str):
    global _API_KEY, _API_SECRET
    _API_KEY    = key
    _API_SECRET = secret


def _server_time() -> int:
    r = requests.get(f"{BASE_URL}/v3/time", timeout=5)
    return r.json()['serverTime']


def _sign(secret: str, params: str) -> str:
    return hmac.new(
        secret.encode('utf-8'),
        params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def _signed_get(endpoint: str, extra: str = "") -> dict:
    ts = _server_time()
    params = f"timestamp={ts}"
    if extra:
        params = f"{extra}&{params}"
    sig = _sign(_API_SECRET, params)
    url = f"{BASE_URL}{endpoint}?{params}&signature={sig}"
    headers = {"X-MBX-APIKEY": _API_KEY}
    r = requests.get(url, headers=headers, timeout=10)
    return r.json()


def _signed_post(endpoint: str, extra: str = "") -> dict:
    ts = _server_time()
    params = f"timestamp={ts}"
    if extra:
        params = f"{extra}&{params}"
    sig = _sign(_API_SECRET, params)
    full_params = f"{params}&signature={sig}"
    url = f"{BASE_URL}{endpoint}"
    headers = {"X-MBX-APIKEY": _API_KEY}
    r = requests.post(url, data=full_params, headers=headers, timeout=10)
    return r.json()


def sync_time(client: Client) -> int:
    local = int(time.time() * 1000)
    server = _server_time()
    return server - local


def _parse_klines(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
        df[col] = df[col].astype(float)
    df.set_index('timestamp', inplace=True)
    return df


def get_multi_timeframe_data(client: Client, symbol: str = SYMBOL) -> dict:
    """
    Recupere les donnees sur 3 timeframes via l'API publique testnet.
    Le 4H testnet peut avoir peu de bougies : on prend ce qui est dispo.
    """
    timeframes = {
        TIMEFRAME_TREND:  '4h',
        TIMEFRAME_SIGNAL: '30m',
        TIMEFRAME_EXEC:   '5m',
    }
    data = {}
    for label, interval in timeframes.items():
        url = f"{BASE_URL}/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": LIMIT_CANDLES}
        r = requests.get(url, params=params, timeout=10)
        klines = r.json()
        data[label] = _parse_klines(klines)
    return data


def get_current_price(client: Client, symbol: str = SYMBOL) -> float:
    r = requests.get(f"{BASE_URL}/v3/ticker/price",
                     params={"symbol": symbol}, timeout=5)
    return float(r.json()['price'])


def get_account_balance(client: Client, asset: str = 'USDT') -> float:
    data = _signed_get("/v3/account")
    if 'balances' not in data:
        print(f"Erreur get_account_balance : {data}")
        return 0.0
    for b in data['balances']:
        if b['asset'] == asset:
            return float(b['free'])
    return 0.0


def get_btc_balance(client: Client) -> float:
    return get_account_balance(client, asset='BTC')


def get_symbol_filters(client: Client, symbol: str = SYMBOL):
    r = requests.get(f"{BASE_URL}/v3/exchangeInfo",
                     params={"symbol": symbol}, timeout=10)
    info = r.json()
    sym = info['symbols'][0]
    filters = {f['filterType']: f for f in sym['filters']}
    lot_size = float(filters['LOT_SIZE']['stepSize'])
    if 'MIN_NOTIONAL' in filters:
        min_notional = float(filters['MIN_NOTIONAL']['minNotional'])
    elif 'NOTIONAL' in filters:
        min_notional = float(filters['NOTIONAL'].get('minNotional', 10.0))
    else:
        min_notional = 10.0
    return lot_size, min_notional


def place_order(client: Client, symbol: str, side: str, quantity: float) -> dict:
    extra = (f"symbol={symbol}&side={side}&type=MARKET"
             f"&quantity={quantity}")
    return _signed_post("/v3/order", extra)
