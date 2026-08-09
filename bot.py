import time
import pandas as pd #analysise du jeu de données
import ta 
from binance.client import Client #API Binance pour se connecter à Binance
from binance.enums import * #importer toutes les constantes de Binance

# Clé API et secret pour se connecter à Binance
api_key= 'API_KEY'
api_secret= 'API_SECREAT_KEY'

#se connecter à Binance
Client = Client(api_key, api_secret)
# interagir avec l'API Binance pour récupérer les données de trading

symbol = 'BTCUSDT' #symbole de trading bitcoin contre le dollar américain
Quantity = 0.001 #quantité de bitcoin à acheter ou vendre

timeframe = Client.KLINE_INTERVAL_30MINUTE #intervalle de temps pour les données de trading (30 minutes)
LIMIT = 50 #nombre de données de trading à récupérer
SLEEP_TIME = 60*60 #temps d'attente entre chaque exécution du bot (1 heure)

def get_market_data(symbol, timeframe, limit):
    # recuperer les dernieres bougies ohlcv
    klines= Client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') #convertir le timestamp en format datetime
    df['close'] = df['close'].astype(float) #convertir les prix de clôture en float
    return df

def add_indicators(df):
    # ajouter des indicateurs techniques pour l'analyse du marché
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi() #indice de force relative (RSI)
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator() #moyenne mobile simple sur 50 périodes
    df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator() #moyenne mobile simple sur 200 périodes
    return df

def get_trading_signal(df):
    last_row = df.iloc[-1] #récupérer la dernière ligne du DataFrame pour analyser les indicateurs
    if last_row['rsi'] < 30 and last_row['sma_50'] > last_row['sma_200']:
        return 'buy' #signal d'achat
    elif last_row['rsi'] > 70 and last_row['sma_50'] < last_row['sma_200']:
        return 'sell' #signal de vente
    else:
        return 'hold' #pas de signal clair, rester en position
    
def place_buy_order(symbol, quantity):
    try:
        order = Client.order_market_buy(
            symbol=symbol,
            quantity=quantity
        )
        print("Achat effectué:", {order})
    except Exception as e:
        print("Erreur lors de l'achat:", {e})

def place_sell_order(symbol, quantity):
    try:
        order = Client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )
        print("Vente effectuée:", {order})
    except Exception as e:
        print("Erreur lors de la vente:", {e})

while True:
    try:
        print("Récupération des données de marché...")
        df = get_market_data(symbol, timeframe, LIMIT) #récupérer les données de marché
        df = add_indicators(df) #ajouter les indicateurs techniques
        signal = get_trading_signal(df) #obtenir le signal de trading
        print(f"Dernier prix: {df.iloc[-1]['close']:.2f},RSI: {df.iloc[-1]['rsi']:.2f}, SMA 50: {df.iloc[-1]['sma_50']:.2f}, SMA 200: {df.iloc[-1]['sma_200']:.2f}, Signal: {signal}")
        if signal == 'buy':
            place_buy_order(symbol, Quantity) #placer un ordre d'achat
        elif signal == 'sell':
            place_sell_order(symbol, Quantity) #placer un ordre de vente
            print(f"Pause de {SLEEP_TIME/60} minutes avant la prochaine exécution...")
        else:
            print("Pas de signal clair, aucune action prise.")
    except Exception as e:
        print("Erreur lors de l'exécution du bot:", {e})
    time.sleep(10) #attendre avant la prochaine exécution du bot