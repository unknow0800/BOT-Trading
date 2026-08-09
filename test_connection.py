from binance.client import Client
import time
import hmac
import hashlib
import requests
import os

API_KEY    = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")
BASE_URL = "https://testnet.binance.vision/api"

# Récupère le temps serveur directement
r = requests.get(f"{BASE_URL}/v3/time")
server_time = r.json()['serverTime']
print(f"Serveur time : {server_time}")
print(f"Local time   : {int(time.time() * 1000)}")
print(f"Différence   : {server_time - int(time.time() * 1000)} ms")

# Construit la requête signée manuellement avec le temps SERVEUR
params = f"timestamp={server_time}"
signature = hmac.new(
    API_SECRET.encode('utf-8'),
    params.encode('utf-8'),
    hashlib.sha256
).hexdigest()

headers = {"X-MBX-APIKEY": API_KEY}
url = f"{BASE_URL}/v3/account?{params}&signature={signature}"

response = requests.get(url, headers=headers)
print(f"\nStatus: {response.status_code}")
print(response.json())