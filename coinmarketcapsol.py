
import requests
import pandas as pd
from datetime import datetime

# 🔑 Tu API Key de CoinMarketCap
API_KEY = "6607a9e2-d5a5-4433-9602-b7f7803be5fb"
symbol = "SOL"

url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
params = {"symbol": symbol, "convert": "USD"}
headers = {"X-CMC_PRO_API_KEY": API_KEY}

response = requests.get(url, headers=headers, params=params)
data = response.json()

# 🔍 Ver la respuesta completa
print(data)