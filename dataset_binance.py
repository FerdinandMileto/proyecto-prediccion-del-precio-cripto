from pycoingecko import CoinGeckoAPI
import pandas as pd

# Inicializar cliente
cg = CoinGeckoAPI()

# Parámetros
coin_id = "bitcoin"  # Cambia por ethereum, solana, etc.
vs_currency = "usd"
days = "30"  # Últimos 30 días

# Obtener datos
data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)

# Convertir a DataFrame
prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

# Unir los DataFrames
df = prices.merge(market_caps, on='timestamp').merge(volumes, on='timestamp')

# Convertir timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(df.head())
# Guardar el DataFrame en un CSV
nombre_archivo = f"{coin_id}_market_data.csv"
df.to_csv(nombre_archivo, index=False)

print(f"✅ Datos guardados en: {nombre_archivo}")

