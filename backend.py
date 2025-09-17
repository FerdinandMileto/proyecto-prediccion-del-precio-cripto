# backend.py
# Guardar order book BTC/MXN de Bitso usando REST API (corregido)

import requests
import pandas as pd
from datetime import datetime, timezone
import time
import os

CSV_FILE = "order_book.csv"
SYMBOL = "btc_mxn"
INTERVAL = 5  # segundos entre requests

def fetch_order_book():
    url = f"https://api.bitso.com/v3/order_book/?book={SYMBOL}&aggregate=true"
    response = requests.get(url)
    data = response.json()
    
    if data["success"]:
        bids = data["payload"]["bids"]
        asks = data["payload"]["asks"]
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Cada bid/ask es un diccionario {"price": "...", "amount": "..."}
        df = pd.DataFrame({
            "side": ["bid"]*len(bids) + ["ask"]*len(asks),
            "price": [float(x["price"]) for x in bids] + [float(x["price"]) for x in asks],
            "size": [float(x["amount"]) for x in bids] + [float(x["amount"]) for x in asks],
            "time": [timestamp]*(len(bids)+len(asks))
        })
        return df
    else:
        print("Error al obtener datos:", data)
        return None

if __name__ == "__main__":
    while True:
        df = fetch_order_book()
        if df is not None:
            # Guardar en CSV (append)
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode="a", index=False, header=header)
            print(f"{len(df)} filas guardadas en {CSV_FILE}")
        time.sleep(INTERVAL)
