import requests
import pandas as pd
import zipfile
import io

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/"

# Lista de archivos (ejemplo desde 2017-08 a 2025-08)
years_months = [(year, month) for year in range(2017, 2026) for month in range(1, 13)]
dataframes = []

for year, month in years_months:
    file_name = f"BTCUSDT-1d-{year}-{month:02d}.zip"
    url = BASE_URL + file_name
    
    try:
        r = requests.get(url)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_file = z.namelist()[0]
            df = pd.read_csv(z.open(csv_file), header=None)
            dataframes.append(df)
            print(f"Descargado: {file_name}")
        else:
            print(f"No existe: {file_name}")
    except Exception as e:
        print(f"Error con {file_name}: {e}")

# Combinar todo
if dataframes:
    full_df = pd.concat(dataframes, ignore_index=True)
    full_df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume", 
                       "Close Time", "Quote Asset Volume", "Number of Trades", 
                       "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]

    full_df.to_csv("BTCUSDT_1d_full.csv", index=False)
    print("Dataset completo guardado: BTCUSDT_1d_full.csv")
else:
    print("No se descargó ningún archivo.")
