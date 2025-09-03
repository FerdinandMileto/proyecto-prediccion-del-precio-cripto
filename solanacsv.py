import requests
import pandas as pd
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor

# === Configuración ===
symbol = "SOLUSDT"
interval = "1d"
start_year = 2020
end_year = 2025
base_url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/"

# Lista de años y meses
years_months = [(year, month) for year in range(start_year, end_year+1) for month in range(1, 13)]

dataframes = []

# Función para descargar cada archivo
def download_month(year_month):
    year, month = year_month
    file_name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = base_url + file_name
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_file = z.namelist()[0]
            df = pd.read_csv(z.open(csv_file), header=None)
            print(f"✅ Descargado: {file_name}")
            return df
        else:
            print(f"⚠️ No existe: {file_name}")
            return None
    except Exception as e:
        print(f"❌ Error con {file_name}: {e}")
        return None

# Descargar con multithreading
with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(download_month, years_months)

# Filtrar resultados válidos
for df in results:
    if df is not None:
        dataframes.append(df)

# Combinar y procesar
if dataframes:
    full_df = pd.concat(dataframes, ignore_index=True)
    full_df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume", 
                       "Close Time", "Quote Asset Volume", "Number of Trades", 
                       "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]

    # Convertir fechas a datetime y manejar errores
    full_df["Open Time"] = pd.to_datetime(full_df["Open Time"], unit='ms', errors='coerce')
    full_df["Close Time"] = pd.to_datetime(full_df["Close Time"], unit='ms', errors='coerce')
    full_df = full_df.dropna(subset=["Open Time", "Close Time"])

    # Convertir columnas numéricas a float
    numeric_cols = ["Open", "High", "Low", "Close", "Volume", 
                    "Quote Asset Volume", "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume"]
    full_df[numeric_cols] = full_df[numeric_cols].astype(float)

    # Ordenar por fecha
    full_df = full_df.sort_values("Open Time").reset_index(drop=True)

    # Guardar CSV completo
    output_file = f"{symbol}_{interval}_full.csv"
    full_df.to_csv(output_file, index=False)
    print(f"\n🎉 Dataset completo guardado: {output_file}")

    # Resumen anual
    full_df['Year'] = full_df['Open Time'].dt.year
    resumen = full_df.groupby('Year').agg({
        'Close': ['min', 'max', 'mean'],
        'Volume': ['min', 'max', 'mean']
    })
    print("\n📊 Resumen anual:")
    print(resumen)

else:
    print("⚠️ No se descargó ningún archivo válido, CSV no generado.")


