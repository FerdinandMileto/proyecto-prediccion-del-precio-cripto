# fetch_ohlcv_bitget.py
import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
import math

# CONFIG
SYMBOL = "XRP/USDT:USDT"
# símbolo spot en ccxt/bitget (ver bitget.load_markets() si falla)
TIMEFRAME = "5m"
TOTAL_CANDLES = 2000    # objetivo
LIMIT_PER_CALL = 1000   # límite por llamada (ccxt/bitget suele aceptar <=1000)
RATE_LIMIT_SLEEP = 0.2  # fallback sleep (s) entre llamadas



exchange = ccxt.bitget()
markets = exchange.load_markets()



# Mostramos los primeros 30 pares para verificar cómo se llama XRP/USDT

def fetch_n_ohlcv(exchange, symbol, timeframe='5m', limit_total=5000, limit_per_call=1000):
    """
    Obtiene 'limit_total' velas usando múltiples llamadas fetch_ohlcv si es necesario.
    Devuelve un DataFrame con columnas: timestamp, open, high, low, close, volume, datetime
    """
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    # Empezamos pidiendo desde (ahora - total*tf) para asegurarnos de obtener el rango completo
    now_ms = exchange.milliseconds()
    since = int(now_ms - (limit_total * timeframe_ms))

    all_rows = []  # lista de listas [ts,o,h,l,c,v]
    attempts = 0
    max_attempts = 5

    while True:
        remaining = limit_total - len(all_rows)
        if remaining <= 0:
            break
        fetch_limit = min(limit_per_call, remaining)
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=fetch_limit)
        except Exception as e:
            attempts += 1
            print(f"[WARN] Error fetch_ohlcv (intento {attempts}): {e}")
            if attempts >= max_attempts:
                print("[ERROR] Muchos intentos fallidos, abortando.")
                break
            time.sleep(1 + attempts*0.5)
            continue

        attempts = 0

        if not candles:
            # si no llegan datos puede ser que since esté fuera de rango; avanzar un bloque pequeño hacia adelante
            since += fetch_limit * timeframe_ms
            print("[INFO] fetch_ohlcv devolvió 0 filas; avanzando 'since' y reintentando...")
            time.sleep(exchange.rateLimit / 1000 if hasattr(exchange, 'rateLimit') else RATE_LIMIT_SLEEP)
            continue

        # Filtrar duplicados respecto a all_rows
        if all_rows and candles[0][0] <= all_rows[-1][0]:
            last_ts = all_rows[-1][0]
            new = [c for c in candles if c[0] > last_ts]
        else:
            new = candles

        if not new:
            # ninguna vela nueva -> probablemente ya llegamos al límite de datos históricos disponibles
            print("[INFO] No hay velas nuevas en esta llamada. Terminando.")
            break

        all_rows.extend(new)
        # preparar next since: timestamp del último + timeframe_ms
        since = all_rows[-1][0] + timeframe_ms

        # dormir para respetar rate limit
        sleep_for = (exchange.rateLimit / 1000) if hasattr(exchange, 'rateLimit') else RATE_LIMIT_SLEEP
        time.sleep(max(0.01, sleep_for))

    # DataFrame final
    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
    # eliminar duplicados si los hubiera, ordenar y mantener solo las últimas limit_total
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp', ascending=True).reset_index(drop=True)
    if len(df) > limit_total:
        df = df.iloc[-limit_total:].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

if __name__ == "__main__":
    # inicializar cliente bitget (solo lectura pública)
    bitget = ccxt.bitget({
        'enableRateLimit': True,
        'timeout': 15000,
    })
    # comprobar que el símbolo existe
    try:
        markets = bitget.load_markets()
        if SYMBOL not in markets:
            print(f"[ERROR] Símbolo {SYMBOL} no encontrado en Bitget. Revisa bitget.load_markets()")
            print("Pares disponibles ejemplo (primeros 20):")
            print(list(markets.keys())[:20])
            raise SystemExit(1)
    except Exception as e:
        print(f"[ERROR] falló load_markets: {e}")
        raise

    print(f"[{datetime.now(timezone.utc)}] -> Descargando {TOTAL_CANDLES} velas {TIMEFRAME} para {SYMBOL} desde Bitget...")
    df = fetch_n_ohlcv(bitget, SYMBOL, timeframe=TIMEFRAME, limit_total=TOTAL_CANDLES, limit_per_call=LIMIT_PER_CALL)
    print(f"[{datetime.now(timezone.utc)}] -> Velas recibidas: {len(df)}")
    print(df.tail(5).to_string(index=False))

    # Guardar
    import re

# limpiar nombre: solo letras y números
safe_symbol = re.sub(r'[^A-Za-z0-9]', '', SYMBOL)
out_csv = f"{safe_symbol}_{TIMEFRAME}_{TOTAL_CANDLES}.csv"

df.to_csv(out_csv, index=False)
print(f"[{datetime.now(timezone.utc)}] -> Guardado en {out_csv}")
