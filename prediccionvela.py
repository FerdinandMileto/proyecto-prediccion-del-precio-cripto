import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ====================== CONFIG ======================
PAIR = "BTCUSDT"
INTERVAL = "1h"
LOOKBACK = 100          # Velas usadas para entrenar
FORECAST_STEPS = 1      # Predecir 1 vela adelante
RETRAIN_EVERY = 100     # Reentrenar cada N velas nuevas
UPDATE_DELAY = 30       # Segundos entre predicciones
CSV_FILE = "btc_predictions.csv"
BUFFER_SIZE =600       # Velas recientes para reentrenamiento
EPOCHS = 10
BATCH_SIZE = 16
# ====================== CONFIGURACIÓN AUTOMÁTICA DE REENTRENAMIENTO ======================
INTERVAL_MAP = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440
}

# Convierte el intervalo a minutos (por defecto 1)
interval_minutes = INTERVAL_MAP.get(INTERVAL, 1)

# Reentrenar cada cierto número de horas según el timeframe
# (por ejemplo cada 2h si es M1, cada 6h si es M5, etc.)
if interval_minutes <= 1:
    RETRAIN_EVERY = int((2 * 60) / interval_minutes)   # cada 2 horas
elif interval_minutes <= 5:
    RETRAIN_EVERY = int((6 * 60) / interval_minutes)   # cada 6 horas
elif interval_minutes <= 15:
    RETRAIN_EVERY = int((12 * 60) / interval_minutes)  # cada 12 horas
else:
    RETRAIN_EVERY = int((24 * 60) / interval_minutes)  # cada 24 horas

print(f"[⚙️] Intervalo {INTERVAL} detectado → Reentrenamiento cada {RETRAIN_EVERY} velas "
      f"({RETRAIN_EVERY * interval_minutes / 60:.1f} horas)")

# ====================== FUNCIONES ======================
def fetch_klines(symbol, interval, limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df['close'] = df['close'].astype(float)
    return df[['close']]

def create_sequences(data, lookback=LOOKBACK):
    X, y_price, y_dir = [], [], []
    for i in range(len(data) - lookback - FORECAST_STEPS + 1):
        seq_x = data[i:i+lookback]
        target_price = data[i+lookback+FORECAST_STEPS-1]
        target_dir = 1 if target_price > data[i+lookback-1] else 0
        X.append(seq_x)
        y_price.append(target_price)
        y_dir.append(target_dir)
    return np.array(X), np.array(y_price), np.array(y_dir)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ====================== INICIALIZACIÓN ======================
historical = fetch_klines(PAIR, INTERVAL, limit=BUFFER_SIZE)
prices = historical['close'].values.reshape(-1,1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

X, y_price, y_dir = create_sequences(prices_scaled)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = build_model((LOOKBACK,1))
model.fit(X, y_price, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=2)])

new_klines_count = 0
total_predictions = 0
correct_dir_predictions = 0

# ====================== LOOP PRINCIPAL ======================
while True:
    try:
        df_new = fetch_klines(PAIR, INTERVAL, limit=BUFFER_SIZE)
        df_closed = df_new.iloc[:-1]  # eliminar la vela en curso
        close_prices = df_closed['close'].values.reshape(-1,1)

        close_scaled = scaler.transform(close_prices)

        seq_input = close_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        pred_scaled = model.predict(seq_input, verbose=0)[0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        last_price = close_prices[-1][0]
        pred_dir = 1 if pred_price > last_price else 0
        pred_dir_symbol = "↑" if pred_dir==1 else "↓"
        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_price = abs(pred_price - last_price)
        total_predictions += 1
        actual_dir = 1 if close_prices[-1] < close_prices[-FORECAST_STEPS-1] else 0
        if pred_dir == actual_dir:
            correct_dir_predictions += 1
        acc_dir = correct_dir_predictions / total_predictions * 100

# Mostrar en consola
        print(f"[{ahora}] Precio actual: {last_price:.6f} | "
            f"Predicción próxima vela: {pred_price:.6f} {pred_dir_symbol} | "
            f"Error={error_price:.6f} | DirAcc={acc_dir:.2f}%")

        # ====================== Métricas ======================
        total_predictions += 1
        actual_dir = 1 if close_prices[-1] < close_prices[-FORECAST_STEPS-1] else 0
        if pred_dir == actual_dir:
            correct_dir_predictions += 1
        acc_dir = correct_dir_predictions / total_predictions * 100
        error_price = abs(pred_price - last_price)

        # ====================== Logging ======================
        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if total_predictions % 10 == 0:
            print(f"[{ahora}] Precio actual: {last_price:.6f} | Predicción próxima vela: {pred_price:.6f} {pred_dir_symbol} | Error={error_price:.6f} | DirAcc={acc_dir:.2f}%")

        # Guardar resultados en CSV
        df_out = pd.DataFrame({
            'timestamp':[ahora],
            'close_actual':[last_price],
            'pred_close':[pred_price],
            'pred_dir':[pred_dir_symbol],
            'error_price':[error_price],
            'dir_acc':[acc_dir]
        })
        df_out.to_csv(CSV_FILE, mode='a', header=not pd.io.common.file_exists(CSV_FILE), index=False)

        new_klines_count += 1

        # ====================== Reentrenamiento automático ======================
        if new_klines_count >= RETRAIN_EVERY:
            print(f"[{ahora}] 🔄 Reentrenando modelo con últimas {BUFFER_SIZE} velas...")
            scaler = MinMaxScaler()
            close_scaled = scaler.fit_transform(close_prices)
            X, y_price, y_dir = create_sequences(close_scaled)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            model.fit(X, y_price, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=2)])
            new_klines_count = 0
            print(f"[{ahora}] ✅ Modelo actualizado.")

        time.sleep(UPDATE_DELAY)

    except KeyboardInterrupt:
        print("Bot detenido manualmente.")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(10)
