import pandas as pd
import numpy as np
import requests
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ====================== CONFIG ======================
LOOKBACK = 100
FORECAST_STEPS = 1
BUFFER_SIZE = 600
EPOCHS = 10
BATCH_SIZE = 16
CSV_FILE = "crypto_predictions.csv"

# ====================== SELECCIÓN INTERACTIVA ======================
available_pairs = ["BTCUSDT","ETHUSDT","XRPUSDT","BNBUSDT","SOLUSDT","LTCUSDT","BASEUSDT","HYPEUSDT","BGBUSDT"]
available_intervals = ["1m","5m","15m","30m","1h","4h","1d"]

print("🔹 Pares disponibles para análisis:")
for i, p in enumerate(available_pairs):
    print(f"{i+1}. {p}")

while True:
    choice = input("Selecciona el par (número o símbolo, ej. BTCUSDT): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(available_pairs):
        PAIR = available_pairs[int(choice)-1]
        break
    elif choice.upper() in available_pairs:
        PAIR = choice.upper()
        break
    else:
        print("❌ Opción no válida. Intenta de nuevo.")

print("\n🔹 Temporalidades disponibles:")
for i, t in enumerate(available_intervals):
    print(f"{i+1}. {t}")

while True:
    choice = input("Selecciona la temporalidad (número o intervalo, ej. 1h): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(available_intervals):
        INTERVAL = available_intervals[int(choice)-1]
        break
    elif choice in available_intervals:
        INTERVAL = choice
        break
    else:
        print("❌ Opción no válida. Intenta de nuevo.")

print(f"\n✅ Has seleccionado: {PAIR} con temporalidad {INTERVAL}\n")

# ====================== FUNCIONES ======================
def fetch_klines(symbol, interval, limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df[['open','high','low','close','volume']]

def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(data) - lookback - FORECAST_STEPS + 1):
        seq_x = data[i:i+lookback]
        target = data[i+lookback+FORECAST_STEPS-1, 0]  # close
        X.append(seq_x)
        y.append(target)
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ====================== INDICADORES ======================
def EMA(series, period):
    ema = [series[0]]
    alpha = 2 / (period + 1)
    for price in series[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return np.array(ema)

def RSI(series, period=14):
    deltas = np.diff(series)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = [100 - 100 / (1 + rs)]
    for delta in deltas[period:]:
        u = max(delta, 0)
        d = -min(delta, 0)
        up = (up * (period - 1) + u) / period
        down = (down * (period - 1) + d) / period
        rs = up / down if down != 0 else 0
        rsi.append(100 - 100 / (1 + rs))
    rsi = [50] * (period) + rsi
    return np.array(rsi)

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def ATR(high, low, close, period=14):
    min_len = min(len(high), len(low), len(close))
    high, low, close = high[:min_len], low[:min_len], close[:min_len]
    tr = []
    for i in range(1, len(close)):
        tr.append(max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    tr = np.array(tr)
    if len(tr) < period:
        return np.full(len(close), np.mean(tr) if len(tr) > 0 else 0)
    atr_vals = np.convolve(tr, np.ones(period)/period, mode='valid')
    atr = np.concatenate((np.full(period, atr_vals[0]), atr_vals))
    if len(atr) < len(close):
        atr = np.append(atr, [atr[-1]]*(len(close)-len(atr)))
    elif len(atr) > len(close):
        atr = atr[:len(close)]
    return atr

def Bollinger_Bands(series, period=20, std_dev=2):
    sma = pd.Series(series).rolling(period).mean().to_numpy()
    std = pd.Series(series).rolling(period).std().to_numpy()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

# ====================== INICIALIZACIÓN ======================
print(f"🚀 Iniciando análisis LSTM para {PAIR} con temporalidad {INTERVAL}...")

historical = fetch_klines(PAIR, INTERVAL, limit=BUFFER_SIZE)
df_closed = historical.iloc[:-1].copy()

# Indicadores
df_closed['ema_short'] = EMA(df_closed['close'], 12)
df_closed['ema_long'] = EMA(df_closed['close'], 26)
df_closed['rsi'] = RSI(df_closed['close'])
df_closed['macd_line'], df_closed['signal'], _ = MACD(df_closed['close'])
df_closed['atr'] = ATR(df_closed['high'], df_closed['low'], df_closed['close'])
df_closed['bb_upper'], df_closed['bb_lower'] = Bollinger_Bands(df_closed['close'])

features = ['close','ema_short','ema_long','rsi','macd_line','signal','atr','bb_upper','bb_lower']
X_all = df_closed[features].fillna(method='bfill').values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)
scaler_close = MinMaxScaler()
scaler_close.fit(df_closed['close'].values.reshape(-1,1))

X_seq, y_seq = create_sequences(X_scaled)
model = build_model((LOOKBACK, X_seq.shape[2]))
model.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
          callbacks=[EarlyStopping(monitor='loss', patience=2)])

# ====================== BACKTESTING ======================
close_prices = df_closed['close'].values
pred_prices, correct_dirs, equity_curve = [], 0, []
equity = 1000

for i in range(LOOKBACK, len(X_scaled)-1):
    seq_input = X_scaled[i-LOOKBACK:i].reshape(1, LOOKBACK, X_scaled.shape[1])
    pred_scaled = model.predict(seq_input, verbose=0)[0][0]
    pred_price = scaler_close.inverse_transform([[pred_scaled]])[0][0]
    actual_price = close_prices[i]
    pred_dir = np.sign(pred_price - close_prices[i-1])
    actual_dir = np.sign(actual_price - close_prices[i-1])
    if pred_dir == actual_dir:
        correct_dirs += 1
    equity += pred_dir * (actual_price - close_prices[i-1])
    equity_curve.append(equity)
    pred_prices.append(pred_price)

total_preds = len(pred_prices)
acc = correct_dirs / total_preds * 100
avg_abs_error = np.mean(np.abs(np.array(pred_prices) - close_prices[LOOKBACK:LOOKBACK+total_preds]))
drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve)
sharpe = (np.mean(np.diff(equity_curve)) / (np.std(np.diff(equity_curve)) + 1e-8)) * np.sqrt(252*24)

# ====================== PREDICCIONES FUTURAS ======================
lookback_seq = X_scaled[-LOOKBACK:].copy()
future_steps = [1,50,100,200]
future_preds = []
price_buffer = list(df_closed['close'].values)

for step in range(1, max(future_steps)+1):
    seq_input = lookback_seq.reshape(1, LOOKBACK, X_scaled.shape[1])
    pred_scaled = model.predict(seq_input, verbose=0)[0][0]
    pred_price = scaler_close.inverse_transform([[pred_scaled]])[0][0]
    price_buffer.append(pred_price)

    # recalcular indicadores con la nueva predicción
    ema_short = EMA(price_buffer,12)[-1]
    ema_long = EMA(price_buffer,26)[-1]
    rsi_val = RSI(price_buffer)[-1]
    macd_line, signal_line, _ = MACD(price_buffer)
    atr_val = ATR(price_buffer, price_buffer, price_buffer)[-1]  # estable
    bb_upper, bb_lower = Bollinger_Bands(price_buffer)
    new_row = np.array([pred_price, ema_short, ema_long, rsi_val, macd_line[-1], signal_line[-1], atr_val, bb_upper[-1], bb_lower[-1]])
    new_row_scaled = scaler.transform(new_row.reshape(1,-1))[0]
    lookback_seq = np.vstack([lookback_seq[1:], new_row_scaled])

    if step in future_steps:
        future_preds.append(pred_price)

# ====================== RESULTADOS ======================
ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
last_price = df_closed['close'].iloc[-1]

print("\n🎯 BACKTESTING COMPLETO")
print("────────────────────────────────────────────")
print(f"📊 Total predicciones: {total_preds}")
print(f"✅ Correctas: {correct_dirs}")
print(f"❌ Incorrectas: {total_preds-correct_dirs}")
print(f"📈 Acierto: {acc:.2f}%")
print(f"📏 Error abs. medio: {avg_abs_error:.2f}")
print(f"💹 Equity final: {equity:.2f} USD")
print(f"📉 Máx. Drawdown: {drawdown:.2f} USD")
print(f"📊 Sharpe ratio: {sharpe:.2f}")
print("────────────────────────────────────────────\n")

print("🎯 PREDICCIONES VELAS FUTURAS")
print("────────────────────────────────────────────")
print(f"⏰ Hora: {ahora}")
print(f"💰 Precio actual: {last_price:.2f} USD")
print(f"🔮 Próxima vela: {future_preds[0]:.2f} USD")
print(f"🔮 Vela 50 después: {future_preds[1]:.2f} USD")
print(f"🔮 Vela 100 después: {future_preds[2]:.2f} USD")
print(f"🔮 Vela 200 después: {future_preds[3]:.2f} USD")
print("────────────────────────────────────────────\n")

df_out = pd.DataFrame({
    'timestamp':[ahora],
    'close_actual':[last_price],
    'pred_1':[future_preds[0]],
    'pred_50':[future_preds[1]],
    'pred_100':[future_preds[2]],
    'pred_200':[future_preds[3]],
    'acc':[acc],
    'avg_error':[avg_abs_error]
})
df_out.to_csv(CSV_FILE, mode='a', header=not pd.io.common.file_exists(CSV_FILE), index=False)
print("✅ Resultados guardados en", CSV_FILE)
