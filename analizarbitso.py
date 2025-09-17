import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Cargar CSV
df = pd.read_csv('order_book.csv')  # tu archivo descargado

# 2. Convertir timestamp a datetime y filtrar valores nulos
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp', 'mid_price'])
df = df.sort_values('timestamp')

# 3. Agregar por hora
df_hourly = df.resample('H', on='timestamp').median().reset_index()

# 4. Graficar evolución del precio
plt.figure(figsize=(12,6))
plt.plot(df_hourly['timestamp'], df_hourly['mid_price'], label='Precio medio por hora')
plt.xlabel('Hora')
plt.ylabel('Precio (MXN)')
plt.title('Evolución del precio por hora')
plt.legend()
plt.show()

# 5. Preparar datos para Prophet
prophet_df = df_hourly.rename(columns={'timestamp':'ds', 'mid_price':'y'})

# Entrenar modelo
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(prophet_df)

# Hacer predicción para las próximas 24 horas
future = model.make_future_dataframe(periods=24, freq='H')
forecast = model.predict(future)

# 6. Graficar predicción
model.plot(forecast)
plt.title('Predicción horaria del precio')
plt.show()

# 7. Mostrar predicción de las próximas 24 horas
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24))
