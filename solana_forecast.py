# =============================
# Script completo: Predicción de Solana
# =============================

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1️⃣ Cargar datos
csv_file = 'solana_last365.csv'  # Asegúrate de que el CSV esté en la misma carpeta
df = pd.read_csv(csv_file)
df['ds'] = pd.to_datetime(df['ds'])
df = df[['ds','y']]  # Confirmar columnas

print("✅ Datos cargados:")
print(df.head())

# 2️⃣ Crear y entrenar modelo Prophet
model = Prophet(daily_seasonality=True)
model.fit(df)
print("\n✅ Modelo entrenado con los últimos 365 días de datos")

# 3️⃣ Crear DataFrame futuro y predecir
future_days = 30
future = model.make_future_dataframe(periods=future_days)
forecast = model.predict(future)

print(f"\n✅ Predicción de los próximos {future_days} días:")
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())

# 4️⃣ Graficar resultados
plt.figure(figsize=(12,6))
plt.plot(df['ds'], df['y'], label='Precio real', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Predicción', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
plt.title('Predicción de precio de Solana (USD)')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.show()

# 5️⃣ Guardar predicciones en CSV
forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv('solana_forecast.csv', index=False)
print("\n✅ Predicciones guardadas en solana_forecast.csv")
