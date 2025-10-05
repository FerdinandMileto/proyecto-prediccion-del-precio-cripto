import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ======================
# Función para descargar OHLCV desde Binance
# ======================
def get_binance_ohlcv(symbol="BTCUSDT", interval="1h", days=1095):
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    ms_interval = 60 * 60 * 1000  # 1h
    if interval == "1d":
        ms_interval = 24 * 60 * 60 * 1000
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    all_data = []
    while start_time < end_time:
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "limit": limit}
        resp = requests.get(base_url, params=params)
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        last_open_time = data[-1][0]
        start_time = last_open_time + ms_interval
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base",
        "taker_buy_quote","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["open_time","open","high","low","close","volume"]]

# ======================
# Funciones auxiliares
# ======================
def porcentaje(df,columna_objetivo,n_de_intervalos,columna_nueva):
    for i in range(n_de_intervalos,len(df[columna_objetivo])):
        cambio=(df[columna_objetivo][i-n_de_intervalos]-df[columna_objetivo][i])/df[columna_objetivo][i-n_de_intervalos]
        df.loc[i,columna_nueva]=cambio*100

def calcular_RSI(df, column="close", period=14):
    delta = df[column].diff()
    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)
    media_gan = ganancias.ewm(span=period, adjust=False).mean()
    media_per = perdidas.ewm(span=period, adjust=False).mean()
    RS = media_gan / media_per
    RSI = 100 - (100 / (1 + RS))
    df["RSI"] = RSI
    return df

def agregar_emas(df, column="close", spans=[9,21,50,200]):
    for span in spans:
        df[f"EMA_{span}"] = df[column].ewm(span=span, adjust=False).mean()
    return df

def volumen_por_precio(df, price_col="close", volume_col="volume", bins=50):
    precios = df[price_col]
    volumenes = df[volume_col]
    hist, edges = np.histogram(precios, bins=bins, weights=volumenes)
    price_levels = (edges[:-1] + edges[1:]) / 2
    vp = pd.DataFrame({"price_level": price_levels, "volume": hist})
    vp = vp.sort_values(by="volume", ascending=False).reset_index(drop=True)
    return vp

# ======================
# Variables globales
# ======================
restricciones = ["open_time","pred_dia%"]
medir = "pred_dia%"
csv_results_file = "predicciones_backtesting.csv"

# ======================
# Bucle principal para predicción y reentrenamiento por cada vela nueva
# ======================
while True:
    try:
        df = get_binance_ohlcv("BTCUSDT", "1h", days=1095)

        # Features
        df_porcentual = df.copy()
        porcentaje(df_porcentual,"close",1,"1hora%")
        porcentaje(df_porcentual,"close",4,"4hora%")
        porcentaje(df_porcentual,"close",8,"8hora%")
        porcentaje(df_porcentual,"close",24,"dia%")
        porcentaje(df_porcentual,"close",48,"2dias%")
        porcentaje(df_porcentual,"close",24*7,"1semana%")
        porcentaje(df_porcentual,"close",24*7*2,"2semanas%")
        porcentaje(df_porcentual,"close",24*7*4,"1mes%")
        df_porcentual.dropna(inplace=True)
        df_porcentual.reset_index(drop=True,inplace=True)
        df_porcentual["pred_dia%"] = df_porcentual["dia%"].shift(-24)

        calcular_RSI(df_porcentual)
        agregar_emas(df_porcentual)
        resistencias = volumen_por_precio(df_porcentual)

        df_porcentual["resistencia"] = 0
        lista_redondeada_high = [round(x,0) for x in list(df_porcentual["high"])]
        lista_redondeada_low = [round(x,0) for x in list(df_porcentual["low"])]
        lista_redondeada_close = [round(x,0) for x in list(df_porcentual["close"])]
        for i in list(round(resistencias["price_level"],0)):
            if i in lista_redondeada_high:
                indice = lista_redondeada_high.index(i)
                df_porcentual.loc[indice,"resistencia"]=1
            elif i in lista_redondeada_low:
                indice = lista_redondeada_low.index(i)
                df_porcentual.loc[indice,"resistencia"]=1
            elif i in lista_redondeada_close:
                indice = lista_redondeada_close.index(i)
                df_porcentual.loc[indice,"resistencia"]=1

        df_porcentual.dropna(inplace=True)
        df_porcentual.reset_index(drop=True,inplace=True)

        # Entrenamiento y predicción
        relativ_error=[]
        aciertos_direccion_list=[]
        error_aciertos_list=[]
        y_pred_guardadas=[]
        y_real_guardadas=[]

        for i in range(81):
            entreno=df_porcentual[20000+i:25500+i]
            outsider=df_porcentual[25500+i+1:25500+i+2]
            outsider.reset_index(drop=True,inplace=True)
            entreno.reset_index(drop=True,inplace=True)

            X_entreno=entreno.drop(restricciones,axis=1)
            y=entreno[medir]

            X_train, X_test, y_train, y_test = train_test_split(X_entreno, y, test_size=0.001, random_state=42)
            model=RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train,y_train)

            X_outsider=outsider.drop(restricciones,axis=1)
            y_outsider=outsider[medir]
            predicciones_outsider=model.predict(X_outsider)

            y_pred_guardadas.append(predicciones_outsider[0])
            y_real_guardadas.append(y_outsider[0])

            error_relativo_porcentual=abs(y_outsider[0]-predicciones_outsider)/y_outsider[0]
            relativ_error.append(error_relativo_porcentual)

            if y_outsider[0]*predicciones_outsider>0:
                aciertos_direccion_list.append(predicciones_outsider)
                error_relativo_aciertos=abs(y_outsider[0]-predicciones_outsider)/y_outsider[0]
                error_aciertos_list.append(error_relativo_aciertos)

        error_total=abs(np.array(relativ_error).sum())/len(relativ_error)
        aciertos_direccion=len(aciertos_direccion_list)
        error_aciertos=abs(np.array(error_aciertos_list).sum())/len(error_aciertos_list)

        # Mostrar resultados en consola
        print("============================================")
        print(f"📊 Última vela analizada:")
        print(f"⏰ Fecha: {df_porcentual.iloc[-1]['open_time']}")
        print(f"🟢 Open: {df_porcentual.iloc[-1]['open']}")
        print(f"🔺 High: {df_porcentual.iloc[-1]['high']}")
        print(f"🔻 Low: {df_porcentual.iloc[-1]['low']}")
        print(f"🔴 Close: {df_porcentual.iloc[-1]['close']}")
        print(f"📦 Volumen: {df_porcentual.iloc[-1]['volume']}")
        print("--------------------------------------------")

        if len(y_pred_guardadas) > 0:
            ultima_pred = y_pred_guardadas[-1]
            direccion = "Alta ↑" if ultima_pred > 0 else "Baja ↓"
            print("📈 Predicción siguiente vela:")
            print(f"👉 Dirección: {direccion}")
            print(f"💹 Cambio porcentual estimado: {ultima_pred:.4f}")
        else:
            print("⚠️ No se generaron predicciones.")

        print("============================================")
        print("📊 Backtesting:")
        print(f"✅ Precisión dirección: {aciertos_direccion/len(relativ_error):.2%}")
        print(f"📉 Error medio total: {error_total:.6f}")
        print(f"📉 Error medio en aciertos: {error_aciertos:.6f}")
        print("============================================")

        # Guardar resultados para backtesting y reentrenamiento
        resultados_df = pd.DataFrame({
            "open_time": df_porcentual["open_time"].iloc[-1:],
            "open": df_porcentual["open"].iloc[-1:],
            "high": df_porcentual["high"].iloc[-1:],
            "low": df_porcentual["low"].iloc[-1:],
            "close": df_porcentual["close"].iloc[-1:],
            "volume": df_porcentual["volume"].iloc[-1:],
            "pred_dia%": [ultima_pred]
        })
        resultados_df.to_csv(csv_results_file, mode='a', header=not pd.io.common.file_exists(csv_results_file), index=False)

        # Esperar hasta la siguiente vela (aprox. 60 minutos)
        print("⏳ Esperando la siguiente vela para reentrenar...")
        time.sleep(60*60)

