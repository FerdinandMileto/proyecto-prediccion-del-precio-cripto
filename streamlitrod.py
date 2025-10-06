import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================
# Función para cargar datos
# =========================
@st.cache_data
def cargar_resultados():
    # Suponiendo que ya ejecutaste tu código y tienes estos arrays
    y_real = y_real_guardadas  # reemplaza con tus listas
    y_pred = y_pred_guardadas
    df_result = pd.DataFrame({"Real": y_real, "Predicción": y_pred})
    return df_result

# =========================
# Streamlit Layout
# =========================
st.set_page_config(page_title="Predicción de Dirección de SOL/USDT", layout="wide")
st.title("📈 Dashboard de Predicción de Dirección")

# Cargar resultados
resultados = cargar_resultados()

# =========================
# Métricas resumen
# =========================
total = len(resultados)
aciertos = (resultados["Real"] == resultados["Predicción"]).sum()
precision = aciertos / total * 100

st.subheader("🔹 Métricas del Modelo")
st.metric("Total de predicciones", total)
st.metric("Aciertos de dirección", aciertos)
st.metric("Precisión de dirección (%)", f"{precision:.2f}%")

# =========================
# Selección de número de iteraciones a mostrar
# =========================
num_iter = st.slider("Número de iteraciones a mostrar", min_value=10, max_value=total, value=50)

# =========================
# Gráfico Real vs Predicción
# =========================
st.subheader("📊 Gráfico Real vs Predicción")
plt.figure(figsize=(12,5))
sns.lineplot(data=resultados.tail(num_iter), palette=["#00ffff","#ff33cc"])
plt.plot(resultados["Real"].tail(num_iter), label="Real", color="#00ffff", marker='o')
plt.plot(resultados["Predicción"].tail(num_iter), label="Predicción", color="#ff33cc", linestyle="--", marker='x')
plt.title("Predicciones vs Real")
plt.xlabel("Iteración")
plt.ylabel("Dirección (0=Venta,1=Compra)")
plt.legend()
plt.grid(alpha=0.3)
st.pyplot(plt.gcf())

# =========================
# Tabla de últimos resultados
# =========================
st.subheader("📋 Últimos resultados")
st.dataframe(resultados.tail(num_iter))

# =========================
# Descargar resultados
# =========================
csv = resultados.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar resultados como CSV",
    data=csv,
    file_name='resultados_predicciones.csv',
    mime='text/csv'
)
