import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Descargamos datos históricos de Ethereum
data = yf.download('ETH-USD', start='2020-01-01')

# 2. Preparamos la tabla para el modelo estadístico
df = data[['Close', 'Volume']].reset_index()
df.columns = ['ds', 'y', 'volume']

# 3. Creamos el modelo e incluimos el volumen de transacciones
modelo = Prophet(daily_seasonality=True)
modelo.add_regressor('volume')
modelo.fit(df)

# 4. Creamos las fechas futuras (los próximos 90 días para cubrir fin de año)
futuro = modelo.make_future_dataframe(periods=90)

# 5. Rellenamos el volumen futuro estimado usando el promedio reciente
volumen_promedio_reciente = df['volume'].tail(30).mean()
volumen_completo = df[['ds', 'volume']].merge(futuro, on='ds', how='right')
volumen_completo['volume'] = volumen_completo['volume'].fillna(volumen_promedio_reciente)
futuro['volume'] = volumen_completo['volume'].values

# 6. Realizamos la predicción matemática
prediccion = modelo.predict(futuro)

# 7. Diseñamos el gráfico personalizado en Modo Oscuro
plt.figure(figsize=(12, 6))
plt.gca().set_facecolor('#1e1e24')  
plt.gcf().patch.set_facecolor('#111116')  

# Sombra de incertidumbre (Mínimo y Máximo)
plt.fill_between(prediccion['ds'], prediccion['yhat_lower'], prediccion['yhat_upper'], 
                 color='#bc34fa', alpha=0.15, label='Rango de Incertidumbre')

# Línea de predicción promedio
plt.plot(prediccion['ds'], prediccion['yhat'], 
         color='#ff007f', linewidth=2.5, label='Precio Predicho (yhat)')

# Precios reales pasados
plt.scatter(df['ds'], df['y'], 
            color='#00f0ff', s=10, alpha=0.6, label='Precios Reales Pasados')

# Configuración de textos, etiquetas y cuadrícula
plt.title("🤖 Predicción Avanzada de Ethereum (ETH-USD) - Rumbo a Fin de Año 2026", 
          color='white', fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Fecha de Análisis", color='#a0a0a5', fontsize=11)
plt.ylabel("Precio en Dólares (USD)", color='#a0a0a5', fontsize=11)
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(True, color='#2d2d35', linestyle='--', alpha=0.5)
plt.legend(facecolor='#111116', edgecolor='#2d2d35', labelcolor='white', loc='upper left')

# 8. Guardamos el gráfico final como archivo de imagen
plt.savefig('grafico_cyberpunk_eth.png', dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
