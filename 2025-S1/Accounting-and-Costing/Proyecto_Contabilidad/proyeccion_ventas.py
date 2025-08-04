import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

# === CONFIGURACIÓN ===
excel_input = 'Reporte de Ventas - Historico - Empresa FCI_Estudiantes.xlsx'
excel_output = 'proyeccion_resultado.xlsx'
html_output = 'grafico_proyeccion.html'

# === CARGA DE DATOS ===
xls = pd.ExcelFile(excel_input)
df = xls.parse('Resumen Historico')

# === VENTAS REALES ===
anios = df.iloc[3, 2:14].astype(int).values.reshape(-1, 1)
mayoristas = df.iloc[4, 2:14].astype(float).values
minoristas = df.iloc[5, 2:14].astype(float).values

# === ENTRENAMIENTO DEL MODELO POLINOMIAL GRADO 2 ===
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(anios)
model_may = LinearRegression().fit(x_poly, mayoristas)
model_min = LinearRegression().fit(x_poly, minoristas)

# === AÑOS FUTUROS Y PREDICCIONES ===
anios_futuros = np.arange(2025, 2031).reshape(-1, 1)
x_futuro = poly.transform(anios_futuros)
pro_may = model_may.predict(x_futuro)
pro_min = model_min.predict(x_futuro)

# === COMBINAR HISTÓRICO Y PROYECCIÓN ===
todos_anios = np.vstack((anios, anios_futuros)).flatten()
todos_may = np.concatenate((mayoristas, pro_may))
todos_min = np.concatenate((minoristas, pro_min))

# === GENERAR GRÁFICO ===
plt.figure(figsize=(10, 6))
plt.plot(todos_anios, todos_may, marker='o', label='Mayoristas')
plt.plot(todos_anios, todos_min, marker='s', label='Minoristas')
plt.axvline(x=2025, color='gray', linestyle='--', label='Proyección Futura')
plt.title('Proyección de Ventas Mayoristas y Minoristas (2014–2030)')
plt.xlabel('Año')
plt.ylabel('Ventas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_temp.png')

# === HTML OUTPUT ===
with open(html_output, 'w', encoding='utf-8') as f:
    f.write("""
    <html>
        <head><title>Proyección de Ventas</title></head>
        <body>
            <h2>Gráfica de Proyección de Ventas Mayoristas y Minoristas</h2>
            <img src='grafico_temp.png' width='800'>
        </body>
    </html>
    """)

# === GUARDAR RESULTADOS EN EXCEL ===
df_resultados = pd.DataFrame({
    'Año': todos_anios,
    'Ventas Mayoristas': todos_may,
    'Ventas Minoristas': todos_min
})
df_resultados.to_excel(excel_output, index=False)

print("✅ Proceso completado. Revisa 'grafico_proyeccion.html' y 'proyeccion_resultado.xlsx'")
