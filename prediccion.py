import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load("modelo_regresion.pkl")

# Nueva inversión en publicidad
nueva_inversion = np.array([[3000]])  # Cambia este valor para probar otros casos

# Hacer predicción
prediccion = modelo.predict(nueva_inversion)

print(f"Para una inversión de ${nueva_inversion[0][0]}, se esperan {prediccion[0]:.2f} ventas.")
