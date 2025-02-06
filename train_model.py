import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 📌 Datos de entrenamiento (Inversión en Publicidad y Ventas)
df = pd.DataFrame({
    "Inversion": [500, 700, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Ventas": [50, 65, 78, 95, 105, 130, 160, 180, 195, 220]
})

# 📌 Separar variables de entrada (X) y salida (y)
X = df[['Inversion']].values
y = df['Ventas'].values

# 📌 Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Entrenar modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 📌 Guardar el modelo entrenado
joblib.dump(modelo, "modelo_regresion.pkl")

print("✅ Modelo guardado exitosamente en 'modelo_regresion.pkl'")
