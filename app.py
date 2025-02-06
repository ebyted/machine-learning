from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load("modelo_regresion.pkl")

# Datos históricos de inversión y ventas
datos_historicos = pd.DataFrame({
    "Inversion": [500, 700, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Ventas": [50, 65, 78, 95, 105, 130, 160, 180, 195, 220]
})

@app.route("/")
def home():
    return render_template("index.html", datos=datos_historicos.to_dict(orient="records"))

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.get_json()
        inversion = float(data["inversion"])

        # Hacer la predicción
        prediccion = modelo.predict(np.array([[inversion]]))[0]

        return jsonify({"prediccion": round(prediccion, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
