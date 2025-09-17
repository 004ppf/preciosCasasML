# uvicorn main:app --reload --host 0.0.0.0 --port 8000

# main.py
import pandas as pd
import joblib
import os
from fastapi import FastAPI

# Crear la app de FastAPI
app = FastAPI(title="API Árbol de Regresión - Precios de Casas")

# Ruta al modelo entrenado
MODEL_PATH = os.path.join("modelos", "modelo_regresion_casas.pkl")

# Cargar el modelo al iniciar
model = joblib.load(MODEL_PATH)
print(f"✅ Modelo cargado desde {MODEL_PATH}")

@app.get("/")
def home():
    return {"mensaje": "API Árbol de Regresión - Precios de Casas está activa 🚀"}

@app.post("/predecir")
def predecir(superficie: float, habitaciones: int, antiguedad: float, ubicacion_rural: int, ubicacion_urbano: int):
    """
    Endpoint para predecir el precio de una casa.
    Ejemplo de uso:
    /predecir?superficie=120&habitaciones=3&antiguedad=10&ubicacion_rural=0&ubicacion_urbano=1
    """
    df = pd.DataFrame([{
        "superficie": superficie,
        "habitaciones": habitaciones,
        "antiguedad": antiguedad,
        "ubicacion_rural": ubicacion_rural,
        "ubicacion_urbano": ubicacion_urbano
    }])

    # Reordenar columnas para que coincidan con el modelo
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediccion = model.predict(df)[0]
    return {"precio_estimado": round(float(prediccion), 2)}
