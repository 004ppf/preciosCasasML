# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

# Inicializar FastAPI
app = FastAPI(title="API Árbol de Regresión - Precios de Casas")

# 📂 Ruta del modelo entrenado (precios colombianos)
MODEL_PATH = os.path.join("modelos", "modelo_regresion_casas_colombia.pkl")
model = joblib.load(MODEL_PATH)
print(f"✅ Modelo colombiano cargado desde {MODEL_PATH}")

# 📌 Definición del esquema de entrada (JSON esperado)
class Features(BaseModel):
    superficie: float
    habitaciones: int
    antiguedad: float
    ubicacion_rural: int
    ubicacion_urbano: int

# Endpoint de bienvenida
@app.get("/")
def home():
    return {"mensaje": "API Árbol de Regresión - Precios de Casas Colombia está activa 🇨🇴"}

# Endpoint de predicción con JSON en body
@app.post("/predecir")
def predecir(features: Features):
    # Convertir JSON a DataFrame
    df = pd.DataFrame([features.dict()])
    
    # Asegurar que las columnas estén en el orden correcto
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Hacer predicción
    prediccion = model.predict(df)[0]
    return {
        "precio_estimado": round(float(prediccion), 2),
        "unidad": "millones de pesos colombianos (COP)",
        "precio_formateado": f"${round(float(prediccion), 2):,.2f} millones COP"
    }
