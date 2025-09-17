# test_model.py
import pandas as pd
import joblib
import os

def cargar_modelo(path_pkl: str):
    """Carga un modelo previamente entrenado."""
    model = joblib.load(path_pkl)
    print(f"✅ Modelo '{path_pkl}' cargado exitosamente.")
    return model


def probar_modelo(model, df_prueba: pd.DataFrame):
    """Hace predicciones con el modelo ya entrenado."""
    print("📋 Columnas que espera el modelo:", list(model.feature_names_in_))
    print("📋 Columnas entregadas en prueba:", list(df_prueba.columns))

    # Reordenar columnas de df_prueba para que coincidan con el modelo
    df_prueba = df_prueba.reindex(columns=model.feature_names_in_, fill_value=0)

    predicciones = model.predict(df_prueba)
    return predicciones


if __name__ == "__main__":
    # Ruta al modelo
    model_path = os.path.join("modelos", "modelo_regresion_casas.pkl")
    model = cargar_modelo(model_path)

    # 🏠 Ejemplo de prueba (ajusta valores si quieres)
    df_prueba = pd.DataFrame([{
        "superficie": 100,
        "habitaciones": 4,
        "antiguedad": 5,
        "ubicacion_rural": 1,
        "ubicacion_urbano": 0
    }])

    # Predicciones
    resultados = probar_modelo(model, df_prueba)

    for i, pred in enumerate(resultados):
        print(f"🔮 Ejemplo {i+1} → Precio estimado: {pred:.2f}")
