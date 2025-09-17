# train_model.py
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Importamos lo necesario de scikit-learn
from sklearn.model_selection import train_test_split      # para dividir el dataset en train/test
from sklearn.tree import DecisionTreeRegressor, plot_tree # el modelo de árbol y función para graficarlo
from sklearn.metrics import mean_absolute_error, r2_score # métricas de evaluación


def cargar_datos(path_csv: str) -> pd.DataFrame:
    """Carga el dataset limpio desde CSV."""
    return pd.read_csv(path_csv)


def entrenar_modelo(df: pd.DataFrame):
    """
    Entrena un Árbol de Regresión para predecir precios de casas.
    Devuelve el modelo entrenado y los conjuntos de datos para evaluación.
    """

    # ----------------- 1. Variables -----------------
    # X = variables predictoras (todas menos precio)
    X = df.drop("precio", axis=1)
    # y = variable objetivo (precio de la casa)
    y = df["precio"]

    # ----------------- 2. División -----------------
    # train_test_split divide en entrenamiento y prueba
    # test_size=0.2 → 20% de datos para test, 80% para train
    # random_state=42 → asegura reproducibilidad (misma división siempre)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------- 3. Modelo -----------------
    # DecisionTreeRegressor → Árbol de regresión
    # max_depth=4 → profundidad máxima del árbol (para evitar sobreajuste)
    model = DecisionTreeRegressor(max_depth=4, random_state=42)

    # Entrenamos el árbol con los datos de entrenamiento
    model.fit(X_train, y_train)

    # ----------------- 4. Evaluación -----------------
    # Predecimos en el conjunto de prueba
    y_pred = model.predict(X_test)

    # mean_absolute_error → diferencia absoluta promedio
    mae = mean_absolute_error(y_test, y_pred)

    # r2_score → coeficiente de determinación (0 a 1)
    r2 = r2_score(y_test, y_pred)

    print("📊 Evaluación del modelo:")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - R² : {r2:.4f}")

    return model, X_train, y_train


def guardar_modelo(model, path_pkl: str):
    """Guarda el modelo entrenado en un archivo .pkl usando joblib."""
    os.makedirs(os.path.dirname(path_pkl), exist_ok=True)
    joblib.dump(model, path_pkl)
    print(f"✅ Modelo guardado en: {path_pkl}")


def graficar_arbol(model, X_train):
    """
    Genera y muestra el gráfico del árbol entrenado.
    plot_tree → función de scikit-learn que dibuja la estructura.
    """
    plt.figure(figsize=(16, 10))  # tamaño del gráfico
    plot_tree(
        model,
        feature_names=X_train.columns,  # nombres de variables
        filled=True,   # colorear nodos
        rounded=True,  # bordes redondeados
        fontsize=10
    )
    plt.title("Árbol de Regresión - Predicción de precios de casas")
    plt.show()


if __name__ == "__main__":
    # Rutas
    input_path = os.path.join("data", "casas_limpias.csv")
    output_model = os.path.join("modelos", "modelo_regresion_casas.pkl")

    # Cargar
    print("📥 Cargando datos limpios...")
    df = cargar_datos(input_path)

    # Entrenar
    print("🌳 Entrenando modelo...")
    model, X_train, y_train = entrenar_modelo(df)

    # Guardar
    guardar_modelo(model, output_model)

    # Graficar
    print("📈 Mostrando árbol de regresión...")
    graficar_arbol(model, X_train)
