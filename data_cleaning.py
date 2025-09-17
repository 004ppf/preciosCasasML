# data_cleaning.py
import pandas as pd
import os

def cargar_datos(path_csv: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(path_csv)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset según las reglas del ejercicio."""

    # --- Superficie ---
    df['superficie'] = df['superficie'].astype(str).str.replace('m2', '', regex=False)
    df['superficie'] = pd.to_numeric(df['superficie'], errors='coerce')
    df['superficie'] = df['superficie'].fillna(df['superficie'].median())

    # --- Habitaciones ---
    df['habitaciones'] = df['habitaciones'].astype(str).str.replace('tres', '3', regex=False)
    df['habitaciones'] = pd.to_numeric(df['habitaciones'], errors='coerce')
    df['habitaciones'] = df['habitaciones'].fillna(df['habitaciones'].median())
    med = df['habitaciones'].median()
    df['habitaciones'] = df['habitaciones'].apply(lambda x: med if x > 10 else x)

    # --- Antigüedad ---
    df['antiguedad'] = df['antiguedad'].astype(str).str.replace('nueva', '0', regex=False)
    df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')
    df['antiguedad'] = df['antiguedad'].fillna(df['antiguedad'].mean())
    df['antiguedad'] = df['antiguedad'].abs()

    # --- Precio ---
    df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
    df = df[(df['precio'] > 10000) & (df['precio'] < 1000000)]

    # --- Ubicación ---
    # Si ya existen columnas dummy de ubicación, unificarlas
    if "ubicacion_rural" in df.columns and "ubicacion_urbano" in df.columns:
        # Si es rural -> 1, si es urbano -> 0
        df["ubicacion_rural"] = df["ubicacion_rural"].astype(int)
        df = df.drop(columns=["ubicacion_urbano"], errors="ignore")

    elif "ubicacion" in df.columns:
        # Si todavía existe la columna original
        df['ubicacion'] = df['ubicacion'].astype(str).str.lower()
        df['ubicacion'] = df['ubicacion'].replace({
            'urbnaa': 'urbano',
            'urban': 'urbano',
            'true': 'urbano',
            'false': 'rural',
            'rurall': 'rural'
        })
        df['ubicacion'] = df['ubicacion'].fillna(df['ubicacion'].mode()[0])
        df = pd.get_dummies(df, columns=['ubicacion'], drop_first=True)
        if "ubicacion_rural" not in df.columns:
            df["ubicacion_rural"] = 0

    return df


def guardar_datos(df: pd.DataFrame, path_csv: str):
    """Guarda el DataFrame limpio en un archivo CSV."""
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df.to_csv(path_csv, index=False)
    print(f"✅ Archivo limpio guardado en: {path_csv}")


if __name__ == "__main__":
    # Rutas de entrada y salida
    input_path = os.path.join("data", "casas_sucias.csv")
    output_path = os.path.join("data", "casas_limpias.csv")

    # Cargar
    print("📥 Cargando datos...")
    df = cargar_datos(input_path)

    # Limpiar
    print("🧹 Limpiando datos...")
    df_limpio = limpiar_datos(df)

    # Guardar
    guardar_datos(df_limpio, output_path)
