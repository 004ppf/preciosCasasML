# API de Predicción de Precios de Casas 🏠

Una API desarrollada con FastAPI que utiliza un modelo de Árbol de Regresión para predecir precios de casas basándose en características como superficie, habitaciones, antigüedad y ubicación.

## 🚀 Características

- **Modelo de Machine Learning**: Árbol de Regresión entrenado con scikit-learn
- **API REST**: Desarrollada con FastAPI
- **Documentación automática**: Swagger UI integrado
- **Predicciones en tiempo real**: Endpoint para obtener estimaciones de precios

## 📊 Variables del Modelo

El modelo utiliza las siguientes características para predecir el precio:

- `superficie`: Superficie de la casa en m²
- `habitaciones`: Número de habitaciones
- `antiguedad`: Antigüedad de la casa en años
- `ubicacion_rural`: 1 si es ubicación rural, 0 si no
- `ubicacion_urbano`: 1 si es ubicación urbana, 0 si no

## 🛠️ Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/004ppf/preciosCasasML.git
cd preciosCasasML
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Ejecución

Para ejecutar la API:

```bash
uvicorn main:app --reload --port 8000
```

La API estará disponible en: `http://localhost:8000`

## 📖 Uso de la API

### Endpoint de Bienvenida
```bash
GET http://localhost:8000/
```

### Endpoint de Predicción
```bash
POST http://localhost:8000/predecir
Content-Type: application/json

{
    "superficie": 120.5,
    "habitaciones": 3,
    "antiguedad": 10.0,
    "ubicacion_rural": 0,
    "ubicacion_urbano": 1
}
```

**Respuesta:**
```json
{
    "precio_estimado": 139704.17
}
```

## 📚 Documentación

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧪 Entrenamiento del Modelo

Para entrenar un nuevo modelo:

```bash
python train_model.py
```

## 📁 Estructura del Proyecto

```
├── data/
│   ├── casas_limpias.csv      # Datos limpios para entrenamiento
│   └── casas_sucias.csv       # Datos originales
├── modelos/
│   └── modelo_regresion_casas.pkl  # Modelo entrenado
├── main.py                    # API FastAPI
├── train_model.py            # Script de entrenamiento
├── data_cleaning.py          # Limpieza de datos
├── test_model.py             # Pruebas del modelo
└── requirements.txt          # Dependencias
```

## 🔧 Dependencias

- FastAPI
- scikit-learn
- pandas
- numpy
- joblib
- matplotlib
- uvicorn

## 📈 Métricas del Modelo

El modelo de Árbol de Regresión utiliza:
- **Profundidad máxima**: 4 niveles
- **División de datos**: 80% entrenamiento, 20% prueba
- **Métricas evaluadas**: MAE (Error Absoluto Medio) y R²

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
