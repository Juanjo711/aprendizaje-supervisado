# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time # Para medir el tiempo de entrenamiento

# ==============================================================================
# PASO 1: Cargar Datos DESDE EL ARCHIVO CSV
# ==============================================================================

nombre_archivo_csv = 'dataset_simulado_transporte_medellin.csv'

print(f"Cargando datos desde el archivo: '{nombre_archivo_csv}'...")

try:
    df = pd.read_csv(
        nombre_archivo_csv,
        parse_dates=['Timestamp']
    )
    print("¡Datos cargados con éxito!")
    print(f"El dataset contiene {len(df)} filas y {len(df.columns)} columnas.")

    print("\n--- Primeras 5 filas del Dataset Cargado ---")
    print(df.head())
    print("\n--- Información del DataFrame Cargado ---")
    df.info()

except FileNotFoundError:
    print(f"Error: El archivo '{nombre_archivo_csv}' no se encontró.")
    print("Asegúrate de que el archivo CSV esté en el mismo directorio que este script,")
    print("o proporciona la ruta completa al archivo.")
    exit()
except Exception as e:
    print(f"Error al cargar el archivo CSV: {e}")
    exit()

# ==============================================================================
# PASO 2: Preparar Datos (Seleccionar Features y Target)
# ==============================================================================

features = ['ID_Estacion', 'Dia_Semana', 'Es_Festivo', 'Es_Domingo', 'Hora_del_Dia',
            'Es_Hora_Pico', 'Evento_Cercano', 'Temperatura_C', 'Precipitacion_mm']
target = 'Flujo_Pasajeros'

missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    print(f"Error: Las siguientes columnas requeridas no se encontraron en el CSV: {missing_cols}")
    exit()

X = df[features]
y = df[target]

# ==============================================================================
# PASO 3: Dividir Datos en Entrenamiento y Prueba
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

print(f"\nTamaño del set de Entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del set de Prueba: {X_test.shape[0]} muestras")

# ==============================================================================
# PASO 4: Preprocesamiento (Escalado de Features)
# ==============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# PASO 5: Seleccionar y Entrenar el Modelo
# ==============================================================================
model = RandomForestRegressor(n_estimators=150,
                              random_state=42,
                              n_jobs=-1,
                              max_depth=20,
                              min_samples_split=10,
                              min_samples_leaf=5
                              )

print("\nEntrenando el modelo Random Forest Regressor...")
start_time = time.time()
model.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

# ==============================================================================
# PASO 6: Realizar Predicciones
# ==============================================================================
print("Realizando predicciones en el set de prueba...")
y_pred = model.predict(X_test_scaled)

# ==============================================================================
# PASO 7: Evaluar el Modelo
# ==============================================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluación del Modelo ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} pasajeros")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} pasajeros")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# ==============================================================================
# PASO 8: (Opcional) Importancia de Features
# ==============================================================================
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Importancia de Features ---")
print(feature_importance_df)