# 🚇 Predicción de Flujo de Pasajeros en el Transporte Masivo de Medellín 🧠

Este proyecto desarrolla un modelo de **Aprendizaje Supervisado (Machine Learning)** utilizando Python para predecir el flujo horario de pasajeros en estaciones seleccionadas del Sistema Integrado de Transporte del Valle de Aburrá (SITVA) en Medellín, Colombia.

**Nota Importante:** Debido a la limitada disponibilidad pública de datos granulares sobre el flujo de pasajeros, este proyecto utiliza un **dataset sintético/simulado** generado programáticamente. El objetivo principal es demostrar el flujo de trabajo de un proyecto de Machine Learning aplicado a este dominio, desde la generación de datos hasta la evaluación del modelo.

## 📜 Descripción

El objetivo es predecir la variable `Flujo_Pasajeros` (número de pasajeros ingresando a una estación en una hora determinada) basándose en características como la hora del día, el día de la semana, si es festivo, la estación específica, y condiciones climáticas simuladas. Se utiliza un modelo de Regresión (Random Forest) implementado con las librerías `pandas` para manipulación de datos y `scikit-learn` para el modelado y evaluación.

## 📊 Dataset Utilizado

* **Fuente:** Datos Sintéticos / Simulados.
* **Generación:** Los datos se generan mediante el script `1_generar_datos.py`. Este script crea patrones realistas basados en factores conocidos que influyen en el flujo de pasajeros (horas pico, fines de semana, festivos, clima, etc.).
* **Archivo:** El dataset generado se guarda (por defecto) como `dataset_simulado_transporte_medellin.csv`.
* **Estructura Clave:**
    * `Timestamp`: Marca de tiempo horaria.
    * `ID_Estacion`: Identificador numérico de la estación.
    * `Dia_Semana`: Día de la semana (Lunes=0, Domingo=6).
    * `Es_Festivo`: Indicador binario (1 si es festivo, 0 si no).
    * `Hora_del_Dia`: Hora (0-23).
    * `Es_Hora_Pico`: Indicador binario.
    * `Temperatura_C`: Temperatura simulada.
    * `Precipitacion_mm`: Precipitación simulada.
    * `Flujo_Pasajeros`: **Variable objetivo** a predecir (número de pasajeros).

## ⚙️ Instalación y Configuración

Sigue estos pasos para configurar el proyecto localmente:

1.  **Clonar el Repositorio (si está en Git):**
    ```bash
    git clone https://github.com/Juanjo711/aprendizaje-supervisado
    cd aprendizaje-supervisado
    ```

2.  **Crear un Entorno Virtual (Recomendado):**
    ```bash
    # Crear el entorno
    python -m venv venv

    # Activar el entorno
    # En Windows (cmd/powershell):
    .\venv\Scripts\activate
    # En Linux/macOS (bash/zsh):
    source venv/bin/activate
    ```

3.  **Instalar Dependencias:**
    Asegúrate de tener un archivo `requirements.txt` con las librerías necesarias. Si no lo tienes, puedes crearlo (después de instalar manualmente las librerías en tu entorno virtual) con `pip freeze > requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Las dependencias clave son:
    * `pandas`
    * `numpy`
    * `scikit-learn`

## ▶️ Uso

El proyecto se ejecuta en dos pasos principales:

1.  **Generar el Dataset Simulado:**
    Este script crea o sobrescribe el archivo `dataset_simulado_transporte_medellin.csv`.
    ```bash
    python 1_generar_datos.py
    ```

2.  **Entrenar y Evaluar el Modelo:**
    Este script carga los datos desde el CSV, los prepara, entrena el modelo RandomForest, realiza predicciones en el conjunto de prueba y muestra las métricas de evaluación y la importancia de las características.
    ```bash
    python 2_entrenar_modelo.py
    ```
    La salida incluirá métricas como MAE, RMSE y R².

## 📁 Estructura de Archivos (Sugerida)