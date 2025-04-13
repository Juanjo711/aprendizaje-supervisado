import pandas as pd
import numpy as np
from datetime import timedelta
import time


def generar_dataset_transporte(start_date='2024-01-01', end_date='2024-06-30', estaciones=[1, 2, 3, 4]):
    """
    Genera un dataset sintético de flujo de pasajeros para estaciones del Metro de Medellín.
    (Misma función de la respuesta anterior)
    """
    print(f"Generando datos desde {start_date} hasta {end_date} para estaciones {estaciones}...")
    start_gen_time = time.time()
    date_rng = pd.date_range(start=start_date, end=end_date, freq='h', tz='America/Bogota')
    df = pd.DataFrame(date_rng, columns=['Timestamp'])
    df['key'] = 1
    stations_df = pd.DataFrame({'ID_Estacion': estaciones, 'key': 1})
    df = pd.merge(df, stations_df, on='key').drop('key', axis=1)
    df['Dia_Semana'] = df['Timestamp'].dt.dayofweek
    df['Hora_del_Dia'] = df['Timestamp'].dt.hour
    df['Mes'] = df['Timestamp'].dt.month
    df['Dia_del_Anio'] = df['Timestamp'].dt.dayofyear
    festivos_colombia_2024 = [
        '2024-01-01', '2024-01-08', '2024-03-25', '2024-03-28', '2024-03-29',
        '2024-05-01', '2024-05-13', '2024-06-03', '2024-06-10'
    ]
    festivos_dates = pd.to_datetime(festivos_colombia_2024).date
    df['Es_Festivo'] = df['Timestamp'].dt.date.isin(festivos_dates).astype(int)
    df['Es_Domingo'] = (df['Dia_Semana'] == 6).astype(int)
    df['Es_Hora_Pico'] = 0
    df.loc[ (df['Dia_Semana'] < 5) &
             (((df['Hora_del_Dia'] >= 6) & (df['Hora_del_Dia'] <= 8)) |
              ((df['Hora_del_Dia'] >= 16) & (df['Hora_del_Dia'] <= 18))),
             'Es_Hora_Pico'] = 1
    temp_base = 22
    temp_amplitude_daily = 4
    df['Temperatura_C'] = (temp_base -
                           temp_amplitude_daily * np.cos(2 * np.pi * df['Hora_del_Dia'] / 24) +
                           0.5 * np.sin(2 * np.pi * df['Dia_del_Anio'] / 365) +
                           np.random.normal(0, 0.5, size=len(df))
                          ).round(1)
    prob_lluvia = 0.02
    prob_lluvia += np.where(df['Mes'].isin([4, 5]), 0.05, 0)
    prob_lluvia += np.where((df['Hora_del_Dia'] >= 14) & (df['Hora_del_Dia'] <= 18), 0.03, 0)
    llueve = np.random.rand(len(df)) < prob_lluvia
    df['Precipitacion_mm'] = np.where(llueve, np.random.uniform(0.1, 8, size=len(df)), 0).round(1)
    df['Evento_Cercano'] = (np.random.rand(len(df)) < 0.015).astype(int)
    flujo_base = {1: 500, 2: 450, 3: 350, 4: 300}
    flujo = df['ID_Estacion'].map(flujo_base).astype(float)
    factor_hora = np.sin(np.pi * df['Hora_del_Dia'] / 24)**0.5
    factor_hora += np.where(df['Es_Hora_Pico'] == 1, 1.5 + np.random.normal(0, 0.2, size=len(df)), 0)
    factor_hora += np.where((df['Hora_del_Dia'] >= 10) & (df['Hora_del_Dia'] <= 14) & (df['Dia_Semana'] < 5), 0.4, 0)
    factor_hora += np.where((df['Hora_del_Dia'] >= 19) & (df['Hora_del_Dia'] <= 21), 0.3, 0)
    factor_hora = np.maximum(0.05, factor_hora)
    factor_dia = np.ones(len(df))
    factor_dia[df['Dia_Semana'] == 5] = 0.8
    factor_dia[df['Dia_Semana'] == 6] = 0.5
    factor_festivo = np.where(df['Es_Festivo'] == 1, 0.35, 1.0)
    factor_evento = np.where(df['Evento_Cercano'] == 1, 1.8 + np.random.uniform(0, 0.5, size=len(df)), 1.0)
    factor_lluvia = np.maximum(0.6, 1.0 - df['Precipitacion_mm'] * 0.03)
    flujo_final = (flujo * factor_hora * factor_dia * factor_festivo * factor_evento * factor_lluvia *
                   (1 + np.random.normal(0, 0.15, size=len(df))))
    df['Flujo_Pasajeros'] = np.maximum(0, flujo_final).astype(int)
    end_gen_time = time.time()
    print(f"Dataset generado con {len(df)} registros en {end_gen_time - start_gen_time:.2f} segundos.")
    columnas_finales = ['Timestamp', 'ID_Estacion', 'Dia_Semana', 'Es_Festivo', 'Es_Domingo',
                      'Hora_del_Dia', 'Es_Hora_Pico', 'Evento_Cercano',
                      'Temperatura_C', 'Precipitacion_mm', 'Flujo_Pasajeros']
    return df[columnas_finales]

df_transporte = generar_dataset_transporte()


nombre_archivo_csv = 'dataset_simulado_transporte_medellin.csv'

print(f"\nExportando el DataFrame a '{nombre_archivo_csv}'...")

try:
    df_transporte.to_csv(
        nombre_archivo_csv,
        index=False,
        sep=',',
        encoding='utf-8'
    )
    print(f"¡Exportación completada con éxito! El archivo '{nombre_archivo_csv}' ha sido creado.")
    print(f"Contiene {len(df_transporte)} filas y {len(df_transporte.columns)} columnas.")

except Exception as e:
    print(f"Error durante la exportación a CSV: {e}")