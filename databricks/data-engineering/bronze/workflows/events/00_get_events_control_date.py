# Databricks notebook source
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

# Inicia la sesión de Spark
spark = SparkSession.builder.appName("Control Fechas GDELT").getOrCreate()

# Consulta para obtener la última fecha procesada de la tabla de control
query = """
SELECT LAST_UPDATE_DATE 
FROM BRONZE.TABLE_CONTROL 
WHERE TABLE_NAME = 'gdelt_events'
"""

# Ejecuta la consulta
df = spark.sql(query)

# Verifica si se obtuvo algún resultado
if df.count() > 0:
    # Obtén la última fecha procesada (ya es un objeto datetime.date)
    last_processed_date = df.collect()[0]['LAST_UPDATE_DATE']
    
    # Suma un día para obtener la próxima fecha a procesar
    next_date_to_process = (last_processed_date + timedelta(1)).strftime('%Y-%m-%d')
    dbutils.jobs.taskValues.set("next_date_to_process", next_date_to_process)
    
    print(f"Última fecha procesada: {last_processed_date}")
    print(f"Próxima fecha a procesar: {next_date_to_process}")
else:
    print("No se encontró la última fecha procesada. Asegúrate de que la tabla de control está correctamente inicializada.")

