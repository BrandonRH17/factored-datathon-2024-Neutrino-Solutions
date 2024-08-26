# Databricks notebook source
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

# Inicia la sesión de Spark
spark = SparkSession.builder.appName("Control Fechas GDELT").getOrCreate()

# Consulta para obtener la última fecha procesada de la tabla de control
query = """
SELECT TABLE_NAME, LAST_UPDATE_DATE 
FROM BRONZE.TABLE_CONTROL 
""" 

# Ejecuta la consulta
df = spark.sql(query)

# Verifica si se obtuvo algún resultado
if df.count() > 0:
    # Obtén la última fecha procesada (ya es un objeto datetime.date)
    gdelt_gkg_last_modified = df.filter(df.TABLE_NAME == 'gold-gdelt-gkg').select('LAST_UPDATE_DATE').collect()[0][0]

    # Suma un día para obtener la próxima fecha a procesar
    next_date_to_process_gkg    = (gdelt_gkg_last_modified + timedelta(1)).strftime('%Y-%m-%d')
    dbutils.jobs.taskValues.set("next_date_to_process_gkg", next_date_to_process_gkg)

    print(f"Última fecha procesada tabla gkg:    {gdelt_gkg_last_modified}")
    print(f"Próxima fecha a procesar gkg:        {next_date_to_process_gkg}")
else:
    print("No se encontró la última fecha procesada. Asegúrate de que la tabla de control está correctamente inicializada.")
