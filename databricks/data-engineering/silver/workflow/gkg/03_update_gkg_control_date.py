# Databricks notebook source
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

# Inicia la sesión de Spark
spark = SparkSession.builder.appName("Actualizar Fecha de Control").getOrCreate()

# Calcula la fecha de ayer
yesterday_date = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

# Consulta para actualizar la tabla de control con la fecha de ayer
update_query = f"""
UPDATE BRONZE.TABLE_CONTROL
SET LAST_UPDATE_DATE = '{yesterday_date}', 
    LAST_MODIFIED = CURRENT_TIMESTAMP
WHERE TABLE_NAME = 'BRONZE-GDELT_GKG'
"""

# Ejecuta la actualización
spark.sql(update_query)

print(f"Tabla de control actualizada con la fecha de ayer: {yesterday_date}")

