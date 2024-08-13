# Databricks notebook source
import requests
import zipfile
import io
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from datetime import datetime, timedelta

# Inicializar SparkSession si no está ya inicializada
spark = SparkSession.builder.appName("GDELT Daily Loader").getOrCreate()

# URL base del archivo
url_base = "http://data.gdeltproject.org/events/"

# Obtener la fecha de ayer
yesterday_date = (datetime.now() - timedelta(1)).strftime("%Y%m%d")

# Construir el URL completo usando la fecha de ayer
url = f"{url_base}{yesterday_date}.export.CSV.zip"

# Extraer la fecha del URL para usarla en la columna extraction_date
extraction_date = re.search(r'(\d{8})', url).group(1)

# Descargar el archivo ZIP
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    try:
        # Intentar abrir el archivo ZIP
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Verificar el encabezado de contenido y los primeros bytes del archivo
        print("Content-Type:", response.headers.get('Content-Type'))
        print("Primeros 100 bytes del contenido descargado:", response.content[:100])
        
        # Extraer el archivo CSV del ZIP
        csv_filename = zip_file.namelist()[0]  # Asumiendo que hay un solo archivo en el ZIP
        csv_file = zip_file.open(csv_filename)

        # Leer el CSV en un DataFrame de Spark
        gdelt_df = spark.read.format("csv").option("header", "false").option("inferSchema", "true").load(csv_file)

        # Añadir la columna extraction_date
        gdelt_df = gdelt_df.withColumn("extraction_date", lit(extraction_date))

        # Mostrar el esquema para asegurarse de que los datos se cargaron correctamente
        gdelt_df.printSchema()

        # Mostrar los primeros registros para verificar
        gdelt_df.show(5)

        # Insertar los datos en la tabla creada en Unity Catalog
        gdelt_df.write.format("delta").mode("append").saveAsTable("catalog_name.schema_name.table_name")

    except zipfile.BadZipFile:
        print("El archivo descargado no es un archivo ZIP válido.")
else:
    print(f"Error al descargar el archivo: {response.status_code}")

