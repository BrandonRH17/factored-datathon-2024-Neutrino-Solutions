# Databricks notebook source
import pandas as pd
import requests
import zipfile
import io

# URL del archivo ZIP
url = f"http://data.gdeltproject.org/gkg/20240813.gkg.csv.zip"

# Descargar el archivo ZIP
response = requests.get(url) 
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extraer el archivo CSV del ZIP
csv_file_name = zip_file.namelist()[0]  # Obtener el nombre del archivo CSV dentro del ZIP
csv_file = zip_file.open(csv_file_name)

# Cargar el archivo CSV con los nombres de columnas correctos
df = pd.read_csv(csv_file, sep='\t')

# Verificar si todas las filas tienen el mismo número de columnas
column_counts = df.apply(lambda row: len(row), axis=1).value_counts()

# Mostrar las primeras filas del DataFrame
print(column_counts)
df.head(50)


# COMMAND ----------

import pandas as pd
import requests
import zipfile
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from datetime import datetime

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("GDK Data Loader").getOrCreate()

# Especificar la fecha del archivo
date = "20230813"

# URL del archivo ZIP
url = f"http://data.gdeltproject.org/gkg/{date}.gkgcounts.csv.zip"

# Descargar el archivo ZIP
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extraer el archivo CSV del ZIP
csv_file_name = zip_file.namelist()[0]  # Obtener el nombre del archivo CSV dentro del ZIP
csv_file = zip_file.open(csv_file_name)

# Cargar el CSV en un DataFrame de pandas
df = pd.read_csv(csv_file, sep='\t')

# # Convertir la columna DATE de pandas a un formato de fecha
# df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d', errors='coerce').dt.date

# # Convertir las columnas numéricas a string
df['NUMARTS'] = df['NUMARTS'].astype(int)
df['DATE'] = df['DATE'].astype(int)
df['NUMBER'] = df['NUMBER'].astype(int)
df['GEO_TYPE'] = df['GEO_TYPE'].astype(int)
df['GEO_LAT'] = df['GEO_LAT'].astype(float)
df['GEO_LONG'] = df['GEO_LONG'].astype(float)

# # Convertir las columnas restantes a string si es necesario
# string_columns = ["COUNTS","THEMES","LOCATIONS","PERSONS","ORGANIZATIONS","TONE","CAMEOEVENTIDS","SOURCES","SOURCEURLS"]
# df[string_columns] = df[string_columns].astype(str)

# # Agregar la columna extraction_date con la fecha del archivo
df['extraction_date'] = datetime.strptime(date, '%Y%m%d').date()
df
# # Definir el esquema para la tabla de Spark
schema = StructType([
    StructField("DATE", IntegerType(), True),
    StructField("NUMARTS", IntegerType(), True),
    StructField("COUNTTYPE", StringType(), True),
    StructField("NUMBER", IntegerType(), True),
    StructField("OBJECTTYPE", StringType(), True),
    StructField("GEO_TYPE", IntegerType(), True),
    StructField("GEO_FULLNAME", StringType(), True),
    StructField("GEO_COUNTRYCODE", StringType(), True),
    StructField("GEO_ADM1CODE", StringType(), True),
    StructField("GEO_LAT", FloatType(), True),
    StructField("GEO_LONG", FloatType(), True),
    StructField("GEO_FEATUREID", StringType(), True),
    StructField("CAMEOEVENTIDS", StringType(), True),
    StructField("SOURCES", StringType(), True),
    StructField("SOURCEURLS", StringType(), True),
    StructField("extraction_date", DateType(), True)
])

# # Convertir el DataFrame de pandas a un DataFrame de Spark
spark_df = spark.createDataFrame(df, schema=schema)

# # Mostrar los primeros registros del DataFrame de Spark
pandas_df = spark_df.toPandas()
display(pandas_df.head(20))  # Mostrar las primeras 20 filas


# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * 
# MAGIC FROM bronze.gdelt_events
# MAGIC WHERE SOURCEURL != 'nan'
# MAGIC LIMIT 1000

# COMMAND ----------

import pandas as pd
import requests
import zipfile
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from delta.tables import DeltaTable
from datetime import datetime, timedelta

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("GKG Data Loader").getOrCreate()

# Especificar la ruta de la tabla Delta en la ubicación predeterminada de S3
delta_table_path = "s3://databricks-workspace-stack-e63e7-bucket/unity-catalog/2600119076103476/bronze/gkg/events"

# Verificar si la tabla ya existe en formato Delta
table_exists = DeltaTable.isDeltaTable(spark, delta_table_path)

# Si la tabla no existe, crearla en formato Delta
if not table_exists:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS bronze.gkg_events
        USING DELTA
        LOCATION '{delta_table_path}'
    """)

# Definir las fechas de inicio y fin
start_date = datetime.strptime("2023-08-13", "%Y-%m-%d")
end_date = datetime.strptime("2024-08-12", "%Y-%m-%d")

# Definir el esquema para la tabla de Spark
schema = StructType([
    StructField("DATE", IntegerType(), True),
    StructField("NUMARTS", IntegerType(), True),
    StructField("COUNTTYPE", StringType(), True),
    StructField("NUMBER", IntegerType(), True),
    StructField("OBJECTTYPE", StringType(), True),
    StructField("GEO_TYPE", IntegerType(), True),
    StructField("GEO_FULLNAME", StringType(), True),
    StructField("GEO_COUNTRYCODE", StringType(), True),
    StructField("GEO_ADM1CODE", StringType(), True),
    StructField("GEO_LAT", FloatType(), True),
    StructField("GEO_LONG", FloatType(), True),
    StructField("GEO_FEATUREID", StringType(), True),
    StructField("CAMEOEVENTIDS", StringType(), True),
    StructField("SOURCES", StringType(), True),
    StructField("SOURCEURLS", StringType(), True),
    StructField("extraction_date", DateType(), True)
])

# Procesar cada día en el rango de fechas
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y%m%d')
    url = f"http://data.gdeltproject.org/gkg/{date}.gkgcounts.csv.zip"
    
    try:
        # Descargar el archivo ZIP
        response = requests.get(url)
        response.raise_for_status()  # Verificar si la descarga fue exitosa
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        # Extraer el archivo CSV del ZIP
        csv_file_name = zip_file.namelist()[0]
        csv_file = zip_file.open(csv_file_name)

        # Cargar el CSV en un DataFrame de pandas
        df = pd.read_csv(csv_file, sep='\t')

        # Convertir las columnas numéricas
        df['NUMARTS'] = df['NUMARTS'].astype(int)
        df['DATE'] = df['DATE'].astype(int)
        df['NUMBER'] = df['NUMBER'].astype(int)
        df['GEO_TYPE'] = df['GEO_TYPE'].astype(int)
        df['GEO_LAT'] = df['GEO_LAT'].astype(float)
        df['GEO_LONG'] = df['GEO_LONG'].astype(float)

# # Convertir las columnas restantes a string si es necesario
# string_columns = ["COUNTS","THEMES","LOCATIONS","PERSONS","ORGANIZATIONS","TONE","CAMEOEVENTIDS","SOURCES","SOURCEURLS"]
# df[string_columns] = df[string_columns].astype(str)

# # Agregar la columna extraction_date con la fecha del archivo

        # Agregar la columna extraction_date con la fecha del archivo
        df['extraction_date'] = current_date.date()

        # Convertir el DataFrame de pandas a un DataFrame de Spark
        spark_df = spark.createDataFrame(df, schema=schema)

        # Verificar si la tabla ya existe en formato Delta y realizar el upsert
        # if table_exists:
        #     delta_table = DeltaTable.forPath(spark, delta_table_path)
        #     delta_table.alias("tgt").merge(
        #         source=spark_df.alias("src"),
        #         condition="tgt.GlobalEventID = src.GlobalEventID"
        #     ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        #     print(f"Datos actualizados para {date_str}.")
        # else:
            # Si la tabla no existe, crearla en formato Delta
        spark_df.write.format("delta").mode("overwrite").save(delta_table_path)
        print(f"Tabla creada y datos insertados para {date_str}.")
        table_exists = True

    except requests.exceptions.RequestException as e:
        print(f"Error al descargar o procesar los datos para la fecha {date_str}: {e}")
    except Exception as e:
        print(f"Error al procesar los datos para la fecha {date_str}: {e}")

    # Avanzar al siguiente día
    current_date += timedelta(days=1)

# Finalizar la sesión de Spark
spark.stop()



# COMMAND ----------


