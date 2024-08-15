# Databricks notebook source
import pandas as pd
import requests
import zipfile
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from delta.tables import DeltaTable
from datetime import datetime, timedelta

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("GDELT Data Loader").getOrCreate()

# Especificar la ruta de la tabla Delta en la ubicación predeterminada de S3
delta_table_path = ""

# Verificar si la tabla ya existe en formato Delta
table_exists = DeltaTable.isDeltaTable(spark, delta_table_path)

# Si la tabla no existe, crearla en formato Delta
if not table_exists:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS bronze.gdelt_events
        USING DELTA
        LOCATION '{delta_table_path}'
    """)

# Calcular la fecha del día anterior
current_date = datetime.now() - timedelta(days=1)
date_str = current_date.strftime('%Y%m%d')

# Nombres de las columnas según GDELT
column_names = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate", 
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", 
    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", 
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", 
    "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", 
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", 
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", 
    "Actor1Geo_Type", "Actor1Geo_Fullname", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", 
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", 
    "Actor2Geo_Fullname", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", 
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", 
    "ActionGeo_Fullname", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", 
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"
]

# Definir el esquema para la tabla de Spark
schema = StructType([
    StructField("GlobalEventID", LongType(), True),
    StructField("Day", IntegerType(), True),
    StructField("MonthYear", IntegerType(), True),
    StructField("Year", IntegerType(), True),
    StructField("FractionDate", FloatType(), True),
    StructField("Actor1Code", StringType(), True),
    StructField("Actor1Name", StringType(), True),
    StructField("Actor1CountryCode", StringType(), True),
    StructField("Actor1KnownGroupCode", StringType(), True),
    StructField("Actor1EthnicCode", StringType(), True),
    StructField("Actor1Religion1Code", StringType(), True),
    StructField("Actor1Religion2Code", StringType(), True),
    StructField("Actor1Type1Code", StringType(), True),
    StructField("Actor1Type2Code", StringType(), True),
    StructField("Actor1Type3Code", StringType(), True),
    StructField("Actor2Code", StringType(), True),
    StructField("Actor2Name", StringType(), True),
    StructField("Actor2CountryCode", StringType(), True),
    StructField("Actor2KnownGroupCode", StringType(), True),
    StructField("Actor2EthnicCode", StringType(), True),
    StructField("Actor2Religion1Code", StringType(), True),
    StructField("Actor2Religion2Code", StringType(), True),
    StructField("Actor2Type1Code", StringType(), True),
    StructField("Actor2Type2Code", StringType(), True),
    StructField("Actor2Type3Code", StringType(), True),
    StructField("IsRootEvent", IntegerType(), True),
    StructField("EventCode", IntegerType(), True),
    StructField("EventBaseCode", IntegerType(), True),
    StructField("EventRootCode", IntegerType(), True),
    StructField("QuadClass", IntegerType(), True),
    StructField("GoldsteinScale", FloatType(), True),
    StructField("NumMentions", IntegerType(), True),
    StructField("NumSources", IntegerType(), True),
    StructField("NumArticles", IntegerType(), True),
    StructField("AvgTone", FloatType(), True),
    StructField("Actor1Geo_Type", IntegerType(), True),
    StructField("Actor1Geo_Fullname", StringType(), True),
    StructField("Actor1Geo_CountryCode", StringType(), True),
    StructField("Actor1Geo_ADM1Code", StringType(), True),
    StructField("Actor1Geo_Lat", FloatType(), True),
    StructField("Actor1Geo_Long", FloatType(), True),
    StructField("Actor1Geo_FeatureID", StringType(), True),
    StructField("Actor2Geo_Type", IntegerType(), True),
    StructField("Actor2Geo_Fullname", StringType(), True),
    StructField("Actor2Geo_CountryCode", StringType(), True),
    StructField("Actor2Geo_ADM1Code", StringType(), True),
    StructField("Actor2Geo_Lat", FloatType(), True),
    StructField("Actor2Geo_Long", FloatType(), True),
    StructField("Actor2Geo_FeatureID", StringType(), True),
    StructField("ActionGeo_Type", IntegerType(), True),
    StructField("ActionGeo_Fullname", StringType(), True),
    StructField("ActionGeo_CountryCode", StringType(), True),
    StructField("ActionGeo_ADM1Code", StringType(), True),
    StructField("ActionGeo_Lat", FloatType(), True),
    StructField("ActionGeo_Long", FloatType(), True),
    StructField("ActionGeo_FeatureID", StringType(), True),
    StructField("DATEADDED", LongType(), True),
    StructField("SOURCEURL", StringType(), True),
    StructField("extraction_date", DateType(), True)
])

# Procesar el archivo del día anterior
url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"

try:
    # Descargar el archivo ZIP
    response = requests.get(url)
    response.raise_for_status()  # Verificar si la descarga fue exitosa
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    # Extraer el archivo CSV del ZIP
    csv_file_name = zip_file.namelist()[0]
    csv_file = zip_file.open(csv_file_name)

    # Cargar el CSV en un DataFrame de pandas
    df = pd.read_csv(csv_file, sep='\t', header=None, names=column_names)

    # Convertir las columnas numéricas
    numeric_columns = [
        'GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate', 'IsRootEvent', 
        'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass', 'GoldsteinScale', 
        'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 
        'Actor1Geo_Type', 'Actor1Geo_Lat', 'Actor1Geo_Long', 
        'Actor2Geo_Type', 'Actor2Geo_Lat', 'Actor2Geo_Long', 
        'ActionGeo_Type', 'ActionGeo_Lat', 'ActionGeo_Long', 
        'DATEADDED'
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Convertir las columnas a string
    string_columns = [
        'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode',
        'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
        'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode', 'Actor2EthnicCode',
        'Actor2Religion1Code', 'Actor2Religion2Code', 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code', 'Actor1Geo_Fullname', 'Actor1Geo_CountryCode', 
        'Actor1Geo_ADM1Code', 'Actor1Geo_FeatureID', 'Actor2Geo_Fullname', 
        'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_FeatureID', 
        'ActionGeo_Fullname', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code', 
        'ActionGeo_FeatureID', 'SOURCEURL'
    ]
    df[string_columns] = df[string_columns].astype(str)

    # Agregar la columna extraction_date con la fecha del archivo
    df['extraction_date'] = current_date.date()

    # Convertir el DataFrame de pandas a un DataFrame de Spark
    spark_df = spark.createDataFrame(df, schema=schema)

    # Verificar si la tabla ya existe en formato Delta y realizar el upsert
    if table_exists:
        delta_table = DeltaTable.forPath(spark, delta_table_path)
        delta_table.alias("tgt").merge(
                source=spark_df.alias("src"),
                condition="tgt.GlobalEventID = src.GlobalEventID"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        print(f"Datos actualizados para {date_str}.")
    else:
        # Si la tabla no existe, crearla en formato Delta
        spark_df.write.format("delta").mode("overwrite").save(delta_table_path)
        print(f"Tabla creada y datos insertados para {date_str}.")
        table_exists = True

except requests.exceptions.RequestException as e:
    print(f"Error al descargar o procesar los datos para la fecha {date_str}: {e}")
except Exception as e:
    print(f"Error al procesar los datos para la fecha {date_str}: {e}")

# Finalizar la sesión de Spark
spark.stop()


