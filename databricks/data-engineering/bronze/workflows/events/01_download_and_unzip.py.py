# Databricks notebook source
import requests
import zipfile
import io
import pandas as pd
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import NoCredentialsError
from datetime import datetime, timedelta

def download_and_prepare_data():
    # Calcular la fecha del día anterior
    next_day_to_process = dbutils.jobs.taskValues.get("00_get_events_control_date", "next_date_to_process_events")
    date_str = next_day_to_process.replace("-", "")
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    
    try:
        # Descargar el archivo ZIP
        response = requests.get(url)
        response.raise_for_status()
        
        # Descomprimir el archivo ZIP en memoria
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_file_name = zip_file.namelist()[0]
        csv_file = zip_file.open(csv_file_name)
        
        # Leer el archivo CSV en un DataFrame de pandas
        df = pd.read_csv(csv_file, sep='\t', header=None, names=[
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
        ])
        
        print(f"Datos para {date_str} descargados y preparados.")
        return df, date_str

    except requests.exceptions.RequestException as e:
        print(f"Error al descargar o descomprimir los datos para la fecha {date_str}: {e}")
        raise

def upload_to_s3_parquet(df, file_name, bucket_name, s3_prefix=""):
    # Configuración de credenciales de AWS
    aws_access_key_id = dbutils.widgets.get("aws_access_key")
    aws_secret_access_key = dbutils.widgets.get("aws_secret_access_key")
    
    # Inicializar el cliente de S3 con las credenciales
    s3 = boto3.client('s3', 
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)
    
    # Convertir el DataFrame a un archivo Parquet en memoria
    table = pa.Table.from_pandas(df)
    parquet_buffer = io.BytesIO()
    pq.write_table(table, parquet_buffer)
    parquet_buffer.seek(0)
    
    try:
        # Construir la key para el archivo en S3 (ruta dentro del bucket)
        s3_key = f"{s3_prefix}gdelt/{file_name}.parquet"
        
        # Subir el archivo Parquet al bucket S3
        s3.upload_fileobj(parquet_buffer, bucket_name, s3_key)
        print(f"Archivo {s3_key} subido a S3 en el bucket {bucket_name}.")
    except NoCredentialsError:
        print("Credenciales de AWS no encontradas.")
    except Exception as e:
        print(f"Error al subir el archivo a S3: {e}")
        raise

# Ejecución de las funciones
if __name__ == "__main__":
    # Paso 1: Descargar y preparar los datos
    df, date_str = download_and_prepare_data()
    
    # Paso 2: Subir el DataFrame como archivo Parquet a S3
    upload_to_s3_parquet(df, date_str, 'factored-datalake-raw', 'events/')


