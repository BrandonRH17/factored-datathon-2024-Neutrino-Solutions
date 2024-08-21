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
    next_day_to_process = dbutils.jobs.taskValues.get("00_get_events_control_date", "next_date_to_process_gkg")
    date_str = next_day_to_process.replace("-", "")
    url = f"http://data.gdeltproject.org/gkg/20240819.gkg.csv.zip"
    
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
            "DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS",
            "PERSONS", "ORGANIZATIONS", "TONE", "CAMEOEVENTIDS",
            "SOURCES", "SOURCEURLS"
        ], dtype=str)
        
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
    upload_to_s3_parquet(df, date_str, 'factored-datalake-raw', 'gkg/')

# COMMAND ----------


