# Databricks notebook source
# MAGIC %md
# MAGIC ## TESTING PHASE

# COMMAND ----------

import pandas as pd
import requests
import zipfile
import io

url = f"http://data.gdeltproject.org/gkg/20240813.gkg.csv.zip"

# DOWLOAD ZIP FILE
response = requests.get(url) 
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# EXTRACT ZIP FILE
csv_file_name = zip_file.namelist()[0] 
csv_file = zip_file.open(csv_file_name)

# LOAD CSV FILE
df = pd.read_csv(csv_file, sep='\t')

# CHECK COLUMN COUNT
column_counts = df.apply(lambda row: len(row), axis=1).value_counts()

# SHOW DATAFRAME
print(column_counts)
df.head(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ## LOAD DATA FROM A PERIOD - PROD PHASE

# COMMAND ----------

import pandas as pd
import requests
import zipfile
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from datetime import datetime, timedelta
from pyspark.sql.functions import col

# INIT SPARK SESSION
spark = SparkSession.builder.appName("GKG Data Loader").getOrCreate()

# S3 DELTA TABLE PATH
delta_table_path = "s3://databricks-workspace-stack-e63e7-bucket/unity-catalog/2600119076103476/bronze/gdelt/gkg/"

# DEFINE THE RANGE OF DATES YOU WANT TO LOAD
start_date = datetime.strptime("2023-03-27", "%Y-%m-%d")
end_date = datetime.strptime("2024-08-18", "%Y-%m-%d")  # Example end date

# DEFINE COLUMN NAMES
column_names = [
    "DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS",
    "PERSONS", "ORGANIZATIONS", "TONE", "CAMEOEVENTIDS",
    "SOURCES", "SOURCEURLS"
]

# DEFINE SPARK SCHEMA 
schema = StructType([
    StructField("DATE", IntegerType(), True),  
    StructField("NUMARTS", IntegerType(), True),  
    StructField("COUNTS", StringType(), True), 
    StructField("THEMES", StringType(), True), 
    StructField("LOCATIONS", StringType(), True), 
    StructField("PERSONS", StringType(), True),
    StructField("ORGANIZATIONS", StringType(), True), 
    StructField("TONE", StringType(), True),
    StructField("CAMEOEVENTIDS", StringType(), True), 
    StructField("SOURCES", StringType(), True), 
    StructField("SOURCEURLS", StringType(), True),
    StructField("extraction_date", DateType(), True)  
])

# PROCESS THE RANGE OF DATES
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y%m%d')
    url = f"http://data.gdeltproject.org/gkg/{date_str}.gkg.csv.zip"
    
    try:
        # DOWNLOAD ZIP FILE
        response = requests.get(url)
        response.raise_for_status()
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        # EXTRACT ZIP FILE
        csv_file_name = zip_file.namelist()[0]
        csv_file = zip_file.open(csv_file_name)

        # LOAD DATA TO A DATAFRAME
        df = pd.read_csv(csv_file, sep='\t', header=None, names=column_names, low_memory=False)

        # CONVERT NUMERIC COLUMNS
        numeric_columns = ['DATE', 'NUMARTS']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # CONVERT STRING COLUMNS
        string_columns = [
            'COUNTS', 'THEMES', 'LOCATIONS', 'PERSONS', 'ORGANIZATIONS', 
            'TONE', 'CAMEOEVENTIDS', 'SOURCES', 'SOURCEURLS'
        ]
        df[string_columns] = df[string_columns].astype(str)

        # ADD EXTRACTION DATE COLUMN
        df['extraction_date'] = current_date.date()

        # CONVERT TO SPARK DATAFRAME
        spark_df = spark.createDataFrame(df, schema=schema)

        # INSERT DATA INTO THE DELTA TABLE
        spark_df.write.format("delta").mode("append").save(delta_table_path)
        print(f"NEW RECORDS INSERTED FOR DATE {date_str}.")

    except requests.exceptions.RequestException as e:
        print(f"AN ERROR HAS OCURRED FOR DATE {date_str}: {e}")
    except Exception as e:
        print(f"ERROR HAS OCURRED FOR DATE {date_str}: {e}")

    # INCREMENT DATE
    current_date += timedelta(days=1)

# STOP SPARK SESSION
spark.stop()

