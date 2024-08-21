# Databricks notebook source
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from delta.tables import DeltaTable
from pyspark.sql.functions import lit
import io

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("GDELT Delta Table Creator").getOrCreate()
spark.conf.set("fs.s3a.access.key", dbutils.widgets.get("aws_access_key"))
spark.conf.set("fs.s3a.secret.key", dbutils.widgets.get("aws_secret_access_key"))
spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")

# Especificar la ruta de la tabla Delta en la ubicación predeterminada de S3
delta_table_path = dbutils.widgets.get("aws_delta_table_path")

# Verificar si la tabla ya existe en formato Delta
table_exists = DeltaTable.isDeltaTable(spark, delta_table_path)

# Nombres de las columnas según GDELT
column_names = [
    "DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS",
    "PERSONS", "ORGANIZATIONS", "TONE", "CAMEOEVENTIDS",
    "SOURCES", "SOURCEURLS"
]

# Definir el esquema para la tabla de Spark
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

# Leer el archivo Parquet desde S3
extraction_date = dbutils.jobs.taskValues.get("00_get_events_control_date", "next_date_to_process_gkg")
extraction_date_for_s3 = extraction_date.replace("-", "")

s3_base_path = dbutils.widgets.get("aws_raw_path")
s3_path = f"{s3_base_path}/{extraction_date_for_s3}.parquet"

spark_df = spark.read.parquet(s3_path)

# Cast columns to match the schema of the Delta table
spark_df = spark_df.select(
    spark_df["DATE"].cast(IntegerType()),
    spark_df["NUMARTS"].cast(IntegerType()),
    spark_df["COUNTS"].cast(StringType()),
    spark_df["THEMES"].cast(StringType()),
    spark_df["LOCATIONS"].cast(StringType()),
    spark_df["PERSONS"].cast(StringType()),
    spark_df["ORGANIZATIONS"].cast(StringType()),
    spark_df["TONE"].cast(StringType()),
    spark_df["CAMEOEVENTIDS"].cast(StringType()),
    spark_df["SOURCES"].cast(StringType()),
    spark_df["SOURCEURLS"].cast(StringType())
)

# Add the extraction_date and last_modified columns
spark_df = spark_df.withColumn("extraction_date", lit(extraction_date))
spark_df = spark_df.withColumn("last_modified", lit(extraction_date))

# Write to Delta table
spark_df.write.format("delta").mode("append").save(delta_table_path)
print(f"Tabla creada y datos insertados desde el archivo {s3_path}.")

# Finalizar la sesión de Spark
spark.stop()

# COMMAND ----------


