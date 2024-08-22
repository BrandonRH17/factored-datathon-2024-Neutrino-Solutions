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
    "SOURCES", "SOURCEURLS",  "extraction_date"
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
spark_df = spark_df.withColumn("extraction_date", lit(extraction_date))
spark_df = spark_df.withColumn("last_modified", lit(extraction_date))

if table_exists:
    delta_table = DeltaTable.forPath(spark, delta_table_path)
    delta_table.alias("tgt").merge(
        source=spark_df.alias("src"),
        condition="tgt.DATE = src.DATE AND tgt.NUMARTS = src.NUMARTS"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print(f"Datos actualizados para el archivo {s3_path}.")
else:
    spark_df.write.format("delta").mode("append").save(delta_table_path)
    print(f"Tabla creada y datos insertados desde el archivo {s3_path}.")
    table_exists = True

# Finalizar la sesión de Spark
spark.stop()

# COMMAND ----------


