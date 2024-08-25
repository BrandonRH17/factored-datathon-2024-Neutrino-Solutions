# Databricks notebook source
!pip install bs4

# Loading Libraries for the notebook
import datetime
import requests
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from bs4 import BeautifulSoup
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, split, to_date


# COMMAND ----------


gdelt_gkg = spark.sql("SELECT * FROM BRONZE.GDELT_GKG")

# COMMAND ----------


# Define los códigos para cada ruta
transpacifica_codes = ["CA02", "CA10", "USCA", "CH23", "CH30", "CH02", "JA40", "JA19", "JA01"]
transatlantica_codes = ["UKF2", "UKN5", "GM04", "GM03", "SP51", "SP60", "USNY", "USGA", "USFL", "CA10", "CA07"]

# Modifica el DataFrame para incluir las nuevas columnas
gdelt_gkg_filtered = (
    gdelt_gkg
    .withColumn("Date", to_date(col("DATE"), "yyyyMMdd"))  # CREATE DATE COLUMN
    .withColumn("CountryCode", split(col("LOCATIONS"), "#").getItem(2))  # GET COUNTRY CODE
    .withColumn("LocationCode", split(col("LOCATIONS"), "#").getItem(3))  # GET LOCATION CODE
    .withColumn("AverageTone", split(col("TONE"), ",").getItem(0))  # GET AVERAGE TONE
    .withColumn("TonePositiveScore", split(col("TONE"), ",").getItem(1))  # GET TONE POSITIVE SCORE
    .withColumn("ToneNegativeScore", split(col("TONE"), ",").getItem(2))  # GET TONE NEGATIVE SCORE
    .withColumn("Polarity", split(col("TONE"), ",").getItem(3))  # GET TONE POLARITY
    .withColumn("is_ruta_transpacifica", when(col("LocationCode").isin(transpacifica_codes), 1).otherwise(0))  # Ruta Transpacífica
    .withColumn("is_ruta_transatlantica", when(col("LocationCode").isin(transatlantica_codes), 1).otherwise(0))  # Ruta Transatlántica
    .withColumn("is_ruta_del_cabo_de_buena_esperanza", lit(0))  # Ruta del Cabo de Buena Esperanza (Placeholder)
    .filter(
        col("LocationCode").isin(transpacifica_codes) &
        (
            (col("THEMES").like("%PORT%")) |
            (col("THEMES").like("%TRANSPORT%")) |
            (col("THEMES").like("%SHIPPING%")) |
            (col("THEMES").like("%MARITIME%")) |
            (col("THEMES").like("%TRADE_PORT%")) |
            (col("THEMES").like("%NAVAL_PORT%")) |
            (col("THEMES").like("%LOGISTICS_PORT%"))
        ) &
        (~col("THEMES").like("%AIRPORT%"))
    )  # FILTER THE NEWS RELATED TO THE LOCATIONS OF THE PORT
)


# COMMAND ----------

def scrape_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content
    except Exception as e:
        return str(e)

# Registrar la función como UDF en Spark
scrape_udf = udf(scrape_content, StringType())

# COMMAND ----------

gdelt_gkg_filtered_with_content = gdelt_gkg_filtered.withColumn('contenturl', scrape_udf(gdelt_gkg_filtered['SOURCEURLS']))

display(gdelt_gkg_filtered_with_content.limit(20))


# COMMAND ----------

table_name = "silver.s_gdelt_gkg_scraping"
delta_table_path = f"s3://databricks-workspace-stack-e63e7-bucket/unity-catalog/2600119076103476/silver/gdelt/gkg/"  # O especifica una ruta personalizada si usas S3 o DBFS


# COMMAND ----------


from pyspark.sql.functions import year, col

# Filtrar los registros donde el año de extraction_date sea 2024
gdelt_gkg_filtered_with_content = gdelt_gkg_filtered_with_content.filter(year(col("extraction_date")) == 2024)
if DeltaTable.isDeltaTable(spark, delta_table_path):
    # Si la tabla existe, inserta los datos
    print("La tabla existe, insertando los datos")
    delta_table = DeltaTable.forPath(spark, delta_table_path)
    delta_table.alias("tgt").merge(
        source=gdelt_gkg_filtered_with_content.alias("src"),
        condition="tgt.DATE = src.DATE AND "
        "tgt.NUMARTS = src.NUMARTS AND "
        "tgt.CAMEOEVENTIDS = src.CAMEOEVENTIDS AND "
        "tgt.THEMES = src.THEMES AND "
        "tgt.TONE = src.TONE AND "
        "tgt.LOCATIONS = src.LOCATIONS AND "
        "tgt.PERSONS = src.PERSONS AND "
        "tgt.ORGANIZATIONS = src.ORGANIZATIONS AND "
        "tgt.SOURCES = src.SOURCES AND "
        "tgt.SOURCEURLS = src.SOURCEURLS"  
    ).whenNotMatchedInsertAll().execute()
else:
    # Si la tabla no existe, créala e inserta los datos
    print("La tabla no existe, creando la tabla y insertando los datos")
    gdelt_gkg_filtered_with_content.write.format("delta").mode("overwrite").save(delta_table_path)
    print("Se ha escrito en formato delta.")
    spark.sql(f"CREATE TABLE {table_name} USING DELTA LOCATION '{delta_table_path}'")


