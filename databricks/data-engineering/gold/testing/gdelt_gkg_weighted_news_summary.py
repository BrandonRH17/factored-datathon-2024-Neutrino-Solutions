# Databricks notebook source
import datetime
import requests
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, split, to_date

# COMMAND ----------

gkg_scraped = spark.sql("SELECT * FROM SILVER.S_GDELT_GKG_SCRAPING")

# COMMAND ----------

gkg_scraped_without_emotional_charge = (gkg_scraped
.withColumn("Neutrality", when((col("AverageTone") >= -0.5) & (col("AverageTone") <= 0.5),1).otherwise(0)) # GET NEUTRALITY OF THE NEW (FLAG 1=NEUTRAL, 0=NOT NEUTRAL)
.withColumn("EC", when((col("Neutrality") == 1) & (col('Polarity') >= 9),1).otherwise(0)) #GET EMOTIONAL CHARGED FLAG (1=EMOTIONAL CHARGE,0=NOT EMOTIONAL CHARGED)
.filter("EC == 0") # GET ONLY THE NEWS THAT ARE NOT EMOTIONAL CHARGED
)

# COMMAND ----------

gkg_with_weights = (    
gkg_scraped_without_emotional_charge
.withColumn("BaseNews", when((col("THEMES").like("%PORT%")) & (col("THEMES").like("%TRANSPORT%") & (~col("THEMES").like("%AIRPORT%"))),1))
.filter("BaseNews == 1")
.withColumn("NewsWithTINFA", when(col("THEMES").like("%TRANSPORT_INFRASTRUCTURE%"),1))
.withColumn("NewsWithTRADE", when(col("THEMES").like("%TRADE%"),1))
.withColumn("NewsWithME", when(col("THEMES").like("%MACROECONOMIC%"),1))
.withColumn("NewsWithPS", when(col("THEMES").like("%PUBLIC_SECTOR%"),1))
.withColumn("NewsWithMI", when(col("THEMES").like("%MARITIME_INCIDENT%"),1))
.fillna(0)
.withColumn("Total", col("NewsWithTINFA")+col("NewsWithTRADE")+col("NewsWithME") + col("NewsWithPS") + col("NewsWithMI"))
.filter("AverageTone < 0")
.groupby("Date","Total","LocationCode", "is_ruta_transpacifica", "is_ruta_transatlantica", "is_ruta_del_cabo_de_buena_esperanza").agg(count("Total").alias("NumberOfNews"))
.withColumn('WeightedCountOfNews', when(col("Total") == 5, col("NumberOfNews") * 500)
.when(col('Total') == 4, col("NumberOfNews") * 250)
.when(col("NumberOfNews") ==3, col("NumberOfNews") * 100)
.when(col("Total") == 2, col("NumberOfNews") * 5)
.otherwise(0))
)

# COMMAND ----------

display(gkg_with_weights.limit(20))

# COMMAND ----------

table_name = "gold.g_gdelt_gkg_weights_report"
delta_table_path = f"s3://databricks-workspace-stack-e63e7-bucket/unity-catalog/2600119076103476/gold/gdelt/gkg_weights_report/"  # O especifica una ruta personalizada si usas S3 o DBFS

# COMMAND ----------


# Filtrar los registros donde el año de extraction_date sea 2024
if DeltaTable.isDeltaTable(spark, delta_table_path):
    # Si la tabla existe, inserta los datos
    print("La tabla existe, insertando los datos")
    delta_table = DeltaTable.forPath(spark, delta_table_path)
    delta_table.alias("tgt").merge(
        source=gkg_with_weights.alias("src"),
        condition="tgt.DATE = src.DATE AND "
        "tgt.Total = src.Total AND "
        "tgt.LocationCode = src.LocationCode AND "
        "tgt.NumberOfNews = src.NumberOfNews"  
    ).whenNotMatchedInsertAll().execute()
else:
    # Si la tabla no existe, créala e inserta los datos
    print("La tabla no existe, creando la tabla y insertando los datos")
    gkg_with_weights.write.format("delta").mode("overwrite").save(delta_table_path)
    print("Se ha escrito en formato delta.")
    spark.sql(f"CREATE TABLE {table_name} USING DELTA LOCATION '{delta_table_path}'")
