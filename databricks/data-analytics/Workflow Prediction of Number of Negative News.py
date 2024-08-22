# Databricks notebook source
# Loading Libraries for the notebook
from pyspark.sql.functions import *
import datetime
from pyspark.sql.window import Window

# COMMAND ----------

GKG = spark.sql("SELECT * FROM BRONZE.GDELT_GKG")

# COMMAND ----------

GKG_PRINCIPAL_CLEANING =(GKG
.withColumn("Date", to_date(col("DATE"), "yyyyMMdd")) # CREATE DATE COLUMN
.filter("Date >= '2022-01-01' and Date < '2024-08-01'") # SELECTE NEWS FROM 2022 TO JULY 2024
.withColumn("CountryCode", split(col("LOCATIONS"),"#").getItem(2)) # GET COUNTRY CODE
.withColumn("LocationCode", split(col("LOCATIONS"),"#").getItem(3)) # GET LOCATION CODE
.withColumn("AverageTone", split(col("TONE"),",").getItem(0)) # GET AVERAGE TONE
.withColumn("TonePositiveScore", split(col("TONE"),",").getItem(1)) # GET TONE POSITIVE SCORE
.withColumn("ToneNegativeScore", split(col("TONE"),",").getItem(2)) # GET TONE NEGATIVE SCORE
.withColumn("Polarity", split(col("TONE"),",").getItem(3))  # GET TONE POLARITY
.filter(col("LocationCode").isin("CA02","CA02","CA10","USCA","CH23",'CH30',"CH02","JA40","JA19","JA01")) # FILTER THE NEWS RELATED TO THE LOCATIONS OF THE PORT (CHECK DEFINITION IN THE CELL ABOVE FOR MORE DETAILS IN THE CODES)
)

# COMMAND ----------

TABLE_OF_DATES = (
GKG_PRINCIPAL_CLEANING
.select("Date").distinct()
)

# COMMAND ----------

GKG_NOT_EMOTIONAL_CHARE = (GKG_PRINCIPAL_CLEANING
.withColumn("Neutrality", when((col("AverageTone") >= -0.5) & (col("AverageTone") <= 0.5),1).otherwise(0)) # GET NEUTRALITY OF THE NEW (FLAG 1=NEUTRAL, 0=NOT NEUTRAL)
.withColumn("EC", when((col("Neutrality") == 1) & (col('Polarity') >= 9),1).otherwise(0)) #GET EMOTIONAL CHARGED FLAG (1=EMOTIONAL CHARGE,0=NOT EMOTIONAL CHARGED)
.filter("EC == 0") # GET ONLY THE NEWS THAT ARE NOT EMOTIONAL CHARGED
)

# COMMAND ----------

display(
GKG_NOT_EMOTIONAL_CHARE
.withColumn("BaseNews", when(
((col("THEMES").like("%PORT%")) | (col("THEMES").like("%TRANSPORT%")) | 
(col("THEMES").like("%SHIPPING%")) | (col("THEMES").like("%MARITIME%")) | (col("THEMES").like("%TRADE_PORT%")) | (col("THEMES").like("%NAVAL_PORT%")) | (col("THEMES").like("%LOGISTICS_PORT%"))) & (~col("THEMES").like("%AIRPORT%")),1).otherwise("Not Port Related"))
.filter("AverageTone < 0")
.filter("BaseNews == 1")
.groupby("Date").agg(count("Date").alias("NegativeNews"), avg("ToneNegativeScore").alias("ToneNegativeScore"))
)

# COMMAND ----------


