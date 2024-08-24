# Databricks notebook source
# Loading Libraries for the notebook
import datetime
import requests
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from bs4 import BeautifulSoup
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# COMMAND ----------

gdelt_gkg = spark.sql("SELECT * FROM BRONZE.GDELT_GKG")


# COMMAND ----------

gdelt_gkg_filtered =(gdelt_gkg
.withColumn("Date", to_date(col("DATE"), "yyyyMMdd")) # CREATE DATE COLUMN
.withColumn("CountryCode", split(col("LOCATIONS"),"#").getItem(2)) # GET COUNTRY CODE
.withColumn("LocationCode", split(col("LOCATIONS"),"#").getItem(3)) # GET LOCATION CODE
.withColumn("AverageTone", split(col("TONE"),",").getItem(0)) # GET AVERAGE TONE
.withColumn("TonePositiveScore", split(col("TONE"),",").getItem(1)) # GET TONE POSITIVE SCORE
.withColumn("ToneNegativeScore", split(col("TONE"),",").getItem(2)) # GET TONE NEGATIVE SCORE
.withColumn("Polarity", split(col("TONE"),",").getItem(3))  # GET TONE POLARITY
.filter(col("LocationCode").isin("CA02","CA02","CA10","USCA","CH23",'CH30',"CH02","JA40","JA19","JA01")&
        (
            (col("THEMES").like("%PORT%")) |
            (col("THEMES").like("%TRANSPORT%")) |
            (col("THEMES").like("%SHIPPING%")) |
            (col("THEMES").like("%MARITIME%")) |
            (col("THEMES").like("%TRADE_PORT%")) |
            (col("THEMES").like("%NAVAL_PORT%")) |
            (col("THEMES").like("%LOGISTICS_PORT%"))
        ) &
        (~col("THEMES").like("%AIRPORT%"))) # FILTER THE NEWS RELATED TO THE LOCATIONS OF THE PORT (CHECK DEFINITION IN THE CELL ABOVE FOR MORE DETAILS IN THE CODES)
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


