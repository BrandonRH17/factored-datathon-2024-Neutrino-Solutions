# Databricks notebook source
# Loading Libraries for the notebook
from pyspark.sql.functions import *
import datetime
from pyspark.sql.window import Window

# COMMAND ----------


# Function to Calculate Distance in Kilometers in a straight line between two points.

'''
CAL_LAT_LONG_DIST(df,lat1,long1,lat2,long2):
df = dataframe when we want to add the columns of "distance_in_kms"
lat1 = Latitued of Place 1
long1 = Longitude of Place 1
lat 2 = Latitude of Place 2
long2 = Longitued of Place 2

Outputs: DataFrame with a new column named = "distance_in_kms". The number represent the distance in KMS between the two points for each row.
'''

def cal_lat_log_dist(df, lat1, long1, lat2, long2):
        df = df.withColumn('distance_in_kms' , \
            round((acos((sin(radians(col(lat1))) * sin(radians(col(lat2)))) + \
                   ((cos(radians(col(lat1))) * cos(radians(col(lat2)))) * \
                    (cos(radians(long1) - radians(long2))))
                       ) * lit(6371.0)), 4))
        return df

# COMMAND ----------

# Loading Data from BRONZE database

GDELT_EVENTS = spark.sql("SELECT * FROM BRONZE.GDELT_EVENTS")
PORT_LOCATIONS_DIM = spark.sql("SELECT * FROM BRONZE.PORTS_DICTIONARY")
CAMEO_DICTIONARY = spark.sql("SELECT * FROM BRONZE.CAMEO_DICTIONARY")

# COMMAND ----------

# Claeaning RAW data from PORT_LOCATIONS

PORT_LOCATIONS_DIM_CLEANED = (
PORT_LOCATIONS_DIM
.filter("LATITUDE IS NOT NULL") #Filter for Latitud is nos null
.filter("LONGITUDE IS NOT NULL") #Filter for Longitud is nos null
.withColumn("LATITUDE", regexp_replace(col("LATITUDE")," ","")) #Eliminate black spaces in LATITUD column
.withColumn("LONGITUDE", regexp_replace(col("LONGITUDE")," ","")) #Eliminate black spaces in LATITUD column
.withColumn("Lat_Ori", substring(col("LATITUDE"),-1,1)) # Get N,S,W,E Orientation from latitud
.withColumn("Long_Ori", substring(col("LONGITUDE"),-1,1)) # Get N,S,W,E Orientation from longitude
.withColumn("LATITUDE_CORRECTED", #THIS NEW COLUMN CORRECT THE COORINDATES DEPENDING ON THE ORIENTATION N,S,W,E
            when(col("Lat_Ori") == 'S', expr("substring(LATITUDE,1,length(LATITUDE) - 1 )") * - 1) #GET CORRECT COORDINATES
            .when(col("Lat_Ori") == 'N', expr("substring(LATITUDE,1,length(LATITUDE) - 1 )")) #GET CORRECT COORDINATES
            .when(col("Lat_Ori") == 'E', expr("substring(LATITUDE,1,length(LATITUDE) - 1 )") * -1) #GET CORRECT COORDINATES
            .otherwise(999.999) # ID FOR CHECKING IF SOME VALUE ISN'T TAKEN INTO ACCOUNT
)
.withColumn("LONGITUDE_CORRECTED", #THIS NEW COLUMN CORRECT THE COORINDATES DEPENDING ON THE ORIENTATION N,S,W,E
            when(col("Long_Ori") == 'E', expr("substring(LONGITUDE,1,length(LONGITUDE) - 1 )")) #GET CORRECT COORDINATES
            .when(col("Long_Ori") == 'W', expr("substring(LONGITUDE,1,length(LONGITUDE) - 1 )") * -1)#GET CORRECT COORDINATES
            .when(col("Lat_Ori") == 'N', expr("substring(LATITUDE,1,length(LATITUDE) - 1 )") * -1) #GET CORRECT COORDINATES
            .otherwise(999.999) # ID FOR CHECKING IF SOME VALUE ISN'T TAKEN INTO ACCOUNT
)
.select("COUNTRY","PORT","LATITUDE_CORRECTED","LONGITUDE_CORRECTED") # SELECT COUNTRIES OF INTEREST
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DATA ANALYSIS FOR WHOLE COUNTRYS IN THE TRANSPACIFIC ROUTE:
# MAGIC * CANADA
# MAGIC * USA
# MAGIC * CHINA
# MAGIC * JAPON
# MAGIC * SOUTH KOREA
# MAGIC * TAIWAN
# MAGIC * VIETNAM
# MAGIC * HONG KONG

# COMMAND ----------

GDELT_EVENTS_TPR = (GDELT_EVENTS
.filter(col("ActionGeo_CountryCode").isin("US","CA","VM","CH","JA","HK","KS")) # FILTER FOR COUNTRIES OF INTEREST
.join(CAMEO_DICTIONARY,col("EventRootCode") == col("CAMEO CODE"), "left") #GET NAME FOR EventRootCode
.filter("DESCRIPTION is not null") #NO NEWS WITH NO CLEAR DESCRIPTION
.withColumn("Date", to_date(col("Day").cast("string"), "yyyyMMdd")) # CREATE COLUMN OF DATE TYPE
.withColumn("YearWeek", weekofyear(col("Date"))) # GET NUMBER OF WEEk OF THE YEAR
.withColumn("MonthYearWeek", concat(col("MonthYear"),col("YearWeek"))) #GET DATE ID for MONTH,YEAR,WEEK
)

# COMMAND ----------

# MAGIC %md 
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Objective
# MAGIC In this notebook, I will use the example of China's deteriorating relationship with Hong Kong. For this, I will take the "Ley de Seguridad Nacional, en Hong Kong" which ocurred in the 30th of June of 2020. 

# COMMAND ----------


GDELT_EVENTS_CHINA_HONGKONG_EVENT = (
    GDELT_EVENTS_TPR
    .filter(
        (col("ActionGeo_CountryCode").isin("CH")) &
        (col("DATE") >= "2023-02-01") & 
        (col("DATE") <= "2020-05-30")
    )
)

GDELT_EVENTS_CHINA_HONGKONG_EVENT.show()


# COMMAND ----------

goldstein_variation = (
    GDELT_EVENTS_CHINA_HONGKONG_EVENT
    .withColumn('Goldstein positive', when(col("GoldsteinScale") > 0, col("GoldsteinScale")).otherwise(0))
    .withColumn('Goldstein negative', when(col("GoldsteinScale") < 0, col("GoldsteinScale")).otherwise(0))
    )

daily_goldstein_variation = (
    goldstein_variation
    .groupBy("DATE").
    agg(
        mean("Goldstein positive").alias("Goldstein positive"),
        mean("Goldstein negative").alias("Goldstein negative")
    )
    .orderBy("DATE")
)

display(daily_goldstein_variation)

# COMMAND ----------

display(GDELT_EVENTS_CHINA_HONGKONG_EVENT.filter(col("DATE") == "2020-06-26"))

# COMMAND ----------


