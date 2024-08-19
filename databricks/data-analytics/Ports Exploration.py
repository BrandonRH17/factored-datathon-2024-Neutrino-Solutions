# Databricks notebook source
# Loading Libraries for the notebook
from pyspark.sql.functions import *
from geopy.distance import geodesic
from pyspark.sql.functions import lit, col, udf
from pyspark.sql.types import DoubleType
import datetime
from pyspark.sql.window import Window


# COMMAND ----------

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Convertir la función en un UDF (User Defined Function)
calculate_distance_udf = udf(calculate_distance, DoubleType())

# COMMAND ----------

# Loading Data from BRONZE database

GDELT_EVENTS = spark.sql(f"""SELECT *
FROM BRONZE.GDELT_EVENTS
WHERE 
    ActionGeo_CountryCode = 'CA'
    AND TO_DATE(CAST(DATEADDED AS STRING), 'yyyyMMdd') BETWEEN '2023-06-01' AND '2023-07-30'
    AND EventRootCode IN (14, 15, 16, 17, 19, 20, 21, 22)
ORDER BY DATEADDED DESC""")
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

display(PORT_LOCATIONS_DIM_CLEANED) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## DATA ANALYSIS FOR WHOLE COUNTRIES IN THE TRANSPACIFIC ROUTE:
# MAGIC * CANADA
# MAGIC * USA
# MAGIC * CHINA
# MAGIC * JAPON
# MAGIC * SOUTH KOREA
# MAGIC * TAIWAN
# MAGIC * VEITNAM
# MAGIC * HONG KONG

# COMMAND ----------

canada_port = PORT_LOCATIONS_DIM_CLEANED.filter("PORT = 'Vancouver, B.C., Canada '")
display(canada_port)

# COMMAND ----------

vancouver_lat = 49.17
vancouver_long = -123.07

events_with_distance = GDELT_EVENTS.withColumn(
    "distance_to_vancouver",
    calculate_distance_udf(col("ActionGeo_Lat"), col("ActionGeo_Long"), lit(vancouver_lat), lit(vancouver_long))
)


# COMMAND ----------

events_near_vancouver = events_with_distance.filter(col("distance_to_vancouver") <= 25)
display(events_near_vancouver)

# COMMAND ----------


display(events_with_distance)
