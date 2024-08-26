# Databricks notebook source
import datetime
from pyspark.sql.functions import *
import pandas as pd
import keras
import joblib
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# COMMAND ----------

GKG = spark.sql("SELECT * FROM BRONZE.GDELT_GKG")

# COMMAND ----------

today = datetime.date.today()
days8past = today - datetime.timedelta(days=8)

# COMMAND ----------

GKG_PRINCIPAL_CLEANING =(GKG
.withColumn("Date", to_date(col("DATE"), "yyyyMMdd")) # CREATE DATE COLUMN
.filter(col("Date") >= days8past) # SELECTE NEWS FROM 2022 TO JULY 2024
.withColumn("CountryCode", split(col("LOCATIONS"),"#").getItem(2)) # GET COUNTRY CODE
.withColumn("LocationCode", split(col("LOCATIONS"),"#").getItem(3)) # GET LOCATION CODE
.withColumn("AverageTone", split(col("TONE"),",").getItem(0)) # GET AVERAGE TONE
.withColumn("TonePositiveScore", split(col("TONE"),",").getItem(1)) # GET TONE POSITIVE SCORE
.withColumn("ToneNegativeScore", split(col("TONE"),",").getItem(2)) # GET TONE NEGATIVE SCORE
.withColumn("Polarity", split(col("TONE"),",").getItem(3))  # GET TONE POLARITY
.filter(col("LocationCode").isin("CA02","CA10","USCA","CH23",'CH30',"CH02","JA40","JA19","JA01")) # FILTER THE NEWS RELATED TO THE LOCATIONS OF THE PORT (CHECK DEFINITION IN THE CELL ABOVE FOR MORE DETAILS IN THE CODES)
)

# COMMAND ----------

GKG_NOT_EMOTIONAL_CHARGE = (GKG_PRINCIPAL_CLEANING
.withColumn("Neutrality", when((col("AverageTone") >= -0.5) & (col("AverageTone") <= 0.5),1).otherwise(0)) # GET NEUTRALITY OF THE NEW (FLAG 1=NEUTRAL, 0=NOT NEUTRAL)
.withColumn("EC", when((col("Neutrality") == 1) & (col('Polarity') >= 9),1).otherwise(0)) #GET EMOTIONAL CHARGED FLAG (1=EMOTIONAL CHARGE,0=NOT EMOTIONAL CHARGED)
.filter("EC == 0") # GET ONLY THE NEWS THAT ARE NOT EMOTIONAL CHARGED
)

# COMMAND ----------

GKG_NEWS_AND_TONE = (
GKG_NOT_EMOTIONAL_CHARGE
.withColumn("BaseNews", when(
((col("THEMES").like("%PORT%")) | (col("THEMES").like("%TRANSPORT%")) | 
(col("THEMES").like("%SHIPPING%")) | (col("THEMES").like("%MARITIME%")) | (col("THEMES").like("%TRADE_PORT%")) | (col("THEMES").like("%NAVAL_PORT%")) | (col("THEMES").like("%LOGISTICS_PORT%"))) & (~col("THEMES").like("%AIRPORT%")),1).otherwise("Not Port Related")) #CREATE A FLAG THAT TELLS US IF THE NEW IS WHITHIN THE THEMES WE WANT TO SELECT
.filter("AverageTone < 0") # FILTER THAT THE AVERAGE TONE OF THE NEWS IS NEGATIVE
.filter("BaseNews == 1")) # KEEP ONLY NEWS THAT ACCOMPLISH THE THEME FILTER

# COMMAND ----------

GKG_NEWS_AND_TONE_DAILY = (GKG_NEWS_AND_TONE
.groupby("Date").agg(count("Date").alias("NegativeNews"), avg("ToneNegativeScore").alias("ToneNegativeScore"), avg("Polarity").alias("Polarity"),max("ToneNegativeScore").alias("MaxToneNeg"),min("ToneNegativeScore").alias("MinToneNeg"))
) # GROUP OUR DATA BY DATE AND CREATE THE VARIABLES DESCRIBED BELOW

# COMMAND ----------

VAR_THEMES_LAG_DAILY = (
GKG_NEWS_AND_TONE
# GENERATE FLAGS IF THE NEWS HAVE THE THEMES WE ARE INTERESTED IN RELATION WITH PORT PROBLEMS
.withColumn("NewsAboutDisruption", when(col("THEMES").like("%DISRUPTION%"),1).otherwise(0)) 
.withColumn("NewsAboutStrikes", when(col("THEMES").like("%STRIKES%"),1).otherwise(0))
.withColumn("NewsAboutProtest", when(col("THEMES").like("%PROTEST%"),1).otherwise(0))
.withColumn("NewsAboutMaritimeDisaster", when(col("THEMES").like("%MARITIME_DISASTER%"),1).otherwise(0))
.withColumn("NewsAboutTerrorism", when(col("THEMES").like("%TERRORISM%"),1).otherwise(0))
.withColumn("NewsAboutPiracy", when(col("THEMES").like("%PIRACY%"),1).otherwise(0))
.withColumn("NewsAboutCongestion", when(col("THEMES").like("%CONGESTION%"),1).otherwise(0))
# GROUPING OUR DATA BY DATE AND SUM OUR FLAGS TO GET HOW MANY NEWS ARE RELATED TO EACH THEME PER DAY
.groupBy("Date").agg(sum("NewsAboutDisruption").alias("DisruptionNews"),sum("NewsAboutStrikes").alias("StrikeNews"),sum("NewsAboutProtest").alias("ProtestNews"),sum("NewsAboutMaritimeDisaster").alias('MDNews'),sum("NewsAboutTerrorism").alias("TerrorismNews"),sum("NewsAboutPiracy").alias("PiracyNews"),sum("NewsAboutCongestion").alias("CongestionNews"))
)


# COMMAND ----------

GKG_DAILY_FINAL = (GKG_NEWS_AND_TONE_DAILY
.join(VAR_THEMES_LAG_DAILY,"Date","left")
.fillna(0)
.persist()
) # JOINING OUR VARIABLES BY DATE

GKG_DAILY_FINAL.count()

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL.toPandas()

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS['Date'] = pd.to_datetime(GKG_DAILY_FINAL_PANDAS['Date']) # ADJUST DATE TO DATE TIME OBJECT
GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL_PANDAS.sort_values('Date') 

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS['Diff'] = GKG_DAILY_FINAL_PANDAS['NegativeNews'].diff() # APPLY DIFFERENCING
GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL_PANDAS.dropna()

# COMMAND ----------

n_input=7
n_features=13

# COMMAND ----------

TPRMODEL = keras.saving.load_model("TPR_model.keras")
X_SCALER = joblib.load("scalerTPR.save")
Y_SCALER = joblib.load("scalerTPR_Obj.save") 

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS_PRED = GKG_DAILY_FINAL_PANDAS.drop("Date", axis = 1)

# COMMAND ----------

scaled_data = X_SCALER.transform(GKG_DAILY_FINAL_PANDAS_PRED)

# COMMAND ----------

pred_generator = TimeseriesGenerator(scaled_data, # DATAFRAME
                                     scaled_data[:, 12], # OBJECTIVE
                                      n_input # NUMBER OF LAGS
)

# COMMAND ----------

preds =  (TPRMODEL.predict(pred_generator)) #MODEL PREDICTIONS

# COMMAND ----------

real_preds = []

transformed_preds = Y_SCALER.inverse_transform(preds)

for i in range(0,len(transformed_preds)):
    real_preds.append(transformed_preds[i][0])

# COMMAND ----------

real_preds

# COMMAND ----------

PREDICTIONS_DATAFRAME = GKG_DAILY_FINAL_PANDAS[7:]
PREDICTIONS_DATAFRAME['DifferencialPredictions'] = real_preds
PREDICTIONS_DATAFRAME['NextDayNegativeNewsPredictions'] = PREDICTIONS_DATAFRAME['NegativeNews'] + PREDICTIONS_DATAFRAME['DifferencialPredictions']

OUTPUT_DATAFRAME = PREDICTIONS_DATAFRAME[['Date','NegativeNews','NextDayNegativeNewsPredictions']]
OUTPUT_DATAFRAME['PredictionLocation'] = 'transpacific_route'

# COMMAND ----------

OUTPUT_DATAFRAME
