# Databricks notebook source
# MAGIC %md
# MAGIC # Models Generation to Predict Status in Ports and Transpacific Route

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries
# MAGIC
# MAGIC In this piece of code you can see and import all the libraries that im going to use to train the models

# COMMAND ----------

# Loading Libraries for the notebook
from pyspark.sql.functions import *
import datetime
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit,KFold
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from scipy import stats
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import GradientBoostingRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Data
# MAGIC
# MAGIC In the next piece of code we load the GKG dataset, that is the data that i'm going to use to create our objective variable and create predictors

# COMMAND ----------

GKG = spark.sql("SELECT * FROM BRONZE.GDELT_GKG")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Principal Cleaning
# MAGIC
# MAGIC GKG need some Principal Cleaning we need to perform. 
# MAGIC 1. Create a Date Column 
# MAGIC
# MAGIC 2. Get from column LOCATIONS
# MAGIC
# MAGIC CountryCode: ID for Countries, LocationCode: ID for Locations in a Country (ex:California)
# MAGIC
# MAGIC 3. Get from column TONE
# MAGIC
# MAGIC AverageTone: Average Tone of the News Related to Event, TonePositiveScore: Positve Tone Score of the News Related to the event, ToneNegativeScore: Negative Tone Score of the News Related to the event, Polarity: Negative and Positive Word directly index of the event
# MAGIC
# MAGIC 4. Filter for locations of interest, In our case as we are focusing on the transpacific route, we would focus on this locations: "CA02","CA02","CA10","USCA","CH23","CH30","CH02","JA40","JA19","JA01"

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
.filter(col("LocationCode").isin("CA02","CA10","USCA","CH23",'CH30',"CH02","JA40","JA19","JA01")) # FILTER THE NEWS RELATED TO THE LOCATIONS OF THE PORT (CHECK DEFINITION IN THE CELL ABOVE FOR MORE DETAILS IN THE CODES)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eliminating Emotional Charges

# COMMAND ----------

GKG_NOT_EMOTIONAL_CHARGE = (GKG_PRINCIPAL_CLEANING
.withColumn("Neutrality", when((col("AverageTone") >= -0.5) & (col("AverageTone") <= 0.5),1).otherwise(0)) # GET NEUTRALITY OF THE NEW (FLAG 1=NEUTRAL, 0=NOT NEUTRAL)
.withColumn("EC", when((col("Neutrality") == 1) & (col('Polarity') >= 9),1).otherwise(0)) #GET EMOTIONAL CHARGED FLAG (1=EMOTIONAL CHARGE,0=NOT EMOTIONAL CHARGED)
.filter("EC == 0") # GET ONLY THE NEWS THAT ARE NOT EMOTIONAL CHARGED
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base Table

# COMMAND ----------

GKG_NEWS_AND_TONE = (
GKG_NOT_EMOTIONAL_CHARGE
.withColumn("BaseNews", when(
((col("THEMES").like("%PORT%")) | (col("THEMES").like("%TRANSPORT%")) | 
(col("THEMES").like("%SHIPPING%")) | (col("THEMES").like("%MARITIME%")) | (col("THEMES").like("%TRADE_PORT%")) | (col("THEMES").like("%NAVAL_PORT%")) | (col("THEMES").like("%LOGISTICS_PORT%"))) & (~col("THEMES").like("%AIRPORT%")),1).otherwise("Not Port Related"))
.filter("AverageTone < 0")
.filter("BaseNews == 1"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Objective Variables Count of Negative News and Tone Negative Scores for Daily and Weekly temporality

# COMMAND ----------

GKG_NEWS_AND_TONE_DAILY = (GKG_NEWS_AND_TONE
.groupby("Date").agg(count("Date").alias("NegativeNews"), avg("ToneNegativeScore").alias("ToneNegativeScore"), avg("Polarity").alias("Polarity"),max("ToneNegativeScore").alias("MaxToneNeg"),min("ToneNegativeScore").alias("MinToneNeg"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Variables

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variable 1: Lags of Negative News and Negative Tone, Daily and Weekly

# COMMAND ----------

VAR_LAGS_DAILY = (
GKG_NEWS_AND_TONE_DAILY
.withColumn("Part", lit(1))
.withColumn("3daysAverage", avg(col("NegativeNews")).over(Window.partitionBy("Part").orderBy("Date").rowsBetween(-7,-1)))
.withColumn("7daysAverage", avg(col("NegativeNews")).over(Window.partitionBy("Part").orderBy("Date").rowsBetween(-15,-1)))
#.withColumn("5daysAverage", avg(col("NegativeNews")).over(Window.partitionBy("Part").orderBy("Date").rowsBetween(-5,-1)))
#.withColumn("7daysAverage", avg(col("NegativeNews")).over(Window.orderBy("Date").rowsBetween(-7,-1)))
#.withColumn("15daysAverage", avg(col("NegativeNews")).over(Window.orderBy("Date").rowsBetween(-15,-1)))
#.withColumn("30daysAverage", avg(col("NegativeNews")).over(Window.orderBy("Date").rowsBetween(-30,-1)))
.drop("NegativeNews","ToneNegativeScore","Part")
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable Two Average Polarity - Lagged

# COMMAND ----------

VAR_LAGS_POLARITY_DAILY = (
GKG_NEWS_AND_TONE
.groupby("Date").agg(avg("Polarity").alias("AveragePolarity"))
.withColumn("Part", lit(1))
.withColumn("OneDayLagPolarity", lag(col("AveragePolarity"),1).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("TwoDayLagPolarity", lag(col("AveragePolarity"),2).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("ThreeDayLagPolarity", lag(col("AveragePolarity"),3).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FourDayLagPolarity", lag(col("AveragePolarity"),4).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FiveDayLagPolarity", lag(col("AveragePolarity"),5).over(Window.partitionBy("Part").orderBy("Date")))
.drop("AveragePolarity","Part")
.fillna(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable 3 Lagged High Intensity New and Low Intensity New

# COMMAND ----------

VAR_LAGS_HIGH_LOW_INTENSITY_DAILY = (
GKG_NEWS_AND_TONE
.groupby("Date").agg(max("ToneNegativeScore").alias("MaxToneNegativeScore"), min("ToneNegativeScore").alias("MinToneNegativeScore"))
.withColumn("Part", lit(1))
.withColumn("OneDayLagMax", lag(col("MaxToneNegativeScore"),1).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("TwoDayLagMax", lag(col("MaxToneNegativeScore"),2).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("ThreeDayLagMax", lag(col("MaxToneNegativeScore"),3).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FourDayLagMax", lag(col("MaxToneNegativeScore"),4).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FiveDayLagMax", lag(col("MaxToneNegativeScore"),5).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("OneDayLagMin", lag(col("MinToneNegativeScore"),1).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("TwoDayLagMin", lag(col("MinToneNegativeScore"),2).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("ThreeDayLagMin", lag(col("MinToneNegativeScore"),3).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FourDayLagMin", lag(col("MinToneNegativeScore"),4).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FiveDayLagMin", lag(col("MinToneNegativeScore"),5).over(Window.partitionBy("Part").orderBy("Date")))
.fillna(0)
.drop("MaxToneNegativeScore","MinToneNegativeScore","Part"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable 4 Lagged Growth Percentages in Number of News

# COMMAND ----------

VAR_LAGS_GROWTH_DAYLY = (
GKG_NEWS_AND_TONE_DAILY
.withColumn("Part", lit(1))
.withColumn("PastDayNumNews", lag(col("NegativeNews"),1).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("Growth", (col("NegativeNews") - col("PastDayNumNews")/col("PastDayNumNews")))
.fillna(0)
.withColumn("OneDayLagGrowht", lag(col("Growth"),1).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("TwoDayLagGrowht", lag(col("Growth"),2).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("ThreeDayLagGrowht", lag(col("Growth"),3).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FourDayLagGrowht", lag(col("Growth"),4).over(Window.partitionBy("Part").orderBy("Date")))
.withColumn("FiveDayLagGrowht", lag(col("Growth"),5).over(Window.partitionBy("Part").orderBy("Date")))
.drop("Part","PastDayNumNews","NegativeNews","ToneNegativeScore","Growth")
.fillna(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable 5. Themes related to port problems

# COMMAND ----------

VAR_THEMES_LAG_DAILY = (
GKG_NEWS_AND_TONE
.withColumn("NewsAboutDisruption", when(col("THEMES").like("%DISRUPTION%"),1).otherwise(0))
.withColumn("NewsAboutStrikes", when(col("THEMES").like("%STRIKES%"),1).otherwise(0))
.withColumn("NewsAboutProtest", when(col("THEMES").like("%PROTEST%"),1).otherwise(0))
.withColumn("NewsAboutMaritimeDisaster", when(col("THEMES").like("%MARITIME_DISASTER%"),1).otherwise(0))
.withColumn("NewsAboutTerrorism", when(col("THEMES").like("%TERRORISM%"),1).otherwise(0))
.withColumn("NewsAboutPiracy", when(col("THEMES").like("%PIRACY%"),1).otherwise(0))
.withColumn("NewsAboutCongestion", when(col("THEMES").like("%CONGESTION%"),1).otherwise(0))
.groupBy("Date").agg(sum("NewsAboutDisruption").alias("DisruptionNews"),sum("NewsAboutStrikes").alias("StrikeNews"),sum("NewsAboutProtest").alias("ProtestNews"),sum("NewsAboutMaritimeDisaster").alias('MDNews'),sum("NewsAboutTerrorism").alias("TerrorismNews"),sum("NewsAboutPiracy").alias("PiracyNews"),sum("NewsAboutCongestion").alias("CongestionNews"))
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Tables for Model Training

# COMMAND ----------

 ### TABLAS FINALES

GKG_DAILY_FINAL = (GKG_NEWS_AND_TONE_DAILY
.join(VAR_THEMES_LAG_DAILY,"Date","left")
.fillna(0)
)

# COMMAND ----------

display(GKG_DAILY_FINAL)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training and Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ## Models for Number of News

# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL.toPandas()

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS['Date'] = pd.to_datetime(GKG_DAILY_FINAL_PANDAS['Date'])
GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL_PANDAS.sort_values('Date')

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller

adfuller(GKG_DAILY_FINAL_PANDAS['NegativeNews'], regression = "c", autolag= 'AIC')

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS['Diff'] = GKG_DAILY_FINAL_PANDAS['NegativeNews'].diff()
GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL_PANDAS.dropna()

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS

# COMMAND ----------

plt.plot(GKG_DAILY_FINAL_PANDAS['Date'], GKG_DAILY_FINAL_PANDAS['Diff'])


# COMMAND ----------

adfuller(GKG_DAILY_FINAL_PANDAS['Diff'], regression = "c", autolag= 'AIC')

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS_TRAIN = GKG_DAILY_FINAL_PANDAS.copy()
GKG_DAILY_FINAL_PANDAS_TRAIN = GKG_DAILY_FINAL_PANDAS_TRAIN[GKG_DAILY_FINAL_PANDAS_TRAIN['Date'] < '2024-01-01']
GKG_DAILY_FINAL_PANDAS_TRAIN_PROC = GKG_DAILY_FINAL_PANDAS_TRAIN.drop("Date", axis = 1)

GKG_DAILY_FINAL_PANDAS_TEST = GKG_DAILY_FINAL_PANDAS.copy()
GKG_DAILY_FINAL_PANDAS_TEST = GKG_DAILY_FINAL_PANDAS_TEST[GKG_DAILY_FINAL_PANDAS_TEST['Date'] >= '2024-01-01']
GKG_DAILY_FINAL_PANDAS_TEST_PROC = GKG_DAILY_FINAL_PANDAS_TEST.drop("Date", axis = 1)

# COMMAND ----------

print("Training Data :", len(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC))
print("Test Data :", len(GKG_DAILY_FINAL_PANDAS_TEST_PROC))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sequence Creator

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import keras

# COMMAND ----------

n_input=12
n_features=5

# COMMAND ----------

scaler = MinMaxScaler()
scaler = scaler.fit(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC)
scaled_data = scaler.transform(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC)
scaled_data_test = scaler.transform(GKG_DAILY_FINAL_PANDAS_TEST_PROC)

# COMMAND ----------

scaler_objective = MinMaxScaler()
scaler_objective = scaler.fit(np.array(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC['Diff']).reshape(-1,1))

# COMMAND ----------

train_generator= TimeseriesGenerator(scaled_data,
                                     scaled_data[:, 4],
                                      n_input,
                                      batch_size=32)

# COMMAND ----------

test_generator = TimeseriesGenerator(scaled_data_test,
                                     scaled_data_test[:, 4],
                                      n_input,
                                      batch_size=32)

# COMMAND ----------

model=Sequential()
model.add(LSTM(200,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
model.add(keras.layers.Dropout(0.20))
model.add(LSTM(100,activation='relu',return_sequences=True))
model.add(keras.layers.Dropout(0.20))
model.add(LSTM(50,activation='relu'))
model.add(Dense(1))

# COMMAND ----------

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='mae')
model.summary()

# COMMAND ----------

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        loss = logs["val_loss"]
        if loss <= self.threshold:
            self.model.stop_training = True

my_callback = MyThresholdCallback(threshold=0.07)

# COMMAND ----------

model.fit(train_generator,validation_data = test_generator, epochs=500, batch_size = 32, callbacks = [my_callback])

# COMMAND ----------

pred_train = (model.predict(train_generator))
pred_test =  (model.predict(test_generator))

# COMMAND ----------

predictions_test = []
predictions_train = []
real_train = []
real_test = []

transformed_train = scaler.inverse_transform(pred_train)
transformed_test = scaler.inverse_transform(pred_test)
transformed_train_real = scaler.inverse_transform(scaled_data[:,4].reshape(-1, 1))
transformed_test_real = scaler.inverse_transform(scaled_data_test[:,4].reshape(-1, 1))

for i in range(0,len(transformed_train)):
    predictions_train.append(transformed_train[i][0])

for i in range(0,len(transformed_test)):
    predictions_test.append(transformed_test[i][0])

for i in range(0,len(transformed_train_real)):
    real_train.append(transformed_train_real[i][0])

for i in range(0,len(transformed_test_real)):
    real_test.append(transformed_test_real[i][0])

# COMMAND ----------

from sklearn.metrics import mean_absolute_error

print(" Training MAE : " , mean_absolute_error(scaled_data[8:,4], pred_train))
print(" Test MAE : " , mean_absolute_error(scaled_data_test[8:,4], pred_test))

print(" Training MAE Real: " , mean_absolute_error(real_train[8:], predictions_train))
print(" Test MAE Real: " , mean_absolute_error(real_test[8:], predictions_test))

# COMMAND ----------

plt.hist(np.array(predictions_train) - np.array(real_train[8:]))

# COMMAND ----------

plt.hist(np.array(predictions_test) - np.array(real_test[8:]))

# COMMAND ----------

plt.scatter(y_train[8:],predictions_train)

# COMMAND ----------

plt.scatter(y_test[8:],predictions_test)

# COMMAND ----------

TRAIN_RESULTS = pd.DataFrame()

TRAIN_RESULTS['test'] = y_train[8:]
TRAIN_RESULTS['preds'] = predictions_train
TRAIN_RESULTS['Date'] = GKG_DAILY_FINAL_PANDAS_TRAIN['Date'][3:]
#PRUEBA['test'] = y_test
TRAIN_RESULTS = TRAIN_RESULTS.sort_values('Date')

plt.plot(TRAIN_RESULTS['Date'], TRAIN_RESULTS['test'], label = "Real Value")
plt.plot(TRAIN_RESULTS['Date'], TRAIN_RESULTS['preds'], label = "Predictions")
plt.legend()
plt.show()

# COMMAND ----------

TEST_RESULTS = pd.DataFrame()

TEST_RESULTS['test'] = y_test[8:]
TEST_RESULTS['preds'] = predictions_test
TEST_RESULTS['Date'] = GKG_DAILY_FINAL_PANDAS_TEST['Date'][3:]
#PRUEBA['test'] = y_test
TEST_RESULTS = TEST_RESULTS.sort_values('Date')

plt.plot(TEST_RESULTS['Date'], TEST_RESULTS['test'], label = "Real Value")
plt.plot(TEST_RESULTS['Date'], TEST_RESULTS['preds'], label = "Predictions")
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Necessary Preproccesing

# COMMAND ----------

# MAGIC %md
# MAGIC * Convert DataFrames to Pandas

# COMMAND ----------

# MAGIC %md
# MAGIC * Separating Objective Variable and Predictors

# COMMAND ----------

# MAGIC %md
# MAGIC # DAILY MODEL

# COMMAND ----------

# MAGIC %md
# MAGIC ## Number of News

# COMMAND ----------

# MAGIC %md
# MAGIC * Train Test Split

# COMMAND ----------

print(len(X_train))
print(len(X_test))

# COMMAND ----------

kernel_ridge = KernelRidge()
alphas = np.random.uniform(low=0, high=10, size=10)
mms = MinMaxScaler()
pipe = Pipeline([('mms',mms),('model', kernel_ridge)])

param_grid = [
        {
        "model__alpha": alphas,
        "model__kernel": ["poly"],
        "model__degree": [2,3,4,5,6,7,8]
        }
    ]

grid_search = GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
preds_test = best_model.predict(X_test)
preds_train = best_model.predict(X_train)

print(best_params)
print("R2 Train: "+str(r2_score(y_train,preds_train)))
print("R2 Test: "+str(r2_score(y_test,preds_test)))
print()
print("Spearman Train: ",stats.spearmanr(preds_train, y_train))
print("Spearman Test: ",stats.spearmanr(preds_test, y_test))

# COMMAND ----------

plt.hist(preds_test - y_test)
plt.xlabel("Errors")
plt.title("Distriuition of Errors Graph")

plt.show()

# COMMAND ----------

plt.scatter(preds_test,y_test)
plt.xlabel("Predictions")
plt.ylabel("Real Values")
plt.title("Predictions vs Real Values Graph")

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST = GKG_DAILY_FINAL_PANDAS_TRAIN.copy().fillna(0).sort_values('Date')
GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'] = best_model.predict(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST.drop(['Date','NegativeNews','ToneNegativeScore'],axis = 1))

# plot lines
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NegativeNews'], label = "Real Value")
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'], label = "Predictions")
plt.legend()
plt.show()

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST = GKG_DAILY_FINAL_PANDAS_TEST.copy().fillna(0).sort_values('Date')
GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'] = best_model.predict(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST.drop(['Date','NegativeNews','ToneNegativeScore'],axis = 1))

# plot lines
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NegativeNews'], label = "Real Value")
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'], label = "Predictions")
plt.legend()
plt.show()

# COMMAND ----------

gbt = GradientBoostingRegressor()
mms = MinMaxScaler()

pipe = Pipeline([('mms',mms),('model', gbt)])

param_grid = [
        {
        "model__n_estimators": [10,12,13,14,15],
        "model__criterion": ['squared_error'],
        "model__max_depth": [2,3,4],
        "model__min_samples_split" : [20,21,22,23,24,25]
        }
    ]
grid_search = GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
preds_test = best_model.predict(X_test)
preds_train = best_model.predict(X_train)

print(best_params)
print(X_features)
print("R2 Train: "+str(r2_score(y_train,preds_train)))
print("R2 Test: "+str(r2_score(y_test,preds_test)))
print()
print("Spearman Train: ",stats.spearmanr(preds_train, y_train))
print("Spearman Test: ",stats.spearmanr(preds_test, y_test))

# COMMAND ----------

plt.hist(preds_test - y_test)
plt.xlabel("Errors")
plt.title("Distriuition of Errors Graph")

plt.show()

# COMMAND ----------

plt.scatter(preds_test,y_test)
plt.xlabel("Predictions")
plt.ylabel("Real Values")
plt.title("Predictions vs Real Values Graph")

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST = GKG_DAILY_FINAL_PANDAS_TRAIN.copy().fillna(0).sort_values('Date')
GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'] = best_model.predict(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST.drop(['Date','NegativeNews','ToneNegativeScore'],axis = 1))

# plot lines
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NegativeNews'], label = "Real Value")
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'], label = "Predictions")
plt.legend()
plt.show()

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST = GKG_DAILY_FINAL_PANDAS_TEST.copy().fillna(0).sort_values('Date')
GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'] = best_model.predict(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST.drop(['Date','NegativeNews','ToneNegativeScore'],axis = 1))

# plot lines
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NegativeNews'], label = "Real Value")
plt.plot(GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['Date'], GKG_DAILY_FINAL_PANDAS_RANFOM_FOREST['NewNewsPrediction'], label = "Predictions")
plt.legend()
plt.show()

# COMMAND ----------


