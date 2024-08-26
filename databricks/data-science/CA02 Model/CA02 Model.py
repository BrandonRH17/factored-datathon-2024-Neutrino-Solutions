# Databricks notebook source
# MAGIC %md
# MAGIC # Models Generation to Predict Status of CA02 Port
# MAGIC
# MAGIC In this notebook you can find the training process for the model for CO2 Port status predictions. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries
# MAGIC
# MAGIC Loading Libraries to be used in this notebook

# COMMAND ----------

# Loading Libraries for the notebook
from pyspark.sql.functions import *
import datetime
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import keras
import statsmodels.api as sm
import numpy as np
import joblib

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Data
# MAGIC
# MAGIC In the next piece of code we load the GKG dataset, that is the data that we are going to use to create our objective variable and create predictors

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
# MAGIC 4. Filter for locations of interest, In our case as we are focusing on the CA02 port, we would focus on this location: "CA02"

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
.filter(col("LocationCode").isin("CA02")) # FILTER THE NEWS RELATED TO THE LOCATIONS OF THE PORT (CHECK DEFINITION IN THE CELL ABOVE FOR MORE DETAILS IN THE CODES)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eliminating Emotional Charges
# MAGIC
# MAGIC As a logistic company, we can't base our decisions in emotions or a interpretation of a situation with a lot of emotion charge to it. This could easily cause wrong interpretations and could generate serious problems to a company.
# MAGIC
# MAGIC In GKG, with the TONE varible we could make some decissions by extracting the average tone and the polarity. 
# MAGIC
# MAGIC In the GKG codebook, it states that news with high emotional charge can be identify following this rule: The Average Tone is very neutral (close to 0) and the Polarity is High.
# MAGIC
# MAGIC So, now the question that arise is "how close to 0 to considere it neutral" and "what is high in polarity"
# MAGIC * We define that a neutral average tone is going to be between -0.5 and 0.5 
# MAGIC * A high polarity is a value higher than 9. This value was obtained by analyzing percentiles. This 9 is approximately the 85% percentile.
# MAGIC
# MAGIC The following piece of code accomplish the task to eliminate Emotional Charged News.

# COMMAND ----------

GKG_NOT_EMOTIONAL_CHARGE = (GKG_PRINCIPAL_CLEANING
.withColumn("Neutrality", when((col("AverageTone") >= -0.5) & (col("AverageTone") <= 0.5),1).otherwise(0)) # GET NEUTRALITY OF THE NEW (FLAG 1=NEUTRAL, 0=NOT NEUTRAL)
.withColumn("EC", when((col("Neutrality") == 1) & (col('Polarity') >= 9),1).otherwise(0)) #GET EMOTIONAL CHARGED FLAG (1=EMOTIONAL CHARGE,0=NOT EMOTIONAL CHARGED)
.filter("EC == 0") # GET ONLY THE NEWS THAT ARE NOT EMOTIONAL CHARGED
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base Table
# MAGIC
# MAGIC As we are a logistics company. We don't care about all news that are happening in the world. We care about the news that could have effect or impact in our operations. The Themes we are going to include to create a base table is the ones that are going to help us to relate the news to some port activities: PORT, TRANSPORT, SHIPPING, MARITIME, TRADE_PORT, NAVAL_PORT, LOGISTICS_PORT are the THEMES that we are going to use for filtering our news.
# MAGIC
# MAGIC Also, Its important to note that we are interested in News that could have impact in operations in Ports. As we did in the Data Analysis part of the project. We identiy that this news have a negative Average Tone. So, we are going to keep news that
# MAGIC
# MAGIC 1. Are related to ports, ports activities or maritime activity
# MAGIC 2. News that have a negative average tone

# COMMAND ----------

GKG_NEWS_AND_TONE = (
GKG_NOT_EMOTIONAL_CHARGE
.withColumn("BaseNews", when(
((col("THEMES").like("%PORT%")) | (col("THEMES").like("%TRANSPORT%")) | 
(col("THEMES").like("%SHIPPING%")) | (col("THEMES").like("%MARITIME%")) | (col("THEMES").like("%TRADE_PORT%")) | (col("THEMES").like("%NAVAL_PORT%")) | (col("THEMES").like("%LOGISTICS_PORT%"))) & (~col("THEMES").like("%AIRPORT%")),1).otherwise("Not Port Related")) #CREATE A FLAG THAT TELLS US IF THE NEW IS WHITHIN THE THEMES WE WANT TO SELECT
.filter("AverageTone < 0") # FILTER THAT THE AVERAGE TONE OF THE NEWS IS NEGATIVE
.filter("BaseNews == 1")) # KEEP ONLY NEWS THAT ACCOMPLISH THE THEME FILTER

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Objective Variables and Some Predictors
# MAGIC
# MAGIC Our date unit is daily. So we are going to group our data within the Date, and generate some variables we are going to use in the training model part. These variables are:
# MAGIC
# MAGIC 1. Number of Negative News
# MAGIC 2. Average of the Negative Score Tone
# MAGIC 3. Avergae of the Polarity
# MAGIC 4. Maximun Negative Tone of the day
# MAGIC 5. Minimiun Negative Tone of the dat
# MAGIC
# MAGIC
# MAGIC Remeber, our objective is to predict the number of negative news that would have the next day

# COMMAND ----------

GKG_NEWS_AND_TONE_DAILY = (GKG_NEWS_AND_TONE
.groupby("Date").agg(count("Date").alias("NegativeNews"), avg("ToneNegativeScore").alias("ToneNegativeScore"), avg("Polarity").alias("Polarity"),max("ToneNegativeScore").alias("MaxToneNeg"),min("ToneNegativeScore").alias("MinToneNeg"))
) # GROUP OUR DATA BY DATE AND CREATE THE VARIABLES DESCRIBED BELOW

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extra Variable: Themes Related To Port Problems
# MAGIC
# MAGIC This variable would have various categories. We are interested in the count of news that the THEME is related to a category that could cause some port problems. The THEMES we select are:
# MAGIC DISRUPTION, STRIKES, PROTEST, MARITIME_DISASTER, TERRORISM, PIRACY, and CONGESTION

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

# MAGIC %md
# MAGIC ## Generating Table For Model Training
# MAGIC
# MAGIC Now, we are going to join the both tables we have created to combine this information and start with model training.

# COMMAND ----------

GKG_DAILY_FINAL = (GKG_NEWS_AND_TONE_DAILY
.join(VAR_THEMES_LAG_DAILY,"Date","left")
.fillna(0)
.persist()
) # JOINING OUR VARIABLES BY DATE

GKG_DAILY_FINAL.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM
# MAGIC
# MAGIC After various iterations within a lot of techniques like: Random Forest, GBT and Kernel Ridge Regressions. We choose that the best approach to tackle our objetive is to use LSTMS for time series forecasting.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting DataFrame to Pandas
# MAGIC
# MAGIC Converting Spark dataframe to Pandas to use tensorflow for our LSTM model

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL.toPandas() # CONVERTING SPARK DATAFRAME TO PANDAS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Date Adjustments
# MAGIC
# MAGIC We adjust our date column to a datetime column and sort our values in ascending order.

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS['Date'] = pd.to_datetime(GKG_DAILY_FINAL_PANDAS['Date']) # ADJUST DATE TO DATE TIME OBJECT
GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL_PANDAS.sort_values('Date') # SORT VALUES BY DATE

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoCorrelation

# COMMAND ----------

sm.graphics.tsa.plot_acf(GKG_DAILY_FINAL_PANDAS['NegativeNews'])

# COMMAND ----------

# MAGIC %md
# MAGIC By analyzing this graph we are going to use a lag of 7, as it represent a growing autocorrelation as the lag period advance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test for Stationary Time Series.
# MAGIC
# MAGIC Before doing some training. We need to be sure that our objective is Stationary. Non-Stationary Series could be a really big problem for Time Series Forecasting because the mean and standar deviation of the series could change over time and affect the predictibility of the series.
# MAGIC
# MAGIC We are going to use an adfuller test to check this situation

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller

# AD FULLER TEST FOR STATIONARY SERIES
adfuller(GKG_DAILY_FINAL_PANDAS['NegativeNews'], regression = "c", autolag= 'AIC')

# COMMAND ----------

# MAGIC %md
# MAGIC The P value in our series is  0.26427396773673184. Is not lower than 0.05. this means that our series in NOT STATIONARY. So we have to make the serie Stationary to have Better Forecasting Results

# COMMAND ----------

# MAGIC %md
# MAGIC The technique we are going to use to solve this issue is Differencing. This technique uses the difference between the past value of the series to normalize our data. In context of the problem. Our model is going to predict the **difference of negative news expected for the next day**

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS['Diff'] = GKG_DAILY_FINAL_PANDAS['NegativeNews'].diff() # APPLY DIFFERENCING
GKG_DAILY_FINAL_PANDAS = GKG_DAILY_FINAL_PANDAS.dropna() # DROP NULL VALUES 

# COMMAND ----------

# PLOT DIFFERENCING BY DATE
plt.plot(GKG_DAILY_FINAL_PANDAS['Date'], GKG_DAILY_FINAL_PANDAS['Diff'])
plt.xlabel("Date")
plt.ylabel("Differencing")
plt.xticks(rotation=90)
plt.title("Differencing Time Series")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outliers
# MAGIC
# MAGIC To treat outliers we are going to keep values from de 0.02% percentile to the 99% percentile.

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS['Diff'] = np.where(GKG_DAILY_FINAL_PANDAS['Diff'] <= GKG_DAILY_FINAL_PANDAS['Diff'].quantile(0.02),GKG_DAILY_FINAL_PANDAS['Diff'].quantile(0.02), np.where(GKG_DAILY_FINAL_PANDAS['Diff'] >= GKG_DAILY_FINAL_PANDAS['Diff'].quantile(0.99),GKG_DAILY_FINAL_PANDAS['Diff'].quantile(0.99),GKG_DAILY_FINAL_PANDAS['Diff']))

# COMMAND ----------

plt.plot(GKG_DAILY_FINAL_PANDAS['Date'], GKG_DAILY_FINAL_PANDAS['Diff'])
plt.xlabel("Date")
plt.ylabel("Differencing")
plt.xticks(rotation=90)
plt.title("Differencing Time Series")

# COMMAND ----------

# MAGIC %md
# MAGIC By looking at the Graph Above, we have a pretty good insight that our data is now Stationary. Nevertheless, we are going to run the AD FULLER test to be 100% sure

# COMMAND ----------

adfuller(GKG_DAILY_FINAL_PANDAS['Diff'], regression = "c", autolag= 'AIC') # RUN AD FULLER TEST FOR STATIONARY

# COMMAND ----------

# MAGIC %md
# MAGIC Now, are P value is   2.8245834394494916e-13, Very below to 0.05. So now we can start thinking abour Time Series Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and Test
# MAGIC
# MAGIC As we don't have a big set of data. We are going to use all date below 2024 for training and all data from 2024 for testing

# COMMAND ----------

GKG_DAILY_FINAL_PANDAS_TRAIN = GKG_DAILY_FINAL_PANDAS.copy() # COPY DATASET
GKG_DAILY_FINAL_PANDAS_TRAIN = GKG_DAILY_FINAL_PANDAS_TRAIN[GKG_DAILY_FINAL_PANDAS_TRAIN['Date'] < '2024-01-01'] # DATA BELOW 2024
GKG_DAILY_FINAL_PANDAS_TRAIN_PROC = GKG_DAILY_FINAL_PANDAS_TRAIN.drop(["Date"], axis = 1) # DROP DATE OBJECT FOR MODEL AS IS NOT A VARIABLE

GKG_DAILY_FINAL_PANDAS_TEST = GKG_DAILY_FINAL_PANDAS.copy() # COPY DATASET
GKG_DAILY_FINAL_PANDAS_TEST = GKG_DAILY_FINAL_PANDAS_TEST[GKG_DAILY_FINAL_PANDAS_TEST['Date'] >= '2024-01-01'] # DATA FROM 2024
GKG_DAILY_FINAL_PANDAS_TEST_PROC = GKG_DAILY_FINAL_PANDAS_TEST.drop(["Date",], axis = 1) # DROP DATE OBJECT FOR MODEL AS IS NOT A VARIABLE


y_train = GKG_DAILY_FINAL_PANDAS_TRAIN_PROC['Diff'] # GET DIFF VALUES OF TRAINING DATA
y_test = GKG_DAILY_FINAL_PANDAS_TEST_PROC['Diff'] # GET DIFF VALUES OF TEST DATA

# COMMAND ----------

print("Training Data :", len(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC))
print("Test Data :", len(GKG_DAILY_FINAL_PANDAS_TEST_PROC))

# COMMAND ----------

# MAGIC %md
# MAGIC ## IMPORTANT LSTMS SPECIFICATIONS
# MAGIC
# MAGIC These two variables are very important. n_inputs is how many times are we going to lag all our variables for the LSTM and n_features the number of features we are going to use in the model.
# MAGIC
# MAGIC * n_inputs: 8 lags of the data
# MAGIC * n_features: 13 features that we have in our dataset

# COMMAND ----------

n_input=7
n_features=13

# COMMAND ----------

# MAGIC %md
# MAGIC ## Min Max Sacaling
# MAGIC
# MAGIC It's very important that our data is Normalized. Even our objetive variables. LSTMS perform better when our data ir nomalized. In this codes we are going to create two scalers
# MAGIC 1. One Scaler for X variables (All the dataset)
# MAGIC 2. Scaler for Objectiv Variable to adjust to real values

# COMMAND ----------

scaler = MinMaxScaler() #START SCALER
scaler = scaler.fit(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC) # FIT TO TRAINING DATA
scaled_data = scaler.transform(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC) # TRANSFORM TRAINING DATA
scaled_data_test = scaler.transform(GKG_DAILY_FINAL_PANDAS_TEST_PROC) # TRANSFORMT TEST DATA

# COMMAND ----------

scaler_objective = MinMaxScaler() # START SCALER
scaler_objective = scaler_objective.fit(np.array(GKG_DAILY_FINAL_PANDAS_TRAIN_PROC['Diff']).reshape(-1,1)) # FIT TO TRAINING DIFFERENCING

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Series Generators
# MAGIC
# MAGIC This functions helps us to generate the data that LSTSM models understand. The return numpy arrays that have the lags of the variables accumulated and the objective prediction value. 
# MAGIC
# MAGIC To see more about the TimeSeriesGeneration you can check: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator

# COMMAND ----------

train_generator= TimeseriesGenerator(scaled_data, # DATAFRAME
                                     scaled_data[:, 12], # OBJECTIVE
                                      n_input, # NUMBER OF LAGS
                                      batch_size=32) #BATCH SIZE

# COMMAND ----------

test_generator = TimeseriesGenerator(scaled_data_test, # DATAFRAME
                                     scaled_data_test[:, 12], # OBJECTIVE
                                      n_input, # NUMBER OF LAGS
                                      batch_size=32) #BATCH SIZE

# COMMAND ----------

# MAGIC %md
# MAGIC ## LSTM model Architecture
# MAGIC
# MAGIC The model architecture is output of lots of iteratons made to find the best fit. The next pieces of code create this architecture, and compile the model. 
# MAGIC
# MAGIC 1. The architecture is build upon 3 LSTM layers, with 128,64,32 units and all with a ReLu activation function.
# MAGIC 2. The input of the firs layer is (n_input,n_features) or (8,13)
# MAGIC 3. The optimizer we choose is ADAM with a learning rate of 0.0001 and the loss we consider is the best fit to judge the performance of the model is Mean Absolute Error
# MAGIC

# COMMAND ----------

model=Sequential() # CREATE MODEL INSTANCE
model.add(LSTM(128,activation='relu',input_shape=(n_input,n_features),return_sequences=True)) # FIRST LAYER CONSTRUCTOR
model.add(LSTM(64,activation='relu',return_sequences=True)) # SECOND LAYER CONSTRUCTOR
model.add(LSTM(32,activation='relu')) # THIRD LAYER CONSTRUCTOR
model.add(Dense(1)) # OUTPUT LAYER CONSTRUCTOR

# COMMAND ----------

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='mae') # COMPILE MODEL
model.summary() # MODEL SUMMARY

# COMMAND ----------

# MAGIC %md
# MAGIC ## CallBacks
# MAGIC
# MAGIC We use one CallBack to stop the model from overfitting. Our call back stops when the val_loss (loss from test samples) are below a certain trashold. This with the objective to prevent overfitting and test if the model is still improving from certain point.

# COMMAND ----------

# CREATING THRESHOLD CALL BACK CLASS AND FUNCTION
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        loss = logs["val_loss"]
        if loss <= self.threshold:
            self.model.stop_training = True

my_callback = MyThresholdCallback(threshold=0.158) # CREATING CALL BACK

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training

# COMMAND ----------

model.fit(train_generator,validation_data = test_generator, epochs=1000, batch_size = 32, callbacks = [my_callback]) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Predictions for Test and for Training
# MAGIC
# MAGIC In the next part of the code we interact with the model Predict function that let us use what the model learn to predict values. The last part of the code is used to extract this values and put them in a list.

# COMMAND ----------

pred_train = (model.predict(train_generator)) #MODEL PREDICTIONS
pred_test =  (model.predict(test_generator)) #TEST PREDICTIONS

# COMMAND ----------

# EMPTY LISTS FOR STORING DATA
predictions_test = []
predictions_train = []
real_train = []
real_test = []


# IN THIS PART OF THE CODE WE USE THE SCALES WE BUILD TO INVERSE TRANSFORM THE VALUES TO THE ORIGINAL ONES.

transformed_train = scaler_objective.inverse_transform(pred_train)
transformed_test = scaler_objective.inverse_transform(pred_test)
transformed_train_real = scaler_objective.inverse_transform(scaled_data[:,12].reshape(-1, 1))
transformed_test_real = scaler_objective.inverse_transform(scaled_data_test[:,12].reshape(-1, 1))

# FOR LOOP TO EXTRACT THE VALUES TO INSERT THEM INTO THE EMPTY LISTS
for i in range(0,len(transformed_train)):
    predictions_train.append(transformed_train[i][0])

for i in range(0,len(transformed_test)):
    predictions_test.append(transformed_test[i][0])

for i in range(0,len(transformed_train_real)):
    real_train.append(transformed_train_real[i][0])

for i in range(0,len(transformed_test_real)):
    real_test.append(transformed_test_real[i][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## First Metric Mean Absolute Error
# MAGIC
# MAGIC The first Metrics we are going to use is the MAE (Mean Absolute Error). This metrics is te average of the absolute value of the errors (-2 = 2, 2=2) for all the observations. We have to be aware tha averages are sensitive to outliers. a Higher value dosen't mean that the model is totally wrong. 

# COMMAND ----------

from sklearn.metrics import mean_absolute_error

print(" Training MAE : " , mean_absolute_error(scaled_data[7:,12], pred_train))
print(" Test MAE : " , mean_absolute_error(scaled_data_test[7:,12], pred_test))

print(" Training MAE Real: " , mean_absolute_error(real_train[7:], predictions_train))
print(" Test MAE Real: " , mean_absolute_error(real_test[7:], predictions_test))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Errors Histogram
# MAGIC
# MAGIC As I said before MAE is a very good metric to give a quick sight of how the model is performing. Now the erros histogram gives us a more deeply understanding on how are errors are distribuited. For goodness in the model we look for the high concentrations be between the 0's

# COMMAND ----------

plt.hist(np.array(predictions_train) - np.array(real_train[7:]))
plt.xlabel("Errors")
plt.ylabel("Count")
plt.title("Erros in the Training Set Distribuition")

# COMMAND ----------

plt.hist(np.array(predictions_test) - np.array(real_test[7:]))
plt.xlabel("Errors")
plt.ylabel("Count")
plt.title("Erros in the Test Set Distribuition")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predictions vs Real Values of Time
# MAGIC
# MAGIC Now we are going to se in the train and test set how well our predicted line fits within each date

# COMMAND ----------

TRAIN_RESULTS = pd.DataFrame()

TRAIN_RESULTS['test'] = y_train[7:] 
TRAIN_RESULTS['preds'] = predictions_train
TRAIN_RESULTS['Date'] = GKG_DAILY_FINAL_PANDAS_TRAIN['Date'][3:]
#PRUEBA['test'] = y_test
TRAIN_RESULTS = TRAIN_RESULTS.sort_values('Date')

plt.plot(TRAIN_RESULTS['Date'], TRAIN_RESULTS['test'], label = "Real Value")
plt.plot(TRAIN_RESULTS['Date'], TRAIN_RESULTS['preds'], label = "Predictions")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Difference")
plt.xticks(rotation=90)
plt.title("Real vs Predicted Values over Time Train")
plt.show()

# COMMAND ----------

TEST_RESULTS = pd.DataFrame()

TEST_RESULTS['test'] = y_test[7:]
TEST_RESULTS['preds'] = predictions_test
TEST_RESULTS['Date'] = GKG_DAILY_FINAL_PANDAS_TEST['Date'][3:]
#PRUEBA['test'] = y_test
TEST_RESULTS = TEST_RESULTS.sort_values('Date')

plt.plot(TEST_RESULTS['Date'], TEST_RESULTS['test'], label = "Real Value")
plt.plot(TEST_RESULTS['Date'], TEST_RESULTS['preds'], label = "Predictions")
plt.xlabel("Date")
plt.ylabel("Difference")
plt.xticks(rotation=90)
plt.title("Real vs Predicted Values over Time Testing")
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Spearman Correlation
# MAGIC
# MAGIC Last but not least, the Spearman Correlation is a great form on testing the goodnes of a model in sort a different way. The Spearmn correlation measures correlation based on Ranks. This is going to helps us understand how well are we capturing the "Movements" in our time series

# COMMAND ----------

stats.spearmanr(predictions_test, y_test[7:])

# COMMAND ----------

stats.spearmanr(predictions_train, y_train[7:])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving Models and Scalers

# COMMAND ----------

model.save("CA02_model.keras")

# COMMAND ----------

scaler_filename = "scalerCA02.save"
joblib.dump(scaler, scaler_filename) 

# COMMAND ----------

scaler_filename = "scalerCA02_Obj.save"
joblib.dump(scaler_objective, scaler_filename) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perfomance on Prediction Next Day Negative News
# MAGIC
# MAGIC As we can remember, our model predicts "difference" in news. Not the actual news. So, we are going to use our test set to try to predict the Next Day Negative News by using our model that predict the difference. Next pieces of code is all the preprocessing steps need to achive this

# COMMAND ----------

RP_ANALYSIS = GKG_DAILY_FINAL_PANDAS_TEST.copy() #COPY TEST DATASET
RP_ANALYSIS = RP_ANALYSIS.sort_values("Date") # SORT VALUES BY DATE
RP_ANALYSIS = RP_ANALYSIS.drop("Date", axis = 1) # DROP DATE AS IS NOT PART OF PREDICTORS

# COMMAND ----------

scaled_data_real_test = scaler.transform(RP_ANALYSIS) # TRANSFORMS VALUE WITH MIN MAX SCALER

# COMMAND ----------

preds_generator = TimeseriesGenerator(scaled_data_real_test,
                                     scaled_data_real_test[:, 12],
                                      8,
                                      batch_size=32) # GENERATE DATASET FOR TIME SERIES FORECASTING AND PREDICTIONS

# COMMAND ----------

prediction_real_test =scaler_objective.inverse_transform(model.predict(test_generator)) # PREDICT VALUES AND RETURN THEM TO REAL VALUES USING THE SCALER OBJECTIVE SCALER

# COMMAND ----------

NEWS_PREDICTION = GKG_DAILY_FINAL_PANDAS_TEST[7:] # GET VALUES THAT WOULD HAVE A PREDICTION (8 DAY LAG)
NEWS_PREDICTION['DiffPred'] = prediction_real_test # ADD PREDICTIONS OF DIFFERENCE TO DATAFRAME
NEWS_PREDICTION['NextDayNewsPred'] = NEWS_PREDICTION['NegativeNews'] + NEWS_PREDICTION['DiffPred'] # CALCULATE NEXT DAT NEWS
NEWS_PREDICTION['NextDayNewsPredAdjusted'] = NEWS_PREDICTION['NextDayNewsPred'].shift(1) # LAG NEXDAY NEWS PRED TO ANALYZE
NEWS_PREDICTION = NEWS_PREDICTION.dropna() # DROP NULL VALUES (8 DAYS LAG)

# COMMAND ----------

plt.plot(NEWS_PREDICTION['Date'], NEWS_PREDICTION['NegativeNews'], label = "Real Value")
plt.plot(NEWS_PREDICTION['Date'], NEWS_PREDICTION['NextDayNewsPredAdjusted'], label = "Predictions")
plt.xlabel("Date")
plt.ylabel("Difference")
plt.xticks(rotation=90)
plt.title("Real vd Predicted Negative News Values by Date")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC It's pretty good! Now this is the end of the model generating model for the CA02
