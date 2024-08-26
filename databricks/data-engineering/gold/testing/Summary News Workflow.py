# Databricks notebook source
from transformers import pipeline
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import datetime
from delta.tables import DeltaTable

# COMMAND ----------

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# COMMAND ----------

gkg_scraped = spark.sql("SELECT * FROM SILVER.S_GDELT_GKG_SCRAPING")

# COMMAND ----------

NEWS_RELATED_TO_PORT_PROBLEMS_TPR = (
gkg_scraped
.filter(col("Date") >= yesterday)
.filter(col("AverageTone") < 0)
.withColumn("NewsWithTINFA", when(col("THEMES").like("%TRANSPORT_INFRASTRUCTURE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRANSPORT INFRASTRCTURE
.withColumn("NewsWithTRADE", when(col("THEMES").like("%TRADE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRADE 
.withColumn("NewsWithME", when(col("THEMES").like("%MACROECONOMIC%"),1)) # FLAG THAT THE NEWS HAS THE THEME MACROECONOMICS
.withColumn("NewsWithPS", when(col("THEMES").like("%PUBLIC_SECTOR%"),1)) # FLAG THAT THE NEWS HAS THE THEME PUBLIC SECTOR
.withColumn("NewsWithMI", when(col("THEMES").like("%MARITIME_INCIDENT%"),1)) # FLAG THAT THE NEWS HAS THE THEME MARITIME_INCIDIENT
.fillna(0) # FILL WITH 0 THE COLUMNS WHERE OCURRENCES ARE INEXISTENT
.filter(~col("contenturl").like('%Client Error%'))
.filter(~col("contenturl").like('%OK XID%'))
.filter("is_ruta_transpacifica == 1")
.withColumn("Total", col("NewsWithTINFA")+col("NewsWithTRADE")+col("NewsWithME") + col("NewsWithPS") + col("NewsWithMI"))
.withColumn('Rank', row_number().over(Window.partitionBy("Date").orderBy(col("Total").desc())))
.filter("Rank <= 3") # TOTAL
)

NEWS_RELATED_TO_PORT_PROBLEMS_TAR = (
gkg_scraped
.filter(col("Date") >= '2024-08-15')
.filter(col("AverageTone") < 0)
.withColumn("NewsWithTINFA", when(col("THEMES").like("%TRANSPORT_INFRASTRUCTURE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRANSPORT INFRASTRCTURE
.withColumn("NewsWithTRADE", when(col("THEMES").like("%TRADE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRADE 
.withColumn("NewsWithME", when(col("THEMES").like("%MACROECONOMIC%"),1)) # FLAG THAT THE NEWS HAS THE THEME MACROECONOMICS
.withColumn("NewsWithPS", when(col("THEMES").like("%PUBLIC_SECTOR%"),1)) # FLAG THAT THE NEWS HAS THE THEME PUBLIC SECTOR
.withColumn("NewsWithMI", when(col("THEMES").like("%MARITIME_INCIDENT%"),1)) # FLAG THAT THE NEWS HAS THE THEME MARITIME_INCIDIENT
.fillna(0) # FILL WITH 0 THE COLUMNS WHERE OCURRENCES ARE INEXISTENT
.filter(~col("contenturl").like('%Client Error%'))
.filter(~col("contenturl").like('%OK XID%'))
.filter("is_ruta_transatlantica == 1")
.withColumn("Total", col("NewsWithTINFA")+col("NewsWithTRADE")+col("NewsWithME") + col("NewsWithPS") + col("NewsWithMI"))
.withColumn('Rank', row_number().over(Window.partitionBy("Date").orderBy(col("Total").desc())))
.filter("Rank <= 3") # TOTAL
)

# COMMAND ----------

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# COMMAND ----------

GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TPR = NEWS_RELATED_TO_PORT_PROBLEMS_TPR.select('Date','Rank','contenturl').toPandas()
GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TAR = NEWS_RELATED_TO_PORT_PROBLEMS_TAR.select('Date','Rank','contenturl').toPandas()

# COMMAND ----------

content_list_TPR = GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TPR['contenturl'].to_list()
content_list_TAR = GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TAR['contenturl'].to_list()

# COMMAND ----------

content_list_cleaned_TPR = []

for i in range(0,len(content_list_TPR)):
    content_list_cleaned_TPR.append(content_list_TPR[i][200:3500])

content_list_cleaned_TAR = []

for i in range(0,len(content_list_TAR)):
    content_list_cleaned_TAR.append(content_list_TAR[i][200:3500])

# COMMAND ----------

resume_content_TPR = []

for i in range(0,len(content_list_cleaned_TPR)):
    text = summarizer(content_list_cleaned_TPR[i],min_length=30, do_sample=False)[0]['summary_text']
    print(text)
    resume_content_TPR.append(text)
    print(i)

# COMMAND ----------

len(resume_content_TPR)

# COMMAND ----------

resume_content_TAR = []

for i in range(0,len(content_list_cleaned_TAR)):
    text = summarizer(content_list_cleaned_TAR[i],min_length=30, do_sample=False)[0]['summary_text']
    print(text)
    resume_content_TAR.append(text)
    print(i)

# COMMAND ----------

GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TPR['Summary'] = resume_content_TPR
GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TAR['Summary'] = resume_content_TAR

# COMMAND ----------

GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TPR['is_ruta_transpacifica'] = 1
GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TAR['is_ruta_transatlantica'] = 1

# COMMAND ----------

SPARK_GKG_NEWS_TPR = spark.createDataFrame(GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TPR)
SPARK_GKG_NEWS_TAR = spark.createDataFrame(GKG_2024_MOST_IMPORTANT_NEWS_PER_DAY_P_TAR)

# COMMAND ----------

NEWS_RELATED_TO_PORT_PROBLEMS_TPR_SUMMARIZED = (
gkg_scraped
.filter(col("Date") >= '2024-08-15')
.filter(col("AverageTone") < 0)
.withColumn("NewsWithTINFA", when(col("THEMES").like("%TRANSPORT_INFRASTRUCTURE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRANSPORT INFRASTRCTURE
.withColumn("NewsWithTRADE", when(col("THEMES").like("%TRADE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRADE 
.withColumn("NewsWithME", when(col("THEMES").like("%MACROECONOMIC%"),1)) # FLAG THAT THE NEWS HAS THE THEME MACROECONOMICS
.withColumn("NewsWithPS", when(col("THEMES").like("%PUBLIC_SECTOR%"),1)) # FLAG THAT THE NEWS HAS THE THEME PUBLIC SECTOR
.withColumn("NewsWithMI", when(col("THEMES").like("%MARITIME_INCIDENT%"),1)) # FLAG THAT THE NEWS HAS THE THEME MARITIME_INCIDIENT
.fillna(0) # FILL WITH 0 THE COLUMNS WHERE OCURRENCES ARE INEXISTENT
.filter(~col("contenturl").like('%Client Error%'))
.filter(~col("contenturl").like('%OK XID%'))
.filter("is_ruta_transpacifica == 1")
.withColumn("Total", col("NewsWithTINFA")+col("NewsWithTRADE")+col("NewsWithME") + col("NewsWithPS") + col("NewsWithMI"))
.withColumn('Rank', row_number().over(Window.partitionBy("Date").orderBy(col("Total").desc())))
.join(SPARK_GKG_NEWS_TPR.drop('contenturl'),['Date','Rank','is_ruta_transpacifica'],'left')
.select('DATE', 'EXTRACTION_DATE', 'COUNTRYCODE', 'LOCATIONCODE', 'AVERAGETONE', 'IS_RUTA_TRANSPACIFICA', 'IS_RUTA_TRANSATLANTICA', 'IS_RUTA_DEL_CABO_DE_BUENA_ESPERANZA', 'SOURCEURLS', 'SUMMARY')
)

NEWS_RELATED_TO_PORT_PROBLEMS_TAR_SUMMARIZED = (
gkg_scraped
.filter(col("Date") >= '2024-08-15')
.filter(col("AverageTone") < 0)
.withColumn("NewsWithTINFA", when(col("THEMES").like("%TRANSPORT_INFRASTRUCTURE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRANSPORT INFRASTRCTURE
.withColumn("NewsWithTRADE", when(col("THEMES").like("%TRADE%"),1)) # FLAG THAT THE NEWS HAS THE THEME TRADE 
.withColumn("NewsWithME", when(col("THEMES").like("%MACROECONOMIC%"),1)) # FLAG THAT THE NEWS HAS THE THEME MACROECONOMICS
.withColumn("NewsWithPS", when(col("THEMES").like("%PUBLIC_SECTOR%"),1)) # FLAG THAT THE NEWS HAS THE THEME PUBLIC SECTOR
.withColumn("NewsWithMI", when(col("THEMES").like("%MARITIME_INCIDENT%"),1)) # FLAG THAT THE NEWS HAS THE THEME MARITIME_INCIDIENT
.fillna(0) # FILL WITH 0 THE COLUMNS WHERE OCURRENCES ARE INEXISTENT
.filter(~col("contenturl").like('%Client Error%'))
.filter(~col("contenturl").like('%OK XID%'))
.filter("is_ruta_transatlantica == 1")
.withColumn("Total", col("NewsWithTINFA")+col("NewsWithTRADE")+col("NewsWithME") + col("NewsWithPS") + col("NewsWithMI"))
.withColumn('Rank', row_number().over(Window.partitionBy("Date").orderBy(col("Total").desc())))
.join(SPARK_GKG_NEWS_TAR.drop('contenturl'),['Date','Rank','is_ruta_transatlantica'],'left')
.select('DATE', 'EXTRACTION_DATE', 'COUNTRYCODE', 'LOCATIONCODE', 'AVERAGETONE', 'IS_RUTA_TRANSPACIFICA', 'IS_RUTA_TRANSATLANTICA', 'IS_RUTA_DEL_CABO_DE_BUENA_ESPERANZA', 'SOURCEURLS', 'SUMMARY')
)

FINAL_DATASET_NEWS = (
NEWS_RELATED_TO_PORT_PROBLEMS_TPR_SUMMARIZED.union(NEWS_RELATED_TO_PORT_PROBLEMS_TAR_SUMMARIZED)
)

# COMMAND ----------

table_name = "gold.g_gdelt_gkg_weights_detail_report"
delta_table_path = f"s3://databricks-workspace-stack-e63e7-bucket/unity-catalog/2600119076103476/gold/gdelt/g_gdelt_gkg_weights_detail_report/"  # O especifica una ruta personalizada si usas S3 o DBFS

# COMMAND ----------

# Filtrar los registros donde el año de extraction_date sea 2024
if DeltaTable.isDeltaTable(spark, delta_table_path):
    # Si la tabla existe, inserta los datos
    print("La tabla existe, insertando los datos")
    delta_table = DeltaTable.forPath(spark, delta_table_path)
    delta_table.alias("tgt").merge(
        source=FINAL_DATASET_NEWS.alias("src"),
        condition="tgt.DATE = src.DATE AND "
        "tgt.Total = src.Total AND "
        "tgt.LocationCode = src.LocationCode AND "
        "tgt.NumberOfNews = src.NumberOfNews"  
    ).whenNotMatchedInsertAll().execute()
else:
    # Si la tabla no existe, créala e inserta los datos
    print("La tabla no existe, creando la tabla y insertando los datos")
    FINAL_DATASET_NEWS.write.format("delta").mode("overwrite").save(delta_table_path)
    print("Se ha escrito en formato delta.")
    spark.sql(f"CREATE TABLE {table_name} USING DELTA LOCATION '{delta_table_path}'")

# COMMAND ----------


