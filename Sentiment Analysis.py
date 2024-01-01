#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark 
findspark.init()


# In[2]:


get_ipython().system('pip install findspark')


# In[3]:


get_ipython().system('pip install pyspark')


# In[5]:


from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.ml.clustering import KMeans
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


# In[6]:


spark = SparkSession     .builder     .appName("Introdution to Spark DataFrame")     .config("spark.some.config.option", "some-value")     .getOrCreate()


# In[7]:


purchaseSchema = StructType([
    StructField("Date", DateType(), True),
    StructField("Time", StringType(), True),
    StructField("City", StringType(), True),
    StructField("Item", StringType(), True),
    StructField("Total", FloatType(), True),
    StructField("Payment", StringType(), True),
])    


# In[8]:


purchaseDataframe = spark.read.csv(
    "dataset/purchases.csv", 
    header=True, schema=purchaseSchema, sep="\t")
purchaseDataframe.show(3)


# In[9]:


num_rows = purchaseDataframe.count()
print("number of rows: ", num_rows)
purchaseDataframe.printSchema()
purchaseDataframe.describe('Total').show()


# In[14]:


newDataframe = purchaseDataframe.select(purchaseDataframe['City'], 
                                              purchaseDataframe['Total'])
newDataframe.show(3);


# In[ ]:





# In[12]:


purchaseDataframe.filter(purchaseDataframe['Total'] > 200).show(3)


# In[13]:


sortedByCity = purchaseDataframe.orderBy('City').show(4)


# In[15]:


numTransactionEachCity = purchaseDataframe.groupBy("City").count()
numTransactionEachCity.show(5)


# In[15]:


from pyspark.sql.functions import monotonically_increasing_id

newPurchasedDataframe = purchaseDataframe.withColumn(
    "index", monotonically_increasing_id())
newPurchasedDataframe.show(7)


# In[17]:


dataRow2ColumnTotal = newPurchasedDataframe.filter(newPurchasedDataframe['index']==2).select('Total')
dataRow2ColumnTotal.show()


# In[17]:


purchaseDataframe.createOrReplaceTempView("purchaseSql")
anotherNewDataframe = spark.sql("SELECT Total, Payment FROM purchaseSql")
anotherNewDataframe.show(3)


# In[18]:


orderByCity = spark.sql("SELECT * FROM purchaseSql ORDER BY City")
orderByCity.show(5)


# In[20]:


filterAndSortWithSQL = spark.sql("SELECT * FROM purchaseSql WHERE Total>200 ORDER BY Payment")
filterAndSortWithSQL.show(4)


# In[20]:


flightSchema = StructType([
  StructField("DayofMonth", IntegerType(), False),
  StructField("DayOfWeek", IntegerType(), False),
  StructField("Carrier", StringType(), False),
  StructField("OriginAirportID", IntegerType(), False),
  StructField("DestAirportID", IntegerType(), False),
  StructField("DepDelay", IntegerType(), False),
  StructField("ArrDelay", IntegerType(), False),
])


# In[21]:


flights = spark.read.csv('dataset/raw-flight-data.csv', 
                         schema=flightSchema, header=True)
flights.show(5)


# In[33]:


airportSchema = StructType([
  StructField("airport_id", IntegerType(), False),
  StructField("city", StringType(), False),
  StructField("state", StringType(), False),
  StructField("name", StringType(), False),
])

airports = spark.read.csv('dataset/airports.csv', header=True, 
                          schema=airportSchema)
airports.show(2)


# In[34]:


flightsByOrigin = flights.join(airports,
                               flights.OriginAirportID == 
                               airports.airport_id).groupBy("city").count()
flightsByOrigin.show(3)


# In[35]:


n1 = flights.count()
print("number of original data rows: ", n1)
#count the number of data rows after deleting duplicated data
n2 = flights.dropDuplicates().count()
print("number of data rows after deleting duplicated data: ", n2)
n3 = n1 - n2
print("number of duplicated data: ", n3)


# In[39]:


meanArrDelay = flights.groupBy().avg("ArrDelay").take(1)[0][0]
print("mean ArrDelay: ", meanArrDelay)
meanDepDelay = flights.groupBy().avg("DepDelay").take(1)[0][0]
print("mean DepDelay: ", meanDepDelay)
#drop duplicated data and fill missing data with mean value
flightsCleanData=flights.fillna(
    {'ArrDelay': meanArrDelay, 'DepDelay': meanDepDelay})
#just for experiment
flights.groupBy().avg("ArrDelay").show()


# In[40]:


flightsCleanData.describe('DepDelay','ArrDelay').show()


# In[41]:


correlation = flightsCleanData.corr('DepDelay', 'ArrDelay')
print("correlation between departure delay and arrival delay: ", 
      correlation)


# In[42]:


flightSchema = StructType([
  StructField("DayofMonth", IntegerType(), False),
  StructField("DayOfWeek", IntegerType(), False),
  StructField("Carrier", StringType(), False),
  StructField("OriginAirportID", IntegerType(), False),
  StructField("DestAirportID", IntegerType(), False),
  StructField("DepDelay", IntegerType(), False),
  StructField("ArrDelay", IntegerType(), False),
])
#read csv data with our defined schema
flightDataFrame = spark.read.csv('dataset/flights.csv', 
                                 schema=flightSchema, header=True)
flightDataFrame.show(3)


# In[43]:


data = flightDataFrame.select("DayofMonth", "DayOfWeek", 
                              "OriginAirportID", "DestAirportID", 
                              "DepDelay", "ArrDelay")
data.show(3)


# In[44]:


dividedData = data.randomSplit([0.7, 0.3]) 
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)


# In[52]:


assembler = VectorAssembler(inputCols = [
    "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", 
    "DepDelay"], outputCol="features")
trainingDataFinal = assembler.transform(trainingData).select(
    col("features"), (col("ArrDelay").cast("Int").alias("label")))
trainingDataFinal.show(truncate=False , n=3)


# In[53]:


algoritma = LinearRegression(
    labelCol="label",featuresCol="features", 
    maxIter=10, regParam=0.3)
#train the model
model = algoritma.fit(trainingDataFinal)
print ("Regression model is trained!")


# In[54]:


testingDataFinal = assembler.transform(
    testingData).select(
    col("features"), (col("ArrDelay")).cast("Int").alias("trueLabel"))
testingDataFinal.show(truncate=False, n=2)


# In[55]:


prediction = model.transform(testingDataFinal)
#show some prediction results
prediction.show(3)


# In[56]:


from pyspark.ml.evaluation import RegressionEvaluator

#define our evaluator
evaluator = RegressionEvaluator(
    labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
#calculate RMSE of our trained model
rmse = evaluator.evaluate(prediction)
print ("Root Mean Square Error (RMSE):", rmse)


# In[57]:


csv = spark.read.csv('dataset/flights.csv', schema=flightSchema, header=True)
csv.show(3)


# In[58]:


data = csv.select(
    "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", 
    "DepDelay", ((col("ArrDelay") > 15).cast("Int").alias("Late")))
data.show(3)


# In[59]:


dividedData = data.randomSplit([0.7, 0.3]) 
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)


# In[60]:


assembler = VectorAssembler(inputCols = [
    "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", 
    "DepDelay"], outputCol="features")
trainingDataFinal = assembler.transform(
    trainingData).select(col("features"), col("Late").alias("label"))
trainingDataFinal.show(truncate=False, n=2)


# In[63]:


classifier = LogisticRegression(
    labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)
#train our classifier
model = classifier.fit(trainingDataFinal)
print ("Classifier model is trained!")


# In[64]:


testingDataFinal = assembler.transform(
    testingData).select(col("features"), col("Late").alias("trueLabel"))
testingDataFinal.show(3)


# In[65]:


prediction = model.transform(testingDataFinal)
predictionFinal = prediction.select(
    "features", "prediction", "probability", "trueLabel")
predictionFinal.show(truncate=False, n=3)
prediction.show(truncate=False, n=3)


# In[66]:


correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['trueLabel']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)


# In[67]:


from pyspark.ml.classification import RandomForestClassifier

model2 = RandomForestClassifier(
    numTrees=3, maxDepth=5, seed=42, labelCol="label",featuresCol="features")
model2 = model2.fit(trainingDataFinal)
print ("Model is trained!")


# In[68]:


prediction = model2.transform(testingDataFinal)
predictionFinal = prediction.select(
    "features", "prediction", "probability", "trueLabel")
predictionFinal.show(truncate=False, n=3)
correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['trueLabel']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", 
      totalData, ", accuracy", correctPrediction/totalData)


# In[69]:


tweets_csv = spark.read.csv('dataset/tweets.csv', inferSchema=True, header=True)
tweets_csv.show(truncate=False, n=3)


# In[70]:


data = tweets_csv.select("SentimentText", col("Sentiment").cast("Int").alias("label"))
data.show(truncate = False,n=5)


# In[71]:


dividedData = data.randomSplit([0.7, 0.3]) 
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)


# In[74]:


tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
tokenizedTrain = tokenizer.transform(trainingData)
tokenizedTrain.show(truncate=False, n=5)


# In[75]:


swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), 
                       outputCol="MeaningfulWords")
SwRemovedTrain = swr.transform(tokenizedTrain)
SwRemovedTrain.show(truncate=False, n=5)


# In[76]:


hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
numericTrainData = hashTF.transform(SwRemovedTrain).select(
    'label', 'MeaningfulWords', 'features')
numericTrainData.show(truncate=False, n=3)


# In[77]:


lr = LogisticRegression(labelCol="label", featuresCol="features", 
                        maxIter=10, regParam=0.01)
model = lr.fit(numericTrainData)
print ("Training is done!")


# In[78]:


tokenizedTest = tokenizer.transform(testingData)
SwRemovedTest = swr.transform(tokenizedTest)
numericTest = hashTF.transform(SwRemovedTest).select(
    'Label', 'MeaningfulWords', 'features')
numericTest.show(truncate=False, n=2)


# In[79]:


prediction = model.transform(numericTest)
predictionFinal = prediction.select(
    "MeaningfulWords", "prediction", "Label")
predictionFinal.show(n=4, truncate = False)
correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['Label']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)


# In[81]:


customers = spark.read.csv(
    'dataset/customers.csv', inferSchema=True, header=True)
customers.show(3)


# In[82]:


assembler = VectorAssembler(inputCols = [
    "Age", "MaritalStatus", "IncomeRange", "Gender", "TotalChildren", 
    "ChildrenAtHome", "Education", "Occupation", "HomeOwner", "Cars"], 
                            outputCol="features")
data = assembler.transform(customers).select('CustomerName', 'features')
data.show(truncate = False, n=3)


# In[83]:


kmeans = KMeans(
    featuresCol=assembler.getOutputCol(), 
    predictionCol="cluster", k=5)
model = kmeans.fit(data)
print ("Model is successfully trained!")


# In[84]:


centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[85]:


prediction = model.transform(data)#cluster given data
prediction.groupBy("cluster").count().orderBy("cluster").show()#count members in each cluster
prediction.select('CustomerName', 'cluster').show(5)#show several clustered data


# In[87]:


ratings = spark.read.csv('dataset/ratings.csv', inferSchema=True, header=True)
movies = spark.read.csv('dataset/movies.csv', inferSchema=True, header=True)
#merge "movies" and "ratings" dataFrame based on "movieId"
ratings.join(movies, "movieId").show(3)


# In[88]:


data = ratings.select("userId", "movieId", "rating")
#divide data, 70% for training and 30% for testing
splits = data.randomSplit([0.7, 0.3])
train = splits[0].withColumnRenamed("rating", "label")
test = splits[1].withColumnRenamed("rating", "trueLabel")
#calculate number of rows
train_rows = train.count()
test_rows = test.count()
print ("number of training data rows:", train_rows, 
       ", number of testing data rows:", test_rows)


# In[89]:


als = ALS(maxIter=19, regParam=0.01, userCol="userId", 
          itemCol="movieId", ratingCol="label")
#train our ALS model
model = als.fit(train)
print("Training is done!")


# In[90]:


prediction = model.transform(test)
print("testing is done!")


# In[91]:


prediction.join(movies, "movieId").select(
    "userId", "title", "prediction", "trueLabel").show(n=10, truncate=False)


# In[93]:


evaluator = RegressionEvaluator(
    labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print ("Root Mean Square Error (RMSE):", rmse)


# In[94]:


prediction.count()
a = prediction.count()
print("number of original data rows: ", a)
#drop rows with any missing data
cleanPred = prediction.dropna(how="any", subset=["prediction"])
b = cleanPred.count()
print("number of rows after dropping data with missing value: ", b)
print("number of missing data: ", a-b)


# In[95]:


rmse = evaluator.evaluate(cleanPred)
print ("Root Mean Square Error (RMSE):", rmse)


# In[ ]:




