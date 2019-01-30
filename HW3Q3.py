# Databricks notebook source
from pyspark.mllib.recommendation import ALS, Rating

# COMMAND ----------

data = sc.textFile("/FileStore/tables/ratings.dat")

# COMMAND ----------

# print data.take(10)

# COMMAND ----------

ratings = data.map(lambda l: l.split('::')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# COMMAND ----------

training, test = ratings.randomSplit([0.6, 0.4])

# COMMAND ----------

rank=10
numIterations=10

# COMMAND ----------

model = ALS.train(training, rank, numIterations)

# COMMAND ----------

testdata = test.map(lambda p: (p[0], p[1]))

# COMMAND ----------

# print testdata.take(10)

# COMMAND ----------

predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

# COMMAND ----------

ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

# COMMAND ----------

MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

# COMMAND ----------

print("Mean Squared Error = " + str(MSE))

# COMMAND ----------


