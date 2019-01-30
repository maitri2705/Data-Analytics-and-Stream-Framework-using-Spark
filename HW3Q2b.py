# Databricks notebook source
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

# COMMAND ----------

def parseLine(line):
    parts = line.split(',')
    label = float(parts[10])
    features = Vectors.dense([float(x) for x in parts[1:10]])
    return LabeledPoint(label, features)

# COMMAND ----------

data = sc.textFile('/FileStore/tables/glass.data').map(parseLine)

# COMMAND ----------

print(data.take(10))

# COMMAND ----------

training, test = data.randomSplit([0.6, 0.4])

# COMMAND ----------

print test.take(10)

# COMMAND ----------

model = NaiveBayes.train(training)

# COMMAND ----------

predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))

# COMMAND ----------

accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()

# COMMAND ----------

print accuracy

# COMMAND ----------


