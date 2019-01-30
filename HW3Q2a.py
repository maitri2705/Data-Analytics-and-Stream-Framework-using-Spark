# Databricks notebook source
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.util import MLUtils

# COMMAND ----------

def parseLine(line):
    parts = line.split(',')
    label = int(parts[10])
    features = Vectors.dense([float(x) for x in parts[1:10]])
    return LabeledPoint(label, features)

# COMMAND ----------

data = sc.textFile('/FileStore/tables/glass.data').map(parseLine)

# COMMAND ----------

training, test = data.randomSplit([0.6, 0.4])

# COMMAND ----------

print training.take(10)

# COMMAND ----------

model = DecisionTree.trainClassifier(training, numClasses=8, categoricalFeaturesInfo={},impurity='gini', maxDepth=4, maxBins=32)

# COMMAND ----------

predictions = model.predict(test.map(lambda x: x.features))

# COMMAND ----------

labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

# COMMAND ----------

accuracy = 1.0*labelsAndPredictions.filter(lambda lp: lp[0]==lp[1]).count() /test.count()

# COMMAND ----------

print accuracy

# COMMAND ----------


