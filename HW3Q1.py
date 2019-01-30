# Databricks notebook source
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession

# COMMAND ----------

conf=SparkConf().setAppName("HW3Q1")

# COMMAND ----------

sc = SparkContext.getOrCreate()

# COMMAND ----------

def item_mapper(row):
  splits=row.split(maxsplit=1)
  second_split=splits[1].split(' ')
  temp=[]
  for x in second_split:
    temp.append(float(x))
  return [splits[0],temp]

# COMMAND ----------

items=sc.textFile("/FileStore/tables/itemusermat").map(item_mapper).collect()

# COMMAND ----------

vectors = sc.parallelize(items).map(lambda vec: vec[1])

# COMMAND ----------

clusters = KMeans.train(vectors, 10, maxIterations=100, initializationMode="random", seed=1)

# COMMAND ----------

movies= sc.parallelize(items).map(lambda review: (clusters.predict(review[1]), review[0]))

# COMMAND ----------

clusterAndMovies=movies.reduceByKey(lambda x, y: x + " " + y).mapValues(lambda x: x.split(" ")[:5]).sortByKey().collect()

# COMMAND ----------

movieCluster = sc.parallelize(clusterAndMovies).flatMap(lambda x: [(movieId, x[0]) for movieId in x[1]])

# COMMAND ----------

def movies_mapper(row):
  parts=row.split("::")
  return (parts[0],[parts[1],parts[2]])

# COMMAND ----------

movieInfo=sc.textFile("/FileStore/tables/movies.dat").map(movies_mapper)

# COMMAND ----------

join = movieCluster.join(movieInfo).collect()

# COMMAND ----------

result = sc.parallelize(join).map(lambda tpl: (tpl[1][0],[[tpl[0],tpl[1][1][0],tpl[1][1][1]]])).reduceByKey(lambda x, y: x + y).collect()

# COMMAND ----------

for x in result:
		print("Cluster "+str(x[0]+1))
		for movie in x[1]:
			print("\t"+movie[0]+", "+movie[1]+", "+movie[2])

# COMMAND ----------


