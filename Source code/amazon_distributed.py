#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.recommendation import ALS

import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import sys
import pandas as pd
import seaborn as sns
from time import time
from pyspark.sql import Row
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit,CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline, PipelineModel
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import lit
from pyspark.ml.feature import IndexToString


# In[2]:


# spark config

spark = SparkSession     .builder     .appName("Amazon")     .config("spark.driver.maxResultSize", "5g")     .config("spark.driver.memory", "5g")     .config("spark.executor.memory", "5g")     .config("spark.master", "local[5]")     .getOrCreate()
# get spark context
sc = spark.sparkContext
sc.setLogLevel("Error")


# In[3]:


data_path = 'C:\\Users\\1\\Documents\\WSU\\big data\\Project'

ratings = spark.read.load(os.path.join(data_path, 'Customer.csv'), format='csv', header=True, inferSchema=True)


# In[8]:


# transform asin and CustomerID alphanumeric string to index using spark StringIndexer function
asinIndexer = StringIndexer(inputCol="ASIN", outputCol="item",handleInvalid='error') # create indexer for asins
userIndexer = StringIndexer(inputCol='CustomerID',outputCol='userid',handleInvalid='error') # create indexer for Customer
asinIndexed = asinIndexer.fit(ratings).transform(ratings) # apply asin indexer
userIndexed = userIndexer.fit(asinIndexed).transform(asinIndexed) # apply Customer indexer
data_indexed = userIndexed.drop('ASIN').drop('CustomerID') # remove old columns with alphanumeric strings


# In[5]:


# 70-30 train-test split
(data_train, data_test) = data_indexed.randomSplit([0.7,0.3])
# cache them in memory across clusters since we access this data frequently 
data_train.cache() 
data_test.cache()


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:



# model
als = ALS(userCol="userid", itemCol="item", ratingCol="rating",coldStartStrategy='drop',nonnegative=False)
     
# evaluator
rmseevaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# parameter grid
paramGrid = ParamGridBuilder()    .addGrid(als.rank, [1, 5, 10,50,70])     .addGrid(als.maxIter, [15])    .addGrid(als.regParam, [0.05, 0.1, 0.5,5])    .build()

# train validation split
tvs = TrainValidationSplit(estimator=als,
                           estimatorParamMaps=paramGrid,
                           evaluator=rmseevaluator,
                           trainRatio=0.8)
# fit model and time

tvsmodel = tvs.fit(data_train)
    
# zip train validation and parameter into one list
paramMap = list(zip(tvsmodel.validationMetrics,tvsmodel.getEstimatorParamMaps()))
paramMax = min(paramMap)
print(paramMax)


# In[ ]:





# In[ ]:


def Customer_List(model, user):
    # Create a dataset with distinct Customers as one column and the asin as another column
    Customer = data_train.select("userid").distinct().withColumn("item", lit(user))

#     # convert index back to original CustomerID 
    userconverter = IndexToString(inputCol="userid", outputCol="List of Customers")
    userString = userconverter.transform(Customer)
    userString.drop("userid").drop("item").show()

# In[ ]:


Customer_List(tvsmodel,241)


# In[ ]:




