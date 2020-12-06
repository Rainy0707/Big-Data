#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import sys
# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.recommendation import ALS

# data science imports
import math
import numpy as np
import pandas as pd

# visualization imports
import seaborn as sns
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[2]:


# spark config
spark = SparkSession     .builder     .appName("movie recommendation")     .config("spark.driver.maxResultSize", "96g")     .config("spark.driver.memory", "96g")     .config("spark.executor.memory", "8g")     .config("spark.master", "local[12]")     .getOrCreate()
# get spark context
sc = spark.sparkContext


# In[3]:


data_path = 'C:\\Users\\1\\Documents\\WSU\\big data\\Project\\Source code\\Data'


# In[4]:


ratings = spark.read.load(os.path.join(data_path, 'Customer.csv'), format='csv', header=True, inferSchema=True)


# In[5]:


ratings.show()


# In[6]:


ratings.describe().show()
ratings.printSchema()


# In[7]:


# transform asin and user alphanumeric string to index using spark StringIndexer function
asinIndexer = StringIndexer(inputCol="ASIN", outputCol="item",handleInvalid='error') # create indexer for asins
userIndexer = StringIndexer(inputCol='CustomerID',outputCol='userid',handleInvalid='error') # create indexer for user
asinIndexed = asinIndexer.fit(ratings).transform(ratings) # apply asin indexer
userIndexed = userIndexer.fit(asinIndexed).transform(asinIndexed) # apply user indexer
df_indexed = userIndexed.drop('ASIN').drop('CustomerID') # remove old columns with alphanumeric strings

# 70-30 train-test split
(df_train, df_test) = df_indexed.randomSplit([0.7,0.3])
# cache them in memory across clusters since we access this data frequently 
df_train.cache() 
df_test.cache()

# Display dataset size
print('Train set size: {}'.format(df_train.count()))
print('Test set size: {}'.format(df_test.count()))

print('Matrix size, percentage of matrix filled and number of distinct users and itmes:')
# calculate percentage of the user-item matrix that is filled
df_train.createOrReplaceTempView('df_train')
spark.sql("""
      SELECT *, 100 * rating/matrix_size AS percentage
        FROM (
          SELECT userid, item, rating, userid * item AS matrix_size
            FROM(
              SELECT COUNT(*) AS rating, COUNT(DISTINCT(item)) AS item, COUNT(DISTINCT(userid)) AS userid
                FROM df_train
                )
            )
""").show()


# In[ ]:


df_viz = df_indexed.sample(False,0.09) # sample a small portion of the dataset for visualization
pdf = df_viz.toPandas() # convert to pandas dataframe


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




