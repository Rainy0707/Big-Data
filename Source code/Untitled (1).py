#!/usr/bin/env python
# coding: utf-8

# In[11]:


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

#get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[10]:


# spark config

# conf = SparkConf().setAppName("AmazonALS").set("spark.executor.memory", "2g")
# sc = SparkContext(conf=conf)
spark = SparkSession \
    .builder \
    .appName("Amazon") \
    .config("spark.driver.maxResultSize", "5g") \
    .config("spark.driver.memory", "5g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.master", "local[5]") \
    .getOrCreate()
# # get spark context
sc = spark.sparkContext
sc.setLogLevel("Error")


# In[3]:


data_path = 'Data'


# In[4]:


ratings = spark.read.load(os.path.join(data_path, 'Customer.csv'), format='csv', header=True, inferSchema=True)


# In[5]:


ratings.show()


# In[6]:


ratings.describe().show()
ratings.printSchema()


# In[8]:


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


# df_viz = df_indexed.sample(False,0.09) # sample a small portion of the dataset for visualization
# pdf = df_viz.toPandas() # convert to pandas dataframe


# In[ ]:


# numuniquser = pdf['userid'].value_counts().count() # to set axis
# numuniqitem = pdf['item'].value_counts().count() # to set axis
# custompal = sns.xkcd_palette(['red', 'orange', 'sandy yellow', 'yellowgreen', 'vibrant green']) # traffic-light style palette
# scplot = sns.lmplot('userid','item',pdf,hue='rating',fit_reg=False,size=10 # use seaborn lmplot to plot user vs item and rating as hue
#            , aspect=2,palette=custompal,scatter_kws={'alpha':0.5})
# axes = scplot.axes
# axes[0,0].set_ylim(0,numuniqitem)  
# axes[0,0].set_xlim(0,numuniquser) 
# scplot
# plt.savefig('scatterplot.png',dpi=50)


# In[ ]:


df_indexed.createOrReplaceTempView('df_ind') # create temp SQL view
# count number of ratings in each category 
ratingcount = spark.sql("""
      SELECT COUNT(rating) as count
      ,rating
      FROM df_ind
      GROUP BY rating
""")
pandas_rc = ratingcount.toPandas()  # convert to pandas
pandas_rc.sort_values('rating',axis=0,inplace=True) # sort 
# pandas_rc.plot(x='rating',y='count',kind='bar',legend=False,color=custompal,figsize=(8,5)) # plot using the traffic-light palette
# plt.savefig('barchart.png',dpi=70) # save to disk
# plt.show()


# In[ ]:


pandas_rc


# In[ ]:


# mean = float(df_train.describe().toPandas()['rating'][1]) # mean
# print('Training set mean: {}'.format(mean))
# print('Test set baseline MSE and RMSE')     
# se_rdd = df_test.rdd.map(lambda x: (x[0]-mean)**2) #  squared error
# row = Row("val") # create row
# se_df = se_rdd.map(row).toDF() # convert to df
# se_df.createOrReplaceTempView('se_df') # create temp SQL view
# baseline = spark.sql('SELECT AVG(val) as MSE,SQRT(AVG(val)) as RMSE  FROM se_df') # calculate MSE and RMSE
# baseline.show()
# baseline_rmse = float(baseline.toPandas()['RMSE'][0])


# In[ ]:


# define overall pipeline 

def als_pipeline(df_train,df_test,trainingdownsampling=0.99):
    """
      Args: 
        df_train: pyspark train dataframe  
        df_test: pyspark test dataframe
        trainingdownsampling: percentage of full training set
      Returns:
        testset_rmse
        baseline_rmse
        wallclock
    """

    # model
    als = ALS(userCol="userid", itemCol="item", ratingCol="rating",coldStartStrategy='drop',nonnegative=False)
    
    # evaluator
    rmseevaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    # parameter grid
    paramGrid = ParamGridBuilder()        .addGrid(als.rank, [1, 5, 10,50,70])         .addGrid(als.maxIter, [15])        .addGrid(als.regParam, [0.05, 0.1, 0.5,5])        .build()

    # train validation split
    tvs = TrainValidationSplit(estimator=als,
                               estimatorParamMaps=paramGrid,
                               evaluator=rmseevaluator,
                               trainRatio=0.8)

    
    # sample, Note : spark sample does is not guaranteed to provide exactly the fraction specified of the total
    training = df_train.sample(False,trainingdownsampling)
    print('Full training set size: {}'.format(df_train.count()))
    print('Downsampled training set size: {} \n'.format(training.count()))
  
    # fit model and time it
    print('Fitting model...')
    startTime = time()
    tvsmodel = tvs.fit(training)
    endTime = time()
    wallclock = ( endTime - startTime )
    
    print('Wall-clock time: {}'.format(wallclock))
    
    print('\n')
    paramMap = list(zip(tvsmodel.validationMetrics,tvsmodel.getEstimatorParamMaps())) # zip validation rmse and selected parameters
    paramMax = min(paramMap)
    print('Best parameters and validation set RMSE:')
    print(paramMax)
    print('\n')
    
    # predict and evaluate test set
    predictions = tvsmodel.transform(df_test)
    testset_rmse = rmseevaluator.evaluate(predictions)
    print('Test set RMSE: {}'.format(testset_rmse))    
    return testset_rmse,wallclock,paramMax


# In[ ]:


# downsamples = [0.01,0.1,0.5,0.8] # list of percentages to downsample training set
rmses = [] 
wallclocks = []
params = []
# # loop through the list and apply the pipeline function, append the results to the above empty lists
# for s in downsamples:
#     print('Fitting als model for {} % of the training set'.format(s*100))
#     test_rmse,wallclock,parammax = als_pipeline(df_train,df_test,trainingdownsampling=s)
#     rmses.append(test_rmse)
#     wallclocks.append(wallclock)
#     params.append(parammax)


# In[ ]:


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
startTime = time()
tvsmodel = tvs.fit(df_train)
endTime = time()
wallclock = ( endTime - startTime )
    
print('Wall-clock time: {}'.format(wallclock))
    
print('\n')
# zip train validation and parameter into one list
paramMap = list(zip(tvsmodel.validationMetrics,tvsmodel.getEstimatorParamMaps()))
paramMax = min(paramMap)
print(paramMax)

# predict and evaluate test set
predictions = tvsmodel.transform(df_test)
testset_rmse = rmseevaluator.evaluate(predictions)
print('Test set RMSE: {}'.format(testset_rmse))


# In[ ]:


# append with full train set
rmses.append(testset_rmse)
wallclocks.append(wallclock)
params.append(paramMax)


# In[ ]:


# getting terms separately since dictionaries are only ordered in Python 3.6 onwards, we are still on Python 3.5
# rank1 = list(params[0][1].values())[0]
# iter1 = list(params[0][1].values())[1]
# reg1 = list(params[0][1].values())[2]

# rank2 = list(params[1][1].values())[1]
# iter2 = list(params[1][1].values())[2]
# reg2 = list(params[1][1].values())[0]

# rank3 = list(params[2][1].values())[0]
# iter3 = list(params[2][1].values())[2]
# reg3 = list(params[2][1].values())[1]

# rank4 = list(params[3][1].values())[1]
# iter4 = list(params[3][1].values())[0]
# reg4 = list(params[3][1].values())[2]

# rank5 = list(params[4][1].values())[0]
# iter5 = list(params[4][1].values())[1]
# reg5 = list(params[4][1].values())[2]


# In[ ]:


# convert to lists
# val_rmse = list(map(lambda x: x[0], params))
# ranks = [rank1,rank2,rank3,rank4,rank5]
# iters = [iter1,iter2,iter3,iter4,iter5]
# regParams = [reg1,reg2,reg3,reg4,reg5]
#downsamples.append(1)


# In[ ]:


# pd_results = pd.DataFrame(
#     {'Downsample percentage': downsamples,
#      'Number of latent factors': ranks,
#      'Maximum number of iterations': iters,
#      'Regularization parameter': regParams,
#      'Wall-clock time': wallclocks,
#      'Validation RMSE': val_rmse,
#      'Test RMSE': rmses
#     })


# In[ ]:


# display dataframe
# pd_results


# In[ ]:


# save to disk
# pd_results.to_csv('results.csv')


# In[ ]:


df_train.show(n=3)


# In[ ]:


from pyspark.sql.functions import lit
from pyspark.ml.feature import IndexToString

def recommendGames(model, user, num_rec):
    # Create a dataset with distinct games as one column and the user of interest as another column
    itemsuser = df_train.select("item").distinct().withColumn("userid", lit(user))
    #itemsuser.show(n=5)

    # filter out games that user has already rated 
    gamesrated = df_train.filter(df_train.userid == user).select("item", "userid")

    # apply trained recommender system
    predictions = model.transform(itemsuser.subtract(gamesrated)).dropna().orderBy("prediction", ascending=False).limit(num_rec).select("item", "prediction")
    predictions.show()
    
    # convert index back to original ASIN 
    converter = IndexToString(inputCol="item", outputCol="originalCategory")
    converted = converter.transform(predictions)
    converted.show()


# In[ ]:



# pick a random user id (696) and display 3 recommendations
recommendGames(tvsmodel,387,3)


# In[ ]:




