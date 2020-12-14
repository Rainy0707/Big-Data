#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD


# In[2]:


amazon_ratings = pd.read_csv('C:\\Users\\1\\Documents\\WSU\\big data\\Project\\Source Code\\Data\\Customer.CSV')
amazon_ratings = amazon_ratings.dropna()
amazon_ratings


# In[3]:


if __name__=="__main__":
    
    if(len(sys.argv) != 2):
        print("USAGE: python AmazonALS.py ProductID")
        sys.exit(1)
        
    i = sys.argv[1]
    popular_products = pd.DataFrame(amazon_ratings.groupby('ASIN')['rating'].count())
    most_popular = popular_products.sort_values('rating', ascending=False)
    amazon_ratings1 = amazon_ratings.head(10000)
    ratings_utility_matrix = amazon_ratings1.pivot_table(values='rating', index='CustomerID', columns='ASIN', fill_value=0)
    ratings_utility_matrix.head()
    ratings_utility_matrix.shape
    X = ratings_utility_matrix.T
    X.head()
    X.shape
    X1 = X
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    decomposed_matrix.shape
    correlation_matrix = np.corrcoef(decomposed_matrix)
    product_names = list(X.index)
    if i in product_names:
        product_ID = product_names.index(i)
        correlation_product_ID = correlation_matrix[product_ID]
        Recommend = list(X.index[correlation_product_ID > 0.90])
        # Removes the item already bought by the customer
        Recommend.remove(i) 
        print("\n")
        print(Recommend[0:4])
        print("\n")
    else:
        print("product ID not found")
            

