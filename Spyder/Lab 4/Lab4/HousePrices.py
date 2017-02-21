#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:09:01 2017

@author: atreyamisra
"""

import numpy as np
import pandas as pd
import scipy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from PIL import Image
from scipy.misc import imread
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, LassoCV, Lasso
from sklearn.model_selection import cross_val_score
import xgboost as xgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
#prices.hist()
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
                               
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

#creating a runner for alpha values
def ridgeRegTest(a):
    clf = Ridge(alpha=a)
    clf.fit(X_train, y)
    predict = np.expm1(clf.predict(X_test))

    results = pd.DataFrame({"Id":test.Id, "SalePrice":predict})
    results.to_csv("RidgeResults.csv", index = False)

    results.plot(x='Id', y='SalePrice', kind='line')
    plt.show()
#ridgeRegTest(10.10077)
def lassoTest(a):
    clf = Lasso(alpha=a)
    clf.fit(X_train, y)
    predict = np.expm1(clf.predict(X_test))

    results = pd.DataFrame({"Id":test.Id, "SalePrice":predict})
    results.to_csv("LassoResults.csv", index = False)

    results.plot(x='Id', y='SalePrice', kind='line')
    plt.show()

#checkeing rmse on different values of alpha
alphas = []
for i in range(0,200):
    alphas.append(i*.0000001+10.10076)
    
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
model_ridge = Ridge()

#alphas = [.001, .01, .05, .1, .5, 1, 5, 10, 50]
#first try small range then determine min from graph in order to save running time

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

print cv_ridge.min()

alphas2 = []
zeronorm = []

def lnorm(mini, maxi, increment):
    for i in range(mini,maxi):
        alphas2.append(i*increment)
        alphas3 = i*increment
        #print type(alphas3)
        model_lasso = Lasso(alpha = alphas3).fit(X_train, y)
        coefficients = model_lasso.coef_
        nonzeros = 0;
        for hey in coefficients:
            if (coefficients[hey] != 0.0):
                nonzeros+=1
        zeronorm.append(nonzeros)
    #print zeronorm
    plt.plot(alphas2, zeronorm)
    plt.show()
                
lnorm(1,100,.0001)        
#model_lasso = LassoCV()

#cv_lasso = [rmse_cv(LassoCV(alphas = alpha)).mean()
#            for alpha in alphas2]
#
#cv_lasso = pd.Series(cv_lasso, index = alphas2)
#cv_lasso.plot(title = "Validation - Just Do It2")
#plt.xlabel("alpha")
#plt.ylabel("rmse")
#plt.show()
def findBestLasso(mini,maxi,increment):
    bestAlpha = 0
    currentMin = 1
    for i in range(mini,maxi):
        alphaTest = i*increment
        model_lasso_test = Lasso(alpha = alphaTest).fit(X_train, y)
        score=rmse_cv(model_lasso_test).mean()
        #print score
        print currentMin
        if(score < currentMin):
            bestAlpha=alphaTest
            currentMin=score
    return bestAlpha
bestone = findBestLasso(52550,52650,.00000001)
print 'best lasso alpha is:'
print bestone

#lassoTest(bestone)


#print alphas2
print 'lasso mean:'
print rmse_cv(model_lasso).mean()
print 'lasso min:'
print rmse_cv(model_lasso).min()


model_lasso = LassoCV(alphas = [1, 0.5, 0.1, 0.001, 0.0005]).fit(X_train, y)
print rmse_cv(model_lasso)
print 'real lasso mean:'
print rmse_cv(model_lasso).mean()
print 'real lasso min:'
print rmse_cv(model_lasso).min()







