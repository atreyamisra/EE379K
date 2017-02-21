"""
Created on Mon Feb 20 19:05:15 2017

@author: atreyamisra
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.cluster import KMeans
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


weekly = pd.read_csv("Weekly.csv")
print weekly.corr() #print correlation matrix to discover most impactful corss correlations

weekly.Today.hist()
plt.title('Histogram of Todays Returns')
plt.xlabel('Today Return Level')
plt.ylabel('Frequency')

#weekly.Direction.hist()
plt.title('Histogram of Direction')
plt.xlabel('Today Return Level')
plt.ylabel('Frequency')

#pd.crosstab(weekly.Today, weekly.Direction.astype(bool)).plot(kind='bar')
#plt.title('Todays Return based on market direction')
#plt.xlabel('Market Rating')
#plt.ylabel('Frequency')

y = weekly.Direction
X = weekly.drop(['Direction', 'Today', 'Year'], 1)
#X = pd.read_csv("Weekly.csv", usecols = [Lag1, Lag2, Lag3, Lag4, Lag5, Volume])
#print y.head()
#print X.head()
#print weekly.head()

log = LogisticRegression()
log = log.fit(X,y)
print "LOGISTIC ACCURACY:"
print log.score(X,y)
print "LOGISTIC SUMMARY:"
print pd.DataFrame(zip(X.columns, np.transpose(log.coef_))) #nothing seems to be correlated

conmatrixLogistic = confusion_matrix(y, log.predict(X))

print "LOGISTIC CONFUSION MATRIX:"
print conmatrixLogistic


lda=LDA()
lda.fit(X,y)
print "LDA ACCURACY:"
print lda.score(X,y)
print "LDA SUMMARY:"
print pd.DataFrame(zip(X.columns, np.transpose(lda.coef_))) 
conmatrixLDA = confusion_matrix(y, lda.predict(X))

print "LDA CONFUSION MATRIX:"
print conmatrixLDA


qda=QDA()
qda.fit(X,y)
print "QDA ACCURACY:"
print qda.score(X,y)
print "QDA SUMMARY:"
print pd.DataFrame(zip(X.columns, np.transpose(qda.rotations_))) 
conmatrixQDA = confusion_matrix(y, qda.predict(X))

print "QDA CONFUSION MATRIX:"
print conmatrixQDA

knn=KMeans(n_clusters=2)
knn.fit(X,y)
print "KNN ACCURACY:"
print knn.score(X,y)
#print "KNN SUMMARY:"
#print pd.DataFrame(zip(X.columns, np.transpose(knn.rotations_))) 
#conmatrixKNN = confusion_matrix(y, knn.predict(X))

#print "QDA CONFUSION MATRIX:"
#print conmatrixKNN







