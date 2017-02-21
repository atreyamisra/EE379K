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
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


weekly = pd.read_csv("Weekly.csv")
print weekly.corr() #print correlation matrix to discover most impactful corss correlations

#weekly.Today.hist()
#plt.title('Histogram of Todays Returns')
#plt.xlabel('Today Return Level')
#plt.ylabel('Frequency')

#weekly.Direction.hist()
#plt.title('Histogram of Direction')
#plt.xlabel('Today Return Level')
#plt.ylabel('Frequency')

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

knn=KMeans(n_clusters=1)
knn.fit(X,y)
print "KNN ACCURACY:"
print knn.score(X,y)
#print "KNN SUMMARY:"
#print pd.DataFrame(zip(X.columns, np.transpose(knn.rotations_))) 
#conmatrixKNN = confusion_matrix(y, knn.predict(X))

#print "QDA CONFUSION MATRIX:"
#print conmatrixKNN

#KNN is definitely the most error prone especially with just one cluster

# -------------NUMBER 5 STARTING HERE ------------------------

default = pd.read_excel("Data/Default.xlsx")
y1 = default.default
X1 = default.drop(['default', 'student'], 1)
log1 = LogisticRegression()
log1 = log1.fit(X1,y1)
print "LOGISTIC ACCURACY FOR 5:"
print log1.score(X1,y1)

def CrossValidate(train, validate):
    yt = train.default
    Xt = train.drop(['default', 'student'], 1)
    yv = validate.default
    logt = LogisticRegression()
    logt = logt.fit(Xt,yt)
    print "LOGISTIC ACCURACY ON OWN TRAINING SET:"
    print logt.score(Xt,yt)
    print "LOGISTIC ACCURACY ON VALIDATE SET:"
    print logt.score(Xt,yv)
    pred = logt.predict(Xt)
    #print mean_squared_error(yv, pred)
    

train1 = default[0:4999]
validate1 = default[5000:9999]
CrossValidate(train1, validate1)

train2 = default[0:2500]
validate2 = default[2501:5001]
CrossValidate(train2, validate2)

train3 = default[0:4000]
validate3 = default[4001:8001]
CrossValidate(train3, validate3)

train4 = default[0:1000]
validate4 = default[1001:2001]
CrossValidate(train4, validate4)

default['StudentBool'] = (default.student == 'Yes').astype(int)

y2 = default.default
X2 = default.drop(['default', 'student'], 1)
log2 = LogisticRegression()
log2 = log1.fit(X2,y2)
print "LOGISTIC ACCURACY FOR 5 WHEN STUDENT DUMMY ADDED:"
print log1.score(X2,y2)

traind1 = default[0:4999]
validated1 = default[5000:9999]
CrossValidate(traind1, validated1)

traind2 = default[0:2500]
validated2 = default[2501:5001]
CrossValidate(traind2, validated2)

traind3 = default[0:4000]
validated3 = default[4001:8001]
CrossValidate(traind3, validated3)

traind4 = default[0:1000]
validated4 = default[1001:2001]
CrossValidate(traind4, validated4)

#----------------Number 6------------------------


np.random.seed(2)
y = np.random.normal(0, 1, 100)
x = np.random.normal(0, 1, 100)

y = x - 2 * x**2 + np.random.normal(0, 1, 100)

plt.scatter(x, y)
plt.show()


from sklearn.cross_validation import LeaveOneOut
for deg in range(1,5):
    print("Degree: "+str(deg))
    fit=np.polyfit(x,y,deg)
    estimator=0
    estimatorStr="y="
    for d in range(deg,-1,-1):
        estimator+=fit[deg-d]*x**d
        estimatorStr+="  "+str(fit[deg-d])+"x^"+str(d)
    
    print(estimatorStr)

    MSESum=0.0
    loo = LeaveOneOut(100)
    for train, test in loo:
        fit=np.polyfit(x[train],y[train],deg)
        estimator=0
        for d in range(deg,-1,-1):
            estimator+=fit[deg-d]*x[test]**d
        MSESum+=(y[test]-estimator)**2.0
    MSESum*=(1.0/len(y))
    print("ERROR IS: "+ str(MSESum[0]))
    print("STATITICAL SIGNIFICANCE:")
    print pd.DataFrame(zip(x, fit))
    

np.random.seed(1012312)

for deg in range(1,5):
    print("Degree: "+str(deg))
    fit=np.polyfit(x,y,deg)
    estimator=0
    estimatorStr="y="
    for d in range(deg,-1,-1):
        estimator+=fit[deg-d]*x**d
        estimatorStr+="  "+str(fit[deg-d])+"x^"+str(d)
    
    #plot and show estimate using all data
    print(estimatorStr)
    #plt.scatter(x,estimator)
    #plt.show()
    
    #Find loocv error
    MSESum=0.0
    loo = LeaveOneOut(100)
    for train, test in loo:
        fit=np.polyfit(x[train],y[train],deg)
        estimator=0
        for d in range(deg,-1,-1):
            estimator+=fit[deg-d]*x[test]**d
        MSESum+=(y[test]-estimator)**2.0
    MSESum*=(1.0/len(y))
    print("ERROR IS: "+ str(MSESum[0]))
    print("STATITICAL SIGNIFICANCE:")
    print pd.DataFrame(zip(x, fit))
    




