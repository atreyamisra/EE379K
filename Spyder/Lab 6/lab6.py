"""
Created on Mon Feb 27 20:00:24 2017

@author: atreyamisra
"""


import numpy as np
import pandas as pd
import math
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

print "---------------------#2-------------------" 
def createConcentricRings(points, rad1, rad2, var1, var2):
    inner = rad1
    outer = rad2
    thetaIn = np.random.normal(0,1,points)*2*math.pi
    thetaOut = np.random.normal(0,1,points)*2*math.pi
    innerNoiseX = np.random.normal(0, var1, points)                          
    innerNoiseY = np.random.normal(0, var1, points)
    outerNoiseX = np.random.normal(0, var2, points)                          
    outerNoiseY = np.random.normal(0, var2, points)
    xIn = [None] * len(thetaIn)
    yIn = [None] * len(thetaIn)                          
    xOut = [None] * len(thetaOut)
    yOut = [None] * len(thetaOut)   
    for i in range (len(thetaIn)):
        xIn[i] = inner*math.cos(thetaIn[i]) +innerNoiseX[i]
        yIn[i] = inner*math.sin(thetaIn[i]) +innerNoiseY[i]                 
        xOut[i] = outer*math.cos(thetaOut[i]) +outerNoiseX[i]
        yOut[i] = outer*math.sin(thetaOut[i]) +outerNoiseY[i]
    dataInner = np.transpose(([xIn,yIn]))
    dataOuter = np.transpose(([xOut,yOut]))
    data = np.vstack((dataInner, dataOuter))
    plt.figure(figsize=(10,10))
    plt.scatter(xIn,yIn, c = 'g')
    plt.scatter(xOut,yOut, c = 'r')
    plt.show()
    return(data)
datapoints=createConcentricRings(100, 1, 5, 1, 1)

def makeEuclidianDistanceMatrix(data):
    distanceMatrix=np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            distance = ((data[i,0]-data[j,0])**2+(data[i,1]-data[j,1])**2)**.5
            distanceMatrix[i,j]=distance
    return distanceMatrix

def makeSimilarityGraph(data, option, var=1):
    if(option==0):
        print "E neighborhood graph"
        similarityGraph = makeEuclidianDistanceMatrix(data)
        for i in range(len(similarityGraph)):
            for j in range(len(similarityGraph)):
                if(similarityGraph[i, j] < var):
                    similarityGraph[i,j] = 1
                else:
                    similarityGraph[i,j] = 0
    elif(option == 1):
        print "k-nearest neighbor graph - ignore direction"
    elif(option == 2):
        print "k-nearest neighbor graph - mutual neighbors"
    elif(option == 3):
        print "The fully connected graph"
        similarityGraph = np.zeros((len(data),len(data)))
        for i in range(len(similarityGraph)):
            for j in range(len(similarityGraph)):
                sigma = 1
                similarityGraph[i,j]=math.exp(-(((data[i,0] - data [j,0])**2)+((data[i,1] - data[j,1])**2))/(2*sigma**2))
        print similarityGraph
    else:
        print"Invalid input"
    return similarityGraph

g=makeSimilarityGraph(datapoints, 0)
print g
