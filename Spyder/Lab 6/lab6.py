"""
Created on Mon Feb 27 20:00:24 2017

@author: atreyamisra
"""


import numpy as np
import pandas as pd
import math
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import random
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


min = 0
max = 1
means = []
numClusters = 2
centroids = []
clusters = [[] for _ in range(numClusters)]
threshHold = .1

def pythag(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def kMeanLoop(generatedPoints, means):
    converged = False
    while not converged:
        for i in range(0, len(generatedPoints)):
            dists = []
            for j in range(0, numClusters):
                dists.append(pythag(generatedPoints[i], means[j]))
                # print dists
            minIndex = np.argmin(dists)
            (clusters[minIndex]).append(generatedPoints[i])

        # calculate centroids
        centroids[:] = []
        for i in range(0, numClusters):
            thisCluster = clusters[i]
            centroids.append([np.mean([x[0] for x in thisCluster]), np.mean([x[1] for x in thisCluster])])

        diff = 0.0
        converged = True
        for i in range(0, numClusters):
            diff = pythag(centroids[i], means[i])
            # print 'Difference between mean and centroid of ' + str(i) + ': ' + str(diff)
            if diff > threshHold:
                converged = False

        if not converged:
            means = centroids


def kMeans(generatedPoints, k):
    for i in range(0, k):
        potentialMean = [random.uniform(min, max), random.uniform(min, max)]
#         while potentialMean in means:
#             potentialMean = [random.uniform(min, max), random.uniform(min, max)]
        means.append(potentialMean)

    kMeanLoop(generatedPoints, means)
    
def display():
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'y']
    for i in range(0, numClusters):
        x_val = [x[0] for x in clusters[i]]
        y_val = [y[1] for y in clusters[i]]

        plt.scatter(x_val, y_val, color=colors[i], label='Cluster ' + str(i))

    # x_val = [x[0] for x in means]
    # y_val = [y[1] for y in means]
    #
    # plt.scatter(x_val, y_val, marker='x', label='Initial Means')

    x_val = [x[0] for x in centroids]
    y_val = [y[1] for y in centroids]

    plt.scatter(x_val, y_val, marker='x', color='k', s=50, label='Final Means')
    plt.title('Generated Points and Corresponding Clusters')
    plt.legend()
    plt.show()

def makeEuclidianDistanceMatrix(data):
    distanceMatrix=np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            distance = ((data[i,0]-data[j,0])**2+(data[i,1]-data[j,1])**2)**.5
            distanceMatrix[i,j]=distance
    return distanceMatrix

def makeSimilarityGraph(data, option, var):
    if(option==0):
        print "E neighborhood graph"
        similarityGraph = makeEuclidianDistanceMatrix(data)
        for i in range(len(similarityGraph)):
            for j in range(len(similarityGraph)):
                if(similarityGraph[i, j] > var):
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

def spectralCluster(simMatrix, k):
    w = np.zeros((len(simMatrix),len(simMatrix)))
    l = np.zeros((len(simMatrix),len(simMatrix)))
    for i in range(len(simMatrix)):
        sum=0.0
        for j in range(len(simMatrix)):
            sum+=simMatrix[j, i]
        w[i,i]=sum
    for i in range(len(simMatrix)):
        for j in range(len(simMatrix)):
            l[i,j]=w[i,j]-simMatrix[i,j]
    values,e =np.linalg.eig(l)
    e=e[:k]
    y=np.transpose(e)
    kMeans(y, k)
    print "SPECTRAL RESULT:"
    display()

datapoints=createConcentricRings(100, 1, 10, 1, 1)
g=makeSimilarityGraph(datapoints, 0, 1)
spectralCluster(g, 2)
kMeans(g, 2)
print "K-MEANS RESULT:"
display()
#print g

