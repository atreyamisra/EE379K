import pandas as pd
import numpy as np

# Determine the nan mean of corresponding feature
def getMean(s):
    meanS=np.nanmean(s)
    return s.apply(replaceValues, mean=meanS)


# Replace nan values with mean if necessary
def replaceValues(val, mean):
    if pd.isnull(val):
        return mean
    return val

# Read CSV and fill '?'s' with means
data=pd.read_csv('PatientData.csv',header=None).apply(pd.to_numeric,errors='coerce')
data=data.apply(getMean)

# Save new CSV
data.to_csv('output.csv', sep=',')


correlationsList = []


# Read pairs of columns and computer the corresponding correlations
for i in range(280):
     df = pd.read_csv('output.csv', usecols=[i, 280])
     correlation = df.corr().iloc[1, 0]
     correlationsList.insert(i, correlation)


# Take the absolute value of correlations
allCorrelations = map(abs, correlationsList)

# Sort correlations and get index of 3 largest values
idx = (-np.asarray(allCorrelations)).argsort()[:3]

print 'Features with highest correlations:'
print idx