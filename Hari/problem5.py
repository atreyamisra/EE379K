import pandas as pd
import numpy as np


def getMean(s):
    meanS=np.mean(s)
    return s.apply(replaceValues, mean=meanS)

def replaceValues(val, mean):
    if pd.isnull(val):
        return mean
    return val

data=pd.read_csv('PatientData.csv',header=None).apply(pd.to_numeric,errors='coerce')
data=data.apply(getMean)


data.to_csv('output.csv', sep=',')


trials = []

for i in range(280):
     df = pd.read_csv('output.csv', usecols=[i, 280])
     correlation = df.corr().iloc[1, 0]
     trials.insert(i, correlation)

#print trials

allCorrelations = map(abs, trials);

idx = (-np.asarray(trials)).argsort()[:3]

print 'Features with highest correlations:'
print idx

