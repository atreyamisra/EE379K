import numpy as np
import matplotlib.pyplot as plt

mean = [-5, 5]
cov = [[20, .8], [.8, 30]]

s = np.random.multivariate_normal(mean, cov, 10000)

x = []
y = []

for i in range(10000):
    x.insert(i, s[i][0])
    y.insert(i, s[i][1])

mean = [sum(x)/10000, sum(y)/10000]

# Output calculated covariance
print [ [np.sum((x - mean[0])**2)/len(x) , np.sum((x - mean[0])*(y - mean[1]))/len(x) ], [np.sum((x - mean[0])*(y - mean[1]))/len(x), np.sum((y - mean[1])**2)/len(y)] ]
