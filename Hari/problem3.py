import numpy as np
import matplotlib.pyplot as plt


# Get samples for normal
s = np.random.normal(0, 5, 25000)
mean = np.sum(s)/len(s)
variance = np.sum((s - mean)**2)/len(s)
std = variance**(.5)

print(mean)
print(std)
