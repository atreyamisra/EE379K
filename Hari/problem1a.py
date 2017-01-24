import numpy as np
import matplotlib.pyplot as plt

# Get samples from normal distribution
s1 = np.random.normal(-10, 5, 1000)
s2 = np.random.normal(10, 5, 1000)

sum = s1 + s2

newMean = np.mean(sum)
newVar = (np.std(sum))**2

print(newMean)
print(newVar)

# Construct and create plot
count, bins, ignored = plt.hist(sum, 30, normed=False)
plt.show()
