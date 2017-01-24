import numpy as np
import matplotlib.pyplot as plt

s1 = np.random.normal(-10, 5, 1000)
s2 = np.random.normal(10, 5, 1000)

sum = s1 + s2
#print(sum)

# plt.plot(sum)
# plt.show()

newMean = np.mean(sum)
newVar = (np.std(sum))**2

print(newMean)
print(newVar)


count, bins, ignored = plt.hist(sum, 30, normed=False)
plt.show()