import numpy as np
import matplotlib.pyplot as plt



n = 10000
trials = []


for i in range(1000):
    s = np.random.binomial(1, .5, n)

    for x in range(len(s)):
        if(s[x] == 0):
            s[x] = -1

    #print(s)

    z = (1.0/n) * (sum(s))
    trials.insert(i, z)


print(trials)
print(len(trials))


count, bins, ignored = plt.hist(trials, 30, normed=False)
plt.show()