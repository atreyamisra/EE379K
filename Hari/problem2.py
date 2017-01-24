import numpy as np
import matplotlib.pyplot as plt



n = 10000
trials = []


for i in range(1000):
	# Get samples from binomial
    s = np.random.binomial(1, .5, n)

	# Replace all 0's with -1
    for x in range(len(s)):
        if(s[x] == 0):
            s[x] = -1

    z = (1.0/n) * (sum(s))
    trials.insert(i, z)


print(trials)
print(len(trials))


# Plot histogram
count, bins, ignored = plt.hist(trials, 30, normed=False)
plt.show()
