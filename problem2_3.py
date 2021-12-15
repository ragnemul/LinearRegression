#
# MACHINE LEARNING WEEK 7 Project
# Linear Regression
# Author: Luis Menendez (luis.menendez@gmail.com)
#

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import itertools
import statistics
from mpl_toolkits.mplot3d import Axes3D
from decimal import *



data = pd.read_csv('./input2.csv', header=None, names = ["age","weight","height"])

# need to use population standard deviation
pstdev = data.apply(statistics.pstdev, axis=0)


data_norm = pd.DataFrame()
data_norm["age"] = data["age"].apply(lambda x: x / pstdev["age"])
data_norm["weight"] = data["weight"].apply(lambda x: x / pstdev["weight"])
data_norm["height"] = data["height"].apply(lambda x: x / pstdev["height"])

# X contains first column with intercept (ones) and features (age and weight)
#X = np.c_[np.ones(data_norm.shape[0]), data_norm.iloc[:,[0,1]]]
X = np.c_[np.ones(data_norm.shape[0]), data_norm.iloc[:,[0,1]]]

# Y contains the desired result
Y = np.c_[data.iloc[:,2]]

# W contains the weights (B)
W = np.array([0.0, 0.0, 0.0])


"""
returns the evolution of errors in R
"""
def gradient_descend(W,X,Y,alpha,ep=1e-10,iterations=1000000):
    # resets the weights vector
    for i, item in enumerate(W):
        W[i] = 0.0
    # temp weights
    _W = [0] * 3


    n = X.shape[0]

    _e = sum([np.float128((Y[i] - np.matmul(X[i], W.transpose()))) ** 2 for i in range(n)])
    converged = False
    num_iters = 0
    while not converged:

        for w_i in range(W.size):
            _W[w_i] = (1 / n) * sum([np.float128((np.matmul(X[i], W.transpose()) - Y[i])) * X[i][w_i] for i in range(n)])

        for w_i in range(W.size):
            W[w_i] = W[w_i] - alpha * _W[w_i]

        e = sum([np.float128((Y[i] - np.matmul(X[i], W.transpose()))) ** 2 for i in range(n)])

        if abs(_e - e) <= ep:
            converged = True

        _e = e  # update error

        num_iters += 1  # update iter

        if num_iters == iterations:
            converged = True

    return [num_iters, W[1], W[2], W[0]]




output = []
for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
    result = gradient_descend(W,X,Y,alpha,0.0001,100)
    output.append([alpha, result[0], Decimal(result[1]), Decimal(result[2]),Decimal(result[3])])
    print ("%f,%i,%F,%F,%F" % (alpha, result[0], result[1], result[2],result[3]))



alpha=0.1
result = gradient_descend(W,X,Y,alpha)
output.append([alpha, result[0], Decimal(result[1]), Decimal(result[2]), Decimal(result[3])])
print ("%f,%i,%F,%F,%F" % (alpha, result[0], result[1], result[2],result[3]))


with open("output2.csv", 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter=',')
    for idx, row in enumerate(output):
        writer.writerow(row)
csvFile.close()


# plotting the normalized points, so using data_norm

# plot points
fig = plt.figure()
ax = plt.subplot(111, projection='3d')

# Age (years)
age_x = data_norm["age"].values.tolist()
ax.set_xlabel('Age (years)')

# Weight(Kilograms)
weight_y= data_norm["weight"].values.tolist()
ax.set_ylabel('Weight (Kilograms)')

# Height (meters)
height_z = data["height"].values.tolist()
ax.set_zlabel('Height (meters)')


ax.scatter(age_x, weight_y, height_z)




# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r, c] = result[1] * X[r, c] + result[2] * Y[r, c] + result[3]
ax.plot_wireframe(X,Y,Z, color='k')

plt.show()




