import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import statistics

N_POINTS = 100
TARGET_X_SLOPE = -3
TARGET_y_SLOPE = 2
TARGET_OFFSET  = -5
EXTENTS = 5
NOISE = 5

"""
# create random data
xs = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
ys = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
zs = []
for i in range(N_POINTS):
    zs.append(xs[i]*TARGET_X_SLOPE + \
              ys[i]*TARGET_y_SLOPE + \
              TARGET_OFFSET + np.random.normal(scale=NOISE))

# data normalization
x_stdev = statistics.pstdev(xs)
xs = [x / x_stdev for x in xs]
y_stdev = statistics.pstdev(ys)
ys = [y / y_stdev for y in ys]
z_stdev = statistics.pstdev(zs)
zs = [z / z_stdev for z in zs]

"""

data = pd.read_csv('./input2.csv', header=None, names = ["age","weight","height"])
pstdev = data.apply(statistics.pstdev, axis=0)

data_norm = pd.DataFrame()
data_norm["age"] = data["age"].apply(lambda x: x / pstdev["age"])
data_norm["weight"] = data["weight"].apply(lambda x: x / pstdev["weight"])
data_norm["height"] = data["height"].apply(lambda x: x / pstdev["height"])

xs = data_norm["age"].values.tolist()
ys = data_norm["weight"].values.tolist()
zs = data["height"].values.tolist()




# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(xs, ys, zs, color='b')


def normal_equation(xs,ys):
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    return fit


def gradient_descent(alpha, x, y, z, max_iter=100000, ep=0.0001):

        converged = False
        iter = 0
        m = np.size(xs) # number of samples

        # initial theta
        t0 = 0
        t1 = 0
        t2 = 0

        # total error, J(theta)
        J = sum([(t0 + t1 * x[i] + t2 * y[i] - z[i]) ** 2 for i in range(m)])

        # Iterate Loop
        # for iter in range(iterations):
        while not converged:
            # for each training sample, compute the gradient (d/d_theta j(theta))
            grad0 = 1.0 / m * sum([(t0 + t1 * x[i] + t2 * y[i] - z[i]) for i in range(m)])
            grad1 = 1.0 / m * sum([(t0 + t1 * x[i] + t2 * y[i] - z[i]) * x[i] for i in range(m)])
            grad2 = 1.0 / m * sum([(t0 + t1 * x[i] + t2 * y[i] - z[i]) * y[i] for i in range(m)])

            # update the theta_temp
            temp0 = t0 - alpha * grad0
            temp1 = t1 - alpha * grad1
            temp2 = t2 - alpha * grad2

            # update theta
            t0 = temp0
            t1 = temp1
            t2 = temp2

            # mean squared error
            e = sum([np.float128(t0 + t1 * x[i] + t2 * y[i] - z[i]) ** 2 for i in range(m)])

            if abs(J - e) <= ep:
                print('Converged, iterations: ', iter, '!!!')
                converged = True

            J = e  # update error
            iter += 1  # update iter

            if iter == max_iter:
                print('Max interactions exceeded!')
                converged = True

        # attention to the order of elements: t1 for x, t2 for y, and t0 for z!!
        return [t1, t2, t0]


fit = normal_equation(xs,ys)

print ("solution normal equation:")
#print ("%.6E x + %.6E y + %.6E = z" % (fit[0], fit[1], fit[2]))
print (str(fit[0])+" x + ", str(fit[1]) + " y " + str(fit[2]))

#print ("errors:")
#print (errors)
#print ("residual:")
#print (residual)


for i in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    fit_gradient = gradient_descent(i,xs,ys,zs,100)
    print ("solution gradient descend:",i,fit_gradient[0], "x", fit_gradient[1],"y",fit_gradient[2])

fit_gradient = gradient_descent(0.1,xs,ys,zs)
print ("solution gradient descend:",0.1,fit_gradient[0], "x", fit_gradient[1],"y",fit_gradient[2])

# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        #Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
        Z[r, c] = fit_gradient[0] * X[r, c] + fit_gradient[1] * Y[r, c] + fit_gradient[2]
ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
