#%%

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = []
Z = []
for x in range(-7, 7):
    for y in range(-7, 7):
        r = numpy.sqrt(x**2 + y**2)

        z = r#numpy.sin(r) / (r + 1e-10)

        data.append((x, y))
        Z.append(z)

data = numpy.array(data)
Z = numpy.array(Z)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.scatter(data[:, 0], data[:, 1], Z)
plt.show()

#%%

N2 = 10

A1 = numpy.random.randn(N2, 2)# / numpy.sqrt(2 + N2)
b1 = numpy.random.randn(N2, 1)# / numpy.sqrt(1 + N2)

A2 = numpy.random.randn(1, N2)# / numpy.sqrt(1 + N2)
b2 = numpy.random.randn(1, 1)# / numpy.sqrt(1 + N2)

As = []

f = A1.dot(data.transpose()) + b1
ef = numpy.exp(-f)
g1 = 1 / (1 + ef)
g2 = A2.dot(g1) + b2

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.scatter(data[:, 0], data[:, 1], g2)
plt.show()


losses = []
for i in range(1000):
    blk = 49

    i = numpy.random.randint(0, data.shape[0] - blk)

    data2 = data[i : i + blk, :].transpose()
    label = Z[i : i + blk]

    f = A1.dot(data2) + b1
    ef = numpy.exp(-f)
    g1 = 1 / (1 + ef)
    g2 = A2.dot(g1) + b2
    loss = (g2 - label)**2

    losses.append(sum(loss))
    print sum(loss)
    #print b

    a = 0.0001 * 0.995**i

    dLdg2 = -(label - g2)
    dg2dA2 = g1.transpose()
    dg1df = (1 - g1) * g1
    dfdA1 = data2.transpose()

    for i in range(blk):
        dLdA2 = dLdg2[:, i] * dg2dA2[i, :]

        d = dLdg2[:, i].dot(A2.dot(dg1df[:, i]))

        A2 -= a * dLdA2
        b2 -= a * dLdg2[:, i]

        A1 -= a * d * data2[:, i]
        b1 -= a * d
        #1/0

    #1/0
    #for d, sample in zip((dLdg * dgdf).reshape(-1), dfdA):
        #print -a * d * sample
    #    A -= a * d * sample
    #    b -= a * d

    #As.append(A.copy())

plt.plot(losses)
plt.show()

f = A1.dot(data.transpose()) + b1
ef = numpy.exp(-f)
g1 = 1 / (1 + ef)
g2 = A2.dot(g1) + b2

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.scatter(data[:, 0], data[:, 1], Z, c = 'r')
surf = ax.scatter(data[:, 0], data[:, 1], g2)
plt.gcf().set_size_inches((10, 10))
plt.show()

plt.plot(data[:, 1], Z, 'rx')
plt.plot(data[:, 1], g2.flatten(), 'bx')
plt.show()

#As2 = numpy.array(As)

#plt.plot(As2[:, 1, 0], As2[:, 1, 1])
#plt.show()

#losses = -labels + numpy.log(ef.sum(axis = 1).reshape(ef.shape[0], 1))

