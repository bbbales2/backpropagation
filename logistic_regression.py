#%%

import numpy

data = []
labels = []
for i in range(100):
    x = 10.0 + numpy.random.randn()
    y = 7.0 + numpy.random.randn()
    
    data.append((x, y))
    labels.append(1.0)    
    
    x = 5.0 + numpy.random.randn()
    y = 8.0 + numpy.random.randn()
    
    data.append((x, y))
    labels.append(0.0)

data = numpy.array(data)
labels = numpy.array(labels)

#%%

A = numpy.random.randn(2, 2)
b = numpy.random.randn(2, 1)


As = []

for i in range(1000):
    blk = 20
    
    i = numpy.random.randint(0, data.shape[0] - blk)
    
    data2 = data[i : i + blk, :].transpose()
    label = labels[i : i + blk]
    
    f = A.dot(data2) + b
    ef = numpy.exp(-f)
    g = 1 / (1 + ef)#ef / ef.sum(axis = 1).reshape(ef.shape[0], 1)
    loss = (g - label)**2#-sum((labels * ef).flatten())
    
    print sum(loss)
    #print b
    
    a = 0.1
    
    dLdg = -(label - g)
    dgdf = (1 - g) * g
    dfdA = data2.transpose()
    
    for d, sample in zip((dLdg * dgdf).reshape(-1), dfdA):
        #print -a * d * sample
        A -= a * d * sample
        b -= a * d

    As.append(A.copy())

As2 = numpy.array(As)

plt.plot(As2[:, 1, 0], As2[:, 1, 1])
plt.show()

#losses = -labels + numpy.log(ef.sum(axis = 1).reshape(ef.shape[0], 1))

