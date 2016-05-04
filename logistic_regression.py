#%%

import numpy

A = numpy.random.randn(2, 2)
b = numpy.random.randn(2)

data = []
labels = []
for i in range(100):
    x = 10.0 + numpy.random.randn()
    y = 7.0 + numpy.random.randn()
    
    data.append((x, y))
    labels.append([1.0, 0.0])    
    
    x = 5.0 + numpy.random.randn()
    y = 8.0 + numpy.random.randn()
    
    data.append((x, y))
    labels.append([0.0, 1.0])

data = numpy.array(data)
labels = numpy.array(labels)

#%%

f = data.dot(A) + b
ef = numpy.exp(f)
ef = ef / ef.sum(axis = 1).reshape(ef.shape[0], 1)
loss = labels.dot(ef.transpose())
#losses = -labels + numpy.log(ef.sum(axis = 1).reshape(ef.shape[0], 1))

