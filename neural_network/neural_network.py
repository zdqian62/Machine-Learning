import numpy as np
from numpy import linalg as LA
#N = 1, R = 0
'''
converge at
[1.17813651]
layer1
[[  1.25742272   1.25742272   1.25742272   1.25742272   1.25742272
    1.25742272   1.25742272]
 [  4.64961425   4.64961425   4.64961425   4.64961425   4.64961425
    4.64961425   4.64961425]
 [-13.42909596 -13.42909596 -13.42909596 -13.42909596 -13.42909596
  -13.42909596 -13.42909596]
 [ 44.06104693  44.06104693  44.06104693  44.06104693  44.06104693
   44.06104693  44.06104693]
 [ 42.1497166   42.1497166   42.1497166   42.1497166   42.1497166
   42.1497166   42.1497166 ]
 [-19.438933   -19.438933   -19.438933   -19.438933   -19.438933
  -19.438933   -19.438933  ]
 [-19.67437007 -19.67437007 -19.67437007 -19.67437007 -19.67437007
  -19.67437007 -19.67437007]
 [ 12.54555525  12.54555525  12.54555525  12.54555525  12.54555525
   12.54555525  12.54555525]]
layer2
[[2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]]
result for whole data
[3.49999999]
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def gettrainingdata(filename):
    data = np.loadtxt(filename, delimiter=",", usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    res = np.loadtxt(filename, delimiter=",", usecols=[1])
    return data, res

def normalizing(data):
    mean = np.mean(data, 0)
    sstd = np.std(data, 0, ddof=1)
    normalized = (data - mean[None, :]) / sstd
    return normalized

def forward(data, res, w_1_model, w_2_model):
    W1 = sigmoid(np.dot(data, w_1_model))
    result = sigmoid(np.dot(W1, w_2_model))
    err = (res- result)**2
    return W1, result, err

def back(data, res, layer1, w_1_model, w_2_model, result):
    d_weights2 = layer1 * (2*(res - result) * sigmoid_derivative(result))
    d_weights1 = (np.matrix(data).T * ((2*(res - result) * sigmoid_derivative(result) * w_2_model.T) * sigmoid_derivative(layer1)))
    w_1_model += 0.01 * d_weights1
    w_2_model += 0.01 * np.matrix(d_weights2).T
    return w_1_model, w_2_model

def gettestdata(filename):
    data = np.loadtxt(filename, delimiter=",", usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    res = np.loadtxt(filename, delimiter=",", usecols=[1], dtype='str')
    return data, res

w_1_model = np.ones((8, 7))
w_2_model = np.ones((7, 1))
data, res = gettrainingdata("trainingdata.data")


data = normalizing(data)
wtw = np.dot(np.transpose(data), data)
w, v = LA.eig(wtw)
newdata = np.dot(wtw, v)
newdata = newdata[:, 0:8]
data = newdata


totalerr = 1
lasterr = 0
while np.abs(totalerr - lasterr) > 0.0000001:
    lasterr = totalerr
    totalerr = 0
    for i in range(len(data)):
        layer1, result, err = forward(data[i], res[i], w_1_model, w_2_model)
        w_1_model, w_2_model = back(data[i], res[i], layer1, w_1_model, w_2_model, result)
        totalerr = totalerr + err
    print totalerr

print w_1_model
print w_2_model

testdata, testres = gettestdata("wpbc.data")
for i in range(len(testres)):
    if testres[i] == 'N':
        testres[i] = 1
    else:
        testres[i] = 0
testres = testres.astype(int)

testdata = normalizing(testdata)
wtw = np.dot(np.transpose(testdata), testdata)
w, v = LA.eig(wtw)
newdata = np.dot(wtw, v)
newdata = newdata[:, 0:8]
testdata = newdata


total = 0
for i in range(len(testdata)):
    layer1, result, err = forward(testdata[i], testres[i], w_1_model, w_2_model)
    total = total + err
print total

'''
converge at
[1.17813651]
layer1
[[  1.25742272   1.25742272   1.25742272   1.25742272   1.25742272
    1.25742272   1.25742272]
 [  4.64961425   4.64961425   4.64961425   4.64961425   4.64961425
    4.64961425   4.64961425]
 [-13.42909596 -13.42909596 -13.42909596 -13.42909596 -13.42909596
  -13.42909596 -13.42909596]
 [ 44.06104693  44.06104693  44.06104693  44.06104693  44.06104693
   44.06104693  44.06104693]
 [ 42.1497166   42.1497166   42.1497166   42.1497166   42.1497166
   42.1497166   42.1497166 ]
 [-19.438933   -19.438933   -19.438933   -19.438933   -19.438933
  -19.438933   -19.438933  ]
 [-19.67437007 -19.67437007 -19.67437007 -19.67437007 -19.67437007
  -19.67437007 -19.67437007]
 [ 12.54555525  12.54555525  12.54555525  12.54555525  12.54555525
   12.54555525  12.54555525]]
layer2
[[2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]
 [2.78475679]]
result for whole data
[3.49999999]
'''

