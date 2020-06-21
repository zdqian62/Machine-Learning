import csv
import numpy as np
import matplotlib.pylab as plt
import pickle, sys
from cvxopt import matrix, solvers
import datetime

def readvector(filename):
    x0 = []
    x1 = []
    x2 = []
    with open(filename) as f:
        for line in f:
            row = line.split(",")
            x0.append(1.0)
            x1.append(float(row[0]))
            x2.append(float(row[1]))
    return np.column_stack((np.array(x0), np.array(x1), np.array(x2)))

def readlable(filename):
    y = []
    with open(filename) as f:
        for line in f:
            y.append(float(line))
    return np.array(y)

def logistic(x, y, learning_rate = 0.1, max_num_iterations = 200):
    theta = np.zeros(3)
    count = 0
    for m in range(max_num_iterations):
        theta -= learning_rate * cost_function_derivative(x, y, theta)
        if count == int(0.2 * max_num_iterations):
            plotDecisionBoundary(theta, x, y)
            print((theta[0]*theta[0] + theta[1]*theta[1] + theta[2]*theta[2])**0.5)
            count = 0
        count += 1
    #plt.show()
    return theta

def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g

def cost_function_derivative(x, y, theta):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    grad = x.T.dot(h-y)
    return grad

def plotDecisionBoundary(theta, x, y):
    p1 = plt.scatter(x[:1000, 1], x[:1000, 2], marker='+', color='b')
    p2 = plt.scatter(x[1000:, 1], x[1000:, 2], marker='.', color='r')
    plot_x = np.array([-1.0, 0.8])
    plot_y = -1/theta[2]*(theta[1]*plot_x+theta[0])
    plt.axis([-1.2, 1.2, -1, 1])
    plt.plot(plot_x, plot_y)

def perceptron_online(x, y, learning_rate=0.01, max_num_iterations=200):
    theta = np.zeros(3)
    shuffled_index = np.random.permutation(y.size)
    xx = x[shuffled_index, :]
    yy = y[shuffled_index]
    count = 0
    idx = 0
    for i, label in enumerate(yy):
        result = theta.dot(xx[i])
        output = 1 if result >= 0 else 0
        update = learning_rate*(label-output)
        theta[0] += update
        theta[1:] += update * xx[i][1:]
        if count == int(0.2 * max_num_iterations):
            plotDecisionBoundary(theta, x, y)
            #plt.show()
            count = 0
        count += 1
        if idx > max_num_iterations:
            break
        idx += 1
    #plt.show()
    return theta

def perceptron_batch(x, y, learning_rate=0.01, max_num_iterations=200):
    theta = np.zeros(3)
    count = 0
    for m in range(max_num_iterations):
        result = []
        output = []
        for i in range(len(x)):
            result.append(theta.dot(x[i]))
            if theta.dot(x[i]) >= 0:
                output.append(1)
            else:
                output.append(0)
        update = sum(learning_rate*(y-output))
        theta[0] += update
        theta[1:] += update * x[i][1:]
        if count == int(0.2 * max_num_iterations):
            plotDecisionBoundary(theta, x, y)
            count = 0
        count += 1
    #plt.show()
    return theta

def svm_hard(X, y): 
    x = []
    for i in range(len(X)):
        x.append(X[i][1:])
    x = np.array(x)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    NUM = x.shape[0]
    DIM = x.shape[1]
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((NUM, 1)))
    G = matrix(-np.eye(NUM))
    h = matrix(np.zeros(NUM))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    w = np.sum(alphas * y[:, None] * x, axis = 0)
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    bias = b[0]
    norm = np.linalg.norm(w)
    w, bias = w / norm, bias / norm
    slope = -w[0] / w[1]
    intercept = -bias / w[1]
    print slope, intercept
    xx = np.arange(-1, 2)
    plt.plot(xx, xx * slope + intercept, 'k-')
    #plt.show()

def svm_soft(X, y): 
    x = []
    for i in range(len(X)):
        x.append(X[i][1:])
    x = np.array(x)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    NUM = x.shape[0]
    DIM = x.shape[1]
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((NUM, 1)))

    G_std = matrix(np.diag(np.ones(NUM) * -1))
    h_std = matrix(np.zeros(NUM))

    G_slack = matrix(np.diag(np.ones(NUM)))
    h_slack = matrix(np.ones(NUM) * 100)

    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.vstack((h_std, h_slack)))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    w = np.sum(alphas * y[:, None] * x, axis = 0)
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    bias = b[0]
    norm = np.linalg.norm(w)
    w, bias = w / norm, bias / norm
    slope = -w[0] / w[1]
    intercept = -bias / w[1]
    print slope, intercept
    xx = np.arange(-1, 2)
    plt.plot(xx, xx * slope + intercept, 'g-')
    #plt.show()

x = readvector('data/hw04_sample_vectors.csv')
y = readlable('data/hw04_labels.csv')
p1 = plt.scatter(x[:1000, 1], x[:1000, 2], marker='+', color='b')
p2 = plt.scatter(x[1000:, 1], x[1000:, 2], marker='.', color='r')
starttime = datetime.datetime.now()
logistic(x, y)
endtime = datetime.datetime.now()
print (endtime - starttime)

starttime = datetime.datetime.now()
perceptron_online(x, y)
endtime = datetime.datetime.now()
print (endtime - starttime)

starttime = datetime.datetime.now()
perceptron_batch(x, y)
endtime = datetime.datetime.now()
print (endtime - starttime)

starttime = datetime.datetime.now()
svm_hard(x, y)
endtime = datetime.datetime.now()
print (endtime - starttime)

starttime = datetime.datetime.now()
svm_soft(x, y)
endtime = datetime.datetime.now()
print (endtime - starttime)
