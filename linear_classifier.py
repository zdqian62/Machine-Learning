import cvxpy as cp
import numpy as np
import csv
import matplotlib.pyplot as plt

def readfiles():

    with open("data/male_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        rows= [row for row in reader]
        data = np.array(rows)
        data = np.delete(data, 0, axis=0)
        male_data = np.delete(data, 0, axis=1)
        print("out0=",type(male_data),male_data.shape)
        print("out1=",male_data)
        
    csv_file.close()

    with open("data/female_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        rows= [row for row in reader]
        data = np.array(rows)
        data = np.delete(data, 0, axis=0)
        female_data = np.delete(data, 0, axis=1)
        print("out0=",type(female_data),female_data.shape)
        print("out1=",female_data)

    csv_file.close()
    return male_data, female_data
    
def readtestfiles():

    with open("data/male_test_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        rows= [row for row in reader]
        data = np.array(rows)
        data = np.delete(data, 0, axis=0)
        male_data = np.delete(data, 0, axis=1)
        print("out0=",type(male_data),male_data.shape)
        print("out1=",male_data)
        
    csv_file.close()

    with open("data/female_test_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        rows= [row for row in reader]
        data = np.array(rows)
        data = np.delete(data, 0, axis=0)
        female_data = np.delete(data, 0, axis=1)
        print("out0=",type(female_data),female_data.shape)
        print("out1=",female_data)

    csv_file.close()
    return male_data, female_data


def classify_np(male, female):
    X = np.vstack((male, female))
    X = X.astype(np.float)
    a, b = X[:len(male)-1].T
    c, d = X[len(male):].T
    m = len(X)
    print m
    X = np.array([np.ones(m), X[:, 0], X[:, 1]]).T
    print X
    y = np.concatenate((-1*np.ones(len(male)), np.ones(len(female)))).T
    y = y.astype(np.float)
    print y
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    print("optimal weight vector from equation: ")
    print beta
    print a, b
    plt.scatter(a, b, marker='.', color='b', s = 2)
    plt.scatter(c, d, marker='.', color='r', s = 2)
    line_x = np.linspace(0, 100)
    line_y = -beta[0] / beta[2] - (beta[1] / beta[2]) * line_x
    #plt.plot(line_x, line_y)
    plt.show()

def classify_cvx(male, female):
    A = np.vstack((male, female))
    A = A.astype(np.float)
    
    a, e = A[:len(male)-1].T
    c, d = A[len(male):].T
    m = len(A)
    A = np.array([np.ones(m), A[:, 0], A[:, 1]]).T
    print("here is A")
    print A
    b = np.concatenate((-1*np.ones(len(male)), np.ones(len(female)))).T
    print("here is b")
    print b
    x = cp.Variable(3)
    objective = cp.Minimize(cp.sum_squares(A*x - b))
    prob = cp.Problem(objective)
    result = prob.solve()
    print("optimal weight vector from cvx: ")
    print(x.value)
    beta = x.value
    plt.scatter(a, e, marker='.', color='b', s = 2)
    plt.scatter(c, d, marker='.', color='r', s = 2)
    line_x = np.linspace(0, 100)
    line_y = -beta[0] / beta[2] - (beta[1] / beta[2]) * line_x
    plt.plot(line_x, line_y)
    plt.show()

def test(male, female):
    sm = 0
    sf = 0
    male = male.astype(np.float)
    female = female.astype(np.float)
    a, b = male.T
    c, d = female.T
    beta = np.array([ 1.07017505e+01, 1.23396767e-02, -6.67486843e-03])
    plt.scatter(a, b, marker='.', color='b', s = 2)
    plt.scatter(c, d, marker='.', color='r', s = 2)
    line_x = np.linspace(0, 100)
    line_y = -beta[0] / beta[2] - (beta[1] / beta[2]) * line_x
    plt.plot(line_x, line_y)
    for i in range(len(male)):
        data = male[i][0]
        predict = -beta[0] / beta[2] - (beta[1] / beta[2]) * data
        if male[i][1] > predict:
            sm = sm + 1
    for i in range(len(female)):
        data = female[i][0]
        predict = -beta[0] / beta[2] - (beta[1] / beta[2]) * data
        if female[i][1] < predict:
            sf = sf + 1
    print sm, sf
    print len(male)+len(female)
    print ("success rate: ")
    print float(sm+sf)/(len(male)+len(female))
    
    plt.show()

def withlambda(male, female):
    X = np.vstack((male, female))
    X = X.astype(np.float)
    a, b = X[:len(male)-1].T
    c, d = X[len(male):].T
    m = len(X)
    print m
    X = np.array([np.ones(m), X[:, 0], X[:, 1]]).T
    print X
    y = np.concatenate((-1*np.ones(len(male)), np.ones(len(female)))).T
    y = y.astype(np.float)
    print y
    for L in range(1, 100, 10):
        L = L / 10.0
        beta = np.dot(np.linalg.inv(np.dot(X.T, X) + L*np.ones((3, 3))), np.dot(X.T, y))
        print("optimal weight vector from equation: ")
        print beta
        print a, b
        plt.scatter(a, b, marker='.', color='b', s = 2)
        plt.scatter(c, d, marker='.', color='r', s = 2)
        line_x = np.linspace(0, 100)
        line_y = -beta[0] / beta[2] - (beta[1] / beta[2]) * line_x
        plt.plot(line_x, line_y)
    plt.show()

#male, female = readfiles()
#classify_np(male, female)
#classify_cvx(male, female)
#male, female = readtestfiles()
#test(male, female)
male, female = readfiles()
withlambda(male, female)

