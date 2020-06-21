import numpy as np
import matplotlib.pyplot as plt
import os

def my_training(train_cat, train_grass):
    mu_cat = train_cat.mean(axis=1, keepdims=True)
    mu_grass = train_grass.mean(axis=1, keepdims=True)
    Sigma_cat = np.cov(train_cat)
    Sigma_grass = np.cov(train_grass)
    return mu_cat, mu_grass, Sigma_cat, Sigma_grass

def my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass, shift=8):
    
    d = mu_cat.shape[0]
    M, N = Y.shape
    output = np.zeros((M-8, N-8))
    prior_cat = K_cat / (K_cat + K_grass)
    prior_grass = K_grass / (K_cat + K_grass)
    sigma_cat_inv = np.linalg.pinv(Sigma_cat)
    sigma_grass_inv = np.linalg.pinv(Sigma_grass)
    c_cat = 1/(2*np.pi)**(d/2)/np.sqrt(np.linalg.det(Sigma_cat))*prior_cat
    c_grass = 1/(2*np.pi)**(d/2)/np.sqrt(np.linalg.det(Sigma_grass))*prior_grass
    for i in np.arange(0, M-8, shift):
        for j in np.arange(0, N-8, shift):
            z = np.reshape(Y[i:i+8, j:j+8], (64,1),'F')
            f_cat = c_cat*np.exp(-0.5*(np.matmul(np.matmul((z-mu_cat).T , sigma_cat_inv) , (z-mu_cat))))
            f_grass = c_grass*np.exp(-0.5*(np.matmul(np.matmul((z-mu_grass).T , sigma_grass_inv) , (z-mu_grass))))
            output[i:i+shift][j:j+shift] = f_cat > f_grass
    return output
    '''
    f_cat = np.log(K_cat / (K_cat + K_grass))
    f_grass = np.log(K_grass / (K_cat + K_grass))
    M = len(Y)
    N = len(Y[0])
    c_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(Sigma_cat))))
    c_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(Sigma_grass))))
    inv_cat = np.linalg.pinv(Sigma_cat)
    inv_grass = np.linalg.pinv(Sigma_grass)
    output = np.zeros((M - 8, N - 8))
    for i in range(M - 8):
        for j in range(N - 8):
            z = Y[i : i + 8, j : j + 8]
            z = z.flatten('F')
            z = z.reshape((64,1) )
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass
            func_cat = f_cat + c_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            func_grass = f_grass + c_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)
            if(func_cat > func_grass):
                output[i][j] = 1
    return output
    '''
def my_testing_no(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass):
    f_cat = np.log(K_cat / (K_cat + K_grass))
    f_grass = np.log(K_grass / (K_cat + K_grass))
    M = len(Y)
    N = len(Y[0])
    c_cat = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(Sigma_cat))))
    c_grass = -np.log(((2 * np.pi) ** (64 / 2) * np.sqrt(np.linalg.det(Sigma_grass))))
    inv_cat = np.linalg.pinv(Sigma_cat)
    inv_grass = np.linalg.pinv(Sigma_grass)
    output = np.zeros((M - 8, N - 8))
    for i in range(0, M - 8, 8):
        for j in range(0, N - 8, 8):
            z = Y[i : i + 8, j : j + 8]
            z = z.flatten('F')
            z = z.reshape((64,1) )
            diff_cat = z - mu_cat
            diff_grass = z - mu_grass
            func_cat = f_cat + c_cat - 0.5 * np.matmul(np.matmul(np.transpose(diff_cat), inv_cat), diff_cat)
            func_grass = f_grass + c_grass - 0.5 * np.matmul(np.matmul(np.transpose(diff_grass), inv_grass), diff_grass)
            if(func_cat > func_grass):
                output[i:i+8, j:j+8] = 1
    return output


if __name__ == '__main__':
    
    Y = plt.imread('../HW5/resultimg/lambda1_100.jpg', 0) / 255
    Y = Y[:,:,0]
    train_cat = np.loadtxt('train_cat.txt', delimiter = ',')
    train_grass = np.loadtxt('train_grass.txt', delimiter = ',')
    K_cat, K_grass = train_cat.shape[1], train_grass.shape[1]
    

    mu_cat, mu_grass, Sigma_cat, Sigma_grass = my_training(train_cat, train_grass)
    output = my_testing_no(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass)
    output = np.asfarray(output)
    plt.imsave('../HW5/resultimg/lambda1_100res.png', output, cmap = 'gray')
    truth = plt.imread('truth.png')
    diff = np.abs(np.subtract(output, truth[4:len(truth)-4, 4:len(truth[0])-4]))
    mae = np.sum(np.abs(np.subtract(output, truth[4:len(truth)-4, 4:len(truth[0])-4]))) / output.size
    print(mae)








