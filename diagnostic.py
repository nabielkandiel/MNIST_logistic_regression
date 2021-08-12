import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
This is a support file for the model in main.py

Here is where different techniques will be used to analyze the performance
of the model
'''

'''
Takes in a list of possible lambda values and trains the model with each of them

The error will be recorded and the data will then be plotted for analysis
'''

path_to_data = "../data/"
train_data   = "train.csv"
test_data    = "test.csv"
output_data  = "output.csv"
# 0 - 9
labels       = 10
alpha        = .00003
lambda_reg   = .3
max_itters   = 1000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linearRegressionCost(X, y, theta, lambda_reg):
    h = sigmoid(X @ theta)
    y1 = (y.T @ np.log(h))
    y2 = ((1-y).T @ np.log(1-h) )
    j_reg = ((lambda_reg/(2*X.shape[0])) * np.sum(theta[2:]**2))
    j = -(1/X.shape[0]) * (y1 + y2) + j_reg

    hy = h - y
    hyX = X.T @ hy
    reg_theta = theta
    reg_theta[0] = 0
    grad_reg = ((lambda_reg/X.shape[0]) * reg_theta)
    grad = ((alpha/X.shape[0]) * hyX) + grad_reg
    return j, grad

'''
Performs gradient descent to find the best thetas for each 
output label

Also creates graphs to ensure the cost is decrease over time
'''
def gradient_descent(X, y, alpha, lambda_reg, max_itters):
    #Set up graph to see how gradient descent performed
    fig, axs = plt.subplots(5, 2)
    x_axis = list(range(0, max_itters))
    theta = np.zeros((labels, X.shape[1]))
    j_hist = {}

    #Find theta values for each output lable
    for i in range(labels):
        temp_theta = np.zeros(X.shape[1])
        yi = (y == i).astype(int)
        j_hist[i] = []
        for k in range(max_itters):
            j, grad = linearRegressionCost(X, yi, temp_theta, lambda_reg)
            j_hist[i].append(j)
            temp_theta = temp_theta - grad
            if k % 50 == 0:
                print("Label : {} \t Itter : {} \t Cost : {}".format(i, k, j), end='\r')
        theta[i, :] = temp_theta.T

        #Print final cost and set up cost graph
        print("Label : {} \t Itter : {} \t Cost : {}".format(i, k+1, j))
        subplt_row = i % 5
        subplt_col = int(i >= 5)
        axs[subplt_row, subplt_col].plot(x_axis, j_hist[i])
        axs[subplt_row, subplt_col].set_title("Label:" + str(i))
    
    #Show cost graph and print out cost values for each itteration
    for ax in axs.flat:
        ax.set(xlabel="Itteration", ylabel="Cost")
    plt.show()
    for i in range(len(j_hist)):
        print("Cost on itter {:4} : {:10}".format(i, j_hist[i][-1]))
    return theta



def validateLambda(X, y, Xval, yval, lambda_vec):
    error_train = np.zeros((len(lambda_vec), 0))
    error_val   = np.zeros((len(lambda_vec), 0))
    for i in range(len(lambda_vec)):
        test_lambda = lambda_vec[i]
        theta = trainLinearReg(X, y, test_lambda)
        error_train[i] = linearRegCostFunction(X, y, theta, 0)
        error_val[i] = linearRegCostFunction(X, y, theta, 0)
    return error_train, error_val
