import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Performs logisitc regression on the MNIST data set

Save predictions to a csv to be submitted to Kaggle
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
            h = sigmoid(X @ temp_theta)
            y1 = (yi.T @ np.log(h))
            y2 = ((1-yi).T @ np.log(1-h) )
            j_reg = ((lambda_reg/(2*X.shape[0])) * np.sum(temp_theta[2:]**2))
            j = -(1/X.shape[0]) * (y1 + y2) + j_reg
            j_hist[i].append(j)
            hy = h - yi
            hyX = X.T @ hy
            reg_theta = temp_theta
            reg_theta[0] = 0
            grad_reg = ((lambda_reg/X.shape[0]) * reg_theta)
            temp_theta = (temp_theta - ((alpha/X.shape[0]) * hyX)) + grad_reg
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

'''
Returns the index with the highest probabily, which is the predicted number
'''
def predict(theta, X):
    test_ans = sigmoid(X @ theta.T)
    return np.argmax(test_ans, axis=1)

def main():
    #TRAIN
    print("Reading training data")
    df = pd.read_csv(path_to_data + train_data)

    #Organize data
    y = df['label'].to_numpy()
    X = df.iloc[:, 1:].to_numpy()
    
    # add collumn of ones to X matrix for theta0 term
    x_one = np.ones((X.shape[0],1))
    X = np.hstack((x_one, X))
    
    
    #Fit model
    print("Gradient Descent with alpha = {}, lambda = {} and {} itterations".format(alpha, lambda_reg, max_itters))
    theta = gradient_descent(X, y, alpha, lambda_reg, max_itters)
    print(theta.shape)

    #TEST
    print("Reading testing data")
    df = pd.read_csv(path_to_data + test_data)

    #Organize data
    X = df.to_numpy()

    # add collumn of ones to X matrix for theta0 term
    x_one = np.ones((X.shape[0],1))
    X = np.hstack((x_one, X))


    #Predict and print results
    pred = predict(theta, X).reshape(X.shape[0], 1)
    pred_id = np.arange(1, X.shape[0]+1, 1).reshape((X.shape[0], 1))
    out = np.hstack((pred_id, pred))
    outdf = pd.DataFrame(out, columns=["ImageID", "Label"])
    outdf.to_csv(path_to_data + output_data, index=False)



if __name__ == "__main__":
    main()