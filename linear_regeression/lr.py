import numpy as np 


class linear:
    def __init__(self, lr=0.0001, numofiteration=100) -> None:
        # Constructor for the linear regression model
        # lr: learning rate for gradient descent algorithm
        # numofiteration: number of iterations for gradient descent
        self.lr = lr
        self.numofiteration = numofiteration
        self.w = None
        self.bias = None

    def fit(self,x,y):
        # Fit the linear regression model to the training data
        # x: input data of shape (n_sample, n_feature)
        # y: output data of shape (n_sample, 1)

        n_sample, n_feature = x.shape

        # Initialize the weight matrix and bias to zeros
        self.w = np.zeros((n_feature,1))
        self.bias = 0

        # Perform gradient descent to optimize the weight matrix and bias
        for _ in range(self.numofiteration):
            y_predict = np.dot(x,self.w) + self.bias

            # Compute the gradients of the loss function w.r.t. weight matrix and bias
            dw = 1/n_sample * np.dot(x.T,(y_predict - y))
            db = 1/n_sample * np.sum(y_predict - y)

            # Update the weight matrix and bias using the gradients and learning rate
            self.w -= self.lr * dw
            self.bias -= self.lr * db

    def predicted(self,x):
        # Predict the output values for the input data using the trained weight matrix and bias
        # x: input data of shape (n_sample, n_feature)
        y_predict = np.dot(x,self.w) + self.bias
        return y_predict