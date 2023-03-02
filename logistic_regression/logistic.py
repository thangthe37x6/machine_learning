import numpy as np

class logistic_regression:
    def __init__(self,lr=0.0001, numofiteration=1000) -> None:
        self.lr = lr
        self.numofiteration = numofiteration
        self.w = None
        self.bias = None
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def fit(self,x,y):

        n_sample, n_feature = x.shape
        self.w = np.zeros(n_feature)
        self.bias = 0
        for _ in range(self.numofiteration):
            y_predict = self.sigmoid(np.dot(x,self.w))

            dw = 1/n_sample * np.dot(x.T,(y_predict - y))
            db = 1/n_sample * np.sum(y_predict - y)

            self.w -= self.lr * dw
            self.bias -= self.lr * db
    def predicted(self,x):
        y_pred = self.sigmoid(np.dot(x,self.w))
        labels = [1 if i >= 0.5 else 0 for i in y_pred]
        return labels

