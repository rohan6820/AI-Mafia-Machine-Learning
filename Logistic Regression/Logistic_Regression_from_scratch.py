import numpy as n
import pandas as p
import matplotlib.pyplot as plt

class LogisticRegression:
    
    def  __init__(self,lr=0.01,iter=1000):
        self.lr=lr
        self.iter=iter
        self.weight=None
        self.bias=None
    
    def fit(self,X,y):
        x_samples,x_features=X.shape
        self.weight=n.zeros(x_features)
        self.bias=0
        for _ in range(self.iter):
            y_pred=self.sigmoid(n.dot(X,self.weight)+self.bias)
            dw=(1/x_samples)*n.dot(X.T,(y_pred-y))*2
            db=(1/x_samples)*n.sum(y_pred-y)
            self.weight-=self.lr*dw
            self.bias-=self.lr*db
    def sigmoid(self,x):
        return 1/(1+n.exp(-x))
    def predict(self,x):
        y_pred=self.sigmoid(n.dot(x,self.weight)+self.bias)
        return [1 if i>0.5 else 0 for i in y_pred]

d_x=p.read_csv("Logistic_X_Train.csv").values
d_y=p.read_csv("Logistic_Y_Train.csv").values
d_x_test=p.read_csv("Logistic_X_Test.csv").values
d_y=d_y.reshape((3000,))
l=LogisticRegression()
l.fit(d_x,d_y)
d_y_test=p.DataFrame(l.predict(d_x_test))


            