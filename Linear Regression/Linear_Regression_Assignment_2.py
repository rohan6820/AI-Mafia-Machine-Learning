import numpy as n
import pandas as p
import matplotlib.pyplot as plt
class LinearRegression:
    
    def __init__(self,lr=0.01,iter=1000):
        self.lr=lr
        self.iter=iter
        self.weight=None
        self.bias=None
    
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weight=n.zeros(n_features)
        self.bias=0
        for _ in range(self.iter):
            y_pred=n.dot(X,self.weight)+self.bias
            dw=(1/n_samples)*n.dot(X.T,(y_pred-y))*2
            db=(1/n_samples)*n.sum(y_pred-y)
            self.weight-=self.lr*dw
            self.bias-=self.lr*db
    
    def predict(self,X):
        return n.dot(X,self.weight)+self.bias
    
    def meanSquareError(self,Y,Y_pred):
        return n.mean((Y-Y_pred)**2)
    
d=p.read_csv("Train.csv")
d_x=d.iloc[:,:5].values
d_y=d.iloc[:,-1].values
test=p.read_csv("Test.csv")
r=LinearRegression()
r.fit(d_x,d_y)
f=p.DataFrame(r.predict(test))