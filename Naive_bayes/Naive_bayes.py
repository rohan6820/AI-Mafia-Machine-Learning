import numpy as n
import pandas as p

class NaiveBayes:
    
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.classes=n.unique(y)
        n_classes=len(self.classes)
        self.mean=n.zeros((n_classes,n_features),dtype=n.float64)
        self.var=n.zeros((n_classes,n_features),dtype=n.float64)
        self.prior=n.zeros(n_classes,dtype=n.float64)
        for i in self.classes:
            x_c=x[y==i]
            self.mean[i,:]=x_c.mean(axis=0)
            self.var[i,:]=x_c.var(axis=0)
            self.prior[i]=x_c.shape[0]/float(n_samples)
        
    def predict(self,x):
        y_pred=[self.helper(i) for i in x]
        return y_pred
    
    def helper(self,x):
        prosteriors=[]
        for i,val in enumerate(self.classes):
            prior=n.log(self.prior[i])
            con=n.sum(n.log(self.pdf(x,i)))
            prosteriors.append(prior+con)
        return self.classes[n.argmax(prosteriors)]
    
    def pdf(self,x,i):
        mean=self.mean[i]
        var=self.var[i]
        numerator=n.exp(-(x-mean)**2/2*(var**2))
        denominator=n.sqrt(2*n.pi*(var**2))
        return numerator/denominator
    

    
d_x=p.read_csv("Logistic_X_Train.csv").values
d_y=p.read_csv("Logistic_Y_Train.csv").values
d_y=d_y.reshape((3000,))
nb=NaiveBayes()
nb.fit(d_x,d_y)
d_x_test=p.read_csv("Logistic_X_Test.csv").values
y_pred=nb.predict(d_x_test)


