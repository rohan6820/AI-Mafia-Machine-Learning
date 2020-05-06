import pandas as p
import numpy as n
import matplotlib.pyplot as plt

class KNN:
    
    def __init__(self,k=10):
        self.k=k
    
    def _distance(self,x1,x2):
        return n.sqrt(sum((x1-x2)**2))
    
    def predict(self,x,y,test):
        val=n.array(sorted([(self._distance(x[i],test),y[i]) for i in range(x.shape[0])])[:self.k])
        j=n.unique(val[:,1],return_counts=True)
        index=j[1].argmax()
        return j[0][index].astype("int")
    
    def plotFreq(self,x,y):
        f=x
        f["Labels"]=y
        classes=n.unique(y)
        class_=[]
        for i in classes:
            class_.append(f.loc[f["Labels"]==i].shape[0])
        plt.bar(classes,class_,width=.1)
        for a,b in zip(classes,class_):
            plt.text(a,b,str(b))
        plt.show()
            
d_x=p.read_csv("Diabetes_XTrain.csv")
d_y=p.read_csv("Diabetes_YTrain.csv")
d_x_test=p.read_csv("Diabetes_Xtest.csv")
k=KNN()
ans=p.DataFrame([k.predict(d_x.values,d_y.values,d_x_test.values[i]) for i in range(d_x_test.shape[0])])
k.plotFreq(d_x_test,ans)