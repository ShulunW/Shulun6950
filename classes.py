#!/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics
import statsmodels.api as sm
from sklearn import linear_model

df = pd.read_csv ("winequality-white.csv")
dfwine=pd.DataFrame(df)

class draw_pics:
    def __init__(self,array1,array2,title1):
        self.array1=array1
        self.array2=array2
        self.title1=title1

    def draw(self):
        plt.plot(dfwine[self.array1],dfwine[self.array2],"o")
        plt.title(self.title1)
        plt.xlabel(self.array1)
        plt.ylabel(self.array2)
        plt.show()

class cook_dis:
    def __init__(self,range0,range1,range2,range3,range4,x_label2,y_label2):
        self.range0=range0
        self.range1=range1
        self.range2=range2
        self.range3=range3
        self.range4=range4
        self.x_label2=x_label2
        self.y_label2=y_label2
        
    def scatter(self):
        plt.figure(figsize=(self.range0,self.range1))
        plt.plot(self.range2, self.range3,"o")
        plt.xlim(self.range4)
        plt.xlabel(self.x_label2)
        plt.ylabel(self.y_label2)
        plt.show()

class metrics_:
    def __init__(self,y_test, y_pred,rowname,colname):
        self.y_test=y_test
        self.y_pred=y_pred
        self.rowname=rowname
        self.colname=colname

    def produce(self):
        cm = pd.crosstab(self.y_test, self.y_pred, rownames=[self.rowname], colnames=[self.colname])
        sn.heatmap(cm, annot=True)
        print('Accuracy: ',metrics.accuracy_score(self.y_test, self.y_pred))
        plt.show()


class tabels:
    def __init__(self,index1,y_name):
        self.index1=index1
        self.y_name=y_name
        self.fit_linear = 0 
    def draw_tab(self):
        X = dfwine.drop(columns=[self.index1])
        y = dfwine[self.y_name]
        X = sm.add_constant(X)
        self.fit_linear = sm.OLS(y,X).fit()
        print(self.fit_linear.summary())
    def get_linear(self):
        return self.fit_linear
