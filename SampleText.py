import pandas
from yayo import Regression
from yayo import Classification
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def C():
    dataset = pd.read_csv(r"C:\\Users\\Raunaq\\Desktop\\Machine Learning Project\\ClassificationData.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(len(y),1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    demo=Classification(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    # print(demo.Logistic())
    # print(demo.DecisionTree())
    # print(demo.RandomForest())
    # print(demo.KNN())
    # print(demo.NaiveBayes())
    # print(demo.SVM())
    # print(demo.XGBoost())
    # print(demo.ADABoost())
    # print(demo.CatBoost())

def R():
    dataset = pd.read_csv(r"C:\\Users\\Raunaq\\Desktop\\Machine Learning Project\\RegressionData.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(len(y),1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    demo=Regression(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    # print(demo.Linear())
    # print(demo.Polynomial())
    # print(demo.DecisionTree())
    # print(demo.SVM())
    # print(demo.RandomForest())
    # print(demo.XGBoost())
    # print(demo.ADABoost())
    # print(demo.CatBoost())  
    
C()