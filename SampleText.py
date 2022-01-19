import pandas
from yayo import Regression
from yayo import Classification
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
base_path = Path(__file__).parent
def C():
    file_path1 = (base_path/"../Machine Learning Project/ClassificationData.csv").resolve()
    dataset = pd.read_csv(file_path1)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    # y = y.reshape(len(y),1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    demo=Classification(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    # print(demo.Logistic())
    # print(demo.DecisionTree())
    # print(demo.RandomForest())
    # print(demo.KNN())
    # print(demo.NaiveBayes())
    # print(demo.SVM())
    # print(demo.XGBoost())
    # print(demo.CatBoost())
    # print(demo.ADABoost())

def R():
    file_path2 = (base_path/"../Machine Learning Project/RegressionData.csv").resolve()
    dataset = pd.read_csv(file_path2)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(len(y),1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    demo=Regression(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    # print(demo.Polynomial())
    # print(demo.DecisionTree())
    print(demo.SVM())
    # print(demo.RandomForest())
    # print(demo.XGBoost())
    # print(demo.CatBoost())
    # print(demo.ADABoost())
      
R()