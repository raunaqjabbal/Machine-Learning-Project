import pandas
from yayo import Regression
import os

# Importing the dataset
import pandas as pd
dataset = pd.read_csv(r"C:\\Users\\Raunaq\\Desktop\\Demo\\Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
demo=Regression(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
print(demo.Linear())
# print(demo.Polynomial())
# print(demo.DecisionTree())
# print(demo.SVM())
# print(demo.RandomForest())
# print(demo.XGBoost())
# print(demo.ADABoost())
# print(demo.CatBoost())