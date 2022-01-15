import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
            

class Regression:
    def __init__(self, X_train=None,y_train=None,X_test=None,y_test=None):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        y_train = sc_y.fit_transform(y_train)   
        X_test = sc_X.transform(X_test)
        y_test = sc_y.transform(y_test)
        self.y_train=self.y_train.reshape(-1)
    def main(self):
        val=[]
        
    def Linear(self):
        regressor=LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        type='LinearRegression'
        y_pred=regressor.predict(self.X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output
        
    def Polynomial(self):
        poly_reg = PolynomialFeatures(degree=4)
        X_train=poly_reg.fit_transform(self.X_train)
        X_test=poly_reg.transform(self.X_test)
        regressor=LinearRegression()
        regressor.fit(X_train, self.y_train)
        type='PolynomialRegression'
        y_pred=regressor.predict(X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output
            
    def DecisionTree(self):
        regressor= DecisionTreeRegressor()
        regressor.fit(self.X_train, self.y_train)
        type='DecisionTreeRegression'
        y_pred=regressor.predict(self.X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output
    
    def SVM(self):
        regressor=SVR(kernel='linear')
        regressor.fit(self.X_train, self.y_train.ravel())
        type='SupportVectorMachine'
        y_pred=regressor.predict(self.X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output
        
    def RandomForest(self):
        regressor=RandomForestRegressor()
        regressor.fit(self.X_train,self.y_train)
        type='RandomForest'
        y_pred=regressor.predict(self.X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output
        
    def XGBoost(self):
        regressor=XGBRegressor(use_label_encoder=False)
        regressor.fit(self.X_train, self.y_train)
        type='XGBoost'
        y_pred=regressor.predict(self.X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output

    def CatBoost(self):
        regressor=CatBoostRegressor()
        regressor.fit(self.X_train, self.y_train)
        type='CatBoost'
        y_pred=regressor.predict(self.X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output
        
    def ADABoost(self):
        regressor=AdaBoostRegressor(n_estimators=100)
        lab_enc = LabelEncoder()
        self.y_train= lab_enc.fit_transform(self.y_train)
        regressor.fit(self.X_train, self.y_train)
        type='ADABoost'
        y_pred=regressor.predict(self.X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output