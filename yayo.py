import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class Classification:
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
    def __init__(self, X_train=None,y_train=None,X_test=None,y_test=None):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train) 
        X_test = sc_X.transform(X_test)
        self.y_train=self.y_train.reshape(-1)
    def main(self):
        val=[]
        
    def Logistic(self):
        classifier=LogisticRegression()
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='LogisticRegression'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        
        return output
            
    def DecisionTree(self):
        classifier= DecisionTreeClassifier()
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='DecisionTree'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output
    
    def RandomForest(self):
        classifier=RandomForestClassifier()
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='RandomForest'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output
    
    def KNN(self):
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='KNearestNeighbours'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output
    
    def NaiveBayes(self):
        classifier = GaussianNB()
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='NaiveBayes'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output
    
    def SVM(self):
        classifier=SVC(kernel='linear')
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='SupportVectorMachine'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output
        
    def XGBoost(self):
        classifier=XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='XGBoost'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output

    def CatBoost(self):
        classifier=CatBoostClassifier()
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='CatBoost'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output
        
    def ADABoost(self):
        classifier=AdaBoostClassifier()
        lab_enc = LabelEncoder()
        self.y_train= lab_enc.fit_transform(self.y_train)
        classifier.fit(self.X_train, self.y_train)
        y_pred=classifier.predict(self.X_test)
        type='ADABoost'
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        output=[classifier,type,accuracy,cm]
        return output


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