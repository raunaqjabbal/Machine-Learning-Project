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
from catboost import CatBoostRegressor, cv
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
from sklearn.model_selection import GridSearchCV

class Classification:
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
    def __init__(self, X_train=None,y_train=None,X_test=None,y_test=None):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        sc= StandardScaler()
        self.X_train = sc.fit_transform(self.X_train) 
        self.X_test = sc.transform(self.X_test)
        # self.y_train=self.y_train.reshape(-1)
        # df_train.info()
    def main(self):
        val=[]
        
    def Logistic(self):
        classifier=LogisticRegression(max_iter=500)
        type='Logistic'
        parameters =[{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'max_iter':[500],'solver': ['newton-cg','lbfgs','sag'], 'penalty': ['l2']},
                     {'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'max_iter':[500], 'solver': ['saga','liblinear'], 'penalty': ['l1','l2']}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output
            
    def DecisionTree(self):
        classifier= DecisionTreeClassifier()
        type='DecisionTree'
        parameters =[{'criterion': ['gini','entropy'], 'min_samples_leaf':[1,3,5,7,9]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output
    
    def RandomForest(self):
        classifier=RandomForestClassifier()
        type='RandomForest'
        parameters =[{'criterion': ['gini','entropy'], 'max_features':['log2','sqrt', 'auto']}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output
    
    def KNN(self):
        classifier = KNeighborsClassifier()
        type='KNearestNeighbours'
        parameters =[{'n_neighbors': [3,5,7], 'weights':['uniform','distance'], 'algorithm':['auto','ball_tree','kd_tree'], 'p':[1,2], 'n_jobs':[-1]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output
    
    def NaiveBayes(self):
        classifier = GaussianNB()
        type='NaiveBayes'
        parameters =[{'var_smoothing':[1e-9]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output
    
    def SVM(self):      
        classifier=SVC()
        type='SupportVectorMachine'
        parameters =[{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'kernel': ['linear', 'sigmoid']},
                     {'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'scale']},
                     {'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'kernel': ['poly'], 'degree': [3,4,5]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output
        
    def XGBoost(self):
        classifier=XGBClassifier()
        type='XGBoost'
        parameters =[{'n_jobs':[-1],'use_label_encoder':[False],'eval_metric':['error','logloss', 'auc'], 'objective':['binary:logistic'],
        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'max_depth': [1, 3, 5, 7, 9]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output

    def CatBoost(self):
        classifier=CatBoostClassifier()
        type='CatBoost'
        parameters =[{'custom_loss':['AUC', 'Accuracy'], 'verbose':[False], 'allow_writing_files':[False] }]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output
        
    def ADABoost(self):
        classifier=AdaBoostClassifier()
        type='ADABoost'
        parameters =[{'n_estimators':[75], 'algorithm':['SAMME', 'SAMME.R']}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        return output


class Regression:
    def __init__(self, X_train=None,y_train=None,X_test=None,y_test=None):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.y_train = sc_y.fit_transform(self.y_train)   
        self.X_test = sc_X.transform(self.X_test)
        self.y_test = sc_y.transform(self.y_test)
        self.y_train=self.y_train.reshape(-1)
    def main(self):
        val=[]
                
    def Polynomial(self):
        poly_reg = PolynomialFeatures(degree=4)
        type='PolynomialRegression'
        X_train=poly_reg.fit_transform(self.X_train)
        X_test=poly_reg.transform(self.X_test)
        regressor=LinearRegression(n_jobs=-1)
        regressor.fit(X_train, self.y_train)
        y_pred=regressor.predict(X_test)
        r2=r2_score(self.y_test,y_pred)*100
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        accuracy=accuracies.mean()*100
        std=accuracies.std()*100
        output=[regressor,type,r2,accuracy,std]
        return output
            
    def DecisionTree(self):
        regressor= DecisionTreeRegressor()
        type='DecisionTreeRegression'       
        parameters =[{  'criterion':['squared_error'],
                        "min_samples_leaf":[1,3,5,7,9],
                        "max_features":["log2","sqrt",None], }]
        grid_search = GridSearchCV(estimator = regressor,param_grid = parameters,scoring = 'neg_mean_squared_error',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_*100
        best_parameters=grid_search.best_params_
        regressor=grid_search.best_estimator_
        y_pred=regressor.predict(self.X_test)
        testaccuracy=regressor.score(self.X_test,self.y_test)
        trainaccuracy=regressor.score(self.X_train,self.y_train)
        output=[type,trainaccuracy,testaccuracy,best_accuracy,best_parameters]
        return output
    
    def SVM(self):
        regressor=SVR()
        type='SupportVectorMachine'
        parameters =[{'C': [0.2, 0.4, 0.6, 0.8, 1], 'kernel': ['linear']},
                     {'C': [0.2, 0.4, 0.6, 0.8, 1], 'kernel': ['rbf'], 'gamma': [0.2, 0.4, 0.6, 0.8, 1, 'scale', 'auto']},
                     { 'C': [0.2, 0.4, 0.6, 0.8, 1], 'kernel': ['poly'],'degree': [3,4,5]}]
        grid_search = GridSearchCV(estimator = regressor,param_grid = parameters,scoring = 'neg_mean_squared_error', n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_*100
        best_parameters=grid_search.best_params_
        regressor=grid_search.best_estimator_
        y_pred=regressor.predict(self.X_test)
        testaccuracy=regressor.score(self.X_test,self.y_test)
        trainaccuracy=regressor.score(self.X_train,self.y_train)
        output=[type,trainaccuracy,testaccuracy,best_accuracy,best_parameters]
        return output
        
    def RandomForest(self):
        regressor=RandomForestRegressor()
        type='RandomForest'
        parameters =[{'criterion': ['squared_error','absolute_error'], 'max_features':['log2','sqrt', 'auto']}]
        grid_search = GridSearchCV(estimator = regressor,param_grid = parameters,scoring = 'neg_mean_squared_error',n_jobs = -1, cv=10, verbose=3)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_*100
        best_parameters=grid_search.best_params_
        regressor=grid_search.best_estimator_
        y_pred=regressor.predict(self.X_test)
        testaccuracy=regressor.score(self.X_test,self.y_test)
        trainaccuracy=regressor.score(self.X_train,self.y_train)
        output=[type,trainaccuracy,testaccuracy,best_accuracy,best_parameters]
        return output
        
    def XGBoost(self):
        regressor=XGBRegressor(use_label_encoder=False)
        type='XGBoost'
        parameters =[{'n_jobs':[-1],'use_label_encoder':[False],'eval_metric':['rmse'], 'objective':['reg:squarederror'], 'booster':['gblinear', 'gbtree']}]
        grid_search = GridSearchCV(estimator = regressor,param_grid = parameters,scoring = 'neg_mean_squared_error',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_*100
        best_parameters=grid_search.best_params_
        regressor=grid_search.best_estimator_
        y_pred=regressor.predict(self.X_test)
        testaccuracy=regressor.score(self.X_test,self.y_test)
        trainaccuracy=regressor.score(self.X_train,self.y_train)
        output=[type,trainaccuracy,testaccuracy,best_accuracy,best_parameters]
        return output

    def CatBoost(self):
        #CatBoost .score looks broken
        regressor=CatBoostRegressor(loss_function='RMSE', verbose=False, allow_writing_files=False, iterations=300)
        type='CatBoost'
        parameters ={'learning_rate': [0.03, 0.1, 0.3, 0.1]}
        grid_search = GridSearchCV(estimator = regressor,param_grid = parameters,scoring = 'neg_mean_squared_error',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_*100
        best_parameters=grid_search.best_params_
        regressor=grid_search.best_estimator_
        
        y_pred=regressor.predict(self.X_test)
        trainaccuracy=r2_score(self.y_train,regressor.predict(self.X_train))*100
        testaccuracy=r2_score(self.y_test,y_pred)*100
        output=[type,trainaccuracy,testaccuracy,best_accuracy,best_parameters]
        return output

    def ADABoost(self):
        regressor=AdaBoostRegressor()
        type='ADABoost'
        parameters =[{'n_estimators':[10,15,25,50,75], 'loss':['linear'],'learning_rate': [0.01,0.1, 0.3, 1, 3]}]
        grid_search = GridSearchCV(estimator = regressor,param_grid = parameters,scoring = 'neg_mean_squared_error',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train,self.y_train)
        best_accuracy = grid_search.best_score_*100
        best_parameters=grid_search.best_params_
        regressor=grid_search.best_estimator_
        y_pred=regressor.predict(self.X_test)
        testaccuracy=regressor.score(self.X_test,self.y_test)
        trainaccuracy=regressor.score(self.X_train,self.y_train)
        output=[type,trainaccuracy,testaccuracy,best_accuracy,best_parameters]
        return output