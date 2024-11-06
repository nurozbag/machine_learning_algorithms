import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification

class Knn_opt():
    def  knn_alg():
        data_set= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
        data_set.head()
        data_set = data_set.replace(0, pd.NA).dropna()
        #######datada gezinmek
        X = data_set.iloc[:,1:-1].values
        y = data_set.iloc[:,-1:].values
        X, y = make_classification(n_samples=11500, n_features=180, n_informative=100, n_classes=6, random_state=42)
        X.shape
        y.shape
        
        #####normalizasyon
        scaler_minmax = MinMaxScaler()
        Xscaled_data = scaler_minmax.fit_transform(X)
        #####train-test split
        X_train, X_test, y_train, y_test = train_test_split(Xscaled_data,y,test_size=0.3,shuffle=True)
        #normalizasyon
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 
        
        ###model fit
        knn_model = KNeighborsClassifier(n_neighbors=1)#burasını kullanıcıdan alınacak
        knn_model.fit(X_train, y_train)
        y_predict = knn_model.predict(X_test)
        
        ####karisiklik matrisi ve metrikler
        cm = confusion_matrix(y_test,y_predict)
        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict,average='weighted')
        recall = recall_score(y_test,y_predict,average='weighted')
        f1_s = f1_score(y_test,y_predict,average='weighted')
        specificity = recall_score(y_test,y_predict,average='weighted')
        print({"Confusion matrix":cm,"Accuracy":accuracy,"Precision":precision,"recall":recall,"Specificity":specificity,"F1_score":f1_s})
        
        ###saving model with joblib
        knn_classifier = 'knn_trained_model.sav'
        joblib.dump(knn_model, knn_classifier)
    knn_alg("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")