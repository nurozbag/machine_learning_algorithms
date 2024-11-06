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
class Kullanici_girisli_knn():
    def  kullanici_girisli_knn(self,data_yol,t_deger,komsuluk):
        data_set= pd.read_csv(data_yol)
        data_set.head()
        
        ndata=data_set.isnull().sum()
        data_set = data_set.replace(0, pd.NA).dropna()
        #######datada gezinmek
        X = data_set.iloc[:,1:-1].values
        y = data_set.iloc[:,-1:].values
        y = np.ravel(y)
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=6, random_state=42)
        X.shape
        y.shape
        
        #####normalizasyon
        scaler_minmax = MinMaxScaler()
        Xscaled_data = scaler_minmax.fit_transform(X)
        #####train-test split
        X_train, X_test, y_train, y_test = train_test_split(Xscaled_data,y,test_size=t_deger,shuffle=True)
        #normalizasyon
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        ###model fit
        knn_model = KNeighborsClassifier(komsuluk)#burasını kullanıcıdan alınacak
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
        knn_classifier = 'knn_kullanici_trained_model.sav'
        joblib.dump(knn_model, knn_classifier)
    kullanici_girisli_knn("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")