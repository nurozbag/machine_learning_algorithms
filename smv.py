import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from keras.utils import to_categorical
from sklearn.model_selection import KFold

data_set= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
data_set.head()
data_set = data_set.replace(0, pd.NA).dropna()

X = data_set.iloc[:,1:-1].values
y = data_set.iloc[:,-1:].values
kfold = KFold(n_splits=40, shuffle=True, random_state=42)
accuracy_scores = []
for train_index, test_index in kfold.split(X, y):
    X_kfold, X_test = X[train_index], X[test_index]
    y_kfold, y_test = y[train_index], y[test_index]
    X_train, _, y_train, _ = train_test_split(X_kfold, y_kfold, test_size=0.3, shuffle=True, random_state=42)
    y_train_categorical = to_categorical(y_train, num_classes=6)
    y_test_categorical = to_categorical(y_test, num_classes=6)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 

    svm_model= svm.SVC(C=2,kernel='linear')
    y_train_flat = y_train_categorical.ravel()
    y_test_flat = y_test_categorical.ravel()
    svm_model.fit(X_train,y_train_flat)
    y_predict = svm_model.predict(X_test)
    
    
    cm = confusion_matrix(y_test_flat,y_predict)
    accuracy = accuracy_score(y_test_flat, y_predict)
    precision = precision_score(y_test_flat, y_predict,average='weighted')
    recall = recall_score(y_test_flat,y_predict,average='weighted')
    f1_s = f1_score(y_test_flat,y_predict,average='weighted')
    specificity = recall_score(y_test_flat,y_predict,average='weighted')
    print({"Confusion matrix":cm,"Accuracy":accuracy,"Precision":precision,"recall":recall,"Specificity":specificity,"F1_score":f1_s})
    
    ###saving model with joblib
    svm_classifier = 'svm_trained_model.sav'
    joblib.dump(svm_model, svm_classifier)