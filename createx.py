import pandas as pd 
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2

def X_y_for_classif():
    dataset= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
    dataset = dataset.replace(0, pd.NA).dropna()
    
    y1=dataset[dataset['y']==1]
    y1_downsampled=resample(y1, replace=False,n_samples=2000,random_state=42)
    y2=dataset[dataset['y']==2]
    y2_downsampled=resample(y2, replace=False,n_samples=2000,random_state=42)
    y3=dataset[dataset['y']==3]
    y3_downsampled=resample(y3, replace=False,n_samples=2000,random_state=42)
    y4=dataset[dataset['y']==4]
    y4_downsampled=resample(y4, replace=False,n_samples=2000,random_state=42)
    y5=dataset[dataset['y']==5]
    y5_downsampled=resample(y5, replace=False,n_samples=2000,random_state=42)
    
    dataset=pd.concat([y1_downsampled,y2_downsampled,y3_downsampled,y4_downsampled,y5_downsampled])
    dataset=dataset.sample(frac=1,random_state=42).reset_index(drop=True)
    scaler=MinMaxScaler()
    X_scaled=scaler.fit_transform(dataset.iloc[:,:-1])
    
    chi2_selector=SelectKBest(chi2,k=178)
    X_chi2_selected = chi2_selector.fit_transform(X_scaled,dataset['y'])
    
    return X_chi2_selected,dataset['y']

def create_train_test():
    X,y = X_y_for_classif()
    X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=42)
    X_train, X_val, y_train, y_val =train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_val,X_test,y_train,y_val,y_test
    
    