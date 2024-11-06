import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import joblib
import seaborn as sns
from svm_arayuz import Ui_Form
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

class SVM(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.radio_optm.toggled.connect(self.buton_goster_opt)
        self.ui.radio_verigir.toggled.connect(self.buton_goster_giris)
        self.ui.C_degeri_al.hide()
        self.ui.test_al.hide()
        self.ui.girisli_egit.hide()
        self.ui.optimum_egit.hide()
        self.ui.optimum_egit.clicked.connect(self.optimum)
        self.ui.girisli_egit.clicked.connect(self.Girisle_egit)    
    def buton_goster_opt(self):
        self.ui.optimum_egit.setVisible(self.ui.radio_optm.isChecked())
    
    def buton_goster_giris(self):
        self.ui.girisli_egit.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.C_degeri_al.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.test_al.setVisible(self.ui.radio_verigir.isChecked())
        
    def Girisle_egit(self):
        test_degeri = float(self.ui.test_al.text())
        k_fold = int(self.ui.fold_deger.text())
        c_degeri = int(self.ui.C_degeri_al.text())
        def girisle_egit(test_degeri,k_fold,c_degeri):
            data_set= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
            data_set.head()
            data_set = data_set.replace(0, pd.NA).dropna()

            X = data_set.iloc[:,1:-1].values
            y = data_set.iloc[:,-1:].values
            kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            for train_index, test_index in kfold.split(X, y):
                X_kfold, X_test = X[train_index], X[test_index]
                y_kfold, y_test = y[train_index], y[test_index]
                X_train, _, y_train, _ = train_test_split(X_kfold, y_kfold, test_size=test_degeri, shuffle=True, random_state=42)
                y_train_categorical = to_categorical(y_train, num_classes=6)
                y_test_categorical = to_categorical(y_test, num_classes=6)

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test) 

                svm_model= svm.SVC(C=c_degeri,kernel='linear')
                y_train_flat = y_train.ravel()
                y_test_flat = y_test.ravel()
                svm_model.fit(X_train,y_train_flat)
                y_predict = svm_model.predict(X_test)
                
                
                cm = confusion_matrix(y_test_flat,y_predict)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap="Blues", ax=None)
                plt.savefig("C:\\Users\\Pc\\Desktop\\ML_project\\svm_cm_matrix")
                cm_matris_path = "C:\\Users\\Pc\\Desktop\\ML_project\\svm_cm_matrix"
               
                pixmap = QPixmap(cm_matris_path)
                self.ui.LblGoruntu.setPixmap(pixmap)
                self.ui.LblGoruntu.setScaledContents(True)
               
                accuracy = accuracy_score(y_test_flat, y_predict)
                precision = precision_score(y_test_flat, y_predict,average='weighted')
                recall = recall_score(y_test_flat,y_predict,average='weighted')
                f1_s = f1_score(y_test_flat,y_predict,average='weighted')
                specificity = recall_score(y_test_flat,y_predict,average='weighted')
                print({"Accuracy":accuracy,"Precision":precision,"recall":recall,"Specificity":specificity,"F1_score":f1_s})
                cv_scores = cross_val_score(svm_model, X, y, cv=5)
                print("Cross-Validation Scores:", cv_scores)
                cv_mean_score = np.mean(cv_scores)
                string_accuracy = f"Accuracy: {accuracy:.2f}"  
                string_precision = f"Precision: {precision:.2f}"
                string_recall = f"Recall: {recall:.2f}"
                string_f1_s = f"F1 Score: {f1_s:.2f}"
                string_specificity = f"Specificity: {specificity:.2f}"
                string_crossval = f"CrossValidation: {cv_mean_score :.2f}"
                self.ui.Accuracy.setText(string_accuracy)
                self.ui.Precision.setText(string_precision)
                self.ui.Recall.setText(string_recall)
                self.ui.F1Score.setText(string_f1_s)
                self.ui.Specificity.setText(string_specificity)
                self.ui.CrossVal.setText(string_crossval)
                self.ui.TableX_test.setRowCount(len(X_train) + len(X_test))
                self.ui.TableX_test.setColumnCount(2)
                self.ui.TableX_test.setHorizontalHeaderLabels(['Index', 'Value'])
               
                self.ui.TableX_train.setRowCount(len(X_train) + len(X_train))
                self.ui.TableX_train.setColumnCount(2)
                self.ui.TableX_train.setHorizontalHeaderLabels(['Index', 'Value'])
               
                self.ui.TableY_train.setRowCount(len(y_train) + len(y_train))
                self.ui.TableY_train.setColumnCount(2)
                self.ui.TableY_train.setHorizontalHeaderLabels(['Index', 'Value'])
               
                self.ui.TableY_test.setRowCount(len(y_test) + len(y_test))
                self.ui.TableY_test.setColumnCount(2)
                self.ui.TableY_test.setHorizontalHeaderLabels(['Index', 'Value'])
               
               

                row_index_train = 0
                row_index_test = 0
                row_index_y_train = 0
                row_index_y_test = 0
                for i, x_train_item in enumerate(X_train):
                   self.ui.TableX_train.setItem(row_index_train, 0, QTableWidgetItem(f"X_train[{i}]"))
                   self.ui.TableX_train.setItem(row_index_train, 1, QTableWidgetItem(str(x_train_item)))
                   row_index_train += 1
               
                for i, x_test_item in enumerate(X_test):
                   self.ui.TableX_test.setItem(row_index_test, 0, QTableWidgetItem(f"X_test[{i}]"))
                   self.ui.TableX_test.setItem(row_index_test, 1, QTableWidgetItem(str(x_test_item)))
                   row_index_test += 1
               
                for i, y_test_item in enumerate(y_test):
                   self.ui.TableY_test.setItem(row_index_y_test, 0, QTableWidgetItem(f"Y_test[{i}]"))
                   self.ui.TableY_test.setItem(row_index_y_test, 1, QTableWidgetItem(str(y_test_item)))
                   row_index_y_test += 1
                for i, y_train_item in enumerate(y_train):
                   self.ui.TableY_train.setItem(row_index_y_train, 0, QTableWidgetItem(f"Y_train[{i}]"))
                   self.ui.TableY_train.setItem(row_index_y_train, 1, QTableWidgetItem(str(y_train_item)))
                   row_index_y_train += 1
                svm_classifier = 'svm_trained_model.sav'
                joblib.dump(svm_model, svm_classifier)
                return 0
        return girisle_egit(test_degeri,k_fold,c_degeri)
    def optimum(self):
            data_set= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
            data_set.head()
            data_set = data_set.replace(0, pd.NA).dropna()

            X = data_set.iloc[:,1:-1].values
            y = data_set.iloc[:,-1:].values
        
            kfold = KFold(n_splits=40, shuffle=True, random_state=42)
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
                y_train_flat = y_train.ravel()
                y_test_flat = y_test.ravel()
                svm_model.fit(X_train, y_train_flat)
                y_predict = svm_model.predict(X_test)
                
                cm = confusion_matrix(y_test_flat, y_predict)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap="Blues", ax=None)
                plt.savefig("C:\\Users\\Pc\\Desktop\\ML_project\\svm_cm_matrix")
                cm_matris_path = "C:\\Users\\Pc\\Desktop\\ML_project\\svm_cm_matrix"
           
                pixmap = QPixmap(cm_matris_path)
                self.ui.LblGoruntu.setPixmap(pixmap)
                self.ui.LblGoruntu.setScaledContents(True)
           
                accuracy = accuracy_score(y_test_flat, y_predict)
                precision = precision_score(y_test_flat, y_predict,average='weighted')
                recall = recall_score(y_test_flat,y_predict,average='weighted')
                f1_s = f1_score(y_test_flat,y_predict,average='weighted')
                specificity = recall_score(y_test_flat,y_predict,average='weighted')
                print({"Accuracy":accuracy,"Precision":precision,"recall":recall,"Specificity":specificity,"F1_score":f1_s})
                cv_scores = cross_val_score(svm_model, X, y, cv=5)
                print("Cross-Validation Scores:", cv_scores)
                cv_mean_score = np.mean(cv_scores)
                string_accuracy = f"Accuracy: {accuracy:.2f}"  
                string_precision = f"Precision: {precision:.2f}"
                string_recall = f"Recall: {recall:.2f}"
                string_f1_s = f"F1 Score: {f1_s:.2f}"
                string_specificity = f"Specificity: {specificity:.2f}"
                string_crossval = f"CrossValidation: {cv_mean_score :.2f}"
           
                self.ui.Accuracy.setText(string_accuracy)
                self.ui.Precision.setText(string_precision)
                self.ui.Recall.setText(string_recall)
                self.ui.F1Score.setText(string_f1_s)
                self.ui.Specificity.setText(string_specificity)
                self.ui.CrossVal.setText(string_crossval)
           
           
           #LISTELEME
                self.ui.TableX_test.setRowCount(len(X_train) + len(X_test))
                self.ui.TableX_test.setColumnCount(2)
                self.ui.TableX_test.setHorizontalHeaderLabels(['Index', 'Value'])
            
                self.ui.TableX_train.setRowCount(len(X_train) + len(X_train))
                self.ui.TableX_train.setColumnCount(2)
                self.ui.TableX_train.setHorizontalHeaderLabels(['Index', 'Value'])
           
                self.ui.TableY_train.setRowCount(len(y_train) + len(y_train))
                self.ui.TableY_train.setColumnCount(2)
                self.ui.TableY_train.setHorizontalHeaderLabels(['Index', 'Value'])
           
                self.ui.TableY_test.setRowCount(len(y_test) + len(y_test))
                self.ui.TableY_test.setColumnCount(2)
                self.ui.TableY_test.setHorizontalHeaderLabels(['Index', 'Value'])
           
           

                row_index_train = 0
                row_index_test = 0
                row_index_y_train = 0
                row_index_y_test = 0
                for i, x_train_item in enumerate(X_train):
                    self.ui.TableX_train.setItem(row_index_train, 0, QTableWidgetItem(f"X_train[{i}]"))
                    self.ui.TableX_train.setItem(row_index_train, 1, QTableWidgetItem(str(x_train_item)))
                    row_index_train += 1
               
                for i, x_test_item in enumerate(X_test):
                    self.ui.TableX_test.setItem(row_index_test, 0, QTableWidgetItem(f"X_test[{i}]"))
                    self.ui.TableX_test.setItem(row_index_test, 1, QTableWidgetItem(str(x_test_item)))
                    row_index_test += 1
           
                for i, y_test_item in enumerate(y_test):
                    self.ui.TableY_test.setItem(row_index_y_test, 0, QTableWidgetItem(f"Y_test[{i}]"))
                    self.ui.TableY_test.setItem(row_index_y_test, 1, QTableWidgetItem(str(y_test_item)))
                    row_index_y_test += 1
                for i, y_train_item in enumerate(y_train):
                    self.ui.TableY_train.setItem(row_index_y_train, 0, QTableWidgetItem(f"Y_train[{i}]"))
                    self.ui.TableY_train.setItem(row_index_y_train, 1, QTableWidgetItem(str(y_train_item)))
                    row_index_y_train += 1
       
      
                return 0     
            
            ###saving model with joblib
                svm_classifier = 'svm_trained_model.sav'
                joblib.dump(svm_model, svm_classifier)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
   app = QApplication(sys.argv)
   pencere = SVM()
   pencere.show()
   sys.exit(app.exec_())
                
    