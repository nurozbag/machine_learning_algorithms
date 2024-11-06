import sys
from ann_arayuz import Ui_Form
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pyp
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from PyQt5.QtGui import QPixmap
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier


class ANN(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.radio_optm.toggled.connect(self.buton_goster_opt)
        self.ui.radio_verigir.toggled.connect(self.buton_goster_giris)
        self.ui.batch_deger.hide()
        self.ui.test_al.hide()
        self.ui.girisli_egit.hide()
        self.ui.optimum_egit.hide()
        self.ui.batch_deger.hide()
        self.ui.epoch_deger.hide()
        self.ui.patience_deger.hide()
        self.ui.validation_deger.hide()
        self.ui.fold_deger.hide()
        self.ui.label.hide()
        self.ui.label_2.hide()
        self.ui.label_3.hide()
        self.ui.label_4.hide()
        self.ui.label_5.hide()
        self.ui.label_6.hide()
        self.ui.radio_optm.toggled.connect(self.buton_goster_opt)
        self.ui.radio_verigir.toggled.connect(self.buton_goster_giris)
        self.ui.optimum_egit.clicked.connect(self.optimum_egit)
        self.ui.girisli_egit.clicked.connect(self.Girisle_egit)
        
    def buton_goster_opt(self):
        self.ui.optimum_egit.setVisible(self.ui.radio_optm.isChecked())
    
    def buton_goster_giris(self):
        self.ui.girisli_egit.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.fold_deger.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.test_al.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.batch_deger.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.epoch_deger.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.validation_deger.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.patience_deger.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.label.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.label_2.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.label_3.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.label_4.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.label_5.setVisible(self.ui.radio_verigir.isChecked())
        self.ui.label_6.setVisible(self.ui.radio_verigir.isChecked())
        
        
    def Girisle_egit(self):
        test_deger = float(self.ui.test_al.text())
        fold_degeri = int(self.ui.fold_deger.text())
        batch_degeri = int(self.ui.batch_deger.text())
        epoch_degeri = int(self.ui.epoch_deger.text())
        validation = float(self.ui.validation_deger.text())
        patience = int(self.ui.patience_deger.text())
        def Girisle_egit(test_deger,fold_degeri,batch_degeri,epoch_degeri,validation,patience):
            data_set= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
            data_set = data_set.replace(0, pd.NA).dropna()
            X = data_set.iloc[:,1:-1].values
            y = data_set.iloc[:,-1:].values
            X, y = make_classification(n_samples=11500, n_features=180, n_informative=100, n_classes=6, random_state=42)
            kfold = KFold(n_splits=fold_degeri, shuffle=True, random_state=42)
            accuracy_scores = []
            for train_index, test_index in kfold.split(X, y):
                X_kfold, X_test = X[train_index], X[test_index]
                y_kfold, y_test = y[train_index], y[test_index]
                X_train, _, y_train, _ = train_test_split(X_kfold, y_kfold, test_size=test_deger, shuffle=True, random_state=42)
                y_train_categorical = to_categorical(y_train, num_classes=6)
                y_test_categorical = to_categorical(y_test, num_classes=6)

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test) 

            ann_model = Sequential()
            ann_model.add(Dense(units=512,activation='relu'))
            ann_model.add(Dense(units=512,activation='relu'))
            ann_model.add(Dense(units=128,activation='relu'))
            ann_model.add(Dense(units=128,activation='relu'))
            ann_model.add(Dense(units=128,activation='relu'))
            ann_model.add(Dense(units=64,activation='relu'))
            ann_model.add(Dense(units=6,activation='softmax'))
            ann_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
            early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
            chckpt = ModelCheckpoint(filepath='C:\\Users\\Pc\\Desktop\\ML_project"/check_point' , verbose=1, save_best_only=True)

            ann_model.fit(X_train,y_train_categorical,batch_size=batch_degeri,epochs = epoch_degeri,validation_split=(validation),callbacks=[early],verbose=1)
            #estimator = KerasClassifier(build_fn=ann_model, epochs=200, batch_size=5, verbose=0)
            accuracy_score = ann_model.evaluate(X_test, y_test_categorical)
            accuracy_scores.append(accuracy_score[1])
            print(accuracy_score,kfold)
            print("Mean Accuracy:", np.mean(accuracy_scores))
            classifier = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(512, 512, 128, 128, 128, 64),
                                                           max_iter=1000,
                                                           random_state=42))
            y_train_binary = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5])
            y_test_binary = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
            classifier.fit(X_train, y_train_binary)
            y_score = classifier.predict_proba(X_test)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(6): 
                fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.figure(figsize=(10, 8))
            for i in range(6):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive')
            plt.ylabel('True Positive')
            plt.title('(ROC) Curve for Multi-class')
            plt.legend(loc="lower right")
            plt.savefig("C:\\Users\\Pc\\Desktop\\ML_project\\roc")
            roc_auc__path = "C:\\Users\\Pc\\Desktop\\ML_project\\roc"
            
            pixmap = QPixmap(roc_auc__path)
            self.ui.LblGoruntu.setPixmap(pixmap)
            self.ui.LblGoruntu.setScaledContents(True)
            
            #self.ui.label_9.setText(accuracy_text)
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
            #results = cross_val_score(estimator, X, y, cv=kfold)
            ann_model.save("C:\\Users\\Pc\\Desktop\\ML_project")
        return Girisle_egit(test_deger, fold_degeri, batch_degeri, epoch_degeri, validation, patience)
    def optimum_egit(self):
        data_set= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
        data_set = data_set.replace(0, pd.NA).dropna()
        X = data_set.iloc[:,1:-1].values
        y = data_set.iloc[:,-1:].values
        X, y = make_classification(n_samples=11500, n_features=180, n_informative=100, n_classes=6, random_state=42)
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

        ann_model = Sequential()
        ann_model.add(Dense(units=512,activation='relu'))
        ann_model.add(Dense(units=512,activation='relu'))
        ann_model.add(Dense(units=128,activation='relu'))
        ann_model.add(Dense(units=128,activation='relu'))
        ann_model.add(Dense(units=128,activation='relu'))
        ann_model.add(Dense(units=64,activation='relu'))
        ann_model.add(Dense(units=6,activation='softmax'))
        ann_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        chckpt = ModelCheckpoint(filepath='C:\\Users\\Pc\\Desktop\\ML_project"/check_point' , verbose=1, save_best_only=True)

        ann_model.fit(X_train,y_train_categorical,batch_size=37,epochs = 3,validation_split=(0.2),callbacks=[early],verbose=1)
        #estimator = KerasClassifier(build_fn=ann_model, epochs=200, batch_size=5, verbose=0)
        accuracy_score = ann_model.evaluate(X_test, y_test_categorical)
        accuracy_scores.append(accuracy_score[1])
        print(accuracy_score,kfold)
        print("Mean Accuracy:", np.mean(accuracy_scores))
        mean_accuracy = np.mean(accuracy_scores)
        accuracy_text = f"Accuracy Score: {accuracy_score[1]:.4f}, Mean Accuracy: {mean_accuracy:.4f}"
        classifier = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(512, 512, 128, 128, 128, 64),
                                                       max_iter=1000,
                                                       random_state=42))
        y_train_binary = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5])
        y_test_binary = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
        classifier.fit(X_train, y_train_binary)
        y_score = classifier.predict_proba(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(6): 
            fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure(figsize=(10, 8))
        for i in range(6):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.title('(ROC) Curve for Multi-class')
        plt.legend(loc="lower right")
        plt.savefig("C:\\Users\\Pc\\Desktop\\ML_project\\roc")
        roc_auc__path = "C:\\Users\\Pc\\Desktop\\ML_project\\roc"
        
        pixmap = QPixmap(roc_auc__path)
        self.ui.LblGoruntu.setPixmap(pixmap)
        self.ui.LblGoruntu.setScaledContents(True)
        
        self.ui.label_9.setText(accuracy_text)
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
        #results = cross_val_score(estimator, X, y, cv=kfold)
        ann_model.save("C:\\Users\\Pc\\Desktop\\ML_project")
        return 0
    
    
    
if __name__ == '__main__':
   app = QApplication(sys.argv)
   pencere = ANN()
   pencere.show()
   sys.exit(app.exec_())
                